"""Ray actor that serves batched generation for RLOO training.

This process owns a vLLM engine and is restarted when checkpoints change.
"""

import warnings
warnings.filterwarnings("ignore")

import re

import ray
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from rloo_trainer.context_restoration import (
    backtrack_stop_strings, build_cleanup, parse_backtrack_stop,
)

# Strip a trailing `<backtrack_K>` (with optional whitespace) from a segment
# whose cleanup failed. Without this, the kept_text ends with an orphan
# backtrack token — violating the "every <backtrack_K> is followed by
# <state_K>" invariant the context manager is supposed to enforce. The
# resulting trace has no <answer> block so compute_score and the structural
# PRM both score it 0 (NO-ANSWER), same as without the strip.
_TRAILING_BACKTRACK_RE = re.compile(r"<backtrack_\d+>\s*$")

@ray.remote(num_gpus=1)
class SamplingWorker:
    """GPU worker responsible for policy rollouts (text sampling)."""
    def __init__(
        self,
        model_path,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
        max_num_seqs=64,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        max_tokens=1024,
        group_size=16,
        # SELS-only, default-off. When False, this worker is byte-identical to
        # the baseline configuration (default `generate()` path is the only one
        # used by `rloo.py` unless --context_restoration is set).
        enable_context_restoration=False,
        max_restoration_rounds=4,
    ):
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_chunked_prefill = enable_chunked_prefill
        self.max_num_seqs = max_num_seqs
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_tokens = max_tokens
        self.group_size = group_size
        self.enable_context_restoration = enable_context_restoration
        self.max_restoration_rounds = max_restoration_rounds

    def load_checkpoint(self):
        """(Re)load tokenizer + vLLM engine for current model path."""
        self.tear_down()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        effective_max_model_len = self.max_model_len
        tokenizer_max_length = getattr(self.tokenizer, "model_max_length", None)
        if (
            effective_max_model_len is not None
            and isinstance(tokenizer_max_length, int)
            and 0 < tokenizer_max_length < 1_000_000
        ):
            effective_max_model_len = min(effective_max_model_len, tokenizer_max_length)
        effective_max_num_batched_tokens = self.max_num_batched_tokens
        if (
            effective_max_model_len is not None
            and effective_max_num_batched_tokens is not None
            and effective_max_num_batched_tokens < effective_max_model_len
        ):
            effective_max_num_batched_tokens = effective_max_model_len

        # Build vLLM config once so callers can hot-swap model paths cleanly.
        llm_kwargs = {
            "model": self.model_path,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_batched_tokens": effective_max_num_batched_tokens,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "max_num_seqs": self.max_num_seqs,
        }
        if effective_max_model_len is not None:
            llm_kwargs["max_model_len"] = effective_max_model_len

        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            max_tokens=self.max_tokens,
            n=self.group_size,
            stop=["</answer>"],
            include_stop_str_in_output=True,
            logprobs=1,
            # SELS checkpoints keep <state_*>/<backtrack_*> in output; default unchanged.
            skip_special_tokens="<state_0>" not in self.tokenizer.get_vocab(),
        )

        # Segmented-sampling params, only built when context restoration is
        # enabled. Default `self.sampling_params` above is untouched, so the
        # baseline `generate()` path stays byte-identical.
        if self.enable_context_restoration:
            seg_stops = ["</answer>"] + backtrack_stop_strings(max_depth=4)
            common = dict(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                max_tokens=self.max_tokens,
                stop=seg_stops,
                include_stop_str_in_output=True,
                logprobs=1,
                skip_special_tokens=False,           # SELS-only path; tokens are registered.
            )
            # Round 0: one prompt → group_size completions (KV-shared).
            self._seg_params_initial = SamplingParams(n=self.group_size, **common)
            # Rounds 1..N: one resume-prefix per active completion → n=1.
            self._seg_params_resume  = SamplingParams(n=1,                **common)

    def update_model_path(self, model_path):
        """Switch to a new checkpoint and reload generation engine."""
        self.model_path = model_path
        self.load_checkpoint()

    def tear_down(self):
        """Release GPU memory and distributed state before reload/exit."""
        import gc
        import torch
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'llm'):
            # vLLM requires explicit cleanup of distributed resources
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel
                destroy_model_parallel()
            except Exception:
                pass
            del self.llm
        if hasattr(self, 'sampling_params'):
            del self.sampling_params
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @staticmethod
    def _extract_sequence_logprob(output) -> float:
        """Best-effort extraction of sequence logprob across vLLM versions."""
        if hasattr(output, "cumulative_logprob") and output.cumulative_logprob is not None:
            return float(output.cumulative_logprob)
        if hasattr(output, "logprob") and output.logprob is not None:
            return float(output.logprob)

        # Fallback path: reconstruct from per-token logprobs if cumulative logprob
        # is not present in this vLLM version.
        token_ids = getattr(output, "token_ids", None)
        token_logprobs = getattr(output, "logprobs", None)
        if token_ids is None or token_logprobs is None:
            raise RuntimeError(
                "Could not extract sequence logprob from vLLM output: missing cumulative_logprob/logprob and token-level logprobs."
            )

        seq_logprob = 0.0
        for token_id, token_topk in zip(token_ids, token_logprobs):
            if token_topk is None:
                continue

            token_entry = None
            if isinstance(token_topk, dict):
                token_entry = token_topk.get(token_id)
                if token_entry is None and len(token_topk) == 1:
                    token_entry = next(iter(token_topk.values()))
            else:
                token_entry = token_topk

            if token_entry is None:
                continue

            if hasattr(token_entry, "logprob"):
                seq_logprob += float(token_entry.logprob)
            else:
                seq_logprob += float(token_entry)

        return float(seq_logprob)

    def generate(self, prompts: list[str]) -> tuple[list[list[str]], list[list[float]]]:
        """Sample `group_size` responses per prompt and return sequence logprobs."""
        outputs = self.llm.generate(prompts, self.sampling_params)
        all_responses = []
        all_logprobs = []
        for output in outputs:
            curr_responses = []
            curr_logprobs = []
            for o in output.outputs:
                curr_responses.append(o.text)
                curr_logprobs.append(self._extract_sequence_logprob(o))
            all_responses.append(curr_responses)
            all_logprobs.append(curr_logprobs)
        return all_responses, all_logprobs

    # --- Context-restoration path (SELS, opt-in) ---------------------------
    #
    # Standalone alternative to `generate()`. Same return shape so `rloo.py`
    # can dispatch to one or the other behind a single flag without any other
    # downstream changes (the update worker re-tokenizes the final response
    # and recomputes logprobs from scratch).
    #
    # On each segment: if vLLM stops on `<backtrack_K>`, we cut the response
    # at the most recent depth-K ancestor on the active branch (via
    # `context_restoration.build_cleanup`), append a clean
    # `<backtrack_K>\n<state_K> [list]\n` marker, and re-prompt vLLM with
    # `original_prompt + cleaned_so_far` to generate the next segment in
    # *clean* context. The model never sees the failed branch in its
    # conditioning window during either sampling or inference.
    #
    # Caveats made explicit so future-me doesn't get burned:
    #   - Returned logprob = sum of per-segment cumulative_logprobs. This
    #     over-counts: it includes tokens that ended up dropped by cleanup.
    #     The update worker recomputes from scratch on the final response, so
    #     this scalar is used only for sampling-side diagnostics / KL.
    #   - Each backtrack costs one full re-prompt with cold KV cache. With
    #     1–2 backtracks/rollout this is ~2–3× the baseline vLLM cost.
    #   - Hard caps: `max_restoration_rounds` segments per completion, and
    #     final response is allowed to exceed `max_tokens` (the update worker
    #     truncates downstream — same as baseline budget overflow handling).
    def generate_with_context_restoration(
        self, prompts: list[str]
    ) -> tuple[list[list[str]], list[list[float]]]:
        if not self.enable_context_restoration:
            raise RuntimeError(
                "generate_with_context_restoration called but "
                "enable_context_restoration=False on this worker."
            )

        batch_size  = len(prompts)
        group_size  = self.group_size

        # Per-completion accumulator: (kept_text, logprob_sum, done_flag).
        kept_text = [["" for _ in range(group_size)] for _ in range(batch_size)]
        logprob   = [[0.0 for _ in range(group_size)] for _ in range(batch_size)]
        done      = [[False for _ in range(group_size)] for _ in range(batch_size)]

        # --- Round 0: standard batched generation, n=group_size per prompt ---
        outputs = self.llm.generate(prompts, self._seg_params_initial)
        for i, output in enumerate(outputs):
            for j, o in enumerate(output.outputs):
                seg_text = o.text
                seg_lp   = self._extract_sequence_logprob(o)
                K = parse_backtrack_stop(getattr(o, "stop_reason", None))
                if K is None:
                    kept_text[i][j] = seg_text
                    logprob[i][j]   = seg_lp
                    done[i][j]      = True
                    continue
                cleanup = build_cleanup(seg_text, K)
                if cleanup is None:
                    # Malformed (no valid ancestor): drop the trailing orphan
                    # <backtrack_K> so the invariant holds, then stop. The
                    # trace has no <answer> → outcome reward + structural PRM
                    # both score it 0, same as a NO-ANSWER failure.
                    kept_text[i][j] = _TRAILING_BACKTRACK_RE.sub("", seg_text)
                    logprob[i][j]   = seg_lp
                    done[i][j]      = True
                    continue
                cleaned, marker = cleanup
                kept_text[i][j] = cleaned + marker
                logprob[i][j]   = seg_lp

        # --- Rounds 1..N: re-prompt only the still-active completions, n=1 ---
        for _round in range(self.max_restoration_rounds):
            pending = [(i, j) for i in range(batch_size)
                              for j in range(group_size) if not done[i][j]]
            if not pending:
                break
            resume_prompts = [prompts[i] + kept_text[i][j] for i, j in pending]
            outputs = self.llm.generate(resume_prompts, self._seg_params_resume)
            for (i, j), output in zip(pending, outputs):
                o = output.outputs[0]
                seg_text = o.text
                seg_lp   = self._extract_sequence_logprob(o)
                logprob[i][j] += seg_lp
                K = parse_backtrack_stop(getattr(o, "stop_reason", None))
                if K is None:
                    kept_text[i][j] += seg_text
                    done[i][j]      = True
                    continue
                # Cleanup operates on the *combined* response (kept so far +
                # this segment) so the branch walk sees the full history.
                combined = kept_text[i][j] + seg_text
                cleanup  = build_cleanup(combined, K)
                if cleanup is None:
                    kept_text[i][j] += _TRAILING_BACKTRACK_RE.sub("", seg_text)
                    done[i][j]      = True
                    continue
                cleaned, marker = cleanup
                kept_text[i][j] = cleaned + marker

        # Anything still not done has exhausted the round budget — accept its
        # current kept_text (terminates with a `<state_K> [list]\n` marker;
        # no `</answer>` block, so it'll score 0 by compute_score / be flagged
        # as NO-ANSWER by the structural PRM, same as the baseline path's
        # max-token cut-off behavior).
        return kept_text, logprob
