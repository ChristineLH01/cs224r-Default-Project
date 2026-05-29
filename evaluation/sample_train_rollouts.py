"""Sample N rollouts per prompt from the TRAIN split for Stage 2 SFT-data
generation. Output JSON format matches `evaluation/countdown_eval.py` so all
existing analyzers (analyze_failure_modes.py, compare_failure_modes.py) work
directly on the result.

Differences from countdown_eval.py:
  - Loads `split='train'`
  - Subsamples the first `--max_prompts` rows (train set is ~490K).
  - Default budget: 8 rollouts × 2000 prompts (favor prompt diversity over
    redundant samples — Stage 2 SFT needs ONE (CORRECT, rambling) pair per
    prompt, not many).
"""
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tenacity import retry, stop_after_attempt, wait_exponential
import argparse
import os
from datasets import load_dataset, Dataset
from evaluation.countdown import compute_score


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def load_checkpoint(model_path, max_model_len, gpu_memory_utilization,
                    max_num_batched_tokens, enable_chunked_prefill, max_num_seqs):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, max_model_len=max_model_len,
              gpu_memory_utilization=gpu_memory_utilization,
              max_num_batched_tokens=max_num_batched_tokens,
              enable_chunked_prefill=enable_chunked_prefill,
              max_num_seqs=max_num_seqs)
    return tokenizer, llm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--max_model_len', type=int, default=2048)
    p.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    p.add_argument('--max_num_batched_tokens', type=int, default=4096)
    p.add_argument('--enable_chunked_prefill', type=bool, default=True)
    p.add_argument('--temperature', type=float, default=0.6)
    p.add_argument('--top_p', type=float, default=0.95)
    p.add_argument('--top_k', type=int, default=20)
    p.add_argument('--min_p', type=float, default=0.0)
    p.add_argument('--max_tokens', type=int, default=1024)
    p.add_argument('--max_num_seqs', type=int, default=16)
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--eval_dataset', type=str, default='asingh15/countdown_tasks_3to4')
    p.add_argument('--output_dir', type=str, default='/vol/sft_data')
    p.add_argument('--output_name', type=str, required=True)
    p.add_argument('--num_responses', type=int, default=8,
                   help='Rollouts per prompt. Default 8 favors prompt diversity over redundancy.')
    p.add_argument('--max_prompts', type=int, default=2000,
                   help='Subsample N training prompts starting at --start_idx (train set is ~490K).')
    p.add_argument('--start_idx', type=int, default=0,
                   help='Skip the first N prompts. Use for incremental sampling: '
                        'relaunch with --start_idx 2000 --output_name ..._p2 to get '
                        'prompts 2000-4000 in a separate file.')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tokenizer, llm = load_checkpoint(args.model_path, args.max_model_len,
                                      args.gpu_memory_utilization,
                                      args.max_num_batched_tokens,
                                      args.enable_chunked_prefill,
                                      args.max_num_seqs)
    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        min_p=args.min_p, max_tokens=args.max_tokens, n=args.num_responses,
        skip_special_tokens=False, stop=["</answer>"],
        include_stop_str_in_output=True)

    loaded_dataset = load_dataset(args.eval_dataset, split='train')
    prompt_df = (loaded_dataset.to_pandas()
                 .iloc[args.start_idx : args.start_idx + args.max_prompts]
                 .reset_index(drop=True))
    print(f"Sampling {args.num_responses} rollouts × {len(prompt_df)} prompts "
          f"(idx {args.start_idx}..{args.start_idx + len(prompt_df) - 1}) "
          f"= {args.num_responses * len(prompt_df)} total rollouts")

    outputs = llm.generate(prompt_df['prompt'].tolist(), sampling_params)

    response, scores = [], []
    for i, output in enumerate(outputs):
        row = prompt_df.iloc[i]
        gt = {'target': row['target'], 'numbers': row['nums']}
        curr_response, curr_scores = [], []
        for o in output.outputs:
            curr_response.append(o.text)
            curr_scores.append(compute_score(o.text, gt))
        response.append(curr_response)
        scores.append(curr_scores)

    prompt_df['response'] = response
    prompt_df['scores']   = scores
    os.makedirs(args.output_dir, exist_ok=True)
    Dataset.from_pandas(prompt_df).to_json(f'{args.output_dir}/{args.output_name}.json')
    print(f"Saved to {args.output_dir}/{args.output_name}.json")
