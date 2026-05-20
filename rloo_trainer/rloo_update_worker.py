"""Ray actor that applies policy-gradient updates for RLOO.

The orchestrator (`rloo.py`) samples responses and computes rewards, then
calls this worker with tokenized sequences to perform gradient updates.

This file is intentionally incomplete. Students are expected to implement
`update(...)` while reusing the data/model/sampling setup provided here.
"""

import os
import warnings
import ray
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import Optional

warnings.filterwarnings("ignore")

@ray.remote(num_gpus=1)
class RLOOUpdateWorker:
    """Owns policy/ref models and optimizer state for RLOO updates."""
    def __init__(
        self, 
        model_path, 
        optimizer_path, 
        scheduler_path,
        tokenizer_path=None, 
        ref_model_path=None,
        batch_size=64,
        gradient_accumulation_steps=1,
        gradient_clipping=1.0,
        group_size=16, 
        entropy_coefficient=0.01, 
        kl_divergence_coefficient=0.0, 
        lr_schedule='constant',
        learning_rate=1e-5, 
        weight_decay=0.01, 
        warmup_ratio=0.0,
        num_training_steps=250,
    ):
        self.model_path = model_path
        self.ref_model_path = ref_model_path if ref_model_path is not None else model_path
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
        self.optimizer_path = optimizer_path
        self.scheduler_path = scheduler_path
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clipping = gradient_clipping
        self.group_size = group_size
        if self.group_size < 2:
            raise ValueError(f"group_size must be >= 2 for RLOO, got {self.group_size}")
        self.entropy_coefficient = entropy_coefficient
        self.kl_divergence_coefficient = kl_divergence_coefficient
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        if warmup_ratio > 0:
            raise NotImplementedError("Warmup ratio > 0 is not supported for constant learning rate schedule")
        self.num_training_steps = num_training_steps

    def tear_down(self):
        """Release model/optimizer objects and clear GPU memory."""
        import gc
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'ref_model'):
            del self.ref_model
        if hasattr(self, 'optimizer'):
            del self.optimizer
        if hasattr(self, 'scheduler'):
            del self.scheduler
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def update_checkpoint_paths(self, model_path, optimizer_path, scheduler_path, load_checkpoint=False):
        """Update output paths (and optionally reload state immediately)."""
        self.model_path = model_path
        self.optimizer_path = optimizer_path
        self.scheduler_path = scheduler_path
        if load_checkpoint:
            self.load_checkpoint()

    def load_checkpoint(self):
        """Load policy model, optional reference model, and optimizer/scheduler."""
        self.tear_down()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        ).to(device="cuda")
        self.model.gradient_checkpointing_enable()

        if self.kl_divergence_coefficient > 0:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.ref_model_path,
                torch_dtype=torch.bfloat16,
            ).to(device="cuda")
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        if self.optimizer_path and self.scheduler_path and os.path.exists(self.optimizer_path) and os.path.exists(self.scheduler_path):
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            self.optimizer.load_state_dict(torch.load(self.optimizer_path))
            if self.lr_schedule == 'constant':
                self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
            else:
                raise ValueError(f"Invalid learning rate schedule: {self.lr_schedule}")
            
            self.scheduler.load_state_dict(torch.load(self.scheduler_path))
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            
            if self.lr_schedule == 'constant':
                self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
            else:
                raise ValueError(f"Invalid learning rate schedule: {self.lr_schedule}")

        self.model.train()

    def save_checkpoint(self):
        """Persist optimizer/scheduler state plus model+tokenizer weights."""
        torch.save(self.optimizer.state_dict(), self.optimizer_path)
        torch.save(self.scheduler.state_dict(), self.scheduler_path)

        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)


    def update_gradient_accumulation(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        is_response_token: np.ndarray,
        rewards: np.ndarray,
        sample_log_probs: Optional[np.ndarray] = None,
        device='cuda',
    ):
        """Split incoming batch into microbatches and call `update(...)`."""
        update_metrics = None
        if self.gradient_accumulation_steps > 1:
            curr_batch_size = input_ids.shape[0]
            assert curr_batch_size % self.gradient_accumulation_steps == 0, (
                f"Flattened batch size {curr_batch_size} must be divisible by gradient_accumulation_steps "
                f"{self.gradient_accumulation_steps}."
            )
            group_per_gradient_accumulation_step = curr_batch_size // self.gradient_accumulation_steps
            # Ensure each microbatch still contains full RLOO groups so the baseline is meaningful
            assert group_per_gradient_accumulation_step % self.group_size == 0, (
                f"Microbatch size {group_per_gradient_accumulation_step} must be divisible by group_size {self.group_size} "
                f"when using gradient_accumulation_steps={self.gradient_accumulation_steps}."
            )
            all_metrics = []
            for i in range(self.gradient_accumulation_steps):
                curr_input_ids = input_ids[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                curr_attention_mask = attention_mask[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                curr_is_response_token = is_response_token[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                curr_rewards = rewards[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                curr_sample_log_probs = None
                if sample_log_probs is not None:
                    curr_sample_log_probs = sample_log_probs[i * group_per_gradient_accumulation_step:(i + 1) * group_per_gradient_accumulation_step]
                
                is_update_step = (i == self.gradient_accumulation_steps - 1)
                curr_update_metrics = self.update(
                    curr_input_ids,
                    curr_attention_mask,
                    curr_is_response_token,
                    curr_rewards,
                    curr_sample_log_probs,
                    is_update_step,
                    device,
                )
                all_metrics.append(curr_update_metrics)
            update_metrics = {}
            for metric_name in all_metrics[0].keys():
                update_metrics[metric_name] = np.mean([metric[metric_name] for metric in all_metrics]).item()
        else:
            update_metrics = self.update(
                input_ids,
                attention_mask,
                is_response_token,
                rewards,
                sample_log_probs,
                True,
                device,
            )

        return update_metrics

    # `is_update_step` is False on intermediate microbatches so we can
    # accumulate gradients before stepping optimizer/scheduler.
    def update(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        is_response_token: np.ndarray,
        rewards: np.ndarray,
        sample_log_probs: Optional[np.ndarray] = None,
        is_update_step: bool = True,
        device='cuda',
    ):
        # TODO(student): implement one RLOO policy update.
        # Inputs arrive flattened as [batch_size * group_size, seq_len].
        # Required pieces:
        # 1) Compute per-token log-probs on target tokens under current policy.
        # 2) Build leave-one-out baseline within each response group.
        # 3) Compute policy-gradient loss using advantages (and importance weights
        #    if sample_log_probs are provided).
        # 4) Add entropy regularization and optional KL penalty to ref model.
        # 5) Backward pass; if `is_update_step`, clip and step optimizer/scheduler.
        # 6) Return scalar metrics used by trainer logging.

        input_ids_t = torch.as_tensor(input_ids, dtype=torch.long, device=device)
        attention_mask_t = torch.as_tensor(attention_mask, dtype=torch.long, device=device)
        is_response_token_t = torch.as_tensor(is_response_token, dtype=torch.long, device=device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)

        bsz_flat = input_ids_t.shape[0]
        assert bsz_flat % self.group_size == 0, (
            f"Flattened batch {bsz_flat} not divisible by group_size {self.group_size}"
        )
        num_groups = bsz_flat // self.group_size

        # Predict token at position t+1 from logits at position t.
        logits = self.model(input_ids=input_ids_t, attention_mask=attention_mask_t).logits
        shift_logits = logits[:, :-1, :].float()
        shift_targets = input_ids_t[:, 1:]
        # A token contributes to the loss only if it is a response token AND attended to.
        loss_mask = (is_response_token_t[:, 1:] * attention_mask_t[:, 1:]).float()

        log_probs_full = F.log_softmax(shift_logits, dim=-1)
        target_log_probs = log_probs_full.gather(-1, shift_targets.unsqueeze(-1)).squeeze(-1)
        target_log_probs = target_log_probs * loss_mask  # zero out non-response positions

        # Sequence logprob under current policy pi_theta (sum over response tokens).
        seq_log_probs = target_log_probs.sum(dim=-1)  # [bsz_flat]

        # Importance weight w = exp(log pi_theta - log mu), clipped in log-space.
        if sample_log_probs is not None:
            mu_log_probs = torch.as_tensor(sample_log_probs, dtype=torch.float32, device=device)
            log_iw = seq_log_probs - mu_log_probs
            # Clip for stability — keeps an outlier rollout from dominating the batch.
            log_iw_clipped = torch.clamp(log_iw, max=2.0)
            iw = torch.exp(log_iw_clipped)
        else:
            iw = torch.ones_like(seq_log_probs)

        # RLOO leave-one-out baseline within each group of `group_size`.
        rewards_grouped = rewards_t.view(num_groups, self.group_size)
        group_sum = rewards_grouped.sum(dim=-1, keepdim=True)
        # baseline_i = mean over the OTHER k-1 samples in the group
        baseline = (group_sum - rewards_grouped) / (self.group_size - 1)
        advantages = (rewards_grouped - baseline).view(bsz_flat)
        advantages = advantages.detach()

        # Per-sequence mean response logprob (normalize so long/short responses are comparable).
        token_count = loss_mask.sum(dim=-1).clamp(min=1.0)
        seq_log_probs_mean = target_log_probs.sum(dim=-1) / token_count

        # Policy gradient surrogate: -E[ w * A * log pi_theta(y|x) ].
        pg_loss = -(iw.detach() * advantages * seq_log_probs_mean).mean()

        # Entropy bonus: -sum_v p log p, averaged over response tokens.
        probs_full = log_probs_full.exp()
        token_entropy = -(probs_full * log_probs_full).sum(dim=-1)  # [bsz, seq-1]
        entropy = (token_entropy * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
        entropy_loss = -self.entropy_coefficient * entropy

        # KL-to-ref penalty (exact token-level KL on response positions).
        kl_loss_val = torch.zeros((), device=device)
        if self.kl_divergence_coefficient > 0 and hasattr(self, 'ref_model'):
            with torch.no_grad():
                ref_logits = self.ref_model(input_ids=input_ids_t, attention_mask=attention_mask_t).logits
                ref_log_probs_full = F.log_softmax(ref_logits[:, :-1, :].float(), dim=-1)
            # KL(pi_theta || pi_ref) = sum_v p_theta * (log p_theta - log p_ref)
            token_kl = (probs_full * (log_probs_full - ref_log_probs_full)).sum(dim=-1)
            kl_loss_val = (token_kl * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

        total_loss = pg_loss + entropy_loss + self.kl_divergence_coefficient * kl_loss_val
        # Scale so accumulated gradient ≈ one full-batch step.
        scaled_loss = total_loss / self.gradient_accumulation_steps
        scaled_loss.backward()

        weight_mse = 0.0
        weight_max_abs_diff = 0.0
        weight_nonzero_diff_ratio = 0.0
        weight_changed_tensor_ratio = 0.0

        if is_update_step:
            old_params = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

            if self.gradient_clipping is not None and self.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            total_elems = 0
            sq_sum = 0.0
            max_abs = 0.0
            nonzero_diff = 0
            changed_tensors = 0
            total_tensors = 0
            for n, p in self.model.named_parameters():
                if n not in old_params:
                    continue
                diff = (p.detach() - old_params[n]).float()
                total_elems += diff.numel()
                sq_sum += float(diff.pow(2).sum())
                max_abs = max(max_abs, float(diff.abs().max()))
                nz = int((diff != 0).sum())
                nonzero_diff += nz
                total_tensors += 1
                if nz > 0:
                    changed_tensors += 1
            if total_elems > 0:
                weight_mse = sq_sum / total_elems
                weight_nonzero_diff_ratio = nonzero_diff / total_elems
            weight_max_abs_diff = max_abs
            if total_tensors > 0:
                weight_changed_tensor_ratio = changed_tensors / total_tensors
            del old_params

        metrics = {
            "pg_loss": float(pg_loss.detach()),
            "kl_loss": float(kl_loss_val.detach()) if isinstance(kl_loss_val, torch.Tensor) else float(kl_loss_val),
            "importance_weight_mean": float(iw.detach().mean()),
            "lr": float(self.optimizer.param_groups[0]['lr']),
            "weight_mse": weight_mse,
            "weight_max_abs_diff": weight_max_abs_diff,
            "weight_nonzero_diff_ratio": weight_nonzero_diff_ratio,
            "weight_changed_tensor_ratio": weight_changed_tensor_ratio,
        }
        return metrics
