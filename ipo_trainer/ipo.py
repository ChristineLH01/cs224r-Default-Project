"""Starter IPO training entrypoint for the class project.

This script wires model loading, data loading, and optimizer setup.
Students are expected to implement `train(...)` for the IPO objective.
"""

import sys
from pathlib import Path

# Allow `python ipo_trainer/ipo.py` to resolve imports from project root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import gc
import argparse
import os
from ipo_trainer.ipo_dataset import get_dataloaders
import wandb
import torch.nn.functional as F
import tqdm.auto as tqdm
import copy
# os.environ['WANDB_MODE'] = 'offline'

def get_model(model_name, device, use_gradient_checkpointing=True):
    """Load trainable policy model and frozen reference model."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Enable gradient checkpointing to reduce memory (trades compute for memory)
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    model.train()

    # IPO compares policy preferences to a fixed baseline policy.
    reference_model = copy.deepcopy(model)
    for param in reference_model.parameters():
        param.requires_grad = False
    reference_model.eval()
    return model, tokenizer, reference_model

def clear_cache(model):
    """Best-effort GPU/CPU cache cleanup between heavy steps."""
    torch.cuda.empty_cache()
    gc.collect()

def save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir):
    """Save model/tokenizer plus optimizer/scheduler states."""
    os.makedirs(output_dir, exist_ok=True)

    model_dir = os.path.join(output_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model and tokenizer saved to {model_dir}")

    torch.save({
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(output_dir, 'train_states.pth'))
    print(f"Model saved to {output_dir}")

def compute_response_logps(model, input_ids, attention_mask, is_response_token, average_logps=False):
    """Compute sum or mean log-prob of response tokens"""
    # Shift for next token prediction
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    labels = input_ids[:, 1:]   
    # Valid response tokens (not padding)
    mask = (is_response_token[:, 1:] * attention_mask[:, 1:]).to(logits.dtype)

    logps = F.log_softmax(logits, dim=-1)
    token_logps = logps.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    sum = (token_logps * mask).sum(dim=-1)
    if average_logps:
        n = mask.sum(dim=-1).clamp(min=1.0)
        return sum / n
    return sum

def forward(model, reference_model, batch, device, beta, average_logps, loss_type):
    """Compute IPO loss and reward margin for a single batch."""
    input_ids_w = batch["input_ids_w"].to(device)
    attention_mask_w = batch["attention_mask_w"].to(device)
    is_response_token_w = batch["is_response_token_w"].to(device)

    input_ids_l = batch["input_ids_l"].to(device)
    attention_mask_l = batch["attention_mask_l"].to(device)
    is_response_token_l = batch["is_response_token_l"].to(device)

    policy_logps_w = compute_response_logps(model, input_ids_w, attention_mask_w, is_response_token_w, average_logps)
    policy_logps_l = compute_response_logps(model, input_ids_l, attention_mask_l, is_response_token_l, average_logps)

    with torch.no_grad():
        ref_logps_w = compute_response_logps(reference_model, input_ids_w, attention_mask_w, is_response_token_w, average_logps)
        ref_logps_l = compute_response_logps(reference_model, input_ids_l, attention_mask_l, is_response_token_l, average_logps)

    h = (policy_logps_w - ref_logps_w) - (policy_logps_l - ref_logps_l)

    if loss_type == 'ipo':
        loss = ((h - 1.0 / (2.0 * beta)) ** 2).mean()
    elif loss_type == 'dpo':
        loss = -F.logsigmoid(beta * h).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    reward_margin = (beta * h).mean()
    return loss, reward_margin

@torch.no_grad()
def evaluate(model, reference_model, dataloader, device, beta, average_logps, loss_type):
    model.eval()
    total_loss = 0.0
    total_margin = 0.0
    n = 0
    for batch in dataloader:
        loss, margin = forward(model, reference_model, batch, device, beta, average_logps, loss_type)
        total_loss += loss.item()
        total_margin += margin.item()
        n += 1
    model.train()
    return total_loss / n, total_margin / n

def train(
    model, 
    tokenizer, 
    reference_model,
    train_dataloader, 
    test_dataloader, 
    optimizer, 
    scheduler, 
    num_epochs, 
    device='cuda', 
    save_model=1, 
    output_dir='sft_model', 
    gradient_accumulation_steps=1, 
    gradient_clipping=1.0,
    beta=0.1,
    average_logps=False,
    loss_type='ipo',
):
    # TODO(student): implement IPO/DPO-style pairwise optimization.
    # Expected high-level flow:
    # 1) Compute policy log-probs for chosen/rejected responses.
    # 2) Compute frozen-reference log-probs for chosen/rejected responses.
    # 3) Build the pairwise objective (IPO or related variant).
    # 4) Apply gradient accumulation, clipping, logging, and checkpointing.
    
    total_batch = 0
    best_eval_loss = float('inf')

    for epoch in range(num_epochs):
        tqdm_bar = tqdm.tqdm(train_dataloader, desc=f"epoch {epoch}")
        optimizer.zero_grad()

        # To track effective batch metrics when using gradient accumulation (microbatch)
        total_loss = 0.0
        total_margin = 0.0

        # To track whether save the last model
        last_is_best = False

        for i, microbatch in enumerate(tqdm_bar):
            loss, reward_margin = forward(
                model, reference_model, microbatch, device, beta, average_logps, loss_type
            )
            # Accumulate average loss across microbatch if enabled
            (loss / gradient_accumulation_steps).backward()

            total_loss += loss.item() 
            total_margin += reward_margin.item() 

            # Effective batch after gradient accumulation
            if (i + 1) % gradient_accumulation_steps == 0:
                if gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                train_loss = total_loss / gradient_accumulation_steps
                train_margin = total_margin / gradient_accumulation_steps

                eval_loss, eval_margin = evaluate(
                    model, reference_model, test_dataloader, device, beta, average_logps, loss_type
                )

                total_batch += 1

                wandb.log({
                    "train/loss": train_loss,
                    "train/reward_margin": train_margin,
                    "eval/loss": eval_loss,
                    "eval/reward_margin": eval_margin,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch+1,
                    "train/step": total_batch,
                })
                tqdm_bar.set_postfix({"loss": f"{train_loss:.4f}", "step": total_batch})

                # Reset totals for next effective batch
                total_loss = 0.0
                total_margin = 0.0

        print(f"Epoch {epoch+1}: eval_loss={eval_loss:.4f}, eval_reward_margin={eval_margin:.4f}")

        # Save the best model
        if save_model and eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            last_is_best = True
            save_checkpoint(model, tokenizer, optimizer, scheduler, os.path.join(output_dir, 'best'))

        clear_cache(model)

    # Save the last trained model only if it differs from the best
    if save_model and not last_is_best:
        save_checkpoint(model, tokenizer, optimizer, scheduler, os.path.join(output_dir, 'last_model'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--dataset_name', type=str, default='asingh15/countdown_tasks_3to4-dpo')
    parser.add_argument('--output_dir', type=str, default='sft_model')
    parser.add_argument('--max_prompt_length', type=int, default=512)
    parser.add_argument('--max_response_length', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb_project', type=str, default='sft_default_project')
    parser.add_argument('--wandb_name', type=str, default='test')
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', type=int, default=1)
    parser.add_argument('--gradient_clipping', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--average_logps', type=int, default=0)
    parser.add_argument('--loss_type', type=str, default='dpo')
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_name)
    wandb.config.update(vars(args))

    model, tokenizer, reference_model = get_model(args.model_name, args.device, use_gradient_checkpointing=args.gradient_checkpointing)

    dataloaders = get_dataloaders(
        dataset_name=args.dataset_name, 
        tokenizer=tokenizer, 
        max_prompt_length=args.max_prompt_length, 
        max_response_length=args.max_response_length, 
        batch_size=args.batch_size, 
        splits=['train', 'test'],
        pin_memory=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    train_dataloader, test_dataloader = dataloaders['train'], dataloaders['test']
    # Scheduler steps happen only after an optimizer step, so account for
    # gradient accumulation when estimating total training steps.
    num_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(num_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps)

    full_output_dir = os.path.join(args.output_dir, args.wandb_project, args.wandb_name)
    os.makedirs(full_output_dir, exist_ok=True)

    train(
        model, 
        tokenizer, 
        reference_model,
        train_dataloader, 
        test_dataloader, 
        optimizer, 
        scheduler, 
        args.num_epochs, 
        args.device, 
        args.save_model, 
        full_output_dir, 
        args.gradient_accumulation_steps, 
        args.gradient_clipping,
        args.beta,
        args.average_logps,
        args.loss_type
    )

if __name__ == "__main__":
    main()
