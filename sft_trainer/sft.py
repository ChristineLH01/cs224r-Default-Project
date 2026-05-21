"""Starter SFT training entrypoint for the class project.

This file is intentionally incomplete. Students are expected to implement
`train(...)` while reusing the data/model setup provided here.
"""

import sys
from pathlib import Path

# Allow `python sft_trainer/sft.py` to resolve imports from project root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import gc
import argparse
import os
from sft_trainer.sft_dataset import get_dataloaders
import wandb
import torch.nn.functional as F
import tqdm.auto as tqdm
# os.environ['WANDB_MODE'] = 'offline'

def get_model(model_name, device='cuda', use_gradient_checkpointing=True):
    """Load policy model + tokenizer for SFT training."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # SELS special tokens — register all 10 (state_0..4, backtrack_0..4) so they
    # tokenize as single atomic IDs. Idempotent: re-running adds nothing.
    sels_specials = [f"<state_{k}>" for k in range(5)] + [f"<backtrack_{k}>" for k in range(5)]
    n_added = tokenizer.add_special_tokens({"additional_special_tokens": sels_specials})
    if n_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Registered {n_added} SELS special tokens; vocab now {len(tokenizer)}")

    # Enable gradient checkpointing to reduce memory (trades compute for memory)
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    model.train()
    return model, tokenizer

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

# Compute token-level average log-prob loss and metrics
def forward(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    is_response_token = batch["is_response_token"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Shift for next token prediction
    shift_logits = logits[:, :-1, :].contiguous()   # ignore last token no next prediction
    shift_labels = input_ids[:, 1:].contiguous()    # ignore t0 no conditioning token
    shift_mask = (is_response_token[:, 1:] * attention_mask[:, 1:]).contiguous()    # Valid response token

    # Token NLL = -log p(label). Flatten then reshape back to (batch, seq_len)
    token_nll = F.cross_entropy(
        shift_logits.float().view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none',
    ).view(shift_labels.shape)

    mask = shift_mask.float()
    num_tokens = mask.sum()
    loss = (token_nll * mask).sum() / num_tokens

    # Token accuracy
    preds = shift_logits.argmax(dim=-1)
    num_correct = ((preds == shift_labels).float() * mask).sum()
    return loss, num_tokens, num_correct


@torch.no_grad()
def evaluate(model, test_dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    for batch in test_dataloader:
        loss, num_tokens, num_correct = forward(model, batch, device)
        total_loss += loss.item() * num_tokens.item()
        total_correct += num_correct.item()
        total_tokens += num_tokens.item()
    model.train()
    return total_loss / total_tokens, total_correct / total_tokens

def train(
    model,
    tokenizer,
    train_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
    num_epochs,
    device='cuda',
    save_model=1,
    output_dir='sft_model',
    gradient_accumulation_steps=1,
    gradient_clipping=1.0
):
    # TODO(student): implement the SFT optimization loop.
    # Expected high-level flow:
    # 1) Forward pass on `input_ids` and compute token-level log-probs.
    # 2) Mask loss to response tokens only using `is_response_token`.
    # 3) Backprop, optionally clip gradients, then optimizer/scheduler steps.
    # 4) Periodically evaluate on `test_dataloader` and log metrics to W&B.
    # 5) Save checkpoints under `output_dir` when requested.

    best_eval_loss = float('inf')
    total_batch = 0

    for epoch in range(num_epochs):
        tqdm_bar = tqdm.tqdm(train_dataloader, desc=f"epoch {epoch}")

        optimizer.zero_grad()

        # To track effective batch metrics when using gradient accumulation (microbatch)
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        # To track whether save the last model
        last_is_best = False

        # batch is microbatch if gradient_accumulation_steps > 1
        for i, microbatch in enumerate(tqdm_bar):
            loss, num_tokens, num_correct = forward(model, microbatch, device)
            # Accumulate average loss across microbatch if enabled
            (loss / gradient_accumulation_steps).backward()

            total_loss += loss.item() * num_tokens.item()
            total_correct += num_correct.item()
            total_tokens += num_tokens.item()

            # Effective batch after gradient accumulation
            if (i + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                train_loss = total_loss / total_tokens
                train_acc = total_correct / total_tokens
                # Reset totals for next effective batch
                total_loss = total_correct = total_tokens = 0.0

                eval_loss, eval_acc = evaluate(model, test_dataloader, device)

                total_batch += 1
                wandb.log({
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "train/lr": scheduler.get_last_lr()[0],
                    "eval/loss": eval_loss,
                    "eval/accuracy": eval_acc,
                    "train/epoch": epoch,
                    "train/step": total_batch,
                })
                tqdm_bar.set_postfix({"loss": f"{train_loss:.4f}", "acc": f"{train_acc:.4f}", "step": total_batch})

        print(f"Epoch {epoch+1} eval loss: {eval_loss:.4f} | eval acc: {eval_acc:.4f}")

        # Save the best model
        if save_model and eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            last_is_best = True
            save_checkpoint(model, tokenizer, optimizer, scheduler, os.path.join(output_dir, 'best_model'))

        clear_cache(model)

    # Save the last trained model only if it differs from the best
    if save_model and not last_is_best:
        save_checkpoint(model, tokenizer, optimizer, scheduler, os.path.join(output_dir, 'last_model'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--dataset_name', type=str, default='Asap7772/cog_behav_all_strategies')
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
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_name)
    wandb.config.update(vars(args))

    model, tokenizer = get_model(args.model_name, args.device, use_gradient_checkpointing=args.gradient_checkpointing)

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
        train_dataloader, 
        test_dataloader, 
        optimizer, 
        scheduler, 
        args.num_epochs, 
        args.device, 
        args.save_model, 
        full_output_dir, 
        args.gradient_accumulation_steps, 
        args.gradient_clipping
    )

if __name__ == "__main__":
    main()
