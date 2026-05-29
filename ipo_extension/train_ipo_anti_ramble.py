"""IPO anti-ramble trainer — wraps ipo_trainer.ipo's get_model/train (untouched)
with a local-disk pair dataset.

Why this lives here, not in ipo_trainer/: keeps the starter code unmodified and
keeps the IPO extension self-contained.
"""
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import os
import torch
from transformers import get_cosine_schedule_with_warmup
import wandb

from ipo_trainer.ipo import get_model, train
from ipo_extension.ipo_dataset_local import get_local_dataloaders


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True,
                   help="Warm-start checkpoint (e.g. Arith PRM cont. latest).")
    p.add_argument("--dataset_name", type=str, required=True,
                   help="Local path to the synthesized IPO pair DatasetDict.")
    p.add_argument("--output_dir", type=str, default="/vol/checkpoints/ipo_anti_ramble")
    p.add_argument("--max_prompt_length", type=int, default=384)
    p.add_argument("--max_response_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--wandb_project", type=str, default="ipo_anti_ramble")
    p.add_argument("--wandb_name", type=str, default="ipo_v1")
    p.add_argument("--save_model", type=int, default=1)
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    p.add_argument("--gradient_clipping", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--average_logps", type=int, default=0)
    p.add_argument("--loss_type", type=str, default="ipo",
                   help="ipo (squared) or dpo (sigmoid). Default ipo per the "
                        "anti-ramble plan.")
    args = p.parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_name)
    wandb.config.update(vars(args))

    model, tokenizer, reference_model = get_model(
        args.model_name,
        args.device,
        use_gradient_checkpointing=bool(args.gradient_checkpointing),
    )

    dataloaders = get_local_dataloaders(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    train_dl, test_dl = dataloaders["train"], dataloaders["test"]
    print(f"train pairs: {len(train_dl.dataset)} | test pairs: {len(test_dl.dataset)}")

    num_steps = len(train_dl) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(num_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps
    )

    full_output_dir = os.path.join(args.output_dir, args.wandb_project, args.wandb_name)
    os.makedirs(full_output_dir, exist_ok=True)

    train(
        model,
        tokenizer,
        reference_model,
        train_dl,
        test_dl,
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
        args.loss_type,
    )


if __name__ == "__main__":
    main()
