"""Local-disk IPO dataset wrapper.

Inherits ipo_trainer.ipo_dataset.IPODataset so we reuse its collate_fn and
the chat-template map_fn unchanged. The single override is `__init__` —
load via `load_from_disk` (HF DatasetDict on disk) instead of
`load_dataset` (HF hub).

`ipo_trainer/` is NOT modified; we only import from it.
"""
from __future__ import annotations

import os

from torch.utils.data import DataLoader
from datasets import load_from_disk

from ipo_trainer.ipo_dataset import IPODataset, get_map_fn


class LocalIPODataset(IPODataset):
    def __init__(
        self,
        dataset_name,
        tokenizer,
        prompt_key="query",
        response_w_key="response_ws",
        response_l_key="response_ls",
        max_prompt_length=512,
        max_response_length=1024,
        padding=True,
        truncation=True,
        num_proc=os.cpu_count(),
        split="train",
    ):
        # Mirror parent attribute setup (we skip its __init__ because the only
        # difference is the load path).
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.prompt_key = prompt_key
        self.response_w_key = response_w_key
        self.response_l_key = response_l_key
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.padding = padding
        self.truncation = truncation
        self.num_proc = num_proc
        self.split = split

        # The only behavioral difference: load_from_disk vs load_dataset.
        self.dataset = load_from_disk(dataset_name)[split]

        # Apply the same chat-template wrap the upstream IPODataset uses.
        map_fn = get_map_fn(
            self.tokenizer, self.prompt_key, self.response_w_key, self.response_l_key
        )
        self.dataset = self.dataset.map(
            map_fn, num_proc=self.num_proc, desc="Applying chat template"
        )

        self.all_prompts = self.dataset[self.prompt_key]
        self.all_responses_w = self.dataset[self.response_w_key]
        self.all_responses_l = self.dataset[self.response_l_key]


def get_local_dataloaders(
    dataset_name,
    tokenizer,
    max_prompt_length=512,
    max_response_length=1024,
    padding=True,
    truncation=True,
    num_proc=os.cpu_count(),
    batch_size=16,
    splits=("train", "test"),
    num_workers=2,
    pin_memory=True,
    drop_last=True,
    prompt_key="query",
    response_w_key="response_ws",
    response_l_key="response_ls",
    shuffle=True,
    gradient_accumulation_steps=1,
):
    """Mirror ipo_trainer.ipo_dataset.get_dataloaders but with LocalIPODataset."""
    assert batch_size % gradient_accumulation_steps == 0
    micro_bs = batch_size // gradient_accumulation_steps
    out = {}
    for split in splits:
        ds = LocalIPODataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            prompt_key=prompt_key,
            response_w_key=response_w_key,
            response_l_key=response_l_key,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            padding=padding,
            truncation=truncation,
            num_proc=num_proc,
            split=split,
        )
        out[split] = DataLoader(
            ds,
            batch_size=micro_bs,
            shuffle=(shuffle if split == "train" else False),
            collate_fn=ds.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(drop_last if split == "train" else False),
        )
    return out
