import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.encoder_utils import build_self_attn_cond_masks
from utils.logging_utils import log_for_0


def get_pad_token_id(tokenizer, pad_token: str = "pad") -> int:
    """Resolve the token id used for padding, optionally using EOS as pad."""
    token_id = tokenizer.eos_token_id if pad_token == "eos" else tokenizer.pad_token_id
    if token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id.")
    return token_id


def prepare_batch(batch: Dict) -> Dict:
    """Convert collated NumPy arrays to tensors, preserving text metadata."""
    return {
        key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        for key, value in batch.items()
    }


def pad_and_truncate(ids_list, target_len, pad_token_id):
    """Pad or truncate sequences to target_len, return stacked array and lengths."""
    padded, lengths = [], []
    for ids in ids_list:
        orig_len = min(len(ids), target_len)
        ids = ids[:target_len]
        if orig_len < target_len:
            ids = np.concatenate([ids, np.full(target_len - orig_len, pad_token_id, dtype=ids.dtype)])
        padded.append(ids)
        lengths.append(orig_len)
    return np.stack(padded), np.array(lengths)


def get_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    max_seq_length: int = 512,
    pad_token_id: int = 0,
    max_input_seq_length: Optional[int] = None,
    distributed: bool = True,
    encoder_mask_2d: bool = False,
):
    """Create a DataLoader."""

    def collate_fn(batch_list):
        input_ids_list = [np.asarray(item["input_ids"], dtype=np.int64) for item in batch_list]

        if "condition_input_ids" in batch_list[0]:
            seq_list, cond_lens = [], []
            for item in batch_list:
                cond = np.asarray(item["condition_input_ids"], dtype=np.int64)[:max_input_seq_length]
                inp = np.asarray(item["input_ids"], dtype=np.int64)
                seq_list.append(np.concatenate([cond, inp]))
                cond_lens.append(len(cond))
            cond_lens = np.array(cond_lens)
        else:
            seq_list = input_ids_list
            cond_lens = np.zeros(len(input_ids_list), dtype=np.int32)

        ids, total_lens = pad_and_truncate(seq_list, max_seq_length, pad_token_id)
        pos = np.arange(max_seq_length)[None, :]
        is_cond = pos < cond_lens[:, None]
        is_valid = pos < total_lens[:, None]
        if encoder_mask_2d:
            # GPT-2 supplies the causal mask; only key padding is needed here.
            encoder_attn = attn = is_valid.astype(np.float32)
            pred = is_cond.astype(np.float32)
        else:
            encoder_attn, attn, pred = build_self_attn_cond_masks(is_cond, is_valid)
        result = {
            "input_ids": ids,
            "encoder_attention_mask": encoder_attn,
            "attention_mask": attn,
            "cond_seq_mask": pred,
        }
        for key in ("index", "input", "target"):
            if key in batch_list[0]:
                result[key] = [item[key] for item in batch_list]
        return result

    common = dict(
        batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn,
        drop_last=drop_last, persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    if distributed:
        distributed_run = dist.is_available() and dist.is_initialized()
        sampler = DistributedSampler(
            dataset, num_replicas=dist.get_world_size() if distributed_run else 1,
            rank=dist.get_rank() if distributed_run else 0,
            shuffle=shuffle, drop_last=drop_last,
        )
        return DataLoader(dataset, sampler=sampler, **common)
    return DataLoader(dataset, shuffle=shuffle, **common)


def load_jsonl_dataset(path, tokenizer):
    """Load a JSONL eval set (one `{input, output}` example per line)."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append({
                "index": i,
                "input": data["input"],
                "target": data["output"],
                "condition_input_ids": tokenizer(data["input"], add_special_tokens=False)["input_ids"],
                "input_ids": tokenizer(data["output"], add_special_tokens=False)["input_ids"],
            })
    return examples


# ============================================
# Dataset loading
# ============================================

def _looks_like_save_to_disk_arrow(ds) -> bool:
    """Detect HF datasets uploaded via `save_to_disk` (returns 1-row of metadata)."""
    return (
        len(ds) == 1
        and any(c.startswith("_") for c in ds.column_names)
        and not any(not c.startswith("_") for c in ds.column_names)
    )


def load_dataset_split(path: str, *, split=None):
    """Load one local or Hub dataset split; Parquet downloads select only that split."""
    from datasets import (
        Dataset, DatasetDict, concatenate_datasets,
        load_dataset as hf_load_dataset, load_dataset_builder, load_from_disk,
    )

    def select_split(dataset):
        if not isinstance(dataset, DatasetDict):
            return dataset
        splits = list(dataset.keys())
        if split is not None:
            if split not in dataset:
                raise ValueError(f"Unknown split {split!r} at {path!r}; available: {splits}.")
            return dataset[split]
        if len(splits) != 1:
            raise ValueError(f"Expected dataset at {path!r} to have a single split, got {splits}. "
                             "Set data_split or eval_data_split explicitly.")
        return dataset[splits[0]]

    def load_from_hub(cache_dir=None):
        cache_kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
        if split is None:
            return hf_load_dataset(path, **cache_kwargs)
        builder = load_dataset_builder(path, **cache_kwargs)
        if builder.info.builder_name == "parquet":
            files = builder.config.data_files
            if split not in files:
                raise ValueError(f"Unknown split {split!r} at {path!r}; available: {list(files)}.")
            # A split= argument alone still prepares every configured split. Use
            # only the selected files, retaining the dataset's declared features.
            return hf_load_dataset("parquet", data_files={split: files[split]}, split=split,
                                   features=builder.info.features, **cache_kwargs)
        return hf_load_dataset(path, split=split, **cache_kwargs)

    ds = None
    if os.path.isdir(path):
        try:
            ds = load_from_disk(path)
        except FileNotFoundError:
            # A saved dataset with missing shards must not load only the remaining files.
            if any((Path(path) / name).exists() for name in ("state.json", "dataset_dict.json")):
                raise
            shards = sorted(Path(path).glob("*.arrow"))
            if not shards:
                raise
            ds = concatenate_datasets([Dataset.from_file(str(shard)) for shard in shards])
    elif os.path.isfile(path) and path.endswith(".arrow"):
        ds = Dataset.from_file(path)
    else:
        try:
            ds = load_from_hub()
        except PermissionError as error:
            from datasets import config as datasets_config

            cache_root = Path(os.path.abspath(Path(datasets_config.HF_DATASETS_CACHE).expanduser()))
            fallback = Path.home() / ".cache" / "huggingface" / "datasets"
            if (error.filename is None or fallback == cache_root
                    or not Path(os.path.abspath(error.filename)).is_relative_to(cache_root)):
                raise
            log_for_0(
                f"Dataset cache is not writable at {error.filename}; retrying with {fallback}. "
                "Set HF_DATASETS_CACHE to choose another writable directory.",
                level=logging.WARNING,
            )
            ds = load_from_hub(cache_dir=fallback)

    ds = select_split(ds)

    if _looks_like_save_to_disk_arrow(ds):
        from huggingface_hub import snapshot_download
        log_for_0(
            f"Dataset at {path!r} looks like a save_to_disk-format HF repo; "
            f"re-downloading via snapshot_download + load_from_disk."
        )
        local_dir = snapshot_download(repo_id=path, repo_type="dataset")
        ds = select_split(load_from_disk(local_dir))

    ds.set_format(type="numpy", columns=ds.column_names)
    return ds


def load_dataset(config, tokenizer=None):
    """Resolve config.data_path / config.eval_data_path into train/eval datasets."""
    log_for_0(f"Loading dataset from {config.data_path}...")
    train_dataset = load_dataset_split(config.data_path, split=config.data_split)
    log_for_0(f"Train size: {len(train_dataset)}")

    eval_dataset = None
    if config.eval_data_path:
        if config.eval_data_path.endswith(".jsonl"):
            if tokenizer is None:
                raise ValueError("JSONL evaluation data requires a tokenizer")
            eval_dataset = load_jsonl_dataset(config.eval_data_path, tokenizer)
        else:
            eval_dataset = load_dataset_split(config.eval_data_path, split=config.eval_data_split)
        log_for_0(f"Eval size: {len(eval_dataset)}")
    else:
        log_for_0("No eval dataset")
    return train_dataset, eval_dataset
