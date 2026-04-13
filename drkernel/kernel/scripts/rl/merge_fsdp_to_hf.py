#!/usr/bin/env python3
"""Merge FSDP sharded checkpoints into HuggingFace safetensors format."""

import argparse
import json
import os
import shutil
from collections import OrderedDict

import torch
from safetensors.torch import save_file


def merge_fsdp_to_hf(ckpt_dir: str, output_dir: str, world_size: int = 16):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {world_size} FSDP shards from {ckpt_dir}...")

    # With FSDP full_state_dict_type, rank 0 has the complete state dict.
    # Try rank 0 first.
    rank0_path = os.path.join(ckpt_dir, f"model_world_size_{world_size}_rank_0.pt")
    if not os.path.exists(rank0_path):
        raise FileNotFoundError(f"Rank 0 shard not found: {rank0_path}")

    full_state_dict = torch.load(rank0_path, map_location="cpu", weights_only=False)
    print(f"Loaded rank 0: {len(full_state_dict)} keys")

    # Copy tokenizer/config files from huggingface subdir
    src_hf = os.path.join(ckpt_dir, "huggingface")
    if os.path.isdir(src_hf):
        for f in os.listdir(src_hf):
            src = os.path.join(src_hf, f)
            dst = os.path.join(output_dir, f)
            if not os.path.exists(dst) and os.path.isfile(src):
                shutil.copy2(src, dst)
        print(f"Copied config/tokenizer files from {src_hf}")

    # Save as sharded safetensors
    max_shard_size = 5 * 1024 * 1024 * 1024  # 5GB per shard
    current_shard = OrderedDict()
    current_size = 0
    shard_files = []
    index = {"metadata": {"total_size": 0}, "weight_map": {}}

    for key, tensor in full_state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()
        index["metadata"]["total_size"] += tensor_size

        if current_size + tensor_size > max_shard_size and current_shard:
            shard_name = f"model-{len(shard_files)+1:05d}-of-PLACEHOLDER.safetensors"
            save_file(current_shard, os.path.join(output_dir, shard_name))
            shard_files.append(shard_name)
            for k in current_shard:
                index["weight_map"][k] = shard_name
            current_shard = OrderedDict()
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_size

    # Last shard
    if current_shard:
        shard_name = f"model-{len(shard_files)+1:05d}-of-PLACEHOLDER.safetensors"
        save_file(current_shard, os.path.join(output_dir, shard_name))
        shard_files.append(shard_name)
        for k in current_shard:
            index["weight_map"][k] = shard_name

    # Fix placeholder in filenames
    total = len(shard_files)
    total_str = f"{total:05d}"
    for old_name in shard_files:
        new_name = old_name.replace("PLACEHOLDER", total_str)
        if old_name != new_name:
            os.rename(
                os.path.join(output_dir, old_name),
                os.path.join(output_dir, new_name),
            )
            for k in index["weight_map"]:
                if index["weight_map"][k] == old_name:
                    index["weight_map"][k] = new_name

    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"Saved {total} shards ({index['metadata']['total_size'] / 1e9:.1f} GB) to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True, help="FSDP actor checkpoint directory")
    parser.add_argument("--output-dir", required=True, help="Output HF directory")
    parser.add_argument("--world-size", type=int, default=16)
    args = parser.parse_args()
    merge_fsdp_to_hf(args.ckpt_dir, args.output_dir, args.world_size)
