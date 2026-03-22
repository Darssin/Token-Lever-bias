#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def get_special_tokens(num_levels: int = 4, codebook_size: int = 256) -> list[str]:
    special_tokens = ["<|sid_begin|>", "<|sid_end|>"]
    for level in range(num_levels):
        prefix = f"s_{chr(97 + level)}"
        for idx in range(codebook_size):
            special_tokens.append(f"<{prefix}_{idx}>")
    return special_tokens


def round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be a positive integer")
    return ((value + multiple - 1) // multiple) * multiple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Expand tokenizer/model vocabulary with SID special tokens."
    )
    parser.add_argument("--base_model_dir", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--num_levels", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--pad_to_multiple", type=int, default=256)
    return parser.parse_args()


def expand_vocabulary(
    base_model_dir: Path,
    save_dir: Path,
    num_levels: int,
    codebook_size: int,
    pad_to_multiple: int,
) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model config from: {base_model_dir}")
    config = AutoConfig.from_pretrained(base_model_dir)

    print("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(base_model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Available device for quick encode check: {device}")
    if device == "cuda":
        model = model.to(device)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

    new_tokens = get_special_tokens(num_levels=num_levels, codebook_size=codebook_size)
    print(f"Preparing to add {len(new_tokens)} special tokens.")
    tokens_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": new_tokens},
        replace_additional_special_tokens=False,
    )
    print(f"Successfully added {tokens_added} tokens.")

    updated_vocab_size = len(tokenizer)
    target_vocab_size = round_up_to_multiple(updated_vocab_size, pad_to_multiple)
    print(
        f"Current vocab size: {updated_vocab_size}, resizing embeddings to {target_vocab_size}."
    )
    model.resize_token_embeddings(target_vocab_size)
    config.vocab_size = target_vocab_size

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving expanded model to: {save_dir}")
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    config.save_pretrained(save_dir)

    sample_text = "<|sid_begin|><s_a_0><s_b_0><s_c_0><s_d_0><|sid_end|>"
    sample_ids = tokenizer.encode(sample_text, return_tensors="pt").to(device)
    print(f"Sample SID text encoded shape: {sample_ids.shape}")


def main() -> None:
    args = parse_args()
    expand_vocabulary(
        base_model_dir=args.base_model_dir.resolve(),
        save_dir=args.save_dir.resolve(),
        num_levels=args.num_levels,
        codebook_size=args.codebook_size,
        pad_to_multiple=args.pad_to_multiple,
    )


if __name__ == "__main__":
    main()
