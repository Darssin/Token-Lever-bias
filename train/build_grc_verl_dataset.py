#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from grc_pipeline_utils import (
    SYSTEM_MESSAGE,
    build_metadata_lookup,
    build_sid_constraint_ids,
    extract_sid_from_text,
    get_grc_special_tokens,
    make_sid_prefix_allowed_tokens_fn,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build GRC verl RL dataset from SID interactions.")
    parser.add_argument("--base_model_path", type=Path, required=True)
    parser.add_argument("--interaction_data_path", type=Path, required=True)
    parser.add_argument("--metadata_path", type=Path, required=True)
    parser.add_argument("--verl_output_path", type=Path, required=True)
    parser.add_argument("--metadata_cache_output_path", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_beams", type=int, default=8)
    parser.add_argument("--num_return_sequences", type=int, default=4)
    parser.add_argument("--draft_max_new_tokens", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_levels", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--repair_input", type=Path, default=None)
    parser.add_argument("--repair_output", type=Path, default=None)
    return parser.parse_args()


def get_distributed_context() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def setup_distributed() -> Tuple[int, int, int]:
    rank, world_size, local_rank = get_distributed_context()
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def resolve_dtype(dtype_name: str):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None


def chunked(items: Sequence[Any], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def shard_dataframe(dataframe: pd.DataFrame, rank: int, world_size: int) -> pd.DataFrame:
    if world_size <= 1:
        return dataframe.reset_index(drop=True)
    return dataframe.iloc[rank::world_size].reset_index(drop=True)


def gather_object_list(local_rows: List[Dict[str, Any]], world_size: int) -> List[Dict[str, Any]]:
    if world_size <= 1:
        return local_rows
    gathered: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_rows)
    merged: List[Dict[str, Any]] = []
    for part in gathered:
        if part:
            merged.extend(part)
    return merged


def add_grc_tokens_if_missing(tokenizer, model, num_levels: int):
    tokens = get_grc_special_tokens(num_levels=num_levels)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": tokens},
        replace_additional_special_tokens=False,
    )
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
    return added


def build_base_sid_prompt(user_input: str) -> str:
    return f"""<|im_start|>system
{SYSTEM_MESSAGE}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
<think>

</think>
"""


def build_verl_prompt_messages(user_input: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_input},
    ]


def generate_sid_candidates(
    model,
    tokenizer,
    batch_inputs: Sequence[str],
    num_levels: int,
    codebook_size: int,
    num_beams: int,
    num_return_sequences: int,
    draft_max_new_tokens: int,
) -> List[List[Tuple[Optional[str], float]]]:
    prompts = [build_base_sid_prompt(text) for text in batch_inputs]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    prompt_length = encoded["input_ids"].shape[1]
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    sid_constraints = build_sid_constraint_ids(
        tokenizer=tokenizer,
        num_levels=num_levels,
        codebook_size=codebook_size,
    )
    outputs = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded.get("attention_mask"),
        max_new_tokens=draft_max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=sid_constraints["eos_id"],
        prefix_allowed_tokens_fn=make_sid_prefix_allowed_tokens_fn(
            prompt_length=prompt_length,
            sid_constraints=sid_constraints,
        ),
    )
    generated_ids = outputs.sequences[:, prompt_length:]
    decoded = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    scores = outputs.sequences_scores.detach().cpu().tolist() if outputs.sequences_scores is not None else None

    grouped: List[List[Tuple[Optional[str], float]]] = []
    for batch_index in range(len(batch_inputs)):
        start = batch_index * num_return_sequences
        end = start + num_return_sequences
        per_example: List[Tuple[Optional[str], float]] = []
        seen = set()
        for local_index, text in enumerate(decoded[start:end]):
            sid = extract_sid_from_text(text)
            if sid in seen:
                continue
            seen.add(sid)
            score = float(scores[start + local_index]) if scores is not None else 0.0
            per_example.append((sid, score))
        grouped.append(per_example)
    return grouped


def format_preview_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def print_preview_rows(title: str, rows: Sequence[Dict[str, Any]], limit: int = 3):
    print(f"\n===== {title} Preview ({min(len(rows), limit)}/{len(rows)}) =====")
    if not rows:
        print("No rows available.")
        return
    for index, row in enumerate(rows[:limit], start=1):
        print(f"[{title} #{index}]")
        print("prompt:")
        print(format_preview_value(row.get("prompt", "")))
        print(f"ground_truth: {row.get('ground_truth', '')}")
        print("reward_model:")
        print(format_preview_value(row.get("reward_model", {})))
        print("extra_info:")
        print(format_preview_value(row.get("extra_info", {})))
        print("-" * 80)


def normalize_extra_info(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def normalize_reward_model(value: Any, fallback_ground_truth: Any = None) -> Dict[str, Any]:
    if isinstance(value, dict):
        result = dict(value)
        if "ground_truth" not in result and fallback_ground_truth is not None:
            result["ground_truth"] = fallback_ground_truth
        return result
    if fallback_ground_truth is None:
        return {}
    return {"style": "rule", "ground_truth": fallback_ground_truth}


def repair_verl_parquet(input_path: Path, output_path: Path):
    dataframe = pd.read_parquet(input_path).copy()
    if "extra_info" in dataframe.columns:
        dataframe["extra_info"] = dataframe["extra_info"].map(normalize_extra_info)
    else:
        dataframe["extra_info"] = [{} for _ in range(len(dataframe))]
    ground_truths = dataframe["ground_truth"] if "ground_truth" in dataframe.columns else [None] * len(dataframe)
    if "reward_model" in dataframe.columns:
        dataframe["reward_model"] = [
            normalize_reward_model(value, fallback_ground_truth=ground_truth)
            for value, ground_truth in zip(dataframe["reward_model"], ground_truths)
        ]
    else:
        dataframe["reward_model"] = [
            normalize_reward_model(None, fallback_ground_truth=ground_truth) for ground_truth in ground_truths
        ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)
    print(f"Repaired verl dataset saved to {output_path} ({len(dataframe)} rows)")


def build_verl_rows(
    interactions: pd.DataFrame,
    sid_lookup: Dict[str, Dict[str, Any]],
    model,
    tokenizer,
    args,
    rank: int,
) -> List[Dict[str, Any]]:
    records = interactions.to_dict("records")
    verl_rows: List[Dict[str, Any]] = []

    iterable = chunked(records, args.batch_size)
    try:
        from tqdm.auto import tqdm

        iterable = tqdm(
            iterable,
            total=(len(records) + args.batch_size - 1) // args.batch_size,
            desc=f"Building GRC verl dataset (rank {rank})",
            disable=not is_main_process(rank),
        )
    except Exception:
        pass

    model.eval()
    for batch in iterable:
        batch_inputs = [row["input"] for row in batch]
        candidate_groups = generate_sid_candidates(
            model=model,
            tokenizer=tokenizer,
            batch_inputs=batch_inputs,
            num_levels=args.num_levels,
            codebook_size=args.codebook_size,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            draft_max_new_tokens=args.draft_max_new_tokens,
        )

        for row, candidate_group in zip(batch, candidate_groups):
            del candidate_group
            target_sid = row["output"]
            target_meta = sid_lookup.get(target_sid)
            if extract_sid_from_text(target_sid) is None or target_meta is None:
                continue

            verl_rows.append(
                {
                    "source_row_index": int(row["__row_idx__"]),
                    "prompt": build_verl_prompt_messages(row["input"]),
                    "ground_truth": target_sid,
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": target_sid,
                    },
                    "data_source": args.interaction_data_path.stem,
                    "extra_info": {
                        "index": int(row["__row_idx__"]),
                        "user_id": str(row.get("user_id", "")),
                        "target_item_id": target_meta["item_id"],
                        "target_leaf_category": target_meta["leaf_category"],
                        "target_brand": target_meta["brand"],
                    },
                }
            )

    verl_rows = sorted(verl_rows, key=lambda row: row["source_row_index"])
    for row in verl_rows:
        row.pop("source_row_index", None)
    return verl_rows


def save_normalized_metadata(sid_lookup: Dict[str, Dict[str, Any]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sid, meta in sid_lookup.items():
            payload = dict(meta)
            payload["sid"] = sid
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    if args.repair_input is not None:
        if args.repair_output is None:
            raise ValueError("--repair_output must be provided with --repair_input")
        repair_verl_parquet(args.repair_input, args.repair_output)
        return

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    try:
        interactions = pd.read_parquet(args.interaction_data_path)
        required_columns = {"input", "output"}
        missing = required_columns - set(interactions.columns)
        if missing:
            raise ValueError(f"Interaction parquet missing columns: {sorted(missing)}")
        if args.max_samples is not None:
            interactions = interactions.iloc[: args.max_samples].reset_index(drop=True)
        interactions = interactions.reset_index(drop=True)
        interactions["__row_idx__"] = interactions.index
        total_interactions = len(interactions)
        interactions = shard_dataframe(interactions, rank=rank, world_size=world_size)

        sid_lookup, _ = build_metadata_lookup(args.metadata_path)
        if is_main_process(rank):
            print(f"Loaded {len(sid_lookup)} SID metadata records from {args.metadata_path}")
            print(f"Building verl dataset with world_size={world_size} on device={device}")
            print(f"Total interaction rows: {total_interactions}")
        print(f"[rank {rank}] Local shard rows: {len(interactions)}")

        model = AutoModelForCausalLM.from_pretrained(
            str(args.base_model_path.resolve()),
            torch_dtype=resolve_dtype(args.dtype),
        )
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(str(args.base_model_path.resolve()))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        add_grc_tokens_if_missing(tokenizer, model, num_levels=args.num_levels)

        local_verl_rows = build_verl_rows(
            interactions=interactions,
            sid_lookup=sid_lookup,
            model=model,
            tokenizer=tokenizer,
            args=args,
            rank=rank,
        )

        verl_rows = gather_object_list(local_verl_rows, world_size=world_size)
        if not is_main_process(rank):
            return

        print_preview_rows("VERL", verl_rows)
        verl_df = pd.DataFrame(verl_rows)
        args.verl_output_path.parent.mkdir(parents=True, exist_ok=True)
        verl_df.to_parquet(args.verl_output_path, index=False)
        print(f"Saved verl dataset to {args.verl_output_path} ({len(verl_df)} rows)")

        metadata_cache_output_path = (
            args.metadata_cache_output_path
            if args.metadata_cache_output_path is not None
            else args.verl_output_path.with_name(f"{args.verl_output_path.stem}.metadata_cache.jsonl")
        )
        save_normalized_metadata(sid_lookup, metadata_cache_output_path)
        print(f"Saved normalized metadata cache to {metadata_cache_output_path}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
