#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grc_pipeline_utils import (
    build_base_sid_prompt,
    build_brand_reflection,
    build_generation_prompt,
    build_leaf_category_reflection,
    build_localization_reflection,
    build_metadata_lookup,
    build_sft_text,
    build_sid_constraint_ids,
    extract_sid_from_text,
    get_grc_special_tokens,
    get_sid_special_tokens,
    make_sid_prefix_allowed_tokens_fn,
    maybe_to_serializable,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build GRC SFT and verl RL datasets from SID interactions.")
    parser.add_argument("--base_model_path", type=Path, required=True)
    parser.add_argument("--interaction_data_path", type=Path, required=True)
    parser.add_argument("--metadata_path", type=Path, required=True)
    parser.add_argument("--sft_output_path", type=Path, required=True)
    parser.add_argument("--verl_output_path", type=Path, default=None)
    parser.add_argument("--summary_output_path", type=Path, default=None)
    parser.add_argument("--metadata_cache_output_path", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_beams", type=int, default=8)
    parser.add_argument("--num_return_sequences", type=int, default=4)
    parser.add_argument("--draft_max_new_tokens", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_levels", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    return None


def chunked(items: Sequence[Any], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def shorten_text(text: Any, max_length: int = 240) -> str:
    text = str(text or "").replace("\n", "\\n")
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def print_preview_rows(title: str, rows: Sequence[Dict[str, Any]], limit: int = 3):
    print(f"\n===== {title} Preview ({min(len(rows), limit)}/{len(rows)}) =====")
    if not rows:
        print("No rows available.")
        return

    for index, row in enumerate(rows[:limit], start=1):
        print(f"[{title} #{index}]")
        if "user_id" in row:
            print(f"user_id: {row.get('user_id', '')}")
        if "input" in row:
            print(f"input: {shorten_text(row.get('input', ''))}")
        if "target_sid" in row:
            print(f"target_sid: {row.get('target_sid', '')}")
        if "draft_sid" in row:
            print(f"draft_sid: {row.get('draft_sid', '')}")
        if "reflection_sequence" in row:
            print(f"reflection_sequence: {row.get('reflection_sequence', '')}")
        if "loc_label_token" in row:
            print(f"loc_label_token: {row.get('loc_label_token', '')}")
        if "leaf_label_token" in row:
            print(
                "leaf_label_token: "
                f"{row.get('leaf_label_token', '')} "
                f"(score={row.get('leaf_severity_score', '')}, bucket={row.get('leaf_severity_bucket', '')})"
            )
        if "brand_label_token" in row:
            print(
                "brand_label_token: "
                f"{row.get('brand_label_token', '')} "
                f"(score={row.get('brand_severity_score', '')}, bucket={row.get('brand_severity_bucket', '')})"
            )
        if "prompt" in row:
            print(f"prompt: {shorten_text(row.get('prompt', ''))}")
        if "ground_truth" in row:
            print(f"ground_truth: {row.get('ground_truth', '')}")
        if "extra_info" in row:
            print(f"extra_info: {shorten_text(row.get('extra_info', ''))}")
        if "text" in row:
            print(f"text: {shorten_text(row.get('text', ''), max_length=360)}")
        print("-" * 80)


def add_grc_tokens_if_missing(tokenizer, model, num_levels: int):
    tokens = get_grc_special_tokens(num_levels=num_levels)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": tokens},
        replace_additional_special_tokens=False,
    )
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
    return added


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


def build_sft_rows(
    interactions: pd.DataFrame,
    sid_lookup: Dict[str, Dict[str, Any]],
    model,
    tokenizer,
    args,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    records = interactions.to_dict("records")
    sft_rows: List[Dict[str, Any]] = []
    verl_rows: List[Dict[str, Any]] = []

    skipped_target_missing = 0
    skipped_target_invalid = 0
    total_candidates = 0

    iterable = chunked(records, args.batch_size)
    try:
        from tqdm.auto import tqdm

        iterable = tqdm(
            iterable,
            total=(len(records) + args.batch_size - 1) // args.batch_size,
            desc="Building GRC datasets",
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
            target_sid = row["output"]
            target_meta = sid_lookup.get(target_sid)
            if extract_sid_from_text(target_sid) is None:
                skipped_target_invalid += 1
                continue
            if target_meta is None:
                skipped_target_missing += 1
                continue

            verl_rows.append(
                {
                    "prompt": build_generation_prompt(row["input"]),
                    "ground_truth": target_sid,
                    "data_source": args.interaction_data_path.stem,
                    "extra_info": json.dumps(
                        {
                            "user_id": str(row.get("user_id", "")),
                            "target_item_id": target_meta["item_id"],
                            "target_leaf_category": target_meta["leaf_category"],
                            "target_brand": target_meta["brand"],
                        },
                        ensure_ascii=False,
                    ),
                }
            )

            for rank, (draft_sid, draft_score) in enumerate(candidate_group, start=1):
                total_candidates += 1
                draft_meta = sid_lookup.get(draft_sid) if draft_sid else None
                localization = build_localization_reflection(
                    candidate_sid=draft_sid or "",
                    target_sid=target_sid,
                    num_levels=args.num_levels,
                )
                leaf_reflection = build_leaf_category_reflection(
                    candidate_meta=draft_meta,
                    target_meta=target_meta,
                )
                brand_reflection = build_brand_reflection(
                    candidate_meta=draft_meta,
                    target_meta=target_meta,
                )

                text = build_sft_text(
                    user_input=row["input"],
                    draft_sid=draft_sid or "",
                    localization_token=localization.label_token,
                    leaf_token=leaf_reflection.label_token,
                    brand_token=brand_reflection.label_token,
                    target_sid=target_sid,
                )

                sft_rows.append(
                    {
                        "user_id": str(row.get("user_id", "")),
                        "input": row["input"],
                        "target_sid": target_sid,
                        "target_item_id": target_meta["item_id"],
                        "target_leaf_category": target_meta["leaf_category"],
                        "target_brand": target_meta["brand"],
                        "draft_sid": draft_sid,
                        "draft_rank": rank,
                        "draft_score": draft_score,
                        "draft_item_id": draft_meta["item_id"] if draft_meta else None,
                        "draft_leaf_category": draft_meta["leaf_category"] if draft_meta else "",
                        "draft_brand": draft_meta["brand"] if draft_meta else "",
                        "draft_brand_source": draft_meta["brand_source"] if draft_meta else "missing",
                        "draft_brand_confidence": maybe_to_serializable(
                            float(draft_meta["brand_confidence"]) if draft_meta else 0.0
                        ),
                        "loc_position": localization.position,
                        "loc_label_token": localization.label_token,
                        "token_hits_draft": localization.token_hits,
                        "token_misses_draft": localization.token_misses,
                        "leaf_label_token": leaf_reflection.label_token,
                        "leaf_match": leaf_reflection.is_match,
                        "leaf_shared_depth": leaf_reflection.shared_depth,
                        "leaf_overlap_ratio": leaf_reflection.overlap_ratio,
                        "leaf_severity_score": leaf_reflection.severity_score,
                        "leaf_severity_bucket": leaf_reflection.severity_bucket,
                        "leaf_target_value": leaf_reflection.target_leaf,
                        "leaf_draft_value": leaf_reflection.draft_leaf,
                        "brand_label_token": brand_reflection.label_token,
                        "brand_match": brand_reflection.is_match,
                        "brand_similarity": brand_reflection.similarity,
                        "brand_severity_score": brand_reflection.severity_score,
                        "brand_severity_bucket": brand_reflection.severity_bucket,
                        "brand_target_value": brand_reflection.target_brand,
                        "brand_draft_value": brand_reflection.draft_brand,
                        "brand_target_source": brand_reflection.target_brand_source,
                        "brand_draft_source": brand_reflection.draft_brand_source,
                        "brand_target_confidence": maybe_to_serializable(
                            brand_reflection.target_brand_confidence
                        ),
                        "brand_draft_confidence": maybe_to_serializable(
                            brand_reflection.draft_brand_confidence
                        ),
                        "reflection_sequence": (
                            f"{localization.label_token}"
                            f"{leaf_reflection.label_token}"
                            f"{brand_reflection.label_token}"
                        ),
                        "reflection_combo": (
                            f"{localization.label_token}|"
                            f"{leaf_reflection.label_token}|"
                            f"{brand_reflection.label_token}"
                        ),
                        "text": text,
                    }
                )

    summary = {
        "num_raw_samples": int(len(interactions)),
        "num_sft_rows": int(len(sft_rows)),
        "num_verl_rows": int(len(verl_rows)),
        "num_generated_candidates": int(total_candidates),
        "skipped_target_missing_in_metadata": int(skipped_target_missing),
        "skipped_target_invalid_sid": int(skipped_target_invalid),
    }
    return sft_rows, verl_rows, summary


def save_normalized_metadata(sid_lookup: Dict[str, Dict[str, Any]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sid, meta in sid_lookup.items():
            payload = dict(meta)
            payload["sid"] = sid
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_summary(sft_df: pd.DataFrame, summary: Dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def count_series(series_name: str):
        if series_name not in sft_df.columns:
            return {}
        counts = sft_df[series_name].astype(str).value_counts(dropna=False).to_dict()
        return {str(key): int(value) for key, value in counts.items()}

    summary = dict(summary)
    if len(sft_df) > 0:
        summary.update(
            {
                "loc_distribution": count_series("loc_label_token"),
                "leaf_distribution": count_series("leaf_label_token"),
                "brand_distribution": count_series("brand_label_token"),
                "leaf_severity_bucket_distribution": count_series("leaf_severity_bucket"),
                "brand_severity_bucket_distribution": count_series("brand_severity_bucket"),
                "reflection_combo_top20": {
                    str(key): int(value)
                    for key, value in sft_df["reflection_combo"].value_counts().head(20).to_dict().items()
                },
            }
        )
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    interactions = pd.read_parquet(args.interaction_data_path)
    required_columns = {"input", "output"}
    missing = required_columns - set(interactions.columns)
    if missing:
        raise ValueError(
            f"Interaction parquet is missing columns: {sorted(missing)}. "
            f"Available columns: {list(interactions.columns)}"
        )
    if args.max_samples is not None:
        interactions = interactions.iloc[: args.max_samples].reset_index(drop=True)

    sid_lookup, _ = build_metadata_lookup(args.metadata_path)
    print(f"Loaded {len(sid_lookup)} SID metadata records from {args.metadata_path}")

    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model_path.resolve()),
        torch_dtype=resolve_dtype(args.dtype),
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model_path.resolve()))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    add_grc_tokens_if_missing(tokenizer, model, num_levels=args.num_levels)

    sft_rows, verl_rows, summary = build_sft_rows(
        interactions=interactions,
        sid_lookup=sid_lookup,
        model=model,
        tokenizer=tokenizer,
        args=args,
    )

    print_preview_rows("SFT", sft_rows)
    print_preview_rows("VERL", verl_rows)

    sft_df = pd.DataFrame(sft_rows)
    args.sft_output_path.parent.mkdir(parents=True, exist_ok=True)
    sft_df.to_parquet(args.sft_output_path, index=False)
    print(f"Saved SFT dataset to {args.sft_output_path} ({len(sft_df)} rows)")

    if args.verl_output_path is not None:
        verl_df = pd.DataFrame(verl_rows)
        args.verl_output_path.parent.mkdir(parents=True, exist_ok=True)
        verl_df.to_parquet(args.verl_output_path, index=False)
        print(f"Saved verl RL dataset to {args.verl_output_path} ({len(verl_df)} rows)")

    metadata_cache_output_path = (
        args.metadata_cache_output_path
        if args.metadata_cache_output_path is not None
        else args.sft_output_path.with_name(f"{args.sft_output_path.stem}.metadata_cache.jsonl")
    )
    save_normalized_metadata(sid_lookup, metadata_cache_output_path)
    print(f"Saved normalized metadata cache to {metadata_cache_output_path}")

    summary_output_path = (
        args.summary_output_path
        if args.summary_output_path is not None
        else args.sft_output_path.with_name(f"{args.sft_output_path.stem}.summary.json")
    )
    save_summary(sft_df, summary, summary_output_path)
    print(f"Saved dataset summary to {summary_output_path}")


if __name__ == "__main__":
    main()
