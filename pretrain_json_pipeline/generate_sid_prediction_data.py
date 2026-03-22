#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


INPUT_FORMAT_EXAMPLE = """
Input file formats
==================

1. user_sequence.txt
Each line is one user sequence:
user_id item_id_1 item_id_2 item_id_3 ...

Example:
1 101 205 309 888
2 205 777

Meaning:
- column 1 is user_id
- remaining columns are item ids in chronological order

2. item_meta.with_sid.json
Top-level structure is a dict: item_id -> item metadata

Example:
{
  "101": {
    "title": "USB C Cable",
    "categories": "Electronics|Cables",
    "sid": "<|sid_begin|><s_a_3><s_b_19><s_c_8><s_d_44><|sid_end|>"
  },
  "205": {
    "title": "Fast Charger",
    "categories": "Electronics|Chargers",
    "sid": "<|sid_begin|><s_a_5><s_b_9><s_c_11><s_d_20><|sid_end|>"
  }
}
"""


OUTPUT_FORMAT_EXAMPLE = """
Output parquet row examples
===========================

train.parquet row example:
{
  "user_id": "1",
  "description": "The user has purchased the following items: <|sid_begin|><s_a_3><s_b_19><s_c_8><s_d_44><|sid_end|>; <|sid_begin|><s_a_5><s_b_9><s_c_11><s_d_20><|sid_end|>;",
  "groundtruth": "<|sid_begin|><s_a_7><s_b_1><s_c_6><s_d_13><|sid_end|>"
}

valid.parquet row example:
{
  "user_id": "1",
  "description": "The user has purchased the following items: <|sid_begin|><s_a_3><s_b_19><s_c_8><s_d_44><|sid_end|>; <|sid_begin|><s_a_5><s_b_9><s_c_11><s_d_20><|sid_end|>; <|sid_begin|><s_a_7><s_b_1><s_c_6><s_d_13><|sid_end|>;",
  "groundtruth": "<|sid_begin|><s_a_2><s_b_4><s_c_10><s_d_18><|sid_end|>"
}

test.parquet row example:
{
  "user_id": "1",
  "description": "The user has purchased the following items: <|sid_begin|><s_a_3><s_b_19><s_c_8><s_d_44><|sid_end|>; <|sid_begin|><s_a_5><s_b_9><s_c_11><s_d_20><|sid_end|>; <|sid_begin|><s_a_7><s_b_1><s_c_6><s_d_13><|sid_end|>; <|sid_begin|><s_a_2><s_b_4><s_c_10><s_d_18><|sid_end|>;",
  "groundtruth": "<|sid_begin|><s_a_6><s_b_15><s_c_21><s_d_31><|sid_end|>"
}

Split rule:
- train: remove the last 2 items first, then use the remaining prefix to predict its final item
- valid: remove the last 1 item first, then use the remaining prefix to predict its final item
- test: use the full prefix to predict the original final item
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SID prediction train/valid/test parquet files from user_sequence.txt and item_meta.with_sid.json."
    )
    parser.add_argument("--user_sequence", type=Path, required=True)
    parser.add_argument("--item_meta_with_sid", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--train_name", type=str, default="training_prediction_sid_data_train.parquet")
    parser.add_argument("--valid_name", type=str, default="training_prediction_sid_data_valid.parquet")
    parser.add_argument("--test_name", type=str, default="training_prediction_sid_data_test.parquet")
    parser.add_argument("--print_format_example", action="store_true")
    return parser.parse_args()


def load_item_meta(item_meta_with_sid: Path) -> dict:
    print(f"Loading item metadata with sid: {item_meta_with_sid}")
    with item_meta_with_sid.open("r", encoding="utf-8") as f:
        item_meta = json.load(f)
    print(f"Item metadata count: {len(item_meta)}")
    return item_meta


def extract_sid_sequence(item_ids: list[str], user_id: str, item_meta: dict) -> list[str]:
    sid_sequence: list[str] = []
    for item_id in item_ids:
        item_info = item_meta.get(item_id)
        if not item_info:
            print(f"Warning: item_id {item_id} for user {user_id} not found in item_meta.with_sid.json, skipping.")
            continue
        sid = item_info.get("sid")
        if not sid:
            print(f"Warning: item_id {item_id} for user {user_id} missing sid field, skipping.")
            continue
        sid_sequence.append(sid)
    return sid_sequence


def build_dataset_entry(user_id: str, sid_sequence: list[str], tail_remove_count: int) -> dict | None:
    if len(sid_sequence) <= tail_remove_count + 1:
        return None

    candidate_sequence = (
        sid_sequence[: len(sid_sequence) - tail_remove_count]
        if tail_remove_count > 0
        else sid_sequence
    )
    if len(candidate_sequence) < 2:
        return None

    groundtruth = candidate_sequence[-1]
    description_sids = candidate_sequence[:-1]
    if not description_sids:
        return None

    description = "The user has purchased the following items: " + "; ".join(description_sids) + ";"
    return {
        "user_id": user_id,
        "description": description,
        "groundtruth": groundtruth,
    }


def build_output_paths(args):
    output_dir = args.output_dir if args.output_dir else args.user_sequence.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        output_dir / args.train_name,
        output_dir / args.valid_name,
        output_dir / args.test_name,
    )


def preview(df: pd.DataFrame, name: str) -> None:
    print(f"\n{name} first 2 rows preview:")
    if df.empty:
        print("empty dataframe")
        return
    for _, row in df.head(2).iterrows():
        print(f"user_id: {row['user_id']}")
        print(f"description: {row['description']}")
        print(f"groundtruth: {row['groundtruth']}")


def generate_sid_prediction_data(
    user_sequence_file: Path,
    item_meta_with_sid: Path,
    output_train: Path,
    output_valid: Path,
    output_test: Path,
) -> None:
    item_meta = load_item_meta(item_meta_with_sid)

    print(f"Loading user sequence file: {user_sequence_file}")
    with user_sequence_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"User sequence lines: {len(lines)}")

    train_rows: list[dict] = []
    valid_rows: list[dict] = []
    test_rows: list[dict] = []

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        elements = line.split()
        if len(elements) <= 1:
            continue

        user_id = elements[0]
        item_ids = elements[1:]
        sid_sequence = extract_sid_sequence(item_ids, user_id, item_meta)
        if not sid_sequence:
            continue

        train_entry = build_dataset_entry(user_id, sid_sequence, tail_remove_count=2)
        if train_entry:
            train_rows.append(train_entry)

        valid_entry = build_dataset_entry(user_id, sid_sequence, tail_remove_count=1)
        if valid_entry:
            valid_rows.append(valid_entry)

        test_entry = build_dataset_entry(user_id, sid_sequence, tail_remove_count=0)
        if test_entry:
            test_rows.append(test_entry)

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} lines...")

    df_train = pd.DataFrame(train_rows)
    df_valid = pd.DataFrame(valid_rows)
    df_test = pd.DataFrame(test_rows)

    print(f"Training set entries: {len(df_train)}")
    print(f"Validation set entries: {len(df_valid)}")
    print(f"Test set entries: {len(df_test)}")

    print(f"Saving training set to: {output_train}")
    df_train.to_parquet(output_train, engine="pyarrow", index=False)

    print(f"Saving validation set to: {output_valid}")
    df_valid.to_parquet(output_valid, engine="pyarrow", index=False)

    print(f"Saving test set to: {output_test}")
    df_test.to_parquet(output_test, engine="pyarrow", index=False)

    preview(df_train, "Training set")
    preview(df_valid, "Validation set")
    preview(df_test, "Test set")


def main():
    args = parse_args()

    if args.print_format_example:
        print(INPUT_FORMAT_EXAMPLE.strip())
        print()
        print(OUTPUT_FORMAT_EXAMPLE.strip())
        print()

    output_train, output_valid, output_test = build_output_paths(args)
    generate_sid_prediction_data(
        user_sequence_file=args.user_sequence,
        item_meta_with_sid=args.item_meta_with_sid,
        output_train=output_train,
        output_valid=output_valid,
        output_test=output_test,
    )


if __name__ == "__main__":
    main()
