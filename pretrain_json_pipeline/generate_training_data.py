#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate align train/valid/test parquet files from user_sequence.txt and item_meta.with_sid.json."
    )
    parser.add_argument("--user_sequence", type=Path, required=True)
    parser.add_argument("--item_meta_with_sid", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--train_name", type=str, default="training_align_data_train.parquet")
    parser.add_argument("--valid_name", type=str, default="training_align_data_valid.parquet")
    parser.add_argument("--test_name", type=str, default="training_align_data_test.parquet")
    return parser.parse_args()


def load_item_meta(item_meta_with_sid: Path) -> dict:
    print(f"Loading item metadata with sid: {item_meta_with_sid}")
    with item_meta_with_sid.open("r", encoding="utf-8") as f:
        item_meta = json.load(f)
    print(f"Item metadata count: {len(item_meta)}")
    return item_meta


def build_description(item_ids: list[str], user_id: str, item_meta: dict) -> tuple[list[str], int]:
    item_descriptions: list[str] = []
    missing_count = 0

    for item_id in item_ids:
        item_info = item_meta.get(item_id)
        if not item_info:
            print(f"Warning: item_id {item_id} for user {user_id} not found in item_meta.with_sid.json, skipping.")
            missing_count += 1
            continue

        sid = item_info.get("sid", "")
        title = item_info.get("title", "")
        categories = item_info.get("categories", "")

        if sid and title and categories:
            item_descriptions.append(f'{sid}, its title is "{title}", its categories are "{categories}"')
        else:
            print(f"Warning: item_id {item_id} for user {user_id} missing sid/title/categories, skipping.")
            missing_count += 1

    return item_descriptions, missing_count


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


def generate_training_data(
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
    missing_items_count = 0

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        elements = line.split()
        if len(elements) <= 1:
            continue

        user_id = elements[0]
        item_ids = elements[1:]

        test_descriptions, test_missing = build_description(item_ids, user_id, item_meta)
        missing_items_count += test_missing
        if test_descriptions:
            test_rows.append({
                "user_id": user_id,
                "description": "The user has purchased the following items: " + "; ".join(test_descriptions) + ";",
            })

        if len(item_ids) > 1:
            valid_descriptions, valid_missing = build_description(item_ids[:-1], user_id, item_meta)
            missing_items_count += valid_missing
            if valid_descriptions:
                valid_rows.append({
                    "user_id": user_id,
                    "description": "The user has purchased the following items: " + "; ".join(valid_descriptions) + ";",
                })

        if len(item_ids) > 2:
            train_descriptions, train_missing = build_description(item_ids[:-2], user_id, item_meta)
            missing_items_count += train_missing
            if train_descriptions:
                train_rows.append({
                    "user_id": user_id,
                    "description": "The user has purchased the following items: " + "; ".join(train_descriptions) + ";",
                })

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} lines...")

    df_train = pd.DataFrame(train_rows)
    df_valid = pd.DataFrame(valid_rows)
    df_test = pd.DataFrame(test_rows)

    print(f"Training set entries: {len(df_train)}")
    print(f"Validation set entries: {len(df_valid)}")
    print(f"Test set entries: {len(df_test)}")
    print(f"Total skipped item references: {missing_items_count}")

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
    output_train, output_valid, output_test = build_output_paths(args)
    generate_training_data(
        user_sequence_file=args.user_sequence,
        item_meta_with_sid=args.item_meta_with_sid,
        output_train=output_train,
        output_valid=output_valid,
        output_test=output_test,
    )


if __name__ == "__main__":
    main()
