#!/usr/bin/env python3

import argparse
import os
import pickle
import re
from collections import defaultdict

import pandas as pd
from transformers import AutoTokenizer


SID_PATTERN = r'<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>'


def extract_all_sids_from_text(text):
    return re.findall(SID_PATTERN, text or "")


def extract_sid_from_text(text):
    match = re.search(SID_PATTERN, text or "")
    if match:
        return match.group(0)
    return (text or "").strip()


def build_global_trie(test_parquet_file, model_path, output_file):
    print(f"Loading test data from: {test_parquet_file}")
    df = pd.read_parquet(test_parquet_file)
    print(f"Total samples in test set: {len(df)}")

    required_cols = ["input", "output"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found. Available: {list(df.columns)}")

    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Extracting all SIDs from test set (input + output)...")
    valid_sids = set()
    for _, row in df.iterrows():
        for sid in extract_all_sids_from_text(row["input"]):
            valid_sids.add(sid)

        output_sid = extract_sid_from_text(row["output"])
        if output_sid and "<|sid_begin|>" in output_sid and "<|sid_end|>" in output_sid:
            valid_sids.add(output_sid)

    print(f"Found {len(valid_sids)} unique valid SIDs in test set")

    print("Converting SIDs to token sequences...")
    sid_token_sequences = []
    for sid in valid_sids:
        sid_token_sequences.append(tokenizer.encode(sid, add_special_tokens=False))

    exact_trie = defaultdict(lambda: defaultdict(set))
    max_length = max(len(seq) for seq in sid_token_sequences) if sid_token_sequences else 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    for seq in sid_token_sequences:
        for pos in range(len(seq)):
            current_token = seq[pos]
            if pos + 1 < len(seq):
                exact_trie[pos][current_token].add(seq[pos + 1])
            else:
                exact_trie[pos][current_token].add(eos_id)

    final_exact_trie = {}
    for pos, token_map in exact_trie.items():
        final_exact_trie[pos] = {}
        for token_id, next_tokens in token_map.items():
            final_exact_trie[pos][token_id] = list(next_tokens)

    trie_data = {
        "exact_trie": final_exact_trie,
        "valid_sids": valid_sids,
        "valid_sid_tokens": sid_token_sequences,
        "tokenizer_name": model_path,
        "total_samples": len(df),
        "search_space_size": len(valid_sids),
        "max_length": max_length,
        "trie_type": "exact",
    }

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(trie_data, f)

    print(f"Exact trie saved to: {output_file}")
    return trie_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute global trie for only-sid evaluation")
    parser.add_argument("--test_parquet_file", type=str, required=True, help="Only-sid test parquet file")
    parser.add_argument("--model_path", type=str, required=True, help="Model path for tokenizer")
    parser.add_argument("--output_file", type=str, default="./global_trie_onlysid.pkl", help="Output pickle file")
    args = parser.parse_args()

    build_global_trie(args.test_parquet_file, args.model_path, args.output_file)
