#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark-style SID beam-search probability analysis.

This script keeps the same prompt format, trie constraint, and beam-search
settings as `eval/test_model_hitrate.py`, then adds:

1. Per-sample label SID four-level conditional probabilities and joint
   probabilities.
2. Per-step beam-search cutoff probabilities, defined as the lowest retained
   cumulative probability among the kept beams after pruning.
3. Visualization comparing label probabilities against beam cutoff thresholds.
"""

import argparse
import json
import logging
import math
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_MESSAGE = (
    "You are a professional recommendation expert who needs to recommend the next "
    "possible purchase for users based on their purchase history. Please predict "
    "the most likely next product that the user will purchase based on the user's "
    "historical purchase information."
)

SID_PATTERN = re.compile(
    r"<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>"
)
LEVEL_PATTERNS = {
    "a": re.compile(r"<s_a_(\d+)>"),
    "b": re.compile(r"<s_b_(\d+)>"),
    "c": re.compile(r"<s_c_(\d+)>"),
    "d": re.compile(r"<s_d_(\d+)>"),
}
LEVELS = ["a", "b", "c", "d"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SID label probabilities and beam-search cutoffs."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Merged model path.")
    parser.add_argument("--test_parquet_file", type=str, required=True, help="Benchmark parquet path.")
    parser.add_argument("--global_trie_file", type=str, required=True, help="Exact trie pickle path.")
    parser.add_argument("--output_dir", type=str, default="./sid_label_beam_outputs", help="Output directory.")
    parser.add_argument("--num_beams", type=int, default=20, help="Beam size, should match benchmark.")
    parser.add_argument("--batch_size", type=int, default=8, help="Analysis batch size.")
    parser.add_argument("--max_new_tokens", type=int, default=6, help="Max new tokens for one SID.")
    parser.add_argument("--sample_num", type=int, default=-1, help="-1 means all samples.")
    parser.add_argument("--sample_offset", type=int, default=0, help="Optional offset into parquet rows.")
    parser.add_argument("--device", type=str, default=None, help="cuda:0 / cpu. Auto if omitted.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype.",
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Benchmark generation temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Benchmark top_p.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("sid_label_beam_probs")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    return logger


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(parquet_path: str, sample_num: int, sample_offset: int) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if sample_offset > 0:
        df = df.iloc[sample_offset:].reset_index(drop=True)
    if sample_num > 0:
        df = df.iloc[:sample_num].reset_index(drop=True)
    return df


def normalize_sample_row(row: pd.Series, sample_id: int) -> Dict[str, Any]:
    input_text = row["input"] if "input" in row else row["description"]
    output_text = row["output"] if "output" in row else row["groundtruth"]
    return {
        "sample_id": sample_id,
        "input_text": input_text,
        "output_text": output_text,
        "user_id": row["user_id"] if "user_id" in row else f"user_{sample_id}",
    }


def extract_sid_from_text(text: str) -> str:
    matches = SID_PATTERN.findall(text or "")
    return matches[0] if matches else ""


def parse_sid_levels(sid_text: str) -> Optional[Dict[str, str]]:
    if not sid_text:
        return None
    result: Dict[str, str] = {}
    for level, pattern in LEVEL_PATTERNS.items():
        match = pattern.search(sid_text)
        if not match:
            return None
        result[level] = f"<s_{level}_{match.group(1)}>"
    return result


def format_chat_prompt(user_content: str) -> str:
    return f"""<|im_start|>system
{SYSTEM_MESSAGE}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
<think>

</think>
"""


def load_model_and_tokenizer(model_path: str, device: Optional[str], dtype_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_map[dtype_name] if device.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model, tokenizer, device


def load_exact_trie(global_trie_file: str) -> Dict[str, Any]:
    with open(global_trie_file, "rb") as f:
        trie_data = pickle.load(f)
    if trie_data.get("trie_type") != "exact":
        raise ValueError("Only exact trie is supported.")
    return trie_data


def build_prefix_allowed_tokens_fn(tokenizer, trie_data: Dict[str, Any]):
    allowed_tokens = trie_data["exact_trie"]
    sep = tokenizer("</think>", add_special_tokens=False)["input_ids"]
    all_vocab_ids = list(tokenizer.get_vocab().values())
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    def find_last_sublist(lst: Sequence[int], sub: Sequence[int]) -> Optional[int]:
        if not sub:
            return None
        n, m = len(lst), len(sub)
        for start in range(n - m, -1, -1):
            if list(lst[start : start + m]) == list(sub):
                return start
        return None

    def prefix_allowed_tokens_fn(batch_id: int, sentence: torch.Tensor) -> List[int]:
        del batch_id
        sentence_list = sentence.tolist()
        pos = find_last_sublist(sentence_list, sep)
        if pos is None:
            return all_vocab_ids

        pos_after_sep = pos + len(sep)
        generated_after_sep = sentence_list[pos_after_sep:]
        current_pos = len(generated_after_sep)

        if current_pos == 0:
            return tokenizer.encode("\n", add_special_tokens=False)

        sid_pos = current_pos - 1
        if sid_pos == 0:
            return list(allowed_tokens.get(0, {}).keys()) or [eos_id]

        if len(generated_after_sep) > sid_pos:
            prev_token = generated_after_sep[sid_pos]
            prev_pos = sid_pos - 1
            if prev_pos in allowed_tokens and prev_token in allowed_tokens[prev_pos]:
                return allowed_tokens[prev_pos][prev_token]

        return [eos_id]

    return prefix_allowed_tokens_fn


def encode_target_sid(tokenizer, sid_text: str) -> List[int]:
    token_ids = tokenizer.encode(sid_text, add_special_tokens=False)
    if len(token_ids) < 6:
        raise ValueError(f"Unexpected SID tokenization for {sid_text}: {token_ids}")
    return token_ids


def build_left_padded_batch(
    sequences: List[List[int]],
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tensors = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
    padded = pad_sequence(tensors, batch_first=True, padding_value=pad_token_id)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    max_len = padded.size(1)
    shifted = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for idx, (seq_tensor, seq_len) in enumerate(zip(tensors, lengths.tolist())):
        shifted[idx, max_len - seq_len :] = seq_tensor
        attention_mask[idx, max_len - seq_len :] = 1
    return shifted.to(device), attention_mask.to(device)


def safe_exp(log_value: float) -> float:
    if math.isinf(log_value) and log_value < 0:
        return 0.0
    return float(math.exp(log_value))


def trace_label_probabilities(
    model,
    tokenizer,
    prompts: List[str],
    target_sids: List[str],
    prefix_allowed_tokens_fn,
) -> List[Dict[str, Any]]:
    prompt_tokens = tokenizer(
        prompts,
        add_special_tokens=False,
        return_attention_mask=True,
    )
    pad_token_id = tokenizer.pad_token_id
    prompt_id_lists: List[List[int]] = []
    for input_ids, attention_mask in zip(prompt_tokens["input_ids"], prompt_tokens["attention_mask"]):
        valid_len = int(sum(attention_mask))
        prompt_id_lists.append(input_ids[-valid_len:])

    target_token_lists = [encode_target_sid(tokenizer, sid_text) for sid_text in target_sids]
    max_steps = max(len(tokens) for tokens in target_token_lists)

    step_log_probs: List[List[Optional[float]]] = [[None] * max_steps for _ in prompts]
    step_probs: List[List[Optional[float]]] = [[None] * max_steps for _ in prompts]

    for step_idx in range(max_steps):
        active_indices = [idx for idx, tokens in enumerate(target_token_lists) if step_idx < len(tokens)]
        if not active_indices:
            continue

        batch_sequences = [
            prompt_id_lists[idx] + target_token_lists[idx][:step_idx]
            for idx in active_indices
        ]
        input_ids, attention_mask = build_left_padded_batch(
            sequences=batch_sequences,
            pad_token_id=pad_token_id,
            device=model.device,
        )

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
            log_probs = torch.log_softmax(logits.float(), dim=-1)

        for local_idx, sample_idx in enumerate(active_indices):
            full_prefix = torch.tensor(batch_sequences[local_idx], dtype=torch.long)
            allowed_tokens = prefix_allowed_tokens_fn(0, full_prefix)
            masked_log_probs = torch.full_like(log_probs[local_idx], float("-inf"))
            allowed_tensor = torch.tensor(allowed_tokens, dtype=torch.long, device=log_probs.device)
            masked_log_probs[allowed_tensor] = log_probs[local_idx, allowed_tensor]
            normalized_log_probs = torch.log_softmax(masked_log_probs, dim=-1)

            target_token_id = target_token_lists[sample_idx][step_idx]
            step_log_prob = float(normalized_log_probs[target_token_id].detach().cpu())
            step_log_probs[sample_idx][step_idx] = step_log_prob
            step_probs[sample_idx][step_idx] = safe_exp(step_log_prob)

    results = []
    for sid_text, token_ids, token_log_probs, token_probs in zip(
        target_sids, target_token_lists, step_log_probs, step_probs
    ):
        token_texts = tokenizer.convert_ids_to_tokens(token_ids)
        cumulative_log = 0.0
        cumulative_probs_all = []
        for log_prob in token_log_probs:
            if log_prob is None:
                cumulative_probs_all.append(None)
                continue
            cumulative_log += log_prob
            cumulative_probs_all.append(safe_exp(cumulative_log))

        level_conditional_probs = {
            level: float(token_probs[idx + 1]) for idx, level in enumerate(LEVELS)
        }
        level_conditional_log_probs = {
            level: float(token_log_probs[idx + 1]) for idx, level in enumerate(LEVELS)
        }

        cumulative_level_log = 0.0
        cumulative_level_probs: Dict[str, float] = {}
        for level in LEVELS:
            cumulative_level_log += level_conditional_log_probs[level]
            cumulative_level_probs[level] = safe_exp(cumulative_level_log)

        full_sid_log_prob = float(sum(log_prob for log_prob in token_log_probs if log_prob is not None))
        four_level_log_prob = float(sum(level_conditional_log_probs[level] for level in LEVELS))

        results.append(
            {
                "target_sid": sid_text,
                "target_token_ids": [int(token_id) for token_id in token_ids],
                "target_tokens": token_texts,
                "full_step_conditional_probs": [float(prob) if prob is not None else None for prob in token_probs],
                "full_step_cumulative_probs": [float(prob) if prob is not None else None for prob in cumulative_probs_all],
                "level_conditional_probs": level_conditional_probs,
                "level_cumulative_probs": cumulative_level_probs,
                "four_level_log_prob": four_level_log_prob,
                "four_level_total_prob": safe_exp(four_level_log_prob),
                "full_sid_log_prob": full_sid_log_prob,
                "full_sid_total_prob": safe_exp(full_sid_log_prob),
            }
        )
    return results


def run_beam_search_with_cutoff(
    model,
    tokenizer,
    prompts: List[str],
    prefix_allowed_tokens_fn,
    num_beams: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[List[List[str]], List[List[float]], List[Dict[str, float]]]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    with torch.no_grad():
        output = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            output_scores=True,
            return_dict_in_generate=True,
            early_stopping=True,
            temperature=temperature,
            top_p=top_p,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

    decoded = tokenizer.batch_decode(
        output["sequences"],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    raw_scores = output.get("sequences_scores", None)
    if raw_scores is None:
        score_values = [0.0] * len(decoded)
    else:
        score_values = [float(item) for item in raw_scores.detach().cpu().tolist()]

    grouped_texts: List[List[str]] = []
    grouped_scores: List[List[float]] = []
    for idx in range(len(prompts)):
        start = idx * num_beams
        end = start + num_beams
        grouped_texts.append(decoded[start:end])
        grouped_scores.append(score_values[start:end])

    step_labels = [
        "sid_begin",
        "a",
        "b",
        "c",
        "d",
        "sid_end",
    ]
    grouped_cutoffs: List[Dict[str, float]] = []
    for sample_idx in range(len(prompts)):
        sample_cutoffs: Dict[str, float] = {}
        row_start = sample_idx * num_beams
        row_end = row_start + num_beams
        for step_idx, step_scores in enumerate(output.scores):
            label = step_labels[step_idx] if step_idx < len(step_labels) else f"step_{step_idx}"
            sample_rows = step_scores[row_start:row_end]
            flat_scores = sample_rows.reshape(-1)
            top_values, _ = torch.topk(flat_scores, k=min(num_beams, flat_scores.numel()))
            threshold_log_prob = float(top_values[-1].detach().cpu())
            sample_cutoffs[f"{label}_cutoff_log_prob"] = threshold_log_prob
            sample_cutoffs[f"{label}_cutoff_prob"] = safe_exp(threshold_log_prob)
        grouped_cutoffs.append(sample_cutoffs)

    return grouped_texts, grouped_scores, grouped_cutoffs


def render_probability_histogram(records: List[Dict[str, Any]], output_path: str) -> None:
    values = np.asarray(
        [max(record["label_probs"]["four_level_total_prob"], 1e-40) for record in records],
        dtype=np.float64,
    )
    plt.figure(figsize=(8.5, 5.2))
    plt.hist(np.log10(values), bins=50, color="#2c7fb8", alpha=0.85, edgecolor="white")
    plt.xlabel("log10(label four-level total probability)")
    plt.ylabel("Sample count")
    plt.title("Distribution of label four-level total probability")
    plt.grid(alpha=0.18, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def render_level_probability_panels(records: List[Dict[str, Any]], output_path: str) -> None:
    conditional_data = [
        [max(record["label_probs"]["level_conditional_probs"][level], 1e-40) for record in records]
        for level in LEVELS
    ]
    cumulative_data = [
        [max(record["label_probs"]["level_cumulative_probs"][level], 1e-40) for record in records]
        for level in LEVELS
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    axes[0].boxplot([np.log10(values) for values in conditional_data], labels=[f"s_{level}" for level in LEVELS], showfliers=False)
    axes[0].set_title("Label conditional probability by level")
    axes[0].set_ylabel("log10(probability)")
    axes[0].grid(alpha=0.18, linestyle="--")

    axes[1].boxplot([np.log10(values) for values in cumulative_data], labels=[f"s_{level}" for level in LEVELS], showfliers=False)
    axes[1].set_title("Label cumulative probability by level")
    axes[1].set_ylabel("log10(probability)")
    axes[1].grid(alpha=0.18, linestyle="--")

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_label_vs_cutoff(records: List[Dict[str, Any]], output_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), constrained_layout=True)
    for ax, level in zip(axes.reshape(-1), LEVELS):
        x_vals = np.asarray(
            [max(record["beam_cutoff"][f"{level}_cutoff_prob"], 1e-40) for record in records],
            dtype=np.float64,
        )
        y_vals = np.asarray(
            [max(record["label_with_sid_begin_cumulative_probs"][level], 1e-40) for record in records],
            dtype=np.float64,
        )
        ax.scatter(x_vals, y_vals, s=14, alpha=0.35, color="#238b45")
        low = min(x_vals.min(), y_vals.min())
        high = max(x_vals.max(), y_vals.max())
        ax.plot([low, high], [low, high], linestyle="--", color="#cb181d", linewidth=1.1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"s_{level} beam cutoff probability")
        ax.set_ylabel(f"s_{level} label cumulative probability")
        ax.set_title(f"s_{level}: label vs beam cutoff")
        ax.grid(alpha=0.18, linestyle="--")

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_probs = np.asarray(
        [record["label_probs"]["four_level_total_prob"] for record in records],
        dtype=np.float64,
    )
    summary: Dict[str, Any] = {
        "sample_count": int(len(records)),
        "four_level_total_prob": {
            "mean": float(np.mean(total_probs)),
            "median": float(np.median(total_probs)),
            "min": float(np.min(total_probs)),
            "max": float(np.max(total_probs)),
            "p01": float(np.quantile(total_probs, 0.01)),
            "p05": float(np.quantile(total_probs, 0.05)),
            "p95": float(np.quantile(total_probs, 0.95)),
            "p99": float(np.quantile(total_probs, 0.99)),
        },
        "per_level": {},
    }

    for level in LEVELS:
        cond = np.asarray(
            [record["label_probs"]["level_conditional_probs"][level] for record in records],
            dtype=np.float64,
        )
        cumulative = np.asarray(
            [record["label_probs"]["level_cumulative_probs"][level] for record in records],
            dtype=np.float64,
        )
        cutoff = np.asarray(
            [record["beam_cutoff"][f"{level}_cutoff_prob"] for record in records],
            dtype=np.float64,
        )
        survives = np.asarray(
            [1.0 if record["label_with_sid_begin_cumulative_probs"][level] >= record["beam_cutoff"][f"{level}_cutoff_prob"] else 0.0 for record in records],
            dtype=np.float64,
        )
        min_cutoff_idx = int(np.argmin(cutoff))
        summary["per_level"][level] = {
            "label_conditional_mean": float(np.mean(cond)),
            "label_conditional_median": float(np.median(cond)),
            "label_cumulative_mean": float(np.mean(cumulative)),
            "label_cumulative_median": float(np.median(cumulative)),
            "beam_cutoff_mean": float(np.mean(cutoff)),
            "beam_cutoff_median": float(np.median(cutoff)),
            "beam_cutoff_min": float(np.min(cutoff)),
            "beam_cutoff_min_sample_id": int(records[min_cutoff_idx]["sample_id"]),
            "survival_rate_vs_cutoff": float(np.mean(survives)),
        }
    return summary


def main() -> None:
    args = parse_args()
    logger = setup_logger()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading dataset...")
    df = load_dataset(args.test_parquet_file, args.sample_num, args.sample_offset)
    logger.info("Loaded %d samples for analysis.", len(df))

    logger.info("Loading model and tokenizer...")
    model, tokenizer, device = load_model_and_tokenizer(args.model_path, args.device, args.dtype)
    logger.info("Using device: %s", device)

    logger.info("Loading exact trie...")
    trie_data = load_exact_trie(args.global_trie_file)
    prefix_allowed_tokens_fn = build_prefix_allowed_tokens_fn(tokenizer, trie_data)

    normalized_samples = [normalize_sample_row(df.iloc[idx], idx + args.sample_offset) for idx in range(len(df))]
    records: List[Dict[str, Any]] = []

    logger.info("Running benchmark-style beam analysis...")
    for start in range(0, len(normalized_samples), args.batch_size):
        batch = normalized_samples[start : start + args.batch_size]
        prompts = [format_chat_prompt(item["input_text"]) for item in batch]
        target_sids = [extract_sid_from_text(item["output_text"]) for item in batch]

        grouped_texts, grouped_scores, grouped_cutoffs = run_beam_search_with_cutoff(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        label_probs = trace_label_probabilities(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            target_sids=target_sids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        for sample, decoded_list, score_list, cutoff_info, label_info in zip(
            batch, grouped_texts, grouped_scores, grouped_cutoffs, label_probs
        ):
            label_with_sid_begin_cumulative_probs = {
                level: float(label_info["full_step_cumulative_probs"][idx + 1])
                for idx, level in enumerate(LEVELS)
            }
            records.append(
                {
                    "sample_id": int(sample["sample_id"]),
                    "user_id": sample["user_id"],
                    "target_sid": label_info["target_sid"],
                    "target_levels": parse_sid_levels(label_info["target_sid"]),
                    "label_probs": label_info,
                    "label_with_sid_begin_cumulative_probs": label_with_sid_begin_cumulative_probs,
                    "beam_cutoff": cutoff_info,
                    "beam_candidates": [
                        {
                            "rank": rank,
                            "score": float(score),
                            "sid": extract_sid_from_text(decoded_text.split("</think>")[-1]),
                        }
                        for rank, (decoded_text, score) in enumerate(zip(decoded_list, score_list), start=1)
                    ],
                }
            )

        if (start // args.batch_size + 1) % 20 == 0 or (start + len(batch)) == len(normalized_samples):
            logger.info("Processed %d / %d samples", start + len(batch), len(normalized_samples))

    records_path = os.path.join(args.output_dir, "label_beam_probability_records.jsonl")
    with open(records_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Rendering figures...")
    hist_path = os.path.join(args.output_dir, "label_four_level_total_probability_hist.png")
    render_probability_histogram(records, hist_path)

    level_panel_path = os.path.join(args.output_dir, "label_level_probability_panels.png")
    render_level_probability_panels(records, level_panel_path)

    cutoff_scatter_path = os.path.join(args.output_dir, "label_vs_beam_cutoff.png")
    render_label_vs_cutoff(records, cutoff_scatter_path)

    summary = {
        "config": vars(args),
        "summary": summarize_records(records),
        "artifacts": {
            "records": records_path,
            "label_total_histogram": hist_path,
            "level_probability_panels": level_panel_path,
            "label_vs_cutoff": cutoff_scatter_path,
        },
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Analysis complete.")
    logger.info("Summary saved to: %s", summary_path)


if __name__ == "__main__":
    main()
