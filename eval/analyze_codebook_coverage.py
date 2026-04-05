#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run benchmark-style constrained generation on all samples, then visualize
beam-output SID codebook usage and user-history SID codebook usage in the
model output-embedding space.

Outputs:
1. Per-level 2D scatter figures for beam frequency, history frequency, and
   beam-history difference.
2. Coverage summary JSON with beam/history/overlap statistics.
3. Raw frequency CSV for downstream analysis.
"""

import argparse
import csv
import json
import logging
import os
import pickle
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import umap
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install `umap-learn` to run this script.") from exc


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze codebook coverage for benchmark beams.")
    parser.add_argument("--model_path", type=str, required=True, help="Merged model path.")
    parser.add_argument("--test_parquet_file", type=str, required=True, help="Benchmark parquet path.")
    parser.add_argument("--global_trie_file", type=str, required=True, help="Exact trie pickle path.")
    parser.add_argument("--output_dir", type=str, default="./codebook_coverage_outputs", help="Output directory.")
    parser.add_argument("--num_beams", type=int, default=10, help="Beam size, should match benchmark.")
    parser.add_argument("--batch_size", type=int, default=4, help="Generation batch size.")
    parser.add_argument("--max_new_tokens", type=int, default=6, help="Max new tokens for one SID.")
    parser.add_argument("--sample_num", type=int, default=-1, help="-1 means all samples.")
    parser.add_argument("--sample_offset", type=int, default=0, help="Optional offset into parquet rows.")
    parser.add_argument("--device", type=str, default=None, help="cuda:0 / cpu. Auto if omitted.")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--temperature", type=float, default=0.6, help="Benchmark generation temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Benchmark top_p.")
    parser.add_argument("--pca_dim", type=int, default=50, help="PCA dim before UMAP.")
    parser.add_argument("--umap_n_neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("codebook_coverage")
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


def format_chat_prompt(user_content: str) -> str:
    return f"""<|im_start|>system
{SYSTEM_MESSAGE}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
<think>

</think>
"""


def extract_all_sids_from_text(text: str) -> List[str]:
    return SID_PATTERN.findall(text or "")


def extract_sid_from_text(text: str) -> str:
    matches = extract_all_sids_from_text(text)
    if matches:
        return matches[0]
    return text.strip().replace(" ", "")


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


def normalize_sample_row(row: pd.Series, sample_id: int) -> Dict[str, Any]:
    input_text = row["input"] if "input" in row else row["description"]
    output_text = row["output"] if "output" in row else row["groundtruth"]
    return {
        "sample_id": sample_id,
        "input_text": input_text,
        "output_text": output_text,
        "user_id": row["user_id"] if "user_id" in row else f"user_{sample_id}",
    }


def load_dataset(parquet_path: str, sample_num: int, sample_offset: int) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if sample_offset > 0:
        df = df.iloc[sample_offset:].reset_index(drop=True)
    if sample_num > 0:
        df = df.iloc[:sample_num].reset_index(drop=True)
    return df


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
            if list(lst[start:start + m]) == list(sub):
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


def batched_generate_beams(
    model,
    tokenizer,
    prompts: Sequence[str],
    prefix_allowed_tokens_fn,
    num_beams: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[List[List[str]], List[List[float]]]:
    enc = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    output = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc.get("attention_mask"),
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

    scores = output.get("sequences_scores", None)
    if scores is None:
        score_values = [0.0] * len(decoded)
    elif hasattr(scores, "detach"):
        score_values = [float(v) for v in scores.detach().cpu().tolist()]
    else:
        score_values = [float(v) for v in scores]

    grouped_texts: List[List[str]] = []
    grouped_scores: List[List[float]] = []
    for idx in range(len(prompts)):
        start = idx * num_beams
        end = start + num_beams
        grouped_texts.append(decoded[start:end])
        grouped_scores.append(score_values[start:end])
    return grouped_texts, grouped_scores


def find_level_token_ids(tokenizer, level: str) -> List[int]:
    pattern = re.compile(rf"<s_{level}_\d+>")
    token_ids = []
    for idx in range(len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(idx)
        if token and pattern.fullmatch(token):
            token_ids.append(idx)
    if not token_ids:
        raise ValueError(f"No tokens found for level s_{level}.")
    return sorted(token_ids)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return vectors / norms


def build_projection(
    vectors: np.ndarray,
    pca_dim: int,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> np.ndarray:
    normalized = l2_normalize(vectors)
    use_pca_dim = min(pca_dim, normalized.shape[0], normalized.shape[1])
    if use_pca_dim >= 2 and use_pca_dim < normalized.shape[1]:
        reduced = PCA(n_components=use_pca_dim, random_state=seed).fit_transform(normalized)
    else:
        reduced = normalized

    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        transform_seed=seed,
    )
    return reducer.fit_transform(reduced)


def counter_to_array(token_ids: Sequence[int], counts: Counter) -> np.ndarray:
    return np.asarray([float(counts.get(token_id, 0)) for token_id in token_ids], dtype=np.float32)


def log1p_if_needed(values: np.ndarray) -> np.ndarray:
    return np.log1p(values) if np.max(values) > 0 else values


def render_level_figure(
    level: str,
    token_ids: Sequence[int],
    coords: np.ndarray,
    beam_counts: Counter,
    history_counts: Counter,
    tokenizer,
    output_path: str,
) -> Dict[str, Any]:
    beam_values = counter_to_array(token_ids, beam_counts)
    history_values = counter_to_array(token_ids, history_counts)
    beam_norm = beam_values / max(1.0, float(beam_values.sum()))
    history_norm = history_values / max(1.0, float(history_values.sum()))
    diff_values = beam_norm - history_norm

    overlap_mask = (beam_values > 0) & (history_values > 0)
    beam_only_mask = (beam_values > 0) & (history_values == 0)
    history_only_mask = (beam_values == 0) & (history_values > 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    titles = [
        f"s_{level} beam freq",
        f"s_{level} history freq",
        f"s_{level} beam-history delta",
    ]
    panels = [
        (beam_values, "YlOrRd"),
        (history_values, "Blues"),
        (diff_values, "coolwarm"),
    ]

    for ax, title, (panel_values, cmap) in zip(axes, titles, panels):
        size_base = 20.0
        panel_sizes = size_base + 180.0 * (log1p_if_needed(np.abs(panel_values)) / max(1.0, np.max(log1p_if_needed(np.abs(panel_values)))))
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=panel_values,
            cmap=cmap,
            s=panel_sizes,
            alpha=0.9,
            edgecolors="none",
        )
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(alpha=0.18, linestyle="--")
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

    axes[0].scatter(coords[overlap_mask, 0], coords[overlap_mask, 1], s=44, facecolors="none", edgecolors="black", linewidths=0.7, label="overlap")
    axes[0].scatter(coords[beam_only_mask, 0], coords[beam_only_mask, 1], s=44, facecolors="none", edgecolors="#a50f15", linewidths=0.7, label="beam only")
    axes[1].scatter(coords[history_only_mask, 0], coords[history_only_mask, 1], s=44, facecolors="none", edgecolors="#08519c", linewidths=0.7, label="history only")

    top_beam = sorted(((token_id, count) for token_id, count in beam_counts.items() if count > 0), key=lambda x: x[1], reverse=True)[:8]
    top_history = sorted(((token_id, count) for token_id, count in history_counts.items() if count > 0), key=lambda x: x[1], reverse=True)[:8]
    for token_id, count in top_beam[:4]:
        local_idx = token_ids.index(token_id)
        axes[0].annotate(tokenizer.convert_ids_to_tokens(token_id), (coords[local_idx, 0], coords[local_idx, 1]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    for token_id, count in top_history[:4]:
        local_idx = token_ids.index(token_id)
        axes[1].annotate(tokenizer.convert_ids_to_tokens(token_id), (coords[local_idx, 0], coords[local_idx, 1]), xytext=(4, 4), textcoords="offset points", fontsize=8)

    fig.suptitle(f"Output-embedding projection for codebook level s_{level}", fontsize=14)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    coverage = {
        "level": level,
        "token_count": int(len(token_ids)),
        "beam_used_tokens": int(np.sum(beam_values > 0)),
        "history_used_tokens": int(np.sum(history_values > 0)),
        "overlap_tokens": int(np.sum(overlap_mask)),
        "beam_coverage_ratio": float(np.mean(beam_values > 0)),
        "history_coverage_ratio": float(np.mean(history_values > 0)),
        "overlap_ratio": float(np.mean(overlap_mask)),
        "beam_total_frequency": int(beam_values.sum()),
        "history_total_frequency": int(history_values.sum()),
        "top_beam_tokens": [
            {"token_id": int(token_id), "token_text": tokenizer.convert_ids_to_tokens(token_id), "count": int(count)}
            for token_id, count in top_beam
        ],
        "top_history_tokens": [
            {"token_id": int(token_id), "token_text": tokenizer.convert_ids_to_tokens(token_id), "count": int(count)}
            for token_id, count in top_history
        ],
        "figure_path": output_path,
    }
    return coverage


def write_frequency_csv(
    output_path: str,
    per_level_token_ids: Dict[str, List[int]],
    per_level_coords: Dict[str, np.ndarray],
    per_level_beam_counts: Dict[str, Counter],
    per_level_history_counts: Dict[str, Counter],
    tokenizer,
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "level",
                "token_id",
                "token_text",
                "x",
                "y",
                "beam_count",
                "history_count",
                "beam_norm",
                "history_norm",
                "beam_minus_history",
            ],
        )
        writer.writeheader()
        for level, token_ids in per_level_token_ids.items():
            coords = per_level_coords[level]
            beam_counts = per_level_beam_counts[level]
            history_counts = per_level_history_counts[level]
            beam_total = float(sum(beam_counts.values()) or 1)
            history_total = float(sum(history_counts.values()) or 1)
            for idx, token_id in enumerate(token_ids):
                beam_count = int(beam_counts.get(token_id, 0))
                history_count = int(history_counts.get(token_id, 0))
                writer.writerow(
                    {
                        "level": level,
                        "token_id": int(token_id),
                        "token_text": tokenizer.convert_ids_to_tokens(token_id),
                        "x": float(coords[idx, 0]),
                        "y": float(coords[idx, 1]),
                        "beam_count": beam_count,
                        "history_count": history_count,
                        "beam_norm": beam_count / beam_total,
                        "history_norm": history_count / history_total,
                        "beam_minus_history": (beam_count / beam_total) - (history_count / history_total),
                    }
                )


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

    logger.info("Preparing per-level output-embedding projections...")
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("Model has no output embeddings.")
    lm_head = output_embeddings.weight.detach().float().cpu().numpy()

    levels = ["a", "b", "c", "d"]
    per_level_token_ids: Dict[str, List[int]] = {}
    per_level_coords: Dict[str, np.ndarray] = {}
    per_level_beam_counts: Dict[str, Counter] = {level: Counter() for level in levels}
    per_level_history_counts: Dict[str, Counter] = {level: Counter() for level in levels}
    for level in levels:
        token_ids = find_level_token_ids(tokenizer, level)
        coords = build_projection(
            vectors=lm_head[token_ids],
            pca_dim=args.pca_dim,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            seed=args.seed,
        )
        per_level_token_ids[level] = token_ids
        per_level_coords[level] = coords

    beam_sid_counter: Counter = Counter()
    history_sid_counter: Counter = Counter()
    records_path = os.path.join(args.output_dir, "beam_sid_records.jsonl")
    history_records_path = os.path.join(args.output_dir, "history_sid_records.jsonl")

    logger.info("Running constrained beam generation across all samples...")
    with open(records_path, "w", encoding="utf-8") as beam_f, open(history_records_path, "w", encoding="utf-8") as history_f:
        normalized_samples = [normalize_sample_row(df.iloc[idx], idx + args.sample_offset) for idx in range(len(df))]
        for start in range(0, len(normalized_samples), args.batch_size):
            batch = normalized_samples[start:start + args.batch_size]
            prompts = [format_chat_prompt(item["input_text"]) for item in batch]
            grouped_texts, grouped_scores = batched_generate_beams(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            for sample, decoded_list, score_list in zip(batch, grouped_texts, grouped_scores):
                history_sids = extract_all_sids_from_text(sample["input_text"])
                for sid_text in history_sids:
                    history_sid_counter[sid_text] += 1
                    parsed = parse_sid_levels(sid_text)
                    if parsed is None:
                        continue
                    for level in levels:
                        token_id = tokenizer.convert_tokens_to_ids(parsed[level])
                        per_level_history_counts[level][token_id] += 1
                history_f.write(json.dumps({"sample_id": sample["sample_id"], "user_id": sample["user_id"], "history_sids": history_sids}, ensure_ascii=False) + "\n")

                beam_records = []
                for rank, (decoded_text, score) in enumerate(zip(decoded_list, score_list), start=1):
                    sid_text = extract_sid_from_text(decoded_text.split("</think>")[-1])
                    beam_sid_counter[sid_text] += 1
                    parsed = parse_sid_levels(sid_text)
                    if parsed is not None:
                        for level in levels:
                            token_id = tokenizer.convert_tokens_to_ids(parsed[level])
                            per_level_beam_counts[level][token_id] += 1
                    beam_records.append({"rank": rank, "score": float(score), "sid": sid_text})
                beam_f.write(
                    json.dumps(
                        {
                            "sample_id": sample["sample_id"],
                            "user_id": sample["user_id"],
                            "beam_sids": beam_records,
                            "target_sids": extract_all_sids_from_text(sample["output_text"]),
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )

            if (start // args.batch_size + 1) % 20 == 0 or (start + len(batch)) == len(normalized_samples):
                logger.info("Processed %d / %d samples", start + len(batch), len(normalized_samples))

    logger.info("Rendering coverage figures...")
    level_summaries = {}
    for level in levels:
        figure_path = os.path.join(args.output_dir, f"codebook_s_{level}_coverage.png")
        level_summaries[level] = render_level_figure(
            level=level,
            token_ids=per_level_token_ids[level],
            coords=per_level_coords[level],
            beam_counts=per_level_beam_counts[level],
            history_counts=per_level_history_counts[level],
            tokenizer=tokenizer,
            output_path=figure_path,
        )

    csv_path = os.path.join(args.output_dir, "codebook_frequency_projection.csv")
    write_frequency_csv(
        output_path=csv_path,
        per_level_token_ids=per_level_token_ids,
        per_level_coords=per_level_coords,
        per_level_beam_counts=per_level_beam_counts,
        per_level_history_counts=per_level_history_counts,
        tokenizer=tokenizer,
    )

    summary = {
        "config": vars(args),
        "sample_count": int(len(df)),
        "unique_beam_sids": int(len(beam_sid_counter)),
        "unique_history_sids": int(len(history_sid_counter)),
        "beam_history_sid_overlap": int(len(set(beam_sid_counter) & set(history_sid_counter))),
        "level_summaries": level_summaries,
        "artifacts": {
            "beam_records": records_path,
            "history_records": history_records_path,
            "frequency_csv": csv_path,
        },
    }

    summary_path = os.path.join(args.output_dir, "coverage_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Analysis complete.")
    logger.info("Summary saved to: %s", summary_path)


if __name__ == "__main__":
    main()
