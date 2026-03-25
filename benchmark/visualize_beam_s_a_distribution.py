#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize finite-beam candidate concentration on the first SID codebook (s_a).

For one benchmark sample:
1. Build the same chat prompt used in benchmark evaluation.
2. Generate the first product SID with exact-trie-constrained beam search.
3. Use the top-1 beam as the first product, append it to the assistant output,
   and generate the second product SID in the same way.
4. For each step, visualize in output-embedding space:
   - all s_a embeddings as background,
   - top-k s_a candidates (k = num_beams),
   - final beam-retained s_a candidates,
   - the target next-behavior s_a when it exists.
5. Use PCA(50) + UMAP(2, cosine) on L2-normalized output embeddings.
"""

import argparse
import json
import logging
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import umap
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "The `umap-learn` package is required. Please install it with `pip install umap-learn`."
    ) from exc


SYSTEM_MESSAGE = (
    "You are a professional recommendation expert who needs to recommend the next "
    "possible purchase for users based on their purchase history. Please predict "
    "the most likely next product that the user will purchase based on the user's "
    "historical purchase information."
)

SID_PATTERN = r"<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>"
SA_PATTERN = r"<s_a_(\d+)>"


@dataclass
class BeamState:
    tokens: List[int]
    score: float
    finished: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize beam-search concentration with UMAP.")
    parser.add_argument("--model_path", type=str, required=True, help="Merged model path.")
    parser.add_argument("--test_parquet_file", type=str, required=True, help="Benchmark parquet path.")
    parser.add_argument("--global_trie_file", type=str, default=None, help="Optional exact trie pickle.")
    parser.add_argument("--sample_index", type=int, default=0, help="Which sample to visualize.")
    parser.add_argument("--sample_count", type=int, default=1, help="How many consecutive samples to visualize.")
    parser.add_argument("--num_beams", type=int, default=20, help="Beam size and top-k size.")
    parser.add_argument("--max_items", type=int, default=2, help="How many products to generate sequentially.")
    parser.add_argument("--max_sid_steps", type=int, default=8, help="Max token steps for one SID.")
    parser.add_argument("--device", type=str, default=None, help="cuda:0 / cpu. Auto if omitted.")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output_dir", type=str, default="./beam_vis_outputs", help="Output directory.")
    parser.add_argument("--figure_name", type=str, default=None, help="Optional custom figure filename.")
    parser.add_argument("--umap_n_neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--pca_dim", type=int, default=50, help="PCA dimension before UMAP.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("beam_vis")
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
    return re.findall(SID_PATTERN, text or "")


def extract_first_s_a(sid_text: Optional[str]) -> Optional[str]:
    if sid_text is None:
        return None
    match = re.search(SA_PATTERN, sid_text)
    if not match:
        return None
    return f"<s_a_{match.group(1)}>"


def normalize_sample_row(row: pd.Series) -> Dict[str, Any]:
    input_text = row["input"] if "input" in row else row["description"]
    output_text = row["output"] if "output" in row else row["groundtruth"]
    user_id = row["user_id"] if "user_id" in row else "unknown_user"
    return {
        "input_text": input_text,
        "output_text": output_text,
        "user_id": user_id,
    }


def load_dataset(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


def load_sample_from_df(df: pd.DataFrame, sample_index: int) -> Dict[str, Any]:
    if sample_index < 0 or sample_index >= len(df):
        raise IndexError(f"sample_index={sample_index} out of range, total={len(df)}")
    return normalize_sample_row(df.iloc[sample_index])


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


def build_or_load_trie(df: pd.DataFrame, tokenizer, global_trie_file: Optional[str]) -> Dict[str, Any]:
    if global_trie_file:
        with open(global_trie_file, "rb") as f:
            trie_data = pickle.load(f)
        if trie_data.get("trie_type") != "exact":
            raise ValueError("Only exact trie is supported.")
        return trie_data

    valid_sids = set()
    source_text_col = "description" if "description" in df.columns else "input"
    target_text_col = "groundtruth" if "groundtruth" in df.columns else "output"

    for _, row in df.iterrows():
        for sid in extract_all_sids_from_text(row[source_text_col]):
            valid_sids.add(sid)
        for sid in extract_all_sids_from_text(row[target_text_col]):
            valid_sids.add(sid)

    sid_token_sequences = [tokenizer.encode(sid, add_special_tokens=False) for sid in sorted(valid_sids)]
    exact_trie: Dict[int, Dict[int, List[int]]] = {}
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    for seq in sid_token_sequences:
        for pos, current_token in enumerate(seq):
            next_token = eos_id if pos + 1 == len(seq) else seq[pos + 1]
            exact_trie.setdefault(pos, {}).setdefault(current_token, set()).add(next_token)

    final_trie = {
        pos: {token_id: sorted(list(next_tokens)) for token_id, next_tokens in token_map.items()}
        for pos, token_map in exact_trie.items()
    }

    return {
        "exact_trie": final_trie,
        "valid_sids": sorted(valid_sids),
        "valid_sid_tokens": sid_token_sequences,
        "trie_type": "exact",
    }


def get_allowed_tokens(exact_trie: Dict[int, Dict[int, List[int]]], beam_tokens: Sequence[int], eos_id: int) -> List[int]:
    if not beam_tokens:
        return sorted(list(exact_trie.get(0, {}).keys()))
    prev_pos = len(beam_tokens) - 1
    prev_token = beam_tokens[-1]
    return exact_trie.get(prev_pos, {}).get(prev_token, [eos_id])


def encode_with_padding(tokenizer, sequences: Sequence[List[int]], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in sequences)
    pad_id = tokenizer.pad_token_id
    input_ids = []
    attention_mask = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        input_ids.append([pad_id] * pad_len + seq)
        attention_mask.append([0] * pad_len + [1] * len(seq))
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
    )


def batched_next_token_logits(
    model,
    tokenizer,
    prefix_token_ids: List[int],
    beam_states: Sequence[BeamState],
    device: str,
) -> torch.Tensor:
    sequences = [prefix_token_ids + beam.tokens for beam in beam_states]
    input_ids, attention_mask = encode_with_padding(tokenizer, sequences, device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits[:, -1, :]


def constrained_sid_beam_search(
    model,
    tokenizer,
    prefix_text: str,
    trie_data: Dict[str, Any],
    num_beams: int,
    device: str,
    max_sid_steps: int,
) -> List[Dict[str, Any]]:
    exact_trie = trie_data["exact_trie"]
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    prefix_token_ids = tokenizer(prefix_text, add_special_tokens=True)["input_ids"]

    beams = [BeamState(tokens=[], score=0.0, finished=False)]

    for _ in range(max_sid_steps):
        unfinished_beams = [beam for beam in beams if not beam.finished]
        if not unfinished_beams:
            break

        logits = batched_next_token_logits(model, tokenizer, prefix_token_ids, unfinished_beams, device)
        log_probs = F.log_softmax(logits, dim=-1)

        expanded: List[BeamState] = [beam for beam in beams if beam.finished]
        unfinished_idx = 0

        for beam in beams:
            if beam.finished:
                continue

            allowed_tokens = get_allowed_tokens(exact_trie, beam.tokens, eos_id)
            beam_log_probs = log_probs[unfinished_idx]
            unfinished_idx += 1

            allowed_tensor = torch.tensor(allowed_tokens, dtype=torch.long, device=beam_log_probs.device)
            selected_scores = beam_log_probs.index_select(0, allowed_tensor)

            for token_id, token_score in zip(allowed_tokens, selected_scores.tolist()):
                if token_id == eos_id:
                    expanded.append(
                        BeamState(tokens=list(beam.tokens), score=beam.score + float(token_score), finished=True)
                    )
                else:
                    expanded.append(
                        BeamState(
                            tokens=list(beam.tokens) + [int(token_id)],
                            score=beam.score + float(token_score),
                            finished=False,
                        )
                    )

        expanded.sort(key=lambda x: x.score, reverse=True)
        beams = expanded[:num_beams]

    final_results = []
    for rank, beam in enumerate(sorted(beams, key=lambda x: x.score, reverse=True), start=1):
        sid_text = tokenizer.decode(beam.tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        final_results.append(
            {
                "rank": rank,
                "score": float(beam.score),
                "token_ids": list(beam.tokens),
                "sid_text": sid_text,
                "s_a_text": extract_first_s_a(sid_text),
                "finished": beam.finished,
            }
        )
    return final_results


def find_s_a_token_ids(tokenizer) -> List[int]:
    token_ids = []
    for idx in range(len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(idx)
        if token and re.fullmatch(r"<s_a_\d+>", token):
            token_ids.append(idx)
    if not token_ids:
        raise ValueError("No s_a tokens found in tokenizer.")
    return sorted(token_ids)


def compute_s_a_step_scores(
    model,
    tokenizer,
    prefix_text: str,
    trie_data: Dict[str, Any],
    all_s_a_token_ids: Sequence[int],
    device: str,
    topk: int,
) -> Dict[str, Any]:
    exact_trie = trie_data["exact_trie"]
    sid_begin_candidates = sorted(list(exact_trie.get(0, {}).keys()))
    if not sid_begin_candidates:
        raise ValueError("No SID begin candidates found in trie.")

    prefix_token_ids = tokenizer(prefix_text, add_special_tokens=True)["input_ids"]
    sequences = [prefix_token_ids + [sid_begin_id] for sid_begin_id in sid_begin_candidates]
    input_ids, attention_mask = encode_with_padding(tokenizer, sequences, device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]

    sid_begin_scores = []
    for row_idx, sid_begin_id in enumerate(sid_begin_candidates):
        allowed_s_a = exact_trie[0][sid_begin_id]
        allowed_tensor = torch.tensor(allowed_s_a, dtype=torch.long, device=logits.device)
        allowed_scores = logits[row_idx].index_select(0, allowed_tensor)
        best_score, best_local_idx = torch.max(allowed_scores, dim=0)
        sid_begin_scores.append((float(best_score), row_idx, sid_begin_id, allowed_s_a, int(best_local_idx.item())))

    sid_begin_scores.sort(key=lambda x: x[0], reverse=True)
    _, best_row_idx, sid_begin_id, allowed_s_a, _ = sid_begin_scores[0]

    s_a_tensor = torch.tensor(list(all_s_a_token_ids), dtype=torch.long, device=logits.device)
    all_s_a_logits = logits[best_row_idx].index_select(0, s_a_tensor).detach().float().cpu().numpy()

    allowed_s_a_tensor = torch.tensor(allowed_s_a, dtype=torch.long, device=logits.device)
    allowed_s_a_scores = logits[best_row_idx].index_select(0, allowed_s_a_tensor)
    order = torch.argsort(allowed_s_a_scores, descending=True)[:topk]

    topk_records = []
    for rank, idx in enumerate(order.tolist(), start=1):
        token_id = allowed_s_a[idx]
        topk_records.append(
            {
                "rank": rank,
                "token_id": int(token_id),
                "token_text": tokenizer.convert_ids_to_tokens(int(token_id)),
                "score": float(allowed_s_a_scores[idx].item()),
            }
        )

    return {
        "sid_begin_id": int(sid_begin_id),
        "sid_begin_text": tokenizer.convert_ids_to_tokens(int(sid_begin_id)),
        "all_s_a_logits": all_s_a_logits,
        "topk_s_a": topk_records,
    }


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return vectors / norms


def build_umap_projection(
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


def aggregate_points(token_ids: Sequence[int], records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[int, Dict[str, Any]] = {}
    for token_id, record in zip(token_ids, records):
        grouped.setdefault(
            token_id,
            {
                "token_id": token_id,
                "count": 0,
                "best_score": float("-inf"),
                "labels": set(),
                "ranks": [],
            },
        )
        grouped[token_id]["count"] += 1
        grouped[token_id]["best_score"] = max(grouped[token_id]["best_score"], record["score"])
        label = record.get("s_a_text") or record.get("token_text")
        if label:
            grouped[token_id]["labels"].add(label)
        grouped[token_id]["ranks"].append(record["rank"])

    result = list(grouped.values())
    for item in result:
        item["labels"] = sorted(item["labels"])
        item["ranks"] = sorted(item["ranks"])
    return result


def deduplicate_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    ax.legend(unique.values(), unique.keys(), loc="best", fontsize=9)


def plot_step_coverage(
    ax,
    id_to_xy: Dict[int, np.ndarray],
    all_s_a_token_ids: Sequence[int],
    step_result: Dict[str, Any],
    tokenizer,
    color: str,
) -> None:
    background_xy = np.array([id_to_xy[token_id] for token_id in all_s_a_token_ids])
    ax.scatter(
        background_xy[:, 0],
        background_xy[:, 1],
        s=14,
        c="#d9d9d9",
        alpha=0.5,
        label="All s_a",
    )

    topk_records = step_result["topk_s_a"]
    topk_ids = [item["token_id"] for item in topk_records if item["token_id"] in id_to_xy]
    if topk_ids:
        topk_agg = aggregate_points(topk_ids, topk_records)
        first = True
        for item in topk_agg:
            x, y = id_to_xy[item["token_id"]]
            ax.scatter(
                x,
                y,
                s=160 + 16 * item["count"],
                facecolors="none",
                edgecolors=color,
                linewidths=1.2,
                alpha=0.95,
                label="Top-k s_a" if first else None,
            )
            first = False

    beam_records = [beam for beam in step_result["beams"] if beam["s_a_text"] is not None]
    beam_ids = [tokenizer.convert_tokens_to_ids(beam["s_a_text"]) for beam in beam_records]
    beam_ids = [token_id for token_id in beam_ids if token_id in id_to_xy]
    if beam_ids:
        beam_agg = aggregate_points(beam_ids, beam_records)
        first = True
        for item in beam_agg:
            x, y = id_to_xy[item["token_id"]]
            ax.scatter(
                x,
                y,
                s=70 + 24 * item["count"],
                c=color,
                edgecolors="white",
                linewidths=0.8,
                alpha=0.85,
                label="Beam-retained s_a" if first else None,
            )
            label = f"{item['labels'][0]} x{item['count']}"
            ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8, color=color)
            first = False

    chosen_s_a = extract_first_s_a(step_result["chosen_sid"])
    if chosen_s_a:
        chosen_id = tokenizer.convert_tokens_to_ids(chosen_s_a)
        if chosen_id in id_to_xy:
            x, y = id_to_xy[chosen_id]
            ax.scatter(
                x,
                y,
                s=250,
                facecolors="none",
                edgecolors=color,
                linewidths=2.0,
                marker="s",
                label="Beam top-1",
            )

    target_s_a = step_result["target_s_a"]
    if target_s_a:
        target_id = tokenizer.convert_tokens_to_ids(target_s_a)
        if target_id in id_to_xy:
            x, y = id_to_xy[target_id]
            ax.scatter(
                x,
                y,
                s=300,
                c=color,
                marker="*",
                edgecolors="black",
                linewidths=0.8,
                label="Target s_a",
            )

    metrics = step_result["metrics"]
    ax.set_title(
        f"Step {step_result['step']} Coverage\n"
        f"top-k={metrics['topk_size']} | target_in_topk={metrics['target_in_topk']} | "
        f"target_in_beam={metrics['target_in_beam']}"
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.18, linestyle="--")
    deduplicate_legend(ax)


def plot_step_heatmap(
    ax,
    id_to_xy: Dict[int, np.ndarray],
    all_s_a_token_ids: Sequence[int],
    step_result: Dict[str, Any],
    tokenizer,
    color: str,
) -> None:
    logits = np.asarray(step_result["all_s_a_logits"], dtype=np.float32)
    coords = np.array([id_to_xy[token_id] for token_id in all_s_a_token_ids])
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=logits,
        cmap="YlOrRd",
        s=24,
        alpha=0.9,
    )

    topk_ids = [item["token_id"] for item in step_result["topk_s_a"] if item["token_id"] in id_to_xy]
    if topk_ids:
        topk_coords = np.array([id_to_xy[token_id] for token_id in topk_ids])
        ax.scatter(
            topk_coords[:, 0],
            topk_coords[:, 1],
            s=180,
            facecolors="none",
            edgecolors=color,
            linewidths=1.1,
            label="Top-k s_a",
        )

    target_s_a = step_result["target_s_a"]
    if target_s_a:
        target_id = tokenizer.convert_tokens_to_ids(target_s_a)
        if target_id in id_to_xy:
            x, y = id_to_xy[target_id]
            ax.scatter(
                x,
                y,
                s=320,
                c=color,
                marker="*",
                edgecolors="black",
                linewidths=0.8,
                label="Target s_a",
            )

    topk_preview = ", ".join(item["token_text"] for item in step_result["topk_s_a"][:5])
    ax.set_title(f"Step {step_result['step']} s_a Response Heatmap\nTop-5: {topk_preview}")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.18, linestyle="--")
    deduplicate_legend(ax)
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="logit")


def save_metadata(output_path: str, payload: Dict[str, Any]) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_single_sample(
    args: argparse.Namespace,
    logger: logging.Logger,
    model,
    tokenizer,
    device: str,
    df: pd.DataFrame,
    trie_data: Dict[str, Any],
    sample_index: int,
    id_to_xy: Dict[int, np.ndarray],
    all_s_a_token_ids: Sequence[int],
) -> Tuple[str, str]:
    sample = load_sample_from_df(df, sample_index)
    sample_prompt = format_chat_prompt(sample["input_text"])
    target_sids = extract_all_sids_from_text(sample["output_text"])

    logger.info("Running sequential constrained beam search for sample %d...", sample_index)
    step_results: List[Dict[str, Any]] = []
    current_prefix = sample_prompt

    for step_idx in range(args.max_items):
        beam_results = constrained_sid_beam_search(
            model=model,
            tokenizer=tokenizer,
            prefix_text=current_prefix,
            trie_data=trie_data,
            num_beams=args.num_beams,
            device=device,
            max_sid_steps=args.max_sid_steps,
        )

        step_score_info = compute_s_a_step_scores(
            model=model,
            tokenizer=tokenizer,
            prefix_text=current_prefix,
            trie_data=trie_data,
            all_s_a_token_ids=all_s_a_token_ids,
            device=device,
            topk=args.num_beams,
        )

        chosen_sid = beam_results[0]["sid_text"] if beam_results else ""
        target_sid = target_sids[step_idx] if step_idx < len(target_sids) else None
        target_s_a = extract_first_s_a(target_sid)

        topk_token_texts = {item["token_text"] for item in step_score_info["topk_s_a"]}
        beam_s_a_texts = {beam["s_a_text"] for beam in beam_results if beam["s_a_text"] is not None}

        target_rank_in_topk = None
        if target_s_a:
            for item in step_score_info["topk_s_a"]:
                if item["token_text"] == target_s_a:
                    target_rank_in_topk = item["rank"]
                    break

        metrics = {
            "topk_size": args.num_beams,
            "target_rank_in_topk": target_rank_in_topk,
            "target_in_topk": target_s_a in topk_token_texts if target_s_a else False,
            "target_in_beam": target_s_a in beam_s_a_texts if target_s_a else False,
            "beam_s_a_diversity": len(beam_s_a_texts),
            "beam_s_a_diversity_ratio": (len(beam_s_a_texts) / max(1, args.num_beams)),
        }

        step_results.append(
            {
                "step": step_idx + 1,
                "beams": beam_results,
                "chosen_sid": chosen_sid,
                "target_sid": target_sid,
                "target_s_a": target_s_a,
                "topk_s_a": step_score_info["topk_s_a"],
                "all_s_a_logits": step_score_info["all_s_a_logits"].tolist(),
                "sid_begin_text": step_score_info["sid_begin_text"],
                "metrics": metrics,
            }
        )

        current_prefix = current_prefix + chosen_sid + "\n"

    nrows = max(1, len(step_results))
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(15, 6 * nrows), squeeze=False)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for row_idx, step_result in enumerate(step_results):
        color = colors[row_idx % len(colors)]
        plot_step_coverage(
            ax=axes[row_idx, 0],
            id_to_xy=id_to_xy,
            all_s_a_token_ids=all_s_a_token_ids,
            step_result=step_result,
            tokenizer=tokenizer,
            color=color,
        )
        plot_step_heatmap(
            ax=axes[row_idx, 1],
            id_to_xy=id_to_xy,
            all_s_a_token_ids=all_s_a_token_ids,
            step_result=step_result,
            tokenizer=tokenizer,
            color=color,
        )

    fig.suptitle(
        f"Beam Search Candidate Concentration in s_a Output Embedding Space\n"
        f"Sample {sample_index} | top-k = beam width = {args.num_beams}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if args.figure_name and args.sample_count == 1:
        figure_name = args.figure_name
    else:
        figure_name = f"sample_{sample_index:04d}_beam_s_a_umap.png"
    figure_path = os.path.join(args.output_dir, figure_name)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "sample_index": sample_index,
        "user_id": sample["user_id"],
        "input_text": sample["input_text"],
        "output_text": sample["output_text"],
        "target_sids": target_sids,
        "num_beams": args.num_beams,
        "umap": {
            "metric": "cosine",
            "n_neighbors": args.umap_n_neighbors,
            "min_dist": args.umap_min_dist,
            "pca_dim": args.pca_dim,
            "l2_normalize": True,
        },
        "step_results": step_results,
        "figure_path": figure_path,
        "note": (
            "The 2D projection is used for qualitative visualization of cluster structure "
            "and candidate coverage, rather than exact geometric analysis."
        ),
    }
    metadata_path = os.path.join(args.output_dir, f"sample_{sample_index:04d}_beam_s_a_umap.json")
    save_metadata(metadata_path, metadata)

    logger.info("Visualization saved to: %s", figure_path)
    logger.info("Beam metadata saved to: %s", metadata_path)
    return figure_path, metadata_path


def main() -> None:
    args = parse_args()
    logger = setup_logger()
    set_seed(args.seed)

    if args.sample_count < 1:
        raise ValueError("--sample_count must be >= 1")

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_dataset(args.test_parquet_file)
    total_samples = len(df)
    start_index = args.sample_index
    end_index = min(start_index + args.sample_count, total_samples)
    if start_index < 0 or start_index >= total_samples:
        raise IndexError(f"sample_index={start_index} out of range, total={total_samples}")

    logger.info("Loading model and tokenizer...")
    model, tokenizer, device = load_model_and_tokenizer(args.model_path, args.device, args.dtype)

    logger.info("Loading/building exact trie...")
    trie_data = build_or_load_trie(df, tokenizer, args.global_trie_file)

    logger.info("Preparing L2-normalized output embedding UMAP...")
    all_s_a_token_ids = find_s_a_token_ids(tokenizer)
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("Model has no output embeddings.")

    lm_head = output_embeddings.weight.detach().float().cpu().numpy()
    s_a_vectors = lm_head[all_s_a_token_ids]
    projected = build_umap_projection(
        vectors=s_a_vectors,
        pca_dim=args.pca_dim,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        seed=args.seed,
    )
    id_to_xy = {token_id: projected[idx] for idx, token_id in enumerate(all_s_a_token_ids)}

    logger.info(
        "Processing samples from %d to %d (requested count=%d, actual count=%d)...",
        start_index,
        end_index - 1,
        args.sample_count,
        end_index - start_index,
    )
    for sample_index in range(start_index, end_index):
        run_single_sample(
            args=args,
            logger=logger,
            model=model,
            tokenizer=tokenizer,
            device=device,
            df=df,
            trie_data=trie_data,
            sample_index=sample_index,
            id_to_xy=id_to_xy,
            all_s_a_token_ids=all_s_a_token_ids,
        )


if __name__ == "__main__":
    main()
