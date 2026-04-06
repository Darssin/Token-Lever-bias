#!/usr/bin/env python3

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


CURRENT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = CURRENT_DIR.parent / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from grc_pipeline_utils import (  # noqa: E402
    build_correction_prompt,
    build_generation_prompt,
    build_reflection_constraint_ids,
    build_reflection_prompt,
    build_sid_constraint_ids,
    extract_sid_from_text,
    make_reflection_prefix_allowed_tokens_fn,
    make_sid_prefix_allowed_tokens_fn,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark GRC by beam-searching only after <|grc_correct|>.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_parquet_file", type=str, required=True)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=10)
    parser.add_argument("--sample_num", type=int, default=-1)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,hit@20,ndcg@5,ndcg@10,ndcg@20")
    parser.add_argument("--draft_max_new_tokens", type=int, default=6)
    parser.add_argument("--reflection_max_new_tokens", type=int, default=4)
    parser.add_argument("--correction_max_new_tokens", type=int, default=6)
    parser.add_argument("--log_file", type=str, default="./result/grc_eval.log")
    parser.add_argument("--print_generations", action="store_true", default=False)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(f"grc_eval_{os.getpid()}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(stream_handler)
    return logger


def load_model(model_path: str, logger: logging.Logger):
    logger.info("Loading model from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if not torch.cuda.is_available():
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        model = model.to(f"cuda:{torch.cuda.current_device()}")
    model.eval()
    return model, tokenizer


class GrcParquetDataset(Dataset):
    def __init__(self, parquet_file: str, sample_num: int = -1, sample_offset: int = 0):
        dataframe = pd.read_parquet(parquet_file)
        required_columns = {"input", "output"}
        missing = required_columns - set(dataframe.columns)
        if missing:
            raise ValueError(f"Missing columns in test parquet: {sorted(missing)}")

        if sample_offset > 0:
            dataframe = dataframe.iloc[sample_offset:].reset_index(drop=True)
        if sample_num > 0:
            dataframe = dataframe.iloc[:sample_num].reset_index(drop=True)

        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]
        return {
            "input_text": row["input"],
            "target_sid": row["output"],
        }


class GrcCollator:
    def __call__(self, batch: List[Dict[str, Any]]):
        return {
            "input_texts": [item["input_text"] for item in batch],
            "targets": [item["target_sid"] for item in batch],
        }


def hit_k(topk_results: List[List[int]], k: int) -> float:
    hit = 0.0
    for row in topk_results:
        if len(row) >= k and max(row[:k]) == 1:
            hit += 1
    return hit / len(topk_results) if topk_results else 0.0


def ndcg_k(topk_results: List[List[int]], k: int) -> float:
    ndcg = 0.0
    for row in topk_results:
        dcg = 0.0
        for i in range(min(k, len(row))):
            if row[i] == 1:
                dcg += 1.0 / np.log2(i + 2)
        ndcg += dcg
    return ndcg / len(topk_results) if topk_results else 0.0


def get_metrics_results(topk_results: List[List[int]], metrics: List[str]) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for metric in metrics:
        if metric.startswith("hit@"):
            results[metric] = hit_k(topk_results, int(metric.split("@")[1]))
        elif metric.startswith("ndcg@"):
            results[metric] = ndcg_k(topk_results, int(metric.split("@")[1]))
        else:
            raise NotImplementedError(f"Unsupported metric: {metric}")
    return results


def tokenize_batch(tokenizer, prompts: List[str], device: torch.device):
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def generate_sid_stage(
    model,
    tokenizer,
    prompts: List[str],
    sid_constraints: Dict[str, Any],
    max_new_tokens: int,
    num_beams: int,
    num_return_sequences: int,
) -> Tuple[List[str], List[float], List[str]]:
    encoded = tokenize_batch(tokenizer, prompts, model.device)
    prompt_length = encoded["input_ids"].shape[1]
    output = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        output_scores=True,
        return_dict_in_generate=True,
        early_stopping=True,
        prefix_allowed_tokens_fn=make_sid_prefix_allowed_tokens_fn(
            prompt_length=prompt_length,
            sid_constraints=sid_constraints,
        ),
    )
    generated_ids = output["sequences"][:, prompt_length:]
    decoded_generated = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    raw_scores = output.get("sequences_scores")
    if raw_scores is None:
        scores = [0.0] * len(decoded_generated)
    else:
        scores = [float(item) for item in raw_scores.detach().cpu().tolist()]
    sid_texts = [extract_sid_from_text(text) or "" for text in decoded_generated]
    return sid_texts, scores, decoded_generated


def generate_reflection_stage(
    model,
    tokenizer,
    prompts: List[str],
    reflection_constraints: Dict[str, Any],
    max_new_tokens: int,
) -> List[str]:
    encoded = tokenize_batch(tokenizer, prompts, model.device)
    prompt_length = encoded["input_ids"].shape[1]
    output = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        num_beams=1,
        num_return_sequences=1,
        output_scores=False,
        return_dict_in_generate=True,
        early_stopping=True,
        prefix_allowed_tokens_fn=make_reflection_prefix_allowed_tokens_fn(
            prompt_length=prompt_length,
            reflection_constraints=reflection_constraints,
        ),
    )
    decoded = tokenizer.batch_decode(
        output["sequences"],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    reflection_sequences = []
    for text in decoded:
        if "<|grc_reflect|>" in text:
            reflect_part = text.split("<|grc_reflect|>", 1)[1]
        else:
            reflect_part = text
        tokens = [token for token in reflect_part.split("\n")[0].split() if token]
        if not tokens:
            import re

            tokens = re.findall(r"<\|[^|]+?\|>", reflect_part)
        reflection_sequences.append("".join(tokens[:3]))
    return reflection_sequences


def build_topk_results(corrected_candidates: List[List[str]], targets: List[str]) -> List[List[int]]:
    topk_results = []
    for candidates, target in zip(corrected_candidates, targets):
        topk_results.append([1 if candidate == target else 0 for candidate in candidates])
    return topk_results


def log_generation_examples(
    logger: logging.Logger,
    input_texts: List[str],
    targets: List[str],
    draft_prompts: List[str],
    reflection_prompts: List[str],
    correction_prompts: List[str],
    draft_sids: List[str],
    draft_generated_texts: List[str],
    reflection_sequences: List[str],
    corrected_candidates: List[List[str]],
    corrected_scores: List[List[float]],
    corrected_generated_texts: List[List[str]],
    start_index: int,
):
    for sample_index, target in enumerate(targets):
        logger.info("----- SAMPLE %s -----", start_index + sample_index)
        logger.info("INPUT: %s", input_texts[sample_index])
        logger.info("DRAFT_PROMPT: %s", draft_prompts[sample_index])
        logger.info("DRAFT: %s", draft_sids[sample_index])
        logger.info("DRAFT_GENERATED_TEXT: %s", draft_generated_texts[sample_index])
        logger.info("REFLECTION_PROMPT: %s", reflection_prompts[sample_index])
        logger.info("REFLECTION: %s", reflection_sequences[sample_index])
        logger.info("CORRECTION_PROMPT: %s", correction_prompts[sample_index])
        logger.info("CORRECTED_CANDIDATES:")
        for rank, (candidate, score, generated_text) in enumerate(
            zip(corrected_candidates[sample_index], corrected_scores[sample_index], corrected_generated_texts[sample_index]),
            start=1,
        ):
            logger.info("  Rank %s: score=%.4f -> %s", rank, score, candidate)
            logger.info("    GENERATED_TEXT: %s", generated_text)
        logger.info("TARGET: %s", target)


def run_evaluation(args):
    set_seed(args.seed)
    logger = setup_logging(args.log_file)
    logger.info("Starting staged GRC benchmark on GPU %s", args.gpu_id)
    logger.info("Args: %s", vars(args))

    model, tokenizer = load_model(args.model_path, logger)
    sid_constraints = build_sid_constraint_ids(tokenizer)
    reflection_constraints = build_reflection_constraint_ids(tokenizer)

    dataset = GrcParquetDataset(args.test_parquet_file, args.sample_num, args.sample_offset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=GrcCollator(),
    )

    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    all_topk_results: List[List[int]] = []
    total_samples = 0
    draft_top1_hit = 0
    corrected_top1_hit = 0

    with torch.no_grad():
        progress = tqdm(dataloader, desc="GRC Eval")
        for step, batch in enumerate(progress):
            input_texts = batch["input_texts"]
            targets = batch["targets"]

            draft_prompts = [build_generation_prompt(text) for text in input_texts]
            draft_sids, _, draft_generated_texts = generate_sid_stage(
                model=model,
                tokenizer=tokenizer,
                prompts=draft_prompts,
                sid_constraints=sid_constraints,
                max_new_tokens=args.draft_max_new_tokens,
                num_beams=1,
                num_return_sequences=1,
            )

            reflection_prompts = [
                build_reflection_prompt(user_input=text, draft_sid=draft_sid)
                for text, draft_sid in zip(input_texts, draft_sids)
            ]
            reflection_sequences = generate_reflection_stage(
                model=model,
                tokenizer=tokenizer,
                prompts=reflection_prompts,
                reflection_constraints=reflection_constraints,
                max_new_tokens=args.reflection_max_new_tokens,
            )

            correction_prompts = [
                build_correction_prompt(user_input=text, draft_sid=draft_sid, reflection_sequence=reflection_sequence)
                for text, draft_sid, reflection_sequence in zip(input_texts, draft_sids, reflection_sequences)
            ]
            corrected_flat, corrected_scores_flat, corrected_generated_texts_flat = generate_sid_stage(
                model=model,
                tokenizer=tokenizer,
                prompts=correction_prompts,
                sid_constraints=sid_constraints,
                max_new_tokens=args.correction_max_new_tokens,
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
            )

            corrected_candidates = []
            corrected_scores = []
            corrected_generated_texts = []
            for sample_index in range(len(targets)):
                start = sample_index * args.num_beams
                end = start + args.num_beams
                corrected_candidates.append(corrected_flat[start:end])
                corrected_scores.append(corrected_scores_flat[start:end])
                corrected_generated_texts.append(corrected_generated_texts_flat[start:end])

            all_topk_results.extend(build_topk_results(corrected_candidates, targets))
            total_samples += len(targets)
            draft_top1_hit += sum(int(draft_sid == target) for draft_sid, target in zip(draft_sids, targets))
            corrected_top1_hit += sum(
                int(candidates and candidates[0] == target)
                for candidates, target in zip(corrected_candidates, targets)
            )

            if args.print_generations:
                log_generation_examples(
                    logger=logger,
                    input_texts=input_texts,
                    targets=targets,
                    draft_prompts=draft_prompts,
                    reflection_prompts=reflection_prompts,
                    correction_prompts=correction_prompts,
                    draft_sids=draft_sids,
                    draft_generated_texts=draft_generated_texts,
                    reflection_sequences=reflection_sequences,
                    corrected_candidates=corrected_candidates,
                    corrected_scores=corrected_scores,
                    corrected_generated_texts=corrected_generated_texts,
                    start_index=step * args.test_batch_size,
                )

    metric_results = get_metrics_results(all_topk_results, metrics)
    logger.info("=" * 60)
    logger.info("Final Hit Rate Results:")
    logger.info("=" * 60)
    for metric_name, value in metric_results.items():
        logger.info("%10s: %.4f", metric_name, value)
    logger.info("=" * 60)
    logger.info("Total samples: %s", total_samples)
    logger.info("Top1 draft hit: %.4f", draft_top1_hit / total_samples if total_samples else 0.0)
    logger.info("Top1 corrected hit: %.4f", corrected_top1_hit / total_samples if total_samples else 0.0)
    logger.info(
        "Top1 corrected gain over draft: %.4f",
        (corrected_top1_hit - draft_top1_hit) / total_samples if total_samples else 0.0,
    )
    logger.info("Evaluation completed successfully")
    return metric_results


def main():
    args = parse_args()
    try:
        run_evaluation(args)
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
