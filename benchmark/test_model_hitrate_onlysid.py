#!/usr/bin/env python3

import argparse
import logging
import os
import pickle
import random
import re
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SID_PATTERN = r'<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>'
SID_SUFFIX_PATTERN = r'<s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>'


def parse_args():
    parser = argparse.ArgumentParser(description="Only-SID Model Hit Rate Test with Beam Search")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merged_model_path", type=str, required=True, help="Merged model path")
    parser.add_argument("--test_parquet_file", type=str, required=True, help="Only-sid test parquet path")
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--sample_num", type=int, default=-1)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10")
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--print_generations", action="store_true", default=False)
    parser.add_argument("--two_round_beam_search", action="store_true", default=False)
    parser.add_argument("--log_file", type=str, default="./logs/onlysid_test.log")
    parser.add_argument("--global_trie_file", type=str, required=True, help="Pre-computed global trie file")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("onlysid_test")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def extract_sid_from_text(text):
    match = re.search(SID_PATTERN, text or "")
    if match:
        return match.group(0)
    return (text or "").strip().replace(" ", "")


def normalize_predicted_sid(text):
    text = (text or "").strip().replace(" ", "")
    full_match = re.search(SID_PATTERN, text)
    if full_match:
        return full_match.group(0)

    suffix_match = re.search(SID_SUFFIX_PATTERN, text)
    if suffix_match:
        return "<|sid_begin|>" + suffix_match.group(0)

    if text.startswith("<s_a_"):
        return "<|sid_begin|>" + text

    return text


def load_model(model_path, logger=None):
    logger.info(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()}"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)

    logger.info(f"Model loaded successfully, tokenizer vocab size: {tokenizer.vocab_size}")
    return model, tokenizer


class OnlySidParquetDataset(Dataset):
    def __init__(self, parquet_file, sample_num=-1, sample_offset=0, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"Loading test data from parquet file: {parquet_file}")
        self.df = pd.read_parquet(parquet_file)
        self.logger.info(f"Loaded {len(self.df)} samples from parquet")

        if sample_offset > 0:
            self.df = self.df.iloc[sample_offset:].reset_index(drop=True)
            self.logger.info(f"Applied offset {sample_offset}, remaining samples: {len(self.df)}")

        if sample_num > 0 and sample_num < len(self.df):
            self.df = self.df.iloc[:sample_num].reset_index(drop=True)
            self.logger.info(f"Limited to {sample_num} samples for this GPU")

        required_cols = ["input", "output"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' not found. Available: {list(self.df.columns)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "input_text": row["input"],
            "target_text": row["output"],
            "user_id": row.get("user_id", f"user_{idx}"),
        }


class TestCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

    def __call__(self, batch):
        return {
            "inputs": [d["input_text"] + "<|sid_begin|>" for d in batch],
            "raw_inputs": [d["input_text"] for d in batch],
            "targets": [d["target_text"] for d in batch],
            "user_ids": [d["user_id"] for d in batch],
        }


def load_global_trie(global_trie_file):
    if not os.path.exists(global_trie_file):
        raise FileNotFoundError(f"Global trie file not found: {global_trie_file}")
    with open(global_trie_file, "rb") as f:
        trie_data = pickle.load(f)
    if trie_data.get("trie_type") != "exact":
        raise ValueError(f"Expected trie_type='exact', got {trie_data.get('trie_type')}")
    return trie_data


def build_prefix_allowed_tokens_fn(trie_data, tokenizer, current_prompt_lengths):
    allowed_tokens = trie_data["exact_trie"]
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    def prefix_allowed_tokens_fn(batch_id, sentence):
        sentence = sentence.tolist()
        prompt_len = current_prompt_lengths[batch_id % len(current_prompt_lengths)]
        generated = sentence[prompt_len:]

        if len(generated) == 0:
            if 0 in allowed_tokens:
                return list(allowed_tokens[0].keys())
            return [eos_id]

        sid_pos = len(generated) - 1
        prev_token = generated[-1]
        if sid_pos in allowed_tokens and prev_token in allowed_tokens[sid_pos]:
            return allowed_tokens[sid_pos][prev_token]

        return [eos_id]

    return prefix_allowed_tokens_fn


def get_topk_results(predictions, scores, targets, k):
    results = []
    normalized_predictions = [normalize_predicted_sid(pred) for pred in predictions]

    for b in range(len(targets)):
        batch_preds = normalized_predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]
        pairs = sorted(zip(batch_preds, batch_scores), key=lambda x: x[1], reverse=True)
        target_sid = extract_sid_from_text(targets[b])
        results.append([1 if pred == target_sid else 0 for pred, _ in pairs])

    return results


def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        if len(row) >= k and max(row[:k]) == 1:
            hit += 1
    return hit / len(topk_results)


def ndcg_k(topk_results, k):
    ndcg = 0.0
    for row in topk_results:
        dcg = 0.0
        for i in range(min(k, len(row))):
            if row[i] == 1:
                dcg += 1.0 / np.log2(i + 2)
        ndcg += dcg
    return ndcg / len(topk_results)


def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            res[m] = hit_k(topk_results, int(m.split("@")[1]))
        elif m.lower().startswith("ndcg"):
            res[m] = ndcg_k(topk_results, int(m.split("@")[1]))
        else:
            raise NotImplementedError(f"Metric {m} not implemented")
    return res


def run_evaluation(args):
    set_seed(args.seed)
    logger = setup_logging(args.log_file)
    logger.info(f"Starting only-sid beam-search evaluation [GPU {args.gpu_id}]")
    logger.info(f"Args: {vars(args)}")

    model, tokenizer = load_model(args.merged_model_path, logger)
    model.eval()

    trie_data = load_global_trie(args.global_trie_file)
    logger.info(f"Loaded trie with {len(trie_data['valid_sids'])} unique SIDs")

    dataset = OnlySidParquetDataset(args.test_parquet_file, args.sample_num, args.sample_offset, logger)
    loader = DataLoader(
        dataset,
        batch_size=args.test_batch_size,
        collate_fn=TestCollator(tokenizer),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    metrics = args.metrics.split(",")
    all_topk_results = []
    total = 0
    start_time = time.time()

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Testing")
        for step, batch in enumerate(progress_bar):
            inputs_texts = batch["inputs"]
            raw_inputs = batch["raw_inputs"]
            targets = batch["targets"]
            bs = len(targets)

            enc = tokenizer(
                inputs_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
            prompt_lengths = enc["attention_mask"].sum(dim=1).tolist()
            enc = {k: v.to(model.device) for k, v in enc.items()}

            prefix_allowed_tokens_fn = build_prefix_allowed_tokens_fn(
                trie_data,
                tokenizer,
                prompt_lengths,
            )

            num_beams = args.num_beams
            while True:
                try:
                    generate_kwargs = {
                        "input_ids": enc["input_ids"],
                        "attention_mask": enc.get("attention_mask", None),
                        "max_new_tokens": args.max_new_tokens,
                        "num_beams": num_beams,
                        "num_return_sequences": num_beams,
                        "output_scores": True,
                        "return_dict_in_generate": True,
                        "early_stopping": True,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
                    }

                    if args.two_round_beam_search:
                        output = model.generate_two_pass_beam_search(
                            **generate_kwargs,
                            second_pass_separator_token_ids=[],
                        )
                    else:
                        output = model.generate(**generate_kwargs)
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                        logger.warning(f"CUDA OOM with beam={num_beams}. Reducing beam size.")
                        num_beams -= 1
                        if num_beams < 1:
                            raise RuntimeError("Beam search OOM even with beam=1") from e
                        torch.cuda.empty_cache()
                    else:
                        raise

            output_ids = output["sequences"]
            sequences_scores = output.get("sequences_scores", None)
            generated_only_ids = output_ids[:, enc["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(
                generated_only_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            if sequences_scores is not None:
                scores_list = [float(s) for s in sequences_scores.detach().cpu().tolist()]
            else:
                scores_list = [0.0] * len(decoded)

            if args.print_generations:
                for i in range(bs):
                    start = i * num_beams
                    end = start + num_beams
                    logger.info(f"----- SAMPLE {step * bs + i} -----")
                    logger.info(f"INPUT: {raw_inputs[i]}")
                    logger.info(f"INPUT_WITH_SID_BEGIN: {inputs_texts[i]}")
                    logger.info("CANDIDATES:")
                    for rank, (cand, score) in enumerate(zip(decoded[start:end], scores_list[start:end]), start=1):
                        logger.info(f"  Rank {rank}: score={score:.4f} -> {normalize_predicted_sid(cand)}")
                    logger.info(f"TARGET: {targets[i]}")
                    logger.info("-" * 50)

            all_topk_results.extend(get_topk_results(decoded, scores_list, targets, num_beams))
            total += bs

            if (step + 1) % 50 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Progress step {step + 1}/{len(loader)}, processed samples: {total}, elapsed: {elapsed:.1f}s")
                temp_metrics = get_metrics_results(all_topk_results, metrics)
                for metric, value in temp_metrics.items():
                    logger.info(f"{metric}: {value:.4f}")

    final_metrics = get_metrics_results(all_topk_results, metrics)
    logger.info("=" * 60)
    logger.info("Final Hit Rate Results:")
    logger.info("=" * 60)
    for metric, value in final_metrics.items():
        logger.info(f"{metric:>10}: {value:.4f}")
    logger.info("=" * 60)
    logger.info(f"Parquet file: {args.test_parquet_file}")
    logger.info(f"Total samples: {total}")
    logger.info(f"Batch size: {args.test_batch_size}")
    logger.info(f"Beam size: {args.num_beams}")
    logger.info("Evaluation completed successfully!")

    return final_metrics


def main():
    args = parse_args()
    try:
        run_evaluation(args)
        return True
    except Exception:
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
