import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from accelerate.utils import gather_object
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from rq.rqkmeans_faiss import (
    analyze_codes,
    encode_with_rq,
    sinkhorn_uniform_mapping,
    train_faiss_rq,
)
from rq.text2emb.utils import clean_text


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_item_texts(item_map, features):
    item_texts = []
    for item_id, fields in item_map.items():
        parts = []
        for feature in features:
            if feature in fields:
                cleaned = clean_text(fields[feature]).strip()
                if cleaned:
                    parts.append(cleaned)
        if not parts:
            parts = ["unknown item"]
        item_texts.append((str(item_id), " ".join(parts)))
    return item_texts


def load_text_encoder(model_path):
    print("Loading embedding model:", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    return tokenizer, model


def encode_items(accelerator, item_texts, tokenizer, model, max_sent_len, batch_size, word_drop_ratio):
    all_ids, all_texts = zip(*item_texts)
    order_map = {item_id: idx for idx, item_id in enumerate(all_ids)}

    total_items = len(all_texts)
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index
    chunk_size = int(np.ceil(total_items / num_processes))
    start_idx = process_index * chunk_size
    end_idx = min(start_idx + chunk_size, total_items)

    local_ids = all_ids[start_idx:end_idx]
    local_texts = all_texts[start_idx:end_idx]

    if accelerator.is_main_process:
        print(f"Total items: {total_items}")
        print(f"Generating embeddings with {num_processes} processes...")

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    local_results = []
    pbar = tqdm(
        total=len(local_texts),
        desc=f"Proc {process_index}",
        disable=not accelerator.is_local_main_process,
    )

    with torch.no_grad():
        for offset in range(0, len(local_texts), batch_size):
            batch_ids = local_ids[offset : offset + batch_size]
            batch_texts = list(local_texts[offset : offset + batch_size])

            if word_drop_ratio > 0:
                dropped = []
                for text in batch_texts:
                    kept_words = [word for word in text.split(" ") if random.random() > word_drop_ratio]
                    new_text = " ".join(kept_words).strip()
                    dropped.append(new_text if new_text else "[EMPTY]")
                batch_texts = dropped

            batch_texts = [text.strip() if text.strip() else "[EMPTY]" for text in batch_texts]

            encoded = tokenizer(
                batch_texts,
                max_length=max_sent_len,
                truncation=True,
                return_tensors="pt",
                padding=True,
            ).to(accelerator.device)

            outputs = model(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
            )
            hidden = outputs.last_hidden_state
            mask = encoded.attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            summed = torch.sum(hidden * mask, dim=1)
            denom = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = (summed / denom).cpu().numpy()

            for item_id, emb in zip(batch_ids, pooled):
                local_results.append((item_id, emb))
            pbar.update(len(batch_texts))

    pbar.close()
    accelerator.wait_for_everyone()

    gathered = gather_object(local_results)
    if not accelerator.is_main_process:
        return None, None

    gathered.sort(key=lambda x: order_map[x[0]])
    item_ids = [item_id for item_id, _ in gathered]
    embeddings = np.stack([emb for _, emb in gathered], axis=0)
    return item_ids, embeddings


def save_embedding_artifacts(item_ids, embeddings, emb_path, item_ids_path):
    os.makedirs(os.path.dirname(emb_path), exist_ok=True)
    np.save(emb_path, embeddings)
    with open(item_ids_path, "w", encoding="utf-8") as f:
        json.dump(item_ids, f, ensure_ascii=False, indent=2)
    print(f"Saved embeddings to {emb_path}")
    print(f"Saved item id order to {item_ids_path}")


def run_rq_kmeans(embeddings, num_levels, codebook_size, uniform, sinkhorn_batch_size, sinkhorn_iters):
    rq = train_faiss_rq(
        embeddings,
        num_levels=num_levels,
        codebook_size=codebook_size,
        verbose=True,
    )
    codes = encode_with_rq(rq, embeddings, codebook_size, verbose=True)
    analyze_codes(codes, "Before balancing:")

    if uniform:
        codes = sinkhorn_uniform_mapping(
            rq,
            embeddings,
            codes,
            batch_size=sinkhorn_batch_size,
            iters=sinkhorn_iters,
            verbose=True,
        )
        analyze_codes(codes, "After balancing:")

    return codes


def to_index_tokens(code_row, sid_offset=0):
    return [f"<{chr(97 + level)}_{int(code) + sid_offset}>" for level, code in enumerate(code_row)]


def to_wrapped_sid(code_row, sid_offset=0):
    body = "".join(
        f"<s_{chr(97 + level)}_{int(code) + sid_offset}>"
        for level, code in enumerate(code_row)
    )
    return f"<|sid_begin|>{body}<|sid_end|>"


def build_index_json(item_ids, codes, sid_offset=0):
    return {
        str(item_id): to_index_tokens(code_row, sid_offset=sid_offset)
        for item_id, code_row in zip(item_ids, codes)
    }


def merge_sid_into_json(item_map, item_ids, codes, sid_offset=0):
    merged = copy.deepcopy(item_map)
    for item_id, code_row in zip(item_ids, codes):
        merged[str(item_id)]["sid"] = to_wrapped_sid(code_row, sid_offset=sid_offset)
    return merged


def strip_sid_fields(item_map):
    return {
        str(item_id): {k: v for k, v in fields.items() if k != "sid"}
        for item_id, fields in item_map.items()
    }


def resolve_stage1_paths(input_json, output_dir, plm_name):
    input_path = Path(input_json)
    stem = input_path.stem
    out_dir = Path(output_dir) if output_dir else input_path.parent / "pretrain_json_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / f"{stem}.emb-{plm_name}-td.npy"
    item_ids_path = out_dir / f"{stem}.item_ids.json"
    item_meta_path = out_dir / f"{stem}.item.json"
    return out_dir, emb_path, item_ids_path, item_meta_path


def resolve_stage2_paths(input_json, output_dir, output_json):
    input_path = Path(input_json)
    stem = input_path.stem
    out_dir = Path(output_dir) if output_dir else input_path.parent / "pretrain_json_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_json = Path(output_json) if output_json else out_dir / f"{stem}.with_sid.json"
    index_path = out_dir / f"{stem}.index.json"
    item_meta_path = out_dir / f"{stem}.item.json"
    return out_dir, final_json, index_path, item_meta_path
