import json
import os
import random

import numpy as np
import torch
from accelerate.utils import gather_object
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

try:
    from rq.text2emb.utils import clean_text
except ModuleNotFoundError:
    from utils import clean_text


def load_item_feature_map(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_item_text_list(item2feature, features=("title", "description"), default_text="unknown item"):
    item_text_list = []
    for item_id, data in item2feature.items():
        text = []
        for meta_key in features:
            if meta_key in data:
                meta_value = clean_text(data[meta_key])
                cleaned = meta_value.strip()
                if cleaned:
                    text.append(cleaned)

        if not text:
            text = [default_text]

        item_text_list.append((str(item_id), " ".join(text)))

    return item_text_list


def save_item_id_order(item_text_list, output_path):
    if item_text_list and isinstance(item_text_list[0], (list, tuple)):
        item_ids = [item_id for item_id, _ in item_text_list]
    else:
        item_ids = list(item_text_list)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(item_ids, f, ensure_ascii=False, indent=2)


def load_qwen_model(model_path):
    print("Loading Qwen Model:", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    return tokenizer, model


def generate_item_embeddings(
    item_text_list,
    tokenizer,
    model,
    accelerator,
    max_sent_len,
    word_drop_ratio=-1,
    batch_size=1024,
):
    all_ids, all_texts = zip(*item_text_list)

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
        print(f"Start generating embeddings with {num_processes} processes...")

    local_results = []
    pbar = tqdm(
        total=len(local_texts),
        desc=f"Proc {process_index}",
        disable=not accelerator.is_local_main_process,
    )

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        for i in range(0, len(local_texts), batch_size):
            batch_texts = list(local_texts[i : i + batch_size])
            batch_ids = local_ids[i : i + batch_size]

            if word_drop_ratio > 0:
                processed_batch = []
                for text in batch_texts:
                    sent = text.split(" ")
                    new_sent = [wd for wd in sent if random.random() > word_drop_ratio]
                    processed_text = " ".join(new_sent).strip()
                    processed_batch.append(processed_text if processed_text else "[EMPTY]")
                batch_texts = processed_batch

            batch_texts = [s.strip() if s.strip() else "[EMPTY]" for s in batch_texts]

            encoded_sentences = tokenizer(
                batch_texts,
                max_length=max_sent_len,
                truncation=True,
                return_tensors="pt",
                padding=True,
            ).to(accelerator.device)

            outputs = model(
                input_ids=encoded_sentences.input_ids,
                attention_mask=encoded_sentences.attention_mask,
            )

            last_hidden = outputs.last_hidden_state
            mask_expanded = (
                encoded_sentences.attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_output = (sum_embeddings / sum_mask).cpu().numpy()

            for item_id, emb in zip(batch_ids, mean_output):
                local_results.append((item_id, emb))

            pbar.update(len(batch_texts))

    pbar.close()
    accelerator.wait_for_everyone()

    gathered = gather_object(local_results)
    if not accelerator.is_main_process:
        return None, None

    order_map = {item_id: idx for idx, item_id in enumerate(all_ids)}
    gathered.sort(key=lambda x: order_map[x[0]])
    item_ids = [item_id for item_id, _ in gathered]
    final_embeddings = np.stack([emb for _, emb in gathered], axis=0)
    return item_ids, final_embeddings


def save_embeddings(embeddings, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Saved embeddings to {output_path}")
