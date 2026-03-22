#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset
import torch

from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    HfArgumentParser, 
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


def parse_script_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_dir", default="./expanded_model")
    parser.add_argument("--train_data_path", default="./training_align_data_train.parquet")
    parser.add_argument("--val_data_path", default="./training_align_data_valid.parquet")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--tensorboard_dir", default=None)
    parser.add_argument("--freeze_llm", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--start_optimize_embedding_index", type=int, default=0)
    return parser


def prepare_dataset(data_path, sample_size=None, local_rank=0):
    if local_rank == 0:
        print(f"Loading parquet file: {data_path}")
    data_pq = pd.read_parquet(data_path)
    if local_rank == 0:
        print(f"Data shape: {data_pq.shape}")
        print(f"Columns: {list(data_pq.columns)}")

    if sample_size is not None and len(data_pq) > sample_size:
        if local_rank == 0:
            print(f"Sampling {sample_size} samples from {len(data_pq)} total samples")
        data_pq = data_pq.head(sample_size)

    texts = data_pq["description"].tolist()
    if local_rank == 0:
        print(f"Total texts: {len(texts)}")
        print("\nFirst 3 text examples:")
        for i, text in enumerate(texts[:3]):
            print(f"[{i}] {text}")

    return Dataset.from_dict({"text": texts})


def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"],
        padding="longest",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_attention_mask=True,
    )


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def freeze_for_embedding_and_lm_head(model, freeze_llm):
    trainable_names = []
    if not freeze_llm:
        for name, param in model.named_parameters():
            param.requires_grad = True
            trainable_names.append(name)
        return trainable_names

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            param.requires_grad = True
            trainable_names.append(name)
    return trainable_names


class EmbeddingRangeMasker:
    def __init__(self, model, start_index):
        self.start_index = start_index
        self.embed_tokens_weight = model.get_input_embeddings().weight
        output_embeddings = model.get_output_embeddings()
        self.lm_head_weight = output_embeddings.weight if output_embeddings is not None else None

        self.embed_tokens_frozen_prefix = self.embed_tokens_weight[:start_index].detach().clone()
        self.lm_head_frozen_prefix = (
            self.lm_head_weight[:start_index].detach().clone()
            if self.lm_head_weight is not None
            else None
        )

    def restore_frozen_prefix(self):
        with torch.no_grad():
            self.embed_tokens_weight[: self.start_index].copy_(self.embed_tokens_frozen_prefix)
            if self.lm_head_weight is not None and self.lm_head_frozen_prefix is not None:
                self.lm_head_weight[: self.start_index].copy_(self.lm_head_frozen_prefix)


def main():
    script_parser = parse_script_args()
    script_args, remaining_args = script_parser.parse_known_args()
    hf_parser = HfArgumentParser((TrainingArguments,))
    (training_args,) = hf_parser.parse_args_into_dataclasses(args=remaining_args)
    training_args.label_names = ["labels"]

    model_dir = Path(script_args.model_dir).resolve()
    train_data_path = Path(script_args.train_data_path).resolve()
    val_data_path = Path(script_args.val_data_path).resolve()

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not train_data_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_data_path}")
    if not val_data_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_data_path}")
    if script_args.freeze_llm and script_args.start_optimize_embedding_index <= 0:
        raise ValueError("freeze_llm=True requires start_optimize_embedding_index > 0.")

    if training_args.local_rank == 0:
        print(f"Using model_dir: {model_dir}")
        print(f"Training data path: {train_data_path}")
        print(f"Validation data path: {val_data_path}")
        print(
            "freeze_llm="
            f"{script_args.freeze_llm}, start_optimize_embedding_index={script_args.start_optimize_embedding_index}"
        )

    tensorboard_dir = Path(
        script_args.tensorboard_dir
        if script_args.tensorboard_dir
        else Path(training_args.logging_dir) / "tensorboard"
    ).resolve()
    if training_args.local_rank == 0:
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        print(f"TensorBoard directory: {tensorboard_dir}")

    report_to = list(training_args.report_to) if training_args.report_to else []
    if "tensorboard" not in report_to:
        report_to.append("tensorboard")
    training_args.report_to = report_to
    training_args.logging_dir = str(tensorboard_dir)

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    trainable_names = freeze_for_embedding_and_lm_head(model, script_args.freeze_llm)
    if not trainable_names:
        raise ValueError("No trainable parameters matched embed_tokens/lm_head.")

    embedding_masker = None
    if script_args.freeze_llm:
        embedding_masker = EmbeddingRangeMasker(model, script_args.start_optimize_embedding_index)

    total_params, trainable_params = count_parameters(model)
    if training_args.local_rank == 0:
        print("Embedding/lm_head training enabled")
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print("Trainable parameter names:")
        for name in trainable_names:
            print(f"  {name}")

    train_dataset = prepare_dataset(
        train_data_path,
        sample_size=script_args.sample_size,
        local_rank=training_args.local_rank,
    )
    val_dataset = prepare_dataset(
        val_data_path,
        sample_size=script_args.sample_size,
        local_rank=training_args.local_rank,
    )

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, script_args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data",
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, script_args.max_length),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation data",
    )

    class TensorBoardMetricsCallback(TrainerCallback):
        def __init__(self, log_dir):
            self.log_dir = log_dir
            self.writer = None

        def _ensure_writer(self):
            if self.writer is None:
                self.writer = SummaryWriter(log_dir=str(self.log_dir))

        def on_train_begin(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                self._ensure_writer()
                self.writer.add_text("meta/model_dir", str(model_dir))
                self.writer.add_text("meta/train_data_path", str(train_data_path))
                self.writer.add_text("meta/val_data_path", str(val_data_path))
                self.writer.add_scalar("meta/total_parameters", total_params, 0)
                self.writer.add_scalar("meta/trainable_parameters", trainable_params, 0)
                self.writer.add_scalar("meta/train_samples", len(train_dataset), 0)
                self.writer.add_scalar("meta/valid_samples", len(val_dataset), 0)
                self.writer.flush()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero or not logs:
                return
            self._ensure_writer()
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, state.global_step)
            self.writer.flush()

        def on_train_end(self, args, state, control, **kwargs):
            if self.writer is not None:
                self.writer.close()

    class EmbeddingRangeMaskCallback(TrainerCallback):
        def __init__(self, masker):
            self.masker = masker

        def on_step_end(self, args, state, control, **kwargs):
            if self.masker is not None:
                self.masker.restore_frozen_prefix()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            TensorBoardMetricsCallback(tensorboard_dir),
            EmbeddingRangeMaskCallback(embedding_masker),
        ],
    )

    if training_args.local_rank == 0:
        print("Starting full alignment training...")
    trainer.train()

    if training_args.local_rank == 0:
        print("Final evaluation...")
    result = trainer.evaluate()
    if training_args.local_rank == 0:
        print(result)

    trainer.save_model(training_args.output_dir)
    if training_args.local_rank == 0:
        print(f"Model saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()
