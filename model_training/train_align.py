#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset


def parse_script_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_dir", default="./expanded_model")
    parser.add_argument("--train_data_path", default="./training_align_data_train.parquet")
    parser.add_argument("--val_data_path", default="./training_align_data_valid.parquet")
    parser.add_argument("--num_levels", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--tensorboard_dir", default=None)
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


def get_special_tokens(num_levels=4, codebook_size=256):
    special_tokens = ["<|sid_begin|>", "<|sid_end|>"]
    for level in range(num_levels):
        prefix = f"s_{chr(97 + level)}"
        for idx in range(codebook_size):
            special_tokens.append(f"<{prefix}_{idx}>")
    return special_tokens


def main():
    from peft import TrainableTokensConfig, get_peft_model
    from torch.utils.tensorboard import SummaryWriter
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

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

    if training_args.local_rank == 0:
        print(f"Using model_dir: {model_dir}")
        print(f"Training data path: {train_data_path}")
        print(f"Validation data path: {val_data_path}")
        print(f"num_levels={script_args.num_levels}, codebook_size={script_args.codebook_size}")

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

    special_tokens = get_special_tokens(
        num_levels=script_args.num_levels,
        codebook_size=script_args.codebook_size,
    )
    token_ids = tokenizer.convert_tokens_to_ids(special_tokens)
    valid_token_ids = [
        token_id
        for token_id in token_ids
        if token_id != tokenizer.unk_token_id
    ]

    if not valid_token_ids:
        raise ValueError("No valid SID special tokens found in tokenizer. Run expand_vocab.py first.")

    if training_args.local_rank == 0:
        print(f"Valid special tokens: {len(valid_token_ids)}")
        print(f"Training token ID range: {min(valid_token_ids)} to {max(valid_token_ids)}")

    lora_config = TrainableTokensConfig(
        token_indices=valid_token_ids,
        target_modules=["embed_tokens"],
        init_weights=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            TensorBoardMetricsCallback(tensorboard_dir),
        ],
    )

    if training_args.local_rank == 0:
        print("Starting alignment training...")
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
