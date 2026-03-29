#!/bin/bash

MODEL_DIR="${MODEL_DIR:-/path/to/warmup_or_base_model}"
TRAIN_DATA="${TRAIN_DATA:-./train/grc_sft_train.parquet}"
VAL_DATA="${VAL_DATA:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/grc_sft_output}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

DEEPSPEED_CMD=(
  deepspeed
  --num_gpus 8
  ./train/grc_sft.py
  --model_name_or_path "${MODEL_DIR}"
  --train_data_path "${TRAIN_DATA}"
)

if [ -n "${VAL_DATA}" ]; then
  DEEPSPEED_CMD+=(--val_data_path "${VAL_DATA}")
else
  DEEPSPEED_CMD+=(--validation_split_ratio 0.05)
fi

DEEPSPEED_CMD+=(
  --use_lora True
  --lora_r 64
  --lora_alpha 64
  --lora_dropout 0.05
  --rc_loss_weight 1.2
  --per_device_train_batch_size 16
  --num_train_epochs 3
  --gradient_checkpointing True
  --bf16 True
  --deepspeed ./train/ds_config_zero2.json
  --output_dir "${OUTPUT_DIR}"
  --logging_steps 10
  --eval_strategy steps
  --eval_steps 200
  --save_strategy steps
  --save_steps 200
  --save_total_limit 3
  --metric_for_best_model eval_loss
  --greater_is_better False
  --load_best_model_at_end True
  --learning_rate 1e-5
  --warmup_ratio 0.1
  --weight_decay 0.01
  --dataloader_num_workers 4
  --remove_unused_columns False
)

export WANDB_MODE=offline
"${DEEPSPEED_CMD[@]}"
