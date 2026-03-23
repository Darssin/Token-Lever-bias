#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: bash run_training_sid_only.sh <expanded_model_dir> <dataset_dir> <training_output_dir>"
  echo "dataset_dir should contain training_sid_only_data_train.parquet and training_sid_only_data_valid.parquet"
  exit 1
fi

EXPANDED_MODEL_DIR="$1"
DATASET_DIR="$2"
TRAINING_OUTPUT_DIR="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DATA="${DATASET_DIR}/training_sid_only_data_train.parquet"
VAL_DATA="${DATASET_DIR}/training_sid_only_data_valid.parquet"
LOG_FILE="${LOG_FILE:-$(pwd)/train_sid_only.LOG}"
DS_CONFIG="${DS_CONFIG:-$SCRIPT_DIR/ds_config_zero2.json}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$TRAINING_OUTPUT_DIR/tensorboard}"

mkdir -p "$TRAINING_OUTPUT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$TENSORBOARD_DIR"

deepspeed "$SCRIPT_DIR/train_sid_only_sft.py" \
  --model_name_or_path "$EXPANDED_MODEL_DIR" \
  --train_data_path "$TRAIN_DATA" \
  --val_data_path "$VAL_DATA" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-8}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-15}" \
  --gradient_checkpointing True \
  --bf16 "${BF16:-True}" \
  --deepspeed "$DS_CONFIG" \
  --output_dir "$TRAINING_OUTPUT_DIR" \
  --logging_dir "$TENSORBOARD_DIR" \
  --report_to tensorboard \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --eval_strategy epoch \
  --eval_on_start False \
  --save_strategy epoch \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-15}" \
  --metric_for_best_model eval_loss \
  --greater_is_better False \
  --load_best_model_at_end True \
  --optim adamw_torch \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --warmup_ratio "${WARMUP_RATIO:-0.0}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  --remove_unused_columns False 2>&1 | tee -a "$LOG_FILE"

echo "SID-only SFT training completed. Output saved to: $TRAINING_OUTPUT_DIR"
echo "Training log appended to: $LOG_FILE"
echo "TensorBoard logs saved to: $TENSORBOARD_DIR"
