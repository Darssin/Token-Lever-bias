#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: bash run_training_stage1_full.sh <base_model_dir> <expanded_model_dir> <dataset_dir> <training_output_dir>"
  echo "dataset_dir should contain training_align_data_train.parquet and training_align_data_valid.parquet"
  exit 1
fi

BASE_MODEL_DIR="$1"
EXPANDED_MODEL_DIR="$2"
DATASET_DIR="$3"
TRAINING_OUTPUT_DIR="$4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DATA="${DATASET_DIR}/training_align_data_train.parquet"
VAL_DATA="${DATASET_DIR}/training_align_data_valid.parquet"
LOG_FILE="${LOG_FILE:-$(pwd)/train.LOG}"
DS_CONFIG="${DS_CONFIG:-$SCRIPT_DIR/ds_config_zero2.json}"

START_OPTIMIZE_EMBEDDING_INDEX="$(
  python -c "from transformers import AutoTokenizer; print(len(AutoTokenizer.from_pretrained(r'''$BASE_MODEL_DIR''')))"
)"

mkdir -p "$TRAINING_OUTPUT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"
echo "Computed start_optimize_embedding_index=${START_OPTIMIZE_EMBEDDING_INDEX} from base model: $BASE_MODEL_DIR" | tee -a "$LOG_FILE"
deepspeed "$SCRIPT_DIR/train_align_full.py" \
  --model_dir "$EXPANDED_MODEL_DIR" \
  --train_data_path "$TRAIN_DATA" \
  --val_data_path "$VAL_DATA" \
  --freeze_llm True \
  --start_optimize_embedding_index "$START_OPTIMIZE_EMBEDDING_INDEX" \
  --tensorboard_dir "${TENSORBOARD_DIR:-$TRAINING_OUTPUT_DIR/tensorboard}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-8}" \
  --gradient_accumulation_steps 1 \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-15}" \
  --gradient_checkpointing True \
  --bf16 "${BF16:-True}" \
  --deepspeed "$DS_CONFIG" \
  --output_dir "$TRAINING_OUTPUT_DIR" \
  --logging_dir "${LOGGING_DIR:-$TRAINING_OUTPUT_DIR/logs}" \
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

echo "Stage 1 full fine-tuning completed. Output saved to: $TRAINING_OUTPUT_DIR"
echo "Training log appended to: $LOG_FILE"
