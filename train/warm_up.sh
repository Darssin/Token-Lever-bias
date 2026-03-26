#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_DIR="/home/ea-ea-ads-purchasechain-1/models/Qwen3-0.6B"
EXPANDED_MODEL_DIR="/home/ea-ea-ads-purchasechain-1/wubintao/OneRec-0.6B/OneRec-0.6B-expand"
DATASET_DIR="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets"
TRAINING_OUTPUT_DIR="/mnt/cfs/chubaofs_ads_train_image/ouchuang/bias/0.6B_warmup"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DATA="${DATASET_DIR}/training_sid_only_data_train.parquet"
VAL_DATA="${DATASET_DIR}/training_sid_only_data_valid.parquet"

START_OPTIMIZE_EMBEDDING_INDEX="$(
  python -c "from transformers import AutoTokenizer; print(len(AutoTokenizer.from_pretrained(r'''$BASE_MODEL_DIR''')))"
)"

mkdir -p "$TRAINING_OUTPUT_DIR"
echo "Computed start_optimize_embedding_index=${START_OPTIMIZE_EMBEDDING_INDEX} from base model: $BASE_MODEL_DIR"
deepspeed "$SCRIPT_DIR/warm_up.py" \
  --model_dir "$EXPANDED_MODEL_DIR" \
  --train_data_path "$TRAIN_DATA" \
  --val_data_path "$VAL_DATA" \
  --freeze_llm True \
  --start_optimize_embedding_index "$START_OPTIMIZE_EMBEDDING_INDEX" \
  --tensorboard_dir "/export/App/training_platform/PinoModel/models/log_dir" \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 15 \
  --gradient_checkpointing True \
  --bf16 True \
  --deepspeed "$SCRIPT_DIR/ds_config_zero2.json" \
  --output_dir "$TRAINING_OUTPUT_DIR" \
  --logging_dir "/export/App/training_platform/PinoModel/models/log_dir" \
  --report_to tensorboard \
  --logging_steps 10 \
  --eval_strategy epoch \
  --eval_on_start False \
  --save_strategy epoch \
  --save_total_limit 15 \
  --metric_for_best_model eval_loss \
  --greater_is_better False \
  --load_best_model_at_end True \
  --optim adamw_torch \
  --learning_rate 1e-4 \
  --warmup_ratio 0.0 \
  --weight_decay 0.0 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --dataloader_num_workers 4 \
  --remove_unused_columns False

echo "Stage 1 full fine-tuning completed. Output saved to: $TRAINING_OUTPUT_DIR"
