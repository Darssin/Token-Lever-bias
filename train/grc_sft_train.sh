#!/bin/bash

MODEL_DIR="/mnt/cfs/chubaofs_ads_train_image/ouchuang/bias/0.6B_sft"
TRAIN_DATA="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_grc/grc_sft_train.parquet"
VAL_DATA="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_grc/grc_sft_val.parquet"
OUTPUT_DIR="/mnt/cfs/chubaofs_ads_train_image/ouchuang/bias/0.6B_grc_sft"

NUM_GPUS=8
USE_LORA=true
LORA_R=64
LORA_ALPHA=64
LORA_DROPOUT=0.05
RC_LOSS_WEIGHT=1.2

PER_DEVICE_TRAIN_BATCH_SIZE=16
NUM_TRAIN_EPOCHS=3
LEARNING_RATE=1e-5
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.01

LOGGING_STEPS=10
EVAL_STEPS=200
SAVE_STEPS=200
SAVE_TOTAL_LIMIT=3
DATALOADER_NUM_WORKERS=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

LOCAL_LOG_DIR="./train/logs"
LOG_PATH="${LOCAL_LOG_DIR}/grc_sft_train.log"
mkdir -p "${LOCAL_LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

DEEPSPEED_CMD=(
  deepspeed
  --num_gpus "${NUM_GPUS}"
  ./train/grc_sft.py
  --model_name_or_path "${MODEL_DIR}"
  --train_data_path "${TRAIN_DATA}"
)

if [ -n "${VAL_DATA}" ]; then
  DEEPSPEED_CMD+=(--val_data_path "${VAL_DATA}")
else
  DEEPSPEED_CMD+=(--validation_split_ratio 0.05)
fi

if [ "${USE_LORA}" = "true" ]; then
  DEEPSPEED_CMD+=(
    --use_lora True
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
  )
else
  DEEPSPEED_CMD+=(--use_lora False)
fi

DEEPSPEED_CMD+=(
  --rc_loss_weight "${RC_LOSS_WEIGHT}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --gradient_checkpointing True
  --bf16 True
  --deepspeed ./train/ds_config_zero2.json
  --output_dir "${OUTPUT_DIR}"
  --logging_steps "${LOGGING_STEPS}"
  --eval_strategy steps
  --eval_steps "${EVAL_STEPS}"
  --save_strategy steps
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --metric_for_best_model eval_loss
  --greater_is_better False
  --load_best_model_at_end True
  --learning_rate "${LEARNING_RATE}"
  --warmup_ratio "${WARMUP_RATIO}"
  --weight_decay "${WEIGHT_DECAY}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --remove_unused_columns False
)

export WANDB_MODE=offline
echo "Writing training log to ${LOG_PATH}"
"${DEEPSPEED_CMD[@]}" 2>&1 | tee -a "${LOG_PATH}"
