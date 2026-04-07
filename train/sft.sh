#!/bin/bash

MODEL_DIR="/mnt/cfs/chubaofs_ads_train_image/ouchuang/bias/0.6B_warmup/checkpoint-5250"
TRAIN_DATA="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets/training_sid_only_data_train.parquet"
VAL_DATA="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets/training_sid_only_data_valid.parquet"
OUTPUT_DIR="/mnt/cfs/chubaofs_ads_train_image/ouchuang/bias/0.6B_sft/"
USE_LORA=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

DEEPSPEED_CMD=(
    deepspeed
    --num_gpus 8
    ./train/sft.py
    --model_name_or_path "${MODEL_DIR}"
    --train_data_path "${TRAIN_DATA}"
    --val_data_path "${VAL_DATA}"
)

if [ "${USE_LORA}" = "true" ]; then
    DEEPSPEED_CMD+=(
        --use_lora True
        --lora_r 128
        --lora_alpha 128
        --lora_dropout 0.05
        --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
else
    DEEPSPEED_CMD+=(--use_lora False)
fi

DEEPSPEED_CMD+=(
    --per_device_train_batch_size 64
    --num_train_epochs 10
    --gradient_checkpointing True
    --bf16 True
    --deepspeed ./train/ds_config_zero2.json
    --output_dir "${OUTPUT_DIR}"
    --logging_dir "/export/App/training_platform/PinoModel/models/log_dir"
    --logging_steps 10
    --eval_strategy steps
    --eval_steps 20
    --eval_on_start true
    --save_strategy epoch
    --save_total_limit 10
    --metric_for_best_model eval_loss
    --greater_is_better False
    --load_best_model_at_end True
    --optim adamw_torch
    --learning_rate 1e-5
    --warmup_ratio 0.1
    --weight_decay 0.01
    --adam_beta1 0.9
    --adam_beta2 0.999
    --adam_epsilon 1e-8
    --max_grad_norm 1.0
    --dataloader_num_workers 4
    --remove_unused_columns False
)

export WANDB_MODE=offline

nohup env WANDB_MODE=offline "${DEEPSPEED_CMD[@]}" 2>&1 | tee -a beauty_sid_rec.log &
