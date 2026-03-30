#!/bin/bash

BASE_MODEL_PATH="/mnt/cfs/chubaofs_ads_train_image/ouchuang/bias/0.6B_sft"
METADATA_PATH="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_beauty/item_meta.with_sid.json"
TRAIN_INTERACTION_DATA_PATH="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets/training_sid_only_data_train.parquet"
TRAIN_SFT_OUTPUT_PATH="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_grc/grc_sft_train.parquet"
TRAIN_VERL_OUTPUT_PATH="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_grc/grc_verl_train.parquet"
TRAIN_SUMMARY_OUTPUT_PATH="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_grc/grc_dataset_summary.json"
METADATA_CACHE_OUTPUT_PATH="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_grc/grc_metadata_cache.jsonl"

VAL_INTERACTION_DATA_PATH="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets/training_sid_only_data_valid.parquet"
VAL_SFT_OUTPUT_PATH="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_grc/grc_sft_val.parquet"
VAL_VERL_OUTPUT_PATH="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_grc/grc_verl_val.parquet"
VAL_SUMMARY_OUTPUT_PATH="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_grc/grc_val_summary.json"

NUM_GPUS=8
NUM_BEAMS=8
NUM_RETURN_SEQUENCES=4
BATCH_SIZE=32
DRAFT_MAX_NEW_TOKENS=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
LOCAL_LOG_DIR="./train/logs"
mkdir -p "${LOCAL_LOG_DIR}"

run_build() {
    local split_name="$1"
    local interaction_data_path="$2"
    local sft_output_path="$3"
    local verl_output_path="$4"
    local summary_output_path="$5"

    if [ -z "${interaction_data_path}" ]; then
        return
    fi

    local log_path="${LOCAL_LOG_DIR}/$(basename "${sft_output_path%.parquet}").build.log"

    if [ "${NUM_GPUS}" -gt 1 ]; then
        LAUNCH_CMD=(
            torchrun
            --nproc_per_node "${NUM_GPUS}"
            ./train/build_grc_sft_dataset.py
        )
    else
        LAUNCH_CMD=(
            python
            ./train/build_grc_sft_dataset.py
        )
    fi

    LAUNCH_CMD+=(
        --base_model_path "${BASE_MODEL_PATH}"
        --interaction_data_path "${interaction_data_path}"
        --metadata_path "${METADATA_PATH}"
        --sft_output_path "${sft_output_path}"
        --verl_output_path "${verl_output_path}"
        --summary_output_path "${summary_output_path}"
        --metadata_cache_output_path "${METADATA_CACHE_OUTPUT_PATH}"
        --num_beams "${NUM_BEAMS}"
        --num_return_sequences "${NUM_RETURN_SEQUENCES}"
        --batch_size "${BATCH_SIZE}"
        --draft_max_new_tokens "${DRAFT_MAX_NEW_TOKENS}"
    )

    echo "Building ${split_name} dataset..."
    echo "Writing ${split_name} build log to ${log_path}"
    "${LAUNCH_CMD[@]}" 2>&1 | tee -a "${log_path}"
}

run_build \
    "train" \
    "${TRAIN_INTERACTION_DATA_PATH}" \
    "${TRAIN_SFT_OUTPUT_PATH}" \
    "${TRAIN_VERL_OUTPUT_PATH}" \
    "${TRAIN_SUMMARY_OUTPUT_PATH}"

run_build \
    "val" \
    "${VAL_INTERACTION_DATA_PATH}" \
    "${VAL_SFT_OUTPUT_PATH}" \
    "${VAL_VERL_OUTPUT_PATH}" \
    "${VAL_SUMMARY_OUTPUT_PATH}"
