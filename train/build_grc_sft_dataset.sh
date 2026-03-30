#!/bin/bash

BASE_MODEL_PATH="/path/to/base_sid_model"
INTERACTION_DATA_PATH="./train/training_sid_only_data_train.parquet"
METADATA_PATH="./train/Beauty.pretrain.json"
SFT_OUTPUT_PATH="./train/grc_sft_train.parquet"
VERL_OUTPUT_PATH="./train/grc_verl_train.parquet"
SUMMARY_OUTPUT_PATH="./train/grc_dataset_summary.json"
METADATA_CACHE_OUTPUT_PATH="./train/grc_metadata_cache.jsonl"

NUM_BEAMS=8
NUM_RETURN_SEQUENCES=4
BATCH_SIZE=32
DRAFT_MAX_NEW_TOKENS=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

PYTHON_CMD=(
    python
    ./train/build_grc_sft_dataset.py
    --base_model_path "${BASE_MODEL_PATH}"
    --interaction_data_path "${INTERACTION_DATA_PATH}"
    --metadata_path "${METADATA_PATH}"
    --sft_output_path "${SFT_OUTPUT_PATH}"
    --verl_output_path "${VERL_OUTPUT_PATH}"
    --summary_output_path "${SUMMARY_OUTPUT_PATH}"
    --metadata_cache_output_path "${METADATA_CACHE_OUTPUT_PATH}"
    --num_beams "${NUM_BEAMS}"
    --num_return_sequences "${NUM_RETURN_SEQUENCES}"
    --batch_size "${BATCH_SIZE}"
    --draft_max_new_tokens "${DRAFT_MAX_NEW_TOKENS}"
)

"${PYTHON_CMD[@]}"
