#!/bin/bash

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/path/to/base_sid_model}"
INTERACTION_DATA_PATH="${INTERACTION_DATA_PATH:-./train/training_sid_only_data_train.parquet}"
METADATA_PATH="${METADATA_PATH:-./train/Beauty.pretrain.json}"
SFT_OUTPUT_PATH="${SFT_OUTPUT_PATH:-./train/grc_sft_train.parquet}"
VERL_OUTPUT_PATH="${VERL_OUTPUT_PATH:-./train/grc_verl_train.parquet}"
SUMMARY_OUTPUT_PATH="${SUMMARY_OUTPUT_PATH:-./train/grc_dataset_summary.json}"
METADATA_CACHE_OUTPUT_PATH="${METADATA_CACHE_OUTPUT_PATH:-./train/grc_metadata_cache.jsonl}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

python ./train/build_grc_sft_dataset.py \
  --base_model_path "${BASE_MODEL_PATH}" \
  --interaction_data_path "${INTERACTION_DATA_PATH}" \
  --metadata_path "${METADATA_PATH}" \
  --sft_output_path "${SFT_OUTPUT_PATH}" \
  --verl_output_path "${VERL_OUTPUT_PATH}" \
  --summary_output_path "${SUMMARY_OUTPUT_PATH}" \
  --metadata_cache_output_path "${METADATA_CACHE_OUTPUT_PATH}" \
  --num_beams 8 \
  --num_return_sequences 4 \
  --batch_size 32 \
  --draft_max_new_tokens 8
