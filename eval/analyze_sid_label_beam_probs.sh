#!/usr/bin/env bash

MERGED_MODEL_PATH="/mnt/cfs/chubaofs_ads_train_image/ouchuang/bias/0.6B_sft/checkpoint-1400"
TEST_PARQUET="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets/training_sid_only_data_test.parquet"
GLOBAL_TRIE_FILE="./exact_trie.pkl"
OUTPUT_DIR="./sid_label_beam_outputs"

NUM_BEAMS=20
BATCH_SIZE=8
MAX_TOKENS=6

mkdir -p "${OUTPUT_DIR}"

python3 ./eval/analyze_sid_label_beam_probs.py \
  --model_path "${MERGED_MODEL_PATH}" \
  --test_parquet_file "${TEST_PARQUET}" \
  --global_trie_file "${GLOBAL_TRIE_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_beams "${NUM_BEAMS}" \
  --batch_size "${BATCH_SIZE}" \
  --max_new_tokens "${MAX_TOKENS}" \
  --temperature 0.6 \
  --top_p 1.0 \
  "$@"
