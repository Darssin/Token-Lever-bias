#!/usr/bin/env bash
set -euo pipefail

# bash Token-Lever-bias/pretrain_json_pipeline/generate_all_datasets.sh \
#   /path/to/user_sequence.txt \
#   /path/to/item_meta.with_sid.json \
#   /path/to/output_dir

if [ "$#" -lt 3 ]; then
  echo "Usage: bash generate_all_datasets.sh <user_sequence.txt> <item_meta.with_sid.json> <output_dir>"
  exit 1
fi

USER_SEQUENCE="$1"
ITEM_META_WITH_SID="$2"
OUTPUT_DIR="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTPUT_DIR"

python "$SCRIPT_DIR/generate_training_data.py" \
  --user_sequence "$USER_SEQUENCE" \
  --item_meta_with_sid "$ITEM_META_WITH_SID" \
  --output_dir "$OUTPUT_DIR"

python "$SCRIPT_DIR/generate_RA_data.py" \
  --user_sequence "$USER_SEQUENCE" \
  --item_meta_with_sid "$ITEM_META_WITH_SID" \
  --output_dir "$OUTPUT_DIR"

python "$SCRIPT_DIR/generate_sid_prediction_data.py" \
  --user_sequence "$USER_SEQUENCE" \
  --item_meta_with_sid "$ITEM_META_WITH_SID" \
  --output_dir "$OUTPUT_DIR"

echo "All datasets have been generated under: $OUTPUT_DIR"
