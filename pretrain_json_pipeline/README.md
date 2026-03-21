# Pretrain JSON Pipeline

This folder contains a standalone pipeline for data shaped like `*.pretrain.json`.
Input format:
- one JSON file mapping `item_id -> item fields`
- `sid` may be missing

Recommended workflow:
1. Generate embeddings first
2. Run RQ-KMeans from the saved embeddings
3. Merge generated `sid` back into the JSON

## Step 1: Generate Embeddings

```bash
python pretrain_json_pipeline/generate_embeddings.py \
  --input_json data/Beauty.pretrain.json \
  --plm_checkpoint your_emb_model_path \
  --plm_name qwen
```

For multi-GPU embedding:

```bash
accelerate launch --num_processes 8 pretrain_json_pipeline/generate_embeddings.py \
  --input_json data/Beauty.pretrain.json \
  --plm_checkpoint your_emb_model_path \
  --plm_name qwen
```

This stage writes:
- `*.emb-qwen-td.npy`
- `*.item_ids.json`
- `*.item.json`

## Step 2: Generate SIDs And Merge JSON

```bash
python pretrain_json_pipeline/generate_sids_and_merge.py \
  --input_json data/Beauty.pretrain.json \
  --embedding_path data/pretrain_json_outputs/Beauty.pretrain.emb-qwen-td.npy \
  --item_ids_path data/pretrain_json_outputs/Beauty.pretrain.item_ids.json \
  --num_levels 4 \
  --codebook_size 256
```

This stage writes:
- `*.index.json`
- `*.with_sid.json`
- `*.item.json`

## Output Directory

By default, all outputs are written to `pretrain_json_outputs/` next to the input file.
