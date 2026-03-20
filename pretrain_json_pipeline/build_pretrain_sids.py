import argparse
from pathlib import Path

from accelerate import Accelerator

from pretrain_json_pipeline.common import (
    build_index_json,
    build_item_texts,
    encode_items,
    load_json,
    load_text_encoder,
    merge_sid_into_json,
    run_rq_kmeans,
    save_embedding_artifacts,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone pipeline for single-file pretrain JSON: embedding -> 4-level RQ-KMeans -> sid merge-back."
    )
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--plm_checkpoint", type=str, required=True)
    parser.add_argument("--plm_name", type=str, default="qwen")
    parser.add_argument("--features", nargs="+", default=["title", "description"])
    parser.add_argument("--max_sent_len", type=int, default=2048)
    parser.add_argument("--embedding_batch_size", type=int, default=1024)
    parser.add_argument("--word_drop_ratio", type=float, default=-1)
    parser.add_argument("--num_levels", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--uniform", action="store_true")
    parser.add_argument("--sinkhorn_batch_size", type=int, default=8192)
    parser.add_argument("--sinkhorn_iters", type=int, default=30)
    parser.add_argument("--sid_offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def resolve_paths(args):
    input_path = Path(args.input_json)
    stem = input_path.stem
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "pretrain_json_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = Path(args.output_json) if args.output_json else output_dir / f"{stem}.with_sid.json"
    emb_path = output_dir / f"{stem}.emb-{args.plm_name}-td.npy"
    item_ids_path = output_dir / f"{stem}.item_ids.json"
    index_path = output_dir / f"{stem}.index.json"
    item_meta_path = output_dir / f"{stem}.item.json"
    return output_dir, output_json, emb_path, item_ids_path, index_path, item_meta_path


def strip_sid_fields(item_map):
    return {
        str(item_id): {k: v for k, v in fields.items() if k != "sid"}
        for item_id, fields in item_map.items()
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    _, output_json, emb_path, item_ids_path, index_path, item_meta_path = resolve_paths(args)

    item_map = load_json(args.input_json)
    item_texts = build_item_texts(item_map, args.features)
    print(f"Loaded {len(item_texts)} items from {args.input_json}")

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Embedding stage will run on {accelerator.num_processes} process(es).")

    tokenizer, model = load_text_encoder(args.plm_checkpoint)
    model = model.to(accelerator.device)
    model.eval()

    item_ids, embeddings = encode_items(
        accelerator=accelerator,
        item_texts=item_texts,
        tokenizer=tokenizer,
        model=model,
        max_sent_len=args.max_sent_len,
        batch_size=args.embedding_batch_size,
        word_drop_ratio=args.word_drop_ratio,
    )

    if not accelerator.is_main_process:
        return

    print(f"Embeddings shape: {embeddings.shape}")
    save_embedding_artifacts(item_ids, embeddings, str(emb_path), str(item_ids_path))

    codes = run_rq_kmeans(
        embeddings=embeddings,
        num_levels=args.num_levels,
        codebook_size=args.codebook_size,
        uniform=args.uniform,
        sinkhorn_batch_size=args.sinkhorn_batch_size,
        sinkhorn_iters=args.sinkhorn_iters,
    )

    index_json = build_index_json(item_ids, codes, sid_offset=args.sid_offset)
    merged_json = merge_sid_into_json(item_map, item_ids, codes, sid_offset=args.sid_offset)
    item_meta_json = strip_sid_fields(item_map)

    save_json(index_json, str(index_path))
    save_json(merged_json, str(output_json))
    save_json(item_meta_json, str(item_meta_path))

    print(f"Saved merged JSON to {output_json}")
    print(f"Saved index JSON to {index_path}")
    print(f"Saved item metadata JSON to {item_meta_path}")


if __name__ == "__main__":
    main()
