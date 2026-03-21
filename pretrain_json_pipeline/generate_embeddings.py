import argparse

from accelerate import Accelerator

from pretrain_json_pipeline.common import (
    build_item_texts,
    encode_items,
    load_json,
    load_text_encoder,
    resolve_stage1_paths,
    save_embedding_artifacts,
    save_json,
    set_seed,
    strip_sid_fields,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: generate embeddings from single-file pretrain JSON."
    )
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--plm_checkpoint", type=str, required=True)
    parser.add_argument("--plm_name", type=str, default="qwen")
    parser.add_argument("--features", nargs="+", default=["title", "description"])
    parser.add_argument("--max_sent_len", type=int, default=2048)
    parser.add_argument("--embedding_batch_size", type=int, default=128)
    parser.add_argument("--word_drop_ratio", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    _, emb_path, item_ids_path, item_meta_path = resolve_stage1_paths(
        args.input_json, args.output_dir, args.plm_name
    )

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
    save_json(strip_sid_fields(item_map), str(item_meta_path))
    print(f"Saved item metadata JSON to {item_meta_path}")


if __name__ == "__main__":
    main()
