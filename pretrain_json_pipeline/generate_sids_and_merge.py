import argparse

import numpy as np

from pretrain_json_pipeline.common import (
    build_index_json,
    load_json,
    merge_sid_into_json,
    resolve_stage2_paths,
    run_rq_kmeans,
    save_json,
    set_seed,
    strip_sid_fields,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2: run 4-level RQ-KMeans from embeddings and merge sid back into JSON."
    )
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--embedding_path", type=str, required=True)
    parser.add_argument("--item_ids_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--num_levels", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--uniform", action="store_true")
    parser.add_argument("--sinkhorn_batch_size", type=int, default=8192)
    parser.add_argument("--sinkhorn_iters", type=int, default=30)
    parser.add_argument("--sid_offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    _, output_json, index_path, item_meta_path = resolve_stage2_paths(
        args.input_json, args.output_dir, args.output_json
    )

    item_map = load_json(args.input_json)
    item_ids = load_json(args.item_ids_path)
    embeddings = np.load(args.embedding_path)

    print(f"Loaded {len(item_ids)} item ids from {args.item_ids_path}")
    print(f"Loaded embeddings {embeddings.shape} from {args.embedding_path}")

    if len(item_ids) != len(embeddings):
        raise ValueError(
            f"Item id count ({len(item_ids)}) does not match embedding rows ({len(embeddings)})."
        )

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

    save_json(index_json, str(index_path))
    save_json(merged_json, str(output_json))
    save_json(strip_sid_fields(item_map), str(item_meta_path))

    print(f"Saved index JSON to {index_path}")
    print(f"Saved merged JSON to {output_json}")
    print(f"Saved item metadata JSON to {item_meta_path}")


if __name__ == "__main__":
    main()
