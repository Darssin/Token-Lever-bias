import os
import re
import ast
import json
import gzip
import html
import argparse
import collections
from typing import Dict, List, Iterable, Tuple, Any, Optional

def open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def parse_amazon_lines(path: str) -> Iterable[dict]:
    """
    Compatible with common Amazon 2014 raw formats:
    1) JSON lines
    2) Python literal lines used in old UCSD Amazon dumps
    """
    with open_maybe_gzip(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                try:
                    yield ast.literal_eval(line)
                except Exception as e:
                    raise ValueError(f"Failed to parse line {line_no} in {path}: {e}")


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        text = " ".join(str(x) for x in text if x is not None)
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_categories(categories: Any) -> str:
    """
    Convert Amazon 2014 category structures to:
    "Beauty > Hair Care > Conditioners"
    """
    if not categories:
        return ""

    # Common format: [["Beauty", "Hair Care", "Conditioners"]]
    if isinstance(categories, list):
        if len(categories) == 0:
            return ""
        if isinstance(categories[0], list):
            # Prefer the first valid path
            for path in categories:
                cleaned = [clean_text(x) for x in path if clean_text(x)]
                if cleaned:
                    return " > ".join(cleaned)
            return ""
        else:
            cleaned = [clean_text(x) for x in categories if clean_text(x)]
            return " > ".join(cleaned)

    return clean_text(categories)


def load_reviews(review_path: str) -> List[dict]:
    reviews = []
    for obj in parse_amazon_lines(review_path):
        reviewer = obj.get("reviewerID")
        asin = obj.get("asin")
        ts = obj.get("unixReviewTime")

        if reviewer is None or asin is None or ts is None:
            continue

        reviews.append({
            "reviewerID": str(reviewer),
            "asin": str(asin),
            "unixReviewTime": int(ts),
            "overall": obj.get("overall"),
        })
    return reviews


def load_meta(meta_path: str) -> Dict[str, dict]:
    """
    Returns:
        asin -> {
            "title": ...,
            "description": ...,
            "categories": ...
        }
    """
    meta = {}
    for obj in parse_amazon_lines(meta_path):
        asin = obj.get("asin")
        if asin is None:
            continue

        asin = str(asin)
        title = clean_text(obj.get("title", ""))
        description = clean_text(obj.get("description", ""))
        categories = normalize_categories(obj.get("categories", []))

        meta[asin] = {
            "title": title,
            "description": description,
            "categories": categories,
        }
    return meta


def iterative_k_core(
    reviews: List[dict],
    user_k: int = 5,
    item_k: int = 5,
) -> Tuple[List[dict], Dict[str, int], Dict[str, int]]:
    """
    Iteratively filter until all remaining users/items satisfy:
      user interactions >= user_k
      item interactions >= item_k
    """
    filtered = reviews

    while True:
        user_counts = collections.Counter(r["reviewerID"] for r in filtered)
        item_counts = collections.Counter(r["asin"] for r in filtered)

        new_filtered = [
            r for r in filtered
            if user_counts[r["reviewerID"]] >= user_k and item_counts[r["asin"]] >= item_k
        ]

        if len(new_filtered) == len(filtered):
            final_user_counts = collections.Counter(r["reviewerID"] for r in new_filtered)
            final_item_counts = collections.Counter(r["asin"] for r in new_filtered)
            return new_filtered, dict(final_user_counts), dict(final_item_counts)

        filtered = new_filtered


def build_user_sequences(
    reviews: List[dict],
    max_history: Optional[int] = None
) -> Dict[str, List[str]]:
    """
    Build chronological item sequences for each user.
    Keep duplicate interactions, matching the classic SASRec-style behavior.
    """
    user_reviews = collections.defaultdict(list)
    for r in reviews:
        user_reviews[r["reviewerID"]].append(r)

    user_sequences = {}
    for user, rs in user_reviews.items():
        rs.sort(key=lambda x: (x["unixReviewTime"], x["asin"]))
        seq = [r["asin"] for r in rs]
        if max_history is not None:
            seq = seq[-max_history:]
        user_sequences[user] = seq

    return user_sequences


def remap_ids(
    user_sequences: Dict[str, List[str]]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Remap user/item IDs to 1-based integer IDs.
    """
    user2id = {}
    item2id = {}
    next_uid = 1
    next_iid = 1

    for user in sorted(user_sequences.keys()):
        user2id[user] = next_uid
        next_uid += 1

    for user in sorted(user_sequences.keys()):
        for asin in user_sequences[user]:
            if asin not in item2id:
                item2id[asin] = next_iid
                next_iid += 1

    return user2id, item2id


def build_item_json(
    item2id: Dict[str, int],
    meta: Dict[str, dict]
) -> Dict[str, dict]:
    """
    Output format:
    {
      "1": {
        "title": "...",
        "description": "...",
        "categories": "Beauty > Hair Care > Conditioners"
      },
      ...
    }
    """
    output = {}
    for asin, iid in item2id.items():
        m = meta.get(asin, {})
        output[str(iid)] = {
            "title": m.get("title", ""),
            "description": m.get("description", ""),
            "categories": m.get("categories", ""),
        }
    return output


def write_item_json(path: str, item_json: Dict[str, dict]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(item_json, f, ensure_ascii=False, indent=2)


def write_user_sequence_txt(
    path: str,
    user_sequences: Dict[str, List[str]],
    user2id: Dict[str, int],
    item2id: Dict[str, int]
):
    """
    Each line:
    user_id item1 item2 item3 ...
    """
    with open(path, "w", encoding="utf-8") as f:
        for raw_user in sorted(user_sequences.keys(), key=lambda u: user2id[u]):
            uid = user2id[raw_user]
            mapped_items = [str(item2id[asin]) for asin in user_sequences[raw_user]]
            if not mapped_items:
                continue
            f.write(f"{uid} {' '.join(mapped_items)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_file", type=str, required=True, help="Path to meta_Beauty.json(.gz)")
    parser.add_argument("--review_file", type=str, required=True, help="Path to reviews_Beauty.json(.gz)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--user_k", type=int, default=5, help="User k-core threshold")
    parser.add_argument("--item_k", type=int, default=5, help="Item k-core threshold")
    parser.add_argument(
        "--max_history",
        type=int,
        default=50,
        help="Optional max user sequence length. Use 50 to align with OneRec-Think training history limit."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading reviews...")
    reviews = load_reviews(args.review_file)
    print(f"Loaded raw reviews: {len(reviews)}")

    print("Loading metadata...")
    meta = load_meta(args.meta_file)
    print(f"Loaded meta items: {len(meta)}")

    print("Applying iterative k-core filtering...")
    filtered_reviews, user_counts, item_counts = iterative_k_core(
        reviews,
        user_k=args.user_k,
        item_k=args.item_k,
    )
    print(
        f"After filtering: users={len(user_counts)}, "
        f"items={len(item_counts)}, reviews={len(filtered_reviews)}"
    )

    print("Building user sequences...")
    user_sequences = build_user_sequences(
        filtered_reviews,
        max_history=args.max_history
    )

    print("Remapping ids...")
    user2id, item2id = remap_ids(user_sequences)
    print(f"Mapped users={len(user2id)}, mapped items={len(item2id)}")

    print("Building output files...")
    item_json = build_item_json(item2id, meta)

    item_json_path = os.path.join(args.output_dir, "item_meta.json")
    user_seq_path = os.path.join(args.output_dir, "user_sequence.txt")

    write_item_json(item_json_path, item_json)
    write_user_sequence_txt(user_seq_path, user_sequences, user2id, item2id)

    print("Done.")
    print(f"item_meta.json -> {item_json_path}")
    print(f"user_sequence.txt -> {user_seq_path}")


if __name__ == "__main__":
    main()