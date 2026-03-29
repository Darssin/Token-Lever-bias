#!/usr/bin/env python3

import html
import json
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import json5


SYSTEM_MESSAGE = (
    "You are a professional recommendation expert who needs to recommend "
    "the next possible purchase for users based on their purchase history. "
    "Please predict the most likely next product that the user will purchase "
    "based on the user's historical purchase information."
)

SID_PATTERN = re.compile(
    r"<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>"
)
SID_COMPONENT_PATTERN = re.compile(r"<s_[abcd]_\d+>")
TOKEN_CODE_PATTERN = re.compile(r"<s_[abcd]_(\d+)>")

EXPLICIT_BRAND_FIELDS = (
    "brand",
    "brand_name",
    "manufacturer",
    "maker",
    "by",
)

GENERIC_BRAND_STOPWORDS = {
    "pack",
    "set",
    "kit",
    "for",
    "with",
    "and",
    "of",
    "oz",
    "ounce",
    "ounces",
    "ml",
    "liter",
    "litre",
    "lb",
    "lbs",
    "pcs",
    "piece",
    "pieces",
    "count",
}


@dataclass
class CategoryReflection:
    label_token: str
    is_match: Optional[int]
    shared_depth: int
    target_depth: int
    draft_depth: int
    overlap_ratio: float
    severity_score: float
    severity_bucket: str
    target_leaf: str
    draft_leaf: str


@dataclass
class BrandReflection:
    label_token: str
    is_match: Optional[int]
    similarity: float
    severity_score: float
    severity_bucket: str
    target_brand: str
    draft_brand: str
    target_brand_source: str
    draft_brand_source: str
    target_brand_confidence: float
    draft_brand_confidence: float


@dataclass
class LocalizationReflection:
    position: int
    label_token: str
    token_hits: int
    token_misses: int
    exact_match: int


def get_loc_tokens(num_levels: int = 4) -> List[str]:
    return [f"<|loc_{idx}|>" for idx in range(1, num_levels + 2)]


def get_grc_special_tokens(num_levels: int = 4) -> List[str]:
    return [
        "<|grc_draft|>",
        "<|grc_reflect|>",
        "<|grc_correct|>",
        "<|leaf_match|>",
        "<|leaf_mismatch|>",
        "<|leaf_unknown|>",
        "<|brand_match|>",
        "<|brand_mismatch|>",
        "<|brand_unknown|>",
        *get_loc_tokens(num_levels=num_levels),
    ]


def load_json_like(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json5.loads(text)


def iter_metadata_records(raw_data: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if isinstance(raw_data, dict):
        for item_id, record in raw_data.items():
            if isinstance(record, dict):
                yield str(item_id), record
    elif isinstance(raw_data, list):
        for index, record in enumerate(raw_data):
            if not isinstance(record, dict):
                continue
            item_id = record.get("item_id", record.get("id", index))
            yield str(item_id), record


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_brand_text(text: str) -> str:
    text = html.unescape(text or "")
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9'&+\-/ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def infer_brand_from_title(title: str) -> Tuple[str, str, float]:
    title = html.unescape(title or "")
    title = normalize_whitespace(title)
    if not title:
        return "", "missing", 0.0

    head = re.split(r"[,:;(|\[]", title, maxsplit=1)[0].strip()
    cleaned = re.sub(r"[^A-Za-z0-9'&+\-/ ]+", " ", head)
    words = [word for word in cleaned.split() if word]
    brand_tokens = []
    for index, word in enumerate(words[:4]):
        lower = word.lower()
        if index > 0 and (lower in GENERIC_BRAND_STOPWORDS or any(char.isdigit() for char in lower)):
            break
        if index == 0 and lower in GENERIC_BRAND_STOPWORDS:
            continue
        brand_tokens.append(word)
    if not brand_tokens:
        return "", "missing", 0.0

    brand = normalize_brand_text(" ".join(brand_tokens))
    if not brand:
        return "", "missing", 0.0

    confidence = 0.45
    if len(brand_tokens) >= 2:
        confidence += 0.15
    if brand_tokens[0][:1].isupper():
        confidence += 0.15
    if not any(char.isdigit() for char in brand):
        confidence += 0.1
    confidence = min(confidence, 0.85)
    return brand, "title_prefix", confidence


def extract_brand(record: Dict[str, Any]) -> Tuple[str, str, float]:
    for field_name in EXPLICIT_BRAND_FIELDS:
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            brand = normalize_brand_text(value)
            if brand:
                return brand, f"field:{field_name}", 1.0
    return infer_brand_from_title(record.get("title", ""))


def parse_category_path(categories: Any) -> List[str]:
    if categories is None:
        return []
    if isinstance(categories, list):
        parts = [normalize_whitespace(str(part)) for part in categories]
        return [part for part in parts if part]
    text = normalize_whitespace(str(categories))
    if not text:
        return []
    parts = [normalize_whitespace(part) for part in text.split(">")]
    return [part for part in parts if part]


def extract_sid_from_text(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    match = SID_PATTERN.search(text)
    return match.group(0) if match else None


def extract_sid_components(text: str) -> Optional[List[str]]:
    sid = extract_sid_from_text(text)
    if sid is None:
        return None
    parts = SID_COMPONENT_PATTERN.findall(sid)
    return parts if len(parts) == 4 else None


def sid_token_codes(text: str) -> Optional[List[int]]:
    components = extract_sid_components(text)
    if components is None:
        return None
    codes = []
    for token in components:
        match = TOKEN_CODE_PATTERN.fullmatch(token)
        if match is None:
            return None
        codes.append(int(match.group(1)))
    return codes


def compute_token_hits(candidate_sid: str, target_sid: str) -> Tuple[int, int]:
    candidate_codes = sid_token_codes(candidate_sid)
    target_codes = sid_token_codes(target_sid)
    if candidate_codes is None or target_codes is None:
        return 0, 4
    hits = sum(int(left == right) for left, right in zip(candidate_codes, target_codes))
    return hits, len(target_codes) - hits


def build_localization_reflection(candidate_sid: str, target_sid: str, num_levels: int = 4) -> LocalizationReflection:
    candidate_codes = sid_token_codes(candidate_sid)
    target_codes = sid_token_codes(target_sid)
    if candidate_codes is None or target_codes is None:
        hits, misses = compute_token_hits(candidate_sid, target_sid)
        return LocalizationReflection(
            position=1,
            label_token="<|loc_1|>",
            token_hits=hits,
            token_misses=misses,
            exact_match=0,
        )

    hits = 0
    position = num_levels + 1
    for index, (left, right) in enumerate(zip(candidate_codes, target_codes), start=1):
        if left == right:
            hits += 1
            continue
        position = index
        break
    misses = len(target_codes) - hits
    return LocalizationReflection(
        position=position,
        label_token=f"<|loc_{position}|>",
        token_hits=hits,
        token_misses=misses,
        exact_match=int(position == num_levels + 1),
    )


def shared_prefix_depth(left: Sequence[str], right: Sequence[str]) -> int:
    depth = 0
    for left_part, right_part in zip(left, right):
        if left_part != right_part:
            break
        depth += 1
    return depth


def build_leaf_category_reflection(
    candidate_meta: Optional[Dict[str, Any]],
    target_meta: Optional[Dict[str, Any]],
) -> CategoryReflection:
    candidate_path = list(candidate_meta.get("category_path", [])) if candidate_meta else []
    target_path = list(target_meta.get("category_path", [])) if target_meta else []
    candidate_leaf = candidate_meta.get("leaf_category", "") if candidate_meta else ""
    target_leaf = target_meta.get("leaf_category", "") if target_meta else ""

    if not candidate_path or not target_path:
        return CategoryReflection(
            label_token="<|leaf_unknown|>",
            is_match=None,
            shared_depth=0,
            target_depth=len(target_path),
            draft_depth=len(candidate_path),
            overlap_ratio=0.0,
            severity_score=1.0,
            severity_bucket="unknown",
            target_leaf=target_leaf,
            draft_leaf=candidate_leaf,
        )

    depth = shared_prefix_depth(candidate_path, target_path)
    overlap_ratio = depth / max(len(candidate_path), len(target_path))
    is_match = int(candidate_leaf == target_leaf)
    if is_match:
        bucket = "exact_match"
    elif depth >= max(len(candidate_path), len(target_path)) - 1:
        bucket = "same_parent"
    elif depth >= 2:
        bucket = "same_branch"
    elif depth >= 1:
        bucket = "same_root"
    else:
        bucket = "different_root"
    return CategoryReflection(
        label_token="<|leaf_match|>" if is_match else "<|leaf_mismatch|>",
        is_match=is_match,
        shared_depth=depth,
        target_depth=len(target_path),
        draft_depth=len(candidate_path),
        overlap_ratio=overlap_ratio,
        severity_score=0.0 if is_match else 1.0 - overlap_ratio,
        severity_bucket=bucket,
        target_leaf=target_leaf,
        draft_leaf=candidate_leaf,
    )


def build_brand_reflection(
    candidate_meta: Optional[Dict[str, Any]],
    target_meta: Optional[Dict[str, Any]],
) -> BrandReflection:
    candidate_brand = candidate_meta.get("brand", "") if candidate_meta else ""
    target_brand = target_meta.get("brand", "") if target_meta else ""
    candidate_source = candidate_meta.get("brand_source", "missing") if candidate_meta else "missing"
    target_source = target_meta.get("brand_source", "missing") if target_meta else "missing"
    candidate_conf = float(candidate_meta.get("brand_confidence", 0.0)) if candidate_meta else 0.0
    target_conf = float(target_meta.get("brand_confidence", 0.0)) if target_meta else 0.0

    if not candidate_brand or not target_brand:
        return BrandReflection(
            label_token="<|brand_unknown|>",
            is_match=None,
            similarity=0.0,
            severity_score=1.0,
            severity_bucket="unknown",
            target_brand=target_brand,
            draft_brand=candidate_brand,
            target_brand_source=target_source,
            draft_brand_source=candidate_source,
            target_brand_confidence=target_conf,
            draft_brand_confidence=candidate_conf,
        )

    is_match = int(candidate_brand == target_brand)
    similarity = SequenceMatcher(None, candidate_brand, target_brand).ratio()
    if is_match:
        bucket = "exact_match"
    elif similarity >= 0.85:
        bucket = "near_match"
    elif similarity >= 0.5:
        bucket = "partial_match"
    else:
        bucket = "hard_mismatch"
    return BrandReflection(
        label_token="<|brand_match|>" if is_match else "<|brand_mismatch|>",
        is_match=is_match,
        similarity=similarity,
        severity_score=0.0 if is_match else 1.0 - similarity,
        severity_bucket=bucket,
        target_brand=target_brand,
        draft_brand=candidate_brand,
        target_brand_source=target_source,
        draft_brand_source=candidate_source,
        target_brand_confidence=target_conf,
        draft_brand_confidence=candidate_conf,
    )


def build_metadata_lookup(metadata_path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    raw = load_json_like(metadata_path)
    sid_lookup: Dict[str, Dict[str, Any]] = {}
    item_lookup: Dict[str, Dict[str, Any]] = {}

    for item_id, record in iter_metadata_records(raw):
        sid = extract_sid_from_text(record.get("sid", ""))
        category_path = parse_category_path(record.get("categories"))
        brand, brand_source, brand_confidence = extract_brand(record)
        enriched = {
            "item_id": item_id,
            "sid": sid,
            "title": normalize_whitespace(str(record.get("title", ""))),
            "description": record.get("description", "") or "",
            "categories": record.get("categories", ""),
            "category_path": category_path,
            "leaf_category": category_path[-1] if category_path else "",
            "brand": brand,
            "brand_source": brand_source,
            "brand_confidence": float(brand_confidence),
        }
        item_lookup[item_id] = enriched
        if sid:
            sid_lookup[sid] = enriched

    return sid_lookup, item_lookup


def build_generation_prompt(user_input: str) -> str:
    return f"""<|im_start|>system
{SYSTEM_MESSAGE}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
<|grc_draft|>
"""


def build_base_sid_prompt(user_input: str) -> str:
    return f"""<|im_start|>system
{SYSTEM_MESSAGE}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
<think>

</think>
"""


def build_reflection_prompt(user_input: str, draft_sid: str) -> str:
    return build_generation_prompt(user_input) + f"{draft_sid}\n<|grc_reflect|>\n"


def build_correction_prompt(user_input: str, draft_sid: str, reflection_sequence: str) -> str:
    return (
        build_generation_prompt(user_input)
        + f"{draft_sid}\n<|grc_reflect|>\n{reflection_sequence}\n<|grc_correct|>\n"
    )


def build_sft_text(
    user_input: str,
    draft_sid: str,
    localization_token: str,
    leaf_token: str,
    brand_token: str,
    target_sid: str,
) -> str:
    reflection_sequence = f"{localization_token}{leaf_token}{brand_token}"
    return (
        f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<|grc_draft|>\n"
        f"{draft_sid}\n"
        "<|grc_reflect|>\n"
        f"{reflection_sequence}\n"
        "<|grc_correct|>\n"
        f"{target_sid}<|im_end|>\n"
    )


def parse_grc_response(response_text: str) -> Dict[str, Any]:
    text = response_text or ""
    if "<|grc_draft|>" in text:
        text = text.split("<|grc_draft|>", 1)[1]

    if "<|grc_reflect|>" not in text or "<|grc_correct|>" not in text:
        return {
            "draft_sid": extract_sid_from_text(text),
            "reflection_tokens": [],
            "corrected_sid": extract_sid_from_text(text),
            "format_valid": False,
        }

    draft_part, rest = text.split("<|grc_reflect|>", 1)
    reflect_part, correct_part = rest.split("<|grc_correct|>", 1)
    reflection_tokens = re.findall(r"<\|[^|]+?\|>", reflect_part)
    return {
        "draft_sid": extract_sid_from_text(draft_part),
        "reflection_tokens": reflection_tokens,
        "corrected_sid": extract_sid_from_text(correct_part),
        "format_valid": True,
    }


def get_sid_special_tokens(num_levels: int = 4, codebook_size: int = 256) -> List[str]:
    tokens = ["<|sid_begin|>", "<|sid_end|>"]
    for level in range(num_levels):
        prefix = f"s_{chr(97 + level)}"
        for index in range(codebook_size):
            tokens.append(f"<{prefix}_{index}>")
    return tokens


def build_sid_constraint_ids(tokenizer, num_levels: int = 4, codebook_size: int = 256) -> Dict[str, Any]:
    sid_begin_id = tokenizer.convert_tokens_to_ids("<|sid_begin|>")
    sid_end_id = tokenizer.convert_tokens_to_ids("<|sid_end|>")
    if sid_begin_id == tokenizer.unk_token_id or sid_end_id == tokenizer.unk_token_id:
        raise ValueError("Tokenizer is missing SID boundary tokens.")

    level_token_ids = []
    for level in range(num_levels):
        prefix = f"s_{chr(97 + level)}"
        level_tokens = [f"<{prefix}_{index}>" for index in range(codebook_size)]
        token_ids = tokenizer.convert_tokens_to_ids(level_tokens)
        if any(token_id == tokenizer.unk_token_id for token_id in token_ids):
            raise ValueError(f"Tokenizer is missing semantic tokens for {prefix}.")
        level_token_ids.append(token_ids)

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos token id.")

    return {
        "sid_begin_id": sid_begin_id,
        "sid_end_id": sid_end_id,
        "level_token_ids": level_token_ids,
        "eos_id": eos_id,
    }


def make_sid_prefix_allowed_tokens_fn(prompt_length: int, sid_constraints: Dict[str, Any]):
    sid_begin_id = sid_constraints["sid_begin_id"]
    sid_end_id = sid_constraints["sid_end_id"]
    level_token_ids = sid_constraints["level_token_ids"]
    eos_id = sid_constraints["eos_id"]

    def prefix_allowed_tokens_fn(batch_id: int, sentence) -> List[int]:
        generated_len = sentence.shape[0] - prompt_length
        if generated_len == 0:
            return [sid_begin_id]
        if 1 <= generated_len <= len(level_token_ids):
            return level_token_ids[generated_len - 1]
        if generated_len == len(level_token_ids) + 1:
            return [sid_end_id]
        return [eos_id]

    return prefix_allowed_tokens_fn


def build_reflection_constraint_ids(tokenizer, num_levels: int = 4) -> Dict[str, Any]:
    loc_ids = tokenizer.convert_tokens_to_ids(get_loc_tokens(num_levels=num_levels))
    leaf_ids = tokenizer.convert_tokens_to_ids(["<|leaf_match|>", "<|leaf_mismatch|>", "<|leaf_unknown|>"])
    brand_ids = tokenizer.convert_tokens_to_ids(["<|brand_match|>", "<|brand_mismatch|>", "<|brand_unknown|>"])
    if any(token_id == tokenizer.unk_token_id for token_id in loc_ids + leaf_ids + brand_ids):
        raise ValueError("Tokenizer is missing reflection special tokens.")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos token id.")
    return {
        "loc_ids": loc_ids,
        "leaf_ids": leaf_ids,
        "brand_ids": brand_ids,
        "eos_id": eos_id,
    }


def make_reflection_prefix_allowed_tokens_fn(prompt_length: int, reflection_constraints: Dict[str, Any]):
    loc_ids = reflection_constraints["loc_ids"]
    leaf_ids = reflection_constraints["leaf_ids"]
    brand_ids = reflection_constraints["brand_ids"]
    eos_id = reflection_constraints["eos_id"]

    def prefix_allowed_tokens_fn(batch_id: int, sentence) -> List[int]:
        generated_len = sentence.shape[0] - prompt_length
        if generated_len == 0:
            return loc_ids
        if generated_len == 1:
            return leaf_ids
        if generated_len == 2:
            return brand_ids
        return [eos_id]

    return prefix_allowed_tokens_fn


def maybe_to_serializable(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value
