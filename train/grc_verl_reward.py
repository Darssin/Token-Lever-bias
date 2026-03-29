#!/usr/bin/env python3

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from grc_pipeline_utils import (  # noqa: E402
    build_brand_reflection,
    build_leaf_category_reflection,
    build_localization_reflection,
    build_metadata_lookup,
    compute_token_hits,
    parse_grc_response,
    sid_token_codes,
)


LEAF_LABELS = {"<|leaf_match|>", "<|leaf_mismatch|>", "<|leaf_unknown|>"}
BRAND_LABELS = {"<|brand_match|>", "<|brand_mismatch|>", "<|brand_unknown|>"}


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


@lru_cache(maxsize=1)
def _get_sid_lookup():
    metadata_path = os.getenv("GRC_METADATA_PATH")
    if not metadata_path:
        raise ValueError("GRC_METADATA_PATH must be set for verl reward computation.")
    sid_lookup, _ = build_metadata_lookup(Path(metadata_path))
    return sid_lookup


def _safe_json_loads(extra_info: Any) -> Dict[str, Any]:
    if isinstance(extra_info, dict):
        return extra_info
    if isinstance(extra_info, str) and extra_info.strip():
        try:
            return json.loads(extra_info)
        except json.JSONDecodeError:
            return {}
    return {}


def _extract_reflection_labels(tokens) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    loc_token = None
    leaf_token = None
    brand_token = None
    for token in tokens:
        if loc_token is None and token.startswith("<|loc_"):
            loc_token = token
        elif leaf_token is None and token in LEAF_LABELS:
            leaf_token = token
        elif brand_token is None and token in BRAND_LABELS:
            brand_token = token
    return loc_token, leaf_token, brand_token


def _loc_token_to_position(token: Optional[str]) -> int:
    if not token:
        return 1
    try:
        return int(token.removeprefix("<|loc_").removesuffix("|>"))
    except ValueError:
        return 1


def _semantic_correction_score(
    gt_is_match: Optional[int],
    corrected_value: str,
    target_value: str,
) -> Optional[float]:
    if gt_is_match is None or not target_value:
        return None
    if gt_is_match == 1:
        return 0.0
    return float(corrected_value == target_value)


def _compute_reward(ground_truth: str, solution_str: str) -> float:
    sid_lookup = _get_sid_lookup()
    target_meta = sid_lookup.get(ground_truth)
    if target_meta is None:
        return 0.0

    parsed = parse_grc_response(solution_str)
    draft_sid = parsed.get("draft_sid") or ""
    corrected_sid = parsed.get("corrected_sid") or ""
    reflection_tokens = parsed.get("reflection_tokens", [])

    if not draft_sid or not corrected_sid:
        return 0.0

    loc_pred, leaf_pred, brand_pred = _extract_reflection_labels(reflection_tokens)
    draft_meta = sid_lookup.get(draft_sid)
    corrected_meta = sid_lookup.get(corrected_sid)

    loc_gt = build_localization_reflection(draft_sid, ground_truth)
    leaf_gt = build_leaf_category_reflection(draft_meta, target_meta)
    brand_gt = build_brand_reflection(draft_meta, target_meta)

    hits_draft, _ = compute_token_hits(draft_sid, ground_truth)
    hits_corrected, _ = compute_token_hits(corrected_sid, ground_truth)

    beta_cor = _get_env_float("GRC_BETA_COR", 2.2)
    beta_last = _get_env_float("GRC_BETA_LAST", 2.0)
    beta_loc = _get_env_float("GRC_BETA_LOC", 1.0)
    beta_sem = _get_env_float("GRC_BETA_SEM", 0.8)

    reward_task = hits_draft + beta_last * hits_corrected

    reward_loc_label = float(loc_pred == loc_gt.label_token)
    pred_start = _loc_token_to_position(loc_pred)
    draft_codes = sid_token_codes(draft_sid) or []
    corrected_codes = sid_token_codes(corrected_sid) or []
    target_codes = sid_token_codes(ground_truth) or []

    corrected_positions = {
        idx + 1
        for idx, (draft_code, corrected_code, target_code) in enumerate(
            zip(draft_codes, corrected_codes, target_codes)
        )
        if draft_code != target_code and corrected_code == target_code
    }
    pred_region = set(range(pred_start, len(target_codes) + 1)) if target_codes else {1, 2, 3, 4}
    reward_loc_cor = len(pred_region & corrected_positions) / (len(pred_region) + 1e-6)
    reward_loc = reward_loc_label + reward_loc_cor

    sem_label_scores = []
    if leaf_gt.is_match is not None:
        sem_label_scores.append(float(leaf_pred == leaf_gt.label_token))
    if brand_gt.is_match is not None:
        sem_label_scores.append(float(brand_pred == brand_gt.label_token))
    reward_sem_label = sum(sem_label_scores) / len(sem_label_scores) if sem_label_scores else 0.0

    leaf_corrected = corrected_meta.get("leaf_category", "") if corrected_meta else ""
    brand_corrected = corrected_meta.get("brand", "") if corrected_meta else ""
    sem_correction_scores = []
    leaf_correction = _semantic_correction_score(
        gt_is_match=leaf_gt.is_match,
        corrected_value=leaf_corrected,
        target_value=target_meta.get("leaf_category", ""),
    )
    if leaf_correction is not None:
        sem_correction_scores.append(leaf_correction)
    brand_correction = _semantic_correction_score(
        gt_is_match=brand_gt.is_match,
        corrected_value=brand_corrected,
        target_value=target_meta.get("brand", ""),
    )
    if brand_correction is not None:
        sem_correction_scores.append(brand_correction)
    reward_sem_cor = sum(sem_correction_scores) / len(sem_correction_scores) if sem_correction_scores else 0.0
    reward_sem = reward_sem_label + reward_sem_cor

    reward_delta = max(0.0, float(hits_corrected - hits_draft))

    return float(reward_task + beta_cor * (beta_loc * reward_loc + beta_sem * reward_sem + reward_delta))


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Any = None) -> float:
    _safe_json_loads(extra_info)
    try:
        return _compute_reward(ground_truth=ground_truth, solution_str=solution_str)
    except Exception:
        return 0.0
