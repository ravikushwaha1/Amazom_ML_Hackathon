"""
Entity Resolution Challenge: Feature Extraction Engine
Author: Lead ML Engineer
Purpose:
    Extracts high-signal, fast similarity features for candidate entity pairs
    using vectorized token, character n-gram, and numerical distance metrics.
"""

import math
from typing import Dict, List, Set, Any
import numpy as np


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from a string."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Computes Jaccard similarity between two token sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def dice_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Computes Dice similarity coefficient between two n-gram sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    total = len(set_a) + len(set_b)
    return (2.0 * intersection) / total if total > 0 else 0.0


def overlap_coefficient(set_a: Set[str], set_b: Set[str]) -> float:
    """Computes Szymkiewicz-Simpson overlap coefficient."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    min_len = min(len(set_a), len(set_b))
    return intersection / min_len if min_len > 0 else 0.0


def extract_pair_features(s1: Dict[str, Any], target: Dict[str, Any], shared_keys_count: int = 1) -> List[float]:
    """
    Computes a fixed-length numerical feature vector for an (S1, Target) candidate pair.
    
    Feature List (16 features):
    0: exact_clean_name (0 or 1)
    1: exact_core_name (0 or 1)
    2: exact_squash_name (0 or 1)
    3: name_jaccard (0.0 to 1.0)
    4: name_overlap (0.0 to 1.0)
    5: name_3gram_dice (0.0 to 1.0)
    6: name_len_diff (int >= 0)
    7: name_len_ratio (0.0 to 1.0)
    8: legal_suffix_match (1.0 = match, 0.5 = one missing, 0.0 = conflict)
    9: exact_clean_addr (0 or 1)
    10: addr_missing (1 if target address is empty, else 0)
    11: addr_jaccard (0.0 to 1.0)
    12: addr_token_overlap (count of shared tokens)
    13: addr_3gram_dice (0.0 to 1.0)
    14: house_num_match (1 = match, -1 = conflict, 0 = missing)
    15: shared_keys_count (integer >= 1)
    """
    # 1. Name Features
    s1_name = s1.get("clean_name", "")
    t_name = target.get("clean_name", "")
    s1_core = s1.get("core_name", "")
    t_core = target.get("core_name", "")
    s1_squash = s1.get("squash", "")
    t_squash = target.get("squash", "")
    
    f_exact_clean = 1.0 if (s1_name and s1_name == t_name) else 0.0
    f_exact_core = 1.0 if (s1_core and s1_core == t_core) else 0.0
    f_exact_squash = 1.0 if (s1_squash and s1_squash == t_squash) else 0.0
    
    s1_tokens = s1.get("name_tokens", set(s1_name.split()))
    t_tokens = target.get("name_tokens", set(t_name.split()))
    
    f_name_jaccard = jaccard_similarity(s1_tokens, t_tokens)
    f_name_overlap = overlap_coefficient(s1_tokens, t_tokens)
    
    s1_ngrams = s1.get("name_3grams", get_char_ngrams(s1_core))
    t_ngrams = target.get("name_3grams", get_char_ngrams(t_core))
    f_name_dice = dice_similarity(s1_ngrams, t_ngrams)
    
    len1, len2 = len(s1_core), len(t_core)
    f_len_diff = float(abs(len1 - len2))
    f_len_ratio = float(min(len1, len2)) / float(max(len1, len2)) if max(len1, len2) > 0 else 0.0
    
    s1_suff = s1.get("legal_suffix", "")
    t_suff = target.get("legal_suffix", "")
    if s1_suff and t_suff:
        f_suff_match = 1.0 if s1_suff == t_suff else 0.0
    elif not s1_suff and not t_suff:
        f_suff_match = 1.0
    else:
        f_suff_match = 0.5
        
    # 2. Address Features
    s1_addr = s1.get("clean_addr", "")
    t_addr = target.get("clean_addr", "")
    
    f_addr_missing = 1.0 if not t_addr else 0.0
    f_exact_addr = 1.0 if (s1_addr and s1_addr == t_addr) else 0.0
    
    if s1_addr and t_addr:
        s1_addr_toks = s1.get("addr_tokens", set(s1_addr.split()))
        t_addr_toks = target.get("addr_tokens", set(t_addr.split()))
        f_addr_jaccard = jaccard_similarity(s1_addr_toks, t_addr_toks)
        f_addr_overlap = float(len(s1_addr_toks & t_addr_toks))
        f_addr_dice = dice_similarity(
            s1.get("addr_3grams", get_char_ngrams(s1_addr)),
            target.get("addr_3grams", get_char_ngrams(t_addr))
        )
        
        # House number check
        s1_num = s1.get("house_num", "")
        t_num = target.get("house_num", "")
        if s1_num and t_num:
            f_house_match = 1.0 if s1_num == t_num else -1.0
        else:
            f_house_match = 0.0
    else:
        f_addr_jaccard = 0.0
        f_addr_overlap = 0.0
        f_addr_dice = 0.0
        f_house_match = 0.0
        
    f_shared_keys = float(shared_keys_count)
    
    return [
        f_exact_clean,
        f_exact_core,
        f_exact_squash,
        f_name_jaccard,
        f_name_overlap,
        f_name_dice,
        f_len_diff,
        f_len_ratio,
        f_suff_match,
        f_exact_addr,
        f_addr_missing,
        f_addr_jaccard,
        f_addr_overlap,
        f_addr_dice,
        f_house_match,
        f_shared_keys
    ]

FEATURE_NAMES = [
    "exact_clean_name",
    "exact_core_name",
    "exact_squash_name",
    "name_jaccard",
    "name_overlap",
    "name_3gram_dice",
    "name_len_diff",
    "name_len_ratio",
    "legal_suffix_match",
    "exact_clean_addr",
    "addr_missing",
    "addr_jaccard",
    "addr_token_overlap",
    "addr_3gram_dice",
    "house_num_match",
    "shared_keys_count"
]
