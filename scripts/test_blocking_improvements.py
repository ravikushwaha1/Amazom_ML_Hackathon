"""
Evaluation of Targeted Blocking Engine Enhancements
Tests Rule 1 (DBA & Domain stripping), Rule 2 (Prefix/Honorific handling), Rule 3 (Sorted word pairs),
and Source-Balanced Candidate Capping (25 S2 + 25 S3).
Reports exact Candidate Recall, Candidate Growth, and Macro F0.5.
"""

import sys
import os
import csv
import re
from collections import defaultdict
from typing import List, Dict, Set, Tuple
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.normalizer import (
    clean_base_text,
    normalize_business_name,
    normalize_business_address,
    normalize_country
)
from src.features.feature_extractor import extract_pair_features, get_char_ngrams
from src.model.matcher import EntityMatcher
from src.evaluation.metrics import compute_macro_f05

sys.stdout.reconfigure(encoding='utf-8')

GENERIC_WORDS = {
    'the', 'and', 'or', 'of', 'for', 'in', 'on', 'at', 'to', 'a', 'an',
    'inc', 'corp', 'corporation', 'llc', 'llp', 'ltd', 'limited', 'pvt',
    'company', 'co', 'services', 'service', 'center', 'group', 'enterprises',
    'enterprise', 'solutions', 'associates', 'consulting', 'international',
    'holdings', 'industries', 'ventures', 'management', 'global', 'tech',
    'technologies', 'technology', 'system', 'systems', 'trading', 'club',
    'sarl', 'sci', 'sas', 'sa', 'eurl', 'france', 'private'
}

HONORIFICS = {'smt', 'shri', 'shree', 'dr', 'mr', 'mrs', 'ms', 'm/s', 'ms.'}
DOMAIN_RE = re.compile(r'\.(com|org|net|in|co|io|fr|gov|edu|biz|info)\b', re.IGNORECASE)
DBA_RE = re.compile(r'\b(?:d/?b/?a|t/?a|a/?k/?a)\b', re.IGNORECASE)

def get_enhanced_blocking_keys(cl_name: str, core: str, cl_addr: str, country: str, raw_name: str = "") -> List[str]:
    keys = []
    pfx = country + "|"
    
    # Domain stripping
    clean_n = DOMAIN_RE.sub('', cl_name).strip()
    clean_c = DOMAIN_RE.sub('', core).strip()
    
    # 1. Exact Core Name
    if len(clean_c) >= 3:
        keys.append(pfx + "c:" + clean_c)
        
    # 2. Words processing with honorifics handling
    raw_words = [w for w in clean_c.split() if w not in GENERIC_WORDS]
    # Filter out leading honorific if followed by other words
    if len(raw_words) >= 2 and raw_words[0] in HONORIFICS:
        words = raw_words[1:]
    else:
        words = raw_words
        
    if len(words) >= 2:
        keys.append(pfx + "b:" + words[0] + "_" + words[1])
        # Sorted bi-gram for word order invariance
        if words[0] != words[1]:
            s_pair = sorted([words[0], words[1]])
            keys.append(pfx + "sb:" + s_pair[0] + "_" + s_pair[1])
    elif len(words) == 1 and len(words[0]) >= 4 and words[0] not in GENERIC_WORDS and words[0] not in HONORIFICS:
        keys.append(pfx + "w:" + words[0])
        
    # Also if raw name has DBA, add the trade name keys
    if raw_name and DBA_RE.search(raw_name):
        parts = DBA_RE.split(raw_name)
        for part in parts:
            p_cl, p_core, _ = normalize_business_name(part)
            p_clean_c = DOMAIN_RE.sub('', p_core).strip()
            if len(p_clean_c) >= 3 and p_clean_c != clean_c:
                keys.append(pfx + "c:" + p_clean_c)
            p_words = [w for w in p_clean_c.split() if w not in GENERIC_WORDS and w not in HONORIFICS]
            if len(p_words) >= 2:
                keys.append(pfx + "b:" + p_words[0] + "_" + p_words[1])
                s_p = sorted([p_words[0], p_words[1]])
                keys.append(pfx + "sb:" + s_p[0] + "_" + s_p[1])
                
    # 3. Squashed name without domain
    alnum = "".join(c for c in clean_n if c.isalnum())
    if len(alnum) >= 5:
        keys.append(pfx + "s:" + alnum)
        
    # 4. Address Key
    tokens = cl_addr.split()
    nums = [w for w in tokens if any(c.isdigit() for c in w)]
    alphas = [w for w in tokens if w.isalpha() and len(w) >= 4 and w not in {'street', 'road', 'avenue', 'boulevard', 'lane', 'rue'}]
    if nums and alphas:
        keys.append(pfx + "na:" + nums[0] + "_" + alphas[0])
        
    return list(set(keys))

def evaluate_blocking():
    s1_start = 20000
    n_s1 = 2500
    train_gt = os.path.join(PROJECT_ROOT, "dataset", "train", "train_ground_truth.tsv")
    train_s1 = os.path.join(PROJECT_ROOT, "dataset", "train", "train_source1.tsv")
    train_s2 = os.path.join(PROJECT_ROOT, "dataset", "train", "train_source2.tsv")
    train_s3 = os.path.join(PROJECT_ROOT, "dataset", "train", "train_source3.tsv")
    
    gt_map = {}
    needed_s1 = set()
    s1_order = []
    with open(train_gt, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for i, row in enumerate(reader):
            if i < s1_start: continue
            if i >= s1_start + n_s1: break
            sid = row[0].strip()
            matches = set(m.strip() for m in row[1].split(",") if m.strip()) if len(row) > 1 and row[1].strip() else set()
            gt_map[sid] = matches
            needed_s1.add(sid)
            s1_order.append(sid)
            
    total_true = sum(len(m) for m in gt_map.values())
    print(f"Total True Matches: {total_true:,}")
    
    s1_records = {}
    s1_index = defaultdict(list)
    chunk_countries = set()
    with open(train_s1, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            eid = row[0].strip()
            if eid not in needed_s1: continue
            raw_n = row[1]
            cl_name, core, suff = normalize_business_name(raw_n)
            cl_addr = normalize_business_address(row[2])
            country = normalize_country(row[3])
            chunk_countries.add(country)
            
            addr_toks = set(cl_addr.split())
            nums = [w for w in addr_toks if any(ch.isdigit() for ch in w)]
            s1_records[eid] = {
                "entity_id": eid, "clean_name": cl_name, "core_name": core, "legal_suffix": suff,
                "clean_addr": cl_addr, "country": country, "squash": "".join(c for c in cl_name if c.isalnum()),
                "name_tokens": set(cl_name.split()), "name_3grams": get_char_ngrams(core),
                "addr_tokens": addr_toks, "addr_3grams": get_char_ngrams(cl_addr),
                "house_num": nums[0] if nums else ""
            }
            keys = get_enhanced_blocking_keys(cl_name, core, cl_addr, country, raw_name=raw_n)
            for k in keys:
                s1_index[k].append(eid)
            if len(s1_records) >= len(needed_s1): break
            
    print(f"Indexed {len(s1_records):,} S1 entities across {len(s1_index):,} enhanced keys.")
    
    # Stream with balanced cap (25 S2 + 25 S3)
    candidates = defaultdict(list)
    target_records = {}
    
    for src_path, label, cap in [(train_s2, "Source 2", 25), (train_s3, "Source 3", 25)]:
        scanned = 0
        hits = 0
        with open(src_path, "r", encoding="utf-8") as f_src:
            reader = csv.reader(f_src, delimiter="\t")
            next(reader)
            for row in reader:
                scanned += 1
                if len(row) < 4: continue
                country = row[3].strip()
                if country not in chunk_countries: continue
                raw_n = row[1]
                raw_a = row[2]
                cl_name, core, suff = normalize_business_name(raw_n)
                cl_addr_fast = clean_base_text(raw_a)
                keys = get_enhanced_blocking_keys(cl_name, core, cl_addr_fast, country, raw_name=raw_n)
                matched_s1 = set()
                for k in keys:
                    if k in s1_index:
                        s1_list = s1_index[k]
                        if len(s1_list) <= 500:
                            matched_s1.update(s1_list)
                if matched_s1:
                    tid = row[0].strip()
                    prefix = "S2" if label == "Source 2" else "S3"
                    for sid in matched_s1:
                        cur_src_cands = sum(1 for x in candidates[sid] if x.startswith(prefix))
                        if cur_src_cands < cap and tid not in candidates[sid]:
                            candidates[sid].append(tid)
                            hits += 1
                            if tid not in target_records:
                                target_records[tid] = (raw_n, raw_a, country)
        print(f"  Streamed {label}: {hits:,} hits added.")
        
    recovered_true = sum(len(gt_map[s] & set(candidates[s])) for s in s1_order)
    cand_counts = [len(candidates[s]) for s in s1_order]
    avg_cands = np.mean(cand_counts)
    rec_pct = (recovered_true / total_true * 100)
    print(f"\nENHANCED CANDIDATE BLOCKING METRICS:")
    print(f"  Candidate Recall: {recovered_true:,} / {total_true:,} ({rec_pct:.2f}%)  [Previous: 52.50% -> +{rec_pct - 52.50:.2f}% GAIN]")
    print(f"  Average candidates / S1: {avg_cands:.2f} [Previous: 16.89]")
    
    # Preprocess targets and score
    prep_targets = {}
    for tid, (r_name, r_addr, cntry) in target_records.items():
        c_name, c_core, c_suff = normalize_business_name(r_name)
        c_addr = normalize_business_address(r_addr)
        a_toks = set(c_addr.split())
        h_nums = [w for w in a_toks if any(ch.isdigit() for ch in w)]
        prep_targets[tid] = {
            "entity_id": tid, "clean_name": c_name, "core_name": c_core, "legal_suffix": c_suff,
            "clean_addr": c_addr, "country": cntry, "squash": "".join(c for c in c_name if c.isalnum()),
            "name_tokens": set(c_name.split()), "name_3grams": get_char_ngrams(c_core),
            "addr_tokens": a_toks, "addr_3grams": get_char_ngrams(c_addr),
            "house_num": h_nums[0] if h_nums else ""
        }
    del target_records
    
    matcher = EntityMatcher()
    scored_pairs = []
    batch_feats = []
    batch_pairs = []
    for sid in s1_order:
        s1_d = s1_records[sid]
        for tid in candidates.get(sid, []):
            if tid in prep_targets:
                feats = extract_pair_features(s1_d, prep_targets[tid])
                batch_feats.append(feats)
                batch_pairs.append((sid, tid))
                if len(batch_feats) >= 50000:
                    probs = matcher.predict_proba(np.array(batch_feats, dtype=np.float32))
                    for (s, t), p in zip(batch_pairs, probs):
                        scored_pairs.append((s, t, float(p)))
                    batch_feats = []
                    batch_pairs = []
    if batch_feats:
        probs = matcher.predict_proba(np.array(batch_feats, dtype=np.float32))
        for (s, t), p in zip(batch_pairs, probs):
            scored_pairs.append((s, t, float(p)))
            
    print(f"Scored {len(scored_pairs):,} candidate pairs.")
    
    # Evaluate across thresholds
    print("\n--- THRESHOLD PERFORMANCE WITH ENHANCED BLOCKING ---")
    print(f"{'Threshold':<10} | {'Macro F0.5':<12} | {'Precision':<11} | {'Recall':<10} | {'Singleton Acc':<14} | {'Matches':<10}")
    print("-" * 75)
    for T in [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]:
        p_dict = defaultdict(set)
        total_p = 0
        for sid, tid, p in scored_pairs:
            if p >= T:
                p_dict[sid].add(tid)
                total_p += 1
        fp_dict = {sid: p_dict.get(sid, set()) for sid in s1_order}
        m = compute_macro_f05(gt_map, fp_dict)
        print(f"{T:<10.2f} | {m['macro_f05']:<12.5f} | {m['mean_precision']:<11.5f} | {m['mean_recall']:<10.5f} | {m['singleton_accuracy']:<14.5f} | {total_p:<10,}")

if __name__ == "__main__":
    evaluate_blocking()
