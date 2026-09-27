"""
Real End-to-End Held-Out Validation Simulation
Strictly simulates the test inference pipeline on unseen training S1 entities.
No ground-truth leakage or candidate injection.
Evaluates production blocking + native LightGBM matcher.
"""

import sys
import os
import csv
import time
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple
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
from src.prediction.predict_test_chunked import get_blocking_keys

sys.stdout.reconfigure(encoding='utf-8')

def run_real_validation(
    s1_start: int = 20000,
    n_s1: int = 2500,
    max_candidates_per_s1: int = 25,
    threshold: float = 0.85
):
    print("="*80)
    print(f"PHASE 3: REAL HELD-OUT VALIDATION (Entities {s1_start:,} to {s1_start + n_s1:,})")
    print("="*80)

    train_s1 = os.path.join(PROJECT_ROOT, "dataset", "train", "train_source1.tsv")
    train_s2 = os.path.join(PROJECT_ROOT, "dataset", "train", "train_source2.tsv")
    train_s3 = os.path.join(PROJECT_ROOT, "dataset", "train", "train_source3.tsv")
    train_gt = os.path.join(PROJECT_ROOT, "dataset", "train", "train_ground_truth.tsv")

    # 1. Load Ground Truth for evaluation slice ONLY for evaluation
    gt_map: Dict[str, Set[str]] = {}
    needed_s1 = set()
    s1_order = []

    with open(train_gt, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for i, row in enumerate(reader):
            if i < s1_start:
                continue
            if i >= s1_start + n_s1:
                break
            s1_id = row[0].strip()
            matches = set(m.strip() for m in row[1].split(",") if m.strip()) if len(row) > 1 and row[1].strip() else set()
            gt_map[s1_id] = matches
            needed_s1.add(s1_id)
            s1_order.append(s1_id)

    print(f"Held-out S1 entities: {len(needed_s1):,}")
    singletons = sum(1 for m in gt_map.values() if len(m) == 0)
    total_true_matches = sum(len(m) for m in gt_map.values())
    print(f"Ground Truth: {singletons:,} singletons ({singletons/len(gt_map)*100:.2f}%), {total_true_matches:,} true match pairs.")

    # 2. Load and index S1 records
    s1_records = {}
    s1_index = defaultdict(list)
    chunk_countries = set()

    with open(train_s1, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            eid = row[0].strip()
            if eid not in needed_s1:
                continue

            cl_name, core, suff = normalize_business_name(row[1])
            cl_addr = normalize_business_address(row[2])
            country = normalize_country(row[3])
            chunk_countries.add(country)

            addr_toks = set(cl_addr.split())
            nums = [w for w in addr_toks if any(ch.isdigit() for ch in w)]

            s1_records[eid] = {
                "entity_id": eid,
                "clean_name": cl_name,
                "core_name": core,
                "legal_suffix": suff,
                "clean_addr": cl_addr,
                "country": country,
                "squash": "".join(c for c in cl_name if c.isalnum()),
                "name_tokens": set(cl_name.split()),
                "name_3grams": get_char_ngrams(core),
                "addr_tokens": addr_toks,
                "addr_3grams": get_char_ngrams(cl_addr),
                "house_num": nums[0] if nums else ""
            }

            keys = get_blocking_keys(cl_name, core, cl_addr, country)
            for k in keys:
                s1_index[k].append(eid)

            if len(s1_records) >= len(needed_s1):
                break

    print(f"Indexed {len(s1_records):,} S1 entities across {len(s1_index):,} blocking keys.")

    # 3. Stream Source 2 and Source 3 (Exact test pipeline simulation)
    candidates = defaultdict(list)
    target_records = {}

    for src_path, label in [(train_s2, "Source 2"), (train_s3, "Source 3")]:
        t0 = time.time()
        scanned = 0
        hits = 0
        with open(src_path, "r", encoding="utf-8") as f_src:
            reader = csv.reader(f_src, delimiter="\t")
            next(reader)
            for row in reader:
                scanned += 1
                if len(row) < 4:
                    continue
                country = row[3].strip()
                if country not in chunk_countries:
                    continue

                raw_name = row[1]
                raw_addr = row[2]

                cl_name, core, suff = normalize_business_name(raw_name)
                cl_addr_fast = clean_base_text(raw_addr)
                keys = get_blocking_keys(cl_name, core, cl_addr_fast, country)

                matched_s1 = set()
                for k in keys:
                    if k in s1_index:
                        s1_list = s1_index[k]
                        if len(s1_list) <= 500:
                            matched_s1.update(s1_list)

                if matched_s1:
                    tid = row[0].strip()
                    eligible = [sid for sid in matched_s1 if len(candidates[sid]) < max_candidates_per_s1 and tid not in candidates[sid]]
                    if eligible:
                        for sid in eligible:
                            candidates[sid].append(tid)
                            hits += 1
                        if tid not in target_records:
                            target_records[tid] = (raw_name, raw_addr, country)

        print(f"  Streamed {label}: {scanned:,} rows in {time.time() - t0:.2f}s, candidate hits: {hits:,}")

    # Calculate Candidate Blocking Recall
    recovered_true = 0
    all_true_pairs = 0
    missed_pairs_info = []

    for sid in s1_order:
        true_set = gt_map.get(sid, set())
        cands_set = set(candidates.get(sid, []))
        all_true_pairs += len(true_set)
        recovered_true += len(true_set & cands_set)
        missed = true_set - cands_set
        if missed:
            for m_tid in missed:
                missed_pairs_info.append((sid, m_tid))

    candidate_recall = (recovered_true / all_true_pairs * 100) if all_true_pairs > 0 else 0.0
    cand_counts = [len(candidates.get(sid, [])) for sid in s1_order]
    avg_cands = np.mean(cand_counts)
    max_cands = np.max(cand_counts) if cand_counts else 0

    print("\n--- CANDIDATE BLOCKING METRICS ---")
    print(f"Total True Matches: {all_true_pairs:,}")
    print(f"Candidate Blocking Recall: {candidate_recall:.2f}% ({recovered_true:,} / {all_true_pairs:,})")
    print(f"Average candidates / S1: {avg_cands:.2f}")
    print(f"Max candidates / S1: {max_cands}")
    print(f"Total candidate pairs generated: {sum(cand_counts):,}")

    # 4. Preprocess Target Records
    prep_targets = {}
    for tid, (r_name, r_addr, cntry) in target_records.items():
        c_name, c_core, c_suff = normalize_business_name(r_name)
        c_addr = normalize_business_address(r_addr)
        a_toks = set(c_addr.split())
        h_nums = [w for w in a_toks if any(ch.isdigit() for ch in w)]
        prep_targets[tid] = {
            "entity_id": tid,
            "clean_name": c_name,
            "core_name": c_core,
            "legal_suffix": c_suff,
            "clean_addr": c_addr,
            "country": cntry,
            "squash": "".join(c for c in c_name if c.isalnum()),
            "name_tokens": set(c_name.split()),
            "name_3grams": get_char_ngrams(c_core),
            "addr_tokens": a_toks,
            "addr_3grams": get_char_ngrams(c_addr),
            "house_num": h_nums[0] if h_nums else ""
        }
    del target_records

    # 5. Feature Extraction and LightGBM Scoring
    matcher = EntityMatcher()
    print(f"\nScoring pairs with native LightGBM matcher...")
    scored_pairs = [] # (sid, tid, prob)

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

    # 6. Evaluate threshold = 0.85 (and record detailed metrics)
    preds_085 = defaultdict(set)
    for sid, tid, p in scored_pairs:
        if p >= threshold:
            preds_085[sid].add(tid)

    full_preds_085 = {sid: preds_085.get(sid, set()) for sid in s1_order}
    metrics_085 = compute_macro_f05(gt_map, full_preds_085)

    pred_counts = [len(preds_085.get(sid, set())) for sid in s1_order]
    avg_preds = np.mean(pred_counts)
    max_preds = np.max(pred_counts) if pred_counts else 0

    # Calculate false positives per S1
    total_fp = 0
    for sid in s1_order:
        p_set = preds_085.get(sid, set())
        t_set = gt_map.get(sid, set())
        fp = len(p_set - t_set)
        total_fp += fp
    avg_fp = total_fp / len(s1_order)

    print("\n" + "="*80)
    print(f"PHASE 3 RESULTS AT THRESHOLD {threshold:.2f}:")
    print(f"  - candidate recall:       {candidate_recall:.2f}%")
    print(f"  - precision (mean):       {metrics_085['mean_precision']:.5f}")
    print(f"  - recall (mean):          {metrics_085['mean_recall']:.5f}")
    print(f"  - Macro F0.5:             {metrics_085['macro_f05']:.5f}")
    print(f"  - singleton accuracy:     {metrics_085['singleton_accuracy']:.5f}")
    print(f"  - false positives per S1: {avg_fp:.4f}")
    print(f"  - average candidates/S1:  {avg_cands:.2f}")
    print(f"  - average predictions/S1: {avg_preds:.2f}")
    print(f"  - maximum predictions/S1: {max_preds}")
    print("="*80)

    # 7. Sweep thresholds for Phase 5
    sweep_thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]
    print("\n--- PHASE 5: THRESHOLD SWEEP ---")
    print(f"{'Threshold':<10} | {'Macro F0.5':<12} | {'Precision':<11} | {'Recall':<10} | {'Singleton Acc':<14} | {'Predictions':<12}")
    print("-" * 75)
    for T in sweep_thresholds:
        p_dict = defaultdict(set)
        total_p = 0
        for sid, tid, p in scored_pairs:
            if p >= T:
                p_dict[sid].add(tid)
                total_p += 1
        fp_dict = {sid: p_dict.get(sid, set()) for sid in s1_order}
        m = compute_macro_f05(gt_map, fp_dict)
        print(f"{T:<10.2f} | {m['macro_f05']:<12.5f} | {m['mean_precision']:<11.5f} | {m['mean_recall']:<10.5f} | {m['singleton_accuracy']:<14.5f} | {total_p:<12,}")

    return {
        "candidate_recall": candidate_recall,
        "missed_pairs_info": missed_pairs_info[:100],
        "metrics_085": metrics_085,
        "scored_pairs": scored_pairs
    }

if __name__ == "__main__":
    run_real_validation(s1_start=20000, n_s1=2500)
