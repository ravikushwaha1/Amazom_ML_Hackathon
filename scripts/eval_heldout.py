"""
Held-out Validation & Threshold Optimization Benchmark
Evaluates S1 entities 20,000 to 25,000 (completely unseen during training).
Measures exact Macro F0.5, Precision, Recall, and Singleton Accuracy across thresholds.
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

def run_validation(
    s1_start: int = 20000,
    n_s1: int = 5000,
    max_candidates_per_s1: int = 25
):
    print("="*70)
    print(f"RUNNING HELD-OUT VALIDATION (S1 entities {s1_start:,} to {s1_start + n_s1:,})")
    print("="*70)
    
    train_s1 = os.path.join(PROJECT_ROOT, "dataset", "train", "train_source1.tsv")
    train_s2 = os.path.join(PROJECT_ROOT, "dataset", "train", "train_source2.tsv")
    train_s3 = os.path.join(PROJECT_ROOT, "dataset", "train", "train_source3.tsv")
    train_gt = os.path.join(PROJECT_ROOT, "dataset", "train", "train_ground_truth.tsv")
    
    # 1. Load Ground Truth for evaluation slice
    gt_map = {}
    needed_s1 = set()
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
            
    print(f"Loaded ground truth for {len(gt_map):,} entities.")
    singletons = sum(1 for m in gt_map.values() if len(m) == 0)
    total_true_matches = sum(len(m) for m in gt_map.values())
    print(f"Singletons: {singletons:,} ({singletons/len(gt_map)*100:.2f}%), Total True Matches: {total_true_matches:,}")
    
    # 2. Load matching S1 entities from train_source1
    s1_records = {}
    s1_index = defaultdict(list)
    s1_order = []
    countries = set()
    
    with open(train_s1, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            eid = row[0].strip()
            if eid not in needed_s1:
                continue
            s1_order.append(eid)
            
            cl_name, core, suff = normalize_business_name(row[1])
            cl_addr = normalize_business_address(row[2])
            country = normalize_country(row[3])
            countries.add(country)
            
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
                
    print(f"Indexed {len(s1_order):,} S1 entities across {len(s1_index):,} blocking keys.")
    
    # 3. Stream Source 2 and Source 3 for candidates
    candidates = defaultdict(list)
    target_records = {}
    
    for src_path, label in [(train_s2, "Source 2"), (train_s3, "Source 3")]:
        t0 = time.time()
        hits = 0
        with open(src_path, "r", encoding="utf-8") as f_src:
            reader = csv.reader(f_src, delimiter="\t")
            next(reader)
            for row in reader:
                if len(row) < 4:
                    continue
                cntry = row[3].strip()
                if cntry not in countries:
                    continue
                raw_name = row[1]
                raw_addr = row[2]
                cl_name, core, suff = normalize_business_name(raw_name)
                cl_addr_fast = clean_base_text(raw_addr)
                keys = get_blocking_keys(cl_name, core, cl_addr_fast, cntry)
                
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
                            target_records[tid] = (raw_name, raw_addr, cntry)
                            
        print(f"Streamed {label} in {time.time() - t0:.2f}s, candidate hits added: {hits:,}")
        
    print(f"Total unique target records collected: {len(target_records):,}")
    
    # Check Candidate Blocking Recall
    recovered_true = 0
    for sid, true_set in gt_map.items():
        cands_set = set(candidates.get(sid, []))
        recovered_true += len(true_set & cands_set)
    blocking_recall = (recovered_true / total_true_matches * 100) if total_true_matches > 0 else 0.0
    print(f"Candidate Blocking Recall on True Matches: {blocking_recall:.2f}% ({recovered_true:,} / {total_true_matches:,})")
    
    # 4. Preprocess target records
    t0 = time.time()
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
    
    # 5. Extract features & score with EntityMatcher
    matcher = EntityMatcher()
    print(f"Scoring pairs with LightGBM model...")
    all_scored_pairs = [] # (s1_id, target_id, prob)
    
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
                        all_scored_pairs.append((s, t, float(p)))
                    batch_feats = []
                    batch_pairs = []
                    
    if batch_feats:
        probs = matcher.predict_proba(np.array(batch_feats, dtype=np.float32))
        for (s, t), p in zip(batch_pairs, probs):
            all_scored_pairs.append((s, t, float(p)))
            
    print(f"Scored {len(all_scored_pairs):,} candidate pairs in {time.time() - t0:.2f}s.")
    
    # 6. Evaluate over range of thresholds
    thresholds = [0.70, 0.75, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.95]
    results = []
    
    print("\n" + "="*85)
    print(f"{'Threshold':<10} | {'Macro F0.5':<12} | {'Precision':<11} | {'Recall':<10} | {'Singleton Acc':<14} | {'Matches Pred':<12}")
    print("="*85)
    
    for T in thresholds:
        preds = defaultdict(set)
        total_pred = 0
        for sid, tid, p in all_scored_pairs:
            if p >= T:
                preds[sid].add(tid)
                total_pred += 1
                
        # Fill singletons/empty
        full_preds = {sid: preds.get(sid, set()) for sid in s1_order}
        metrics = compute_macro_f05(gt_map, full_preds)
        results.append({
            "threshold": T,
            "macro_f05": metrics["macro_f05"],
            "precision": metrics["mean_precision"],
            "recall": metrics["mean_recall"],
            "singleton_acc": metrics["singleton_accuracy"],
            "total_pred": total_pred
        })
        print(f"{T:<10.2f} | {metrics['macro_f05']:<12.5f} | {metrics['mean_precision']:<11.5f} | {metrics['mean_recall']:<10.5f} | {metrics['singleton_accuracy']:<14.5f} | {total_pred:<12,}")
        
    print("="*85)
    
    # Return best threshold and summary
    best_res = max(results, key=lambda x: x["macro_f05"])
    print(f"\nOptimal Threshold on Held-Out Validation: T = {best_res['threshold']:.2f} (Macro F0.5 = {best_res['macro_f05']:.5f})")
    
    return results

if __name__ == "__main__":
    os.makedirs(os.path.join(PROJECT_ROOT, "scripts"), exist_ok=True)
    run_validation(s1_start=15000, n_s1=2000)
