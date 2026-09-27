"""
Forensic Diagnosis of Missed True Matches in Candidate Blocking
Investigates the exact causes of why 47.5% of true matches are not retrieved as candidates.
"""

import sys
import os
import csv
import re
from collections import defaultdict, Counter
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
from src.prediction.predict_test_chunked import get_blocking_keys

sys.stdout.reconfigure(encoding='utf-8')

def diagnose_missed(s1_start=20000, n_s1=2500):
    train_s1 = os.path.join(PROJECT_ROOT, "dataset", "train", "train_source1.tsv")
    train_s2 = os.path.join(PROJECT_ROOT, "dataset", "train", "train_source2.tsv")
    train_s3 = os.path.join(PROJECT_ROOT, "dataset", "train", "train_source3.tsv")
    train_gt = os.path.join(PROJECT_ROOT, "dataset", "train", "train_ground_truth.tsv")

    # 1. Load GT
    gt_map = {}
    needed_s1 = set()
    needed_targets = set()
    with open(train_gt, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for i, row in enumerate(reader):
            if i < s1_start:
                continue
            if i >= s1_start + n_s1:
                break
            sid = row[0].strip()
            matches = set(m.strip() for m in row[1].split(",") if m.strip()) if len(row) > 1 and row[1].strip() else set()
            gt_map[sid] = matches
            needed_s1.add(sid)
            needed_targets.update(matches)

    # 2. Load S1 raw records
    s1_raw = {}
    with open(train_s1, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            sid = row[0].strip()
            if sid in needed_s1:
                s1_raw[sid] = (row[1], row[2], row[3])
                if len(s1_raw) >= len(needed_s1):
                    break

    # 3. Load Target raw records (only those in GT)
    target_raw = {}
    for path in [train_s2, train_s3]:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                tid = row[0].strip()
                if tid in needed_targets:
                    target_raw[tid] = (row[1], row[2], row[3])

    print(f"Loaded {len(s1_raw):,} S1 raw records and {len(target_raw):,} target ground truth records.")

    # 4. Compare S1 vs Target for all true pairs
    missed_count = 0
    hit_count = 0

    missed_by_source = Counter()
    missed_by_country = Counter()
    reason_counter = Counter()

    detailed_examples = []

    for sid, true_tids in gt_map.items():
        if sid not in s1_raw:
            continue
        s1_name, s1_addr, s1_country = s1_raw[sid]
        s1_cl_name, s1_core, s1_suff = normalize_business_name(s1_name)
        s1_cl_addr = normalize_business_address(s1_addr)
        s1_cntry = normalize_country(s1_country)
        s1_keys = set(get_blocking_keys(s1_cl_name, s1_core, s1_cl_addr, s1_cntry))

        for tid in true_tids:
            if tid not in target_raw:
                continue
            t_name, t_addr, t_country = target_raw[tid]
            t_cl_name, t_core, t_suff = normalize_business_name(t_name)
            t_cl_addr = clean_base_text(t_addr) # as in stream
            t_cntry = normalize_country(t_country)
            t_keys = set(get_blocking_keys(t_cl_name, t_core, t_cl_addr, t_cntry))

            shared_keys = s1_keys & t_keys
            src_type = "Source 2" if tid.startswith("S2") else "Source 3"

            if shared_keys:
                hit_count += 1
            else:
                missed_count += 1
                missed_by_source[src_type] += 1
                missed_by_country[s1_cntry] += 1

                # Analyze reasons
                country_mismatch = (s1_cntry != t_cntry)
                name_exact = (s1_cl_name == t_cl_name)
                core_exact = (s1_core == t_core)
                addr_exact = (clean_base_text(s1_addr) == clean_base_text(t_addr))

                s1_tokens = set(s1_core.split())
                t_tokens = set(t_core.split())
                token_overlap = len(s1_tokens & t_tokens)

                if country_mismatch:
                    reason_counter["country_mismatch"] += 1
                elif name_exact or core_exact:
                    reason_counter["name_exact_but_key_failed"] += 1
                elif token_overlap > 0:
                    if s1_tokens == t_tokens:
                        reason_counter["word_order_variation"] += 1
                    else:
                        reason_counter["partial_name_token_overlap"] += 1
                elif s1_cl_addr and t_cl_addr and addr_exact:
                    reason_counter["exact_addr_different_name"] += 1
                else:
                    reason_counter["complete_name_divergence_or_abbreviation"] += 1

                if len(detailed_examples) < 25:
                    detailed_examples.append({
                        "sid": sid,
                        "tid": tid,
                        "src": src_type,
                        "s1_name": s1_name,
                        "t_name": t_name,
                        "s1_core": s1_core,
                        "t_core": t_core,
                        "s1_addr": s1_addr,
                        "t_addr": t_addr,
                        "s1_keys": list(s1_keys),
                        "t_keys": list(t_keys),
                        "s1_cntry": s1_cntry,
                        "t_cntry": t_cntry
                    })

    print(f"\nTotal Pairs Analyzed: {hit_count + missed_count:,}")
    print(f"Key Intersect Hits: {hit_count:,} ({hit_count/(hit_count+missed_count)*100:.2f}%)")
    print(f"Key Intersect Missed: {missed_count:,} ({missed_count/(hit_count+missed_count)*100:.2f}%)")

    print("\n--- MISSED BY SOURCE ---")
    for src, c in missed_by_source.items():
        print(f"  {src}: {c:,} ({c/missed_count*100:.2f}%)")

    print("\n--- MISSED BY REASON ---")
    for reason, c in reason_counter.most_common():
        print(f"  {reason:<45}: {c:,} ({c/missed_count*100:.2f}%)")

    print("\n--- MISSED BY COUNTRY (Top 5) ---")
    for cntry, c in missed_by_country.most_common(5):
        print(f"  {cntry}: {c:,}")

    print("\n--- SAMPLE MISSED EXAMPLES ---")
    for ex in detailed_examples[:15]:
        print(f"[{ex['src']}] S1: {ex['sid']} vs Target: {ex['tid']}")
        print(f"  S1 Name:     {ex['s1_name']}  (Core: '{ex['s1_core']}')")
        print(f"  Target Name: {ex['t_name']}  (Core: '{ex['t_core']}')")
        print(f"  S1 Addr:     {ex['s1_addr']}")
        print(f"  Target Addr: {ex['t_addr']}")
        print(f"  S1 Keys:     {ex['s1_keys']}")
        print(f"  Target Keys: {ex['t_keys']}")
        print("-" * 60)

if __name__ == "__main__":
    diagnose_missed()
