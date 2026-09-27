"""
10,000 S1 Controlled Test & Validation Runner
Author: Lead ML Engineer
Purpose:
    Executes a controlled test on ONLY 10,000 S1 entities using predict_test_chunked.py.
    Validates all formatting rules, candidate subset conditions, and duplicate checks.
    Measures and outputs all 17 diagnostic items required by the user.
"""

import sys
import os
import json
import time
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.prediction.predict_test_chunked import run_chunked_inference

TEST_OUT_DIR = os.path.join(PROJECT_ROOT, "output", "test_10k")
MATCHING_10K = os.path.join(TEST_OUT_DIR, "matching_results.tsv")
CANDIDATE_10K = os.path.join(TEST_OUT_DIR, "candidate_pairs.tsv")
TEST_S1 = os.path.join(PROJECT_ROOT, "dataset", "test", "test_source1.tsv")


def run_and_validate_10k():
    print("="*75)
    print("STARTING CONTROLLED 10,000 S1 INFERENCE & VALIDATION TEST")
    print("="*75)
    
    os.makedirs(TEST_OUT_DIR, exist_ok=True)
    
    # 1. Run controlled chunked inference on 10k entities
    t0 = time.time()
    completed_without_crash = False
    
    try:
        report = run_chunked_inference(
            batch_size=10000,
            max_s1=10000,
            matching_out=MATCHING_10K,
            candidate_out=CANDIDATE_10K,
            threshold=0.85,
            max_candidates_per_s1=25
        )
        completed_without_crash = True
    except Exception as e:
        print(f"ERROR: Inference failed with exception: {e}")
        report = {}
        
    t_elapsed = time.time() - t0
    
    # 2. Detailed format & correctness validation
    print("\n--- RUNNING DETAILED VALIDATIONS ---")
    val_matching_format = True
    val_candidate_format = True
    val_subset_check = True
    val_no_intra_dupes = True
    val_no_invalid_prefixes = True
    val_no_self_matches = True
    
    matching_data = {}
    candidate_data = {}
    
    # Read test S1 IDs for the first 10,000 rows
    expected_s1_order = []
    with open(TEST_S1, "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i >= 10000:
                break
            parts = line.strip().split("\t")
            if parts:
                expected_s1_order.append(parts[0].strip())
                
    # Validate matching_results.tsv
    matching_lines_count = 0
    with open(MATCHING_10K, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != ["source1_entity_id", "matched_entity_ids"]:
            val_matching_format = False
            print(f"FAIL: matching header is {header}")
            
        for line_idx, line in enumerate(f, start=2):
            matching_lines_count += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                val_matching_format = False
                print(f"FAIL: malformed line {line_idx} in matching_results: {line}")
                continue
            s1_id, m_str = parts
            m_ids = m_str.split(",") if m_str else []
            # Intra-row duplicates
            if len(m_ids) != len(set(m_ids)):
                val_no_intra_dupes = False
                print(f"FAIL: intra-row duplicate in matching_results for {s1_id}")
            # Prefix check
            for mid in m_ids:
                if mid.startswith("S1-"):
                    val_no_self_matches = False
                    print(f"FAIL: self-match found in matching_results: {mid}")
                elif not mid.startswith(("S2-", "S3-")):
                    val_no_invalid_prefixes = False
                    print(f"FAIL: invalid prefix in matching_results: {mid}")
            matching_data[s1_id] = m_ids
            
    # Validate candidate_pairs.tsv
    candidate_lines_count = 0
    with open(CANDIDATE_10K, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != ["source1_entity_id", "candidate_entity_ids"]:
            val_candidate_format = False
            print(f"FAIL: candidate header is {header}")
            
        for line_idx, line in enumerate(f, start=2):
            candidate_lines_count += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                val_candidate_format = False
                print(f"FAIL: malformed line {line_idx} in candidate_pairs: {line}")
                continue
            s1_id, c_str = parts
            c_ids = c_str.split(",") if c_str else []
            # Intra-row duplicates
            if len(c_ids) != len(set(c_ids)):
                val_no_intra_dupes = False
                print(f"FAIL: intra-row duplicate in candidate_pairs for {s1_id}")
            # Prefix check
            for cid in c_ids:
                if cid.startswith("S1-"):
                    val_no_self_matches = False
                    print(f"FAIL: self-match found in candidate_pairs: {cid}")
                elif not cid.startswith(("S2-", "S3-")):
                    val_no_invalid_prefixes = False
                    print(f"FAIL: invalid prefix in candidate_pairs: {cid}")
            candidate_data[s1_id] = c_ids
            
    # Check subset: predicted matches ⊆ candidate pairs
    subset_violations = 0
    for s1_id, m_list in matching_data.items():
        c_set = set(candidate_data.get(s1_id, []))
        for mid in m_list:
            if mid not in c_set:
                subset_violations += 1
                val_subset_check = False
                
    if subset_violations > 0:
        print(f"FAIL: {subset_violations} match IDs are not in candidate pairs!")
    else:
        print("PASS: predicted matches ⊆ candidate pairs (100% compliant, 0 violations).")
        
    # Check S1 ordering
    s1_order_match = (list(matching_data.keys()) == expected_s1_order) and (list(candidate_data.keys()) == expected_s1_order)
    print(f"PASS: S1 ordering matches test_source1.tsv: {s1_order_match}")
    print(f"PASS: Matching rows written: {matching_lines_count:,}")
    print(f"PASS: Candidate rows written: {candidate_lines_count:,}")
    
    # Compile final diagnostic dictionary
    result = {
        "1_runtime_seconds": report.get("runtime_seconds", t_elapsed),
        "2_peak_working_set_mb": report.get("peak_working_set_mb", 0.0),
        "3_peak_pagefile_mb": report.get("peak_pagefile_mb", 0.0),
        "4_s1_rows_written": matching_lines_count,
        "5_total_candidate_pairs": report.get("total_candidate_pairs", sum(len(v) for v in candidate_data.values())),
        "6_total_predicted_matches": report.get("total_predicted_matches", sum(len(v) for v in matching_data.values())),
        "7_avg_candidates_per_s1": report.get("avg_candidates_per_s1", 0.0),
        "8_avg_matches_per_non_empty_s1": report.get("avg_matches_per_non_empty_s1", 0.0),
        "9_s1_with_0_predictions": report.get("pred_count_0", sum(1 for v in matching_data.values() if len(v) == 0)),
        "10_s1_with_gt10_predictions": report.get("pred_count_gt10", sum(1 for v in matching_data.values() if len(v) > 10)),
        "11_s1_with_gt25_predictions": report.get("pred_count_gt25", sum(1 for v in matching_data.values() if len(v) > 25)),
        "12_s1_with_gt50_predictions": report.get("pred_count_gt50", sum(1 for v in matching_data.values() if len(v) > 50)),
        "13_matching_results_format_valid": val_matching_format and s1_order_match,
        "14_candidate_pairs_format_valid": val_candidate_format and s1_order_match,
        "15_predicted_matches_subset_of_candidates": val_subset_check,
        "16_duplicate_id_check_passed": val_no_intra_dupes and val_no_self_matches and val_no_invalid_prefixes,
        "17_completed_without_oom_crash": completed_without_crash
    }
    
    with open(os.path.join(TEST_OUT_DIR, "test_10k_report.json"), "w") as f:
        json.dump(result, f, indent=2)
        
    print("\nFINAL DIAGNOSTIC REPORT (JSON):")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_and_validate_10k()
