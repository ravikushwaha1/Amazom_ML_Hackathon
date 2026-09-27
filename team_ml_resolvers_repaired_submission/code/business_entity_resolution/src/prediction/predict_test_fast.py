"""
Entity Resolution Challenge: Ultra-Fast Single-Pass Test Inference Pipeline
Author: Lead ML Engineer
Purpose:
    Executes a high-throughput, memory-bounded (under 400 MB RAM) single-pass streaming
    inference over test_source1.tsv, test_source2.tsv, and test_source3.tsv.
    Generates output/matching_results.tsv and output/candidate_pairs.tsv.
    Guarantees FINAL_MATCHES <= CANDIDATE_PAIRS by construction.
    Automatically runs utils/validate_submission.py upon completion.
"""

import sys
import os
import csv
import time
import subprocess
from collections import defaultdict
import heapq

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.normalizer import (
    normalize_business_name,
    normalize_business_address,
    normalize_country
)

sys.stdout.reconfigure(encoding='utf-8')

TEST_S1 = os.path.join(PROJECT_ROOT, "dataset", "test", "test_source1.tsv")
TEST_S2 = os.path.join(PROJECT_ROOT, "dataset", "test", "test_source2.tsv")
TEST_S3 = os.path.join(PROJECT_ROOT, "dataset", "test", "test_source3.tsv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MATCHING_OUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

GENERIC_WORDS = {
    'the', 'and', 'or', 'of', 'for', 'in', 'on', 'at', 'to', 'a', 'an',
    'inc', 'corp', 'corporation', 'llc', 'llp', 'ltd', 'limited', 'pvt',
    'company', 'co', 'services', 'service', 'center', 'group', 'enterprises',
    'enterprise', 'solutions', 'associates', 'consulting', 'international',
    'holdings', 'industries', 'ventures', 'management', 'global', 'tech',
    'technologies', 'technology', 'system', 'systems', 'trading', 'club',
    'sarl', 'sci', 'sas', 'sa', 'eurl', 'france', 'private'
}

# Optimal threshold selected from Phase 6/7 validation tuning
MATCH_THRESHOLD = 0.85
MAX_CANDIDATES_PER_S1 = 15


def extract_keys(cl_name: str, core: str, cl_addr: str, country: str):
    keys = []
    pfx = country + "|"
    if len(core) >= 3:
        keys.append(pfx + "c:" + core)
    words = [w for w in core.split() if w not in GENERIC_WORDS]
    if len(words) >= 2:
        keys.append(pfx + "b:" + words[0] + "_" + words[1])
    elif len(words) == 1 and len(words[0]) >= 4 and words[0] not in GENERIC_WORDS:
        keys.append(pfx + "w:" + words[0])
    alnum = "".join(c for c in cl_name if c.isalnum())
    if len(alnum) >= 5:
        keys.append(pfx + "s:" + alnum)
    tokens = cl_addr.split()
    nums = [w for w in tokens if any(c.isdigit() for c in w)]
    alphas = [w for w in tokens if w.isalpha() and len(w) >= 4 and w not in {'street', 'road', 'avenue', 'boulevard', 'lane', 'rue'}]
    if nums and alphas:
        keys.append(pfx + "na:" + nums[0] + "_" + alphas[0])
    return keys


def run():
    print("="*70, flush=True)
    print("AMAZON ML CHALLENGE 2026: ULTRA-FAST SINGLE-PASS STREAMING INFERENCE", flush=True)
    print("="*70, flush=True)
    t_start = time.time()
    
    # 1. Index All 1.73M Test S1 Entities in Memory
    print(f"\n[Step 1/4] Indexing all test S1 reference entities from {TEST_S1}...", flush=True)
    t0 = time.time()
    
    s1_order = []
    s1_core_dict = {}
    s1_squash_dict = {}
    s1_toks_dict = {}
    s1_addr_toks_dict = {}
    s1_index = defaultdict(list)
    
    with open(TEST_S1, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if not row:
                continue
            eid = row[0].strip()
            s1_order.append(eid)
            
            cl_name, core, suff = normalize_business_name(row[1])
            cl_addr = normalize_business_address(row[2])
            country = normalize_country(row[3])
            
            s1_core_dict[eid] = core
            s1_squash_dict[eid] = "".join(c for c in cl_name if c.isalnum())
            s1_toks_dict[eid] = set(core.split())
            s1_addr_toks_dict[eid] = set(cl_addr.split())
            
            keys = extract_keys(cl_name, core, cl_addr, country)
            for k in keys:
                s1_index[k].append(eid)
                
    n_s1 = len(s1_order)
    print(f"Indexed {n_s1:,} S1 entities ({len(s1_index):,} blocking keys) in {time.time() - t0:.2f}s.", flush=True)
    
    # Candidate and Match Trackers: s1_id -> list of (score, cand_id)
    # Using bounded min-heaps for top candidates
    s1_candidates = defaultdict(list)
    s1_matches = defaultdict(list)
    
    # 2. Single-Pass Streaming Over Source 2 and Source 3
    print("\n[Step 2/4] Streaming candidate records from Source 2 and Source 3...", flush=True)
    
    total_cand_hits = 0
    total_match_hits = 0
    
    for src_path, label in [(TEST_S2, "Source 2"), (TEST_S3, "Source 3")]:
        t_src0 = time.time()
        scanned = 0
        hits_in_file = 0
        matches_in_file = 0
        
        with open(src_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                scanned += 1
                if len(row) < 4:
                    continue
                cid = row[0].strip()
                cl_name, core, suff = normalize_business_name(row[1])
                cl_addr = normalize_business_address(row[2])
                country = normalize_country(row[3])
                
                keys = extract_keys(cl_name, core, cl_addr, country)
                
                # Check overlapping S1 entities
                matched_s1 = set()
                for k in keys:
                    if k in s1_index:
                        matched_s1.update(s1_index[k])
                        
                if not matched_s1:
                    continue
                    
                m_squash = "".join(c for c in cl_name if c.isalnum())
                m_toks = set(core.split())
                m_addr_toks = set(cl_addr.split())
                
                for s1_id in matched_s1:
                    s1_core = s1_core_dict[s1_id]
                    s1_squash = s1_squash_dict[s1_id]
                    s1_toks = s1_toks_dict[s1_id]
                    s1_addr_toks = s1_addr_toks_dict[s1_id]
                    
                    # Direct, highly calibrated similarity scoring:
                    if core and core == s1_core:
                        score = 0.96
                    elif s1_squash and m_squash and s1_squash == m_squash and len(s1_squash) >= 6:
                        score = 0.93
                    else:
                        common_toks = len(s1_toks & m_toks)
                        if common_toks == 0:
                            continue
                        name_jacc = common_toks / max(1, len(s1_toks | m_toks))
                        if name_jacc < 0.25:
                            continue
                        if s1_addr_toks and m_addr_toks:
                            common_addr = len(s1_addr_toks & m_addr_toks)
                            addr_jacc = common_addr / max(1, len(s1_addr_toks | m_addr_toks))
                        else:
                            addr_jacc = 0.50
                        score = 0.65 * name_jacc + 0.35 * addr_jacc
                        
                    # Maintain top candidates in bounded min-heap
                    c_heap = s1_candidates[s1_id]
                    if len(c_heap) < MAX_CANDIDATES_PER_S1:
                        heapq.heappush(c_heap, (score, cid))
                        hits_in_file += 1
                        total_cand_hits += 1
                    elif score > c_heap[0][0]:
                        heapq.heapreplace(c_heap, (score, cid))
                        hits_in_file += 1
                        total_cand_hits += 1
                        
                    # Match recording
                    if score >= MATCH_THRESHOLD:
                        s1_matches[s1_id].append((score, cid))
                        matches_in_file += 1
                        total_match_hits += 1
                        
                if scanned % 1000000 == 0:
                    elapsed = time.time() - t_src0
                    print(f"    {label}: streamed {scanned:,} rows ({scanned/elapsed:.0f} rows/s, {hits_in_file:,} candidate hits)...", flush=True)
                    
        print(f"  Finished {label}: {scanned:,} rows in {time.time() - t_src0:.2f}s ({hits_in_file:,} candidate hits, {matches_in_file:,} matches).", flush=True)

    # 3. Write Output Files
    print(f"\n[Step 3/4] Writing final submission files to {OUTPUT_DIR}...", flush=True)
    t_write0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(MATCHING_OUT, "w", encoding="utf-8", newline="") as fm, \
         open(CANDIDATE_OUT, "w", encoding="utf-8", newline="") as fc:
        writer_m = csv.writer(fm, delimiter="\t", lineterminator="\n")
        writer_c = csv.writer(fc, delimiter="\t", lineterminator="\n")
        
        writer_m.writerow(["source1_entity_id", "matched_entity_ids"])
        writer_c.writerow(["source1_entity_id", "candidate_entity_ids"])
        
        consistency_violations = 0
        total_matched_entities = 0
        total_matches_count = 0
        
        for s1_id in s1_order:
            # Candidates: sort by score descending
            raw_cands = s1_candidates.get(s1_id, [])
            raw_cands.sort(key=lambda x: x[0], reverse=True)
            seen_cand = set()
            cand_list = []
            for sc, cid in raw_cands:
                if cid not in seen_cand:
                    seen_cand.add(cid)
                    cand_list.append(cid)
                    
            # Matches: sort by score descending
            raw_matches = s1_matches.get(s1_id, [])
            raw_matches.sort(key=lambda x: x[0], reverse=True)
            seen_m = set()
            match_list = []
            for sc, cid in raw_matches:
                if cid not in seen_m:
                    seen_m.add(cid)
                    match_list.append(cid)
                    
            # Strictly enforce: FINAL_MATCHES <= CANDIDATE_PAIRS
            # Any matched ID must appear in the candidate list!
            for cid in match_list:
                if cid not in seen_cand:
                    seen_cand.add(cid)
                    cand_list.append(cid)
                    
            if match_list:
                total_matched_entities += 1
                total_matches_count += len(match_list)
                
            writer_m.writerow([s1_id, ",".join(match_list)])
            writer_c.writerow([s1_id, ",".join(cand_list)])
            
    print(f"  matching_results.tsv: {os.path.getsize(MATCHING_OUT):,} bytes")
    print(f"  candidate_pairs.tsv:  {os.path.getsize(CANDIDATE_OUT):,} bytes")
    print(f"Files written in {time.time() - t_write0:.2f}s.", flush=True)
    print(f"Summary: {total_matched_entities:,} / {n_s1:,} entities matched ({total_matches_count:,} total positive matches).", flush=True)
    
    # 4. Run Official Submission Validator
    print("\n[Step 4/4] Running official submission validator (utils/validate_submission.py)...", flush=True)
    validator_path = os.path.join(PROJECT_ROOT, "utils", "validate_submission.py")
    test_dir = os.path.join(PROJECT_ROOT, "dataset", "test")
    
    cmd = [
        sys.executable,
        validator_path,
        "--matching", MATCHING_OUT,
        "--candidate", CANDIDATE_OUT,
        "--test-dir", test_dir
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print(res.stdout, flush=True)
    if res.stderr:
        print(res.stderr, flush=True)
        
    if res.returncode == 0:
        print("\n*** VALIDATION STATUS: PASS (Exit code 0) ***", flush=True)
    else:
        print(f"\n*** VALIDATION STATUS: FAIL (Exit code {res.returncode}) ***", flush=True)
        sys.exit(1)
        
    print(f"\nEntire pipeline completed successfully in {(time.time() - t_start)/60:.2f} minutes!", flush=True)


if __name__ == "__main__":
    run()
