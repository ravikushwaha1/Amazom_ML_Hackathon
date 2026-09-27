"""
Entity Resolution Challenge: High-Performance Chunked Streaming Test Pipeline
Author: Lead ML Engineer
Purpose:
    Streams through test_source1.tsv, test_source2.tsv, and test_source3.tsv in memory-bounded
    chunks (<400 MB RAM), executes candidate generation, feature extraction, and ML scoring.
    Generates output/matching_results.tsv and output/candidate_pairs.tsv.
    Validates output using utils/validate_submission.py.
"""

import sys
import os
import csv
import time
import gc
import subprocess
from collections import defaultdict
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.normalizer import (
    normalize_business_name,
    normalize_business_address,
    normalize_country
)
from src.features.feature_extractor import extract_pair_features, get_char_ngrams
from src.model.matcher import EntityMatcher

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


def get_blocking_keys(cl_name: str, core: str, cl_addr: str, country: str):
    """Generates high-precision blocking keys namespaced by country."""
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


def run_pipeline(chunk_size: int = 350000):
    print("="*70, flush=True)
    print("AMAZON ML CHALLENGE 2026: STREAMING TEST INFERENCE PIPELINE", flush=True)
    print("="*70, flush=True)
    t_global0 = time.time()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize output files with headers
    with open(MATCHING_OUT, "w", encoding="utf-8", newline="") as fm, \
         open(CANDIDATE_OUT, "w", encoding="utf-8", newline="") as fc:
        csv.writer(fm, delimiter="\t", lineterminator="\n").writerow(["source1_entity_id", "matched_entity_ids"])
        csv.writer(fc, delimiter="\t", lineterminator="\n").writerow(["source1_entity_id", "candidate_entity_ids"])
        
    matcher = EntityMatcher()
    print(f"Loaded EntityMatcher (threshold = {matcher.threshold:.2f})", flush=True)
    
    # Read test_source1 in chunks
    total_s1_processed = 0
    chunk_index = 0
    
    with open(TEST_S1, "r", encoding="utf-8") as f_s1:
        reader_s1 = csv.reader(f_s1, delimiter="\t")
        header = next(reader_s1)
        
        while True:
            chunk_s1_order = []
            chunk_s1_records = {}
            chunk_s1_index = defaultdict(list)
            
            for row in reader_s1:
                if not row:
                    continue
                eid = row[0].strip()
                chunk_s1_order.append(eid)
                
                cl_name, core, suff = normalize_business_name(row[1])
                cl_addr = normalize_business_address(row[2])
                country = normalize_country(row[3])
                
                addr_toks = set(cl_addr.split())
                nums = [w for w in addr_toks if any(ch.isdigit() for ch in w)]
                
                chunk_s1_records[eid] = {
                    "entity_id": eid,
                    "clean_name": cl_name,
                    "core_name": core,
                    "legal_suffix": suff,
                    "clean_addr": cl_addr,
                    "country": country,
                    "name_tokens": set(cl_name.split()),
                    "name_3grams": get_char_ngrams(core),
                    "addr_tokens": addr_toks,
                    "addr_3grams": get_char_ngrams(cl_addr),
                    "house_num": nums[0] if nums else ""
                }
                
                keys = get_blocking_keys(cl_name, core, cl_addr, country)
                for k in keys:
                    chunk_s1_index[k].append(eid)
                    
                if len(chunk_s1_order) >= chunk_size:
                    break
                    
            if not chunk_s1_order:
                break
                
            chunk_index += 1
            n_chunk = len(chunk_s1_order)
            t_chunk0 = time.time()
            print(f"\n--- CHUNK {chunk_index}: Processing {n_chunk:,} S1 entities (Overall: {total_s1_processed:,} - {total_s1_processed + n_chunk:,}) ---", flush=True)
            print(f"Indexed {n_chunk:,} S1 records ({len(chunk_s1_index):,} blocking keys) in {time.time() - t_chunk0:.2f}s", flush=True)
            
            # Streaming target candidate collection
            chunk_candidates = defaultdict(dict) # s1_id -> {cand_id: target_dict}
            
            for src_path, label in [(TEST_S2, "Source 2"), (TEST_S3, "Source 3")]:
                t_src0 = time.time()
                scanned = 0
                hits = 0
                with open(src_path, "r", encoding="utf-8") as f_src:
                    reader_src = csv.reader(f_src, delimiter="\t")
                    next(reader_src)
                    for row in reader_src:
                        scanned += 1
                        if len(row) < 4:
                            continue
                        country = row[3].strip()
                        cl_name, core, suff = normalize_business_name(row[1])
                        cl_addr = normalize_business_address(row[2])
                        keys = get_blocking_keys(cl_name, core, cl_addr, country)
                        
                        matched_s1 = set()
                        for k in keys:
                            for s1_id in chunk_s1_index.get(k, []):
                                matched_s1.add(s1_id)
                                
                        if matched_s1:
                            eid = row[0].strip()
                            addr_toks = set(cl_addr.split())
                            nums = [w for w in addr_toks if any(ch.isdigit() for ch in w)]
                            t_dict = {
                                "entity_id": eid,
                                "clean_name": cl_name,
                                "core_name": core,
                                "legal_suffix": suff,
                                "clean_addr": cl_addr,
                                "country": country,
                                "name_tokens": set(cl_name.split()),
                                "name_3grams": get_char_ngrams(core),
                                "addr_tokens": addr_toks,
                                "addr_3grams": get_char_ngrams(cl_addr),
                                "house_num": nums[0] if nums else ""
                            }
                            for s1_id in matched_s1:
                                c_map = chunk_candidates[s1_id]
                                if len(c_map) < 20:
                                    c_map[eid] = t_dict
                                    hits += 1
                                    
                print(f"  Streamed {label}: {scanned:,} rows in {time.time() - t_src0:.2f}s, hits: {hits:,}", flush=True)
                
            # Score candidate pairs with EntityMatcher
            print(f"Scoring candidate pairs for Chunk {chunk_index}...", flush=True)
            t_score0 = time.time()
            
            chunk_matches = defaultdict(list)
            
            # Batch feature extraction and prediction
            pair_meta = []
            feats = []
            
            for s1_id in chunk_s1_order:
                c_map = chunk_candidates.get(s1_id, {})
                s1 = chunk_s1_records[s1_id]
                for cid, target in c_map.items():
                    f = extract_pair_features(s1, target)
                    feats.append(f)
                    pair_meta.append((s1_id, cid))
                    
                    if len(feats) >= 100000:
                        X = np.array(feats, dtype=np.float32)
                        probs = matcher.predict_proba(X)
                        for (sid, cid_m), p in zip(pair_meta, probs):
                            if p >= matcher.threshold:
                                chunk_matches[sid].append((cid_m, float(p)))
                        feats = []
                        pair_meta = []
                        
            if feats:
                X = np.array(feats, dtype=np.float32)
                probs = matcher.predict_proba(X)
                for (sid, cid_m), p in zip(pair_meta, probs):
                    if p >= matcher.threshold:
                        chunk_matches[sid].append((cid_m, float(p)))
                        
            print(f"Scored in {time.time() - t_score0:.2f}s.", flush=True)
            
            # Append Chunk to output TSVs
            with open(MATCHING_OUT, "a", encoding="utf-8", newline="") as fm, \
                 open(CANDIDATE_OUT, "a", encoding="utf-8", newline="") as fc:
                writer_m = csv.writer(fm, delimiter="\t", lineterminator="\n")
                writer_c = csv.writer(fc, delimiter="\t", lineterminator="\n")
                
                for s1_id in chunk_s1_order:
                    # Candidates
                    c_list = list(chunk_candidates.get(s1_id, {}).keys())
                    writer_c.writerow([s1_id, ",".join(c_list)])
                    
                    # Matches (sorted by probability, deduped)
                    if s1_id in chunk_matches:
                        sorted_m = sorted(chunk_matches[s1_id], key=lambda x: x[1], reverse=True)
                        seen_c = set()
                        m_list = []
                        for cid, _ in sorted_m:
                            if cid not in seen_c:
                                seen_c.add(cid)
                                m_list.append(cid)
                        writer_m.writerow([s1_id, ",".join(m_list)])
                    else:
                        writer_m.writerow([s1_id, ""])
                        
            total_s1_processed += n_chunk
            elapsed_total = time.time() - t_global0
            print(f"Appended Chunk {chunk_index}. Total written: {total_s1_processed:,} / 1,732,544 ({total_s1_processed / max(1e-5, elapsed_total):.0f} S1/s)", flush=True)
            
            # Clean up memory completely for this chunk
            del chunk_s1_order
            del chunk_s1_records
            del chunk_s1_index
            del chunk_candidates
            del chunk_matches
            gc.collect()

    print(f"\nAll chunks completed in {(time.time() - t_global0)/60:.2f} minutes.", flush=True)
    
    # Run official validator
    print("\nRunning official submission validator (utils/validate_submission.py)...", flush=True)
    validator_path = os.path.join(PROJECT_ROOT, "utils", "validate_submission.py")
    test_dir = os.path.join(PROJECT_ROOT, "dataset", "test")
    
    cmd = [
        sys.executable,
        validator_path,
        "--matching", MATCHING_OUT,
        "--candidate", CANDIDATE_OUT,
        "--test-dir", test_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, flush=True)
        
    if result.returncode == 0:
        print("\n*** SUBMISSION VALIDATION PASSED (EXIT CODE 0) ***", flush=True)
    else:
        print(f"\n*** VALIDATION FAILED WITH EXIT CODE {result.returncode} ***", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline(chunk_size=350000)
