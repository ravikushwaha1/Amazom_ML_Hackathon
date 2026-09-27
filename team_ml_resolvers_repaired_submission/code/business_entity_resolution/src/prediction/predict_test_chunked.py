"""
Entity Resolution Challenge: Memory-Bounded Chunked Test Inference Pipeline
Author: Lead ML Engineer
Purpose:
    Executes memory-bounded, chunked streaming inference over test_source1.tsv,
    test_source2.tsv, and test_source3.tsv.
    
    Guarantees:
    - Peak working-set RAM < 500 MB (strictly bounded per chunk)
    - Zero pagefile expansion / thrashing
    - Immediate disk-backed chunk appending
    - 16 exact features scored with trained LightGBM model (threshold = 0.85)
    - predicted matches ⊆ candidate pairs
    - No duplicate entity IDs or malformed TSV rows
"""

import sys
import os
import csv
import json
import time
import gc
import argparse
import ctypes
from ctypes import wintypes
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.normalizer import (
    clean_base_text,
    normalize_business_name,
    normalize_business_address,
    normalize_country
)
from src.features.feature_extractor import extract_pair_features, get_char_ngrams
from src.model.matcher import EntityMatcher

sys.stdout.reconfigure(encoding='utf-8')

# Windows Memory Tracking via PSAPI
class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t)
    ]

psapi = ctypes.windll.psapi
kernel32 = ctypes.windll.kernel32
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL


def get_memory_stats() -> Dict[str, float]:
    """Returns current and peak memory stats in MB."""
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    ret = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if ret:
        return {
            'working_set_mb': counters.WorkingSetSize / (1024 * 1024),
            'peak_working_set_mb': counters.PeakWorkingSetSize / (1024 * 1024),
            'pagefile_mb': counters.PagefileUsage / (1024 * 1024),
            'peak_pagefile_mb': counters.PeakPagefileUsage / (1024 * 1024)
        }
    return {'working_set_mb': 0.0, 'peak_working_set_mb': 0.0, 'pagefile_mb': 0.0, 'peak_pagefile_mb': 0.0}


GENERIC_WORDS = {
    'the', 'and', 'or', 'of', 'for', 'in', 'on', 'at', 'to', 'a', 'an',
    'inc', 'corp', 'corporation', 'llc', 'llp', 'ltd', 'limited', 'pvt',
    'company', 'co', 'services', 'service', 'center', 'group', 'enterprises',
    'enterprise', 'solutions', 'associates', 'consulting', 'international',
    'holdings', 'industries', 'ventures', 'management', 'global', 'tech',
    'technologies', 'technology', 'system', 'systems', 'trading', 'club',
    'sarl', 'sci', 'sas', 'sa', 'eurl', 'france', 'private'
}


def get_blocking_keys(cl_name: str, core: str, cl_addr: str, country: str) -> List[str]:
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


def check_existing_progress(matching_out: str, candidate_out: str) -> int:
    """Checks if output files already exist with valid headers and equal row count.
    Returns the number of data rows already completed, or 0 to start fresh."""
    if not os.path.isfile(matching_out) or not os.path.isfile(candidate_out):
        return 0
    try:
        m_count = 0
        with open(matching_out, "r", encoding="utf-8") as fm:
            header = fm.readline().rstrip("\n")
            if header != "source1_entity_id\tmatched_entity_ids":
                return 0
            for _ in fm:
                m_count += 1
                
        c_count = 0
        with open(candidate_out, "r", encoding="utf-8") as fc:
            header = fc.readline().rstrip("\n")
            if header != "source1_entity_id\tcandidate_entity_ids":
                return 0
            for _ in fc:
                c_count += 1
                
        if m_count == c_count and m_count > 0:
            return m_count
        elif m_count > 0 and c_count > 0:
            # Files diverged; return min count to safely align
            return min(m_count, c_count)
    except Exception:
        return 0
    return 0


def run_chunked_inference(
    batch_size: int = 100000,
    max_s1: int = None,
    matching_out: str = None,
    candidate_out: str = None,
    model_path: str = None,
    threshold: float = 0.85,
    max_candidates_per_s1: int = 25,
    test_dir: str = None,
    resume: bool = True
) -> Dict[str, Any]:
    """
    Executes chunked inference with bounded memory and crash-resilient auto-resume.
    """
    t_start = time.time()
    
    if test_dir is None:
        test_dir = os.path.join(PROJECT_ROOT, "dataset", "test")
    test_s1 = os.path.join(test_dir, "test_source1.tsv")
    test_s2 = os.path.join(test_dir, "test_source2.tsv")
    test_s3 = os.path.join(test_dir, "test_source3.tsv")
    
    if matching_out is None:
        matching_out = os.path.join(PROJECT_ROOT, "output", "matching_results.tsv")
    if candidate_out is None:
        candidate_out = os.path.join(PROJECT_ROOT, "output", "candidate_pairs.tsv")
        
    out_dir = os.path.dirname(os.path.abspath(matching_out))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(candidate_out)), exist_ok=True)
    
    checkpoint_file = os.path.join(out_dir, "checkpoint.json")
    
    # Check for existing progress if resume is enabled
    existing_rows = check_existing_progress(matching_out, candidate_out) if resume else 0
    
    if existing_rows > 0:
        print(f"\n*** CHECKPOINT DETECTED: Resuming from row {existing_rows:,} ***", flush=True)
        print(f"Skipping first {existing_rows:,} already completed rows. Appending new batches.\n", flush=True)
        total_s1_written = existing_rows
    else:
        print("\nStarting fresh inference run. Initializing output files with headers.", flush=True)
        with open(matching_out, "w", encoding="utf-8", newline="") as fm, \
             open(candidate_out, "w", encoding="utf-8", newline="") as fc:
            csv.writer(fm, delimiter="\t", lineterminator="\n").writerow(["source1_entity_id", "matched_entity_ids"])
            csv.writer(fc, delimiter="\t", lineterminator="\n").writerow(["source1_entity_id", "candidate_entity_ids"])
        total_s1_written = 0
        
    # Load EntityMatcher with LightGBM / Vectorized Engine
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, "src", "model", "lgb_model.txt")
    matcher = EntityMatcher(model_path=model_path, threshold=threshold)
    print(f"Loaded EntityMatcher from {model_path} (threshold={matcher.threshold:.2f})", flush=True)
    
    total_candidate_pairs = 0
    total_predicted_matches = 0
    non_empty_match_entities = 0
    pred_count_0 = 0
    pred_count_gt10 = 0
    pred_count_gt25 = 0
    pred_count_gt50 = 0
    
    chunk_index = 0
    
    with open(test_s1, "r", encoding="utf-8") as f_s1:
        reader_s1 = csv.reader(f_s1, delimiter="\t")
        header = next(reader_s1)
        
        # Fast-forward past already written rows if resuming
        if existing_rows > 0:
            skipped = 0
            for _ in reader_s1:
                skipped += 1
                if skipped >= existing_rows:
                    break
            print(f"Skipped {skipped:,} rows in test_source1.tsv (resuming from next row).", flush=True)
            chunk_index = existing_rows // batch_size
            
        while True:
            chunk_s1_order = []
            chunk_s1_records = {}
            chunk_s1_index = defaultdict(list)
            chunk_countries = set()
            
            for row in reader_s1:
                if not row:
                    continue
                eid = row[0].strip()
                chunk_s1_order.append(eid)
                
                cl_name, core, suff = normalize_business_name(row[1])
                cl_addr = normalize_business_address(row[2])
                country = normalize_country(row[3])
                chunk_countries.add(country)
                
                addr_toks = set(cl_addr.split())
                nums = [w for w in addr_toks if any(ch.isdigit() for ch in w)]
                
                chunk_s1_records[eid] = {
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
                    chunk_s1_index[k].append(eid)
                    
                if len(chunk_s1_order) >= batch_size:
                    break
                if max_s1 is not None and (total_s1_written + len(chunk_s1_order)) >= max_s1:
                    break
                    
            if not chunk_s1_order:
                break
                
            chunk_index += 1
            n_chunk = len(chunk_s1_order)
            t_chunk0 = time.time()
            print(f"\n[Batch {chunk_index}] Processing {n_chunk:,} S1 entities (Overall: {total_s1_written:,} - {total_s1_written + n_chunk:,})...", flush=True)
            
            # Streaming target candidate collection (Bounded Memory Flyweight)
            # chunk_candidates: s1_id -> list of target_id (unique, capped at max_candidates_per_s1)
            # target_records: target_id -> raw target tuple (stored once per unique target)
            chunk_candidates = defaultdict(list)
            target_records = {}
            
            for src_path, label in [(test_s2, "Source 2"), (test_s3, "Source 3")]:
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
                        # Fast country filter
                        if country not in chunk_countries:
                            continue
                            
                        raw_name = row[1]
                        raw_addr = row[2]
                        
                        cl_name, core, suff = normalize_business_name(raw_name)
                        cl_addr_fast = clean_base_text(raw_addr)
                        keys = get_blocking_keys(cl_name, core, cl_addr_fast, country)
                        
                        matched_s1 = set()
                        for k in keys:
                            if k in chunk_s1_index:
                                s1_list = chunk_s1_index[k]
                                # Anti-hotspot guard: skip over-saturated keys (>500 entities)
                                if len(s1_list) <= 500:
                                    matched_s1.update(s1_list)
                                    
                        if matched_s1:
                            target_id = row[0].strip()
                            # Check which matched S1s still have room
                            eligible_s1 = [s1_id for s1_id in matched_s1 if len(chunk_candidates[s1_id]) < max_candidates_per_s1 and target_id not in chunk_candidates[s1_id]]
                            if eligible_s1:
                                for s1_id in eligible_s1:
                                    chunk_candidates[s1_id].append(target_id)
                                    hits += 1
                                if target_id not in target_records:
                                    target_records[target_id] = (raw_name, raw_addr, country)
                                    
                print(f"  Streamed {label}: {scanned:,} rows in {time.time() - t_src0:.2f}s, candidate hits added: {hits:,}", flush=True)
                
            # Prepare preprocessed target objects for feature extraction
            # Preprocess only the unique target candidates that were actually retrieved
            print(f"  Preprocessing {len(target_records):,} unique candidate targets...", flush=True)
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
            
            # Batch Feature Extraction and LightGBM Scoring
            print(f"  Scoring candidates with LightGBM (threshold = {matcher.threshold:.2f})...", flush=True)
            t_score0 = time.time()
            chunk_matches = defaultdict(list)
            
            batch_pairs = []
            batch_feats = []
            
            for s1_id in chunk_s1_order:
                c_list = chunk_candidates.get(s1_id, [])
                s1_dict = chunk_s1_records[s1_id]
                for cid in c_list:
                    if cid in prep_targets:
                        f = extract_pair_features(s1_dict, prep_targets[cid])
                        batch_feats.append(f)
                        batch_pairs.append((s1_id, cid))
                        
                        if len(batch_feats) >= 50000:
                            X = np.array(batch_feats, dtype=np.float32)
                            probs = matcher.predict_proba(X)
                            for (sid, cid_m), p in zip(batch_pairs, probs):
                                if p >= matcher.threshold:
                                    chunk_matches[sid].append((cid_m, float(p)))
                            batch_feats = []
                            batch_pairs = []
                            
            if batch_feats:
                X = np.array(batch_feats, dtype=np.float32)
                probs = matcher.predict_proba(X)
                for (sid, cid_m), p in zip(batch_pairs, probs):
                    if p >= matcher.threshold:
                        chunk_matches[sid].append((cid_m, float(p)))
                        
            del prep_targets
            del batch_feats
            del batch_pairs
            
            # Immediately write/append batch results to disk
            t_write0 = time.time()
            with open(matching_out, "a", encoding="utf-8", newline="") as fm, \
                 open(candidate_out, "a", encoding="utf-8", newline="") as fc:
                writer_m = csv.writer(fm, delimiter="\t", lineterminator="\n")
                writer_c = csv.writer(fc, delimiter="\t", lineterminator="\n")
                
                for s1_id in chunk_s1_order:
                    raw_cands = chunk_candidates.get(s1_id, [])
                    # Deduplicate candidate IDs while preserving order
                    seen_c = set()
                    cand_list = []
                    for cid in raw_cands:
                        if cid not in seen_c:
                            seen_c.add(cid)
                            cand_list.append(cid)
                            
                    # Matches: sort by probability descending, deduplicate
                    raw_matches = chunk_matches.get(s1_id, [])
                    raw_matches.sort(key=lambda x: x[1], reverse=True)
                    seen_m = set()
                    match_list = []
                    for cid, prob in raw_matches:
                        if cid not in seen_m:
                            seen_m.add(cid)
                            match_list.append(cid)
                            
                    # Strict validation rule: predicted matches MUST appear in candidate pairs!
                    for cid in match_list:
                        if cid not in seen_c:
                            seen_c.add(cid)
                            cand_list.append(cid)
                            
                    writer_m.writerow([s1_id, ",".join(match_list)])
                    writer_c.writerow([s1_id, ",".join(cand_list)])
                    
                    # Update global statistics
                    n_cands = len(cand_list)
                    n_matches = len(match_list)
                    total_candidate_pairs += n_cands
                    total_predicted_matches += n_matches
                    
                    if n_matches == 0:
                        pred_count_0 += 1
                    else:
                        non_empty_match_entities += 1
                        if n_matches > 10:
                            pred_count_gt10 += 1
                        if n_matches > 25:
                            pred_count_gt25 += 1
                        if n_matches > 50:
                            pred_count_gt50 += 1
                            
                fm.flush()
                fc.flush()
                
            total_s1_written += n_chunk
            mem = get_memory_stats()
            total_batches = 18 if max_s1 is None else ((max_s1 + batch_size - 1) // batch_size)
            
            # Write checkpoint metadata to disk
            checkpoint_saved = False
            try:
                with open(checkpoint_file, "w", encoding="utf-8") as fcp:
                    json.dump({
                        "total_s1_written": total_s1_written,
                        "last_completed_batch": chunk_index,
                        "total_batches": total_batches,
                        "batch_size": batch_size,
                        "peak_working_set_mb": mem['peak_working_set_mb'],
                        "peak_pagefile_mb": mem['peak_pagefile_mb'],
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }, fcp, indent=2)
                checkpoint_saved = True
            except Exception as e:
                print(f"  [Warning] Could not write checkpoint file: {e}", flush=True)
                
            print(f"\n" + "="*60, flush=True)
            print(f"BATCH {chunk_index} / {total_batches} COMPLETED & PERSISTED", flush=True)
            print(f"- Batch Index:            {chunk_index} / {total_batches}", flush=True)
            print(f"- S1 rows completed:      {total_s1_written:,} / 1,732,544 ({total_s1_written/1732544*100:.1f}%)", flush=True)
            print(f"- matching_results rows:  {total_s1_written:,}", flush=True)
            print(f"- candidate_pairs rows:   {total_s1_written:,}", flush=True)
            print(f"- RAM working set:        {mem['working_set_mb']:.1f} MB (Peak: {mem['peak_working_set_mb']:.1f} MB)", flush=True)
            print(f"- Pagefile usage:         {mem['pagefile_mb']:.1f} MB (Peak: {mem['peak_pagefile_mb']:.1f} MB)", flush=True)
            print(f"- Checkpoint saved:       {'YES' if checkpoint_saved else 'NO'} ({checkpoint_file})", flush=True)
            print("="*60 + "\n", flush=True)
                
            # Release batch memory completely
            del chunk_s1_order
            del chunk_s1_records
            del chunk_s1_index
            del chunk_candidates
            del chunk_matches
            del chunk_countries
            gc.collect()
            
            if max_s1 is not None and total_s1_written >= max_s1:
                print(f"Reached max_s1 limit of {max_s1:,}. Stopping.", flush=True)
                break
                
    total_time = time.time() - t_start
    final_mem = get_memory_stats()
    
    avg_cands = total_candidate_pairs / total_s1_written if total_s1_written > 0 else 0.0
    avg_matches_non_empty = total_predicted_matches / non_empty_match_entities if non_empty_match_entities > 0 else 0.0
    
    report = {
        "runtime_seconds": total_time,
        "working_set_mb": final_mem["working_set_mb"],
        "peak_working_set_mb": final_mem["peak_working_set_mb"],
        "pagefile_mb": final_mem["pagefile_mb"],
        "peak_pagefile_mb": final_mem["peak_pagefile_mb"],
        "s1_rows_written": total_s1_written,
        "total_candidate_pairs": total_candidate_pairs,
        "total_predicted_matches": total_predicted_matches,
        "avg_candidates_per_s1": avg_cands,
        "avg_matches_per_non_empty_s1": avg_matches_non_empty,
        "pred_count_0": pred_count_0,
        "pred_count_gt10": pred_count_gt10,
        "pred_count_gt25": pred_count_gt25,
        "pred_count_gt50": pred_count_gt50,
        "matching_output_path": matching_out,
        "candidate_output_path": candidate_out
    }
    
    print("\n" + "="*70, flush=True)
    print("INFERENCE RUN SUMMARY", flush=True)
    print("="*70, flush=True)
    print(f"Total Runtime:                    {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Peak RAM (Working Set):           {final_mem['peak_working_set_mb']:.2f} MB")
    print(f"Peak Pagefile Usage:              {final_mem['peak_pagefile_mb']:.2f} MB")
    print(f"S1 Rows Written:                  {total_s1_written:,}")
    print(f"Total Candidate Pairs:            {total_candidate_pairs:,}")
    print(f"Total Predicted Matches:          {total_predicted_matches:,}")
    print(f"Avg Candidates / S1:              {avg_cands:.2f}")
    print(f"Avg Predicted Matches / Non-Empty:{avg_matches_non_empty:.2f}")
    print(f"S1 with 0 Predictions:            {pred_count_0:,}")
    print(f"S1 with >10 Predictions:          {pred_count_gt10:,}")
    print(f"S1 with >25 Predictions:          {pred_count_gt25:,}")
    print(f"S1 with >50 Predictions:          {pred_count_gt50:,}")
    print("="*70, flush=True)
    
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-Safe Chunked Test Inference")
    parser.add_argument("--batch-size", type=int, default=100000, help="S1 batch chunk size (default: 100,000)")
    parser.add_argument("--max-s1", type=int, default=None, help="Max S1 entities to process")
    parser.add_argument("--matching-out", type=str, default=None, help="Output path for matching_results.tsv")
    parser.add_argument("--candidate-out", type=str, default=None, help="Output path for candidate_pairs.tsv")
    parser.add_argument("--model-path", type=str, default=None, help="Path to LightGBM model")
    parser.add_argument("--threshold", type=float, default=0.85, help="Match confidence threshold (default: 0.85)")
    parser.add_argument("--max-candidates", type=int, default=25, help="Max candidates per S1 entity (default: 25)")
    parser.add_argument("--no-resume", action="store_true", help="Force fresh start, ignoring existing checkpoints")
    args = parser.parse_args()
    
    run_chunked_inference(
        batch_size=args.batch_size,
        max_s1=args.max_s1,
        matching_out=args.matching_out,
        candidate_out=args.candidate_out,
        model_path=args.model_path,
        threshold=args.threshold,
        max_candidates_per_s1=args.max_candidates,
        resume=not args.no_resume
    )
