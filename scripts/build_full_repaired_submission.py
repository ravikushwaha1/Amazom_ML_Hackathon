"""
Full Submission Builder for Repaired Run
Combines the repaired native LightGBM inference with verified batches,
validates the complete submission with official validator, and builds submission ZIP.
"""

import sys
import os
import csv
import zipfile
import subprocess
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

REPAIRED_DIR = os.path.join(PROJECT_ROOT, "output", "repaired_run")
FINAL_OUT_DIR = os.path.join(PROJECT_ROOT, "output", "repaired_final")
ORIGINAL_MATCHING = os.path.join(PROJECT_ROOT, "output", "matching_results.tsv")
ORIGINAL_CANDIDATE = os.path.join(PROJECT_ROOT, "output", "candidate_pairs.tsv")
TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")
ZIP_NAME = "team_ml_resolvers_repaired_submission.zip"
ZIP_PATH = os.path.join(PROJECT_ROOT, ZIP_NAME)

TOTAL_EXPECTED_S1 = 1732544

def assemble_submission():
    os.makedirs(FINAL_OUT_DIR, exist_ok=True)
    final_matching = os.path.join(FINAL_OUT_DIR, "matching_results.tsv")
    final_candidate = os.path.join(FINAL_OUT_DIR, "candidate_pairs.tsv")
    
    rep_matching = os.path.join(REPAIRED_DIR, "matching_results.tsv")
    rep_candidate = os.path.join(REPAIRED_DIR, "candidate_pairs.tsv")
    
    # Check rows in repaired run
    rep_rows = 0
    if os.path.exists(rep_matching):
        with open(rep_matching, "r", encoding="utf-8") as f:
            next(f, None)
            for _ in f: rep_rows += 1
            
    print(f"Repaired run currently has {rep_rows:,} data rows.")
    
    if rep_rows >= TOTAL_EXPECTED_S1:
        print(f"Full 18 batches completed in {REPAIRED_DIR}! Using directly.")
        target_m = rep_matching
        target_c = rep_candidate
    elif rep_rows >= 500000:
        print(f"Repaired run has {rep_rows:,} rows completed. Synthesizing full submission with verified Batches 11-18...")
        
        # 1. Matching results
        with open(final_matching, "w", encoding="utf-8", newline="") as f_out:
            # Copy header + completed rows from repaired run
            with open(rep_matching, "r", encoding="utf-8") as f_rep:
                header = f_rep.readline()
                f_out.write(header)
                for _ in range(rep_rows):
                    f_out.write(f_rep.readline())
            # Copy remaining rows from original
            with open(ORIGINAL_MATCHING, "r", encoding="utf-8") as f_orig:
                # skip header and completed rows
                for _ in range(rep_rows + 1):
                    f_orig.readline()
                for line in f_orig:
                    f_out.write(line)
                    
        # 2. Candidate pairs
        with open(final_candidate, "w", encoding="utf-8", newline="") as f_out:
            with open(rep_candidate, "r", encoding="utf-8") as f_rep:
                header = f_rep.readline()
                f_out.write(header)
                for _ in range(rep_rows):
                    f_out.write(f_rep.readline())
            with open(ORIGINAL_CANDIDATE, "r", encoding="utf-8") as f_orig:
                for _ in range(rep_rows + 1):
                    f_orig.readline()
                for line in f_orig:
                    f_out.write(line)
                    
        target_m = final_matching
        target_c = final_candidate
        print(f"Synthesized full submission in {FINAL_OUT_DIR} ({rep_rows:,} repaired + {TOTAL_EXPECTED_S1 - rep_rows:,} verified rows).")
    else:
        print(f"Inference is still running ({rep_rows:,} / 500,000 rows). Please wait.")
        return False, None, None

    return True, target_m, target_c

def validate_and_package(target_m, target_c):
    print("\n" + "=" * 80)
    print("RUNNING OFFICIAL CHALLENGE VALIDATOR")
    print("=" * 80)
    val_script = os.path.join(PROJECT_ROOT, "utils", "validate_submission.py")
    cmd = [
        sys.executable,
        val_script,
        "--matching", target_m,
        "--candidate", target_c,
        "--test-dir", TEST_DIR
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
    if res.returncode != 0:
        print("ERROR: Official validator failed!")
        return False
        
    print("OFFICIAL VALIDATOR: 100% PASSED!")
    
    print("\n" + "=" * 80)
    print("INDEPENDENT FORENSIC AUDIT")
    print("=" * 80)
    total_m_rows = 0
    total_matches = 0
    empty_m = 0
    b1_10_matches = 0
    b11_18_matches = 0
    violations = 0
    
    with open(target_m, "r", encoding="utf-8") as fm, \
         open(target_c, "r", encoding="utf-8") as fc:
        rm = csv.reader(fm, delimiter="\t")
        rc = csv.reader(fc, delimiter="\t")
        next(rm)
        next(rc)
        for i, (row_m, row_c) in enumerate(zip(rm, rc)):
            total_m_rows += 1
            m_list = [x.strip() for x in row_m[1].split(",") if x.strip()] if len(row_m) > 1 and row_m[1].strip() else []
            c_set = set(x.strip() for x in row_c[1].split(",") if x.strip()) if len(row_c) > 1 and row_c[1].strip() else set()
            for m in m_list:
                if m not in c_set: violations += 1
            if m_list:
                total_matches += len(m_list)
                if i < 1000000:
                    b1_10_matches += len(m_list)
                else:
                    b11_18_matches += len(m_list)
            else:
                empty_m += 1
                
    print(f"Total S1 Rows:                 {total_m_rows:,}")
    print(f"Candidate Superset Violations: {violations}")
    print(f"Total Predicted Matches:       {total_matches:,}")
    print(f"  - Batches 1–10 Matches:      {b1_10_matches:,} (Previously: 0)")
    print(f"  - Batches 11–18 Matches:     {b11_18_matches:,}")
    print(f"Non-Empty S1 Entities:         {total_m_rows - empty_m:,} ({(total_m_rows - empty_m)/total_m_rows*100:.2f}%)")
    print(f"Empty S1 Entities:             {empty_m:,} ({empty_m/total_m_rows*100:.2f}%)")
    
    assert total_m_rows == TOTAL_EXPECTED_S1
    assert violations == 0
    assert b1_10_matches > 1000000, f"Expected >1M matches in batches 1-10, got {b1_10_matches}"
    
    print("\n" + "=" * 80)
    print("PACKAGING REPAIRED SUBMISSION ZIP")
    print("=" * 80)
    code_dir = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution")
    doc_path = os.path.join(PROJECT_ROOT, "Documentation_template.md")
    
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(target_m, "output/matching_results.tsv")
        print(f"  Added output/matching_results.tsv ({os.path.getsize(target_m):,} bytes)")
        zipf.write(target_c, "output/candidate_pairs.tsv")
        print(f"  Added output/candidate_pairs.tsv ({os.path.getsize(target_c):,} bytes)")
        zipf.write(doc_path, "Documentation_template.md")
        print(f"  Added Documentation_template.md ({os.path.getsize(doc_path):,} bytes)")
        
        for root, dirs, files in os.walk(code_dir):
            dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
            for file in sorted(files):
                if file.endswith(".pyc") or file.endswith(".pyo") or file == ".DS_Store":
                    continue
                full_p = os.path.join(root, file)
                arc_p = os.path.relpath(full_p, PROJECT_ROOT).replace("\\", "/")
                zipf.write(full_p, arc_p)
                print(f"  Added {arc_p}")
                
    zip_size = os.path.getsize(ZIP_PATH)
    print(f"\nSUCCESS: Created {ZIP_PATH} ({zip_size:,} bytes, {zip_size/(1024*1024):.2f} MB)")
    return True

if __name__ == "__main__":
    ready, tm, tc = assemble_submission()
    if ready:
        validate_and_package(tm, tc)
