"""
Automated Final Validation & Packaging Pipeline for Repaired Run
Author: Antigravity Pair Programmer
Purpose:
    1. Runs the official competition validator on output/repaired_run/
    2. Performs forensic verification on row counts, candidate superset condition,
       and prediction distributions.
    3. Builds team_ml_resolvers_repaired_submission.zip without touching the original submission archive.
"""

import sys
import os
import csv
import zipfile
import subprocess
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

MATCHING_FILE = os.path.join(PROJECT_ROOT, "output", "repaired_run", "matching_results.tsv")
CANDIDATE_FILE = os.path.join(PROJECT_ROOT, "output", "repaired_run", "candidate_pairs.tsv")
TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")
ZIP_NAME = "team_ml_resolvers_repaired_submission.zip"
ZIP_PATH = os.path.join(PROJECT_ROOT, ZIP_NAME)

def main():
    print("=" * 80)
    print("STEP 1: RUN OFFICIAL VALIDATOR")
    print("=" * 80)
    
    val_script = os.path.join(PROJECT_ROOT, "utils", "validate_submission.py")
    cmd = [
        sys.executable,
        val_script,
        "--matching", MATCHING_FILE,
        "--candidate", CANDIDATE_FILE,
        "--test-dir", TEST_DIR
    ]
    
    print("Executing:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
        
    if res.returncode != 0:
        print("CRITICAL: Validator failed! Halting.")
        sys.exit(1)
        
    print("OFFICIAL VALIDATOR: PASSED!")
    
    print("\n" + "=" * 80)
    print("STEP 2: INDEPENDENT FORENSIC VERIFICATION")
    print("=" * 80)
    
    total_m_rows = 0
    total_c_rows = 0
    total_matches = 0
    empty_m = 0
    s2_matches = 0
    s3_matches = 0
    violations = 0
    
    with open(MATCHING_FILE, "r", encoding="utf-8") as fm, \
         open(CANDIDATE_FILE, "r", encoding="utf-8") as fc:
        rm = csv.reader(fm, delimiter="\t")
        rc = csv.reader(fc, delimiter="\t")
        
        h_m = next(rm)
        h_c = next(rc)
        assert h_m == ["source1_entity_id", "matched_entity_ids"]
        assert h_c == ["source1_entity_id", "candidate_entity_ids"]
        
        for i, (row_m, row_c) in enumerate(zip(rm, rc)):
            total_m_rows += 1
            total_c_rows += 1
            sid_m = row_m[0].strip()
            sid_c = row_c[0].strip()
            assert sid_m == sid_c, f"ID alignment error at row {i+1}: {sid_m} vs {sid_c}"
            
            m_list = [x.strip() for x in row_m[1].split(",") if x.strip()] if len(row_m) > 1 and row_m[1].strip() else []
            c_set = set(x.strip() for x in row_c[1].split(",") if x.strip()) if len(row_c) > 1 and row_c[1].strip() else set()
            
            # Candidate superset check
            for m in m_list:
                if m not in c_set:
                    violations += 1
                if m.startswith("S2"):
                    s2_matches += 1
                elif m.startswith("S3"):
                    s3_matches += 1
                    
            if m_list:
                total_matches += len(m_list)
            else:
                empty_m += 1
                
    print(f"Total S1 Rows:                 {total_m_rows:,} (Expected: 1,732,544)")
    assert total_m_rows == 1732544, f"Row count mismatch: {total_m_rows}"
    print(f"Candidate Superset Violations: {violations}")
    assert violations == 0, f"Found {violations} violations of candidate superset!"
    print(f"Total Predicted Matches:       {total_matches:,}")
    print(f"Non-Empty S1 Entities:         {total_m_rows - empty_m:,} ({(total_m_rows - empty_m)/total_m_rows*100:.2f}%)")
    print(f"Empty S1 Entities:             {empty_m:,} ({empty_m/total_m_rows*100:.2f}%)")
    print(f"Source 2 Matches:              {s2_matches:,} ({s2_matches/total_matches*100:.2f}%)")
    print(f"Source 3 Matches:              {s3_matches:,} ({s3_matches/total_matches*100:.2f}%)")
    print(f"Average Matches / Non-Empty:   {total_matches / (total_m_rows - empty_m):.2f}")
    
    print("\n" + "=" * 80)
    print("STEP 3: PACKAGE REPAIRED SUBMISSION ZIP")
    print("=" * 80)
    
    code_dir = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution")
    doc_path = os.path.join(PROJECT_ROOT, "Documentation_template.md")
    
    print(f"Building {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(MATCHING_FILE, "output/matching_results.tsv")
        print(f"  Added output/matching_results.tsv ({os.path.getsize(MATCHING_FILE):,} bytes)")
        zipf.write(CANDIDATE_FILE, "output/candidate_pairs.tsv")
        print(f"  Added output/candidate_pairs.tsv ({os.path.getsize(CANDIDATE_FILE):,} bytes)")
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
    print(f"\nSUCCESS: Generated {ZIP_PATH} ({zip_size:,} bytes, {zip_size/(1024*1024):.2f} MB)")
    
    print("\n" + "=" * 80)
    print("STEP 4: VERIFY ZIP CONTENTS")
    print("=" * 80)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        names = zf.namelist()
        for n in sorted(names):
            print(f"  - {n}")
        assert "output/matching_results.tsv" in names
        assert "output/candidate_pairs.tsv" in names
        assert "Documentation_template.md" in names
        assert "code/business_entity_resolution/src/model/matcher.py" in names
        
    print("\nALL VERIFICATIONS PASSED! REPAIRED SUBMISSION READY FOR UNSTOP.")

if __name__ == "__main__":
    main()
