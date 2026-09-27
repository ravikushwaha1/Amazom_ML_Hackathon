"""
Entity Resolution Challenge: Final Submission Zip Builder
Author: Lead ML Engineer
Purpose:
    Packages output/, code/business_entity_resolution/, and Documentation_template.md
    into the official <team_name>_submission.zip archive following the exact competition structure.
"""

import os
import zipfile
import sys

PROJECT_ROOT = r"c:\Users\ravi9\Downloads\student_resource"
ZIP_NAME = "team_ml_resolvers_submission.zip"
ZIP_PATH = os.path.join(PROJECT_ROOT, ZIP_NAME)

def build_zip():
    print(f"Creating submission archive: {ZIP_PATH}...")
    
    # Required files and directories to include
    entries = [
        ("output/matching_results.tsv", "output/matching_results.tsv"),
        ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        ("Documentation_template.md", "Documentation_template.md"),
    ]
    
    # Check that required files exist
    for rel_src, arcname in entries:
        full_src = os.path.join(PROJECT_ROOT, rel_src)
        if not os.path.isfile(full_src):
            print(f"ERROR: Missing required file: {full_src}")
            sys.exit(1)
            
    code_dir = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution")
    if not os.path.isdir(code_dir):
        print(f"ERROR: Missing code directory: {code_dir}")
        sys.exit(1)
        
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add output files and documentation
        for rel_src, arcname in entries:
            full_src = os.path.join(PROJECT_ROOT, rel_src)
            zipf.write(full_src, arcname)
            print(f"  Added {arcname} ({os.path.getsize(full_src):,} bytes)")
            
        # Add code/business_entity_resolution recursively (excluding bytecode and caches)
        for root, dirs, files in os.walk(code_dir):
            dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
            for file in sorted(files):
                if file.endswith(".pyc") or file.endswith(".pyo") or file == ".DS_Store":
                    continue
                full_path = os.path.join(root, file)
                rel_to_code = os.path.relpath(full_path, PROJECT_ROOT).replace("\\", "/")
                zipf.write(full_path, rel_to_code)
                print(f"  Added {rel_to_code}")
                
    print(f"\nSUCCESS: Generated {ZIP_PATH} ({os.path.getsize(ZIP_PATH):,} bytes).")
    
    # Verify contents of zip
    print("\nVerifying archive contents:")
    with zipfile.ZipFile(ZIP_PATH, "r") as zipf:
        names = zipf.namelist()
        for n in sorted(names):
            print(f"  - {n}")
        assert "output/matching_results.tsv" in names
        assert "output/candidate_pairs.tsv" in names
        assert "Documentation_template.md" in names
        assert "code/business_entity_resolution/src/model/matcher.py" in names
        assert "code/business_entity_resolution/README.md" in names
        assert "code/business_entity_resolution/requirements.txt" in names
    print("\nALL ARCHIVE VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    build_zip()
