"""
Entity Resolution Challenge: Reproducible Batch Data Preprocessor
Author: Lead ML Engineer
Purpose:
    Processes raw TSV entity files in a memory-efficient streaming fashion,
    applying safe normalization and saving cleaned outputs separately.
"""

import sys
import os
import csv
import argparse
import time

# Ensure imports work from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.preprocessing.normalizer import (
    normalize_business_name,
    normalize_business_address,
    normalize_country
)

def process_tsv(input_path: str, output_path: str, max_rows: int = None):
    print(f"Starting preprocessing: {input_path} -> {output_path}")
    t0 = time.time()
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    row_count = 0
    with open(input_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(output_path, "w", encoding="utf-8", newline="") as fout:
        
        reader = csv.reader(fin, delimiter="\t")
        writer = csv.writer(fout, delimiter="\t")
        
        header = next(reader, None)
        out_header = [
            "entity_id",
            "business_name_clean",
            "business_name_core",
            "legal_suffix",
            "business_address_clean",
            "country"
        ]
        writer.writerow(out_header)
        
        for row in reader:
            if not row:
                continue
            eid = row[0].strip() if len(row) > 0 else ""
            raw_name = row[1] if len(row) > 1 else ""
            raw_addr = row[2] if len(row) > 2 else ""
            raw_country = row[3] if len(row) > 3 else ""
            
            clean_name, core_name, suffix = normalize_business_name(raw_name)
            clean_addr = normalize_business_address(raw_addr)
            country = normalize_country(raw_country)
            
            writer.writerow([eid, clean_name, core_name, suffix, clean_addr, country])
            row_count += 1
            
            if row_count % 500000 == 0:
                elapsed = time.time() - t0
                print(f"  Processed {row_count:,} rows ({row_count / elapsed:.0f} rows/s)...")
                
            if max_rows and row_count >= max_rows:
                break
                
    elapsed = time.time() - t0
    print(f"Finished {row_count:,} rows in {elapsed:.2f}s ({row_count / max(1e-5, elapsed):.0f} rows/s). Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess and normalize entity TSV files.")
    parser.add_argument("--input", "-i", required=True, help="Path to input raw TSV file")
    parser.add_argument("--output", "-o", required=True, help="Path to output cleaned TSV file")
    parser.add_argument("--max-rows", "-n", type=int, default=None, help="Optional max rows to process (for testing/validation)")
    
    args = parser.parse_args()
    process_tsv(args.input, args.output, args.max_rows)

if __name__ == "__main__":
    main()
