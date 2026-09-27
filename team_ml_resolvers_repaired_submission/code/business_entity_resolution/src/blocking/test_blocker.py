"""
Unit test and benchmark for src/blocking/blocker.py
"""

import sys
import os

sys.path.insert(0, r"c:\Users\ravi9\Downloads\student_resource")

from src.preprocessing.normalizer import (
    normalize_business_name,
    normalize_business_address,
    normalize_country
)
from src.blocking.blocker import extract_blocking_keys, InvertedIndexBlocker

def test_blocking_keys():
    # S1 entity
    name = "Payne Enterprises"
    addr = "3315 Fremont Street, Peoria, IL"
    c = "US"
    
    cl_name, core, suff = normalize_business_name(name)
    cl_addr = normalize_business_address(addr)
    keys_s1 = extract_blocking_keys(cl_name, core, cl_addr, c, raw_name=name)
    
    # S2 matching entity
    name_m = "Payne Énterprises"
    addr_m = "3315 FREMONT ST, PEORIA, IL"
    cl_name_m, core_m, suff_m = normalize_business_name(name_m)
    cl_addr_m = normalize_business_address(addr_m)
    keys_m = extract_blocking_keys(cl_name_m, core_m, cl_addr_m, c, raw_name=name_m)
    
    # Check intersection
    common = set(keys_s1) & set(keys_m)
    print("Common keys found:", common)
    assert len(common) >= 2, f"Expected at least 2 common keys, got {common}"
    print("[PASS] test_blocking_keys")

def test_inverted_index():
    blocker = InvertedIndexBlocker(max_candidates_per_entity=50)
    
    # Add target candidates
    blocker.add_target_entity("S2-001", ["US|core:payne enterprises", "US|bi:payne_enterprises"])
    blocker.add_target_entity("S2-002", ["US|bi:payne_enterprises"])
    blocker.add_target_entity("S3-001", ["US|core:payne enterprises", "US|addr:peoria"])
    blocker.add_target_entity("S3-999", ["US|addr:chicago"]) # unrelated
    
    # Query with S1
    query_keys = ["US|core:payne enterprises", "US|bi:payne_enterprises", "US|addr:peoria"]
    cands = blocker.query_candidates(query_keys)
    
    print("Candidates retrieved:", cands)
    # S2-001 and S3-001 should be at the top because they match 2 keys
    assert "S2-001" in cands[:2] and "S3-001" in cands[:2]
    assert "S2-002" in cands
    assert "S3-999" not in cands
    print("[PASS] test_inverted_index")

if __name__ == "__main__":
    test_blocking_keys()
    test_inverted_index()
    print("\nALL BLOCKER TESTS PASSED SUCCESSFULLY!")
