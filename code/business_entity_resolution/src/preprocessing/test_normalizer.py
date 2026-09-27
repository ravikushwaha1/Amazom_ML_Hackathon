"""
Unit tests for src/preprocessing/normalizer.py
"""

import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.preprocessing.normalizer import (
    normalize_business_name,
    normalize_business_address,
    normalize_country,
    transliterate_devanagari,
    strip_accents
)

def test_transliteration():
    raw = "रेड वेंचर्स प्राइवेट लिमिटेड"
    res = transliterate_devanagari(raw)
    assert "red" in res and "praivet" in res, f"Failed on Devanagari: {res}"
    print("[PASS] test_transliteration")

def test_accent_stripping():
    raw = "Payne Énterprises Léarning S.A.R.L. Àmicale"
    res = strip_accents(raw)
    assert "Enterprises" in res and "Learning" in res and "Amicale" in res
    print("[PASS] test_accent_stripping")

def test_name_normalization():
    # Test case 1: brackets and prefix suffix
    cl, core, suff = normalize_business_name("[Corp] Dick Regional Armada")
    assert core == "dick regional armada"
    assert suff == "corp"
    
    # Test case 2: punctuation and casing
    cl2, core2, suff2 = normalize_business_name("Prem & Sons Pvt Ltd")
    assert "prem and sons" in cl2
    assert "prem and sons" == core2
    assert suff2 in ["pvt ltd", "pvt", "ltd"]
    
    # Test case 3: possessives and noise
    cl3, core3, suff3 = normalize_business_name("<< Orelee's Barbershop >>")
    assert core3 == "orelees barbershop"
    
    print("[PASS] test_name_normalization")

def test_address_normalization():
    raw = "3315 FREMONT ST, <NULL>, PEORIA, IL"
    clean = normalize_business_address(raw)
    assert "<null>" not in clean and "<" not in clean
    assert "3315 fremont street peoria il" == clean
    
    raw_fr = "63 R. DE DIEPPE, LILLE, Hauts-de-France"
    clean_fr = normalize_business_address(raw_fr)
    assert "63 rue de dieppe lille hauts de france" == clean_fr
    print("[PASS] test_address_normalization")

if __name__ == "__main__":
    test_transliteration()
    test_accent_stripping()
    test_name_normalization()
    test_address_normalization()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
