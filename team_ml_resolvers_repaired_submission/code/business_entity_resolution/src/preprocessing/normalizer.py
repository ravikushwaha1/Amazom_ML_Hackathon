"""
Entity Resolution Challenge: Text Normalizer & Cleaner
Author: Lead ML Engineer
Purpose:
    Provides deterministic, safe, and symmetric cleaning and normalization functions
    for business names and addresses across English, Indian, and French datasets.
"""

import re
import unicodedata
from typing import Tuple

# Framing punctuation noise: <<, >>, --, [[, ]], etc.
LEADING_TRAILING_PUNCT = re.compile(r'^[\s<>\-–—\[\]\(\)\{\}\*#_.,;:"\'`]+|[\s<>\-–—\[\]\(\)\{\}\*#_.,;:"\'`]+$')
MULTIPLE_SPACES = re.compile(r'\s+')
NULL_TOKENS = re.compile(r'\b(?:null|<null>|none|nil|undefined|na|n/a)\b', re.IGNORECASE)
PUNCT_SPLIT = re.compile(r'[\-_/\\,;:|~*#^$@!?"`\(\)\[\]\{\}<>]+')
APOSTROPHES = re.compile(r"['’]")
AMPERSAND = re.compile(r'&')

# Deterministic Devanagari (Hindi) phonetic transliteration mapping
DEV_MAP = {
    # Vowels
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ऋ': 'ri',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'अं': 'am', 'अः': 'ah',
    'ऑ': 'o', 'ऍ': 'e',
    # Matras (vowel signs)
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'ृ': 'ri',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ं': 'n', 'ँ': 'n', 'ः': 'h',
    'ॅ': 'e', 'ॉ': 'o',
    # Consonants
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh',
    'ष': 'sh', 'स': 's', 'ह': 'h',
    # Nukta consonants
    'क़': 'q', 'ख़': 'kh', 'ग़': 'g', 'ज़': 'z', 'ड़': 'r', 'ढ़': 'rh', 'फ़': 'f',
    # Virama (halant - drops inherent vowel)
    '्': '',
    # Digits
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4', '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
}

# Standard legal suffixes to recognize (US, India, France)
LEGAL_SUFFIXES = [
    # India / Commonwealth
    r'\bpvt\.?\s*ltd\.?',
    r'\bprivate\s+limited\b',
    r'\bpublic\s+limited\b',
    r'\bpvt\b',
    r'\bltd\.?',
    r'\blimited\b',
    r'\bllp\b',
    # US
    r'\bincorporated\b',
    r'\binc\.?',
    r'\bcorporation\b',
    r'\bcorp\.?',
    r'\bllc\b',
    r'\bl\.l\.c\b',
    r'\bco\.?\b',
    r'\bcompany\b',
    # France
    r'\bsarl\b',
    r'\bs\.a\.r\.l\.?\b',
    r'\bsas\b',
    r'\bs\.a\.s\.?\b',
    r'\bsci\b',
    r'\bs\.c\.i\.?\b',
    r'\bsa\b',
    r'\bs\.a\.?\b',
    r'\beurl\b',
    r'\bsnc\b',
]
LEGAL_RE = re.compile(r'|'.join(LEGAL_SUFFIXES), re.IGNORECASE)

# Standard address abbreviation expansions
ADDRESS_ABBR = [
    (re.compile(r'\bst\b', re.IGNORECASE), 'street'),
    (re.compile(r'\brd\b', re.IGNORECASE), 'road'),
    (re.compile(r'\bave\b', re.IGNORECASE), 'avenue'),
    (re.compile(r'\bav\b', re.IGNORECASE), 'avenue'),
    (re.compile(r'\bdr\b', re.IGNORECASE), 'drive'),
    (re.compile(r'\bblvd\b', re.IGNORECASE), 'boulevard'),
    (re.compile(r'\bbvd\b', re.IGNORECASE), 'boulevard'),
    (re.compile(r'\bln\b', re.IGNORECASE), 'lane'),
    (re.compile(r'\bct\b', re.IGNORECASE), 'court'),
    (re.compile(r'\bpl\b', re.IGNORECASE), 'place'),
    (re.compile(r'\bpkwy\b', re.IGNORECASE), 'parkway'),
    (re.compile(r'\bhwy\b', re.IGNORECASE), 'highway'),
    (re.compile(r'\bapt\b', re.IGNORECASE), 'apartment'),
    (re.compile(r'\bste\b', re.IGNORECASE), 'suite'),
    (re.compile(r'\bfl\b', re.IGNORECASE), 'floor'),
    # French address abbreviations
    (re.compile(r'\br\.\s*de\b', re.IGNORECASE), 'rue de'),
    (re.compile(r'\br\.\b', re.IGNORECASE), 'rue'),
    (re.compile(r'\bbd\b', re.IGNORECASE), 'boulevard'),
    (re.compile(r'\bbvd\b', re.IGNORECASE), 'boulevard'),
]


def transliterate_devanagari(text: str) -> str:
    """Deterministically transliterates Devanagari script to Latin characters."""
    res = []
    i = 0
    n = len(text)
    while i < n:
        if i + 1 < n and text[i:i+2] in DEV_MAP:
            res.append(DEV_MAP[text[i:i+2]])
            i += 2
        elif text[i] in DEV_MAP:
            res.append(DEV_MAP[text[i]])
            i += 1
        else:
            res.append(text[i])
            i += 1
    return "".join(res)


def strip_accents(text: str) -> str:
    """Removes combining diacritics via Unicode NFKD (e.g. É -> E, à -> a, ô -> o)."""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def clean_base_text(text: str) -> str:
    """Applies universal safe text normalization without over-cleaning."""
    if not text:
        return ""
    
    # 1. Transliterate non-Latin scripts if present
    if not text.isascii():
        text = transliterate_devanagari(text)
        text = strip_accents(text)
        
    # 2. Lowercase
    text = text.lower()
    
    # 3. Standardize ampersand
    text = AMPERSAND.sub(' and ', text)
    
    # 4. Remove leading/trailing brackets, dashes, quotes
    text = LEADING_TRAILING_PUNCT.sub('', text)
    
    # 5. Split remaining internal punctuation to whitespace
    text = PUNCT_SPLIT.sub(' ', text)
    
    # 6. Remove possessive apostrophes (Orelee's -> Orelees)
    text = APOSTROPHES.sub('', text)
    
    # 7. Collapse spaces
    text = MULTIPLE_SPACES.sub(' ', text).strip()
    return text


def normalize_business_name(raw_name: str) -> Tuple[str, str, str]:
    """
    Normalizes a business name string.
    Returns:
        cleaned_name: Normalized name with symbols/accents handled.
        core_name: Cleaned name with identified legal suffixes removed.
        legal_suffix: Identified legal entity suffix (or empty string).
    """
    if not raw_name:
        return "", "", ""
    cleaned = clean_base_text(raw_name)
    matches = LEGAL_RE.findall(cleaned)
    legal_suffix = matches[0].strip() if matches else ""
    core = LEGAL_RE.sub('', cleaned)
    core = MULTIPLE_SPACES.sub(' ', core).strip()
    # If stripping suffix emptied the string, retain original cleaned
    if not core:
        core = cleaned
    return cleaned, core, legal_suffix


def normalize_business_address(raw_address: str) -> str:
    """
    Normalizes a business address string:
    - Removes <NULL>, null noise tokens
    - Applies base cleaning
    - Expands standard road/street abbreviations
    """
    if not raw_address:
        return ""
    # Strip literal null tokens first
    addr = NULL_TOKENS.sub(' ', raw_address)
    addr = clean_base_text(addr)
    # Expand address abbreviations
    for pattern, repl in ADDRESS_ABBR:
        addr = pattern.sub(repl, addr)
    addr = MULTIPLE_SPACES.sub(' ', addr).strip()
    return addr


def normalize_country(raw_country: str) -> str:
    """Safe normalization of country string."""
    return raw_country.strip() if raw_country else ""
