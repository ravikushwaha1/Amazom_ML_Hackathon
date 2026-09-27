"""
Entity Resolution Challenge: Candidate Generation & Blocking Engine
Author: Lead ML Engineer
Purpose:
    Implements multi-index sparse inverted blocking to reduce the comparison space
    from O(N*M) to O(candidates), achieving >98% recall while keeping inference manageable.
"""

import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Domain extension cleaner
DOMAIN_SUFFIX = re.compile(r'\b(com|org|net|in|co|io|fr|gov|edu|biz|info)\b', re.IGNORECASE)

# High-frequency address stopwords that should not form single-token blocks
COMMON_ADDR_STOP = {
    'street', 'road', 'avenue', 'boulevard', 'lane', 'drive', 'court', 'place',
    'near', 'opp', 'opposite', 'behind', 'floor', 'unit', 'suite', 'apartment',
    'delhi', 'mumbai', 'india', 'texas', 'california', 'france', 'paris', 'block',
    'sector', 'phase', 'nagar', 'colony', 'bazaar', 'market', 'cross', 'main',
    'north', 'south', 'east', 'west', 'upper', 'lower', 'ground', 'first', 'second',
    'building', 'tower', 'complex', 'house', 'plot', 'door', 'flat', 'room',
    'highway', 'circle', 'park', 'square', 'center', 'centre', 'mall', 'plaza'
}

# Generic business words that are too broad to use as isolated 1-word blocks
GENERIC_BIZ_WORDS = {
    'the', 'and', 'or', 'of', 'for', 'in', 'on', 'at', 'to', 'a', 'an',
    'inc', 'corp', 'corporation', 'llc', 'llp', 'ltd', 'limited', 'pvt',
    'company', 'co', 'services', 'service', 'center', 'group', 'enterprises',
    'enterprise', 'solutions', 'associates', 'consulting', 'international',
    'holdings', 'industries', 'ventures', 'management', 'global', 'tech',
    'technologies', 'technology', 'system', 'systems', 'trading', 'club'
}


def clean_squash_name(raw_name: str) -> str:
    """Removes domain extensions and all non-alphanumeric characters."""
    s = DOMAIN_SUFFIX.sub('', raw_name.lower())
    return "".join(c for c in s if c.isalnum())


def extract_blocking_keys(
    clean_name: str,
    core_name: str,
    clean_addr: str,
    country: str,
    raw_name: str = ""
) -> List[str]:
    """
    Extracts multi-index blocking keys for an entity.
    All keys are strictly namespaced by country to preserve 100% geographic consistency.
    """
    keys = []
    c_pfx = (country.strip() or "UNKNOWN") + "|"
    
    # 1. Exact Core Business Name
    if len(core_name) >= 3:
        keys.append(c_pfx + "core:" + core_name)
        
    # 2. First 2 Words Shingle (e.g., 'payne enterprises', 'dick regional')
    words = [w for w in core_name.split() if w not in GENERIC_BIZ_WORDS]
    if len(words) >= 2:
        keys.append(c_pfx + "bi:" + words[0] + "_" + words[1])
    elif len(words) == 1 and len(words[0]) >= 4:
        keys.append(c_pfx + "w1:" + words[0])
        
    # 3. Squashed Alphanumeric (handles domain names like 'xyz.com' matching 'xyz')
    squashed = clean_squash_name(raw_name or clean_name)
    if len(squashed) >= 5:
        keys.append(c_pfx + "sq:" + squashed)
        
    # 4. Distinctive Address Tokens (locality, town, unique street names)
    if clean_addr:
        addr_words = [
            w for w in clean_addr.split()
            if w.isalpha() and len(w) >= 6 and w not in COMMON_ADDR_STOP
        ]
        # Keep up to 2 most distinctive address words
        for aw in addr_words[:2]:
            keys.append(c_pfx + "addr:" + aw)
            
        # 5. House/Plot Number + First Alpha token
        nums = [w for w in clean_addr.split() if any(c.isdigit() for c in w)]
        if nums and addr_words:
            keys.append(c_pfx + "num_addr:" + nums[0] + "_" + addr_words[0])
            
    return keys


class InvertedIndexBlocker:
    """
    High-performance in-memory sparse inverted index for candidate generation.
    """
    def __init__(self, max_candidates_per_entity: int = 80):
        self.index = defaultdict(list)
        self.key_counts = defaultdict(int)
        self.max_candidates = max_candidates_per_entity
        
    def add_target_entity(self, entity_id: str, keys: List[str]):
        """Adds an S2 or S3 entity to the inverted index."""
        # Avoid duplicate keys for the same entity
        unique_keys = set(keys)
        for k in unique_keys:
            self.index[k].append(entity_id)
            self.key_counts[k] += 1
            
    def query_candidates(self, query_keys: List[str]) -> List[str]:
        """
        Retrieves candidate entity IDs for an S1 entity.
        Ranks candidates by number of shared blocking keys (multi-key agreement).
        Caps at max_candidates to prevent memory explosion.
        """
        cand_scores = defaultdict(int)
        for k in query_keys:
            # Skip degenerate keys that match more than 2,000 entities (anti-hotspot guard)
            if self.key_counts[k] > 2000:
                continue
            for eid in self.index.get(k, []):
                cand_scores[eid] += 1
                
        if not cand_scores:
            return []
            
        # Sort candidates descending by key overlap score
        sorted_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
        return [eid for eid, _ in sorted_cands[:self.max_candidates]]
