# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team ML Resolvers  
**Team Members:** Lead ML Engineer  
**Submission Date:** September 2026  

---

## 1. Executive Summary

This solution presents a scalable, highly accurate, and precision-optimized Entity Resolution (ER) pipeline for the Amazon ML Challenge 2026. Addressing the multi-source matching challenge across noisy, heterogenous business records ($S_1$, $S_2$, $S_3$), our system combines:
1. **Symmetric, script-aware text cleaning** (featuring deterministic Devanagari-to-Latin transliteration and Unicode NFKD accent normalization),
2. **Country-partitioned multi-index sparse inverted blocking** that eliminates $>99.99\%$ of the Cartesian space while achieving **98.08% candidate recall**, and
3. An **MIT-licensed LightGBM pairwise matching classifier** trained on hard negative pairs and calibrated at a precision-optimal decision threshold of **$T = 0.85$**, achieving **$0.9069$ Macro $F_{0.5}$** on validation.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across the 26.4 million records revealed key noise patterns and structural properties:
- **No Cross-Country Matches:** Empirical verification across all 7,638,365 ground-truth pairs revealed an exact **0.0000% cross-country mismatch rate**. Country represents an invariant boundary.
- **The "France" Domain Shift:** While training data covers only `US` and `India`, test data introduces `France` (~15% of records). The pipeline was built without hardcoded country sets to seamlessly generalize.
- **Multilingual Script Disparity:** In $S_1$, nearly 100% of Indian business records are Romanized. In contrast, **21.98% of $S_2$ and 18.72% of $S_3$** are written in native Indic scripts (Devanagari, Hindi). Transliteration was required to enable character overlap.
- **Address Noise & Omissions:** Over 3.3% of $S_2$ and $S_3$ records completely lack addresses, and many contain `<NULL>` tokens or transposed components (`City, State, Street`).
- **Singleton Distribution:** Exactly **5.58%** of $S_1$ entities are singletons (0 matches) across both US and India. Under macro $F_{0.5}$, falsely predicting any match on a singleton results in an immediate score of 0.0, demanding high precision.

### 2.2 Solution Strategy
**Approach Type:** Multi-Index Inverted Blocking + Pairwise GBDT Matching + $F_{0.5}$-Tuned Calibration.  
**Core Innovation:**
1. **Deterministic Script Transliteration & Squash Matching:** Eliminates script mismatch for Indic records and normalizes domain names (e.g., `xyz.com` $\rightarrow$ `xyz`).
2. **Locality Signature Blocking:** Blends street numbers with high-information locality tokens to recover true matches even under heavy name typos or script differences.
3. **Hard-Negative Calibration:** Balances false-candidate mining with true matches to tune decision boundaries specifically for macro $F_{0.5}$.

---

## 3. Candidate Generation (Blocking)

To avoid comparing all $1.73 \times 10^6 \times 9.97 \times 10^6 \approx 1.72 \times 10^{13}$ pairs, we deployed a sparse inverted index blocker operating strictly within country partitions.

- **Blocking Keys Used:**
  1. `country|core:<core_name>`: Exact core business name with legal suffixes isolated.
  2. `country|bi:<token1>_<token2>`: 2-word shingles of the business name.
  3. `country|sq:<squashed>`: Alphanumeric squashed names with web domains stripped.
  4. `country|addr:<locality>`: Distinctive locality/town tokens (length $\ge 6$, non-generic).
  5. `country|num_addr:<house_num>_<street>`: Street number + primary alpha street word.
- **Candidate Pool Control:** Capped at the top 50–80 candidates per $S_1$ entity sorted by multi-key agreement; keys matching $>2,000$ entities are dynamically ignored to eliminate pathological hubs.
- **Candidate Pairs Generated:** Averages ~15 to 40 candidates per $S_1$ entity ($>99.99\%$ reduction ratio).
- **How True Matches Were Preserved:**
  - Evaluated empirically on 17,250 ground-truth pairs:
    - Exact Core Match alone: 41.87% recall
    - 2-Word Shingles: 59.61% recall
    - Squashed Domain Normalization: 93.51% recall
    - Full Multi-Index Engine: **98.08% Candidate Recall**.

---

## 4. Matching Model

### 4.1 Features Used (16 Total)
- **Name Features:**
  - Exact clean name match (`exact_clean_name`)
  - Exact core name match (`exact_core_name`)
  - Exact squashed domain name match (`exact_squash_name`)
  - Token Jaccard similarity (`name_jaccard`)
  - Substring overlap coefficient (`name_overlap`)
  - Character 3-gram Dice similarity (`name_3gram_dice`)
  - Absolute length difference & length ratio (`name_len_diff`, `name_len_ratio`)
  - Legal suffix agreement score (`legal_suffix_match`)
- **Address Features:**
  - Exact clean address match (`exact_clean_addr`)
  - Missing address indicator (`addr_missing`)
  - Address token Jaccard similarity (`addr_jaccard`)
  - Shared address tokens count (`addr_token_overlap`)
  - Address character 3-gram Dice similarity (`addr_3gram_dice`)
  - House number agreement indicator (`house_num_match`, $+1$/$-1$/$0$)
- **Context Features:**
  - Multi-index shared blocking keys count (`shared_keys_count`)

### 4.2 Model Type
- **Algorithm:** LightGBM Gradient Boosted Decision Tree (GBDT).
- **Parameters:** `objective='binary'`, `metric='binary_logloss'`, `learning_rate=0.05`, `num_leaves=31`, `max_depth=6`, `feature_fraction=0.85`.
- **License & Size:** 100% MIT Licensed, file size **874 KB** (sub-1MB memory footprint, compliant with the 8B parameter limit).

### 4.3 Threshold Selection Method
Because the competition metric is Macro $F_{0.5}$ (weighting precision 2× over recall), a threshold sweep $T \in [0.20, 0.90]$ was conducted on an out-of-sample validation split of 2,000 $S_1$ entities:
- Threshold $0.20$: Macro $F_{0.5} = 0.8480$ (Precision $87.43\%$, Recall $85.81\%$, Singleton Acc $71.65\%$)
- Threshold $0.50$: Macro $F_{0.5} = 0.8793$ (Precision $91.12\%$, Recall $85.37\%$, Singleton Acc $79.53\%$)
- **Threshold $0.85$ (Optimal):** **Macro $F_{0.5} = 0.9069$** (Precision **$94.84\%$**, Recall **$82.28\%$**, Singleton Acc **$93.70\%$**)
The elevated threshold of $0.85$ suppresses false merges on singletons and borderline pairs, maximizing the $F_{0.5}$ score.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score:** **0.9069** on held-out validation.
- **Mean Precision:** **94.84%**
- **Mean Recall:** **82.28%**
- **Singleton Accuracy:** **93.70%**
- **Common False Positives (Wrong Merges):**
  - Franchise chains sharing identical brand names and similar city names but different street numbers when street numbers were unparsed.
  - Multi-tenant business complexes where separate companies share an identical building address and generic company words (e.g., "Consulting", "Enterprises").
- **Common False Negatives (Missed Matches):**
  - Records where both name and address were heavily corrupted simultaneously (e.g. extreme phonetic misspelling coupled with a missing address).
  - Heavy script variations in regional languages other than Hindi (e.g., Telugu, Kannada) where only partial phonetic overlap was recovered.

---

## 6. Conclusion

By combining targeted script-aware normalization, high-recall country-partitioned inverted index blocking, and an $F_{0.5}$-calibrated GBDT classifier, this solution delivers an end-to-end reproducible pipeline. The pipeline operates strictly within competition rules, avoids external data, scales smoothly to millions of records, and maximizes precision on business identity resolution.

---

## Appendix

### A. Code Artefacts
All runnable code is located in the submission package under `code/business_entity_resolution/`:
```
code/business_entity_resolution/
├── src/
│   ├── preprocessing/
│   │   ├── normalizer.py           # Core text cleaning & Devanagari transliteration
│   │   └── clean_data.py           # Streaming CLI batch cleaner
│   ├── blocking/
│   │   └── blocker.py              # Multi-index inverted candidate generation engine
│   ├── features/
│   │   └── feature_extractor.py    # 16-feature pairwise extractor
│   ├── model/
│   │   ├── matcher.py              # EntityMatcher inference & thresholding
│   │   ├── lgb_model.txt           # Trained LightGBM booster weights (874 KB)
│   │   └── train_production_model.py # Reproducible training script
│   ├── evaluation/
│   │   └── metrics.py              # Official competition macro F0.5 scorer
│   └── prediction/
│       └── predict_test.py         # Full end-to-end test execution pipeline
├── README.md                       # Complete step-by-step reproduction instructions
└── requirements.txt                # Pinned dependencies (numpy, lightgbm)
```

**Reproducing Outputs:**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run end-to-end test inference
python src/prediction/predict_test.py
```
This generates `output/matching_results.tsv` and `output/candidate_pairs.tsv` and validates them with `utils/validate_submission.py`.

### B. Additional Results
- **Cross-Country Verification:** 0 cross-country matches across 7,638,365 pairs ($0.0000\%$).
- **Blocking Reduction Ratio:** $>99.99\%$ comparison space reduction.
- **Model Parameters:** LightGBM booster with 31 leaves, 6 depth, 250 trees ($<1\text{ MB}$, well within the 8 Billion parameter constraint).
