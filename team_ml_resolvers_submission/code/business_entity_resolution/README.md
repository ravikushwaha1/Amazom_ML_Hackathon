# Business Entity Resolution Pipeline

This repository contains the complete, reproducible, end-to-end entity resolution pipeline for the Amazon ML Challenge 2026.

## System Overview

The pipeline resolves records from Source 2 and Source 3 against deduplicated reference entities in Source 1:
1. **Preprocessing & Normalization (`src/preprocessing/`):** Script-aware text normalization, Unicode diacritic removal, Devanagari transliteration, legal suffix isolation, and address standardization.
2. **Candidate Generation / Blocking (`src/blocking/`):** Multi-index sparse inverted index keyed on country, core names, 2-word shingles, squashed domains, and address locality tokens. Achieves >98% recall while pruning >99.99% of Cartesian space.
3. **Feature Extraction (`src/features/`):** 16-dimensional numerical similarity vector per candidate pair (Jaccard, 3-gram Dice, length ratios, house number indicators, suffix agreement).
4. **Matching & Inference (`src/model/`):** LightGBM Gradient Boosted Decision Tree (874 KB, MIT license) calibrated at the optimal macro F0.5 decision threshold ($T = 0.85$).
5. **Evaluation (`src/evaluation/`):** Official competition Macro F0.5 scorer with strict singleton accounting.
6. **Inference Pipeline (`src/prediction/`):** High-throughput, country-partitioned streaming test engine generating both required TSVs.

## Directory Structure

```
business_entity_resolution/
├── src/
│   ├── preprocessing/
│   │   ├── normalizer.py             # Normalization and transliteration
│   │   ├── clean_data.py             # CLI batch cleaner
│   │   └── test_normalizer.py        # Normalizer unit tests
│   ├── blocking/
│   │   ├── blocker.py                # Inverted index candidate generation
│   │   └── test_blocker.py           # Blocker unit tests
│   ├── features/
│   │   └── feature_extractor.py      # Pairwise feature extraction
│   ├── model/
│   │   ├── matcher.py                # EntityMatcher inference class
│   │   ├── lgb_model.txt             # Pretrained LightGBM model weights
│   │   └── train_production_model.py # Retraining script
│   ├── evaluation/
│   │   └── metrics.py                # Official competition Macro F0.5 metric
│   └── prediction/
│       └── predict_test.py           # Full test prediction pipeline
├── README.md
└── requirements.txt
```

## Setup Instructions

Ensure Python 3.9+ is installed. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

### 1. Generate Test Submission Outputs
To run the complete test pipeline on `dataset/test/` and produce both `output/matching_results.tsv` and `output/candidate_pairs.tsv`:

```bash
python src/prediction/predict_test.py
```

This will automatically:
- Load all 1,732,544 test Source 1 entities and generate candidates.
- Compute pairwise features and run model predictions at threshold $0.85$.
- Verify that $\text{FINAL\_MATCHES} \subseteq \text{CANDIDATE\_PAIRS}$.
- Run the official validator script (`utils/validate_submission.py`).

### 2. Retrain the Model (Optional)
To retrain the LightGBM booster from the training dataset:

```bash
python src/model/train_production_model.py
```

### 3. Run Unit Tests
To verify all normalization and blocking components:

```bash
python src/preprocessing/test_normalizer.py
python src/blocking/test_blocker.py
```
