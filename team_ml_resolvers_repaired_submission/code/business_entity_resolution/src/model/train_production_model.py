import sys
import os
import csv
import time
import random
import numpy as np
import lightgbm as lgb

sys.path.insert(0, r"c:\Users\ravi9\Downloads\student_resource")

from src.preprocessing.normalizer import (
    normalize_business_name,
    normalize_business_address,
    normalize_country
)
from src.blocking.blocker import extract_blocking_keys, InvertedIndexBlocker
from src.features.feature_extractor import extract_pair_features, FEATURE_NAMES, get_char_ngrams

sys.stdout.reconfigure(encoding='utf-8')
random.seed(42)
np.random.seed(42)

TRAIN_S1 = "dataset/train/train_source1.tsv"
TRAIN_S2 = "dataset/train/train_source2.tsv"
TRAIN_S3 = "dataset/train/train_source3.tsv"
TRAIN_GT = "dataset/train/train_ground_truth.tsv"

def main():
    N_TRAIN = 15000
    print(f"Training final production LightGBM model on {N_TRAIN} entities...")
    t0 = time.time()
    
    train_gt = {}
    needed_matches = set()
    needed_s1 = set()
    
    with open(TRAIN_GT, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for i, row in enumerate(reader):
            if i >= N_TRAIN:
                break
            s1_id = row[0].strip()
            matches = set(m.strip() for m in row[1].split(",") if m.strip()) if len(row) > 1 and row[1].strip() else set()
            train_gt[s1_id] = matches
            needed_s1.add(s1_id)
            needed_matches.update(matches)
            
    print(f"Loaded ground truth for {len(needed_s1)} entities ({len(needed_matches)} matches).")
    
    # Load S1
    s1_records = {}
    with open(TRAIN_S1, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            s1_id = row[0].strip()
            if s1_id in needed_s1:
                cl_name, core, suff = normalize_business_name(row[1])
                cl_addr = normalize_business_address(row[2])
                c = normalize_country(row[3])
                addr_toks = set(cl_addr.split())
                nums = [w for w in addr_toks if any(ch.isdigit() for ch in w)]
                s1_records[s1_id] = {
                    "entity_id": s1_id,
                    "clean_name": cl_name,
                    "core_name": core,
                    "legal_suffix": suff,
                    "clean_addr": cl_addr,
                    "country": c,
                    "name_tokens": set(cl_name.split()),
                    "name_3grams": get_char_ngrams(core),
                    "addr_tokens": addr_toks,
                    "addr_3grams": get_char_ngrams(cl_addr),
                    "house_num": nums[0] if nums else "",
                    "blocking_keys": extract_blocking_keys(cl_name, core, cl_addr, c, raw_name=row[1])
                }
                
    # Load targets
    target_records = {}
    blocker = InvertedIndexBlocker(max_candidates_per_entity=50)
    s1_key_set = set()
    for s1 in s1_records.values():
        s1_key_set.update(s1["blocking_keys"])
        
    def scan_target(filepath, max_bg):
        cnt = 0
        loaded = 0
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                cnt += 1
                eid = row[0].strip()
                is_true = (eid in needed_matches)
                if not is_true and cnt > max_bg:
                    continue
                cl_name, core, suff = normalize_business_name(row[1])
                cl_addr = normalize_business_address(row[2])
                c = normalize_country(row[3])
                keys = extract_blocking_keys(cl_name, core, cl_addr, c, raw_name=row[1])
                if is_true or any(k in s1_key_set for k in keys):
                    addr_toks = set(cl_addr.split())
                    nums = [w for w in addr_toks if any(ch.isdigit() for ch in w)]
                    target_records[eid] = {
                        "entity_id": eid,
                        "clean_name": cl_name,
                        "core_name": core,
                        "legal_suffix": suff,
                        "clean_addr": cl_addr,
                        "country": c,
                        "name_tokens": set(cl_name.split()),
                        "name_3grams": get_char_ngrams(core),
                        "addr_tokens": addr_toks,
                        "addr_3grams": get_char_ngrams(cl_addr),
                        "house_num": nums[0] if nums else ""
                    }
                    blocker.add_target_entity(eid, keys)
                    loaded += 1
        print(f"  {filepath}: indexed {loaded:,} records.")

    scan_target(TRAIN_S2, 450000)
    scan_target(TRAIN_S3, 450000)
    
    # Generate balanced training pairs
    X = []
    y = []
    for s1_id, s1 in s1_records.items():
        true_mids = train_gt.get(s1_id, set())
        cands = blocker.query_candidates(s1["blocking_keys"])
        # Positives
        for tm in true_mids:
            if tm in target_records:
                feats = extract_pair_features(s1, target_records[tm])
                X.append(feats)
                y.append(1)
        # Negatives
        neg_cands = [c for c in cands if c not in true_mids and c in target_records]
        num_negs = max(2, min(len(neg_cands), len(true_mids) * 4 if true_mids else 4))
        for neg_id in (random.sample(neg_cands, num_negs) if len(neg_cands) > num_negs else neg_cands):
            feats = extract_pair_features(s1, target_records[neg_id])
            X.append(feats)
            y.append(0)
            
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Dataset: {len(X):,} pairs ({np.sum(y):,} positives, {len(y) - np.sum(y):,} negatives).")
    
    # Train
    train_data = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'verbose': -1,
        'random_state': 42
    }
    
    model = lgb.train(params, train_data, num_boost_round=250)
    save_path = os.path.join(r"c:\Users\ravi9\Downloads\student_resource\src\model", "lgb_model.txt")
    model.save_model(save_path)
    print(f"Successfully trained and saved model to {save_path} in {time.time() - t0:.2f}s.")

if __name__ == "__main__":
    main()
