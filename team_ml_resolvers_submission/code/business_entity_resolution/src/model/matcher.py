"""
Entity Resolution Challenge: Matcher Model & Prediction Engine
Author: Lead ML Engineer
Purpose:
    LightGBM-based pairwise entity matching classifier trained to optimize Macro F0.5.
    Provides inference, probability scoring, and thresholded prediction.
"""

import os
import numpy as np
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except Exception as _e:
    lgb = None
    HAS_LIGHTGBM = False

from typing import List, Dict, Tuple, Set

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "lgb_model.txt")
OPTIMAL_THRESHOLD = 0.85


class EntityMatcher:
    """
    Trained pairwise entity matching classifier using native LightGBM booster.
    Strictly requires native LightGBM; fails loudly if unavailable.
    """
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, threshold: float = OPTIMAL_THRESHOLD):
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        if os.path.isfile(model_path):
            self.load(model_path)
            
    def load(self, model_path: str):
        """Loads a LightGBM booster model from file. Fails loudly if LightGBM is not available."""
        self.model_path = model_path
        if not HAS_LIGHTGBM or lgb is None:
            raise RuntimeError(
                "CRITICAL: Native LightGBM is required but could not be imported! "
                "Silent fallback is prohibited."
            )
        try:
            self.model = lgb.Booster(model_file=model_path)
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Failed to load LightGBM model from '{model_path}': {e}") from e
            
    def save(self, model_path: str):
        """Saves booster model to file."""
        if self.model:
            os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
            self.model.save_model(model_path)
            
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts match probabilities using native LightGBM booster."""
        if self.model is None:
            raise RuntimeError("CRITICAL: LightGBM model has not been loaded.")
        if len(X) == 0:
            return np.array([], dtype=np.float32)
        return self.model.predict(X).astype(np.float32)
        
    def filter_candidates(
        self,
        candidate_pairs: List[Tuple[str, str]],
        features: np.ndarray,
        threshold: float = None
    ) -> Dict[str, List[str]]:
        """
        Filters candidates above the decision threshold.
        Returns mapping: {s1_id: [matched_entity_ids]} sorted by confidence.
        """
        th = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(features)
        
        matches_by_s1 = {}
        for (s1_id, cand_id), prob in zip(candidate_pairs, probs):
            if prob >= th:
                if s1_id not in matches_by_s1:
                    matches_by_s1[s1_id] = []
                matches_by_s1[s1_id].append((cand_id, float(prob)))
                
        # Sort by probability descending and extract IDs
        result = {}
        for s1_id, cand_list in matches_by_s1.items():
            cand_list.sort(key=lambda x: x[1], reverse=True)
            result[s1_id] = [cid for cid, _ in cand_list]
            
        return result
