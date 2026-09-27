"""
Entity Resolution Challenge: Official Competition Metric (Macro F0.5)
Author: Lead ML Engineer
Purpose:
    Computes macro-averaged F0.5 score across all Source 1 entities according to the official formula.
"""

from typing import Dict, Set, List

def compute_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> float:
    """
    Computes F0.5 for a single Source 1 entity according to official challenge rules:
    - True empty (singleton):
        - Predict empty -> 1.0
        - Predict non-empty -> 0.0
    - True non-empty:
        - Predict empty -> 0.0
        - Predict non-empty -> (1.25 * P * R) / (0.25 * P + R)
    """
    is_true_empty = (len(true_matches) == 0)
    is_pred_empty = (len(pred_matches) == 0)
    
    if is_true_empty:
        return 1.0 if is_pred_empty else 0.0
        
    if is_pred_empty:
        return 0.0
        
    # Both non-empty
    intersection = len(true_matches & pred_matches)
    if intersection == 0:
        return 0.0
        
    precision = intersection / len(pred_matches)
    recall = intersection / len(true_matches)
    
    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0
        
    f05 = (1.25 * precision * recall) / denom
    return f05


def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Dict[str, float]:
    """
    Computes overall Macro F0.5, Precision, Recall, and singleton statistics.
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {"macro_f05": 0.0, "mean_precision": 0.0, "mean_recall": 0.0}
        
    total_f05 = 0.0
    total_prec = 0.0
    total_rec = 0.0
    non_singleton_count = 0
    singleton_count = 0
    singleton_correct = 0
    
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        f05 = compute_entity_f05(true_set, pred_set)
        total_f05 += f05
        
        if len(true_set) == 0:
            singleton_count += 1
            if len(pred_set) == 0:
                singleton_correct += 1
        else:
            non_singleton_count += 1
            if len(pred_set) > 0:
                hits = len(true_set & pred_set)
                total_prec += hits / len(pred_set)
                total_rec += hits / len(true_set)
                
    macro_f05 = total_f05 / total_entities
    mean_prec = total_prec / max(1, non_singleton_count)
    mean_rec = total_rec / max(1, non_singleton_count)
    singleton_acc = singleton_correct / max(1, singleton_count)
    
    return {
        "macro_f05": round(macro_f05, 5),
        "mean_precision": round(mean_prec, 5),
        "mean_recall": round(mean_rec, 5),
        "singleton_accuracy": round(singleton_acc, 5),
        "total_evaluated": total_entities,
        "singletons": singleton_count,
        "non_singletons": non_singleton_count
    }
