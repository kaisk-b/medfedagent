"""
MedFedAgent Fairness Metrics Module

Implements fairness metrics commonly used in healthcare AI:
- Demographic Parity: Equal positive prediction rates across groups
- Equalized Odds: Equal TPR and FPR across groups  
- Disparate Impact: Ratio of positive prediction rates
- Accuracy Parity: Equal accuracy across groups

Based on IBM AI Fairness 360 methodology and Fair Federated Learning (FFL) research.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import warnings


@dataclass
class FairnessMetrics:
    """Container for all fairness metrics."""
    
    # Demographic Parity
    demographic_parity_difference: float = 0.0
    demographic_parity_ratio: float = 1.0
    
    # Equalized Odds
    equalized_odds_difference: float = 0.0
    tpr_difference: float = 0.0  # True Positive Rate difference
    fpr_difference: float = 0.0  # False Positive Rate difference
    
    # Disparate Impact
    disparate_impact_ratio: float = 1.0
    
    # Accuracy Parity
    accuracy_parity_difference: float = 0.0
    accuracy_by_group: Dict[str, float] = field(default_factory=dict)
    
    # Per-hospital metrics (for federated setting)
    hospital_accuracy_variance: float = 0.0
    hospital_auc_variance: float = 0.0
    hospital_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Group-level statistics
    group_positive_rates: Dict[str, float] = field(default_factory=dict)
    group_tpr: Dict[str, float] = field(default_factory=dict)
    group_fpr: Dict[str, float] = field(default_factory=dict)
    
    # Summary
    is_fair: bool = True
    fairness_score: float = 1.0  # 0-1, higher is fairer
    violations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'demographic_parity_difference': self.demographic_parity_difference,
            'demographic_parity_ratio': self.demographic_parity_ratio,
            'equalized_odds_difference': self.equalized_odds_difference,
            'tpr_difference': self.tpr_difference,
            'fpr_difference': self.fpr_difference,
            'disparate_impact_ratio': self.disparate_impact_ratio,
            'accuracy_parity_difference': self.accuracy_parity_difference,
            'accuracy_by_group': self.accuracy_by_group,
            'hospital_accuracy_variance': self.hospital_accuracy_variance,
            'hospital_auc_variance': self.hospital_auc_variance,
            'hospital_metrics': self.hospital_metrics,
            'group_positive_rates': self.group_positive_rates,
            'group_tpr': self.group_tpr,
            'group_fpr': self.group_fpr,
            'is_fair': self.is_fair,
            'fairness_score': self.fairness_score,
            'violations': self.violations
        }


def demographic_parity_difference(
    predictions: np.ndarray,
    protected_attribute: np.ndarray,
    favorable_label: int = 1
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate Demographic Parity Difference.
    
    Demographic parity requires that the positive prediction rate is the same
    across all protected groups. This is crucial in healthcare to ensure
    equal access to diagnosis/treatment across demographic groups.
    
    DPD = max(P(Y_hat=1|A=a)) - min(P(Y_hat=1|A=a)) for all groups a
    
    Args:
        predictions: Model predictions (0 or 1)
        protected_attribute: Group membership for each sample
        favorable_label: The positive/favorable prediction label
        
    Returns:
        Tuple of (difference, dict of per-group positive rates)
    """
    unique_groups = np.unique(protected_attribute)
    group_rates = {}
    
    for group in unique_groups:
        mask = protected_attribute == group
        if np.sum(mask) > 0:
            rate = np.mean(predictions[mask] == favorable_label)
            group_rates[str(group)] = float(rate)
    
    if len(group_rates) < 2:
        return 0.0, group_rates
    
    rates = list(group_rates.values())
    dpd = max(rates) - min(rates)
    
    return float(dpd), group_rates


def equalized_odds_difference(
    predictions: np.ndarray,
    labels: np.ndarray,
    protected_attribute: np.ndarray,
    favorable_label: int = 1
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Calculate Equalized Odds Difference.
    
    Equalized odds requires equal True Positive Rates (TPR) and 
    False Positive Rates (FPR) across protected groups. This ensures
    the model is equally accurate for all demographic groups.
    
    EOD = max(|TPR_a - TPR_b|, |FPR_a - FPR_b|) for all group pairs (a, b)
    
    Args:
        predictions: Model predictions (0 or 1)
        labels: Ground truth labels
        protected_attribute: Group membership for each sample
        favorable_label: The positive label
        
    Returns:
        Tuple of (EOD, dict of TPR per group, dict of FPR per group)
    """
    unique_groups = np.unique(protected_attribute)
    group_tpr = {}
    group_fpr = {}
    
    for group in unique_groups:
        mask = protected_attribute == group
        group_labels = labels[mask]
        group_preds = predictions[mask]
        
        # True Positive Rate (sensitivity/recall)
        pos_mask = group_labels == favorable_label
        if np.sum(pos_mask) > 0:
            tpr = np.mean(group_preds[pos_mask] == favorable_label)
            group_tpr[str(group)] = float(tpr)
        else:
            group_tpr[str(group)] = 0.0
        
        # False Positive Rate (1 - specificity)
        neg_mask = group_labels != favorable_label
        if np.sum(neg_mask) > 0:
            fpr = np.mean(group_preds[neg_mask] == favorable_label)
            group_fpr[str(group)] = float(fpr)
        else:
            group_fpr[str(group)] = 0.0
    
    if len(group_tpr) < 2:
        return 0.0, group_tpr, group_fpr
    
    tpr_values = list(group_tpr.values())
    fpr_values = list(group_fpr.values())
    
    tpr_diff = max(tpr_values) - min(tpr_values)
    fpr_diff = max(fpr_values) - min(fpr_values)
    eod = max(tpr_diff, fpr_diff)
    
    return float(eod), group_tpr, group_fpr


def disparate_impact_ratio(
    predictions: np.ndarray,
    protected_attribute: np.ndarray,
    favorable_label: int = 1,
    privileged_group: Optional[Any] = None
) -> float:
    """
    Calculate Disparate Impact Ratio.
    
    Disparate Impact measures the ratio of positive prediction rates
    between unprivileged and privileged groups. A ratio < 0.8 or > 1.25
    typically indicates potential discrimination (80% rule).
    
    DIR = P(Y_hat=1|A=unprivileged) / P(Y_hat=1|A=privileged)
    
    Args:
        predictions: Model predictions
        protected_attribute: Group membership
        favorable_label: Positive prediction label
        privileged_group: The privileged group identifier (if None, uses group with highest rate)
        
    Returns:
        Disparate impact ratio
    """
    unique_groups = np.unique(protected_attribute)
    group_rates = {}
    
    for group in unique_groups:
        mask = protected_attribute == group
        if np.sum(mask) > 0:
            rate = np.mean(predictions[mask] == favorable_label)
            group_rates[group] = rate
    
    if len(group_rates) < 2:
        return 1.0
    
    # Identify privileged group (highest positive rate if not specified)
    if privileged_group is None:
        privileged_group = max(group_rates, key=group_rates.get)
    
    privileged_rate = group_rates.get(privileged_group, 0)
    
    if privileged_rate == 0:
        return 1.0
    
    # Calculate ratio for each unprivileged group
    unprivileged_groups = [g for g in group_rates if g != privileged_group]
    if not unprivileged_groups:
        return 1.0
    
    # Return minimum ratio (worst case)
    min_ratio = min(group_rates[g] / privileged_rate for g in unprivileged_groups)
    
    return float(min_ratio)


def accuracy_parity(
    predictions: np.ndarray,
    labels: np.ndarray,
    protected_attribute: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate Accuracy Parity Difference.
    
    Accuracy parity requires equal prediction accuracy across groups.
    This is especially important in healthcare where we want all patient
    groups to receive equally accurate diagnoses.
    
    APD = max(accuracy_a) - min(accuracy_a) for all groups a
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        protected_attribute: Group membership
        
    Returns:
        Tuple of (difference, dict of accuracy per group)
    """
    unique_groups = np.unique(protected_attribute)
    group_accuracy = {}
    
    for group in unique_groups:
        mask = protected_attribute == group
        if np.sum(mask) > 0:
            acc = np.mean(predictions[mask] == labels[mask])
            group_accuracy[str(group)] = float(acc)
    
    if len(group_accuracy) < 2:
        return 0.0, group_accuracy
    
    accuracies = list(group_accuracy.values())
    apd = max(accuracies) - min(accuracies)
    
    return float(apd), group_accuracy


def calculate_group_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    protected_attribute: Optional[np.ndarray] = None,
    hospital_ids: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive group-level metrics.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        probabilities: Prediction probabilities (for AUC)
        protected_attribute: Demographic group membership
        hospital_ids: Hospital/client IDs for federated fairness
        
    Returns:
        Dictionary of all group metrics
    """
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    
    metrics = {}
    
    # Overall metrics
    metrics['overall'] = {
        'accuracy': float(np.mean(predictions == labels)),
        'precision': float(precision_score(labels, predictions, zero_division=0)),
        'recall': float(recall_score(labels, predictions, zero_division=0)),
        'f1': float(f1_score(labels, predictions, zero_division=0))
    }
    
    if probabilities is not None:
        try:
            metrics['overall']['auc'] = float(roc_auc_score(labels, probabilities))
        except ValueError:
            metrics['overall']['auc'] = 0.0
    
    # Per-demographic group metrics
    if protected_attribute is not None:
        metrics['by_demographic'] = {}
        unique_groups = np.unique(protected_attribute)
        
        for group in unique_groups:
            mask = protected_attribute == group
            if np.sum(mask) > 1:
                group_metrics = {
                    'count': int(np.sum(mask)),
                    'accuracy': float(np.mean(predictions[mask] == labels[mask])),
                    'positive_rate': float(np.mean(predictions[mask] == 1)),
                    'true_positive_rate': 0.0,
                    'false_positive_rate': 0.0
                }
                
                # TPR and FPR
                pos_mask = labels[mask] == 1
                neg_mask = labels[mask] == 0
                
                if np.sum(pos_mask) > 0:
                    group_metrics['true_positive_rate'] = float(
                        np.mean(predictions[mask][pos_mask] == 1)
                    )
                if np.sum(neg_mask) > 0:
                    group_metrics['false_positive_rate'] = float(
                        np.mean(predictions[mask][neg_mask] == 1)
                    )
                
                # AUC per group
                if probabilities is not None and len(np.unique(labels[mask])) > 1:
                    try:
                        group_metrics['auc'] = float(
                            roc_auc_score(labels[mask], probabilities[mask])
                        )
                    except ValueError:
                        group_metrics['auc'] = 0.0
                
                metrics['by_demographic'][str(group)] = group_metrics
    
    # Per-hospital metrics (for federated fairness)
    if hospital_ids is not None:
        metrics['by_hospital'] = {}
        unique_hospitals = np.unique(hospital_ids)
        
        for hospital in unique_hospitals:
            mask = hospital_ids == hospital
            if np.sum(mask) > 1:
                hospital_metrics = {
                    'count': int(np.sum(mask)),
                    'accuracy': float(np.mean(predictions[mask] == labels[mask])),
                    'positive_rate': float(np.mean(predictions[mask] == 1)),
                    'label_positive_rate': float(np.mean(labels[mask] == 1))
                }
                
                if probabilities is not None and len(np.unique(labels[mask])) > 1:
                    try:
                        hospital_metrics['auc'] = float(
                            roc_auc_score(labels[mask], probabilities[mask])
                        )
                    except ValueError:
                        hospital_metrics['auc'] = 0.0
                
                metrics['by_hospital'][str(hospital)] = hospital_metrics
    
    return metrics


def compute_fairness_score(metrics: FairnessMetrics, thresholds: Optional[Dict] = None) -> float:
    """
    Compute an overall fairness score (0-1) based on multiple metrics.
    
    Higher score = fairer model.
    
    Args:
        metrics: FairnessMetrics object
        thresholds: Custom thresholds for each metric
        
    Returns:
        Fairness score between 0 and 1
    """
    if thresholds is None:
        thresholds = {
            'demographic_parity': 0.1,  # Max acceptable difference
            'equalized_odds': 0.1,
            'accuracy_parity': 0.1,
            'disparate_impact_min': 0.8,  # 80% rule
            'disparate_impact_max': 1.25
        }
    
    scores = []
    
    # Demographic parity score (1 - normalized difference)
    dp_score = max(0, 1 - metrics.demographic_parity_difference / thresholds['demographic_parity'])
    scores.append(min(1, dp_score))
    
    # Equalized odds score
    eo_score = max(0, 1 - metrics.equalized_odds_difference / thresholds['equalized_odds'])
    scores.append(min(1, eo_score))
    
    # Accuracy parity score
    ap_score = max(0, 1 - metrics.accuracy_parity_difference / thresholds['accuracy_parity'])
    scores.append(min(1, ap_score))
    
    # Disparate impact score (penalize if outside [0.8, 1.25])
    di = metrics.disparate_impact_ratio
    if thresholds['disparate_impact_min'] <= di <= thresholds['disparate_impact_max']:
        di_score = 1.0
    elif di < thresholds['disparate_impact_min']:
        di_score = di / thresholds['disparate_impact_min']
    else:
        di_score = thresholds['disparate_impact_max'] / di
    scores.append(max(0, min(1, di_score)))
    
    # Average all scores
    return float(np.mean(scores))


def check_fairness_violations(
    metrics: FairnessMetrics,
    thresholds: Optional[Dict] = None
) -> List[str]:
    """
    Check for fairness violations based on thresholds.
    
    Args:
        metrics: FairnessMetrics object
        thresholds: Custom thresholds
        
    Returns:
        List of violation messages
    """
    if thresholds is None:
        thresholds = {
            'demographic_parity': 0.1,
            'equalized_odds': 0.1,
            'accuracy_parity': 0.15,
            'disparate_impact_min': 0.8,
            'hospital_variance': 0.05
        }
    
    violations = []
    
    if metrics.demographic_parity_difference > thresholds['demographic_parity']:
        violations.append(
            f"Demographic Parity violated: {metrics.demographic_parity_difference:.3f} > {thresholds['demographic_parity']}"
        )
    
    if metrics.equalized_odds_difference > thresholds['equalized_odds']:
        violations.append(
            f"Equalized Odds violated: {metrics.equalized_odds_difference:.3f} > {thresholds['equalized_odds']}"
        )
    
    if metrics.accuracy_parity_difference > thresholds['accuracy_parity']:
        violations.append(
            f"Accuracy Parity violated: {metrics.accuracy_parity_difference:.3f} > {thresholds['accuracy_parity']}"
        )
    
    if metrics.disparate_impact_ratio < thresholds['disparate_impact_min']:
        violations.append(
            f"Disparate Impact violated: {metrics.disparate_impact_ratio:.3f} < {thresholds['disparate_impact_min']} (80% rule)"
        )
    
    if metrics.hospital_accuracy_variance > thresholds['hospital_variance']:
        violations.append(
            f"Hospital Performance Parity violated: variance {metrics.hospital_accuracy_variance:.4f} > {thresholds['hospital_variance']}"
        )
    
    return violations
