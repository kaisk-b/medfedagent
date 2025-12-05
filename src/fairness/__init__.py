"""
MedFedAgent Fairness Module

Provides fairness metrics and evaluation tools for federated learning,
including demographic parity, equalized odds, and per-hospital performance analysis.
"""

from .metrics import (
    FairnessMetrics,
    demographic_parity_difference,
    equalized_odds_difference,
    disparate_impact_ratio,
    accuracy_parity,
    calculate_group_metrics
)

from .evaluator import FairnessEvaluator

__all__ = [
    'FairnessMetrics',
    'FairnessEvaluator',
    'demographic_parity_difference',
    'equalized_odds_difference',
    'disparate_impact_ratio',
    'accuracy_parity',
    'calculate_group_metrics'
]
