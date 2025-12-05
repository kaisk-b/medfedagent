"""
MedFedAgent Fairness Evaluator

High-level fairness evaluation for federated learning models.
Computes comprehensive fairness metrics across demographic groups and hospitals.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger
import json
import os
from datetime import datetime

from .metrics import (
    FairnessMetrics,
    demographic_parity_difference,
    equalized_odds_difference,
    disparate_impact_ratio,
    accuracy_parity,
    calculate_group_metrics,
    compute_fairness_score,
    check_fairness_violations
)


@dataclass
class FairnessConfig:
    """Configuration for fairness evaluation."""
    
    # Thresholds for fairness violations
    demographic_parity_threshold: float = 0.1
    equalized_odds_threshold: float = 0.1
    accuracy_parity_threshold: float = 0.15
    disparate_impact_min: float = 0.8  # 80% rule
    disparate_impact_max: float = 1.25
    hospital_variance_threshold: float = 0.05
    
    # Demographic groups to track
    demographic_groups: List[str] = None
    
    # Whether to fail on violations
    fail_on_violation: bool = False
    
    def __post_init__(self):
        if self.demographic_groups is None:
            self.demographic_groups = ["age_group", "sex", "hospital"]
    
    def get_thresholds(self) -> Dict:
        return {
            'demographic_parity': self.demographic_parity_threshold,
            'equalized_odds': self.equalized_odds_threshold,
            'accuracy_parity': self.accuracy_parity_threshold,
            'disparate_impact_min': self.disparate_impact_min,
            'disparate_impact_max': self.disparate_impact_max,
            'hospital_variance': self.hospital_variance_threshold
        }


class FairnessEvaluator:
    """
    Evaluator for computing fairness metrics in federated learning.
    
    This evaluator:
    1. Collects predictions across all clients/hospitals
    2. Computes fairness metrics for demographic groups
    3. Tracks fairness over training rounds
    4. Generates fairness reports
    """
    
    def __init__(
        self,
        config: Optional[FairnessConfig] = None,
        device: str = "cpu"
    ):
        """
        Initialize the fairness evaluator.
        
        Args:
            config: Fairness configuration
            device: Computation device
        """
        self.config = config or FairnessConfig()
        self.device = device
        
        # History tracking
        self.round_history: List[Dict] = []
        self.current_round = 0
        
    def evaluate(
        self,
        model: nn.Module,
        data_loaders: Dict[str, DataLoader],
        demographic_data: Optional[Dict[str, np.ndarray]] = None,
        round_num: Optional[int] = None
    ) -> FairnessMetrics:
        """
        Evaluate fairness metrics for a model across multiple data sources.
        
        Args:
            model: PyTorch model to evaluate
            data_loaders: Dict mapping client/hospital ID to DataLoader
            demographic_data: Dict mapping client ID to demographic attributes
            round_num: Current training round
            
        Returns:
            FairnessMetrics object with all computed metrics
        """
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_hospital_ids = []
        all_demographics = []
        
        with torch.no_grad():
            for client_id, loader in data_loaders.items():
                for batch_idx, (data, target) in enumerate(loader):
                    data = data.to(self.device)
                    
                    # Get predictions
                    output = model(data)
                    probs = torch.softmax(output, dim=1)
                    preds = output.argmax(dim=1)
                    
                    all_predictions.extend(preds.cpu().numpy())
                    all_labels.extend(target.numpy())
                    all_probabilities.extend(probs[:, 1].cpu().numpy())  # Prob of positive class
                    all_hospital_ids.extend([client_id] * len(target))
                    
                    # Add demographic data if available
                    if demographic_data and client_id in demographic_data:
                        demo = demographic_data[client_id]
                        start_idx = batch_idx * loader.batch_size
                        end_idx = start_idx + len(target)
                        all_demographics.extend(demo[start_idx:end_idx])
        
        # Convert to numpy
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        probabilities = np.array(all_probabilities)
        hospital_ids = np.array(all_hospital_ids)
        demographics = np.array(all_demographics) if all_demographics else None
        
        # Compute fairness metrics
        metrics = self._compute_metrics(
            predictions=predictions,
            labels=labels,
            probabilities=probabilities,
            hospital_ids=hospital_ids,
            demographics=demographics
        )
        
        # Store in history
        if round_num is not None:
            self.current_round = round_num
            self.round_history.append({
                'round': round_num,
                'metrics': metrics.to_dict(),
                'timestamp': datetime.now().isoformat()
            })
        
        return metrics
    
    def evaluate_from_predictions(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        hospital_ids: Optional[np.ndarray] = None,
        demographics: Optional[np.ndarray] = None,
        round_num: Optional[int] = None
    ) -> FairnessMetrics:
        """
        Evaluate fairness from pre-computed predictions.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            probabilities: Prediction probabilities
            hospital_ids: Hospital/client IDs
            demographics: Demographic group attributes
            round_num: Current training round
            
        Returns:
            FairnessMetrics object
        """
        metrics = self._compute_metrics(
            predictions=predictions,
            labels=labels,
            probabilities=probabilities,
            hospital_ids=hospital_ids,
            demographics=demographics
        )
        
        if round_num is not None:
            self.current_round = round_num
            self.round_history.append({
                'round': round_num,
                'metrics': metrics.to_dict(),
                'timestamp': datetime.now().isoformat()
            })
        
        return metrics
    
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        hospital_ids: Optional[np.ndarray] = None,
        demographics: Optional[np.ndarray] = None
    ) -> FairnessMetrics:
        """Compute all fairness metrics."""
        
        metrics = FairnessMetrics()
        
        # Use hospital_ids as primary protected attribute if no demographics
        protected_attr = demographics if demographics is not None else hospital_ids
        
        if protected_attr is not None:
            # Demographic Parity
            dpd, group_rates = demographic_parity_difference(
                predictions, protected_attr
            )
            metrics.demographic_parity_difference = dpd
            metrics.group_positive_rates = group_rates
            
            if len(group_rates) > 0:
                rates = list(group_rates.values())
                metrics.demographic_parity_ratio = min(rates) / max(rates) if max(rates) > 0 else 1.0
            
            # Equalized Odds
            eod, tpr_dict, fpr_dict = equalized_odds_difference(
                predictions, labels, protected_attr
            )
            metrics.equalized_odds_difference = eod
            metrics.group_tpr = tpr_dict
            metrics.group_fpr = fpr_dict
            
            if len(tpr_dict) > 1:
                metrics.tpr_difference = max(tpr_dict.values()) - min(tpr_dict.values())
            if len(fpr_dict) > 1:
                metrics.fpr_difference = max(fpr_dict.values()) - min(fpr_dict.values())
            
            # Disparate Impact
            metrics.disparate_impact_ratio = disparate_impact_ratio(
                predictions, protected_attr
            )
            
            # Accuracy Parity
            apd, acc_by_group = accuracy_parity(
                predictions, labels, protected_attr
            )
            metrics.accuracy_parity_difference = apd
            metrics.accuracy_by_group = acc_by_group
        
        # Hospital-specific metrics (federated fairness)
        if hospital_ids is not None:
            hospital_accs = []
            hospital_aucs = []
            metrics.hospital_metrics = {}
            
            for hospital in np.unique(hospital_ids):
                mask = hospital_ids == hospital
                if np.sum(mask) > 1:
                    acc = np.mean(predictions[mask] == labels[mask])
                    hospital_accs.append(acc)
                    
                    hospital_data = {
                        'accuracy': float(acc),
                        'count': int(np.sum(mask)),
                        'positive_rate': float(np.mean(predictions[mask] == 1)),
                        'label_rate': float(np.mean(labels[mask] == 1))
                    }
                    
                    # AUC per hospital
                    if probabilities is not None and len(np.unique(labels[mask])) > 1:
                        try:
                            from sklearn.metrics import roc_auc_score
                            auc = roc_auc_score(labels[mask], probabilities[mask])
                            hospital_aucs.append(auc)
                            hospital_data['auc'] = float(auc)
                        except Exception:
                            pass
                    
                    metrics.hospital_metrics[str(hospital)] = hospital_data
            
            if len(hospital_accs) > 1:
                metrics.hospital_accuracy_variance = float(np.var(hospital_accs))
            if len(hospital_aucs) > 1:
                metrics.hospital_auc_variance = float(np.var(hospital_aucs))
        
        # Compute overall fairness score
        metrics.fairness_score = compute_fairness_score(
            metrics, self.config.get_thresholds()
        )
        
        # Check for violations
        metrics.violations = check_fairness_violations(
            metrics, self.config.get_thresholds()
        )
        metrics.is_fair = len(metrics.violations) == 0
        
        return metrics
    
    def evaluate_per_hospital(
        self,
        client_metrics: Dict[int, Dict[str, float]],
        round_num: Optional[int] = None
    ) -> FairnessMetrics:
        """
        Evaluate fairness based on per-hospital metrics from FL training.
        
        This is useful when you already have per-client metrics from
        the federated learning process.
        
        Args:
            client_metrics: Dict mapping client_id to metrics dict
            round_num: Current training round (optional, for history tracking)
            
        Returns:
            FairnessMetrics focused on hospital parity
        """
        metrics = FairnessMetrics()
        
        accuracies = []
        aucs = []
        
        for client_id, client_data in client_metrics.items():
            acc = client_data.get('accuracy', client_data.get('val_accuracy', 0))
            auc = client_data.get('auc', client_data.get('val_auc', 0))
            
            # Handle NaN
            if acc is not None and not np.isnan(acc):
                accuracies.append(acc)
            if auc is not None and not np.isnan(auc):
                aucs.append(auc)
            
            metrics.hospital_metrics[str(client_id)] = {
                'accuracy': float(acc) if acc else 0.0,
                'auc': float(auc) if auc else 0.0
            }
        
        # Calculate hospital parity
        if len(accuracies) > 1:
            metrics.accuracy_parity_difference = max(accuracies) - min(accuracies)
            metrics.hospital_accuracy_variance = float(np.var(accuracies))
            metrics.accuracy_by_group = {
                f"hospital_{i}": acc for i, acc in enumerate(accuracies)
            }
        
        if len(aucs) > 1:
            metrics.hospital_auc_variance = float(np.var(aucs))
        
        # Fairness score based on hospital parity
        if len(accuracies) > 1:
            # Simple scoring: penalize high variance
            variance_penalty = min(1, metrics.hospital_accuracy_variance / self.config.hospital_variance_threshold)
            parity_penalty = min(1, metrics.accuracy_parity_difference / self.config.accuracy_parity_threshold)
            metrics.fairness_score = max(0, 1.0 - (variance_penalty + parity_penalty) / 2)
        
        # Check violations
        if metrics.hospital_accuracy_variance > self.config.hospital_variance_threshold:
            metrics.violations.append(
                f"Hospital accuracy variance too high: {metrics.hospital_accuracy_variance:.4f}"
            )
        if metrics.accuracy_parity_difference > self.config.accuracy_parity_threshold:
            metrics.violations.append(
                f"Hospital accuracy parity violated: {metrics.accuracy_parity_difference:.3f}"
            )
        
        metrics.is_fair = len(metrics.violations) == 0
        
        # Store in history if round_num provided
        if round_num is not None:
            self.current_round = round_num
            self.round_history.append({
                'round': round_num,
                'metrics': metrics.to_dict(),
                'timestamp': datetime.now().isoformat()
            })
        
        return metrics
    
    def get_history(self) -> List[Dict]:
        """Get fairness metrics history."""
        return self.round_history
    
    def get_summary(self) -> Dict:
        """Get summary of fairness evaluation."""
        if not self.round_history:
            return {'message': 'No evaluations performed'}
        
        latest = self.round_history[-1]['metrics']
        
        # Track trends
        fairness_scores = [r['metrics']['fairness_score'] for r in self.round_history]
        
        return {
            'total_rounds': len(self.round_history),
            'latest_fairness_score': latest['fairness_score'],
            'is_fair': latest['is_fair'],
            'violations': latest['violations'],
            'fairness_trend': 'improving' if len(fairness_scores) > 1 and fairness_scores[-1] > fairness_scores[0] else 'stable',
            'avg_fairness_score': float(np.mean(fairness_scores)),
            'hospital_accuracy_variance': latest.get('hospital_accuracy_variance', 0),
            'demographic_parity_difference': latest.get('demographic_parity_difference', 0)
        }
    
    def save_report(self, filepath: str):
        """Save fairness report to JSON file."""
        report = {
            'summary': self.get_summary(),
            'history': self.round_history,
            'config': {
                'demographic_parity_threshold': self.config.demographic_parity_threshold,
                'equalized_odds_threshold': self.config.equalized_odds_threshold,
                'accuracy_parity_threshold': self.config.accuracy_parity_threshold,
                'disparate_impact_min': self.config.disparate_impact_min,
                'hospital_variance_threshold': self.config.hospital_variance_threshold
            },
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Fairness report saved to {filepath}")
    
    def print_report(self):
        """Print a formatted fairness report."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("FAIRNESS EVALUATION REPORT")
        print("="*60)
        
        print(f"\n📊 Overall Fairness Score: {summary.get('latest_fairness_score', 0):.3f}")
        print(f"   Fair: {'✅ Yes' if summary.get('is_fair', False) else '❌ No'}")
        print(f"   Trend: {summary.get('fairness_trend', 'N/A')}")
        
        if summary.get('violations'):
            print("\n⚠️  Violations:")
            for v in summary['violations']:
                print(f"   - {v}")
        
        print(f"\n📈 Metrics:")
        print(f"   Hospital Accuracy Variance: {summary.get('hospital_accuracy_variance', 0):.4f}")
        print(f"   Demographic Parity Diff: {summary.get('demographic_parity_difference', 0):.4f}")
        
        if self.round_history:
            latest = self.round_history[-1]['metrics']
            if latest.get('hospital_metrics'):
                print("\n🏥 Per-Hospital Performance:")
                for hospital, data in latest['hospital_metrics'].items():
                    acc = data.get('accuracy', 0)
                    auc = data.get('auc', 'N/A')
                    print(f"   {hospital}: Acc={acc:.3f}, AUC={auc if isinstance(auc, str) else f'{auc:.3f}'}")
        
        print("="*60 + "\n")


def create_fairness_evaluator(
    config: Optional[Dict] = None,
    device: str = "cpu"
) -> FairnessEvaluator:
    """
    Factory function to create a FairnessEvaluator.
    
    Args:
        config: Configuration dictionary
        device: Computation device
        
    Returns:
        Configured FairnessEvaluator
    """
    if config:
        fairness_config = FairnessConfig(
            demographic_parity_threshold=config.get('demographic_parity_threshold', 0.1),
            equalized_odds_threshold=config.get('equalized_odds_threshold', 0.1),
            accuracy_parity_threshold=config.get('accuracy_parity_threshold', 0.15),
            disparate_impact_min=config.get('disparate_impact_min', 0.8),
            hospital_variance_threshold=config.get('hospital_variance_threshold', 0.05)
        )
    else:
        fairness_config = FairnessConfig()
    
    return FairnessEvaluator(config=fairness_config, device=device)
