"""
MedFedAgent Federated Learning Server

Flower-based federated learning server with custom aggregation strategies.
Coordinates training rounds and collects metrics from clients.
"""

import flwr as fl
from flwr.common import (
    Parameters,
    FitRes,
    EvaluateRes,
    Scalar,
    NDArrays,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.server.strategy import FedAvg, Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from typing import Dict, List, Tuple, Optional, Union, Callable
import numpy as np
from collections import OrderedDict
from loguru import logger
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class RoundMetrics:
    """Metrics collected from a training round."""
    round_num: int
    timestamp: str
    num_clients: int
    avg_train_loss: float
    avg_val_loss: float
    avg_accuracy: float
    avg_auc: float
    epsilon_per_client: Dict[int, float] = field(default_factory=dict)
    total_epsilon: float = 0.0
    client_samples: Dict[int, int] = field(default_factory=dict)
    anomalies_detected: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "round": self.round_num,
            "timestamp": self.timestamp,
            "num_clients": self.num_clients,
            "avg_train_loss": self.avg_train_loss,
            "avg_val_loss": self.avg_val_loss,
            "avg_accuracy": self.avg_accuracy,
            "avg_auc": self.avg_auc,
            "epsilon_per_client": self.epsilon_per_client,
            "total_epsilon": self.total_epsilon,
            "client_samples": self.client_samples,
            "anomalies_detected": self.anomalies_detected
        }


class MedFedStrategy(FedAvg):
    """
    Custom federated averaging strategy for MedFedAgent.
    
    Extends FedAvg with:
    - Privacy budget tracking
    - Per-round metrics collection
    - Anomaly detection hooks
    - Configurable aggregation
    """
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable] = None,
        on_fit_config_fn: Optional[Callable] = None,
        on_evaluate_config_fn: Optional[Callable] = None,
        accept_failures: bool = False,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[Callable] = None,
        evaluate_metrics_aggregation_fn: Optional[Callable] = None,
        # Custom parameters
        epsilon_budget: float = 8.0,
        enable_anomaly_detection: bool = True,
        anomaly_threshold: float = 3.0,
        results_dir: str = "./results"
    ):
        """
        Initialize the MedFed strategy.
        
        Args:
            fraction_fit: Fraction of clients for training
            fraction_evaluate: Fraction of clients for evaluation
            min_fit_clients: Minimum clients for training
            min_evaluate_clients: Minimum clients for evaluation  
            min_available_clients: Minimum available clients
            evaluate_fn: Server-side evaluation function
            on_fit_config_fn: Function to configure fit
            on_evaluate_config_fn: Function to configure evaluate
            accept_failures: Whether to accept client failures
            initial_parameters: Initial model parameters
            fit_metrics_aggregation_fn: Custom fit metrics aggregation
            evaluate_metrics_aggregation_fn: Custom evaluate metrics aggregation
            epsilon_budget: Total privacy budget
            enable_anomaly_detection: Whether to detect gradient anomalies
            anomaly_threshold: Z-score threshold for anomaly detection
            results_dir: Directory for results
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
        )
        
        self.epsilon_budget = epsilon_budget
        self.enable_anomaly_detection = enable_anomaly_detection
        self.anomaly_threshold = anomaly_threshold
        self.results_dir = results_dir
        
        # Tracking
        self.round_history: List[RoundMetrics] = []
        self.total_epsilon_spent = 0.0
        self.current_round = 0
        self.client_epsilon_history: Dict[int, List[float]] = {}
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info(f"MedFedStrategy initialized with ε_budget={epsilon_budget}")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results from clients.
        
        Args:
            server_round: Current round number
            results: List of (client, fit_result) tuples
            failures: List of failures
            
        Returns:
            Tuple of (aggregated parameters, aggregated metrics)
        """
        self.current_round = server_round
        
        if not results:
            logger.warning(f"Round {server_round}: No results to aggregate")
            return None, {}
        
        # Extract metrics from each client
        train_losses = []
        epsilons = {}
        samples = {}
        gradient_updates = []
        
        for client_proxy, fit_res in results:
            client_id = int(fit_res.metrics.get("client_id", 0))
            train_loss = fit_res.metrics.get("train_loss", 0.0)
            epsilon = fit_res.metrics.get("epsilon", 0.0)
            num_samples = fit_res.num_examples
            
            train_losses.append(train_loss)
            epsilons[client_id] = epsilon
            samples[client_id] = num_samples
            
            # Track epsilon history
            if client_id not in self.client_epsilon_history:
                self.client_epsilon_history[client_id] = []
            self.client_epsilon_history[client_id].append(epsilon)
            
            # Store gradient update magnitude for anomaly detection
            if fit_res.parameters is not None:
                params = parameters_to_ndarrays(fit_res.parameters)
                update_norm = sum(np.linalg.norm(p) for p in params)
                gradient_updates.append((client_id, update_norm))
        
        # Detect anomalies
        anomalies = []
        if self.enable_anomaly_detection and len(gradient_updates) > 2:
            anomalies = self._detect_anomalies(gradient_updates)
        
        # Update total epsilon (use max across clients for conservative estimate)
        max_epsilon = max(epsilons.values()) if epsilons else 0.0
        self.total_epsilon_spent = max_epsilon
        
        # Perform standard FedAvg aggregation
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Store round metrics (will be updated after evaluation)
        round_metrics = RoundMetrics(
            round_num=server_round,
            timestamp=datetime.now().isoformat(),
            num_clients=len(results),
            avg_train_loss=np.mean(train_losses) if train_losses else 0.0,
            avg_val_loss=0.0,  # Updated after evaluation
            avg_accuracy=0.0,  # Updated after evaluation
            avg_auc=0.0,  # Updated after evaluation
            epsilon_per_client=epsilons,
            total_epsilon=self.total_epsilon_spent,
            client_samples=samples,
            anomalies_detected=anomalies
        )
        self.round_history.append(round_metrics)
        
        # Log progress
        status = self._get_budget_status()
        logger.info(
            f"[round={server_round:02d}] "
            f"loss={round_metrics.avg_train_loss:.4f}, "
            f"ε_total={self.total_epsilon_spent:.2f}, "
            f"clients={len(results)}, "
            f"status={status}"
        )
        
        # Add epsilon tracking to metrics
        if aggregated_metrics is None:
            aggregated_metrics = {}
        aggregated_metrics["epsilon_total"] = self.total_epsilon_spent
        aggregated_metrics["budget_status"] = status
        
        return aggregated_params, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from clients.
        
        Args:
            server_round: Current round number
            results: List of (client, evaluate_result) tuples
            failures: List of failures
            
        Returns:
            Tuple of (aggregated loss, aggregated metrics)
        """
        if not results:
            return None, {}
        
        # Extract metrics
        val_losses = []
        accuracies = []
        aucs = []
        total_samples = 0
        
        for client_proxy, eval_res in results:
            val_losses.append(eval_res.loss * eval_res.num_examples)
            total_samples += eval_res.num_examples
            
            accuracies.append(eval_res.metrics.get("accuracy", 0.0))
            aucs.append(eval_res.metrics.get("auc", 0.0))
        
        # Weighted average loss
        avg_loss = sum(val_losses) / total_samples if total_samples > 0 else 0.0
        avg_accuracy = np.mean(accuracies) if accuracies else 0.0
        avg_auc = np.mean(aucs) if aucs else 0.0
        
        # Update latest round metrics
        if self.round_history:
            self.round_history[-1].avg_val_loss = avg_loss
            self.round_history[-1].avg_accuracy = avg_accuracy
            self.round_history[-1].avg_auc = avg_auc
        
        logger.info(
            f"[round={server_round:02d}] eval: "
            f"loss={avg_loss:.4f}, acc={avg_accuracy:.4f}, AUC={avg_auc:.4f}"
        )
        
        return avg_loss, {
            "accuracy": avg_accuracy,
            "auc": avg_auc,
            "epsilon_total": self.total_epsilon_spent
        }
    
    def _detect_anomalies(
        self,
        gradient_updates: List[Tuple[int, float]]
    ) -> List[Dict]:
        """
        Detect anomalous gradient updates using z-score.
        
        Args:
            gradient_updates: List of (client_id, gradient_norm) tuples
            
        Returns:
            List of anomaly dictionaries
        """
        anomalies = []
        
        norms = [norm for _, norm in gradient_updates]
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        if std_norm < 1e-6:
            return anomalies
        
        for client_id, norm in gradient_updates:
            z_score = (norm - mean_norm) / std_norm
            
            if abs(z_score) > self.anomaly_threshold:
                anomaly = {
                    "client_id": client_id,
                    "gradient_norm": norm,
                    "z_score": z_score,
                    "round": self.current_round
                }
                anomalies.append(anomaly)
                logger.warning(
                    f"Anomaly detected: client {client_id}, "
                    f"gradient_norm={norm:.4f}, z_score={z_score:.2f}"
                )
        
        return anomalies
    
    def _get_budget_status(self) -> str:
        """Get current budget status string."""
        ratio = self.total_epsilon_spent / self.epsilon_budget
        
        if ratio >= 1.0:
            return "HALT_BUDGET_EXHAUSTED"
        elif ratio >= 0.9:
            return "WARNING_BUDGET_90%"
        elif ratio >= 0.75:
            return "WARNING_BUDGET_75%"
        else:
            return "OK"
    
    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.total_epsilon_spent >= self.epsilon_budget
    
    def get_round_history(self) -> List[Dict]:
        """Get history of all rounds."""
        return [rm.to_dict() for rm in self.round_history]
    
    def save_results(self, filename: str = "results.json"):
        """Save results to file."""
        filepath = os.path.join(self.results_dir, filename)
        
        results = {
            "total_rounds": len(self.round_history),
            "final_epsilon": self.total_epsilon_spent,
            "epsilon_budget": self.epsilon_budget,
            "rounds": self.get_round_history()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        return filepath


def create_server_config(
    num_rounds: int = 20,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 3,
    min_evaluate_clients: int = 3,
    min_available_clients: int = 3
) -> fl.server.ServerConfig:
    """Create Flower server configuration."""
    return fl.server.ServerConfig(num_rounds=num_rounds)


def create_medfed_strategy(
    initial_parameters: Optional[Parameters] = None,
    num_clients: int = 3,
    epsilon_budget: float = 8.0,
    local_epochs: int = 1,
    results_dir: str = "./results"
) -> MedFedStrategy:
    """
    Factory function to create MedFedStrategy.
    
    Args:
        initial_parameters: Initial model parameters
        num_clients: Number of clients
        epsilon_budget: Privacy budget
        local_epochs: Local epochs per round
        results_dir: Results directory
        
    Returns:
        Configured MedFedStrategy
    """
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return training configuration for each round."""
        return {
            "local_epochs": local_epochs,
            "server_round": server_round
        }
    
    def evaluate_config(server_round: int) -> Dict[str, Scalar]:
        """Return evaluation configuration for each round."""
        return {
            "server_round": server_round
        }
    
    return MedFedStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        epsilon_budget=epsilon_budget,
        enable_anomaly_detection=True,
        anomaly_threshold=3.0,
        results_dir=results_dir
    )


if __name__ == "__main__":
    # Test the server components
    print("Testing FL server components...")
    
    # Create strategy
    strategy = create_medfed_strategy(
        num_clients=3,
        epsilon_budget=8.0
    )
    
    print(f"Strategy created with ε_budget={strategy.epsilon_budget}")
    print(f"Initial ε_spent={strategy.total_epsilon_spent}")
    print(f"Budget status: {strategy._get_budget_status()}")
