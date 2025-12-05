"""
MedFedAgent Orchestrator

Central orchestrator for managing federated learning rounds,
privacy budget tracking, and making rule-based decisions.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from loguru import logger
import numpy as np


class OrchestratorAction(Enum):
    """Actions the orchestrator can take."""
    CONTINUE = "CONTINUE"
    INCREASE_NOISE = "INCREASE_NOISE"
    DECREASE_NOISE = "DECREASE_NOISE"
    HALT = "HALT"
    WARN_BUDGET_75 = "WARNING_BUDGET_75%"
    WARN_BUDGET_90 = "WARNING_BUDGET_90%"
    EXCLUDE_CLIENT = "EXCLUDE_CLIENT"


@dataclass
class OrchestratorState:
    """Current state of the orchestrator."""
    current_round: int = 0
    epsilon_spent: float = 0.0
    epsilon_budget: float = 8.0
    noise_multiplier: float = 1.0
    participating_nodes: List[int] = field(default_factory=list)
    excluded_nodes: List[int] = field(default_factory=list)
    is_halted: bool = False
    halt_reason: str = ""


@dataclass
class RoundLog:
    """Log entry for a single training round."""
    round_num: int
    timestamp: str
    epsilon_round: float
    epsilon_total: float
    noise_multiplier: float
    num_clients: int
    action: str
    status: str
    train_loss: float = 0.0
    val_loss: float = 0.0
    accuracy: float = 0.0
    auc: float = 0.0
    anomalies: List[Dict] = field(default_factory=list)
    client_metrics: Dict[int, Dict] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def __str__(self) -> str:
        return (
            f"[round={self.round_num:02d}] "
            f"ε_round={self.epsilon_round:.2f}, "
            f"ε_total={self.epsilon_total:.2f}, "
            f"σ={self.noise_multiplier:.2f}, "
            f"nodes={self.num_clients}, "
            f"status={self.status}"
        )


class Orchestrator:
    """
    MedFedAgent Orchestrator
    
    Manages:
    - Privacy budget (ε) tracking across rounds
    - Training round management
    - Noise adjustment when approaching budget
    - Decision logging
    - Basic anomaly response
    
    This is a rule-based orchestrator (not AI-driven).
    """
    
    def __init__(
        self,
        epsilon_budget: float = 8.0,
        initial_noise_multiplier: float = 1.0,
        budget_warning_threshold: float = 0.75,
        budget_critical_threshold: float = 0.90,
        noise_increase_factor: float = 1.1,
        enable_anomaly_response: bool = True,
        anomaly_zscore_threshold: float = 3.0,
        log_dir: str = "./logs"
    ):
        """
        Initialize the orchestrator.
        
        Args:
            epsilon_budget: Total privacy budget
            initial_noise_multiplier: Initial noise scale (σ)
            budget_warning_threshold: Fraction of budget to trigger warning
            budget_critical_threshold: Fraction of budget to trigger critical
            noise_increase_factor: Factor to increase noise when approaching budget
            enable_anomaly_response: Whether to respond to anomalies
            anomaly_zscore_threshold: Z-score threshold for anomaly detection
            log_dir: Directory for log files
        """
        self.state = OrchestratorState(
            epsilon_budget=epsilon_budget,
            noise_multiplier=initial_noise_multiplier
        )
        
        self.budget_warning_threshold = budget_warning_threshold
        self.budget_critical_threshold = budget_critical_threshold
        self.noise_increase_factor = noise_increase_factor
        self.enable_anomaly_response = enable_anomaly_response
        self.anomaly_zscore_threshold = anomaly_zscore_threshold
        self.log_dir = log_dir
        
        # Round history
        self.round_history: List[RoundLog] = []
        
        # Gradient norm history for anomaly detection
        self.gradient_norm_history: Dict[int, List[float]] = {}
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        logger.info(
            f"Orchestrator initialized: ε_budget={epsilon_budget}, "
            f"σ={initial_noise_multiplier}, "
            f"warn_threshold={budget_warning_threshold}"
        )
    
    def on_round_start(self, round_num: int, participating_nodes: List[int]) -> Dict[str, Any]:
        """
        Called at the start of a training round.
        
        Args:
            round_num: Current round number
            participating_nodes: List of participating client IDs
            
        Returns:
            Configuration for this round
        """
        self.state.current_round = round_num
        self.state.participating_nodes = participating_nodes
        
        # Check if we should halt before starting
        if self.state.is_halted:
            logger.warning(f"Round {round_num}: Orchestrator is halted - {self.state.halt_reason}")
            return {"action": OrchestratorAction.HALT.value, "reason": self.state.halt_reason}
        
        # Return round configuration
        config = {
            "round_num": round_num,
            "noise_multiplier": self.state.noise_multiplier,
            "participating_nodes": participating_nodes,
            "action": OrchestratorAction.CONTINUE.value
        }
        
        logger.info(f"Round {round_num} starting with {len(participating_nodes)} nodes, σ={self.state.noise_multiplier:.2f}")
        
        return config
    
    def on_round_complete(
        self,
        round_num: int,
        epsilon_this_round: float,
        train_loss: float = 0.0,
        val_loss: float = 0.0,
        accuracy: float = 0.0,
        auc: float = 0.0,
        client_metrics: Optional[Dict[int, Dict]] = None,
        gradient_norms: Optional[Dict[int, float]] = None,
        anomalies: Optional[List[Dict]] = None
    ) -> Tuple[OrchestratorAction, Dict[str, Any]]:
        """
        Called when a training round completes.
        
        Args:
            round_num: Completed round number
            epsilon_this_round: Privacy cost for this round
            train_loss: Average training loss
            val_loss: Average validation loss
            accuracy: Average accuracy
            auc: Average AUC-ROC
            client_metrics: Per-client metrics
            gradient_norms: Gradient norms per client (for anomaly detection)
            anomalies: Pre-detected anomalies
            
        Returns:
            Tuple of (action to take, action details)
        """
        # Update epsilon tracking
        previous_epsilon = self.state.epsilon_spent
        self.state.epsilon_spent += epsilon_this_round
        
        # Detect anomalies if gradient norms provided
        detected_anomalies = anomalies or []
        if gradient_norms and self.enable_anomaly_response:
            new_anomalies = self._detect_gradient_anomalies(gradient_norms)
            detected_anomalies.extend(new_anomalies)
        
        # Determine action based on budget
        action, status = self._determine_action()
        
        # Log the round
        round_log = RoundLog(
            round_num=round_num,
            timestamp=datetime.now().isoformat(),
            epsilon_round=epsilon_this_round,
            epsilon_total=self.state.epsilon_spent,
            noise_multiplier=self.state.noise_multiplier,
            num_clients=len(self.state.participating_nodes),
            action=action.value,
            status=status,
            train_loss=train_loss,
            val_loss=val_loss,
            accuracy=accuracy,
            auc=auc,
            anomalies=detected_anomalies,
            client_metrics=client_metrics or {}
        )
        self.round_history.append(round_log)
        
        # Print log line
        logger.info(str(round_log))
        
        # Execute action
        action_details = self._execute_action(action, round_num)
        
        return action, action_details
    
    def _determine_action(self) -> Tuple[OrchestratorAction, str]:
        """
        Determine what action to take based on current state.
        
        Returns:
            Tuple of (action, status string)
        """
        budget_ratio = self.state.epsilon_spent / self.state.epsilon_budget
        
        if budget_ratio >= 1.0:
            return OrchestratorAction.HALT, "HALT_BUDGET_EXHAUSTED"
        
        elif budget_ratio >= self.budget_critical_threshold:
            return OrchestratorAction.INCREASE_NOISE, "WARNING_BUDGET_90%"
        
        elif budget_ratio >= self.budget_warning_threshold:
            return OrchestratorAction.WARN_BUDGET_75, "WARNING_BUDGET_75%"
        
        else:
            return OrchestratorAction.CONTINUE, "OK"
    
    def _execute_action(self, action: OrchestratorAction, round_num: int) -> Dict[str, Any]:
        """
        Execute the determined action.
        
        Args:
            action: Action to execute
            round_num: Current round number
            
        Returns:
            Details of the executed action
        """
        details = {"action": action.value, "round": round_num}
        
        if action == OrchestratorAction.HALT:
            self.state.is_halted = True
            self.state.halt_reason = "Privacy budget exhausted"
            details["reason"] = self.state.halt_reason
            logger.warning(f"HALT: {self.state.halt_reason}")
        
        elif action == OrchestratorAction.INCREASE_NOISE:
            old_noise = self.state.noise_multiplier
            self.state.noise_multiplier *= self.noise_increase_factor
            details["old_noise"] = old_noise
            details["new_noise"] = self.state.noise_multiplier
            logger.info(f"Increased noise: {old_noise:.2f} -> {self.state.noise_multiplier:.2f}")
        
        return details
    
    def _detect_gradient_anomalies(
        self,
        gradient_norms: Dict[int, float]
    ) -> List[Dict]:
        """
        Detect anomalous gradient norms using z-score.
        
        Args:
            gradient_norms: Dictionary of client_id -> gradient norm
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Update history
        for client_id, norm in gradient_norms.items():
            if client_id not in self.gradient_norm_history:
                self.gradient_norm_history[client_id] = []
            self.gradient_norm_history[client_id].append(norm)
        
        # Need at least some history to detect anomalies
        all_norms = list(gradient_norms.values())
        if len(all_norms) < 2:
            return anomalies
        
        mean_norm = np.mean(all_norms)
        std_norm = np.std(all_norms)
        
        if std_norm < 1e-6:
            return anomalies
        
        for client_id, norm in gradient_norms.items():
            z_score = (norm - mean_norm) / std_norm
            
            if abs(z_score) > self.anomaly_zscore_threshold:
                anomaly = {
                    "type": "gradient_anomaly",
                    "client_id": client_id,
                    "gradient_norm": norm,
                    "z_score": z_score,
                    "round": self.state.current_round
                }
                anomalies.append(anomaly)
                logger.warning(
                    f"Anomaly detected: client {client_id}, "
                    f"gradient_norm={norm:.4f}, z_score={z_score:.2f}"
                )
        
        return anomalies
    
    def get_state(self) -> Dict[str, Any]:
        """Get current orchestrator state as dictionary."""
        return {
            "current_round": self.state.current_round,
            "epsilon_spent": self.state.epsilon_spent,
            "epsilon_budget": self.state.epsilon_budget,
            "epsilon_remaining": max(0, self.state.epsilon_budget - self.state.epsilon_spent),
            "budget_percentage": (self.state.epsilon_spent / self.state.epsilon_budget) * 100,
            "noise_multiplier": self.state.noise_multiplier,
            "participating_nodes": self.state.participating_nodes,
            "excluded_nodes": self.state.excluded_nodes,
            "is_halted": self.state.is_halted,
            "halt_reason": self.state.halt_reason
        }
    
    def get_round_history(self) -> List[Dict]:
        """Get history of all rounds as dictionaries."""
        return [log.to_dict() for log in self.round_history]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.round_history:
            return {"message": "No rounds completed yet"}
        
        losses = [r.train_loss for r in self.round_history if r.train_loss > 0]
        aucs = [r.auc for r in self.round_history if r.auc > 0]
        epsilons = [r.epsilon_round for r in self.round_history]
        
        return {
            "total_rounds": len(self.round_history),
            "final_epsilon": self.state.epsilon_spent,
            "epsilon_budget": self.state.epsilon_budget,
            "budget_used_percent": (self.state.epsilon_spent / self.state.epsilon_budget) * 100,
            "avg_epsilon_per_round": np.mean(epsilons) if epsilons else 0,
            "final_loss": losses[-1] if losses else 0,
            "best_auc": max(aucs) if aucs else 0,
            "final_auc": aucs[-1] if aucs else 0,
            "total_anomalies": sum(len(r.anomalies) for r in self.round_history),
            "noise_multiplier_final": self.state.noise_multiplier,
            "is_halted": self.state.is_halted
        }
    
    def save_logs(self, filename: str = "orchestrator_logs.json"):
        """Save logs to file."""
        filepath = os.path.join(self.log_dir, filename)
        
        data = {
            "summary": self.get_summary(),
            "final_state": self.get_state(),
            "round_history": self.get_round_history()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Logs saved to {filepath}")
        return filepath
    
    def print_log_table(self):
        """Print formatted log table."""
        print("\n" + "="*80)
        print("ORCHESTRATOR LOG")
        print("="*80)
        print(f"{'Round':>6} {'ε_round':>8} {'ε_total':>8} {'σ':>6} {'Nodes':>6} {'Status':<25}")
        print("-"*80)
        
        for log in self.round_history:
            print(
                f"{log.round_num:>6} "
                f"{log.epsilon_round:>8.2f} "
                f"{log.epsilon_total:>8.2f} "
                f"{log.noise_multiplier:>6.2f} "
                f"{log.num_clients:>6} "
                f"{log.status:<25}"
            )
        
        print("="*80)
        summary = self.get_summary()
        print(f"Total rounds: {summary['total_rounds']}")
        print(f"Final epsilon: {summary['final_epsilon']:.2f} / {summary['epsilon_budget']}")
        print(f"Budget used: {summary['budget_used_percent']:.1f}%")
        print(f"Best AUC: {summary['best_auc']:.4f}")
        print("="*80 + "\n")
    
    def should_continue(self) -> bool:
        """Check if training should continue."""
        return not self.state.is_halted


if __name__ == "__main__":
    # Test the orchestrator
    print("Testing Orchestrator...")
    
    orchestrator = Orchestrator(
        epsilon_budget=8.0,
        initial_noise_multiplier=1.0,
        budget_warning_threshold=0.75,
        budget_critical_threshold=0.90
    )
    
    # Simulate 25 rounds
    print("\nSimulating training rounds...")
    
    for round_num in range(1, 26):
        # Start round
        config = orchestrator.on_round_start(round_num, [0, 1, 2])
        
        if config.get("action") == "HALT":
            print(f"Training halted at round {round_num}")
            break
        
        # Simulate round completion
        epsilon_round = np.random.uniform(0.3, 0.5)
        action, details = orchestrator.on_round_complete(
            round_num=round_num,
            epsilon_this_round=epsilon_round,
            train_loss=1.0 - (round_num * 0.03),
            val_loss=1.0 - (round_num * 0.025),
            accuracy=0.5 + (round_num * 0.015),
            auc=0.5 + (round_num * 0.015)
        )
        
        if action == OrchestratorAction.HALT:
            print(f"Training halted at round {round_num}")
            break
    
    # Print summary
    orchestrator.print_log_table()
    
    # Save logs
    orchestrator.save_logs("test_logs.json")
