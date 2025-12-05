"""
MedFedAgent Training Engine

Native integration of federated learning with the web dashboard.
Provides direct control over training without subprocess parsing.
"""

import os
import sys
import threading
import queue
import json
import math
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.model import create_model, get_model_size_mb
from src.data.dataset import load_federated_datasets
from src.privacy.dp_trainer import PrivacyConfig
from src.federated.client import MedFedClient, get_parameters, set_parameters
from src.federated.robust_aggregation import ByzantineRobustAggregator, AggregationConfig, AggregationMethod
from src.orchestrator.orchestrator import Orchestrator, OrchestratorAction
from src.fairness.evaluator import create_fairness_evaluator
from src.privacy.secure_aggregation import create_secure_aggregator


class TrainingStatus(Enum):
    """Training status enumeration."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class TrainingMetrics:
    """Real-time training metrics."""
    round_num: int = 0
    total_rounds: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    auc: float = 0.0
    epsilon_spent: float = 0.0
    epsilon_budget: float = 8.0
    noise_multiplier: float = 1.0
    fairness_score: Optional[float] = None
    is_fair: bool = True
    learning_rate: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrainingState:
    """Complete training state."""
    status: TrainingStatus = TrainingStatus.IDLE
    message: str = "Ready to start"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    current_round: int = 0
    total_rounds: int = 0
    metrics: TrainingMetrics = field(default_factory=TrainingMetrics)
    round_history: List[Dict] = field(default_factory=list)
    log_buffer: List[Dict] = field(default_factory=list)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'status': self.status.value,
            'message': self.message,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'current_round': self.current_round,
            'total_rounds': self.total_rounds,
            'is_running': self.status == TrainingStatus.RUNNING,
            'metrics': self.metrics.to_dict(),
            'error': self.error
        }


class TrainingEngine:
    """
    Native training engine for MedFedAgent.
    
    Provides direct control over federated learning training
    with real-time metrics and state management.
    """
    
    def __init__(
        self,
        results_dir: str = "./results",
        logs_dir: str = "./logs",
        config_dir: str = "./config"
    ):
        self.results_dir = results_dir
        self.logs_dir = logs_dir
        self.config_dir = config_dir
        
        # Training state
        self.state = TrainingState()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._training_thread: Optional[threading.Thread] = None
        
        # Callbacks for real-time updates
        self._on_round_complete: Optional[Callable] = None
        self._on_metrics_update: Optional[Callable] = None
        self._on_status_change: Optional[Callable] = None
        
        # Training components (initialized during training)
        self._clients: List[MedFedClient] = []
        self._global_model = None
        self._orchestrator: Optional[Orchestrator] = None
        self._fairness_evaluator = None
        self._secure_aggregator = None
        self._robust_aggregator = None
        self._initial_learning_rate: Optional[float] = None
        self._stability_config = {
            'loss_spike_threshold': 0.25,
            'auc_drop_threshold': 0.1,
            'decay_factor': 0.5,
            'min_lr': 1e-4
        }
        
        # Ensure directories exist
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        self._add_log("Training engine initialized", "info")
    
    def _add_log(self, message: str, level: str = "info"):
        """Add a log entry."""
        with self._lock:
            log_entry = {
                'time': datetime.now().strftime('%H:%M:%S'),
                'message': message,
                'level': level
            }
            self.state.log_buffer.append(log_entry)
            # Keep last 500 entries
            if len(self.state.log_buffer) > 500:
                self.state.log_buffer = self.state.log_buffer[-500:]
        
        # Also log to loguru
        getattr(logger, level, logger.info)(message)
    
    def _update_status(self, status: TrainingStatus, message: str = ""):
        """Update training status."""
        with self._lock:
            self.state.status = status
            if message:
                self.state.message = message
        
        self._add_log(f"Status: {status.value} - {message}", "info")
        
        if self._on_status_change:
            self._on_status_change(status, message)
    
    def _update_metrics(self, metrics: Dict):
        """Update training metrics."""
        with self._lock:
            if 'loss' in metrics:
                self.state.metrics.loss = metrics['loss']
            if 'accuracy' in metrics:
                self.state.metrics.accuracy = metrics['accuracy']
            if 'auc' in metrics:
                self.state.metrics.auc = metrics['auc']
            if 'epsilon_spent' in metrics:
                self.state.metrics.epsilon_spent = metrics['epsilon_spent']
            if 'fairness_score' in metrics:
                self.state.metrics.fairness_score = metrics['fairness_score']
            if 'is_fair' in metrics:
                self.state.metrics.is_fair = metrics['is_fair']
            if 'learning_rate' in metrics and metrics['learning_rate'] is not None:
                self.state.metrics.learning_rate = metrics['learning_rate']
        
        if self._on_metrics_update:
            self._on_metrics_update(metrics)

    def _current_learning_rate(self) -> Optional[float]:
        """Return current learning rate from first client (all share same LR)."""
        if not self._clients:
            return None
        return getattr(self._clients[0], 'learning_rate', None)

    def _sanitize_learning_rate(self, requested_lr: float, dp_enabled: bool) -> float:
        """Clamp user-provided learning rate to a safe range."""
        if requested_lr <= 0:
            requested_lr = 1e-3
        max_lr = 0.01 if dp_enabled else 0.02
        if requested_lr > max_lr:
            self._add_log(
                f"Learning rate {requested_lr} is high for CPU{' + DP' if dp_enabled else ''}; clamping to {max_lr}",
                "warning"
            )
            return max_lr
        return requested_lr

    def _apply_learning_rate_decay(
        self,
        factor: Optional[float] = None,
        reason: str = "instability"
    ) -> Optional[float]:
        """Decay all client learning rates and log the action."""
        if not self._clients:
            return None
        factor = factor or self._stability_config['decay_factor']
        new_lrs = []
        for client in self._clients:
            new_lr = client.scale_learning_rate(
                factor=factor,
                min_lr=self._stability_config['min_lr']
            )
            new_lrs.append(new_lr)
        avg_lr = sum(new_lrs) / len(new_lrs)
        self._add_log(
            f"Adaptive LR: {reason} triggered decay x{factor:.2f}, avg lr now {avg_lr:.6f}",
            "warning"
        )
        return new_lrs[0] if new_lrs else None

    def _evaluate_round_stability(
        self,
        round_num: int,
        loss: float,
        auc: float,
        previous_round: Optional[Dict]
    ) -> Dict[str, Any]:
        """Detect instability between rounds and adjust learning rate if needed."""
        info = {
            'adjusted': False,
            'learning_rate': self._current_learning_rate()
        }

        if previous_round is None:
            return info

        prev_loss = previous_round.get('val_loss')
        prev_auc = previous_round.get('auc')

        if prev_loss is None or prev_auc is None:
            return info

        loss_spike = loss > prev_loss * (1 + self._stability_config['loss_spike_threshold'])
        auc_drop = (prev_auc - auc) > self._stability_config['auc_drop_threshold']

        if not (loss_spike or auc_drop):
            return info

        reason = 'loss_spike' if loss_spike else 'auc_drop'
        new_lr = self._apply_learning_rate_decay(reason=reason)
        info.update({
            'adjusted': True,
            'reason': reason,
            'learning_rate': new_lr,
            'round': round_num
        })
        return info
    
    def get_state(self) -> Dict:
        """Get current training state."""
        with self._lock:
            return self.state.to_dict()
    
    def get_logs(self, limit: int = 50) -> List[Dict]:
        """Get recent log entries."""
        with self._lock:
            return self.state.log_buffer[-limit:]
    
    def get_round_history(self) -> List[Dict]:
        """Get round history."""
        with self._lock:
            return self.state.round_history.copy()
    
    def get_live_metrics(self) -> Dict:
        """Get live metrics for dashboard."""
        with self._lock:
            return {
                'auc': self.state.metrics.auc,
                'accuracy': self.state.metrics.accuracy,
                'loss': self.state.metrics.loss,
                'epsilon': self.state.metrics.epsilon_spent
            }
    
    def set_callbacks(
        self,
        on_round_complete: Optional[Callable] = None,
        on_metrics_update: Optional[Callable] = None,
        on_status_change: Optional[Callable] = None
    ):
        """Set callbacks for real-time updates."""
        self._on_round_complete = on_round_complete
        self._on_metrics_update = on_metrics_update
        self._on_status_change = on_status_change
    
    def start_training(self, config: Dict) -> bool:
        """
        Start federated learning training.
        
        Args:
            config: Training configuration dictionary
            
        Returns:
            True if training started successfully
        """
        if self.state.status == TrainingStatus.RUNNING:
            self._add_log("Training already running", "warning")
            return False
        
        # Reset state
        self._stop_event.clear()
        self._pause_event.clear()
        
        with self._lock:
            self.state = TrainingState()
            self.state.start_time = datetime.now().isoformat()
        
        # Start training in background thread
        self._training_thread = threading.Thread(
            target=self._training_loop,
            args=(config,),
            daemon=True
        )
        self._training_thread.start()
        
        return True
    
    def stop_training(self) -> bool:
        """Stop training."""
        if self.state.status != TrainingStatus.RUNNING:
            return False
        
        self._stop_event.set()
        self._update_status(TrainingStatus.STOPPED, "Training stopped by user")
        return True
    
    def pause_training(self) -> bool:
        """Pause training."""
        if self.state.status != TrainingStatus.RUNNING:
            return False
        
        self._pause_event.set()
        self._update_status(TrainingStatus.PAUSED, "Training paused")
        return True
    
    def resume_training(self) -> bool:
        """Resume paused training."""
        if self.state.status != TrainingStatus.PAUSED:
            return False
        
        self._pause_event.clear()
        self._update_status(TrainingStatus.RUNNING, "Training resumed")
        return True
    
    def reset(self, clear_results: bool = False) -> bool:
        """Reset training state."""
        if self.state.status == TrainingStatus.RUNNING:
            return False
        
        with self._lock:
            self.state = TrainingState()
        
        if clear_results:
            # Clear result files
            results_path = os.path.join(self.results_dir, 'results.json')
            if os.path.exists(results_path):
                os.remove(results_path)
            
            logs_path = os.path.join(self.logs_dir, 'orchestrator_logs.json')
            if os.path.exists(logs_path):
                os.remove(logs_path)
        
        self._add_log("Training state reset" + (" and results cleared" if clear_results else ""), "info")
        return True
    
    def _training_loop(self, config: Dict):
        """Main training loop (runs in background thread)."""
        try:
            self._update_status(TrainingStatus.INITIALIZING, "Initializing training...")
            
            # Extract config
            fed_config = config.get('federated', {})
            model_config = config.get('model', {})
            train_config = config.get('training', {})
            privacy_config_dict = config.get('privacy', {})
            data_config = config.get('data', {})
            
            num_clients = fed_config.get('num_clients', 3)
            num_rounds = fed_config.get('num_rounds', 10)
            
            with self._lock:
                self.state.total_rounds = num_rounds
                self.state.metrics.total_rounds = num_rounds
                self.state.metrics.epsilon_budget = privacy_config_dict.get('epsilon_budget', 8.0)
            
            # Setup device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._add_log(f"Using device: {device}", "info")
            
            # Load data
            self._add_log("Loading federated datasets...", "info")
            
            # Use realistic dataset for better demonstration
            use_realistic = data_config.get('use_realistic', True)
            label_noise = data_config.get('label_noise', 0.08)
            class_overlap = data_config.get('class_overlap', 0.25)
            
            train_loaders, val_loaders, test_loader = load_federated_datasets(
                num_clients=num_clients,
                samples_per_client=data_config.get('samples_per_client', 500),
                num_classes=model_config.get('num_classes', 2),
                image_size=data_config.get('image_size', 128),
                non_iid=data_config.get('non_iid', {}).get('enabled', True),
                alpha=data_config.get('non_iid', {}).get('alpha', 0.5),
                train_split=data_config.get('train_split', 0.8),
                seed=config.get('experiment', {}).get('seed', 42),
                use_synthetic=not use_realistic,
                batch_size=train_config.get('batch_size', 64),
                use_realistic=use_realistic,
                label_noise=label_noise,
                class_overlap=class_overlap
            )
            
            # Create privacy config
            privacy_cfg = PrivacyConfig(
                enabled=privacy_config_dict.get('enabled', True),
                epsilon_budget=privacy_config_dict.get('epsilon_budget', 8.0),
                delta=privacy_config_dict.get('delta', 1e-5),
                max_grad_norm=privacy_config_dict.get('max_grad_norm', 1.0),
                noise_multiplier=privacy_config_dict.get('noise_multiplier', 1.0)
            )

            base_learning_rate = train_config.get('learning_rate', 0.001)
            learning_rate = self._sanitize_learning_rate(base_learning_rate, privacy_cfg.enabled)
            self._initial_learning_rate = learning_rate
            grad_clip_norm = train_config.get(
                'grad_clip_norm',
                privacy_cfg.max_grad_norm if privacy_cfg.enabled else 2.0
            )
            self._add_log(f"Using effective learning rate: {learning_rate:.5f}", "info")
            
            # Create secure aggregation
            secure_agg_config = config.get('secure_aggregation', {})
            if secure_agg_config.get('enabled', True):
                self._add_log("Initializing secure aggregation...", "info")
                self._secure_aggregator = create_secure_aggregator(
                    config=secure_agg_config,
                    log_dir=self.logs_dir
                )
            
            # Create robust aggregation
            robust_agg_config = config.get('robust_aggregation', {})
            if robust_agg_config.get('enabled', True):
                self._add_log("Initializing Byzantine-robust aggregator...", "info")
                agg_method = AggregationMethod(robust_agg_config.get('method', 'trimmed_mean'))
                self._robust_aggregator = ByzantineRobustAggregator(
                    AggregationConfig(
                        method=agg_method,
                        num_byzantine=robust_agg_config.get('num_byzantine', 0),
                        trim_ratio=robust_agg_config.get('trim_ratio', 0.1),
                        enable_detection=robust_agg_config.get('enable_detection', True),
                        detection_threshold=robust_agg_config.get('detection_threshold', 3.0)
                    )
                )
            
            # Create orchestrator
            self._orchestrator = Orchestrator(
                epsilon_budget=privacy_cfg.epsilon_budget,
                initial_noise_multiplier=privacy_cfg.noise_multiplier,
                budget_warning_threshold=config.get('orchestrator', {}).get('budget_warning_threshold', 0.75),
                budget_critical_threshold=config.get('orchestrator', {}).get('budget_critical_threshold', 0.90),
                noise_increase_factor=config.get('orchestrator', {}).get('noise_increase_factor', 1.1),
                enable_anomaly_response=config.get('orchestrator', {}).get('enable_anomaly_detection', True),
                log_dir=self.logs_dir
            )
            
            # Create fairness evaluator
            fairness_config_dict = config.get('fairness', {})
            if fairness_config_dict.get('enabled', True):
                self._add_log("Creating fairness evaluator...", "info")
                self._fairness_evaluator = create_fairness_evaluator(
                    config=fairness_config_dict.get('thresholds', {}),
                    device=device
                )
            
            # Create clients
            self._add_log(f"Creating {num_clients} federated clients...", "info")
            self._clients = []
            for i in range(num_clients):
                model = create_model(
                    model_name=model_config.get('name', 'simple_cnn'),
                    num_classes=model_config.get('num_classes', 2),
                    pretrained=model_config.get('pretrained', False),
                    dropout=model_config.get('dropout', 0.3)
                )
                
                client = MedFedClient(
                    client_id=i,
                    model=model,
                    train_loader=train_loaders[i],
                    val_loader=val_loaders[i],
                    device=device,
                    privacy_config=privacy_cfg,
                    local_epochs=train_config.get('local_epochs', 1),
                    learning_rate=learning_rate,
                    grad_clip_norm=grad_clip_norm
                )
                self._clients.append(client)
            
            # Initialize global model
            self._add_log("Initializing global model...", "info")
            self._global_model = create_model(
                model_name=model_config.get('name', 'simple_cnn'),
                num_classes=model_config.get('num_classes', 2),
                pretrained=model_config.get('pretrained', False)
            )
            global_params = get_parameters(self._global_model)
            
            model_size = get_model_size_mb(self._global_model)
            self._add_log(f"Model size: {model_size:.2f} MB", "info")
            
            # Start training
            self._update_status(TrainingStatus.RUNNING, "Training in progress")
            self._add_log(f"Starting federated training for {num_rounds} rounds...", "info")
            
            best_auc = 0.0
            round_results = []
            
            for round_num in range(1, num_rounds + 1):
                # Check for stop signal
                if self._stop_event.is_set():
                    self._add_log("Training stopped by user", "warning")
                    break
                
                # Check for pause
                while self._pause_event.is_set():
                    if self._stop_event.is_set():
                        break
                    threading.Event().wait(0.5)
                
                # Update round number
                with self._lock:
                    self.state.current_round = round_num
                    self.state.metrics.round_num = round_num
                
                # Check with orchestrator
                round_config = self._orchestrator.on_round_start(round_num, list(range(num_clients)))
                
                if round_config.get('action') == OrchestratorAction.HALT.value:
                    self._add_log(f"Orchestrator halted training at round {round_num}", "warning")
                    break
                
                self._add_log(f"Starting round {round_num}/{num_rounds}...", "info")
                
                # Distribute global model
                for client in self._clients:
                    client.set_parameters(global_params)
                
                # Local training
                client_updates = []
                client_metrics = {}
                gradient_norms = {}
                total_epsilon = 0.0
                
                for i, client in enumerate(self._clients):
                    updated_params, num_samples, metrics = client.fit(
                        global_params,
                        {"local_epochs": train_config.get('local_epochs', 1)}
                    )
                    
                    client_updates.append((updated_params, num_samples))
                    client_metrics[i] = metrics
                    
                    if metrics.get('epsilon', 0) > total_epsilon:
                        total_epsilon = metrics['epsilon']
                    
                    grad_stats = client.get_gradient_stats()
                    gradient_norms[i] = grad_stats.get('mean', 0.0)
                
                # Aggregation
                total_samples = sum(ns for _, ns in client_updates)
                
                # Secure aggregation
                secure_agg_metadata = None
                if self._secure_aggregator is not None:
                    round_sec_config = self._secure_aggregator.start_round(
                        round_id=round_num,
                        participating_clients=list(range(num_clients))
                    )
                    
                    for i, (params, ns) in enumerate(client_updates):
                        self._secure_aggregator.submit_masked_update(
                            client_id=i,
                            model_update=params,
                            mask_seeds=round_sec_config.get('mask_pairs', {})
                        )
                    
                    _, secure_agg_metadata = self._secure_aggregator.aggregate()
                
                # Robust aggregation
                byzantine_detected = False
                excluded_clients = []
                
                if self._robust_aggregator is not None:
                    robust_updates = [(i, params, ns) for i, (params, ns) in enumerate(client_updates)]
                    agg_result = self._robust_aggregator.aggregate(robust_updates, global_params)
                    aggregated_params = agg_result.aggregated_params
                    byzantine_detected = agg_result.is_byzantine_detected
                    excluded_clients = agg_result.excluded_clients
                    
                    if byzantine_detected:
                        self._add_log(f"Byzantine behavior detected, excluded: {excluded_clients}", "warning")
                else:
                    # Standard FedAvg
                    aggregated_params = []
                    for param_idx in range(len(client_updates[0][0])):
                        weighted_sum = np.zeros_like(client_updates[0][0][param_idx])
                        for params, num_samples in client_updates:
                            weighted_sum += params[param_idx] * (num_samples / total_samples)
                        aggregated_params.append(weighted_sum)
                
                global_params = aggregated_params
                
                # Evaluation
                set_parameters(self._global_model, global_params)
                
                avg_loss = 0.0
                avg_accuracy = 0.0
                avg_auc = 0.0
                valid_auc_count = 0
                per_client_eval_metrics = {}
                
                for i, client in enumerate(self._clients):
                    client.set_parameters(global_params)
                    loss, _, eval_metrics = client.evaluate(global_params, {})
                    avg_loss += loss
                    avg_accuracy += eval_metrics.get('accuracy', 0)
                    
                    client_auc = eval_metrics.get('auc', 0)
                    if client_auc is not None and not (isinstance(client_auc, float) and math.isnan(client_auc)):
                        avg_auc += client_auc
                        valid_auc_count += 1
                    
                    per_client_eval_metrics[i] = {
                        'accuracy': eval_metrics.get('accuracy', 0),
                        'auc': client_auc if client_auc and not math.isnan(client_auc) else 0,
                        'loss': loss
                    }
                
                avg_loss /= num_clients
                avg_accuracy /= num_clients
                avg_auc = avg_auc / valid_auc_count if valid_auc_count > 0 else 0.0
                
                # Fairness evaluation
                fairness_metrics = None
                if self._fairness_evaluator is not None:
                    fairness_metrics = self._fairness_evaluator.evaluate_per_hospital(
                        per_client_eval_metrics,
                        round_num=round_num
                    )
                    
                    if not fairness_metrics.is_fair:
                        self._add_log(f"Fairness violations: {fairness_metrics.violations}", "warning")
                
                previous_round = round_results[-1] if round_results else None
                stability_info = self._evaluate_round_stability(
                    round_num,
                    avg_loss,
                    avg_auc,
                    previous_round
                )

                # Track best model
                if avg_auc > best_auc:
                    best_auc = avg_auc
                    torch.save(
                        self._global_model.state_dict(),
                        os.path.join(self.results_dir, 'best_model.pt')
                    )
                
                # Report to orchestrator
                epsilon_this_round = total_epsilon - (round_results[-1]['epsilon_total'] if round_results else 0)
                if epsilon_this_round < 0:
                    epsilon_this_round = total_epsilon / round_num
                
                action, action_details = self._orchestrator.on_round_complete(
                    round_num=round_num,
                    epsilon_this_round=epsilon_this_round,
                    train_loss=sum(m.get('train_loss', 0) for m in client_metrics.values()) / num_clients,
                    val_loss=avg_loss,
                    accuracy=avg_accuracy,
                    auc=avg_auc,
                    client_metrics=client_metrics,
                    gradient_norms=gradient_norms
                )
                
                # Update metrics
                self._update_metrics({
                    'loss': avg_loss,
                    'accuracy': avg_accuracy,
                    'auc': avg_auc,
                    'epsilon_spent': self._orchestrator.state.epsilon_spent,
                    'fairness_score': fairness_metrics.fairness_score if fairness_metrics else None,
                    'is_fair': fairness_metrics.is_fair if fairness_metrics else True,
                    'learning_rate': stability_info.get('learning_rate') if stability_info else self._current_learning_rate()
                })
                
                # Store round result
                round_result = {
                    'round': round_num,
                    'round_num': round_num,
                    'epsilon_round': epsilon_this_round,
                    'epsilon_total': self._orchestrator.state.epsilon_spent,
                    'noise_multiplier': self._orchestrator.state.noise_multiplier,
                    'num_clients': num_clients,
                    'train_loss': sum(m.get('train_loss', 0) for m in client_metrics.values()) / num_clients,
                    'val_loss': avg_loss,
                    'accuracy': avg_accuracy,
                    'auc': avg_auc,
                    'status': action.value if hasattr(action, 'value') else str(action),
                    'security': {
                        'secure_aggregation': secure_agg_metadata is not None,
                        'robust_aggregation': self._robust_aggregator is not None,
                        'byzantine_detected': byzantine_detected,
                        'excluded_clients': excluded_clients
                    },
                    'stability': stability_info
                }
                
                if fairness_metrics:
                    round_result['fairness'] = {
                        'score': fairness_metrics.fairness_score,
                        'is_fair': fairness_metrics.is_fair,
                        'hospital_accuracy_variance': fairness_metrics.hospital_accuracy_variance,
                        'accuracy_parity_difference': fairness_metrics.accuracy_parity_difference,
                        'hospital_metrics': fairness_metrics.hospital_metrics,
                        'violations': fairness_metrics.violations
                    }
                
                with self._lock:
                    self.state.round_history.append(round_result)
                round_results.append(round_result)
                
                # Log progress
                fairness_info = ""
                if fairness_metrics:
                    fairness_info = f", fairness={fairness_metrics.fairness_score:.3f}"
                lr_value = stability_info.get('learning_rate') if stability_info else self._current_learning_rate()
                lr_info = f", lr={lr_value:.5f}" if lr_value else ""
                
                self._add_log(
                    f"Round {round_num}: loss={avg_loss:.4f}, acc={avg_accuracy:.4f}, "
                    f"AUC={avg_auc:.4f}, ε={self._orchestrator.state.epsilon_spent:.2f}{fairness_info}{lr_info}",
                    "success"
                )
                
                # Callback
                if self._on_round_complete:
                    self._on_round_complete(round_result)
                
                # Check halt
                if action == OrchestratorAction.HALT:
                    self._add_log("Training halted by orchestrator", "warning")
                    break
            
            # Training completed
            with self._lock:
                self.state.end_time = datetime.now().isoformat()
            
            # Save results
            summary = self._orchestrator.get_summary()
            summary['best_auc'] = best_auc
            
            if self._fairness_evaluator:
                summary['fairness'] = self._fairness_evaluator.get_summary()
                self._fairness_evaluator.save_report(
                    os.path.join(self.results_dir, 'fairness_report.json')
                )
            
            if self._secure_aggregator:
                self._secure_aggregator.save_audit_trail()
                summary['security'] = self._secure_aggregator.get_security_report()
            
            if self._robust_aggregator:
                summary['robust_aggregation'] = {
                    'method': self._robust_aggregator.config.method.value,
                    'total_rounds': len(self._robust_aggregator.history),
                    'byzantine_detections': sum(1 for r in self._robust_aggregator.history if r.is_byzantine_detected)
                }
            
            results = {
                'summary': summary,
                'round_history': round_results,
                'config': config,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            
            self._orchestrator.save_logs()
            
            if not self._stop_event.is_set():
                self._update_status(TrainingStatus.COMPLETED, f"Training completed! Best AUC: {best_auc:.4f}")
                self._add_log(f"Training completed! Best AUC: {best_auc:.4f}", "success")
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            self._add_log(f"Training error: {error_msg}", "error")
            self._add_log(traceback.format_exc(), "error")
            
            with self._lock:
                self.state.error = error_msg
            
            self._update_status(TrainingStatus.ERROR, f"Training failed: {error_msg}")


# Global training engine instance
_training_engine: Optional[TrainingEngine] = None


def get_training_engine() -> TrainingEngine:
    """Get or create the global training engine instance."""
    global _training_engine
    if _training_engine is None:
        _training_engine = TrainingEngine()
    return _training_engine
