"""
MedFedAgent - Main Simulation Runner

This script runs the complete federated learning simulation with:
- 3 hospital nodes with non-IID data distribution
- DP-SGD at each node using Opacus
- Federated averaging with Flower
- Privacy budget tracking via Orchestrator
- Fairness evaluation across hospitals and demographics
- Metrics logging and visualization

Usage:
    python run_simulation.py [--config CONFIG_PATH] [--rounds NUM_ROUNDS]
"""

import os
import sys
import argparse
import yaml
import json
import math
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
from loguru import logger

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from src.models.model import create_model, get_model_size_mb
from src.data.dataset import load_federated_datasets, get_transforms
from src.privacy.dp_trainer import PrivacyConfig, DPTrainer
from src.privacy.mia_attack import MembershipInferenceAttack, MIAConfig, run_privacy_audit, print_privacy_report
from src.privacy.secure_aggregation import SecureAggregator, SecureAggConfig, create_secure_aggregator
from src.federated.client import MedFedClient, get_parameters, set_parameters
from src.federated.server import MedFedStrategy, create_medfed_strategy
from src.federated.robust_aggregation import ByzantineRobustAggregator, AggregationConfig, AggregationMethod
from src.orchestrator.orchestrator import Orchestrator, OrchestratorAction
from src.fairness.evaluator import FairnessEvaluator, FairnessConfig, create_fairness_evaluator


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, level="DEBUG", rotation="10 MB")
    
    return log_file


def create_clients(
    num_clients: int,
    train_loaders: List,
    val_loaders: List,
    model_name: str,
    num_classes: int,
    device: str,
    privacy_config: PrivacyConfig,
    local_epochs: int,
    learning_rate: float
) -> List[MedFedClient]:
    """Create federated learning clients."""
    clients = []
    
    for i in range(num_clients):
        model = create_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=True,
            dropout=0.3
        )
        
        client = MedFedClient(
            client_id=i,
            model=model,
            train_loader=train_loaders[i],
            val_loader=val_loaders[i],
            device=device,
            privacy_config=privacy_config,
            local_epochs=local_epochs,
            learning_rate=learning_rate
        )
        clients.append(client)
        
        logger.info(f"Created client {i} with {len(train_loaders[i].dataset)} training samples")
    
    return clients


def run_federated_simulation(
    config: Dict,
    results_dir: str,
    logs_dir: str
) -> Dict:
    """
    Run the complete federated learning simulation.
    
    This simulates the Flower FL process without starting actual servers,
    which is more suitable for local testing and hackathon demos.
    """
    # Extract config
    fed_config = config.get('federated', {})
    model_config = config.get('model', {})
    train_config = config.get('training', {})
    privacy_config_dict = config.get('privacy', {})
    data_config = config.get('data', {})
    
    num_clients = fed_config.get('num_clients', 3)
    num_rounds = fed_config.get('num_rounds', 20)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Load data
    logger.info("Loading federated datasets...")
    train_loaders, val_loaders, test_loader = load_federated_datasets(
        num_clients=num_clients,
        samples_per_client=data_config.get('samples_per_client', 500),  # Use config value
        num_classes=model_config.get('num_classes', 2),
        image_size=data_config.get('image_size', 128),  # Smaller default for CPU
        non_iid=data_config.get('non_iid', {}).get('enabled', True),
        alpha=data_config.get('non_iid', {}).get('alpha', 0.5),
        train_split=data_config.get('train_split', 0.8),
        seed=config.get('experiment', {}).get('seed', 42),
        use_synthetic=True,
        batch_size=train_config.get('batch_size', 64)  # Use config batch size
    )
    
    # Create privacy config
    privacy_cfg = PrivacyConfig(
        enabled=privacy_config_dict.get('enabled', True),
        epsilon_budget=privacy_config_dict.get('epsilon_budget', 8.0),
        delta=privacy_config_dict.get('delta', 1e-5),
        max_grad_norm=privacy_config_dict.get('max_grad_norm', 1.0),
        noise_multiplier=privacy_config_dict.get('noise_multiplier', 1.0)
    )
    
    # Create secure aggregation module
    secure_agg_config = config.get('secure_aggregation', {})
    secure_aggregator = None
    if secure_agg_config.get('enabled', True):
        logger.info("Initializing secure aggregation module...")
        secure_aggregator = create_secure_aggregator(
            config=secure_agg_config,
            log_dir=logs_dir
        )
    
    # Create robust aggregation module
    robust_agg_config = config.get('robust_aggregation', {})
    robust_aggregator = None
    if robust_agg_config.get('enabled', True):
        logger.info("Initializing Byzantine-robust aggregator...")
        agg_method = AggregationMethod(robust_agg_config.get('method', 'trimmed_mean'))
        robust_aggregator = ByzantineRobustAggregator(
            AggregationConfig(
                method=agg_method,
                num_byzantine=robust_agg_config.get('num_byzantine', 0),
                trim_ratio=robust_agg_config.get('trim_ratio', 0.1),
                enable_detection=robust_agg_config.get('enable_detection', True),
                detection_threshold=robust_agg_config.get('detection_threshold', 3.0)
            )
        )
    
    # Create orchestrator
    orchestrator = Orchestrator(
        epsilon_budget=privacy_cfg.epsilon_budget,
        initial_noise_multiplier=privacy_cfg.noise_multiplier,
        budget_warning_threshold=config.get('orchestrator', {}).get('budget_warning_threshold', 0.75),
        budget_critical_threshold=config.get('orchestrator', {}).get('budget_critical_threshold', 0.90),
        noise_increase_factor=config.get('orchestrator', {}).get('noise_increase_factor', 1.1),
        enable_anomaly_response=config.get('orchestrator', {}).get('enable_anomaly_detection', True),
        log_dir=logs_dir
    )
    
    # Create fairness evaluator
    fairness_config_dict = config.get('fairness', {})
    fairness_enabled = fairness_config_dict.get('enabled', True)
    fairness_evaluator = None
    
    if fairness_enabled:
        logger.info("Creating fairness evaluator...")
        fairness_thresholds = fairness_config_dict.get('thresholds', {})
        fairness_evaluator = create_fairness_evaluator(
            config=fairness_thresholds,
            device=device
        )
    
    # Create clients
    logger.info("Creating federated clients...")
    clients = create_clients(
        num_clients=num_clients,
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        model_name=model_config.get('name', 'resnet18'),  # Use smaller model for speed
        num_classes=model_config.get('num_classes', 2),
        device=device,
        privacy_config=privacy_cfg,
        local_epochs=train_config.get('local_epochs', 1),
        learning_rate=train_config.get('learning_rate', 0.001)
    )
    
    # Initialize global model
    logger.info("Initializing global model...")
    global_model = create_model(
        model_name=model_config.get('name', 'resnet18'),
        num_classes=model_config.get('num_classes', 2),
        pretrained=True
    )
    global_params = get_parameters(global_model)
    
    logger.info(f"Model size: {get_model_size_mb(global_model):.2f} MB")
    
    # Training loop
    logger.info(f"Starting federated training for {num_rounds} rounds...")
    
    best_auc = 0.0
    round_results = []
    
    for round_num in range(1, num_rounds + 1):
        # Check with orchestrator before starting
        round_config = orchestrator.on_round_start(round_num, list(range(num_clients)))
        
        if round_config.get('action') == OrchestratorAction.HALT.value:
            logger.warning(f"Orchestrator halted training at round {round_num}")
            break
        
        # Distribute global model to clients
        for client in clients:
            client.set_parameters(global_params)
        
        # Local training at each client
        client_updates = []
        client_metrics = {}
        gradient_norms = {}
        total_epsilon = 0.0
        
        for i, client in enumerate(clients):
            # Train locally
            updated_params, num_samples, metrics = client.fit(
                global_params,
                {"local_epochs": train_config.get('local_epochs', 1)}
            )
            
            client_updates.append((updated_params, num_samples))
            client_metrics[i] = metrics
            
            # Track epsilon (use max across clients)
            if metrics.get('epsilon', 0) > total_epsilon:
                total_epsilon = metrics['epsilon']
            
            # Track gradient norms for anomaly detection
            grad_stats = client.get_gradient_stats()
            gradient_norms[i] = grad_stats.get('mean', 0.0)
        
        # Prepare updates for secure/robust aggregation
        total_samples = sum(ns for _, ns in client_updates)
        
        # Apply Secure Aggregation if enabled
        secure_agg_metadata = None
        if secure_aggregator is not None:
            logger.debug(f"Round {round_num}: Using secure aggregation")
            round_sec_config = secure_aggregator.start_round(
                round_id=round_num,
                participating_clients=list(range(num_clients))
            )
            
            # Submit masked updates from each client
            for i, (params, ns) in enumerate(client_updates):
                secure_aggregator.submit_masked_update(
                    client_id=i,
                    model_update=params,
                    mask_seeds=round_sec_config.get('mask_pairs', {})
                )
            
            # Note: Secure aggregation result is for audit - actual aggregation below
            _, secure_agg_metadata = secure_aggregator.aggregate()
        
        # Apply Robust Aggregation if enabled
        byzantine_detected = False
        excluded_clients = []
        
        if robust_aggregator is not None:
            logger.debug(f"Round {round_num}: Using robust aggregation ({robust_aggregator.config.method.value})")
            # Format updates for robust aggregator
            robust_updates = [(i, params, ns) for i, (params, ns) in enumerate(client_updates)]
            
            agg_result = robust_aggregator.aggregate(robust_updates, global_params)
            aggregated_params = agg_result.aggregated_params
            byzantine_detected = agg_result.is_byzantine_detected
            excluded_clients = agg_result.excluded_clients
            
            if byzantine_detected:
                logger.warning(f"Round {round_num}: Byzantine behavior detected, excluded clients: {excluded_clients}")
        else:
            # Standard FedAvg
            aggregated_params = []
            for param_idx in range(len(client_updates[0][0])):
                weighted_sum = np.zeros_like(client_updates[0][0][param_idx])
                for params, num_samples in client_updates:
                    weighted_sum += params[param_idx] * (num_samples / total_samples)
                aggregated_params.append(weighted_sum)
        
        global_params = aggregated_params
        
        # Evaluate global model
        set_parameters(global_model, global_params)
        
        avg_loss = 0.0
        avg_accuracy = 0.0
        avg_auc = 0.0
        valid_auc_count = 0  # Track clients with valid AUC scores
        per_client_eval_metrics = {}  # Store per-client metrics for fairness
        
        for i, client in enumerate(clients):
            client.set_parameters(global_params)
            loss, _, eval_metrics = client.evaluate(global_params, {})
            avg_loss += loss
            avg_accuracy += eval_metrics.get('accuracy', 0)
            # Handle nan AUC values (occurs when validation set has only one class)
            client_auc = eval_metrics.get('auc', 0)
            if client_auc is not None and not (isinstance(client_auc, float) and math.isnan(client_auc)):
                avg_auc += client_auc
                valid_auc_count += 1
            
            # Store for fairness evaluation
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
        if fairness_evaluator is not None:
            fairness_metrics = fairness_evaluator.evaluate_per_hospital(
                per_client_eval_metrics, 
                round_num=round_num
            )
            
            if not fairness_metrics.is_fair:
                logger.warning(f"Round {round_num} fairness violations: {fairness_metrics.violations}")
        
        # Track best model
        if avg_auc > best_auc:
            best_auc = avg_auc
            # Save best model
            torch.save(global_model.state_dict(), os.path.join(results_dir, 'best_model.pt'))
        
        # Report to orchestrator
        epsilon_this_round = total_epsilon - (round_results[-1]['epsilon_total'] if round_results else 0)
        if epsilon_this_round < 0:
            epsilon_this_round = total_epsilon / round_num  # Estimate
        
        action, action_details = orchestrator.on_round_complete(
            round_num=round_num,
            epsilon_this_round=epsilon_this_round,
            train_loss=sum(m.get('train_loss', 0) for m in client_metrics.values()) / num_clients,
            val_loss=avg_loss,
            accuracy=avg_accuracy,
            auc=avg_auc,
            client_metrics=client_metrics,
            gradient_norms=gradient_norms
        )
        
        # Store round results (including fairness and security)
        round_result = {
            'round': round_num,
            'round_num': round_num,
            'epsilon_round': epsilon_this_round,
            'epsilon_total': orchestrator.state.epsilon_spent,
            'noise_multiplier': orchestrator.state.noise_multiplier,
            'num_clients': num_clients,
            'train_loss': sum(m.get('train_loss', 0) for m in client_metrics.values()) / num_clients,
            'val_loss': avg_loss,
            'accuracy': avg_accuracy,
            'auc': avg_auc,
            'status': action.value if hasattr(action, 'value') else str(action),
            'anomalies': [],
            'security': {
                'secure_aggregation': secure_agg_metadata is not None,
                'robust_aggregation': robust_aggregator is not None,
                'byzantine_detected': byzantine_detected,
                'excluded_clients': excluded_clients
            }
        }
        
        # Add fairness metrics to round results
        if fairness_metrics is not None:
            round_result['fairness'] = {
                'score': fairness_metrics.fairness_score,
                'is_fair': fairness_metrics.is_fair,
                'hospital_accuracy_variance': fairness_metrics.hospital_accuracy_variance,
                'accuracy_parity_difference': fairness_metrics.accuracy_parity_difference,
                'hospital_metrics': fairness_metrics.hospital_metrics,
                'violations': fairness_metrics.violations
            }
        
        round_results.append(round_result)
        
        # Log with fairness info
        fairness_info = ""
        if fairness_metrics is not None:
            fairness_info = f", fairness={fairness_metrics.fairness_score:.3f}"
        
        logger.info(f"Round {round_num}: loss={avg_loss:.4f}, acc={avg_accuracy:.4f}, "
                   f"AUC={avg_auc:.4f}, ε={orchestrator.state.epsilon_spent:.2f}{fairness_info}")
        
        # Check if we should stop
        if action == OrchestratorAction.HALT:
            logger.warning("Training halted by orchestrator")
            break
    
    # Final summary
    summary = orchestrator.get_summary()
    summary['best_auc'] = best_auc
    
    # Add fairness summary
    if fairness_evaluator is not None:
        fairness_summary = fairness_evaluator.get_summary()
        summary['fairness'] = fairness_summary
        
        # Save fairness report
        fairness_report_path = os.path.join(results_dir, 'fairness_report.json')
        fairness_evaluator.save_report(fairness_report_path)
        
        # Print fairness report
        fairness_evaluator.print_report()
    
    # =========================================================================
    # POST-TRAINING PRIVACY AUDIT (Membership Inference Attack)
    # =========================================================================
    mia_config = config.get('privacy_audit', {})
    if mia_config.get('enabled', True):
        logger.info("=" * 60)
        logger.info("Running Membership Inference Attack (Privacy Audit)...")
        logger.info("=" * 60)
        
        try:
            # Use a subset of training and test data for MIA
            mia_results = run_privacy_audit(
                model=global_model,
                train_loader=train_loaders[0],  # Use first client's training data as "member" data
                test_loader=test_loader,         # Test data is non-member data
                device=device,
                attack_types=mia_config.get('attack_types', ['threshold', 'loss_based'])
            )
            
            # Print results
            print_privacy_report(mia_results)
            
            # Add to summary
            summary['privacy_audit'] = {
                attack_type: {
                    'accuracy': result.accuracy,
                    'auc': result.auc,
                    'advantage': result.advantage,
                    'vulnerability_score': result.vulnerability_score,
                    'privacy_grade': result.get_privacy_grade(),
                    'is_vulnerable': result.is_vulnerable()
                }
                for attack_type, result in mia_results.items()
            }
            
            # Save detailed MIA report
            mia_report_path = os.path.join(results_dir, 'mia_privacy_audit.json')
            with open(mia_report_path, 'w') as f:
                json.dump(summary['privacy_audit'], f, indent=2)
            logger.info(f"MIA audit report saved to {mia_report_path}")
            
        except Exception as e:
            logger.error(f"Privacy audit failed: {e}")
            summary['privacy_audit'] = {'error': str(e)}
    
    # Save secure aggregation audit trail
    if secure_aggregator is not None:
        sec_agg_report_path = secure_aggregator.save_audit_trail()
        summary['security'] = {
            **secure_aggregator.get_security_report(),
            'audit_trail_path': sec_agg_report_path
        }
        logger.info(f"Secure aggregation audit saved to {sec_agg_report_path}")
    
    # Save robust aggregation summary
    if robust_aggregator is not None:
        byzantine_summary = {
            'method': robust_aggregator.config.method.value,
            'total_rounds': len(robust_aggregator.history),
            'byzantine_detections': sum(1 for r in robust_aggregator.history if r.is_byzantine_detected),
            'total_excluded': sum(r.num_excluded for r in robust_aggregator.history)
        }
        summary['robust_aggregation'] = byzantine_summary
    
    # Save results
    results = {
        'summary': summary,
        'round_history': round_results,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(results_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save orchestrator logs
    orchestrator.save_logs()
    
    # Print final summary
    orchestrator.print_log_table()
    
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Best AUC: {best_auc:.4f}")
    
    return results


def run_baselines(config: Dict, results_dir: str) -> Dict:
    """Run baseline experiments for comparison."""
    baselines = {}
    
    # 1. Local-only training (no federation)
    logger.info("Running local-only baseline...")
    # This would train each node independently and average results
    # Simplified for hackathon
    baselines['local_only'] = {'auc': 0.76, 'note': 'placeholder'}
    
    # 2. FedAvg without DP
    logger.info("Running FedAvg without DP baseline...")
    # This would run federation without privacy
    baselines['fedavg_no_dp'] = {'auc': 0.81, 'note': 'placeholder'}
    
    return baselines


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='MedFedAgent Simulation Runner')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--rounds', type=int, default=None,
                       help='Override number of training rounds')
    parser.add_argument('--no-dp', action='store_true',
                       help='Disable differential privacy')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Results output directory')
    parser.add_argument('--logs-dir', type=str, default='./logs',
                       help='Logs output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup logging
    log_file = setup_logging(args.logs_dir)
    logger.info(f"Logging to {log_file}")
    
    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    
    logger.info(f"Loading config from {config_path}")
    config = load_config(config_path)
    
    # Apply overrides
    if args.rounds:
        config['federated']['num_rounds'] = args.rounds
    
    if args.no_dp:
        config['privacy']['enabled'] = False
    
    config['experiment']['seed'] = args.seed
    
    # Print configuration
    logger.info("=" * 60)
    logger.info("MedFedAgent - Federated Learning Simulation")
    logger.info("=" * 60)
    logger.info(f"Clients: {config['federated']['num_clients']}")
    logger.info(f"Rounds: {config['federated']['num_rounds']}")
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"DP Enabled: {config['privacy']['enabled']}")
    logger.info(f"ε Budget: {config['privacy']['epsilon_budget']}")
    logger.info("=" * 60)
    
    # Run simulation
    try:
        results = run_federated_simulation(
            config=config,
            results_dir=args.results_dir,
            logs_dir=args.logs_dir
        )
        
        logger.info("Simulation completed successfully!")
        
        # Print summary
        summary = results.get('summary', {})
        print("\n" + "=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        print(f"Total Rounds: {summary.get('total_rounds', 0)}")
        print(f"Final ε: {summary.get('final_epsilon', 0):.2f} / {summary.get('epsilon_budget', 8.0)}")
        print(f"Best AUC: {summary.get('best_auc', 0):.4f}")
        print(f"Final AUC: {summary.get('final_auc', 0):.4f}")
        print("=" * 60)
        
    except Exception as e:
        logger.exception(f"Simulation failed: {e}")
        raise


if __name__ == "__main__":
    main()
