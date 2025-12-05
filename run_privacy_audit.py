"""
MedFedAgent Privacy Audit Script

This script runs a comprehensive privacy audit including:
1. Membership Inference Attack testing
2. Secure Aggregation demonstration
3. Privacy metrics summary

Usage:
    python run_privacy_audit.py [--model-path PATH] [--attack-type TYPE]
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.model import create_model
from src.data.dataset import load_federated_datasets
from src.privacy.mia_attack import (
    MIAConfig,
    MembershipInferenceAttack,
    run_privacy_audit,
    print_privacy_report
)
from src.privacy.secure_aggregation import (
    demonstrate_secure_aggregation,
    create_secure_aggregator,
    SecureAggConfig
)


def setup_logging(log_dir: str = "./logs"):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"privacy_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, level="DEBUG", rotation="10 MB")
    
    return log_file


def run_mia_audit(
    model_path: str,
    model_name: str = "simplecnn",
    num_classes: int = 2,
    attack_types: list = None,
    device: str = "cpu"
):
    """
    Run membership inference attack audit.
    
    Args:
        model_path: Path to trained model checkpoint
        model_name: Model architecture name
        num_classes: Number of output classes
        attack_types: List of attack types to run
        device: Device for computation
    """
    if attack_types is None:
        attack_types = ["threshold", "loss_based"]
    
    print("\n" + "=" * 70)
    print("MEMBERSHIP INFERENCE ATTACK AUDIT")
    print("=" * 70)
    
    # Load model
    print("\n1. Loading model...")
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False
    ).to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"   Loaded model from: {model_path}")
    else:
        print(f"   WARNING: Model file not found at {model_path}")
        print("   Using randomly initialized model for demonstration")
    
    # Load data
    print("\n2. Loading datasets...")
    train_loaders, val_loaders, test_loader = load_federated_datasets(
        num_clients=2,  # Just need train/test split
        samples_per_client=300,
        num_classes=num_classes,
        image_size=128,
        non_iid=False,
        seed=42,
        use_synthetic=True,
        batch_size=32
    )
    
    # Combine loaders for attack
    member_loader = train_loaders[0]  # Training samples (members)
    non_member_loader = test_loader   # Test samples (non-members)
    
    print(f"   Members: {len(member_loader.dataset)} samples")
    print(f"   Non-members: {len(non_member_loader.dataset)} samples")
    
    # Run attacks
    print("\n3. Running Membership Inference Attacks...")
    results = run_privacy_audit(
        model=model,
        train_loader=member_loader,
        test_loader=non_member_loader,
        device=device,
        attack_types=attack_types
    )
    
    # Print results
    print_privacy_report(results)
    
    return results


def run_secure_aggregation_demo(num_clients: int = 3, update_size: int = 1000):
    """
    Run secure aggregation demonstration.
    
    Args:
        num_clients: Number of simulated clients
        update_size: Size of model update vector
    """
    print("\n" + "=" * 70)
    print("SECURE AGGREGATION DEMONSTRATION")
    print("=" * 70)
    
    aggregator, aggregated = demonstrate_secure_aggregation(
        num_clients=num_clients,
        update_size=update_size
    )
    
    # Save audit trail
    audit_path = aggregator.save_audit_trail()
    print(f"\n   Audit trail saved to: {audit_path}")
    
    # Get security report
    report = aggregator.get_security_report()
    
    print("\n" + "-" * 40)
    print("Security Report Summary:")
    print(f"   Total rounds: {report['total_rounds']}")
    print(f"   Success rate: {report['success_rate']:.1%}")
    print(f"   Verification rate: {report['verification_rate']:.1%}")
    print("-" * 40)
    
    return aggregator, report


def generate_privacy_report(
    mia_results: dict,
    secure_agg_report: dict,
    output_path: str = None
):
    """
    Generate comprehensive privacy audit report.
    
    Args:
        mia_results: Results from MIA attacks
        secure_agg_report: Report from secure aggregation
        output_path: Path to save report
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "privacy_audit": {
            "membership_inference_attack": {},
            "secure_aggregation": secure_agg_report
        },
        "overall_assessment": {
            "privacy_grade": "Unknown",
            "recommendations": []
        }
    }
    
    # Process MIA results
    best_auc = 0.0
    for attack_type, result in mia_results.items():
        report["privacy_audit"]["membership_inference_attack"][attack_type] = {
            "accuracy": result.accuracy,
            "auc": result.auc,
            "advantage": result.advantage,
            "vulnerability_score": result.vulnerability_score,
            "privacy_grade": result.get_privacy_grade(),
            "is_vulnerable": result.is_vulnerable()
        }
        best_auc = max(best_auc, result.auc)
    
    # Overall assessment
    if best_auc <= 0.55:
        report["overall_assessment"]["privacy_grade"] = "A (Excellent)"
        report["overall_assessment"]["recommendations"].append(
            "Privacy protection is excellent. Differential privacy is working effectively."
        )
    elif best_auc <= 0.60:
        report["overall_assessment"]["privacy_grade"] = "B (Good)"
        report["overall_assessment"]["recommendations"].append(
            "Privacy protection is good. Consider slightly increasing noise multiplier for better protection."
        )
    elif best_auc <= 0.70:
        report["overall_assessment"]["privacy_grade"] = "C (Fair)"
        report["overall_assessment"]["recommendations"].append(
            "Privacy protection is fair. Increase noise multiplier or reduce training epochs."
        )
    else:
        report["overall_assessment"]["privacy_grade"] = "D or F (Poor)"
        report["overall_assessment"]["recommendations"].append(
            "Privacy protection is poor. Significantly increase differential privacy parameters."
        )
    
    # Add secure aggregation recommendations
    if secure_agg_report.get("encryption_enabled"):
        report["overall_assessment"]["recommendations"].append(
            "Secure aggregation is enabled with encryption simulation."
        )
    else:
        report["overall_assessment"]["recommendations"].append(
            "Consider enabling homomorphic encryption for additional security."
        )
    
    # Save report
    if output_path is None:
        output_path = f"./results/privacy_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n   Full privacy report saved to: {output_path}")
    
    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='MedFedAgent Privacy Audit')
    parser.add_argument('--model-path', type=str, default='./results/best_model.pt',
                       help='Path to trained model')
    parser.add_argument('--model-name', type=str, default='simplecnn',
                       help='Model architecture name')
    parser.add_argument('--attack-types', type=str, nargs='+',
                       default=['threshold', 'loss_based'],
                       help='MIA attack types to run')
    parser.add_argument('--no-mia', action='store_true',
                       help='Skip membership inference attack')
    parser.add_argument('--no-secure-agg', action='store_true',
                       help='Skip secure aggregation demo')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging()
    logger.info(f"Logging to {log_file}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    print("\n" + "=" * 70)
    print("MedFedAgent - Privacy Audit Suite")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print("=" * 70)
    
    mia_results = {}
    secure_agg_report = {}
    
    try:
        # Run MIA audit
        if not args.no_mia:
            mia_results = run_mia_audit(
                model_path=args.model_path,
                model_name=args.model_name,
                attack_types=args.attack_types,
                device=device
            )
        
        # Run secure aggregation demo
        if not args.no_secure_agg:
            _, secure_agg_report = run_secure_aggregation_demo()
        
        # Generate comprehensive report
        if mia_results or secure_agg_report:
            report_path = os.path.join(
                args.output_dir,
                f"privacy_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            report = generate_privacy_report(
                mia_results=mia_results,
                secure_agg_report=secure_agg_report,
                output_path=report_path
            )
            
            # Print summary
            print("\n" + "=" * 70)
            print("PRIVACY AUDIT SUMMARY")
            print("=" * 70)
            print(f"Overall Privacy Grade: {report['overall_assessment']['privacy_grade']}")
            print("\nRecommendations:")
            for i, rec in enumerate(report['overall_assessment']['recommendations'], 1):
                print(f"  {i}. {rec}")
            print("=" * 70)
        
        logger.info("Privacy audit completed successfully!")
        
    except Exception as e:
        logger.exception(f"Privacy audit failed: {e}")
        raise


if __name__ == "__main__":
    main()
