#!/usr/bin/env python3
"""
MedFedAgent Quick Demo Script

This script provides a one-command demonstration of all MedFedAgent features
with clear explanations suitable for hackathon judges and clinical audiences.

Features demonstrated:
✅ Federated Learning across 3 hospital nodes
✅ Differential Privacy (DP-SGD) with budget tracking
✅ Byzantine-robust aggregation (Trimmed Mean)
✅ Secure aggregation protocol with audit trail
✅ Membership Inference Attack (privacy validation)
✅ Fairness evaluation across hospitals
✅ Real-time orchestration with anomaly detection

Usage:
    python run_demo.py                    # Quick 5-round demo
    python run_demo.py --full             # Full 20-round training
    python run_demo.py --dashboard        # Also launch dashboard
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

# Add colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    """Print welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███╗   ███╗███████╗██████╗ ███████╗███████╗██████╗  █████╗  ██████╗ ████████╗ ║
║   ████╗ ████║██╔════╝██╔══██╗██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝ ╚══██╔══╝ ║
║   ██╔████╔██║█████╗  ██║  ██║█████╗  █████╗  ██║  ██║███████║██║  ███╗   ██║    ║
║   ██║╚██╔╝██║██╔══╝  ██║  ██║██╔══╝  ██╔══╝  ██║  ██║██╔══██║██║   ██║   ██║    ║
║   ██║ ╚═╝ ██║███████╗██████╔╝██║     ███████╗██████╔╝██║  ██║╚██████╔╝   ██║    ║
║   ╚═╝     ╚═╝╚══════╝╚═════╝ ╚═╝     ╚══════╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ║
║                                                                              ║
║              Privacy-Preserving Federated Learning for Healthcare            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(Colors.CYAN + banner + Colors.ENDC)


def print_section(title, description=""):
    """Print a formatted section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.GREEN}🔹 {title}{Colors.ENDC}")
    if description:
        print(f"{Colors.CYAN}   {description}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.ENDC}\n")


def print_feature(emoji, title, explanation):
    """Print a feature explanation."""
    print(f"{emoji} {Colors.BOLD}{title}{Colors.ENDC}")
    print(f"   {Colors.CYAN}{explanation}{Colors.ENDC}\n")


def explain_system():
    """Explain the system for judges/clinical audience."""
    print_section("What is MedFedAgent?", 
                  "A complete privacy-preserving federated learning system for medical imaging")
    
    print(f"""{Colors.CYAN}
Imagine 3 hospitals wanting to collaborate on training an AI diagnostic model,
but they cannot share patient data due to HIPAA/privacy regulations.

MedFedAgent solves this by:
{Colors.ENDC}""")
    
    print_feature("🏥", "Federated Learning",
        "Each hospital trains locally on their own data, only sharing model updates (not data)")
    
    print_feature("🔒", "Differential Privacy (DP-SGD)", 
        "Mathematical guarantees that individual patients cannot be identified from the model")
    
    print_feature("🔐", "Secure Aggregation",
        "Model updates are encrypted/masked so even the central server cannot see individual contributions")
    
    print_feature("🛡️", "Byzantine Robustness",
        "Protection against malicious or faulty hospital nodes trying to corrupt the model")
    
    print_feature("⚖️", "Fairness Evaluation",
        "Ensures the model works equally well across different hospitals and patient demographics")
    
    print_feature("🔍", "Privacy Audit (MIA)",
        "Validates privacy by attempting to infer training data membership - lower success = better privacy")
    
    input(f"\n{Colors.WARNING}Press Enter to start the demo...{Colors.ENDC}")


def run_simulation(num_rounds=5, quick=True):
    """Run the federated learning simulation."""
    print_section("Starting Federated Learning Simulation",
                  f"Training across 3 hospital nodes for {num_rounds} rounds")
    
    # Set environment for quick demo
    if quick:
        os.environ['MEDFED_QUICK_DEMO'] = '1'
    
    # Import and run
    import yaml
    
    # Create a demo config
    demo_config_path = os.path.join(os.path.dirname(__file__), 'config', 'demo_config.yaml')
    
    # Load base config and modify for demo
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for demo
    config['federated']['num_rounds'] = num_rounds
    config['data']['samples_per_client'] = 200 if quick else 500
    config['model']['name'] = 'simple_cnn'  # Faster model
    
    # Save demo config
    with open(demo_config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"{Colors.GREEN}✓ Demo configuration created{Colors.ENDC}")
    print(f"{Colors.CYAN}  - Hospitals: 3")
    print(f"  - Training rounds: {num_rounds}")
    print(f"  - Privacy budget (ε): {config['privacy']['epsilon_budget']}")
    print(f"  - Robust aggregation: {config['robust_aggregation']['method']}")
    print(f"  - Secure aggregation: {config['secure_aggregation']['enabled']}{Colors.ENDC}\n")
    
    # Run the simulation
    print(f"{Colors.WARNING}Starting training... (this may take a few minutes){Colors.ENDC}\n")
    
    try:
        # Import the simulation module
        sys.path.insert(0, os.path.dirname(__file__))
        from run_simulation import run_federated_simulation, load_config, setup_logging
        
        # Setup
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
        setup_logging(logs_dir)
        
        # Run
        results = run_federated_simulation(
            config=config,
            results_dir=results_dir,
            logs_dir=logs_dir
        )
        
        return results
        
    except Exception as e:
        print(f"{Colors.FAIL}Error during simulation: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return None


def display_results(results):
    """Display results in a user-friendly format."""
    if not results:
        print(f"{Colors.FAIL}No results to display{Colors.ENDC}")
        return
    
    summary = results.get('summary', {})
    
    print_section("📊 Results Summary")
    
    # Training metrics
    print(f"{Colors.GREEN}Training Results:{Colors.ENDC}")
    print(f"  • Rounds completed: {summary.get('total_rounds', 'N/A')}")
    print(f"  • Best AUC-ROC: {summary.get('best_auc', 0):.4f}")
    print(f"  • Final accuracy: {summary.get('final_accuracy', 0):.4f}")
    
    # Privacy metrics
    print(f"\n{Colors.GREEN}Privacy Metrics:{Colors.ENDC}")
    epsilon_spent = summary.get('final_epsilon', 0)
    epsilon_budget = summary.get('epsilon_budget', 8.0)
    print(f"  • Privacy budget used: {epsilon_spent:.2f} / {epsilon_budget} (ε)")
    print(f"  • Budget remaining: {((epsilon_budget - epsilon_spent) / epsilon_budget * 100):.1f}%")
    
    # Privacy audit results
    privacy_audit = summary.get('privacy_audit', {})
    if privacy_audit and 'error' not in privacy_audit:
        print(f"\n{Colors.GREEN}Privacy Audit (Membership Inference Attack):{Colors.ENDC}")
        for attack_type, result in privacy_audit.items():
            grade = result.get('privacy_grade', 'N/A')
            auc = result.get('auc', 0)
            print(f"  • {attack_type} attack: AUC={auc:.4f}, Grade={grade}")
            if auc < 0.6:
                print(f"    {Colors.GREEN}✓ Well protected - attacker no better than random guessing{Colors.ENDC}")
            else:
                print(f"    {Colors.WARNING}⚠ Some information leakage detected{Colors.ENDC}")
    
    # Fairness metrics
    fairness = summary.get('fairness', {})
    if fairness:
        print(f"\n{Colors.GREEN}Fairness Metrics:{Colors.ENDC}")
        print(f"  • Fairness score: {fairness.get('average_fairness_score', 0):.3f}")
        violations = fairness.get('total_violations', 0)
        if violations == 0:
            print(f"  • {Colors.GREEN}✓ No fairness violations - model works equitably across hospitals{Colors.ENDC}")
        else:
            print(f"  • {Colors.WARNING}⚠ {violations} fairness violations detected{Colors.ENDC}")
    
    # Security metrics
    security = summary.get('security', {})
    if security:
        print(f"\n{Colors.GREEN}Security Features:{Colors.ENDC}")
        print(f"  • Secure aggregation rounds: {security.get('total_rounds', 'N/A')}")
        print(f"  • Verification rate: {security.get('verification_rate', 0)*100:.1f}%")
    
    robust_agg = summary.get('robust_aggregation', {})
    if robust_agg:
        print(f"  • Robust aggregation method: {robust_agg.get('method', 'N/A')}")
        detections = robust_agg.get('byzantine_detections', 0)
        if detections > 0:
            print(f"  • {Colors.WARNING}⚠ {detections} Byzantine behaviors detected and handled{Colors.ENDC}")
        else:
            print(f"  • {Colors.GREEN}✓ No malicious behavior detected{Colors.ENDC}")


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print_section("🖥️ Launching Dashboard",
                  "Opening the visual dashboard in your browser...")
    
    dashboard_path = os.path.join(os.path.dirname(__file__), 'src', 'dashboard', 'app.py')
    
    subprocess.Popen([
        sys.executable, '-m', 'streamlit', 'run', dashboard_path,
        '--server.port', '8501',
        '--browser.gatherUsageStats', 'false'
    ])
    
    print(f"\n{Colors.GREEN}Dashboard launching at: http://localhost:8501{Colors.ENDC}")
    print(f"{Colors.CYAN}Select '🩺 Clinical Summary' for hospital leadership view")
    print(f"Select '🔧 Technical Details' for ML engineering view{Colors.ENDC}")


def main():
    parser = argparse.ArgumentParser(
        description='MedFedAgent Quick Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py              Quick 5-round demo
  python run_demo.py --full       Full 20-round training
  python run_demo.py --dashboard  Launch dashboard after training
  python run_demo.py --explain    Explain the system (no training)
        """
    )
    parser.add_argument('--full', action='store_true', 
                       help='Run full 20-round training instead of quick demo')
    parser.add_argument('--rounds', type=int, default=None,
                       help='Custom number of rounds')
    parser.add_argument('--dashboard', action='store_true',
                       help='Launch dashboard after training')
    parser.add_argument('--explain', action='store_true',
                       help='Only explain the system, do not run training')
    parser.add_argument('--skip-explain', action='store_true',
                       help='Skip explanation and start training immediately')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Explain system
    if not args.skip_explain:
        explain_system()
    
    if args.explain:
        return
    
    # Determine rounds
    num_rounds = args.rounds or (20 if args.full else 5)
    
    # Run simulation
    results = run_simulation(num_rounds=num_rounds, quick=not args.full)
    
    # Display results
    if results:
        display_results(results)
    
    # Launch dashboard if requested
    if args.dashboard:
        launch_dashboard()
        input(f"\n{Colors.WARNING}Press Enter to exit (dashboard will continue running)...{Colors.ENDC}")
    
    print(f"\n{Colors.GREEN}{'='*70}")
    print("Demo complete! Thank you for exploring MedFedAgent.")
    print(f"{'='*70}{Colors.ENDC}\n")


if __name__ == "__main__":
    main()
