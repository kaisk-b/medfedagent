"""
MedFedAgent Premium Dashboard - Flask Application

A modern, professional-grade dashboard for federated learning in healthcare.
Features:
- Real-time training metrics visualization
- Privacy budget tracking with dynamic updates
- Clinical executive summaries for hospital leadership
- Security status and Byzantine detection monitoring
- Fairness evaluation across hospitals
- Native training integration with full control
"""

import os
import sys
import json
import yaml
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from typing import Dict, List, Optional
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

app = Flask(__name__, 
    template_folder='templates',
    static_folder='static'
)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'medfedagent-secret-key-2024')
app.config['RESULTS_DIR'] = os.environ.get('RESULTS_DIR', './results')
app.config['LOGS_DIR'] = os.environ.get('LOGS_DIR', './logs')
app.config['CONFIG_DIR'] = os.environ.get('CONFIG_DIR', './config')

# Import the native training engine
from src.dashboard.training_engine import get_training_engine, TrainingStatus


def load_results() -> Optional[Dict]:
    """Load results from JSON file."""
    results_path = os.path.join(app.config['RESULTS_DIR'], 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


def load_orchestrator_logs() -> Optional[Dict]:
    """Load orchestrator logs from JSON file."""
    logs_path = os.path.join(app.config['LOGS_DIR'], 'orchestrator_logs.json')
    if os.path.exists(logs_path):
        with open(logs_path, 'r') as f:
            return json.load(f)
    return None


def get_dashboard_data() -> Dict:
    """Get combined dashboard data from all sources."""
    orchestrator_logs = load_orchestrator_logs()
    results = load_results()
    
    # Use orchestrator logs if available, otherwise results
    data = orchestrator_logs or results or {}
    
    summary = data.get('summary', {})
    round_history = data.get('round_history', data.get('rounds', []))
    
    # Calculate derived metrics
    if summary:
        epsilon_budget = summary.get('epsilon_budget', 8.0)
        final_epsilon = summary.get('final_epsilon', 0)
        privacy_remaining = max(0, epsilon_budget - final_epsilon)
        privacy_percentage = (privacy_remaining / epsilon_budget) * 100 if epsilon_budget > 0 else 0
        
        summary['privacy_remaining'] = privacy_remaining
        summary['privacy_percentage'] = privacy_percentage
        summary['privacy_status'] = 'excellent' if privacy_percentage > 75 else 'good' if privacy_percentage > 50 else 'warning' if privacy_percentage > 25 else 'critical'
    
    # Process round history for charts
    chart_data = process_chart_data(round_history)
    
    # Get fairness data
    fairness_data = get_latest_fairness(round_history)
    
    # Get security status
    security_data = get_security_status(summary, round_history)
    
    return {
        'summary': summary,
        'round_history': round_history,
        'chart_data': chart_data,
        'fairness': fairness_data,
        'security': security_data,
        'has_data': bool(data),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


def process_chart_data(round_history: List[Dict]) -> Dict:
    """Process round history into chart-ready format."""
    if not round_history:
        return {
            'rounds': [],
            'epsilon_total': [],
            'epsilon_round': [],
            'train_loss': [],
            'val_loss': [],
            'accuracy': [],
            'auc': [],
            'noise_multiplier': []
        }
    
    return {
        'rounds': [r.get('round_num', r.get('round', i+1)) for i, r in enumerate(round_history)],
        'epsilon_total': [r.get('epsilon_total', 0) for r in round_history],
        'epsilon_round': [r.get('epsilon_round', 0) for r in round_history],
        'train_loss': [r.get('train_loss', 0) for r in round_history],
        'val_loss': [r.get('val_loss', 0) for r in round_history],
        'accuracy': [r.get('accuracy', 0) for r in round_history],
        'auc': [r.get('auc', 0) for r in round_history],
        'noise_multiplier': [r.get('noise_multiplier', 1.0) for r in round_history]
    }


def get_latest_fairness(round_history: List[Dict]) -> Dict:
    """Get latest fairness data from round history."""
    for r in reversed(round_history):
        if 'fairness' in r:
            fairness = r['fairness']
            return {
                'score': fairness.get('score', 0),
                'is_fair': fairness.get('is_fair', False),
                'hospital_variance': fairness.get('hospital_accuracy_variance', 0),
                'accuracy_parity_diff': fairness.get('accuracy_parity_difference', 0),
                'violations': fairness.get('violations', []),
                'hospital_metrics': fairness.get('hospital_metrics', {})
            }
    return {
        'score': 0,
        'is_fair': False,
        'hospital_variance': 0,
        'accuracy_parity_diff': 0,
        'violations': [],
        'hospital_metrics': {}
    }


def get_security_status(summary: Dict, round_history: List[Dict]) -> Dict:
    """Get security status information."""
    security = summary.get('security', {})
    robust_agg = summary.get('robust_aggregation', {})
    
    features = []
    
    if security.get('encryption_enabled', False) or security.get('homomorphic_encryption', False):
        features.append({'name': 'Encrypted Aggregation', 'icon': 'lock', 'status': 'active'})
    
    if security.get('masking_enabled', False) or security.get('random_masking', False):
        features.append({'name': 'Random Masking', 'icon': 'eye-slash', 'status': 'active'})
    
    if security.get('audit_trail', False):
        features.append({'name': 'Audit Trail', 'icon': 'file-text', 'status': 'active'})
    
    if robust_agg:
        method = robust_agg.get('method', 'fedavg')
        features.append({'name': f'Byzantine Protection ({method})', 'icon': 'shield', 'status': 'active'})
    
    # Check round history for security events
    has_secure_agg = any(r.get('security', {}).get('secure_aggregation', False) for r in round_history)
    has_robust_agg = any(r.get('security', {}).get('robust_aggregation', False) for r in round_history)
    
    if has_secure_agg and not any(f['name'].startswith('Encrypted') for f in features):
        features.append({'name': 'Secure Aggregation', 'icon': 'lock', 'status': 'active'})
    
    if has_robust_agg and not any('Byzantine' in f['name'] for f in features):
        features.append({'name': 'Byzantine Protection', 'icon': 'shield', 'status': 'active'})
    
    anomaly_count = sum(len(r.get('anomalies', [])) for r in round_history)
    
    return {
        'features': features if features else [
            {'name': 'Differential Privacy', 'icon': 'user-secret', 'status': 'active'},
            {'name': 'Federated Learning', 'icon': 'network-wired', 'status': 'active'}
        ],
        'anomaly_count': anomaly_count,
        'total_rounds': security.get('total_rounds', len(round_history)),
        'successful_rounds': security.get('successful_rounds', len(round_history)),
        'verification_rate': security.get('verification_rate', 1.0)
    }


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main dashboard page."""
    data = get_dashboard_data()
    return render_template('index.html', **data)


@app.route('/clinical')
def clinical():
    """Clinical executive summary view."""
    data = get_dashboard_data()
    return render_template('clinical.html', **data)


@app.route('/technical')
def technical():
    """Technical details view for ML engineers."""
    data = get_dashboard_data()
    return render_template('technical.html', **data)


@app.route('/privacy')
def privacy():
    """Privacy monitoring and audit page."""
    data = get_dashboard_data()
    
    # Load privacy audit report if available
    audit_files = list(Path(app.config['RESULTS_DIR']).glob('privacy_audit_report_*.json'))
    audit_report = None
    if audit_files:
        latest_audit = max(audit_files, key=os.path.getctime)
        with open(latest_audit, 'r') as f:
            audit_report = json.load(f)
    
    data['audit_report'] = audit_report
    return render_template('privacy.html', **data)


@app.route('/fairness')
def fairness():
    """Fairness evaluation page."""
    data = get_dashboard_data()
    
    # Load fairness report if available
    fairness_path = os.path.join(app.config['RESULTS_DIR'], 'fairness_report.json')
    if os.path.exists(fairness_path):
        with open(fairness_path, 'r') as f:
            data['fairness_report'] = json.load(f)
    
    return render_template('fairness.html', **data)


@app.route('/training')
def training():
    """Training configuration and control page."""
    data = get_dashboard_data()
    
    # Load current config
    config_path = os.path.join(app.config['CONFIG_DIR'], 'config.yaml')
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Get training state from native engine
    engine = get_training_engine()
    
    data['config'] = config
    data['training_state'] = engine.get_state()
    return render_template('training.html', **data)


# ============================================================================
# TRAINING API ENDPOINTS
# ============================================================================

def load_config_yaml():
    """Load the current configuration from YAML."""
    config_path = os.path.join(app.config['CONFIG_DIR'], 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def save_config_yaml(config):
    """Save configuration to YAML file."""
    config_path = os.path.join(app.config['CONFIG_DIR'], 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


@app.route('/api/training/config', methods=['GET'])
def get_training_config():
    """Get current training configuration."""
    config = load_config_yaml()
    return jsonify({
        'success': True,
        'config': config
    })


@app.route('/api/training/config', methods=['POST'])
def update_training_config():
    """Update training configuration."""
    try:
        new_config = request.json
        
        # Load existing config and update
        config = load_config_yaml()
        
        # Update federated settings
        if 'federated' in new_config:
            config.setdefault('federated', {}).update(new_config['federated'])
        
        # Update privacy settings
        if 'privacy' in new_config:
            config.setdefault('privacy', {}).update(new_config['privacy'])
        
        # Update training settings
        if 'training' in new_config:
            config.setdefault('training', {}).update(new_config['training'])
        
        # Update model settings
        if 'model' in new_config:
            config.setdefault('model', {}).update(new_config['model'])
        
        save_config_yaml(config)
        
        return jsonify({
            'success': True,
            'message': 'Configuration updated successfully',
            'config': config
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400


@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start the federated learning training process using native engine."""
    engine = get_training_engine()
    state = engine.get_state()
    
    if state['is_running']:
        return jsonify({
            'success': False,
            'message': 'Training is already running'
        }), 400
    
    try:
        # Get optional config overrides from request
        overrides = request.json or {}
        
        # Load full config
        config = load_config_yaml()
        
        # Apply overrides
        if 'num_rounds' in overrides:
            config.setdefault('federated', {})['num_rounds'] = overrides['num_rounds']
        if 'num_clients' in overrides:
            config.setdefault('federated', {})['num_clients'] = overrides['num_clients']
        if 'epsilon_budget' in overrides:
            config.setdefault('privacy', {})['epsilon_budget'] = overrides['epsilon_budget']
        if 'noise_multiplier' in overrides:
            config.setdefault('privacy', {})['noise_multiplier'] = overrides['noise_multiplier']
        
        # Save updated config
        save_config_yaml(config)
        
        # Start training with native engine
        success = engine.start_training(config)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Training started',
                'state': engine.get_state()
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to start training'
            }), 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop the current training process."""
    engine = get_training_engine()
    
    if not engine.get_state()['is_running']:
        return jsonify({
            'success': False,
            'message': 'No training is currently running'
        }), 400
    
    try:
        success = engine.stop_training()
        
        return jsonify({
            'success': success,
            'message': 'Training stopped' if success else 'Failed to stop training'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/training/pause', methods=['POST'])
def pause_training():
    """Pause the current training process."""
    engine = get_training_engine()
    
    success = engine.pause_training()
    
    return jsonify({
        'success': success,
        'message': 'Training paused' if success else 'Cannot pause training'
    })


@app.route('/api/training/resume', methods=['POST'])
def resume_training():
    """Resume paused training."""
    engine = get_training_engine()
    
    success = engine.resume_training()
    
    return jsonify({
        'success': success,
        'message': 'Training resumed' if success else 'Cannot resume training'
    })


@app.route('/api/training/status', methods=['GET'])
def training_status():
    """Get current training status from native engine."""
    engine = get_training_engine()
    state = engine.get_state()
    
    # Also get live metrics
    live_metrics = engine.get_live_metrics()
    
    return jsonify({
        'success': True,
        'state': {
            'is_running': state['is_running'],
            'status': state['status'],
            'message': state['message'],
            'start_time': state['start_time'],
            'current_round': state['current_round'],
            'total_rounds': state['total_rounds'],
            'error': state.get('error'),
            'live_metrics': live_metrics
        }
    })


@app.route('/api/training/logs', methods=['GET'])
def training_logs():
    """Get recent training output logs from native engine."""
    engine = get_training_engine()
    logs = engine.get_logs(limit=50)
    
    return jsonify({
        'success': True,
        'logs': logs
    })


@app.route('/api/training/round_history', methods=['GET'])
def training_round_history():
    """Get training round history from native engine."""
    engine = get_training_engine()
    history = engine.get_round_history()
    
    return jsonify({
        'success': True,
        'round_history': history
    })


@app.route('/api/training/reset', methods=['POST'])
def reset_training():
    """Reset training state and optionally clear results."""
    engine = get_training_engine()
    state = engine.get_state()
    
    if state['is_running']:
        return jsonify({
            'success': False,
            'message': 'Cannot reset while training is running'
        }), 400
    
    clear_results = request.json.get('clear_results', False) if request.json else False
    
    try:
        # Reset using native engine
        success = engine.reset(clear_results=clear_results)
        
        return jsonify({
            'success': success,
            'message': 'Training state reset' + (' and results cleared' if clear_results else '')
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/data')
def api_data():
    """API endpoint for real-time data updates."""
    data = get_dashboard_data()
    return jsonify(data)


@app.route('/api/metrics')
def api_metrics():
    """API endpoint for chart data only."""
    data = get_dashboard_data()
    return jsonify({
        'chart_data': data['chart_data'],
        'summary': data['summary'],
        'last_updated': data['last_updated']
    })


# ============================================================================
# TEMPLATE CONTEXT PROCESSORS
# ============================================================================

@app.context_processor
def utility_processor():
    """Add utility functions to template context."""
    from jinja2 import Undefined
    
    def format_number(value, decimals=2):
        if value is None or isinstance(value, Undefined):
            return 'N/A'
        try:
            return f"{float(value):.{decimals}f}"
        except (ValueError, TypeError):
            return 'N/A'
    
    def format_percentage(value):
        if value is None or isinstance(value, Undefined):
            return 'N/A'
        try:
            return f"{float(value):.1f}%"
        except (ValueError, TypeError):
            return 'N/A'
    
    def status_color(status):
        colors = {
            'excellent': '#10b981',
            'good': '#3b82f6',
            'warning': '#f59e0b',
            'critical': '#ef4444'
        }
        return colors.get(status, '#6b7280')
    
    return dict(
        format_number=format_number,
        format_percentage=format_percentage,
        status_color=status_color
    )


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Ensure template and static directories exist
    templates_dir = Path(__file__).parent / 'templates'
    static_dir = Path(__file__).parent / 'static'
    templates_dir.mkdir(exist_ok=True)
    static_dir.mkdir(exist_ok=True)
    (static_dir / 'css').mkdir(exist_ok=True)
    (static_dir / 'js').mkdir(exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
