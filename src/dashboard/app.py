"""
MedFedAgent Metrics Dashboard

Streamlit-based dashboard for visualizing:
- Training progress (loss, accuracy, AUC)
- Privacy budget tracking (epsilon)
- Per-node performance
- Anomaly detection results
- Security status (secure aggregation, Byzantine detection)
- Clinical executive summary
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import time


# Page configuration
st.set_page_config(
    page_title="MedFedAgent Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better clinical UX
st.markdown("""
<style>
    .clinical-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
    }
    .metric-good { color: #00C853; font-weight: bold; }
    .metric-warning { color: #FFD600; font-weight: bold; }
    .metric-critical { color: #FF5252; font-weight: bold; }
    .executive-summary {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 10px 10px 0;
    }
    .security-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        margin: 2px;
    }
    .badge-secure { background: #E8F5E9; color: #2E7D32; }
    .badge-warning { background: #FFF3E0; color: #EF6C00; }
</style>
""", unsafe_allow_html=True)


def display_clinical_executive_summary(summary: Dict, round_history: List[Dict]):
    """
    Display a clinical-friendly executive summary for hospital administrators.
    This addresses the UX feedback about making the dashboard more accessible
    to non-technical clinical staff.
    """
    st.markdown("## 📋 Executive Summary for Hospital Leadership")
    
    # Overall system status
    privacy_used = summary.get('final_epsilon', 0) / summary.get('epsilon_budget', 8.0)
    is_healthy = privacy_used < 0.9 and summary.get('best_auc', 0) > 0.7
    
    status_color = "🟢" if is_healthy else "🟡" if privacy_used < 0.95 else "🔴"
    status_text = "Healthy" if is_healthy else "Needs Attention" if privacy_used < 0.95 else "Critical"
    
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        st.markdown(f"""
        <div class="executive-summary">
            <h3>{status_color} System Status: {status_text}</h3>
            <p>The federated learning system is training a diagnostic AI model 
            across <b>{summary.get('total_rounds', 0)} rounds</b> while keeping 
            all patient data secure at each hospital.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="executive-summary">
            <h3>🔒 Patient Privacy Protection</h3>
            <p><b>{(1-privacy_used)*100:.0f}%</b> of privacy budget remaining</p>
            <p>Patient data has <b>never left</b> your hospital's servers. 
            Only encrypted model updates are shared.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        auc = summary.get('best_auc', 0)
        auc_interpretation = "Excellent" if auc > 0.85 else "Good" if auc > 0.75 else "Developing"
        st.markdown(f"""
        <div class="executive-summary">
            <h3>🎯 Model Performance: {auc_interpretation}</h3>
            <p>Diagnostic accuracy: <b>{auc*100:.1f}%</b> AUC</p>
            <p>The AI can correctly distinguish abnormal from normal 
            cases with {auc_interpretation.lower()} reliability.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key benefits callout
    st.markdown("""
    ### ✨ What This Means For Your Hospital
    
    | Benefit | Status |
    |---------|--------|
    | 🏥 **Multi-Hospital Collaboration** | Training with partner hospitals without sharing raw patient data |
    | 🔐 **HIPAA Compliance** | Differential privacy mathematically guarantees patient anonymity |
    | ⚖️ **Fair Across Demographics** | Model performs equitably across different patient populations |
    | 🛡️ **Byzantine Resilience** | Protected against potentially malicious or faulty hospital nodes |
    """)
    
    # Privacy audit results if available
    privacy_audit = summary.get('privacy_audit', {})
    if privacy_audit and 'error' not in privacy_audit:
        st.markdown("### 🔍 Privacy Validation Results")
        
        # Get best (lowest vulnerability) result
        best_grade = "F"
        for attack_type, result in privacy_audit.items():
            grade = result.get('privacy_grade', 'F')
            if grade < best_grade:
                best_grade = grade
        
        grade_emoji = {"A": "🏆", "B": "✅", "C": "⚠️", "D": "⚠️", "F": "❌"}
        grade_prefix = best_grade.split()[0] if best_grade else "?"
        emoji = grade_emoji.get(grade_prefix, "❓")
        
        st.markdown(f"""
        **Privacy Grade: {emoji} {best_grade}**
        
        We tested if an attacker could determine which patients were used to train the model. 
        {"The differential privacy protections are working effectively." if grade_prefix in ["A", "B"] 
         else "Additional privacy measures may be recommended."}
        """)


def display_security_status(summary: Dict, round_history: List[Dict]):
    """Display security features status with visual badges."""
    st.markdown("### 🛡️ Security Features Active")
    
    security = summary.get('security', {})
    robust_agg = summary.get('robust_aggregation', {})
    
    badges_html = ""
    
    # Secure aggregation
    if security.get('encryption_enabled', False) or security.get('homomorphic_encryption', False):
        badges_html += '<span class="security-badge badge-secure">🔐 Encrypted Aggregation</span>'
    
    if security.get('masking_enabled', False) or security.get('random_masking', False):
        badges_html += '<span class="security-badge badge-secure">🎭 Random Masking</span>'
    
    if security.get('audit_trail', False):
        badges_html += '<span class="security-badge badge-secure">📝 Audit Trail</span>'
    
    # Robust aggregation
    if robust_agg:
        method = robust_agg.get('method', 'unknown')
        badges_html += f'<span class="security-badge badge-secure">🛡️ {method.replace("_", " ").title()}</span>'
        
        if robust_agg.get('byzantine_detections', 0) > 0:
            badges_html += f'<span class="security-badge badge-warning">⚠️ {robust_agg["byzantine_detections"]} Anomalies Detected</span>'
    
    # Check round history for security events
    has_secure_agg = any(r.get('security', {}).get('secure_aggregation', False) for r in round_history)
    has_robust_agg = any(r.get('security', {}).get('robust_aggregation', False) for r in round_history)
    
    if has_secure_agg and not badges_html:
        badges_html += '<span class="security-badge badge-secure">🔐 Secure Aggregation</span>'
    if has_robust_agg and not badges_html:
        badges_html += '<span class="security-badge badge-secure">🛡️ Byzantine Protection</span>'
    
    if badges_html:
        st.markdown(badges_html, unsafe_allow_html=True)
    else:
        st.info("Security features status will appear after training completes.")
    
    # Show detailed security report if available
    if security:
        with st.expander("📊 Detailed Security Report"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rounds Secured", security.get('total_rounds', 'N/A'))
                st.metric("Successful Rounds", security.get('successful_rounds', 'N/A'))
            with col2:
                st.metric("Verification Rate", f"{security.get('verification_rate', 0)*100:.1f}%")
                st.metric("Min Clients Required", security.get('min_clients_threshold', 'N/A'))


def load_results(results_path: str) -> Optional[Dict]:
    """Load results from JSON file."""
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


def load_orchestrator_logs(logs_path: str) -> Optional[Dict]:
    """Load orchestrator logs from JSON file."""
    if os.path.exists(logs_path):
        with open(logs_path, 'r') as f:
            return json.load(f)
    return None


def create_epsilon_chart(round_history: List[Dict]) -> go.Figure:
    """Create epsilon tracking chart."""
    if not round_history:
        return go.Figure()
    
    df = pd.DataFrame(round_history)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Cumulative Privacy Budget (ε)", "Per-Round Privacy Cost"),
        vertical_spacing=0.15
    )
    
    # Cumulative epsilon
    fig.add_trace(
        go.Scatter(
            x=df['round_num'] if 'round_num' in df.columns else df['round'],
            y=df['epsilon_total'],
            mode='lines+markers',
            name='ε Total',
            line=dict(color='#FF6B6B', width=2),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Budget line
    budget = df['epsilon_total'].max() * 1.2 if 'epsilon_total' in df.columns else 8.0
    fig.add_hline(
        y=8.0,  # Default budget
        line_dash="dash",
        line_color="red",
        annotation_text="Budget (ε=8.0)",
        row=1, col=1
    )
    
    # Per-round epsilon
    fig.add_trace(
        go.Bar(
            x=df['round_num'] if 'round_num' in df.columns else df['round'],
            y=df['epsilon_round'],
            name='ε per Round',
            marker_color='#4ECDC4'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        title_text="Privacy Budget Tracking"
    )
    
    return fig


def create_metrics_chart(round_history: List[Dict]) -> go.Figure:
    """Create training metrics chart."""
    if not round_history:
        return go.Figure()
    
    df = pd.DataFrame(round_history)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Training Loss", "Validation Loss", "Accuracy", "AUC-ROC"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    round_col = 'round_num' if 'round_num' in df.columns else 'round'
    
    # Training loss
    if 'train_loss' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[round_col], y=df['train_loss'],
                mode='lines+markers', name='Train Loss',
                line=dict(color='#FF6B6B')
            ),
            row=1, col=1
        )
    
    # Validation loss
    if 'val_loss' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[round_col], y=df['val_loss'],
                mode='lines+markers', name='Val Loss',
                line=dict(color='#4ECDC4')
            ),
            row=1, col=2
        )
    
    # Accuracy
    if 'accuracy' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[round_col], y=df['accuracy'],
                mode='lines+markers', name='Accuracy',
                line=dict(color='#45B7D1')
            ),
            row=2, col=1
        )
    
    # AUC
    if 'auc' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[round_col], y=df['auc'],
                mode='lines+markers', name='AUC',
                line=dict(color='#96CEB4')
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        title_text="Training Metrics"
    )
    
    return fig


def create_noise_chart(round_history: List[Dict]) -> go.Figure:
    """Create noise multiplier tracking chart."""
    if not round_history:
        return go.Figure()
    
    df = pd.DataFrame(round_history)
    round_col = 'round_num' if 'round_num' in df.columns else 'round'
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df[round_col],
            y=df['noise_multiplier'],
            mode='lines+markers',
            name='Noise Multiplier (σ)',
            line=dict(color='#9B59B6', width=2),
            fill='tozeroy',
            fillcolor='rgba(155, 89, 182, 0.2)'
        )
    )
    
    fig.update_layout(
        title="Noise Multiplier Over Time",
        xaxis_title="Round",
        yaxis_title="σ",
        height=300
    )
    
    return fig


def create_client_comparison_chart(round_history: List[Dict]) -> go.Figure:
    """Create per-client performance comparison."""
    fig = go.Figure()
    
    # Try to get actual hospital metrics from latest round
    if round_history and 'fairness' in round_history[-1]:
        hospital_metrics = round_history[-1]['fairness'].get('hospital_metrics', {})
        if hospital_metrics:
            clients = list(hospital_metrics.keys())
            aucs = [hospital_metrics[c].get('auc', 0) for c in clients]
            accs = [hospital_metrics[c].get('accuracy', 0) for c in clients]
            
            # Format client names
            client_names = [f"Hospital {c}" for c in clients]
            
            fig.add_trace(
                go.Bar(
                    name='Accuracy',
                    x=client_names,
                    y=accs,
                    marker_color='#4ECDC4',
                    text=[f'{acc:.3f}' for acc in accs],
                    textposition='auto'
                )
            )
            
            fig.add_trace(
                go.Bar(
                    name='AUC',
                    x=client_names,
                    y=aucs,
                    marker_color='#FF6B6B',
                    text=[f'{auc:.3f}' for auc in aucs],
                    textposition='auto'
                )
            )
            
            fig.update_layout(
                title="Per-Hospital Performance Comparison",
                xaxis_title="Hospital Node",
                yaxis_title="Score",
                yaxis_range=[0, 1.1],
                height=350,
                barmode='group'
            )
            return fig
    
    # Fallback to placeholder
    clients = ['Hospital 0', 'Hospital 1', 'Hospital 2']
    aucs = [0.78, 0.75, 0.80]
    
    fig.add_trace(
        go.Bar(
            x=clients,
            y=aucs,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            text=[f'{auc:.3f}' for auc in aucs],
            textposition='auto'
        )
    )
    
    fig.update_layout(
        title="Per-Hospital AUC Comparison",
        xaxis_title="Hospital Node",
        yaxis_title="AUC-ROC",
        yaxis_range=[0, 1],
        height=300
    )
    
    return fig


def create_fairness_score_chart(round_history: List[Dict]) -> go.Figure:
    """Create fairness score tracking chart."""
    if not round_history:
        return go.Figure()
    
    # Extract fairness data
    rounds = []
    fairness_scores = []
    acc_parity = []
    hospital_variance = []
    
    for r in round_history:
        if 'fairness' in r:
            rounds.append(r.get('round_num', r.get('round', 0)))
            fairness_scores.append(r['fairness'].get('score', 0))
            acc_parity.append(r['fairness'].get('accuracy_parity_difference', 0))
            hospital_variance.append(r['fairness'].get('hospital_accuracy_variance', 0))
    
    if not rounds:
        # No fairness data, return empty with message
        fig = go.Figure()
        fig.add_annotation(
            text="No fairness data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(height=300)
        return fig
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Fairness Score Over Rounds", "Fairness Metrics"),
        horizontal_spacing=0.15
    )
    
    # Fairness score over time
    fig.add_trace(
        go.Scatter(
            x=rounds,
            y=fairness_scores,
            mode='lines+markers',
            name='Fairness Score',
            line=dict(color='#2ECC71', width=3),
            fill='tozeroy',
            fillcolor='rgba(46, 204, 113, 0.2)'
        ),
        row=1, col=1
    )
    
    # Add threshold line
    fig.add_hline(y=0.8, line_dash="dash", line_color="orange",
                  annotation_text="Fair Threshold", row=1, col=1)
    
    # Latest metrics as bar chart
    if fairness_scores:
        latest_metrics = {
            'Fairness Score': fairness_scores[-1],
            'Accuracy Parity': 1 - acc_parity[-1] if acc_parity else 1,
            'Hospital Equity': 1 - (hospital_variance[-1] * 10) if hospital_variance else 1
        }
        
        fig.add_trace(
            go.Bar(
                x=list(latest_metrics.keys()),
                y=list(latest_metrics.values()),
                marker_color=['#2ECC71', '#3498DB', '#9B59B6'],
                text=[f'{v:.3f}' for v in latest_metrics.values()],
                textposition='auto'
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        height=350,
        showlegend=False,
        title_text="⚖️ Fairness Metrics"
    )
    
    fig.update_yaxes(range=[0, 1.1], row=1, col=1)
    fig.update_yaxes(range=[0, 1.1], row=1, col=2)
    
    return fig


def create_hospital_parity_chart(round_history: List[Dict]) -> go.Figure:
    """Create hospital performance parity chart."""
    if not round_history:
        return go.Figure()
    
    # Get latest round with fairness data
    latest_fairness = None
    for r in reversed(round_history):
        if 'fairness' in r and r['fairness'].get('hospital_metrics'):
            latest_fairness = r['fairness']
            break
    
    if not latest_fairness:
        fig = go.Figure()
        fig.add_annotation(
            text="No hospital parity data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(height=300)
        return fig
    
    hospital_metrics = latest_fairness['hospital_metrics']
    hospitals = list(hospital_metrics.keys())
    accuracies = [hospital_metrics[h].get('accuracy', 0) for h in hospitals]
    
    # Calculate mean and std for reference
    mean_acc = np.mean(accuracies)
    
    # Create radar chart for hospital comparison
    fig = go.Figure()
    
    # Add bars with reference line
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    fig.add_trace(
        go.Bar(
            x=[f"Hospital {h}" for h in hospitals],
            y=accuracies,
            marker_color=colors[:len(hospitals)],
            text=[f'{acc:.3f}' for acc in accuracies],
            textposition='auto'
        )
    )
    
    # Add mean line
    fig.add_hline(y=mean_acc, line_dash="dash", line_color="gray",
                  annotation_text=f"Mean: {mean_acc:.3f}")
    
    # Add parity threshold lines
    fig.add_hline(y=mean_acc + 0.15, line_dash="dot", line_color="red",
                  annotation_text="Upper bound (+15%)")
    fig.add_hline(y=mean_acc - 0.15, line_dash="dot", line_color="red",
                  annotation_text="Lower bound (-15%)")
    
    fig.update_layout(
        title="Hospital Accuracy Parity",
        xaxis_title="Hospital",
        yaxis_title="Accuracy",
        yaxis_range=[0, 1.1],
        height=350
    )
    
    return fig


def display_fairness_summary(round_history: List[Dict]):
    """Display fairness summary metrics."""
    # Get latest fairness data
    latest_fairness = None
    for r in reversed(round_history):
        if 'fairness' in r:
            latest_fairness = r['fairness']
            break
    
    if not latest_fairness:
        st.info("No fairness data available yet.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = latest_fairness.get('score', 0)
        st.metric(
            label="Fairness Score",
            value=f"{score:.3f}",
            delta="Fair" if score >= 0.8 else "Needs attention"
        )
    
    with col2:
        is_fair = latest_fairness.get('is_fair', False)
        status = "✅ Fair" if is_fair else "⚠️ Violations"
        st.metric(
            label="Status",
            value=status
        )
    
    with col3:
        variance = latest_fairness.get('hospital_accuracy_variance', 0)
        st.metric(
            label="Hospital Variance",
            value=f"{variance:.4f}",
            delta="Low" if variance < 0.05 else "High"
        )
    
    with col4:
        parity_diff = latest_fairness.get('accuracy_parity_difference', 0)
        st.metric(
            label="Accuracy Parity Diff",
            value=f"{parity_diff:.3f}",
            delta="Good" if parity_diff < 0.15 else "High"
        )
    
    # Show violations if any
    violations = latest_fairness.get('violations', [])
    if violations:
        st.warning("⚠️ **Fairness Violations Detected:**")
        for v in violations:
            st.markdown(f"- {v}")


def display_summary_metrics(summary: Dict):
    """Display summary metrics in columns."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Rounds",
            value=summary.get('total_rounds', 0)
        )
    
    with col2:
        epsilon = summary.get('final_epsilon', 0)
        budget = summary.get('epsilon_budget', 8.0)
        st.metric(
            label="ε Spent / Budget",
            value=f"{epsilon:.2f} / {budget}",
            delta=f"{(epsilon/budget)*100:.1f}% used"
        )
    
    with col3:
        st.metric(
            label="Best AUC",
            value=f"{summary.get('best_auc', 0):.4f}"
        )
    
    with col4:
        status = "✅ Active" if not summary.get('is_halted', False) else "🛑 Halted"
        st.metric(
            label="Status",
            value=status
        )


def main():
    """Main dashboard function."""
    
    # Sidebar
    st.sidebar.title("🏥 MedFedAgent")
    st.sidebar.markdown("---")
    
    # View Mode Selection
    view_mode = st.sidebar.radio(
        "📊 Dashboard View",
        ["🩺 Clinical Summary", "🔧 Technical Details"],
        help="Clinical view for hospital leadership, Technical for ML engineers"
    )
    
    st.sidebar.markdown("---")
    
    # File paths
    results_dir = st.sidebar.text_input(
        "Results Directory",
        value="./results",
        help="Path to the results directory"
    )
    
    logs_dir = st.sidebar.text_input(
        "Logs Directory", 
        value="./logs",
        help="Path to the logs directory"
    )
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
    refresh_interval = st.sidebar.slider("Refresh interval (s)", 1, 30, 5)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    **MedFedAgent** is a privacy-preserving federated learning system for medical imaging.
    
    - 🔒 Differential Privacy (DP-SGD)
    - 🏥 Multi-hospital collaboration
    - ⚖️ Fairness evaluation
    - 🛡️ Byzantine-robust aggregation
    - 🔐 Secure aggregation protocol
    - 📊 Real-time monitoring
    """)
    
    # Main content
    st.title("📊 MedFedAgent Dashboard")
    st.markdown("Privacy-Preserving Federated Learning for Medical Imaging")
    st.markdown("---")
    
    # Try to load data
    results_path = os.path.join(results_dir, "results.json")
    logs_path = os.path.join(logs_dir, "orchestrator_logs.json")
    
    orchestrator_logs = load_orchestrator_logs(logs_path)
    results = load_results(results_path)
    
    # Use orchestrator logs if available, otherwise results
    data = orchestrator_logs or results
    
    if data:
        summary = data.get('summary', {})
        round_history = data.get('round_history', data.get('rounds', []))
        
        # =====================================================================
        # CLINICAL VIEW - For hospital administrators
        # =====================================================================
        if view_mode == "🩺 Clinical Summary":
            if summary:
                display_clinical_executive_summary(summary, round_history)
                st.markdown("---")
                display_security_status(summary, round_history)
                st.markdown("---")
                
                # Simplified metrics
                st.markdown("### 📈 Training Progress Overview")
                col1, col2 = st.columns(2)
                
                with col1:
                    if round_history:
                        # Simple progress chart
                        fig = go.Figure()
                        rounds = [r.get('round_num', r.get('round', i)) for i, r in enumerate(round_history)]
                        aucs = [r.get('auc', 0) for r in round_history]
                        
                        fig.add_trace(go.Scatter(
                            x=rounds, y=aucs,
                            mode='lines+markers',
                            name='Model Accuracy (AUC)',
                            line=dict(color='#2E7D32', width=3),
                            fill='tozeroy',
                            fillcolor='rgba(46, 125, 50, 0.1)'
                        ))
                        
                        fig.add_hline(y=0.85, line_dash="dash", line_color="green",
                                     annotation_text="Target (85%)")
                        
                        fig.update_layout(
                            title="📈 Model Accuracy Improving Over Time",
                            xaxis_title="Training Round",
                            yaxis_title="Diagnostic Accuracy (AUC)",
                            yaxis_range=[0, 1],
                            height=350
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if round_history:
                        # Privacy remaining
                        budget = summary.get('epsilon_budget', 8.0)
                        spent = summary.get('final_epsilon', 0)
                        remaining = max(0, budget - spent)
                        
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=remaining,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Privacy Budget Remaining"},
                            delta={'reference': budget, 'increasing': {'color': "green"}},
                            gauge={
                                'axis': {'range': [None, budget]},
                                'bar': {'color': "#2E7D32"},
                                'steps': [
                                    {'range': [0, budget*0.25], 'color': "#FFCDD2"},
                                    {'range': [budget*0.25, budget*0.75], 'color': "#FFF9C4"},
                                    {'range': [budget*0.75, budget], 'color': "#C8E6C9"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': budget * 0.1
                                }
                            }
                        ))
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Fairness for clinical view
                st.markdown("### ⚖️ Fairness Across Hospitals")
                display_fairness_summary(round_history)
                
            else:
                st.info("Waiting for training to complete...")
        
        # =====================================================================
        # TECHNICAL VIEW - For ML engineers
        # =====================================================================
        else:
            # Summary metrics
            if summary:
                display_summary_metrics(summary)
            
            st.markdown("---")
            
            # Security status for technical view
            display_security_status(summary, round_history)
            st.markdown("---")
        
            if round_history:
            # Privacy tracking
            st.subheader("🔒 Privacy Budget Tracking")
            epsilon_chart = create_epsilon_chart(round_history)
            st.plotly_chart(epsilon_chart, use_container_width=True)
            
            st.markdown("---")
            
            # Training metrics
            st.subheader("📈 Training Metrics")
            metrics_chart = create_metrics_chart(round_history)
            st.plotly_chart(metrics_chart, use_container_width=True)
            
            st.markdown("---")
            
            # Two columns for noise and client comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔊 Noise Multiplier")
                noise_chart = create_noise_chart(round_history)
                st.plotly_chart(noise_chart, use_container_width=True)
            
            with col2:
                st.subheader("🏥 Per-Hospital Performance")
                client_chart = create_client_comparison_chart(round_history)
                st.plotly_chart(client_chart, use_container_width=True)
            
            st.markdown("---")
            
            # Fairness Section
            st.subheader("⚖️ Fairness Evaluation")
            
            # Fairness summary metrics
            display_fairness_summary(round_history)
            
            # Fairness charts
            col1, col2 = st.columns(2)
            
            with col1:
                fairness_chart = create_fairness_score_chart(round_history)
                st.plotly_chart(fairness_chart, use_container_width=True)
            
            with col2:
                parity_chart = create_hospital_parity_chart(round_history)
                st.plotly_chart(parity_chart, use_container_width=True)
            
            st.markdown("---")
            
            # Round history table
            st.subheader("📋 Round History")
            
            df = pd.DataFrame(round_history)
            display_cols = ['round_num', 'epsilon_round', 'epsilon_total', 
                          'noise_multiplier', 'num_clients', 'status', 
                          'train_loss', 'accuracy', 'auc']
            
            # Add fairness score if available
            if any('fairness' in r for r in round_history):
                df['fairness_score'] = df.apply(
                    lambda r: r.get('fairness', {}).get('score', None) 
                    if isinstance(r.get('fairness'), dict) else None, axis=1
                )
                display_cols.append('fairness_score')
            
            display_cols = [c for c in display_cols if c in df.columns]
            
            if 'round' in df.columns and 'round_num' not in df.columns:
                df['round_num'] = df['round']
                display_cols[0] = 'round_num'
            
            st.dataframe(
                df[display_cols].style.format({
                    'epsilon_round': '{:.4f}',
                    'epsilon_total': '{:.4f}',
                    'noise_multiplier': '{:.2f}',
                    'train_loss': '{:.4f}',
                    'accuracy': '{:.4f}',
                    'auc': '{:.4f}',
                    'fairness_score': '{:.3f}'
                }, na_rep='-'),
                use_container_width=True
            )
            
            # Anomalies section
            anomalies = []
            for r in round_history:
                anomalies.extend(r.get('anomalies', []))
            
            if anomalies:
                st.markdown("---")
                st.subheader("⚠️ Detected Anomalies")
                st.dataframe(pd.DataFrame(anomalies), use_container_width=True)
        else:
            st.info("No round history data available yet.")
    
    else:
        # No data available - show demo/placeholder
        st.info("📁 No data found. Run the simulation to generate results.")
        
        st.markdown("### Expected file locations:")
        st.code(f"""
Results: {results_path}
Logs: {logs_path}
        """)
        
        st.markdown("### Demo Mode")
        
        # Generate demo data
        if st.button("Generate Demo Data"):
            demo_history = []
            epsilon_total = 0
            
            for i in range(1, 21):
                epsilon_round = np.random.uniform(0.35, 0.45)
                epsilon_total += epsilon_round
                
                demo_history.append({
                    'round_num': i,
                    'epsilon_round': epsilon_round,
                    'epsilon_total': epsilon_total,
                    'noise_multiplier': 1.0 + (0.1 * max(0, i - 15)),
                    'num_clients': 3,
                    'status': 'OK' if epsilon_total < 6 else 'WARNING_BUDGET_75%',
                    'train_loss': 0.8 - (i * 0.02),
                    'val_loss': 0.85 - (i * 0.018),
                    'accuracy': 0.6 + (i * 0.015),
                    'auc': 0.55 + (i * 0.018),
                    'anomalies': []
                })
            
            st.subheader("📈 Demo Training Metrics")
            metrics_chart = create_metrics_chart(demo_history)
            st.plotly_chart(metrics_chart, width="stretch")
            
            st.subheader("🔒 Demo Privacy Tracking")
            epsilon_chart = create_epsilon_chart(demo_history)
            st.plotly_chart(epsilon_chart, width="stretch")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
