# MedFedAgent: Technical Documentation

## Privacy-Preserving Federated Learning for Medical Imaging

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Privacy Technologies](#privacy-technologies)
5. [Security Features](#security-features)
6. [API Reference](#api-reference)
7. [Configuration Guide](#configuration-guide)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### What is MedFedAgent?

MedFedAgent is a privacy-preserving federated learning system designed for medical imaging classification. It enables multiple hospitals to collaboratively train diagnostic AI models **without sharing patient data**.

### Key Innovation

Traditional machine learning requires centralizing data, which creates:
- **Privacy risks**: Patient data could be exposed
- **Legal barriers**: HIPAA/GDPR compliance challenges
- **Institutional resistance**: Hospitals won't share data

**MedFedAgent solves this** by:
1. **Keeping data local**: Each hospital's data never leaves their servers
2. **Training collaboratively**: Only model updates (not data) are shared
3. **Mathematical privacy**: Differential privacy provides provable guarantees
4. **Byzantine resilience**: Robust to malicious or faulty participants

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Federated Learning | Custom Implementation | Distributed training |
| Differential Privacy | Opacus (DP-SGD) | Privacy guarantees |
| Model Architecture | PyTorch (CNN) | Medical image classification |
| Secure Aggregation | Custom Implementation | Protect model updates |
| Byzantine Robustness | Krum, Bulyan, etc. | Handle malicious nodes |
| Dashboard | Flask + JavaScript | Real-time monitoring |

---

## System Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      MEDFEDAGENT SYSTEM                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    PRESENTATION LAYER                      │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │  │
│  │  │    Flask     │  │  REST API    │  │  JSON Reports    │  │  │
│  │  │  Dashboard   │  │  Endpoints   │  │                  │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                               │                                  │
│                               ▼                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                   ORCHESTRATION LAYER                      │  │
│  │  ┌─────────────────────┐  ┌─────────────────────────────┐  │  │
│  │  │  Privacy            │  │  Fairness Evaluator         │  │  │
│  │  │  Orchestrator       │  │  • Hospital Parity          │  │  │
│  │  │  • Budget Tracking  │  │  • Demographic Fairness     │  │  │
│  │  │  • Noise Adjustment │  │                             │  │  │
│  │  └─────────────────────┘  └─────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                               │                                  │
│                               ▼                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    FEDERATED LAYER                         │  │
│  │  ┌─────────────────┐  ┌─────────────────────────────────┐  │  │
│  │  │  Training       │  │  Robust Aggregation             │  │  │
│  │  │  Engine         │  │  • FedAvg, Median, Krum, etc.   │  │  │
│  │  │                 │  │  • Byzantine Detection          │  │  │
│  │  └─────────────────┘  └─────────────────────────────────┘  │  │
│  │                                                            │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │                FL CLIENTS (Hospitals)                 │  │  │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐      │  │  │
│  │  │  │ Hospital A │  │ Hospital B │  │ Hospital C │ ...  │  │  │
│  │  │  └────────────┘  └────────────┘  └────────────┘      │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                               │                                  │
│                               ▼                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                      PRIVACY LAYER                         │  │
│  │  ┌─────────────────┐  ┌────────────────┐  ┌─────────────┐  │  │
│  │  │  DP-SGD Trainer │  │  Secure        │  │  MIA        │  │  │
│  │  │  • Opacus       │  │  Aggregation   │  │  Testing    │  │  │
│  │  │  • Grad Clip    │  │  • Masking     │  │             │  │  │
│  │  └─────────────────┘  └────────────────┘  └─────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Training Round Workflow

1. **Initialize Round**: Orchestrator checks privacy budget, adjusts noise if needed
2. **Distribute Model**: Server sends current global model to all clients
3. **Local Training**: Each client trains on local data with DP-SGD
4. **Secure Upload**: Clients send noisy model updates with optional masking
5. **Robust Aggregation**: Server aggregates updates using Byzantine-robust methods
6. **Update & Log**: Update global model, log metrics, evaluate performance
7. **Repeat**: Continue until rounds complete or budget exhausted

---

## Core Components

### 1. Model Architectures (`src/models/model.py`)

Supported CNN architectures for chest X-ray classification:

| Model | Parameters | Use Case | Opacus Compatible |
|-------|------------|----------|-------------------|
| **SimpleCNN** | ~1.5M | Fast training, demos | ✅ Native |
| **ResNet-18** | ~11M | Better accuracy | ✅ Auto-fixed |
| **DenseNet-121** | ~7M | Best accuracy | ✅ Auto-fixed |

**Usage:**
```python
from src.models.model import create_model

model = create_model(
    model_name="simple_cnn",
    num_classes=2,
    pretrained=False,
    opacus_compatible=True
)
```

### 2. Data Pipeline (`src/data/dataset.py`)

**Features:**
- Synthetic medical imaging data generation
- Non-IID data partitioning via Dirichlet distribution
- Hospital-specific domain shift simulation
- Label noise injection for realistic scenarios

**Usage:**
```python
from src.data.dataset import load_federated_datasets

train_loaders, val_loaders, test_loader = load_federated_datasets(
    num_clients=3,
    samples_per_client=600,
    non_iid=True,
    alpha=0.5,
    use_realistic=True
)
```

### 3. Federated Client (`src/federated/client.py`)

**Features:**
- Local training with differential privacy
- Model update generation
- Evaluation on local validation data

### 4. Privacy Orchestrator (`src/orchestrator/orchestrator.py`)

**Features:**
- Privacy budget (ε) tracking
- Automatic noise adjustment when approaching budget limits
- Anomaly detection for suspicious updates
- Round-by-round logging

### 5. Training Engine (`src/dashboard/training_engine.py`)

**Features:**
- Native Python training integration
- Real-time metrics streaming
- Start/stop/pause control
- Comprehensive logging

---

## Privacy Technologies

### Differential Privacy (DP-SGD)

MedFedAgent uses Opacus for differential privacy:

```
DP-SGD Process:
1. Compute per-sample gradients
2. Clip gradients to max norm C
3. Add Gaussian noise with scale σ
4. Aggregate and update model
```

**Key Parameters:**
- `epsilon_budget`: Total privacy budget (typical: 10-100)
- `noise_multiplier`: Noise scale σ (typical: 0.5-2.0)
- `max_grad_norm`: Gradient clipping C (typical: 1.0)

### Secure Aggregation (`src/privacy/secure_aggregation.py`)

**Features:**
- Random masking of model updates
- Simulated secret sharing
- Cryptographic audit trails
- Verification of aggregation integrity

### Membership Inference Attack (`src/privacy/mia_attack.py`)

**Attack Methods:**
- Threshold attack
- Loss-based attack

**Interpretation:**
- AUC ≈ 0.5: Attacker no better than random (good privacy)
- AUC > 0.6: Potential privacy leakage (needs tuning)

---

## Security Features

### Byzantine-Robust Aggregation (`src/federated/robust_aggregation.py`)

| Method | Description | Best For |
|--------|-------------|----------|
| **FedAvg** | Weighted averaging | Trusted environments |
| **Median** | Coordinate-wise median | Single Byzantine |
| **Trimmed Mean** | Remove outliers, average | Multiple outliers |
| **Krum** | Select most representative | Few Byzantine |
| **Multi-Krum** | Select k most representative | Multiple Byzantine |
| **Bulyan** | Krum + trimmed mean | Strong Byzantine |
| **FoolsGold** | Weight by diversity | Sybil attacks |

### Anomaly Detection

- Z-score based gradient norm detection
- Configurable threshold (default: 3.0)
- Automatic flagging of suspicious updates

---

## API Reference

### REST Endpoints

#### Dashboard Data
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/data` | GET | All dashboard data |
| `/api/metrics` | GET | Training metrics only |

#### Training Control
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/training/start` | POST | Start training |
| `/api/training/stop` | POST | Stop training |
| `/api/training/pause` | POST | Pause training |
| `/api/training/resume` | POST | Resume training |
| `/api/training/reset` | POST | Reset training state |
| `/api/training/status` | GET | Get training status |
| `/api/training/logs` | GET | Get training logs |

#### Configuration
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/training/config` | GET | Get current config |
| `/api/training/config` | POST | Update config |

### Example API Usage

```python
import requests

# Start training
response = requests.post('http://127.0.0.1:5000/api/training/start', json={
    'num_rounds': 10,
    'num_clients': 3,
    'epsilon_budget': 50
})

# Get status
status = requests.get('http://127.0.0.1:5000/api/training/status').json()
print(f"Round: {status['state']['current_round']}/{status['state']['total_rounds']}")
```

---

## Configuration Guide

### Main Configuration (`config/config.yaml`)

```yaml
# =============================================================================
# FEDERATED LEARNING
# =============================================================================
federated:
  num_clients: 3              # Number of hospital nodes
  num_rounds: 10              # Training rounds
  fraction_fit: 1.0           # Fraction of clients per round
  min_fit_clients: 3          # Minimum clients required

# =============================================================================
# MODEL
# =============================================================================
model:
  name: simple_cnn            # simple_cnn, resnet18, densenet121
  num_classes: 2              # Binary classification
  dropout: 0.3                # Dropout rate
  pretrained: false           # Use pretrained weights

# =============================================================================
# TRAINING
# =============================================================================
training:
  local_epochs: 2             # Epochs per client per round
  local_steps: 50             # Steps per epoch (if using steps)
  batch_size: 64              # Batch size
  learning_rate: 0.01         # Learning rate
  weight_decay: 0.0001        # L2 regularization
  optimizer: adam             # adam, sgd
  grad_clip_norm: 1.5         # Gradient clipping

# =============================================================================
# DATA
# =============================================================================
data:
  image_size: 128             # Image size (smaller = faster)
  samples_per_client: 600     # Samples per hospital
  use_realistic: true         # Use challenging dataset
  label_noise: 0.08           # Label noise fraction
  class_overlap: 0.25         # Class overlap level
  non_iid:
    enabled: true
    alpha: 0.5                # Dirichlet concentration (lower = more non-IID)

# =============================================================================
# PRIVACY (DIFFERENTIAL PRIVACY)
# =============================================================================
privacy:
  enabled: true
  epsilon_budget: 50          # Total privacy budget (ε)
  delta: 1.0e-05              # Delta parameter
  max_grad_norm: 1.0          # Gradient clipping (C)
  noise_multiplier: 0.8       # Noise scale (σ)
  secure_mode: false          # Use secure RNG

# =============================================================================
# ORCHESTRATOR
# =============================================================================
orchestrator:
  budget_warning_threshold: 0.75    # Warn at 75% budget
  budget_critical_threshold: 0.9    # Critical at 90% budget
  noise_increase_factor: 1.1        # Noise increase multiplier
  enable_anomaly_detection: true
  anomaly_zscore_threshold: 3.0

# =============================================================================
# SECURE AGGREGATION
# =============================================================================
secure_aggregation:
  enabled: true
  min_clients: 2
  threshold: 2
  use_masking: true
  simulate_encryption: true
  audit_enabled: true

# =============================================================================
# ROBUST AGGREGATION
# =============================================================================
robust_aggregation:
  enabled: true
  method: trimmed_mean        # fedavg, median, trimmed_mean, krum, etc.
  num_byzantine: 0            # Expected Byzantine clients
  trim_ratio: 0.1             # Trim ratio for trimmed mean
  enable_detection: true
  detection_threshold: 3.0
```

---

## Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError: No module named 'flask'"
```bash
# Activate virtual environment first
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate     # Linux/macOS

# Install Flask
pip install flask
```

#### 2. Training is slow
- Reduce `image_size` to 128 or 64
- Reduce `samples_per_client`
- Reduce `num_rounds`
- Use `simple_cnn` model

#### 3. Privacy budget exhausted too quickly
- Increase `epsilon_budget`
- Reduce `noise_multiplier`
- Reduce `num_rounds`

#### 4. Poor model accuracy
- Increase `epsilon_budget` (more budget = less noise)
- Increase `learning_rate`
- Use more `local_epochs`
- Try different `model` architecture

#### 5. Dashboard not loading
- Ensure Flask server is running
- Check if port 5000 is available
- Check browser console for errors

### Logs and Debugging

**Log Locations:**
- Training logs: `logs/orchestrator_logs.json`
- Secure aggregation audit: `logs/secure_agg_audit_*.json`
- Console output: Real-time in terminal

**Enable Debug Mode:**
```bash
python run_flask_dashboard.py --debug
```

---

## Performance Tuning

### For Speed
```yaml
data:
  image_size: 64
  samples_per_client: 300
model:
  name: simple_cnn
training:
  local_epochs: 1
  batch_size: 128
federated:
  num_rounds: 5
```

### For Accuracy
```yaml
data:
  image_size: 224
  samples_per_client: 1000
model:
  name: densenet121
  pretrained: true
training:
  local_epochs: 3
  batch_size: 32
federated:
  num_rounds: 20
privacy:
  epsilon_budget: 100
  noise_multiplier: 0.5
```

### For Privacy
```yaml
privacy:
  epsilon_budget: 10
  noise_multiplier: 1.5
  max_grad_norm: 0.5
secure_aggregation:
  enabled: true
  use_masking: true
```

---

## Glossary

| Term | Definition |
|------|------------|
| **Differential Privacy (DP)** | Mathematical framework for privacy guarantees |
| **Epsilon (ε)** | Privacy loss parameter; lower = more private |
| **Delta (δ)** | Probability of privacy breach |
| **DP-SGD** | Differentially private stochastic gradient descent |
| **Federated Learning** | Training ML models across decentralized data |
| **Non-IID** | Non-Independent and Identically Distributed data |
| **Byzantine Fault** | Malicious or arbitrary failures in distributed systems |
| **Secure Aggregation** | Protocol to aggregate data without revealing individual inputs |
| **MIA** | Membership Inference Attack - test if data was in training set |

---

## References

- [Opacus Documentation](https://opacus.ai/)
- [Differential Privacy Primer](https://privacytools.seas.harvard.edu/differential-privacy)
- [Federated Learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
- [Byzantine-Robust Aggregation](https://arxiv.org/abs/1703.02757)
