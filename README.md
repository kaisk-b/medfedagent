# MedFedAgent 🏥

## Privacy-Preserving Federated Learning for Medical Imaging

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MedFedAgent is a **production-ready** privacy-preserving federated learning system designed for medical imaging classification. It enables multiple hospitals to collaboratively train diagnostic AI models **without sharing patient data**, using differential privacy, secure aggregation, and Byzantine-robust protocols.

![Dashboard Preview](https://via.placeholder.com/800x400?text=MedFedAgent+Dashboard)

---

## 🎯 Key Features

### 🔗 Federated Learning
- **Multi-Hospital Training**: Train across multiple hospital nodes without data sharing
- **Non-IID Data Handling**: Realistic data distribution via Dirichlet allocation
- **Model Architectures**: DenseNet-121, ResNet-18, and SimpleCNN support

### 🔐 Privacy Protection
- **Differential Privacy (DP-SGD)**: Per-sample gradient clipping + calibrated noise via Opacus
- **Privacy Budget Orchestration**: Automatic noise adjustment when approaching ε budget
- **Membership Inference Attack (MIA)**: Post-training privacy validation
- **Secure Aggregation**: Cryptographic protection with random masking

### 🛡️ Security & Robustness
- **Byzantine-Robust Aggregation**: 7 methods (FedAvg, Median, Trimmed Mean, Krum, Multi-Krum, Bulyan, FoolsGold)
- **Anomaly Detection**: Gradient norm outlier detection with Z-score thresholds
- **Audit Trails**: Full cryptographic audit logging for compliance

### ⚖️ Fairness & Monitoring
- **Fairness Evaluation**: Hospital parity and demographic fairness metrics
- **Real-time Dashboard**: Modern Flask-based web dashboard with live training metrics
- **Executive Summary**: Hospital leadership-friendly reporting

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/medfedagent.git
cd medfedagent

# Create and activate virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

#### Option 1: Quick Demo
```bash
python run_demo.py
```

#### Option 2: Full Simulation with Dashboard
```bash
# Start the Flask dashboard
python run_flask_dashboard.py

# In another terminal, run the simulation
python run_simulation.py
```

#### Option 3: Dashboard Only (with existing data)
```bash
python run_flask_dashboard.py
# Open http://127.0.0.1:5000 in your browser
```

---

## 📁 Project Structure

```
medfedagent/
├── config/
│   └── config.yaml                 # Main configuration file
├── src/
│   ├── models/
│   │   └── model.py                # Neural network architectures
│   ├── data/
│   │   └── dataset.py              # Dataset loading & non-IID partitioning
│   ├── privacy/
│   │   ├── dp_trainer.py           # DP-SGD trainer with Opacus
│   │   ├── mia_attack.py           # Membership Inference Attack
│   │   └── secure_aggregation.py   # Secure aggregation protocol
│   ├── federated/
│   │   ├── client.py               # Federated learning client
│   │   ├── server.py               # Federated learning server
│   │   └── robust_aggregation.py   # Byzantine-robust aggregation
│   ├── orchestrator/
│   │   └── orchestrator.py         # Privacy budget orchestrator
│   ├── fairness/
│   │   ├── evaluator.py            # Fairness evaluation
│   │   └── metrics.py              # Fairness metrics
│   └── dashboard/
│       ├── flask_app.py            # Flask web application
│       ├── training_engine.py      # Native training integration
│       ├── templates/              # HTML templates
│       └── static/                 # CSS and JavaScript
├── logs/                           # Training logs (auto-generated)
├── results/                        # Results and models (auto-generated)
├── run_flask_dashboard.py          # Dashboard entry point
├── run_simulation.py               # Main simulation runner
├── run_demo.py                     # Quick demo script
├── run_privacy_audit.py            # Standalone privacy audit
└── requirements.txt                # Python dependencies
```

---

## 🖥️ Dashboard Pages

| Page | URL | Description |
|------|-----|-------------|
| **Overview** | `/` | Training metrics, privacy budget, system status |
| **Clinical** | `/clinical` | Executive summary for hospital leadership |
| **Technical** | `/technical` | Detailed ML metrics and performance graphs |
| **Privacy** | `/privacy` | Privacy budget tracking, MIA results |
| **Fairness** | `/fairness` | Hospital parity and demographic fairness |
| **Training** | `/training` | Start/stop training, configure parameters |

---

## ⚙️ Configuration

Edit `config/config.yaml` to customize all aspects of the system:

```yaml
# Federated Learning
federated:
  num_clients: 3              # Number of hospital nodes
  num_rounds: 10              # Training rounds

# Model
model:
  name: simple_cnn            # densenet121, resnet18, simple_cnn
  num_classes: 2              # Binary classification

# Training
training:
  batch_size: 64
  learning_rate: 0.01
  local_epochs: 2

# Privacy (Differential Privacy)
privacy:
  enabled: true
  epsilon_budget: 50.0        # Total privacy budget (ε)
  noise_multiplier: 0.8       # Noise scale (σ)
  max_grad_norm: 1.0          # Gradient clipping (C)

# Secure Aggregation
secure_aggregation:
  enabled: true
  use_masking: true
  audit_enabled: true

# Byzantine-Robust Aggregation
robust_aggregation:
  enabled: true
  method: trimmed_mean        # fedavg, median, trimmed_mean, krum, etc.
  enable_detection: true

# Fairness
fairness:
  thresholds:
    accuracy_parity: 0.15
    hospital_variance: 0.05
```

---

## 🔒 Privacy & Security Stack

### Differential Privacy (DP-SGD)
- Per-sample gradient clipping with configurable max norm
- Calibrated Gaussian noise injection
- RDP-based epsilon accounting via Opacus
- Automatic noise adjustment near budget exhaustion

### Secure Aggregation
- Simulated Shamir-style secret sharing
- Random masking for model updates
- Cryptographic audit trails
- Verification of aggregation integrity

### Byzantine-Robust Aggregation
| Method | Description |
|--------|-------------|
| FedAvg | Standard weighted averaging |
| Median | Coordinate-wise median |
| Trimmed Mean | Remove outliers, then average |
| Krum | Select most representative update |
| Multi-Krum | Select k most representative |
| Bulyan | Krum + trimmed mean hybrid |
| FoolsGold | Weight by update diversity |

---

## 📊 API Endpoints

The Flask backend provides RESTful API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/data` | GET | Get all dashboard data |
| `/api/metrics` | GET | Get training metrics |
| `/api/training/start` | POST | Start training |
| `/api/training/stop` | POST | Stop training |
| `/api/training/status` | GET | Get training status |
| `/api/training/logs` | GET | Get training logs |
| `/api/training/config` | GET/POST | Get/update configuration |

---

## 📈 Sample Output

After training completes, you'll see:

```
======================================================================
PRIVACY AUDIT REPORT - Membership Inference Attack Results
======================================================================

--- THRESHOLD Attack ---
  Accuracy:           0.5123
  AUC-ROC:            0.5089
  Privacy Grade:      A (Excellent)
  Is Vulnerable:      No

--- LOSS_BASED Attack ---
  Accuracy:           0.5234
  AUC-ROC:            0.5156
  Privacy Grade:      A (Excellent)
  Is Vulnerable:      No

INTERPRETATION: AUC ≈ 0.5 means attacker is no better than random guessing
               The differential privacy is working effectively!
======================================================================
```

---

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
# Format code
black src/

# Check types
mypy src/
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- [Opacus](https://opacus.ai/) - Differential Privacy library for PyTorch
- [Flower](https://flower.dev/) - Federated Learning framework
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [PyTorch](https://pytorch.org/) - Deep Learning framework
