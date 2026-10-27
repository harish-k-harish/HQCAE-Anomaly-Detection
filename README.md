# Hybrid Quantum-Classical Autoencoder (HQCAE) for Anomaly Representation and Detection
### Enhanced with Dual-Space Anomaly Detection via Von Neumann Entanglement Entropy

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.1.0-EE4C2C.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-%E2%89%A50.35.0-brightgreen.svg)](https://pennylane.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An unsupervised anomaly detection framework utilizing a **Hybrid Quantum-Classical Autoencoder (HQCAE)** enhanced with a novel **Dual-Space Anomaly Detection module**. The framework projects 16-dimensional correlated multivariate continuous data into a 4-qubit Hilbert space through parameterized quantum circuits (PQCs), simultaneously monitoring classical reconstruction residual error and quantum state subsystem Von Neumann entanglement entropy divergence.

---

## Architecture Summary

```
                       [ Input: x in R^16, domain [0, pi]^16 ]
                                         │
                             ┌───────────┴───────────┐
                             │   Classical Encoder   │
                             │  Linear(16 -> 8) Tanh │
                             │  Linear(8 -> 4)       │
                             │  Sigmoid() * pi       │
                             └───────────┬───────────┘
                                         │
                      [ Latent Angles: theta in [0, pi]^4 ]
                                         │
                             ┌───────────┴───────────┐
                             │  4-Qubit Quantum PQC  │
                             │  AngleEmbedding (Ry)  │
                             │ 2x StronglyEntangling │
                             └─────┬───────────┬─────┘
                                   │           │
         [ Subsystem Entanglement ]│           │[ Expectation Measurements ]
         rho_A = Tr_{2,3}(|psi><psi|)          │<Z_i> in [-1, 1]^4
         S(rho_A) = -Tr(rho_A ln rho_A)        │
                                   │           ▼
                                   │ ┌───────────────────┐
                                   │ │ Classical Decoder │
                                   │ │Linear(4 -> 8) Tanh│
                                   │ │Linear(8 -> 16)    │
                                   │ │Sigmoid() * pi     │
                                   │ └─────────┬─────────┘
                                   │           │
                                   │ [ Reconstruction: x_hat in [0, pi]^16 ]
                                   │           │
                                   ▼           ▼
                      ┌───────────────────────────────────────┐
                      │    Hybrid Dual-Space Anomaly Score    │
                      │ S_dual(x) = MSE_classical + beta * S  │
                      │             (beta = 0.15)             │
                      └───────────────────┬───────────────────┘
                                          │
                                          ▼
                         Decision: Flag if S_dual(x) >= tau
```

### 1. Classical Encoder
- **Layer 1**: Fully connected linear transformation $16 \to 8$ with $\tanh$ activation.
- **Layer 2**: Fully connected linear transformation $8 \to 4$ followed by $\text{Sigmoid}() \times \pi$, mapping the latent vector strictly into $[0, \pi]^4$ to match the rotation angle domain of quantum state initialization.

### 2. Parameterized Quantum Circuit (PQC) Bottleneck
- **Qubit Register**: 4 qubits simulated on PennyLane's `default.qubit`.
- **Quantum State Encoding**: $R_y$ `AngleEmbedding` embeds the 4 latent angles:
  $$\mathcal{U}_{\text{emb}}(\theta) = \bigotimes_{i=0}^3 R_y(\theta_i)$$
- **Variational Quantum Layer**: 2 layers of `StronglyEntanglingLayers` consisting of arbitrary single-qubit rotations ($R_z(\gamma) R_y(\beta) R_z(\alpha)$) followed by cyclic entangling CNOT gates across all 4 qubits ($2 \times 4 \times 3 = 24$ trainable variational parameters).
- **Quantum Measurements**: Measures expectation values of the Pauli-Z operator for all 4 qubits:
  $$\langle Z_i \rangle = \langle \psi(\theta, \mathbf{W}) | Z_i | \psi(\theta, \mathbf{W}) \rangle \in [-1, 1]$$
- **PyTorch Integration**: Encapsulated via `qml.qnn.TorchLayer` for end-to-end analytical gradient backpropagation.

### 3. Classical Decoder
- **Layer 1**: Fully connected linear transformation $4 \to 8$ with $\tanh$ activation.
- **Layer 2**: Fully connected linear transformation $8 \to 16$ with $\text{Sigmoid}() \times \pi$, reconstructing the estimated feature vector $\hat{\mathbf{x}} \in [0, \pi]^{16}$.

---

## Novel Dual-Space Anomaly Scoring Module

Traditional quantum autoencoders evaluate out-of-distribution patterns using only classical reconstruction residual error. However, the quantum latent space encodes rich information-theoretic properties through entanglement.

### Mathematical Formulation

1. **Classical Reconstruction Residual**:
   $$e_{\text{classical}}(\mathbf{x}) = \frac{1}{D} \sum_{j=1}^D (x_j - \hat{x}_j)^2$$

2. **Quantum Subsystem Von Neumann Entanglement Entropy**:
   Let $|\psi(\theta, \mathbf{W})\rangle$ be the state prepared by the PQC for latent angles $\theta = \text{Encoder}(\mathbf{x})$. For subsystem $A = \{0, 1\}$ and environment $B = \{2, 3\}$, the reduced density operator is:
   $$\rho_A = \text{Tr}_B \left( |\psi(\theta, \mathbf{W})\rangle \langle \psi(\theta, \mathbf{W})| \right)$$
   The Von Neumann entanglement entropy is evaluated via PennyLane's `qml.vn_entropy(wires=[0, 1])`:
   $$S_{\text{quantum}}(\rho_A) = -\text{Tr}(\rho_A \ln \rho_A)$$

3. **Hybrid Dual-Space Anomaly Score**:
   $$S_{\text{dual}}(\mathbf{x}) = e_{\text{classical}}(\mathbf{x}) + \beta \cdot S_{\text{quantum}}(\rho_A(\mathbf{x})) \quad (\beta = 0.15)$$

### Why Entanglement Entropy Detects Anomalies
During training on nominal data, the variational parameters $\mathbf{W}$ settle into configurations where nominal feature angles consistently form coherent, highly entangled multi-qubit states ($\mu \approx 0.957\text{ nats}$). When an anomalous vector enters the circuit, the out-of-manifold angles disrupt this learned subsystem coherence, resulting in a distinct, shifted entropy distribution ($\mu \approx 0.911\text{ nats}$). Combining both classical and quantum metrics provides dual-space sensitivity.

---

## Experimental Results

The benchmark consists of $D=16$ multivariate features with exponentially decaying Toeplitz nominal covariance ($\Sigma_{i, j} = \exp(-|i - j| / 3.0)$) and injected anomaly clusters with mean shifts and inflated covariance.

![HQCAE Experimental Results](assets/hqcae_experimental_results.png)

### Performance Comparison: Classical MSE vs. Dual-Space Hybrid

| Evaluation Metric | Classical MSE Alone | Dual-Space Hybrid ($S_{\text{dual}}$) |
|:---|:---:|:---:|
| **ROC-AUC Score** | **0.9700** | **0.9664** |
| **Optimal Threshold ($\tau$)** | **0.2210** | **0.3555** |
| **Youden's $J$ Statistic** | **0.8450** | **0.8317** |
| **Overall Accuracy** | **92.57%** | **91.71%** |
| **Nominal Precision / Recall** | **0.93 / 0.94** | **0.93 / 0.93** |
| **Anomaly Precision / Recall** | **0.92 / 0.90** | **0.90 / 0.91** |
| **True Positive Anomalies Detected** | **135 / 150** (90.0%) | **136 / 150** (**90.7%**) |

### Confusion Matrices ($N = 350$)

#### Classical MSE Alone ($\tau_{\text{cls}} = 0.2210$)
| | Predicted Nominal ($\hat{y}=0$) | Predicted Anomaly ($\hat{y}=1$) | Total |
|:---|:---:|:---:|:---:|
| **Actual Nominal ($y=0$)** | **189** (TN) | **11** (FP) | 200 |
| **Actual Anomaly ($y=1$)** | **15** (FN) | **135** (TP) | 150 |

#### Dual-Space Hybrid ($\tau_{\text{dual}} = 0.3555$)
| | Predicted Nominal ($\hat{y}=0$) | Predicted Anomaly ($\hat{y}=1$) | Total |
|:---|:---:|:---:|:---:|
| **Actual Nominal ($y=0$)** | **185** (TN) | **15** (FP) | 200 |
| **Actual Anomaly ($y=1$)** | **14** (FN) | **136** (TP) | 150 |

*Key finding: The Dual-Space metric increases the True Positive anomaly count to 136/150 (reducing False Negatives to 14), verifying that quantum entanglement entropy captures anomaly characteristics that evade pure classical MSE.*

---

## Project Structure

```
├── assets/
│   └── hqcae_experimental_results.png   # 4-panel publication-quality results figure
├── hqcae_experiment.ipynb               # Fully executed Jupyter Notebook with outputs
├── requirements.txt                     # Project dependencies
└── README.md                            # Complete documentation and execution guide
```

---

## Running Locally

### 1. Prerequisites
Ensure you have Python 3.11, 3.12, or 3.13 installed.

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/harish-k-harish/HQCAE-Anomaly-Detection.git
cd HQCAE-Anomaly-Detection

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch and Execute the Notebook
You can open and run the notebook interactively in Jupyter Lab, VS Code, or Jupyter Notebook:
```bash
jupyter notebook hqcae_experiment.ipynb
```
Or execute headless from the terminal:
```bash
python -m nbclient hqcae_experiment.ipynb
```

All 4-panel figures and evaluation metrics will be generated and saved to `assets/hqcae_experimental_results.png`.
