# Hybrid Quantum-Classical Autoencoder (HQCAE) for Anomaly Representation and Detection

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.1.0-EE4C2C.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-%E2%89%A50.35.0-brightgreen.svg)](https://pennylane.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An unsupervised anomaly detection framework utilizing a **Hybrid Quantum-Classical Autoencoder (HQCAE)**. The model projects 16-dimensional correlated multivariate continuous data into a 4-qubit Hilbert space through parameterized quantum circuits (PQCs), capturing non-linear nominal manifold structures and isolating anomalies via reconstruction residual error divergence.

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
                             │ 4-Qubit Quantum PQC   │
                             │ AngleEmbedding (Ry)   │
                             │ 2x StronglyEntangling │
                             │ Pauli-Z Measurements  │
                             └───────────┬───────────┘
                                         │
                  [ Expectation Vector: <Z_i> in [-1, 1]^4 ]
                                         │
                             ┌───────────┴───────────┐
                             │   Classical Decoder   │
                             │  Linear(4 -> 8) Tanh  │
                             │  Linear(8 -> 16)      │
                             │  Sigmoid() * pi       │
                             └───────────┬───────────┘
                                         │
                    [ Reconstruction: x_hat in [0, pi]^16 ]
                                         │
                                         ▼
                     Reconstruction Residual Anomaly Score:
                       e(x) = (1/D) * sum((x_j - x_hat_j)^2)
                                         │
                                         ▼
                       Decision: Flag if e(x) >= tau
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

### 4. Unsupervised Anomaly Scoring
- The autoencoder is trained **exclusively on nominal samples** using Mean Squared Error (MSE) loss:
  $$\mathcal{L}_{\text{MSE}} = \frac{1}{B} \sum_{b=1}^B \|\mathbf{x}^{(b)} - \hat{\mathbf{x}}^{(b)}\|^2$$
- At inference time, the **reconstruction residual anomaly score** is computed as:
  $$e(\mathbf{x}) = \frac{1}{D} \sum_{j=1}^D (x_j - \hat{x}_j)^2$$
- Anomalous samples violating the nominal correlation manifold produce substantially higher reconstruction errors.
- The decision boundary $\tau$ is established using **Youden's $J$ statistic** ($J = \text{TPR} - \text{FPR}$), marking any sample with $e(\mathbf{x}) \ge \tau$ as an anomaly.

---

## Experimental Results

The experiment was conducted on a synthetic multivariate benchmark ($D=16$) generated with an exponentially decaying Toeplitz correlation structure for nominal baseline samples and mean-shifted, variance-inflated anomaly clusters.

![HQCAE Experimental Results](assets/hqcae_experimental_results.png)

### Summary Metrics

| Metric | Value |
|:---|:---:|
| **ROC-AUC Score** | **0.9700** |
| **Optimal Threshold ($\tau$)** | **0.2210** |
| **Youden's $J$ Statistic** | **0.8450** |
| **Overall Accuracy** | **92.57%** |
| **Nominal Precision / Recall** | **0.93 / 0.94** (F1 = 0.94) |
| **Anomaly Precision / Recall** | **0.92 / 0.90** (F1 = 0.91) |

### Confusion Matrix

| | Predicted Nominal ($\hat{y}=0$) | Predicted Anomaly ($\hat{y}=1$) | Total |
|:---|:---:|:---:|:---:|
| **Actual Nominal ($y=0$)** | **189** (True Negative) | **11** (False Positive) | 200 |
| **Actual Anomaly ($y=1$)** | **15** (False Negative) | **135** (True Positive) | 150 |

---

## Project Structure

```
├── assets/
│   └── hqcae_experimental_results.png   # 3-panel publication-quality results figure
├── hqcae_experiment.ipynb               # Fully executed Jupyter Notebook with outputs
├── requirements.txt                     # Project dependencies
└── README.md                            # Architecture documentation and execution guide
```

---

## Running Locally

### 1. Prerequisites
Ensure you have Python 3.11, 3.12, or 3.13 installed.

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone <repo-url>
cd "Quantum mini project"

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

All figures and evaluation metrics will be generated and saved to `assets/hqcae_experimental_results.png`.
