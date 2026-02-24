# Quantum vs Classical Machine Learning — A Comparative Study

> **Benchmarking Variational Quantum Circuits against Classical ML/DL on the Iris Dataset**

Based on the foundational paper: *"Training a Quantum Neural Network"* — Ventura & Martinez, NIPS 2003

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Classical Models](#classical-models)
  - [Variational Quantum Classifier (VQC)](#variational-quantum-classifier-vqc)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Results & Comparison](#results--comparison)
- [References](#references)

---

## Overview

This project presents a **rigorous comparative analysis** between classical machine learning models and quantum neural network architectures for classification tasks. We implement and evaluate five distinct models on the same datasets under identical experimental conditions to determine whether quantum approaches can achieve competitive performance against well-established classical methods.

### Models Implemented

| # | Model | Type | Framework |
|---|-------|------|-----------|
| 1 | **Support Vector Machine** (RBF Kernel) | Classical ML | scikit-learn |
| 2 | **Feedforward Neural Network** | Classical DL | PyTorch |
| 3 | **Random Forest** | Classical ML (Ensemble) | scikit-learn |
| 4 | **Variational Quantum Classifier (VQC)** | Quantum ML | PennyLane |

---

## Architecture

### Classical Models

**SVM (RBF Kernel):** Hyperparameter-tuned via 5-fold GridSearchCV over `C ∈ {0.1, 1, 10, 100}` and `γ ∈ {scale, auto, 0.01, 0.001}`. Uses all available features with StandardScaler normalization.

**Artificial Neural Network:** PyTorch feedforward network with architecture `Input → 64 → 32 → 16 → Output`, using ReLU activations, BatchNorm, Dropout (0.2), and Adam optimizer with learning rate scheduling.

---


#### VQ Circuit Architecture

| Stage | Gates | Purpose |
|-------|-------|---------|
| **Angle Embedding** | RX(xᵢ) on each qubit | Encode classical features as quantum rotation angles |
| **Variational Layer** (×3) | RY(θ) + RZ(φ) per qubit | Trainable parameterized rotations |
| **Entanglement** (×3) | CNOT ring topology | Create quantum correlations between qubits |
| **Measurement** | ⟨Z⟩ on qubits 0–2 | Extract class probabilities via PauliZ expectation |

- **Qubits:** 4 (one per feature for Iris) / 8 (PCA-reduced for MNIST)
- **Trainable parameters:** 24 (Iris) / 64 (MNIST)
- **Optimizer:** Adam with cross-entropy loss
- **Key insight:** 24 quantum parameters compete with ~3,000+ classical parameters thanks to the exponentially large Hilbert space (2⁴ = 16 dimensions from 4 qubits)

#### How It Works ?

```
Classical Input → [Encode into Qubits] → [Parameterized Quantum Gates] → [Measure] → Prediction
                   (Angle Embedding)      (Trainable RY/RZ + CNOT)       (PauliZ)
```

1. **Encoding:** Each feature `xᵢ` becomes a rotation angle `RX(xᵢ)` on qubit `qᵢ`
2. **Processing:** Variational layers apply learnable rotations and CNOT entanglement — this is where the quantum "computation" happens
3. **Measurement:** PauliZ expectations on selected qubits produce values in [-1, +1], mapped to class probabilities via softmax

---

## Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| **Quantum Simulation** | PennyLane (`default.qubit`) | Variational quantum circuits, automatic differentiation of quantum gates |
| **Classical Deep Learning** | PyTorch | Neural networks, hybrid quantum-classical backpropagation |
| **Classical ML** | scikit-learn | SVM, StandardScaler, MinMaxScaler, GridSearchCV, metrics |
| **Data Processing** | NumPy | Array operations, numerical computation |
| **Visualization** | Matplotlib, Seaborn | Training curves, confusion matrices, comparison charts |
| **Language** | Python 3 | All components |

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
cd "QNN Formulation"
pip install -r requirements.txt
```

### Dependencies

```
numpy
pandas
scikit-learn
matplotlib
seaborn
torch
pennylane
```
---

## Results & Comparison

### Key Metrics Evaluated

- **Accuracy** (Train & Test)
- **Precision** (macro-averaged)
- **Recall** (macro-averaged)
- **F1-Score** (macro-averaged)
- **Confusion Matrix** (per model)
- **Parameter Efficiency** (accuracy per parameter)
- **Training Time**

---

## References

1. **Ventura, D., & Martinez, T.** (2003). *Training a Quantum Neural Network.* Advances in Neural Information Processing Systems (NIPS).
2. **Schuld, M., & Petruccione, F.** (2021). *Machine Learning with Quantum Computers.* Springer.
3. **PennyLane Documentation.** [pennylane.ai](https://pennylane.ai/)
4. **UCI Machine Learning Repository.** Iris & MNIST Datasets.

---

<p align="center">
  <em>Built with ⚛️ PennyLane + 🔥 PyTorch + 🐍 scikit-learn</em>
</p>
