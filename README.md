# ⚛️ Quantum vs Classical Machine Learning — A Comparative Study

> **Benchmarking Variational Quantum Circuits against Classical ML/DL on the Iris & MNIST Datasets**

Based on the foundational paper: *"Training a Quantum Neural Network"* — Ventura & Martinez, NIPS 2003

---

## 📋 Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Architecture](#architecture)
  - [Classical Models](#classical-models)
  - [Variational Quantum Classifier (VQC)](#variational-quantum-classifier-vqc)
  - [Quanvolutional Neural Network](#quanvolutional-neural-network)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
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
| 5 | **Quanvolutional Neural Network** | Hybrid Quantum-Classical | PennyLane + PyTorch |

---

## Motivation

The NIPS 2003 paper *"Training a Quantum Neural Network"* demonstrated that quantum neural networks trained via Grover's quantum search could achieve **97.79% accuracy** on the Iris dataset — outperforming classical backpropagation (96%). This project extends that investigation by:

1. Implementing a **modern variational quantum circuit** (the practical equivalent of the paper's QNN)
2. Comparing against **multiple classical baselines** (not just backpropagation)
3. Introducing a **Quanvolutional Neural Network** — quantum convolutional kernels that process image patches, analogous to CNN filters
4. Evaluating on both the **Iris** and **MNIST** datasets with comprehensive metrics

---

## Architecture

### Classical Models

**SVM (RBF Kernel):** Hyperparameter-tuned via 5-fold GridSearchCV over `C ∈ {0.1, 1, 10, 100}` and `γ ∈ {scale, auto, 0.01, 0.001}`. Uses all available features with StandardScaler normalization.

**Neural Network:** PyTorch feedforward network with architecture `Input → 64 → 32 → 16 → Output`, using ReLU activations, BatchNorm, Dropout (0.2), and Adam optimizer with learning rate scheduling.

**Random Forest:** Ensemble of 200 decision trees with max depth 20.

---

### Variational Quantum Classifier (VQC)

The core quantum model uses a **parameterized quantum circuit** that processes classical data through quantum gates:

<p align="center">
  <img src="assets/vqc_circuit_diagram.png" alt="4-Qubit VQC Circuit Diagram" width="700">
</p>
<p align="center"><em>4-Qubit Variational Quantum Circuit for Iris Classification</em></p>

#### Circuit Architecture

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

#### How It Works

```
Classical Input → [Encode into Qubits] → [Parameterized Quantum Gates] → [Measure] → Prediction
                   (Angle Embedding)      (Trainable RY/RZ + CNOT)       (PauliZ)
```

1. **Encoding:** Each feature `xᵢ` becomes a rotation angle `RX(xᵢ)` on qubit `qᵢ`
2. **Processing:** Variational layers apply learnable rotations and CNOT entanglement — this is where the quantum "computation" happens
3. **Measurement:** PauliZ expectations on selected qubits produce values in [-1, +1], mapped to class probabilities via softmax

---

### Quanvolutional Neural Network

A **hybrid quantum-classical architecture** where quantum circuits replace classical CNN convolution filters:

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐     ┌────────┐
│  8×8 Image   │ ──► │  2×2 Quantum     │ ──► │  Quantum      │ ──► │ Dense  │ ──► Class
│  (64 pixels) │     │  Kernel slides   │     │  Feature Maps │     │ Layers │
│              │     │  across image    │     │  (4×4×4×4)    │     │        │
└──────────────┘     └──────────────────┘     └───────────────┘     └────────┘
                      4 qubits per patch       4 filters output
                      RY encoding + CNOT       256 total features
```

**How the quantum kernel works:**
1. Extract a **2×2 pixel patch** from the image (4 values)
2. **Encode** into 4 qubits via RY rotations
3. Apply **parameterized quantum circuit** (RY/RZ + CNOT entanglement)
4. **Measure** all 4 qubits → 4 output values per patch
5. **Slide** across the entire image (stride 2) → 4×4 quantum feature map
6. Repeat with **4 different quantum filters** (different trainable weights)
7. Feed all quantum features into a **classical dense network** for classification

This architecture is inspired by *"Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits"* (Henderson et al., 2019).

---

## Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| **Quantum Simulation** | PennyLane (`default.qubit`) | Variational quantum circuits, automatic differentiation of quantum gates |
| **Classical Deep Learning** | PyTorch | Neural networks, hybrid quantum-classical backpropagation |
| **Classical ML** | scikit-learn | SVM, Random Forest, PCA, StandardScaler, GridSearchCV, metrics |
| **Data Processing** | NumPy | Array operations, numerical computation |
| **Visualization** | Matplotlib, Seaborn | Training curves, confusion matrices, comparison charts |
| **Language** | Python 3 | All components |

---

## Project Structure

```
QNN Formulation/
│
├── classical_model.py          # SVM + Neural Network + Random Forest
├── qnn_model.py                # Variational Quantum Classifier (VQC)
├── quanvolutional_model.py     # Quanvolutional Neural Network (Quantum CNN)
├── compare_models.py           # Comparison analysis & visualizations
├── run_all.py                  # Pipeline runner (executes everything)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── assets/
│   └── vqc_circuit_diagram.png # Quantum circuit diagram
│
├── results/                    # Generated after running (gitignored)
│   ├── classical_results.json
│   ├── qnn_results.json
│   ├── quanv_results.json
│   ├── comparison_metrics.png
│   ├── comparison_accuracy.png
│   ├── comparison_confusion_matrices.png
│   ├── comparison_params_vs_accuracy.png
│   ├── comparison_report.txt
│   ├── *_confusion_matrix.png
│   └── *_training_curves.png
│
└── NIPS-2003-training-a-quantum-neural-network-Paper.pdf
```

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

## Usage

### Run the Full Pipeline

```bash
python3 run_all.py
```

This executes in sequence:
1. `classical_model.py` — Trains SVM, Neural Network, Random Forest (~seconds)
2. `qnn_model.py` — Trains the Variational Quantum Classifier (~5–15 minutes)
3. `quanvolutional_model.py` — Trains the Quanvolutional NN (~15–30 minutes)
4. `compare_models.py` — Generates comparison plots and report

### Run Individual Models

```bash
python3 classical_model.py         # Classical models only
python3 qnn_model.py               # VQC only
python3 quanvolutional_model.py    # Quanvolutional NN only
python3 compare_models.py          # Comparison (requires results from above)
```

---

## Results & Comparison

After running the pipeline, the `results/` folder will contain:

| Output | Description |
|--------|-------------|
| `comparison_metrics.png` | Bar chart comparing accuracy, precision, recall, F1 across all models |
| `comparison_accuracy.png` | Train vs test accuracy for every model |
| `comparison_confusion_matrices.png` | Side-by-side confusion matrices |
| `comparison_params_vs_accuracy.png` | Parameter efficiency: params vs accuracy scatter plot |
| `comparison_report.txt` | Complete text summary of all results |
| `quanv_feature_maps.png` | Visualization of quantum convolutional feature maps |
| `qnn_pca_variance.png` | PCA explained variance for feature reduction |

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
2. **Henderson, M., Shakya, S., Pradhan, S., & Cook, T.** (2019). *Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits.* arXiv:1904.04767.
3. **Schuld, M., & Petruccione, F.** (2021). *Machine Learning with Quantum Computers.* Springer.
4. **PennyLane Documentation.** [pennylane.ai](https://pennylane.ai/)
5. **UCI Machine Learning Repository.** Iris & MNIST Datasets.

---

<p align="center">
  <em>Built with ⚛️ PennyLane + 🔥 PyTorch + 🐍 scikit-learn</em>
</p>
