# ML Architecture — Cost & Performance Analysis

A PyTorch experiment comparing three neural network architectures (**MLP**, **CNN**, and a **Transformer-based Attention model**) on Fashion-MNIST. The goal is to evaluate the **trade-off between accuracy, parameter count, computational cost (FLOPs/MACs), and inference latency**.

---

## Overview

This project trains and benchmarks three models on the same dataset to answer a practical question:

> *How much accuracy do you gain per parameter, per FLOP, and per millisecond of latency?*

The pipeline covers the full experimental lifecycle:

1. **Data preparation** — Fashion-MNIST split into train / validation / test.
2. **Model definition** — three architectures with distinct inductive biases.
3. **Static analysis** — parameter counts and theoretical FLOPs (hand-calculated + THOP profiling).
4. **Latency benchmarking** — median inference time for batch sizes 1 and 128.
5. **Training & evaluation** — 5 epochs with Adam, tracking loss/accuracy per epoch.
6. **Visualization** — accuracy-vs-parameters and accuracy-vs-latency trade-off plots.

---

---

## 🧠 Architectures

| Model | Description | Key Layers |
|-------|-------------|------------|
| **MLP** | Fully-connected baseline | `Flatten → Linear(784→256) → ReLU → Linear(256→128) → ReLU → Linear(128→10)` |
| **CNN** | Two-block convolutional network | `Conv2d(1→16, 3×3) → ReLU → MaxPool → Conv2d(16→32, 3×3) → ReLU → MaxPool → Flatten → Linear(32·7·7→10)` |
| **Attention** | Patch-based Transformer encoder | `Unfold into 4×4 patches → Linear(49→64) → TransformerEncoder(2 layers, 4 heads) → Mean-pool → Linear(64→10)` |

Each model is designed to be **reasonably sized** for the dataset, so the comparison focuses on architectural trade-offs rather than raw scale.

---

## ⚙️ Setup

### Requirements

```bash
pip install torch torchvision numpy matplotlib thop torchinfo

Run
bash
python main.py
The dataset downloads automatically into data/ on first run. All results, plots, and model weights are written to results/.
