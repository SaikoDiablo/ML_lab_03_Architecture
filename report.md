# Lab 3 Report — Architecture Cost & Performance

## Goal
Compare an MLP, a CNN, and an Attention model on Fashion-MNIST and see which one gives the best accuracy per parameter, per FLOP, and per millisecond of latency.

## Method
Loaded Fashion-MNIST with `ToTensor()` and `Normalize((0.2860,), (0.3530,))`. Split the 60k training set into 55k train / 5k validation, kept the official 10k test set. Batch size 128, seed 42.

Models:
- **MLP**: `784 → 256 → 128 → 10` with ReLU.
- **CNN**: two conv blocks (`Conv2d(1,16,3)` + MaxPool, `Conv2d(16,32,3)` + MaxPool) → Linear(32×7×7, 10).
- **Attention**: 16 patches of 7×7 → Linear to 64 → 2 Transformer encoder layers (d_model=64, nhead=4, ff=128) → mean pool → Linear(64, 10).

Counted parameters with `torchinfo.summary`, hand-calculated FLOPs for one layer of each type using the Ch. 4 formulas, and compared against `thop.profile`. Measured latency at batch 1 (median of 100) and batch 128 (median of 20). Trained all three for 5 epochs with Adam, lr=3e-3.

## Results

**Parameters**

| Model | Params | Size (MB) |
|---|---:|---:|
| MLP | 235,146 | 0.94 |
| CNN | 20,490 | 0.08 |
| Attention | 70,794 | 0.15 |

Only the CNN is close to the manual's 100k–150k target — the MLP overshoots and Attention undershoots. Reporting as measured.

**Hand FLOPs vs THOP**

| Model | Hand FLOPs (one layer) | THOP FLOPs |
|---|---:|---:|
| MLP (first Linear) | 401,408 | 469,504 |
| CNN (first Conv2d) | 225,792 | 2,063,488 |
| Attention (QKV) | 393,216 | 1,182,976 (whole model) |

**Latency**

| Model | Batch 1 (ms) | Batch 128 (ms) |
|---|---:|---:|
| MLP | 0.087 | 0.644 |
| CNN | 0.275 | 25.651 |
| Attention | 0.658 | 11.567 |

**Training (5 epochs)**

| Model | Test accuracy |
|---|---:|
| MLP | 87.34% |
| CNN | **89.52%** |
| Attention | 84.02% |

Trade-off plots saved in `results/`.

## Conclusions
1. **The CNN was the most cost-effective.** Fewest parameters (20k) and highest test accuracy (89.52%), beating the 235k MLP (87.34%) by ~2 points and Attention (84.02%) by ~5.5 points. This matches the Ch. 4 access patterns: the CNN shares weights across spatial positions and exploits local structure, so each parameter does much more work than an MLP's.
2. **The latency ranking flipped with batch size.** At batch 1 it was MLP < CNN < Attention, as expected. At batch 128 it became MLP < Attention < CNN — the CNN turned out slowest. My guess is that at larger batches the CNN's pooling and big spatial feature maps become the CPU bottleneck, while Attention scales more linearly.
3. **Hand FLOPs and THOP differ for a real reason.** The MLP hand-calc on its first Linear is close to THOP's total, but the CNN hand-calc on the first Conv2d is ~9× smaller than THOP's total — the second Conv2d and final Linear contribute most of the MACs, and my formula also skips bias/ReLU/pooling. So the formulas are good for reasoning about which layer dominates, but THOP is what you want for a true model total.