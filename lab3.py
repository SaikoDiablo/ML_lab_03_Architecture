import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import thop

from torch.utils.data import DataLoader, random_split
from torchinfo import summary

# Experiment settings
SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Set random seeds so the results are more reproducible
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Lab 3: Architectures and their Costs")
print("PyTorch version:", torch.__version__)
print("Device:", device)

print("\nTask 1")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_full = torchvision.datasets.FashionMNIST(
    root=str(DATA_DIR),
    train=True,
    download=True,
    transform=transform
)

test_set = torchvision.datasets.FashionMNIST(
    root=str(DATA_DIR),
    train=False,
    download=True,
    transform=transform
)

split_generator = torch.Generator().manual_seed(SEED)

train_set, val_set = random_split(
    train_full,
    [55000, 5000],
    generator=split_generator
)

loader_generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=loader_generator,
    num_workers=0
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Training images:   {len(train_set):,}")
print(f"Validation images: {len(val_set):,}")
print(f"Test images:       {len(test_set):,}")
print(f"Batch size:        {BATCH_SIZE}")

# Verifying the expected input and label shapes
sample_images, sample_labels = next(iter(train_loader))

print("Image batch shape:", tuple(sample_images.shape))
print("Label batch shape:", tuple(sample_labels.shape))
print("Image dimensions:", tuple(sample_images.shape[1:]))

assert len(train_set) == 55000
assert len(val_set) == 5000
assert len(test_set) == 10000
assert tuple(sample_images.shape[1:]) == (1, 28, 28)

# Task 2
class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)


class AttentionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.patch_projection = nn.Linear(49, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
    
        x = x.unfold(2, 7, 7).unfold(3, 7, 7)

        x = x.reshape(x.size(0), 16, 49)

        x = self.patch_projection(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        return self.classifier(x)


models = {
    "MLP": MLP(),
    "CNN": CNN(),
    "Attention": AttentionModel()
}

for model in models.values():
    model.to(device)


print("\nTask 3")

results = {}

for name, model in models.items():
    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"\n{name}")
    print("-" * 40)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print(summary(
        model,
        input_size=(1, 1, 28, 28),
        device=str(device),
        verbose=0
    ))

    results[name] = {
        "params": total_params,
        "trainable_params": trainable_params
    }

# Linear layer: 2 * in * out
mlp_linear_flops = 2 * 784 * 256

# Conv2d:
# 2 * K * K * Cin * Cout * Hout * Wout
cnn_conv_flops = 2 * 3 * 3 * 1 * 16 * 28 * 28

# QKV projections
# 2 * seq * d_model * 3 * d_model
seq_len = 16
d_model = 64

attention_qkv_flops = (
    2 * seq_len * d_model * 3 * d_model
)

# attention score 
# 2 * seq_len^2 * d_model.
attention_scores_flops = (
    2 * seq_len**2 * d_model
)

attention_values_flops = (
    2 * seq_len**2 * d_model
)

hand_calculations = {
    "MLP first Linear layer": mlp_linear_flops,
    "CNN first Conv2d layer": cnn_conv_flops,
    "Attention QKV projections": attention_qkv_flops,
    "Attention score products": attention_scores_flops,
    "Attention weighted-value products": attention_values_flops
}

for description, count in hand_calculations.items():
    print(f"{description}: {count:,} FLOPs")

with open(
    RESULTS_DIR / "hand_calculations.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(hand_calculations, file, indent=4)


dummy_input = torch.randn(1, 1, 28, 28, device=device)


def profile_model(model, inputs):

    model.eval()

    try:
        with torch.inference_mode():
            macs, thop_params = thop.profile(
                model,
                inputs=(inputs,),
                verbose=False
            )

        # THOP generally reports MACs not universally comparable
        # hardware FLOP
        return {
            "macs": float(macs),
            "estimated_flops_2_per_mac": float(2 * macs),
            "thop_params": int(thop_params),
            "profile_ok": True
        }

    except Exception as error:
        print(f"THOP could not fully profile this model: {error}")

        return {
            "macs": None,
            "estimated_flops_2_per_mac": None,
            "thop_params": None,
            "profile_ok": False
        }


for name, model in models.items():
    profile = profile_model(model, dummy_input)

    results[name].update(profile)

    print(f"\n{name}")
    print("THOP MACs:", profile["macs"])
    print(
        "Estimated FLOPs (2 per MAC):",
        profile["estimated_flops_2_per_mac"]
    )
    print("THOP parameters:", profile["thop_params"])

def synchronize_device():

    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_latency(model, batch_size, runs):
    model.eval()

    inputs = torch.randn(
        batch_size, 1, 28, 28, device=device
    )

    # Warm-up runs are excluded from measurements
    with torch.inference_mode():
        for _ in range(10):
            model(inputs)

        synchronize_device()
        timings_ms = []

        for _ in range(runs):
            synchronize_device()
            start = time.perf_counter()

            model(inputs)

            synchronize_device()
            elapsed_ms = (time.perf_counter() - start) * 1000
            timings_ms.append(elapsed_ms)

    return float(np.median(timings_ms))


for name, model in models.items():
    # Manual specifies 100 runs for batch 1 and 20 for batch 128
    lat_1 = measure_latency(model, batch_size=1, runs=100)
    lat_128 = measure_latency(model, batch_size=128, runs=20)

    results[name]["lat_1_ms"] = lat_1
    results[name]["lat_128_ms"] = lat_128

    print(
        f"{name}: batch 1 = {lat_1:.3f} ms; "
        f"batch 128 = {lat_128:.3f} ms"
    )

print("\nTask 4")


def evaluate_model(model, loader, criterion=None):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = 100.0 * correct / total
    average_loss = (
        total_loss / total if criterion is not None else None
    )

    return accuracy, average_loss


def train_model(model, name):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = []

    for epoch in range(EPOCHS):
        model.train()

        total_train_loss = 0.0
        total_train = 0
        correct_train = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_train_loss += loss.item() * batch_size
            total_train += batch_size
            correct_train += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

        train_loss = total_train_loss / total_train
        train_accuracy = 100.0 * correct_train / total_train

        val_accuracy, val_loss = evaluate_model(
            model, val_loader, criterion
        )

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        }

        history.append(epoch_result)

        print(
            f"{name} | Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.2f}% | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_accuracy:.2f}%"
        )

    # Evaluating the official test set after training
    test_accuracy, test_loss = evaluate_model(
        model, test_loader, criterion
    )

    print(
        f"{name} final test accuracy: {test_accuracy:.2f}%"
    )

    torch.save(
        model.state_dict(),
        RESULTS_DIR / f"{name.lower()}_weights.pth"
    )

    return history, test_accuracy, test_loss


histories = {}

for name, model in models.items():
    history, test_accuracy, test_loss = train_model(model, name)

    histories[name] = history

    results[name]["acc"] = test_accuracy
    results[name]["test_loss"] = test_loss

print("\nSaving results")

with open(
    RESULTS_DIR / "metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=4)

with open(
    RESULTS_DIR / "training_history.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(histories, file, indent=4)

with open(
    RESULTS_DIR / "metrics.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:
    fieldnames = [
        "model",
        "params",
        "trainable_params",
        "macs",
        "estimated_flops_2_per_mac",
        "lat_1_ms",
        "lat_128_ms",
        "acc",
        "test_loss"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for name, data in results.items():
        row = {"model": name}

        for field in fieldnames:
            if field != "model":
                row[field] = data.get(field)

        writer.writerow(row)


print("\nTask 5")

names = list(results.keys())

accuracies = [results[name]["acc"] for name in names]
parameters = [results[name]["params"] for name in names]
latencies = [results[name]["lat_1_ms"] for name in names]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy versus parameter count
for name in names:
    axes[0].scatter(
        results[name]["params"],
        results[name]["acc"],
        label=name,
        s=100
    )

    axes[0].annotate(
        name,
        (results[name]["params"], results[name]["acc"]),
        xytext=(5, 5),
        textcoords="offset points"
    )

axes[0].set_xlabel("Number of Parameters")
axes[0].set_ylabel("Test Accuracy (%)")
axes[0].set_title("Accuracy vs Parameters")
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].legend()

# Accuracy versus batch-1 latency
for name in names:
    axes[1].scatter(
        results[name]["lat_1_ms"],
        results[name]["acc"],
        label=name,
        s=100
    )

    axes[1].annotate(
        name,
        (results[name]["lat_1_ms"], results[name]["acc"]),
        xytext=(5, 5),
        textcoords="offset points"
    )

axes[1].set_xlabel("Batch-1 Inference Latency (ms)")
axes[1].set_ylabel("Test Accuracy (%)")
axes[1].set_title("Accuracy vs Latency")
axes[1].grid(True, linestyle="--", alpha=0.5)
axes[1].legend()

fig.tight_layout()
fig.savefig(RESULTS_DIR / "trade_off_plots.png", dpi=200)
plt.show()

print("\nAll tasks completed.")
print("Results saved in:", RESULTS_DIR)
