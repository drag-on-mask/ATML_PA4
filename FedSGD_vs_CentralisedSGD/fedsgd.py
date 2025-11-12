"""
Task 1: FedSGD vs Centralized SGD - Theoretical Equivalence Verification

Based on McMahan et al. (2017) "Communication-Efficient Learning of Deep Networks
from Decentralized Data" - Section 2, Algorithm 1 with E=1 and B=∞

This module implements:
1. FedSGD (K=1, C=1.0, full-batch per client)
2. Centralized SGD (full-batch on combined data)
3. Verification that they produce identical models

Key Insight from Paper (Page 4):
"An equivalent update is given by ∀k, w^k_{t+1} ← w_t − ηg_k and then
w_{t+1} ← Σ(n_k/n)w^k_{t+1}"

With K=1 and full participation, FedSGD mathematically equals Centralized SGD.
"""

import copy
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

os.environ["TQDM_NOTEBOOK"] = "0"
from tqdm import tqdm


# ============================================
# Model Definition
# ============================================


class SimpleCNN(nn.Module):
    """
    Simple CNN for CIFAR-10 as used in Task 2.
    2 Conv layers + 2 FC layers.
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, padding=2
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ============================================
# Data Loading - IID Partition for Task 1
# ============================================


def load_cifar10_iid(
    num_clients: int, batch_size: int = 32, seed: int = 42
) -> Tuple[List[DataLoader], DataLoader, DataLoader, List[int]]:
    """
    Load CIFAR-10 and partition IID across clients.

    Returns:
        train_loaders: List of DataLoader (one per client)
        combined_train_loader: Single DataLoader with ALL training data (for centralized)
        test_loader: Single DataLoader for test set
        client_sizes: [N1, N2, ..., NM] - samples per client
    """
    # Standard CIFAR-10 transforms
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    # IID partition: randomly split train_dataset into num_clients subsets
    np.random.seed(seed)
    total_size = len(train_dataset)
    indices = np.random.permutation(total_size)
    split_size = total_size // num_clients

    train_loaders = []
    client_sizes = []

    for i in range(num_clients):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < num_clients - 1 else total_size
        client_indices = indices[start_idx:end_idx]

        client_subset = Subset(train_dataset, client_indices)

        # Each client uses FULL local dataset as single batch
        # This means B = ∞ in the paper's notation
        client_loader = DataLoader(
            client_subset,
            batch_size=len(client_indices),  # B=∞: Full local dataset as ONE batch
            shuffle=True,  # Keep True for augmentation randomness
            num_workers=2,
        )

        train_loaders.append(client_loader)
        client_sizes.append(len(client_indices))

    # Combined loader for centralized training (FULL dataset as single batch)
    combined_train_loader = DataLoader(
        train_dataset,
        batch_size=len(train_dataset),  # B=∞: Full dataset as ONE batch
        shuffle=True,
        num_workers=2,
    )

    # Test loader
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    return train_loaders, combined_train_loader, test_loader, client_sizes


# ============================================
# FedSGD Implementation (K=1, C=1.0, B=∞)
# ============================================


def fedsgd_train(
    num_clients: int,
    num_rounds: int,
    lr: float,
    device: torch.device,
    seed: int = 42,
) -> Tuple[nn.Module, Dict[str, List]]:
    """
    FedSGD: Federated SGD with K=1 local epoch per round.

    This is the baseline federated algorithm from McMahan et al. paper.
    With K=1, C=1.0, and B=∞, this should equal Centralized SGD.

    Args:
        num_clients: Number of clients (M)
        num_rounds: Number of communication rounds
        lr: Learning rate
        device: cuda or cpu
        seed: Random seed

    Returns:
        global_model: Trained global model
        history: Dict with test_acc, test_loss, param_norm per round
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize model
    global_model = SimpleCNN().to(device)

    # Load data (IID partition)
    train_loaders, _, test_loader, client_sizes = load_cifar10_iid(
        num_clients=num_clients, seed=seed
    )

    total_samples = sum(client_sizes)
    criterion = nn.CrossEntropyLoss()

    # History tracking
    history = {
        "test_acc": [],
        "test_loss": [],
        "param_norm": [],  # Track model parameter norm for comparison
        "rounds": [],
    }

    print(f"--- FedSGD Training ---")
    print(f"  Clients: {num_clients} (all participate each round)")
    print(f"  Rounds: {num_rounds}")
    print(f"  Local Epochs K: 1 (FedSGD)")
    print(f"  Batch Size: FULL (B=∞)")
    print("=" * 70)

    pbar = tqdm(range(num_rounds), desc="FedSGD")

    for round_idx in pbar:
        # Store client updates
        client_weights = []
        client_model_params = []

        # All clients participate (C=1.0)
        for client_idx in range(num_clients):
            # Create local model copy
            local_model = SimpleCNN().to(device)
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))

            # Optimizer for this client
            optimizer = optim.SGD(
                local_model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0
            )

            # K=1: One EPOCH (full pass) through local data with mini-batches
            local_model.train()
            for data, target in train_loaders[client_idx]:
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # Collect updated model parameters
            client_model_params.append(local_model.state_dict())
            client_weights.append(client_sizes[client_idx])

        # Server aggregation: weighted average
        # θ_g^{t+1} = Σ (N_i/N) * θ_i^{(K)}
        aggregated_params = copy.deepcopy(client_model_params[0])
        for key in aggregated_params.keys():
            aggregated_params[key].zero_()

        for key in aggregated_params.keys():
            for i in range(num_clients):
                weight = client_weights[i] / total_samples
                aggregated_params[key] += client_model_params[i][key] * weight

        # Update global model
        global_model.load_state_dict(aggregated_params)

        # Evaluate
        test_acc, test_loss = evaluate_model(global_model, test_loader, device)
        param_norm = compute_model_norm(global_model)

        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)
        history["param_norm"].append(param_norm)
        history["rounds"].append(round_idx + 1)

        pbar.set_postfix({"Test Acc": f"{test_acc:.2f}%", "Loss": f"{test_loss:.4f}"})

    pbar.close()
    print(f"\nFedSGD Complete - Final Accuracy: {history['test_acc'][-1]:.2f}%")

    return global_model, history


# ============================================
# Centralized SGD Implementation
# ============================================


def centralized_train(
    num_rounds: int,
    lr: float,
    device: torch.device,
    seed: int = 42,
) -> Tuple[nn.Module, Dict[str, List]]:
    """
    Centralized SGD: Train on entire dataset with full-batch gradient descent.

    This is the baseline to compare against FedSGD.
    Should produce identical results to FedSGD when K=1, C=1.0, B=∞.

    Args:
        num_rounds: Number of gradient steps (matches FedSGD rounds)
        lr: Learning rate (same as FedSGD)
        device: cuda or cpu
        seed: Random seed (same as FedSGD)

    Returns:
        model: Trained model
        history: Dict with test_acc, test_loss, param_norm per round
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize model (same architecture and seed as FedSGD)
    model = SimpleCNN().to(device)

    # Load combined data (entire training set)
    _, combined_train_loader, test_loader, _ = load_cifar10_iid(
        num_clients=10, seed=seed  # Not used, but needed for function signature
    )

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    # History tracking
    history = {"test_acc": [], "test_loss": [], "param_norm": [], "rounds": []}

    print(f"\n--- Centralized SGD Training ---")
    print(f"  Rounds: {num_rounds}")
    print(f"  Batch Size: FULL (B=∞)")
    print("=" * 70)

    pbar = tqdm(range(num_rounds), desc="Centralized")

    for round_idx in pbar:
        model.train()

        # One full-batch gradient step on entire dataset
        for data, target in combined_train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Evaluate
        test_acc, test_loss = evaluate_model(model, test_loader, device)
        param_norm = compute_model_norm(model)

        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)
        history["param_norm"].append(param_norm)
        history["rounds"].append(round_idx + 1)

        pbar.set_postfix({"Test Acc": f"{test_acc:.2f}%", "Loss": f"{test_loss:.4f}"})

    pbar.close()
    print(f"\nCentralized Complete - Final Accuracy: {history['test_acc'][-1]:.2f}%")

    return model, history


# ============================================
# Evaluation Function
# ============================================


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate model on test set.

    Returns:
        accuracy: Test accuracy (%)
        loss: Average test loss
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = test_loss / total

    return accuracy, avg_loss


# ============================================
# Model Comparison Utilities
# ============================================


def compute_model_norm(model: nn.Module) -> float:
    """
    Compute L2 norm of all model parameters.
    Useful for tracking model evolution.
    """
    param_vector = nn.utils.parameters_to_vector(model.parameters())
    return torch.norm(param_vector, p=2).item()


def compute_model_difference(
    model1_params: Dict[str, torch.Tensor], model2_params: Dict[str, torch.Tensor]
) -> float:
    """
    Compute L2 norm of parameter difference between two models.

    This is the KEY metric for Task 1 verification.
    Should be ≈ 0 (near machine precision) if FedSGD = Centralized.

    Args:
        model1_params: state_dict() from first model
        model2_params: state_dict() from second model

    Returns:
        l2_difference: ||θ_1 - θ_2||_2
    """
    diff_vector = []

    for key in model1_params.keys():
        param1 = model1_params[key].flatten()
        param2 = model2_params[key].flatten()
        diff_vector.append((param1 - param2))

    diff_vector = torch.cat(diff_vector)
    l2_norm = torch.norm(diff_vector, p=2).item()

    return l2_norm


def compute_relative_difference(
    model1_params: Dict[str, torch.Tensor], model2_params: Dict[str, torch.Tensor]
) -> float:
    """
    Compute relative difference: ||θ_1 - θ_2|| / ||θ_1||

    More interpretable than absolute difference.
    Should be < 1e-6 for perfect equivalence.
    """
    diff_norm = compute_model_difference(model1_params, model2_params)

    # Compute norm of first model
    param1_vector = torch.cat(
        [model1_params[key].flatten() for key in model1_params.keys()]
    )
    model1_norm = torch.norm(param1_vector, p=2).item()

    relative_diff = diff_norm / (model1_norm + 1e-10)  # Avoid division by zero

    return relative_diff


# ============================================
# Sanity Check
# ============================================

if __name__ == "__main__":
    """
    Quick sanity check of the implementations.
    """
    print("Running Task 1 framework sanity check...\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Very short test
    NUM_CLIENTS = 5
    NUM_ROUNDS = 3
    LEARNING_RATE = 0.01
    SEED = 42

    print("Testing FedSGD...")
    fedsgd_model, fedsgd_history = fedsgd_train(
        num_clients=NUM_CLIENTS,
        num_rounds=NUM_ROUNDS,
        lr=LEARNING_RATE,
        device=device,
        seed=SEED,
    )

    print("\nTesting Centralized SGD...")
    central_model, central_history = centralized_train(
        num_rounds=NUM_ROUNDS, lr=LEARNING_RATE, device=device, seed=SEED
    )

    # Compare
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    param_diff = compute_model_difference(
        fedsgd_model.state_dict(), central_model.state_dict()
    )

    relative_diff = compute_relative_difference(
        fedsgd_model.state_dict(), central_model.state_dict()
    )

    print(f"L2 Parameter Difference: {param_diff:.8f}")
    print(f"Relative Difference: {relative_diff:.2e}")
    print(f"\nFedSGD Final Accuracy: {fedsgd_history['test_acc'][-1]:.2f}%")
    print(f"Centralized Final Accuracy: {central_history['test_acc'][-1]:.2f}%")
    print(
        f"Accuracy Difference: {abs(fedsgd_history['test_acc'][-1] - central_history['test_acc'][-1]):.4f}%"
    )

    if param_diff < 1e-3:
        print("\n✅ SUCCESS: Models are nearly identical!")
        print("   FedSGD = Centralized SGD equivalence verified.")
    else:
        print("\n⚠️  WARNING: Models differ more than expected.")
        print("   Check hyperparameters and random seeds.")

    print("\nFramework sanity check complete.")
