"""
FedAvg Implementation for ATML PA4 - Task 2
Based on algorithm from McMahan et al. (2017) "Communication-Efficient
Learning of Deep Networks from Decentralized Data"

Implementation patterns learned from FedML library:
https://github.com/FedML-AI/FedML/tree/master/python/fedml/simulation/sp/fedavg

Key references:
- Client-server communication pattern from fedavg_api.py
- Weighted aggregation logic from FedAvgAPI._aggregate()
- Client training wrapper from client.py

All code is original work implementing the FedAvg algorithm for this assignment.
"""

import copy
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

os.environ["TQDM_NOTEBOOK"] = "0"
from tqdm import tqdm

# --- Component 1: Model Definition ---


class SimpleCNN(nn.Module):
    """
    A simple CNN model for CIFAR-10, as specified in the assignment.
    Architecture: 2 Conv layers + 2 FC layers.
    This matches the "small convolutional network" requirement.
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, padding=2
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Fully Connected Layers
        # Image size starts at 32x32
        # After conv1 -> 32x32
        # After pool1 -> 16x16
        # After conv2 -> 16x16
        # After pool2 -> 8x8
        # Flattened size: 64 channels * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1
        x = self.pool(self.relu(self.conv1(x)))
        # Conv block 2
        x = self.pool(self.relu(self.conv2(x)))
        # Flatten
        x = x.view(-1, 64 * 8 * 8)
        # FC block
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# --- Component 2: Data Loading (IID) ---


def load_cifar10_iid(
    num_clients: int, batch_size: int = 32
) -> Tuple[List[DataLoader], DataLoader, List[int]]:
    """
    Load CIFAR-10 and partition IID across clients.

    For Task 2, we use a simple random split.

    Returns:
        train_loaders: List of DataLoader (one per client)
        test_loader: Single DataLoader for global test set
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
    total_size = len(train_dataset)
    indices = np.random.permutation(total_size)
    split_size = total_size // num_clients

    train_loaders = []
    client_sizes = []

    for i in range(num_clients):
        start_idx = i * split_size
        # Assign remainder to the last client
        end_idx = (i + 1) * split_size if i < num_clients - 1 else total_size
        client_indices = indices[start_idx:end_idx]

        client_subset = Subset(train_dataset, client_indices)
        client_loader = DataLoader(
            client_subset, batch_size=batch_size, shuffle=True, num_workers=2
        )

        train_loaders.append(client_loader)
        client_sizes.append(len(client_indices))

    # Global test loader
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    return train_loaders, test_loader, client_sizes


# --- Component 2b: Data Loading (Non-IID) ---


def load_cifar10_noniid_dirichlet(
    num_clients: int, alpha: float, batch_size: int = 32, seed: int = 42
) -> Tuple[List[DataLoader], DataLoader, List[int]]:
    """
    Load CIFAR-10 and partition using Dirichlet distribution for label skew.

    Args:
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
            - Small alpha (e.g., 0.1) = highly skewed (non-IID)
            - Large alpha (e.g., 100) = nearly uniform (IID)
        batch_size: Batch size for training
        seed: Random seed

    Returns:
        train_loaders: List of DataLoader per client
        test_loader: Global test DataLoader
        client_sizes: Number of samples per client
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Standard CIFAR-10 transforms (same as IID)
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

    # Get all labels
    targets = np.array(train_dataset.targets)
    num_classes = 10

    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]

    # For each class, sample from Dirichlet and distribute
    for k in range(num_classes):
        # Get indices for class k
        idx_k = np.where(targets == k)[0]
        np.random.shuffle(idx_k)

        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # Ensure proportions sum to 1 (numerical stability)
        proportions = proportions / proportions.sum()

        # Convert to actual counts (at least 1 sample per client if possible)
        counts = (proportions * len(idx_k)).astype(int)

        # Handle rounding: distribute remaining samples
        counts[np.argmax(proportions)] += len(idx_k) - counts.sum()

        # Split and assign to clients
        start_idx = 0
        for i, count in enumerate(counts):
            if count > 0:  # Only add if client gets samples
                client_indices[i].extend(idx_k[start_idx : start_idx + count])
                start_idx += count

    # Shuffle each client's data
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    # Create DataLoaders
    train_loaders = []
    client_sizes = []

    for i in range(num_clients):
        client_subset = Subset(train_dataset, client_indices[i])
        client_loader = DataLoader(
            client_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            # Drop last to avoid batchnorm errors if a client has < batch_size samples
            drop_last=(len(client_subset) > batch_size),
        )
        train_loaders.append(client_loader)
        client_sizes.append(len(client_indices[i]))

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    return train_loaders, test_loader, client_sizes


# --- Component 3: Client Training ---


def client_update(
    model: nn.Module,
    data_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    mu: float = 0.0,
    rho: float = 0.0,  # <-- ADD THIS
    c_global: List[torch.Tensor] = None,
    c_local_i: List[torch.Tensor] = None,
) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
    """
    Performs K local epochs of SGD on client data.
    ...
    """
    model.to(device)
    model.train()

    # Store initial global model parameters (frozen) for proximal term
    if mu > 0:
        # Use a different name to avoid conflict with SCAFFOLD's global_params
        global_params_prox = [param.clone().detach() for param in model.parameters()]

    # --- SCAFFOLD: Initializations for Option I ---
    avg_grad = None
    if c_global is not None:
        # Initialize a list to store the SUM of gradients, on CPU
        # We only care about parameters that require gradients
        avg_grad = [
            torch.zeros_like(p.cpu()) for p in model.parameters() if p.requires_grad
        ]

    # --- Optimizer Setup ---
    # Check if SCAFFOLD is active (by checking if control variates were passed)
    if c_global is not None:
        # SCAFFOLD's corrected gradients conflict with momentum's velocity buffer.
        # The paper's algorithm uses a plain SGD step.
        current_momentum = 0.5
    else:
        # FedAvg or FedProx work fine with momentum.
        current_momentum = 0.9

    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=current_momentum, weight_decay=5e-4
    )
    criterion = nn.CrossEntropyLoss()

    total_steps = 0
    for epoch in range(epochs):
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            if rho > 0:
                # =================================================
                # --- FedSAM Logic (rho > 0) ---
                # =================================================

                # --- Store original parameters (w) ---
                original_params = [p.clone().detach() for p in model.parameters()]

                # --- Step 1: Ascent Step (Compute grad L(w)) ---
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)  # L_CE(w)

                # Add FedProx loss (if applicable)
                if mu > 0:
                    proximal_term = 0.0
                    for param, global_param_prox in zip(
                        model.parameters(), global_params_prox
                    ):
                        proximal_term += (param - global_param_prox).pow(2).sum()
                    loss += (mu / 2) * proximal_term  # L(w) = L_CE(w) + L_prox(w)

                loss.backward()  # Computes grad L(w)

                # Store the gradients (g = grad L(w))
                grads = [
                    p.grad.clone().detach()
                    for p in model.parameters()
                    if p.grad is not None
                ]

                # --- SCAFFOLD: Accumulate grad L(w) for c_i update ---
                if c_global is not None:
                    for i, grad in enumerate(grads):
                        avg_grad[i] += grad.cpu()
                # -----------------------------------------------------

                # Calculate the L2 norm of the full gradient vector
                grad_norm = torch.norm(nn.utils.parameters_to_vector(grads), p=2)

                # Calculate scaling factor: e = (rho / (grad_norm + 1e-12))
                scale = rho / (grad_norm + 1e-12)

                # --- Manually set model parameters to perturbed state (w_adv) ---
                # w_adv = w + e * g
                with torch.no_grad():
                    grad_idx = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            p.add_(grads[grad_idx], alpha=scale)  # p = p + scale * g
                            grad_idx += 1

                # --- Step 2: Descent Step (Compute grad L(w_adv)) ---
                optimizer.zero_grad()  # Clear grads (g)
                output_adv = model(data)
                loss_adv = criterion(output_adv, target)  # L_CE(w_adv)

                # Add FedProx loss at w_adv (if applicable)
                if mu > 0:
                    proximal_term_adv = 0.0
                    for param, global_param_prox in zip(
                        model.parameters(), global_params_prox
                    ):
                        proximal_term_adv += (param - global_param_prox).pow(2).sum()
                    loss_adv += (
                        mu / 2
                    ) * proximal_term_adv  # L(w_adv) = L_CE(w_adv) + L_prox(w_adv)

                loss_adv.backward()  # Computes grad L(w_adv)

                # --- Restore original weights (w) ---
                # The optimizer's .step() will use p.grad (which is grad L(w_adv))
                # but apply it to the original weights.
                with torch.no_grad():
                    for p, p_orig in zip(model.parameters(), original_params):
                        p.copy_(p_orig)  # p = w

                # --- SCAFFOLD: Apply correction to grad L(w_adv) ---
                if c_global is not None:
                    grad_idx = 0
                    for param in model.parameters():
                        if param.requires_grad and param.grad is not None:
                            # Apply the correction to the update gradient (grad L(w_adv))
                            param.grad.data.add_(
                                c_global[grad_idx].to(device)
                                - c_local_i[grad_idx].to(device)
                            )
                            grad_idx += 1
                # ---------------------------------------------------

                optimizer.step()  # Applies (potentially corrected) grad L(w_adv) to w
                total_steps += 1

            else:
                # =================================================
                # --- Standard Logic (rho = 0) ---
                # =================================================
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)

                if mu > 0:
                    proximal_term = 0.0
                    for param, global_param_prox in zip(
                        model.parameters(), global_params_prox
                    ):
                        proximal_term += (param - global_param_prox).pow(2).sum()
                    loss += (mu / 2) * proximal_term

                loss.backward()

                # --- SCAFFOLD: Option 1 Logic ---
                if c_global is not None:
                    grad_idx = 0
                    for param in model.parameters():
                        if param.requires_grad:
                            if param.grad is not None:
                                # 1. Accumulate the *original* gradient
                                avg_grad[grad_idx] += param.grad.data.clone().cpu()

                                # 2. Apply the correction
                                param.grad.data.add_(
                                    c_global[grad_idx].to(device)
                                    - c_local_i[grad_idx].to(device)
                                )
                                grad_idx += 1
                # --------------------------------

                optimizer.step()
                total_steps += 1

    # --- SCAFFOLD: Compute control variate update (Option I) ---
    delta_c_i = None
    if c_global is not None:
        if total_steps > 0:
            # 1. Get the average gradient: c_i_plus = (1/total_steps) * Sum(gradients)
            for i in range(len(avg_grad)):
                avg_grad[i] /= total_steps

            # 2. Calculate the change: delta_c_i = c_i_plus - c_i_old
            delta_c_i = []
            for c_i_plus, c_i_old in zip(avg_grad, c_local_i):
                delta_c_i.append(c_i_plus.cpu() - c_i_old.cpu())

        else:
            # Edge case: no batches were run, so no change in gradient
            delta_c_i = [torch.zeros_like(p.cpu()) for p in c_global]

    # Return the updated model parameters and the control variate delta
    return model.state_dict(), delta_c_i


# --- Component 4: Server Aggregation ---


def aggregate_models(
    client_models: List[Dict[str, torch.Tensor]], client_weights: List[float]
) -> Dict[str, torch.Tensor]:
    """
    FedAvg aggregation: θ_g^{t+1} = Σ (N_i/N) * θ_i^{(K)}.

    Performs a weighted average of all client model parameters.
    Pattern learned from FedML's FedAvgAPI._aggregate() method.

    Args:
        client_models: List of state_dict() from each client
        client_weights: List of normalized weights (w_i = N_i / N_total)

    Returns:
        global_model_params: Aggregated state_dict()
    """
    # Initialize with the structure of the first client's model
    global_params = copy.deepcopy(client_models[0])

    # Zero out all parameters
    for key in global_params.keys():
        global_params[key].zero_()

    # Perform the weighted sum
    for key in global_params.keys():
        for i in range(len(client_models)):
            global_params[key] += client_models[i][key] * client_weights[i]

    return global_params


def harmonize_gradients(
    client_updates: List[Dict[str, torch.Tensor]],
    global_params: Dict[str, torch.Tensor],
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    """
    FedGH: Harmonize conflicting gradients before aggregation.

    For each pair of clients with negative dot product (conflicting updates),
    project each gradient onto the orthogonal complement of the other.

    Args:
        client_updates: List of client model state_dicts
        global_params: Global model parameters (for computing deltas)
        device: torch device

    Returns:
        harmonized_updates: List of harmonized model state_dicts
    """
    num_clients = len(client_updates)

    # 1. Compute gradient deltas (Δθ_i = θ_i - θ_global) for each client
    deltas = []
    for client_params in client_updates:
        delta_flat = []
        for key in client_params.keys():
            # Ensure both tensors are on the same device for subtraction
            delta = client_params[key].to(device) - global_params[key].to(device)
            delta_flat.append(delta.view(-1))  # Flatten
        deltas.append(torch.cat(delta_flat))  # Concatenate into single vector

    # 2. Harmonize: iterate through all pairs
    # Keep a copy of ORIGINAL deltas for reading (never modify this)
    deltas_original = [d.clone() for d in deltas]
    harmonized_deltas = [d.clone() for d in deltas]  # Working copies

    # Clip gradient norm for stability
    max_norm = 10.0

    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            # Always read from ORIGINAL deltas
            g_i_orig = deltas_original[i]
            g_j_orig = deltas_original[j]

            # Compute dot product with ORIGINAL vectors
            dot_product = torch.dot(g_i_orig, g_j_orig).item()

            # Check for conflict (negative dot product)
            if dot_product < 0:
                # Compute norms using ORIGINAL values
                norm_i_sq = torch.dot(g_i_orig, g_i_orig).item()
                norm_j_sq = torch.dot(g_j_orig, g_j_orig).item()

                # Avoid division by zero
                if norm_i_sq > 1e-10 and norm_j_sq > 1e-10:
                    # Get CURRENT harmonized deltas
                    h_i = harmonized_deltas[i]
                    h_j = harmonized_deltas[j]

                    # Project using ORIGINAL values but apply to CURRENT
                    proj_i = (torch.dot(h_i, g_j_orig) / norm_j_sq) * g_j_orig
                    proj_j = (torch.dot(h_j, g_i_orig) / norm_i_sq) * g_i_orig

                    # Apply projections
                    harmonized_deltas[i] = h_i - proj_i
                    harmonized_deltas[j] = h_j - proj_j

                    # Clip after each projection to prevent explosion
                    for idx in [i, j]:
                        grad_norm = torch.norm(harmonized_deltas[idx])
                        if grad_norm > max_norm:
                            harmonized_deltas[idx] = harmonized_deltas[idx] * (
                                max_norm / grad_norm
                            )

    # Prevent exploding gradients with NORM clipping (not value clipping)
    # This preserves gradient direction while limiting magnitude
    max_norm = 10.0
    for client_idx in range(num_clients):
        grad_norm = torch.norm(harmonized_deltas[client_idx])
        if grad_norm > max_norm:
            harmonized_deltas[client_idx] = harmonized_deltas[client_idx] * (
                max_norm / grad_norm
            )

    # 3. Reconstruct state_dicts from harmonized flat vectors
    harmonized_updates = []
    for client_idx in range(num_clients):
        harmonized_params = {}
        offset = 0

        global_params_device = {k: v.to(device) for k, v in global_params.items()}

        for key in client_updates[client_idx].keys():
            param_shape = client_updates[client_idx][key].shape
            param_numel = client_updates[client_idx][key].numel()

            # Extract this parameter's portion from flat vector
            param_flat = harmonized_deltas[client_idx][offset : offset + param_numel]
            harmonized_delta = param_flat.view(param_shape)

            # Reconstruct: θ_i_new = θ_global + Δθ_i_harmonized
            harmonized_params[key] = global_params_device[key] + harmonized_delta

            offset += param_numel

        harmonized_updates.append(harmonized_params)

    return harmonized_updates


# --- Component 5: Evaluation ---


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate the global model on the hold-out test set.

    Returns:
        accuracy: Test accuracy (%)
        loss: Average test loss
    """
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Sum up batch loss
            test_loss += criterion(output, target).item()
            # Get the index of the max log-probability
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = test_loss / total

    return accuracy, avg_loss


# --- Component 6: Client Drift Metric ---


def compute_client_drift(
    client_models: List[Dict[str, torch.Tensor]],
    global_model_params: Dict[str, torch.Tensor],
) -> float:
    """
    Calculates weight divergence (client drift) as specified in the manual:
    d_θ^t = (1/M) Σ ||θ_i(t,K) - θ_g^t||

    Measures the average L2 norm of the difference between client models
    (after local training) and the global model (before local training).

    Args:
        client_models: List of client model parameters after local training
        global_model_params: Global model parameters before local training

    Returns:
        avg_drift: Average L2 distance between client and global models
    """
    total_drift = 0.0
    num_clients = len(client_models)

    if num_clients == 0:
        return 0.0

    # Convert global model params to a flat vector for easier comparison
    global_vector = nn.utils.parameters_to_vector(
        [global_model_params[key].clone() for key in global_model_params]
    )

    for client_params in client_models:
        client_vector = nn.utils.parameters_to_vector(
            [client_params[key].clone() for key in client_params]
        )

        # Calculate L2 norm of the difference vector
        drift = torch.norm(client_vector - global_vector, p=2)
        total_drift += drift.item()

    avg_drift = total_drift / num_clients
    return avg_drift


# --- Component 7: Main Training Loop ---


def federated_train(
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    client_fraction: float,
    lr: float,
    batch_size: int,
    device: torch.device,
    seed: int = 42,
    mu: float = 0.0,
    rho: float = 0.0,
    use_scaffold: bool = False,
    use_fedgh: bool = False,
    train_loaders: List[DataLoader] = None,
    test_loader: DataLoader = None,
    client_sizes: List[int] = None,
) -> Dict[str, List]:
    """
    Main FedAvg training loop.

    Orchestrates the entire process:
    1. Client Sampling
    2. Broadcast & Local Training
    3. Aggregation
    4. Evaluation

    Pattern learned from FedML's FedAvgAPI.train() method, but simplified.

    Returns:
        history: Dict containing metrics per round
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize model, data, and tracking
    global_model = SimpleCNN().to(device)
    # --- Data Loading Logic ---
    if train_loaders is None:
        print("Loading IID data (Task 2 compatibility mode)...")
        train_loaders, test_loader, client_sizes = load_cifar10_iid(
            num_clients, batch_size
        )
    else:
        print("Using provided data loaders (non-IID mode)...")
        # Data is already provided, just confirm num_clients
        num_clients = len(train_loaders)

    # Calculate base client weights (N_i)
    total_samples = sum(client_sizes)

    c_global = None
    c_local_all = None
    if use_scaffold:
        # Get model param shapes, must be on CPU to save GPU mem
        with torch.no_grad():
            c_global = [
                torch.zeros_like(p.cpu())
                for p in global_model.parameters()
                if p.requires_grad
            ]
        # c_local_all is a list (one per client) of lists of tensors (all on CPU)
        c_local_all = [
            [torch.zeros_like(p) for p in c_global] for _ in range(num_clients)
        ]

    # History dictionary to store metrics
    history = {"test_acc": [], "test_loss": [], "client_drift": [], "rounds": []}

    # --- Determine Algorithm Name ---
    algo_name = "FedAvg"
    if mu > 0:
        algo_name = "FedProx"
    if rho > 0:
        algo_name = "FedSAM"
        if mu > 0:
            algo_name = "FedProx+SAM"
    if use_scaffold:
        if mu > 0:
            algo_name = "FedProx+SCAFFOLD"
        elif rho > 0:
            algo_name = "SCAFFOLD+SAM"
        else:
            algo_name = "SCAFFOLD"
    if use_fedgh:
        algo_name += "+FedGH"

    print(f"--- Starting Federated Training (Algorithm: {algo_name}) ---")
    print(f"  Clients: {num_clients} (sampling {client_fraction*100}%)")
    print(f"  Rounds: {num_rounds}")
    print(f"  Local Epochs (K): {local_epochs}")
    if mu > 0:
        print(f"  FedProx (mu): {mu}")
    if use_scaffold:
        print(f"  SCAFFOLD: Enabled")
    if use_fedgh:
        print(f"  FedGH: Enabled")
    print("=" * 70)

    pbar = tqdm(range(num_rounds), desc="Federated Training")
    for round_idx in pbar:
        # 1. Client Sampling
        num_selected = max(1, int(client_fraction * num_clients))
        selected_clients_indices = np.random.choice(
            range(num_clients), size=num_selected, replace=False
        )
        print(f"Selected clients: {selected_clients_indices}")

        # 2. Broadcast global model and train locally
        client_models_params = []
        selected_client_weights = []
        client_c_deltas = []

        # Get global params *before* local training for drift calculation
        global_params = global_model.state_dict()

        for client_idx in selected_clients_indices:
            # Create a local model copy
            local_model = SimpleCNN()
            # Broadcast: Set local model to global model state
            local_model.load_state_dict(copy.deepcopy(global_params))

            c_g_tensors = c_global if use_scaffold else None
            c_l_i_tensors = c_local_all[client_idx] if use_scaffold else None

            # Local training
            updated_params, delta_c_i = client_update(
                local_model,
                train_loaders[client_idx],
                epochs=local_epochs,
                lr=lr,
                device=device,
                mu=mu,
                rho=rho,
                c_global=c_g_tensors,
                c_local_i=c_l_i_tensors,
            )

            client_models_params.append(updated_params)
            selected_client_weights.append(client_sizes[client_idx])
            if use_scaffold:
                client_c_deltas.append(delta_c_i)

        # 3. Normalize weights (w_i = N_i / N_total_sampled)
        total_selected_weight = sum(selected_client_weights)
        normalized_weights = [
            w / total_selected_weight for w in selected_client_weights
        ]

        # 4. Compute client drift BEFORE aggregation
        drift = compute_client_drift(client_models_params, global_params)

        # 4.5. Apply FedGH harmonization if enabled (NEW)
        if use_fedgh:
            client_models_params = harmonize_gradients(
                client_models_params, global_params, device
            )

        # 5. Aggregate Model Weights
        aggregated_params = aggregate_models(client_models_params, normalized_weights)
        global_model.load_state_dict(aggregated_params)

        if use_scaffold:
            # 5b. Update local control variates (only for selected clients)
            # c_local_i = c_local_i + delta_c_i
            for idx, client_idx in enumerate(selected_clients_indices):
                delta_c_i = client_c_deltas[idx]
                for i in range(len(c_local_all[client_idx])):
                    c_local_all[client_idx][i] += delta_c_i[i].cpu()

            # 5c. Update global control variate by averaging ALL local variates
            # c_global = (1/N) * sum(c_local_all_i)

            # Zero out the old c_global
            for i in range(len(c_global)):
                c_global[i].zero_()

            # Sum all local c_i's
            for c_local_i in c_local_all:
                for i in range(len(c_global)):
                    c_global[i] += c_local_i[i]

            # Divide by N
            N = num_clients
            for i in range(len(c_global)):
                c_global[i] /= N

        # 6. Evaluate
        test_acc, test_loss = evaluate_model(global_model, test_loader, device)

        # 7. Log
        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)
        history["client_drift"].append(drift)
        history["rounds"].append(round_idx + 1)

        # Update the progress bar's description with the latest metrics
        pbar.set_postfix(
            {
                "Test Acc": f"{test_acc:.2f}%",
                "Test Loss": f"{test_loss:.4f}",
                "Drift": f"{drift:.4f}",
            }
        )

    pbar.close()
    print("\n" + "=" * 70)
    print(f"--- Training Complete ---")
    print(f"Final Accuracy: {history['test_acc'][-1]:.2f}%")

    return global_model, history


# --- Main execution block for simple testing ---
if __name__ == "__main__":
    """
    A simple test run to ensure the framework executes.
    """
    print("Running basic framework sanity check...")

    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A small-scale test
    test_history = federated_train(
        num_clients=5,
        num_rounds=3,
        local_epochs=1,
        client_fraction=0.4,  # Test client sampling (2 clients)
        lr=0.01,
        batch_size=32,
        device=test_device,
        seed=42,
    )

    print("\nSanity Check History:")
    print(pd.DataFrame(test_history).to_markdown(index=False))

    print("\nFramework test complete.")
