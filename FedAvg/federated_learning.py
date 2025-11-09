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


# --- Component 3: Client Training ---


def client_update(
    model: nn.Module,
    data_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Performs K local epochs of SGD on client data.

    This function simulates a single client's local training phase.
    Pattern learned from FedML's Client.train() which delegates to a trainer.

    Args:
        model: PyTorch model (a copy of the global model)
        data_loader: Client's local DataLoader
        epochs: K, the number of local epochs
        lr: Learning rate for local SGD
        device: 'cuda' or 'cpu'

    Returns:
        updated_model_params: state_dict() of the trained local model
    """
    model.to(device)
    model.train()

    # Standard SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Return the updated model parameters
    return model.state_dict()


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


# --- Component 5: Evaluation ---


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate the global model on the hold-out test set.
    
    Returns:
        accuracy: Test accuracy (%)
        loss: Average test loss
    """
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss(reduction='sum')
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
    
    accuracy = 100. * correct / total
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
    train_loaders, test_loader, client_sizes = load_cifar10_iid(num_clients, batch_size)

    # Calculate base client weights (N_i)
    total_samples = sum(client_sizes)

    # History dictionary to store metrics
    history = {"test_acc": [], "test_loss": [], "client_drift": [], "rounds": []}

    print(f"--- Starting FedAvg Training ---")
    print(f"  Clients: {num_clients} (sampling {client_fraction*100}%)")
    print(f"  Rounds: {num_rounds}")
    print(f"  Local Epochs (K): {local_epochs}")
    print("="*70)

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

        # Get global params *before* local training for drift calculation
        global_params = global_model.state_dict()

        for client_idx in selected_clients_indices:
            # Create a local model copy
            local_model = SimpleCNN()
            # Broadcast: Set local model to global model state
            local_model.load_state_dict(copy.deepcopy(global_params))

            # Local training
            updated_params = client_update(
                local_model,
                train_loaders[client_idx],
                epochs=local_epochs,
                lr=lr,
                device=device,
            )

            client_models_params.append(updated_params)
            selected_client_weights.append(client_sizes[client_idx])

        # 3. Normalize weights (w_i = N_i / N_total_sampled)
        total_selected_weight = sum(selected_client_weights)
        normalized_weights = [
            w / total_selected_weight for w in selected_client_weights
        ]

        # 4. Compute client drift BEFORE aggregation
        drift = compute_client_drift(client_models_params, global_params)

        # 5. Aggregate
        aggregated_params = aggregate_models(client_models_params, normalized_weights)
        global_model.load_state_dict(aggregated_params)

        # 6. Evaluate
        test_acc, test_loss = evaluate_model(global_model, test_loader, device)

        # 7. Log
        history['test_acc'].append(test_acc)
        history['test_loss'].append(test_loss)
        history['client_drift'].append(drift)
        history['rounds'].append(round_idx + 1)

        # Update the progress bar's description with the latest metrics
        pbar.set_postfix({
            'Test Acc': f"{test_acc:.2f}%",
            'Test Loss': f"{test_loss:.4f}",
            'Drift': f"{drift:.4f}"
        })

    pbar.close()
    print("\n" + "="*70)
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
