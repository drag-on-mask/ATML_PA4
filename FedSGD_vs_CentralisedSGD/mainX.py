# task1_main.py
# Implements Task 1: FedSGD vs. Centralized SGD

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import copy
import matplotlib.pyplot as plt

# --- Component 1: Import from your Task 2 file ---
# We reuse the model, aggregation, and evaluation functions
# from your 'federated_learning.py' file.
try:
    from federated_learning import SimpleCNN, aggregate_models, evaluate_model
    print("Successfully imported components from federated_learning.py")
except ImportError:
    print("ERROR: Could not find 'federated_learning.py'.")
    print("Please make sure this script is in the same directory.")
    exit()

# --- Component 2: Task 1 Configuration ---
NUM_CLIENTS = 6       # As suggested in assignment [cite: 105]
NUM_ROUNDS = 20      # Run for 10-20 iterations [cite: 115]
LEARNING_RATE = 0.01
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Component 3: Task 1 Specific Data Loading ---
def load_data_task1(num_clients: int) -> tuple:
    """
    Loads CIFAR-10 and prepares data for Task 1:
    1. Centralized Loader: Full training set in one batch. 
    2. FedSGD Loaders: IID split, each client's loader has one full batch. [cite: 105, 109]
    3. Test Loader: Standard test set.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # 1. Centralized Loader
    centralized_loader = DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=False
    )
    print(f"Centralized loader: 1 batch of {len(train_dataset)} samples.")

    # 2. FedSGD Loaders (IID)
    total_size = len(train_dataset)
    indices = np.random.permutation(total_size)
    # Split evenly [cite: 105]
    split_size = total_size // num_clients
    
    fedsgd_loaders = []
    client_sizes = []

    for i in range(num_clients):
        start_idx = i * split_size
        # Assign remainder to the last client
        end_idx = (i + 1) * split_size if i < num_clients - 1 else total_size
        client_indices = indices[start_idx:end_idx]
        
        client_subset = Subset(train_dataset, client_indices)
        
        # This is the key change: batch_size = full local dataset size 
        client_loader = DataLoader(
            client_subset, batch_size=len(client_indices), shuffle=False
        )
        
        fedsgd_loaders.append(client_loader)
        client_sizes.append(len(client_indices))

    print(f"FedSGD loaders: {num_clients} clients.")
    print(f"Client sizes: {client_sizes}")

    # Calculate client weights (N_i / N) for aggregation [cite: 101]
    total_samples = sum(client_sizes)
    client_weights = [size / total_samples for size in client_sizes]

    # 3. Test Loader
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    return centralized_loader, fedsgd_loaders, test_loader, client_weights

# --- Component 4: Centralized SGD Trainer ---
def train_centralized(model, data_loader, test_loader, lr, num_rounds):
    """
    Trains a model using centralized, full-batch Gradient Descent. [cite: 111-114]
    """
    model.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {'test_acc': [], 'test_loss': [], 'weights': []}
    
    # Get the single, full batch of data
    full_data, full_target = next(iter(data_loader))
    full_data, full_target = full_data.to(DEVICE), full_target.to(DEVICE)

    print("\n--- Starting Centralized Training ---")
    for round_idx in range(num_rounds):
        model.train()
        
        # Perform one full-batch SGD step [cite: 114]
        optimizer.zero_grad()
        output = model(full_data)
        loss = criterion(output, full_target)
        loss.backward()
        optimizer.step()
        
        # Evaluate
        test_acc, test_loss = evaluate_model(model, test_loader, DEVICE)
        history['test_acc'].append(test_acc)
        history['test_loss'].append(test_loss)
        history['weights'].append(copy.deepcopy(model.state_dict()))
        
        print(f"Round {round_idx+1}/{num_rounds} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
    print("--- Centralized Training Complete ---")
    return history

# --- Component 5: FedSGD Trainer ---

def client_update_task1(model, data_loader, lr):
    """
    Performs K=1 local step of full-batch GD. [cite: 108, 109]
    """
    model.to(DEVICE)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # The data loader was created to have only ONE batch 
    try:
        data, target = next(iter(data_loader))
    except StopIteration:
        print("Error: Client data loader is empty.")
        return model.state_dict()
        
    data, target = data.to(DEVICE), target.to(DEVICE)

    # Perform one full-batch SGD step
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    return model.state_dict()

def train_fedsgd(model, client_loaders, test_loader, client_weights, lr, num_rounds):
    """
    Trains a model using FedSGD.
    - K=1 full-batch step per client [cite: 108]
    - Full client participation 
    """
    model.to(DEVICE)
    history = {'test_acc': [], 'test_loss': [], 'weights': []}
    
    print("\n--- Starting FedSGD Training ---")
    for round_idx in range(num_rounds):
        client_models_params = []
        global_params = model.state_dict()
        
        # All clients participate in each round 
        for client_idx in range(len(client_loaders)):
            local_model = SimpleCNN()
            local_model.load_state_dict(copy.deepcopy(global_params))
            
            # Perform the single local step
            updated_params = client_update_task1(
                local_model,
                client_loaders[client_idx],
                lr=lr
            )
            client_models_params.append(updated_params)
            
        # Aggregate the *model parameters* (not gradients)
        # Note: Aggregating 1-step models is equivalent to aggregating 1-step gradients [cite: 101]
        aggregated_params = aggregate_models(client_models_params, client_weights)
        model.load_state_dict(aggregated_params)
        
        # Evaluate
        test_acc, test_loss = evaluate_model(model, test_loader, DEVICE)
        history['test_acc'].append(test_acc)
        history['test_loss'].append(test_loss)
        history['weights'].append(copy.deepcopy(model.state_dict()))
        
        print(f"Round {round_idx+1}/{num_rounds} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    print("--- FedSGD Training Complete ---")
    return history

# --- Component 6: Main Comparison Logic ---
def main():
    """
    Runs the full comparison for Task 1.
    """
    # 1. Load Data
    central_loader, fedsgd_loaders, test_loader, client_weights = load_data_task1(NUM_CLIENTS)
    
    # 2. Initialize Models
    # Must start with the *exact* same weights 
    model_fedsgd = SimpleCNN()
    model_centralized = SimpleCNN()
    model_centralized.load_state_dict(copy.deepcopy(model_fedsgd.state_dict()))

    # 3. Run Training
    history_fedsgd = train_fedsgd(
        model_fedsgd, fedsgd_loaders, test_loader, client_weights, LEARNING_RATE, NUM_ROUNDS
    )
    history_centralized = train_centralized(
        model_centralized, central_loader, test_loader, LEARNING_RATE, NUM_ROUNDS
    )
    
    # 4. Analyze and Verify Equivalence [cite: 116]
    print("\n--- Verification of Equivalence ---")
    print("Round | FedSGD Acc (%) | Central Acc (%) | Norm of Weight Difference")
    print("----------------------------------------------------------------------")
    
    weight_diffs = []
    rounds = range(1, NUM_ROUNDS + 1)
    
    for i in range(NUM_ROUNDS):
        acc_f = history_fedsgd['test_acc'][i]
        acc_c = history_centralized['test_acc'][i]
        
        # Get weights from this round
        params_f = history_fedsgd['weights'][i]
        params_c = history_centralized['weights'][i]
        
        # Convert to vectors for comparison
        vec_f = nn.utils.parameters_to_vector([p for p in params_f.values()])
        vec_c = nn.utils.parameters_to_vector([p for p in params_c.values()])
        
        # Calculate L2 norm of the difference [cite: 118]
        diff = torch.norm(vec_f - vec_c, p=2).item()
        weight_diffs.append(diff)
        
        print(f"  {i+1:2d}  |    {acc_f:6.2f}    |    {acc_c:6.2f}     | {diff:e}")
        
    # 5. Plot Results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Test Accuracy Comparison [cite: 119]
    ax1.plot(rounds, history_fedsgd['test_acc'], 'bo-', label='FedSGD', markersize=4)
    ax1.plot(rounds, history_centralized['test_acc'], 'rs--', label='Centralized SGD', markersize=4)
    ax1.set_title(f'Task 1: FedSGD vs. Centralized SGD (IID, Full-Batch, {NUM_CLIENTS} Clients)')
    ax1.set_xlabel('Communication Round / SGD Step')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Norm of Weight Difference [cite: 118]
    ax2.plot(rounds, weight_diffs, 'g-o', label='L2 Norm( || W_fedsgd - W_central || )', markersize=4)
    ax2.set_title('Difference Between Model Weights')
    ax2.set_xlabel('Communication Round / SGD Step')
    ax2.set_ylabel('L2 Norm')
    ax2.set_yscale('log') # Use log scale to see small differences
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('task1_comparison.png', dpi=300)
    print("\nSaved comparison plot to 'task1_comparison.png'")
    plt.show()

if __name__ == "__main__":
    main()