# %% [markdown]
# # 📋 ATML PA4 - Task 1: FedSGD vs. Centralized SGD Equivalence
# 
# **Goal:** Demonstrate that FedSGD (with full-batch gradients) is mathematically equivalent to centralized SGD.
# 
# **McMahan's FedSGD:** Each client computes the gradient on its **entire local dataset** (which they denote as `B=∞`, or full-batch), and then the server averages these gradients (which is equivalent to averaging the models after one step with the same LR).
# 
# We will compare this to a centralized model trained for the same number of steps, where each step is also a full-batch gradient computation over the **entire global dataset**.

# %% [markdown]
# ## **Part 1: Environment Setup**

# %% [markdown]
# ### **1.1. Imports and Configuration**

# %%
# ============================================
# 📦 Imports and Environment Setup
# ============================================
import copy
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Add path to federated_learning.py
# (Assuming this notebook is in a subdir and FedAvg is at ../FedAvg)
fed_avg_path = os.path.abspath(os.path.join(os.getcwd(), "../FedAvg"))
if fed_avg_path not in sys.path:
    print(f"Adding path: {fed_avg_path}")
    sys.path.append(fed_avg_path)

try:
    # We only need these specific components for this task
    from federated_learning import SimpleCNN, aggregate_models, evaluate_model

    print("Successfully imported from federated_learning module.")
except ImportError:
    print(f"Error: 'federated_learning.py' not found in path: {fed_avg_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================
# ⚙️ Main Configuration
# ============================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

NUM_CLIENTS = 6
NUM_ROUNDS = 50  # FedSGD rounds / Centralized steps
LEARNING_RATE = 0.01

# %% [markdown]
# ### **1.2. Directory and Experiment Setup**

# %%
# ============================================
# 📂 Directory and Experiment Setup
# ============================================

# Set to True to force re-training even if results.json exists
RETRAIN_TASK1 = True 

# Define directories
PLOT_DIR = 'plots/task1'
JSON_DIR = 'json_results/task1'
MODEL_DIR = 'pth_models/task1'

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Define file paths
json_path = os.path.join(JSON_DIR, 'task1_results.json')
fedsgd_model_path = os.path.join(MODEL_DIR, 'task1_fedsgd_model.pth')
central_model_path = os.path.join(MODEL_DIR, 'task1_centralized_model.pth')

print(f"JSON results will be saved to: {json_path}")
print(f"Models will be saved to: {MODEL_DIR}")

# %% [markdown]
# ## **Part 2: Data Loading (Full-Batch Setup)**

# %% [markdown]
# ### **2.1. Load and Split CIFAR-10**

# %%
# ============================================
# 💾 Load CIFAR-10 and Split IID
# ============================================

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

# Split training data IID across clients
total_size = len(train_dataset)
indices = np.random.permutation(total_size)
split_size = total_size // NUM_CLIENTS

client_datasets = []
client_sizes = []

for i in range(NUM_CLIENTS):
    start_idx = i * split_size
    end_idx = (i + 1) * split_size if i < NUM_CLIENTS - 1 else total_size
    client_indices = indices[start_idx:end_idx]
    client_subset = Subset(train_dataset, client_indices)
    client_datasets.append(client_subset)
    client_sizes.append(len(client_indices))

print(f"Split data across {NUM_CLIENTS} clients")
print(f"Client sizes: {client_sizes}")
print(f"Total training samples: {sum(client_sizes)}")

# %% [markdown]
# ### **2.2. Create Full-Batch Data Loaders**
# 
# This is the key to simulating FedSGD. We create data loaders where the `batch_size` is set to the *entire length* of the client's dataset. This means that when we iterate on the loader, the first (and only) batch contains all of that client's data.

# %%
# ============================================
# 🚚 Create FULL-BATCH Data Loaders
# ============================================

# KEY: Set batch_size = entire client dataset!
fedsgd_loaders = []
for client_data in client_datasets:
    # batch_size = len(dataset) → 1 batch = entire data
    loader = DataLoader(
        client_data,
        batch_size=len(client_data),  # ← FULL BATCH!
        shuffle=True,
        num_workers=2
    )
    fedsgd_loaders.append(loader)

print(f"Created {len(fedsgd_loaders)} full-batch data loaders for FedSGD.")

# Global test loader (standard mini-batch)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# %% [markdown]
# ## **Part 3: Define Training Functions**

# %% [markdown]
# ### **3.1. FedSGD (McMahan Style)**
# 
# We define a custom `fedsgd_client_update` function that performs *exactly one* gradient step using the full-batch loader. We also create a `run_fedsgd` orchestrator that mimics the federated training loop (distribute, update, aggregate, evaluate) but calls our custom update function.

# %%
# ============================================
# 💻 FedSGD Implementation (McMahan Style)
# ============================================

def fedsgd_client_update(model, full_batch_loader, lr, device):
    """
    McMahan's FedSGD: Compute gradient on ENTIRE local dataset.
    
    Since loader has batch_size=full_data, first batch = all data.
    """
    model.to(device)
    model.train()
    
    # Use the same optimizer settings as in federated_learning.py for consistency
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Get the single full-batch
    try:
        data, target = next(iter(full_batch_loader))
    except StopIteration:
        print("Warning: Client data loader was empty.")
        return model.state_dict() # Return unchanged parameters
        
    data, target = data.to(device), target.to(device)
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward() # Compute gradient
    optimizer.step()  # Apply gradient step
    
    return model.state_dict()


def run_fedsgd(num_clients, num_rounds, lr, device, seed=42):
    """
    Run FedSGD for specified rounds.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    global_model = SimpleCNN().to(device)
    
    history = {
        "test_acc": [],
        "test_loss": [],
        "rounds": []
    }
    
    print(f"\n{'='*70}")
    print(f"🚀 Running FedSGD (Full-Batch)")
    print(f"  Clients: {num_clients}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Learning Rate: {lr}")
    print(f"{'='*70}\n")
    
    for round_idx in range(num_rounds):
        # All clients participate (C=1.0)
        client_models_params = []
        client_weights = []
        
        global_params = copy.deepcopy(global_model.state_dict())
        
        for client_idx in range(num_clients):
            # Create local model
            local_model = SimpleCNN()
            local_model.load_state_dict(global_params)
            
            # Full-batch gradient computation & step
            updated_params = fedsgd_client_update(
                local_model,
                fedsgd_loaders[client_idx],
                lr=lr,
                device=device
            )
            
            client_models_params.append(updated_params)
            client_weights.append(client_sizes[client_idx])
        
        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        # Aggregate (using the function from our .py file)
        aggregated_params = aggregate_models(client_models_params, normalized_weights)
        global_model.load_state_dict(aggregated_params)
        
        # Evaluate (using the function from our .py file)
        test_acc, test_loss = evaluate_model(global_model, test_loader, device)
        
        history['test_acc'].append(test_acc)
        history['test_loss'].append(test_loss)
        history['rounds'].append(round_idx + 1)
        
        if (round_idx + 1) % 10 == 0 or round_idx == 0:
            print(f"Round {round_idx + 1:2d}/{num_rounds} | "
                  f"Test Acc: {test_acc:6.2f}% | Test Loss: {test_loss:.4f}")
    
    print(f"\n--- FedSGD Training Complete ---")
    print(f"Final Accuracy: {history['test_acc'][-1]:.2f}%")
    return global_model, history

# %% [markdown]
# ### **3.2. Centralized SGD (Full-Batch)**
# 
# Now we define the centralized equivalent. We concatenate all client datasets into one large global dataset and create a *single* full-batch loader. We then run SGD for the same number of steps as the FedSGD rounds.

# %%
# ============================================
# 💻 Centralized Training (Full-Batch)
# ============================================

def run_centralized(num_steps, lr, device, seed=42):
    """
    Centralized training with full-batch gradient on ALL data.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = SimpleCNN().to(device)
    
    # Combine all training data
    combined_dataset = torch.utils.data.ConcatDataset(client_datasets)
    
    # Full-batch loader (entire 50k samples!)
    full_loader = DataLoader(
        combined_dataset,
        batch_size=len(combined_dataset),  # ← ALL DATA
        shuffle=True,
        num_workers=2
    )
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        "test_acc": [],
        "test_loss": [],
        "rounds": []
    }
    
    print(f"\n{'='*70}")
    print(f"🚀 Running Centralized SGD (Full-Batch)")
    print(f"  Steps: {num_steps}")
    print(f"  Learning Rate: {lr}")
    print(f"{'='*70}\n")
    
    for step in range(num_steps):
        model.train()
        
        # Get full batch
        try:
            data, target = next(iter(full_loader))
        except StopIteration:
            print("Error: Centralized loader was empty.")
            break
            
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Evaluate
        test_acc, test_loss = evaluate_model(model, test_loader, device)
        
        history['test_acc'].append(test_acc)
        history['test_loss'].append(test_loss)
        history['rounds'].append(step + 1)
        
        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step {step + 1:2d}/{num_steps} | "
                  f"Test Acc: {test_acc:6.2f}% | Test Loss: {test_loss:.4f}")
    
    print(f"\n--- Centralized Training Complete ---")
    print(f"Final Accuracy: {history['test_acc'][-1]:.2f}%")
    return model, history

# %% [markdown]
# ## **Part 4: Execute Training**
# 
# We now run both functions. If `task1_results.json` exists and `RETRAIN_TASK1` is `False`, we will skip this and load the results directly from the file.

# %%
# ============================================
# 🚀 Run Experiment (or Load Results)
# ============================================

if not os.path.exists(json_path) or RETRAIN_TASK1:
    print(f"Running training for Task 1...")
    
    # Run FedSGD
    fedsgd_model, fedsgd_history = run_fedsgd(
        num_clients=NUM_CLIENTS,
        num_rounds=NUM_ROUNDS,
        lr=LEARNING_RATE,
        device=device,
        seed=SEED
    )

    # Run Centralized (same number of steps)
    centralized_model, centralized_history = run_centralized(
        num_steps=NUM_ROUNDS,
        lr=LEARNING_RATE,
        device=device,
        seed=SEED
    )
    
    # --- Save Models ---
    print(f"\nSaving models to {MODEL_DIR}...")
    torch.save(fedsgd_model.state_dict(), fedsgd_model_path)
    torch.save(centralized_model.state_dict(), central_model_path)

    # --- Calculate Weight Difference ---
    print("Calculating weight differences...")
    fedsgd_params = fedsgd_model.state_dict()
    central_params = centralized_model.state_dict()
    per_layer_diff = {}
    total_diff = 0
    for key in fedsgd_params.keys():
        diff = torch.norm(fedsgd_params[key].cpu() - central_params[key].cpu()).item()
        per_layer_diff[key] = diff
        total_diff += diff
    
    # --- Save Results to JSON ---
    results = {
        "fedsgd": fedsgd_history,
        "centralized": centralized_history,
        "weight_difference_total": total_diff,
        "weight_difference_per_layer": per_layer_diff
    }
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {json_path}")

else:
    print(f"Loading existing results from {json_path}")
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    fedsgd_history = results['fedsgd']
    centralized_history = results['centralized']
    total_diff = results['weight_difference_total']
    print("...Load complete.")

# %% [markdown]
# ## **Part 5: Analysis and Plotting**

# %% [markdown]
# ### **5.1. Accuracy Comparison**

# %%
# ============================================
# 📈 Compare Final Results
# ============================================

print(f"\n{'='*70}")
print(f"Comparison (After {NUM_ROUNDS} Rounds/Steps)")
print(f"{'='*70}")
fedsgd_final_acc = fedsgd_history['test_acc'][-1]
central_final_acc = centralized_history['test_acc'][-1]

print(f"FedSGD Final Accuracy:       {fedsgd_final_acc:10.6f}%")
print(f"Centralized Final Accuracy:  {central_final_acc:10.6f}%")
print(f"----------------------------------------------")
print(f"Absolute Difference:         {abs(fedsgd_final_acc - central_final_acc):10.6f}%")

# %% [markdown]
# ### **5.2. Plotting**
# 
# We plot the accuracy and loss curves over time. We expect them to be nearly identical, proving the equivalence.

# %%
# ============================================
# 📊 Plot Comparison
# ============================================
plot_path = os.path.join(PLOT_DIR, 'task1_fedsgd_vs_centralized.png')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Task 1: FedSGD vs Centralized (Full-Batch Equivalence)', fontsize=16, y=1.02)

# Accuracy
axes[0].plot(fedsgd_history['rounds'], fedsgd_history['test_acc'], 
             label='FedSGD', marker='o', markersize=4, alpha=0.8)
axes[0].plot(centralized_history['rounds'], centralized_history['test_acc'], 
             label='Centralized', marker='s', markersize=4, linestyle='--', alpha=0.8)
axes[0].set_xlabel('Round/Step')
axes[0].set_ylabel('Test Accuracy (%)')
axes[0].set_title('Test Accuracy')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)

# Loss
axes[1].plot(fedsgd_history['rounds'], fedsgd_history['test_loss'], 
             label='FedSGD', marker='o', markersize=4, alpha=0.8)
axes[1].plot(centralized_history['rounds'], centralized_history['test_loss'], 
             label='Centralized', marker='s', markersize=4, linestyle='--', alpha=0.8)
axes[1].set_xlabel('Round/Step')
axes[1].set_ylabel('Test Loss')
axes[1].set_title('Test Loss')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to {plot_path}")
plt.show()

# %% [markdown]
# ### **5.3. Model Weight Difference Analysis**
# 
# Finally, we load the saved models (if they aren't already in memory) and compute the L2 norm of the difference between their parameters. If the methods are truly equivalent, this difference should be extremely close to zero (e.g., `< 1e-5`), accounting for floating-point arithmetic variations.

# %%
# ============================================
# 🔬 Weight Difference Analysis
# ============================================

# If we didn't train, load the models from disk for analysis
if 'fedsgd_model' not in locals() or 'centralized_model' not in locals():
    print("Loading models from disk for weight comparison...")
    try:
        fedsgd_model = SimpleCNN()
        fedsgd_model.load_state_dict(torch.load(fedsgd_model_path, map_location=device))
        fedsgd_model.to(device)
        
        centralized_model = SimpleCNN()
        centralized_model.load_state_dict(torch.load(central_model_path, map_location=device))
        centralized_model.to(device)
        print("...Models loaded.")
    except FileNotFoundError:
        print("Error: Model files not found. Please run training first by setting RETRAIN_TASK1 = True.")
        fedsgd_model = None # Set to None to skip analysis

if fedsgd_model is not None:
    # Compare final model weights
    fedsgd_params = fedsgd_model.state_dict()
    central_params = centralized_model.state_dict()

    per_layer_diffs_loaded = {}
    total_diff_loaded = 0
    
    print("\nPer-Layer L2 Norm Difference:")
    for key in fedsgd_params.keys():
        diff = torch.norm(fedsgd_params[key] - central_params[key]).item()
        per_layer_diffs_loaded[key] = diff
        total_diff_loaded += diff
        print(f"  {key:20s}: L2 diff = {diff:.8e}")
    
    print(f"\n{'='*70}")
    print(f"Total L2 difference between model weights: {total_diff_loaded:.8e}")
    
    # Final verdict
    # (Floating point errors can accumulate, so we check against a small epsilon)
    if total_diff_loaded < 1e-4:
        print("✅ Equivalence PROVEN. The models are mathematically identical within floating-point error.")
    else:
        print(f"❌ Models are not equivalent. Total difference ({total_diff_loaded:.8e}) is larger than expected.")
else:
    print("\nSkipping weight analysis as models could not be loaded.")

print("\n✅ Task 1 complete!")


