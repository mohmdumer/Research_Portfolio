# main.py
# Purpose: Main script to run the federated learning simulation
# Ties together all components: data partitioning, client creation, server setup, and simulation

import sys
import os
from typing import Dict, List
import torch
import flwr as fl
from flwr.simulation import start_simulation
from flwr.common import Context

# Import our modules
from data import MNISTDataPartitioner
from client import MNISTFlowerClient, create_client
from server import create_server_strategy, run_simulation_with_early_stopping
from model import create_model
from utils import (
    setup_logging,
    create_experiment_config,
    validate_config,
    save_experiment_results,
    check_cuda_availability
)

# Global variables to store data (needed for simulation)
CLIENT_DATALOADERS = {}
TEST_DATALOADER = None
EXPERIMENT_CONFIG = None


def check_dependencies():
    """
    Check if all required dependencies are installed.
    Exit gracefully with helpful error messages if dependencies are missing.
    """
    missing_deps = []
    
    try:
        import flwr
    except ImportError:
        missing_deps.append("flwr[simulation]")
    
    try:
        import torch
        import torchvision
    except ImportError:
        missing_deps.append("torch torchvision")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    if missing_deps:
        print("❌ Missing dependencies detected:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n📦 Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        print("\n   Or manually install:")
        for dep in missing_deps:
            print(f"   pip install {dep}")
        sys.exit(1)
    
    print("✅ All dependencies are available!")


def setup_global_data():
    """
    Load and partition the MNIST dataset for all clients.
    Sets up global variables needed for the simulation.
    """
    global CLIENT_DATALOADERS, TEST_DATALOADER, EXPERIMENT_CONFIG
    
    print("\n🔄 Setting up MNIST dataset with non-IID distribution...")
    
    # Create data partitioner
    partitioner = MNISTDataPartitioner(
        num_clients=EXPERIMENT_CONFIG['num_clients'],
        batch_size=EXPERIMENT_CONFIG['batch_size'],
        non_iid_alpha=EXPERIMENT_CONFIG['non_iid_alpha']
    )
    
    # Load and partition the data
    partitioner.load_data()
    client_datasets = partitioner.partition_data()
    
    # Create data loaders for each client
    CLIENT_DATALOADERS = {}
    for client_id in range(EXPERIMENT_CONFIG['num_clients']):
        train_loader, val_loader = partitioner.create_dataloaders(client_id)
        CLIENT_DATALOADERS[client_id] = {
            'train': train_loader,
            'val': val_loader
        }
    
    # Create test dataloader
    TEST_DATALOADER = partitioner.get_test_dataloader()
    
    # Log data distribution
    partitioner.log_data_distribution()
    
    print(f"✅ Data setup complete!")
    print(f"   - {EXPERIMENT_CONFIG['num_clients']} clients created")
    print(f"   - Training samples per client: ~{60000 // EXPERIMENT_CONFIG['num_clients']}")
    print(f"   - Test samples: {len(TEST_DATALOADER.dataset)}")


def client_fn(context: Context) -> MNISTFlowerClient:
    """
    Create a FlowerClient instance for the given client ID.
    
    Args:
        context: Flower context object containing client configuration
        
    Returns:
        MNISTFlowerClient: Configured client instance
    """
    client_id = int(context.node_config.get("partition-id", context.node_id))
    
    # # Create model for this client
    # model = create_model()
    
    # Get data loaders for this client
    train_loader = CLIENT_DATALOADERS[client_id]['train']
    val_loader = CLIENT_DATALOADERS[client_id]['val']
    
    # Create and return client
    return create_client(
        client_id=client_id,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=EXPERIMENT_CONFIG['local_epochs'],
        learning_rate=EXPERIMENT_CONFIG['learning_rate'],
        device=EXPERIMENT_CONFIG['device']
    )


def run_simulation():
    """
    Run the federated learning simulation using Flower.
    This orchestrates the entire federated learning process.
    """
    global EXPERIMENT_CONFIG
    
    print(f"\n🚀 Starting federated learning simulation...")
    print(f"   - Strategy: FedAvg")
    print(f"   - Clients: {EXPERIMENT_CONFIG['num_clients']}")
    print(f"   - Rounds: {EXPERIMENT_CONFIG['num_rounds']}")
    print(f"   - Epochs per round: {EXPERIMENT_CONFIG['local_epochs']}")
    
    # Create server strategy
    strategy = create_server_strategy(
        test_dataloader=TEST_DATALOADER,
        config=EXPERIMENT_CONFIG
    )
    
    # Configure simulation
    simulation_config = {
        "num_clients": EXPERIMENT_CONFIG['num_clients'],
        "clients_per_round": EXPERIMENT_CONFIG['num_clients'],  # All clients participate
        "client_resources": {
            "num_cpus": 1,
            "num_gpus": 0.0 if not torch.cuda.is_available() else 0.1
        }
    }
    
    print(f"   - Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"   - Client resources: {simulation_config['client_resources']}")
    
    try:
        # Start simulation with early stopping
        history = run_simulation_with_early_stopping(
            client_fn=client_fn,
            num_clients=EXPERIMENT_CONFIG['num_clients'],
            strategy=strategy,
            client_resources=simulation_config['client_resources'],
            max_rounds=EXPERIMENT_CONFIG['num_rounds']
        )
        
        print("\n🎉 Simulation completed successfully!")
        return history
        
    except Exception as e:
        print(f"\n❌ Simulation failed with error: {str(e)}")
        print("   Check the logs above for more details.")
        sys.exit(1)


def print_final_results(history):
    """
    Print final results and summary of the federated learning experiment.
    
    Args:
        history: Simulation history from Flower
    """
    print("\n" + "="*60)
    print("FEDERATED LEARNING EXPERIMENT RESULTS")
    print("="*60)
    
    # if hasattr(history, 'losses_centralized') and history.losses_centralized:
    #     print("\n📊 Global Model Performance:")
    #     for round_num, (loss, accuracy) in enumerate(history.losses_centralized, 1):
    #         print(f"   Round {round_num}: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")
    
    # if hasattr(history, 'losses_distributed') and history.losses_distributed:
    #     print("\n📈 Training Progress:")
    #     for round_num, loss in enumerate(history.losses_distributed, 1):
    #         print(f"   Round {round_num}: Distributed Loss = {loss:.4f}")
    
    # # Print experiment configuration
    # print(f"\n⚙️  Experiment Configuration:")
    # print(f"   - Number of clients: {EXPERIMENT_CONFIG['num_clients']}")
    # print(f"   - Number of rounds: {EXPERIMENT_CONFIG['num_rounds']}")
    # print(f"   - Local epochs: {EXPERIMENT_CONFIG['local_epochs']}")
    # print(f"   - Batch size: {EXPERIMENT_CONFIG['batch_size']}")
    # print(f"   - Learning rate: {EXPERIMENT_CONFIG['learning_rate']}")
    # print(f"   - Non-IID alpha: {EXPERIMENT_CONFIG['non_iid_alpha']}")
    
    # print(f"\n💾 Results saved to: {EXPERIMENT_CONFIG['log_dir']}")
    # print("="*60)

    if hasattr(history, 'metrics_centralized') and history.metrics_centralized.get('accuracy'):
        print("\n📊 Global Model Performance:")
        for round_num, metrics in history.metrics_centralized['accuracy']:
            loss = next(l for r, l in history.losses_centralized if r == round_num)
            print(f"   Round {round_num}: Loss = {loss:.4f}, Accuracy = {metrics:.2f}%")

    if hasattr(history, 'losses_distributed') and history.losses_distributed:
        print("\n📈 Training Progress:")
        for round_num, loss in history.losses_distributed:
            print(f"   Round {round_num}: Distributed Loss = {loss:.4f}")

    print(f"\n⚙️  Experiment Configuration:")
    print(f"   - Number of clients: {EXPERIMENT_CONFIG['num_clients']}")
    print(f"   - Number of rounds: {EXPERIMENT_CONFIG['num_rounds']}")
    print(f"   - Local epochs: {EXPERIMENT_CONFIG['local_epochs']}")
    print(f"   - Batch size: {EXPERIMENT_CONFIG['batch_size']}")
    print(f"   - Learning rate: {EXPERIMENT_CONFIG['learning_rate']}")
    print(f"   - Non-IID alpha: {EXPERIMENT_CONFIG['non_iid_alpha']}")
    
    print(f"\n💾 Results saved to: {EXPERIMENT_CONFIG['log_dir']}")
    print("="*60)

def main():
    """
    Main function that orchestrates the entire federated learning experiment.
    This function:
    1. Checks dependencies
    2. Sets up logging and configuration
    3. Loads and partitions data
    4. Runs the simulation
    5. Saves and displays results
    """
    global EXPERIMENT_CONFIG
    
    print("🔬 FEDERATED LEARNING WITH NON-IID DATA")
    print("=" * 50)
    
    # Step 1: Check dependencies
    check_dependencies()
    
    # Step 2: Check CUDA availability
    check_cuda_availability()
    
    # Step 3: Create experiment configuration
    EXPERIMENT_CONFIG = create_experiment_config()
    
    # Step 4: Validate configuration
    validate_config(EXPERIMENT_CONFIG)
    
    # Step 5: Setup logging
    setup_logging(EXPERIMENT_CONFIG['log_dir'])
    
    # Step 6: Setup data
    setup_global_data()
    
    # Step 7: Run simulation
    print("\n" + "-" * 50)
    history = run_simulation()
    
    # Step 8: Save results
    save_experiment_results(history, EXPERIMENT_CONFIG)
    
    # Step 9: Print final results
    print_final_results(history)


if __name__ == "__main__":
    """
    Entry point for the federated learning simulation.
    
    Usage:
        python main.py
    
    This will:
    - Load MNIST dataset
    - Create 3 clients with non-IID data distribution
    - Run federated learning for 3 rounds
    - Print results and save logs
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Simulation interrupted by user (Ctrl+C)")
        print("   Partial results may be available in the log directory.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("   Please check your installation and try again.")
        sys.exit(1)