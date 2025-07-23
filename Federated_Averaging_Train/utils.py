# utils.py
# Purpose: Utility functions for federated learning implementation
# Includes parameter conversion, logging, evaluation, and helper functions

import torch
import numpy as np
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
import json

def setup_logging(log_dir: str = None, log_level=logging.INFO):
    """
    Set up logging configuration for the federated learning experiment.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup file handler if log directory is specified
    handlers = [console_handler]
    if log_dir:
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'experiment.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # Override existing configuration
    )
    
    return logging.getLogger(__name__)

def weights_to_parameters(weights: List[torch.Tensor]) -> List[np.ndarray]:
    """
    Convert PyTorch model weights to NumPy arrays for Flower.
    
    Args:
        weights: List of PyTorch tensors (model parameters)
        
    Returns:
        List of NumPy arrays
    """
    return [weight.detach().cpu().numpy() for weight in weights]

def parameters_to_weights(parameters: List[np.ndarray]) -> List[torch.Tensor]:
    """
    Convert NumPy arrays back to PyTorch tensors.
    
    Args:
        parameters: List of NumPy arrays
        
    Returns:
        List of PyTorch tensors
    """
    return [torch.tensor(param) for param in parameters]

def set_model_parameters(model: torch.nn.Module, parameters: List[np.ndarray]):
    """
    Set model parameters from a list of NumPy arrays.
    
    Args:
        model: PyTorch model
        parameters: List of NumPy arrays containing model parameters
    """
    params_dict = zip(model.parameters(), parameters)
    for model_param, new_param in params_dict:
        model_param.data.copy_(torch.tensor(new_param))

def get_model_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """
    Get model parameters as a list of NumPy arrays.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of NumPy arrays containing model parameters
    """
    return [param.detach().cpu().numpy() for param in model.parameters()]

def train_model(model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader,
                epochs: int = 5, 
                learning_rate: float = 0.001,
                device: str = "cpu") -> Dict[str, List[float]]:
    """
    Train a PyTorch model on given data.
    
    Args:
        model: PyTorch model to train
        dataloader: DataLoader containing training data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ("cpu" or "cuda")
        
    Returns:
        Dictionary containing training metrics (loss and accuracy per epoch)
    """
    model.to(device)
    model.train()
    
    # Setup optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Track metrics
    metrics = {
        'loss': [],
        'accuracy': []
    }
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        metrics['loss'].append(avg_loss)
        metrics['accuracy'].append(accuracy)
        
        print(f"  Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
    
    return metrics

def evaluate_model(model: torch.nn.Module, 
                   dataloader: torch.utils.data.DataLoader,
                   device: str = "cpu") -> Tuple[float, float]:
    """
    Evaluate a PyTorch model on given data.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader containing evaluation data
        device: Device to evaluate on ("cpu" or "cuda")
        
    Returns:
        Tuple of (loss, accuracy)
    """
    model.to(device)
    model.eval()
    
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

def save_experiment_results(history, config: Dict[str, Any]):
    """
    Save experiment results to a JSON file.
    
    Args:
        history: Flower simulation history
        config: Experiment configuration dictionary
    """
    # Create results dictionary
    results = {
        'config': config,
        'history': str(history),  # Convert to string as history might not be JSON serializable
        'timestamp': datetime.now().isoformat()
    }
    
    # Create results filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"federated_learning_results_{timestamp}.json"
    
    # Ensure log directory exists (use config['log_dir'])
    log_dir = config.get('log_dir', 'results')
    os.makedirs(log_dir, exist_ok=True)
    filepath = os.path.join(log_dir, filename)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2, default=str)
    
    print(f"Experiment results saved to {filepath}")
    return filepath

def load_experiment_results(filepath: str) -> Dict[str, Any]:
    """
    Load experiment results from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary containing experiment results
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results

def print_round_summary(round_num: int, 
                       client_metrics: Dict[int, Dict[str, List[float]]],
                       global_accuracy: float,
                       global_loss: float = None):
    """
    Print a summary of the federated learning round.
    
    Args:
        round_num: Round number
        client_metrics: Dictionary of client training metrics
        global_accuracy: Global model accuracy on test set
        global_loss: Optional global model loss on test set
    """
    print(f"\n{'='*60}")
    print(f"ROUND {round_num} SUMMARY")
    print(f"{'='*60}")
    
    # Print client metrics
    for client_id, metrics in client_metrics.items():
        final_loss = metrics['loss'][-1]
        final_accuracy = metrics['accuracy'][-1]
        print(f"Client {client_id}: Final Loss = {final_loss:.4f}, Final Accuracy = {final_accuracy:.2f}%")
    
    # Print global metrics
    print(f"\nGlobal Test Accuracy: {global_accuracy:.2f}%")
    if global_loss is not None:
        print(f"Global Test Loss: {global_loss:.4f}")
    
    print(f"{'='*60}")

def check_cuda_availability():
    """
    Check if CUDA is available and return appropriate device.
    
    Returns:
        String indicating device to use ("cuda" or "cpu")
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA is not available. Using CPU.")
    
    return device

def create_experiment_config(num_clients: int = 3,
                           num_rounds: int = 20,
                           local_epochs: int = 50,
                           batch_size: int = 32,
                           learning_rate: float = 0.001,
                           non_iid_alpha: float = 0.1,
                           **kwargs) -> Dict[str, Any]:
    """
    Create a configuration dictionary for the federated learning experiment.
    
    Args:
        num_clients: Number of clients
        num_rounds: Number of federated learning rounds
        local_epochs: Number of local training epochs per round
        batch_size: Batch size for training
        learning_rate: Learning rate for optimization
        non_iid_alpha: Alpha parameter for non-IID data distribution
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary containing experiment configuration
    """
    # Create timestamp for unique experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config = {
        'num_clients': num_clients,
        'num_rounds': num_rounds,
        'local_epochs': local_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'non_iid_alpha': non_iid_alpha,
        'early_stopping_patience': 5,  # Stop if no improvement for 5 rounds
        'early_stopping_min_delta': 0.001,  # Minimum improvement threshold
        'log_dir': f"logs/experiment_{timestamp}",
        'save_model': True,
        'model_save_path': f"models/model_{timestamp}.pth",
        'timestamp': datetime.now().isoformat(),
        'device': check_cuda_availability()
    }
    
    # Create directories if they don't exist
    os.makedirs(config['log_dir'], exist_ok=True)
    if config['save_model']:
        os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
    
    # Add any additional parameters
    config.update(kwargs)
    
    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate the experiment configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Boolean indicating if configuration is valid
    """
    required_keys = ['num_clients', 'num_rounds', 'local_epochs', 'batch_size', 'learning_rate', 'non_iid_alpha', 'log_dir']
    
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required configuration key: {key}")
            return False
    
    # Validate values
    if config['num_clients'] <= 0:
        print("Error: num_clients must be positive")
        return False
    
    if config['num_rounds'] <= 0:
        print("Error: num_rounds must be positive")
        return False
    
    if config['local_epochs'] <= 0:
        print("Error: local_epochs must be positive")
        return False
    
    if config['batch_size'] <= 0:
        print("Error: batch_size must be positive")
        return False
    
    if config['learning_rate'] <= 0:
        print("Error: learning_rate must be positive")
        return False
    
    if config['non_iid_alpha'] <= 0:
        print("Error: non_iid_alpha must be positive")
        return False
    
    return True

def calculate_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """
    Calculate the size of the model in terms of parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate size in bytes (assuming float32)
    size_bytes = total_params * 4  # 4 bytes per float32
    size_mb = size_bytes / (1024 * 1024)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'size_bytes': size_bytes,
        'size_mb': size_mb
    }

# Example usage and testing
if __name__ == "__main__":
    # Test logging setup
    logger = setup_logging(log_level=logging.INFO, log_file="test.log")
    logger.info("Testing logging functionality")
    
    # Test configuration creation and validation
    config = create_experiment_config(
        num_clients=5,
        num_rounds=10,
        local_epochs=50,
        batch_size=32,
        learning_rate=0.001
    )
    
    print("Configuration created:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Validate configuration
    is_valid = validate_config(config)
    print(f"\nConfiguration is valid: {is_valid}")
    
    # Test device availability
    device = check_cuda_availability()
    print(f"Selected device: {device}")
    
    # Test model size calculation
    from model import create_model
    model = create_model()
    model_size = calculate_model_size(model)
    print(f"\nModel size information:")
    for key, value in model_size.items():
        print(f"  {key}: {value}")
    
    print("Utils module testing completed!")