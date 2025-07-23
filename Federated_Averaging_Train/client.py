# client.py
# Purpose: Implements the FlowerClient class for federated learning
# This handles local training, parameter sharing, and client-side evaluation

import torch
import flwr as fl
from typing import Dict, List, Tuple, Optional
import numpy as np
from model import create_model
from utils import (
    get_model_parameters, 
    set_model_parameters, 
    train_model, 
    evaluate_model,
    check_cuda_availability
)

class MNISTFlowerClient(fl.client.NumPyClient):
    """
    Flower client for federated learning with MNIST.
    
    This client:
    1. Maintains a local copy of the global model
    2. Receives updated parameters from the server
    3. Trains the model locally on its non-IID data
    4. Sends updated parameters back to the server
    5. Evaluates the model on local data
    """
    
    def __init__(self, 
                 client_id: int, 
                 train_dataloader: torch.utils.data.DataLoader,
                 val_dataloader: torch.utils.data.DataLoader = None,
                 epochs: int = 5,
                 learning_rate: float = 0.01,
                 device: str = None):
        """
        Initialize the Flower client.
        
        Args:
            client_id: Unique identifier for this client
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            epochs: Number of local training epochs per round
            learning_rate: Learning rate for local training
            device: Device to use ("cpu" or "cuda"). If None, auto-detect
        """
        super().__init__()
        
        self.client_id = client_id
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        # Set device
        if device is None:
            self.device = check_cuda_availability()
        else:
            self.device = device
        
        # Initialize model
        self.model = create_model()
        self.model.to(self.device)
        
        # Track training history
        self.training_history = []
        
        print(f"Client {self.client_id} initialized with device: {self.device}")
        print(f"  Training samples: {len(train_dataloader.dataset)}")
        if val_dataloader:
            print(f"  Validation samples: {len(val_dataloader.dataset)}")
    
    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        """
        Get current model parameters.
        
        This method is called by the Flower framework to get the current
        model parameters from this client. These parameters will be sent
        to the server for aggregation.
        
        Args:
            config: Configuration dictionary from server
            
        Returns:
            List of model parameters as NumPy arrays
        """
        print(f"Client {self.client_id}: Getting parameters")
        return get_model_parameters(self.model)
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters.
        
        This method is called by the Flower framework to set new model
        parameters received from the server (after aggregation).
        
        Args:
            parameters: List of model parameters as NumPy arrays
        """
        print(f"Client {self.client_id}: Setting parameters")
        set_model_parameters(self.model, parameters)
    
    def fit(self, 
            parameters: List[np.ndarray], 
            config: Dict[str, fl.common.Scalar]) -> Tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:
        """
        Train the model locally.
        
        This is the core method where local training happens. The client:
        1. Receives global model parameters from the server
        2. Updates its local model with these parameters
        3. Trains the model on its local data
        4. Returns updated parameters to the server
        
        Args:
            parameters: Model parameters from server
            config: Configuration dictionary from server
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        print(f"\nClient {self.client_id}: Starting local training")
        
        # Check for valid dataloader
        if self.train_dataloader is None:
            raise ValueError("fClient {self.client_id}: train_dataloader is None. Cannot train.")
        
        # Set model parameters from server
        self.set_parameters(parameters)
        
        # Extract configuration
        epochs = config.get("epochs", self.epochs)
        learning_rate = config.get("learning_rate", self.learning_rate)
        
        print(f"Client {self.client_id}: Training for {epochs} epochs with LR={learning_rate}")
        
        # Train the model
        try:
            training_metrics = train_model(
                model=self.model,
                dataloader=self.train_dataloader,
                epochs=epochs,
                learning_rate=learning_rate,
                device=self.device
            )
            
            # Store training history
            self.training_history.append(training_metrics)
            
            # Get updated parameters
            updated_parameters = self.get_parameters({})
            
            # Prepare metrics to send back to server
            metrics = {
                "train_loss": training_metrics['loss'][-1],
                "train_accuracy": training_metrics['accuracy'][-1],
                "client_id": self.client_id
            }
            
            num_examples = len(self.train_dataloader.dataset)
            
            print(f"Client {self.client_id}: Training completed")
            print(f"  Final loss: {metrics['train_loss']:.4f}")
            print(f"  Final accuracy: {metrics['train_accuracy']:.2f}%")
            
            return updated_parameters, num_examples, metrics
            
        except Exception as e:
            print(f"Client {self.client_id}: Training failed with error: {e}")
            # Return current parameters if training fails
            return self.get_parameters({}), len(self.train_dataloader.dataset), {"error": str(e)}
    
    def evaluate(self, 
                 parameters: List[np.ndarray], 
                 config: Dict[str, fl.common.Scalar]) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:
        """
        Evaluate the model locally.
        
        This method evaluates the model on the client's local data.
        It can be used for local validation or to contribute to
        distributed evaluation.
        
        Args:
            parameters: Model parameters from server
            config: Configuration dictionary from server
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        print(f"Client {self.client_id}: Starting local evaluation")
        
        # Set model parameters from server
        self.set_parameters(parameters)
        
        # Choose evaluation dataset
        if self.val_dataloader is not None:
            eval_dataloader = self.val_dataloader
            dataset_type = "validation"
        else:
            # Use a small portion of training data for evaluation
            eval_dataloader = self.train_dataloader
            dataset_type = "training"
        
        try:
            # Evaluate the model
            loss, accuracy = evaluate_model(
                model=self.model,
                dataloader=eval_dataloader,
                device=self.device
            )
            
            # Prepare metrics
            metrics = {
                "accuracy": accuracy,
                "client_id": self.client_id,
                "dataset_type": dataset_type
            }
            
            num_examples = len(eval_dataloader.dataset)
            
            print(f"Client {self.client_id}: Evaluation completed")
            print(f"  Loss: {loss:.4f}")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Dataset: {dataset_type} ({num_examples} samples)")
            
            return loss, num_examples, metrics
            
        except Exception as e:
            print(f"Client {self.client_id}: Evaluation failed with error: {e}")
            return float('inf'), len(eval_dataloader.dataset), {"error": str(e)}
    
    def get_training_history(self) -> List[Dict[str, List[float]]]:
        """
        Get the training history for this client.
        
        Returns:
            List of training metrics for each round
        """
        return self.training_history
    
    def get_client_info(self) -> Dict[str, any]:
        """
        Get information about this client.
        
        Returns:
            Dictionary containing client information
        """
        return {
            "client_id": self.client_id,
            "device": self.device,
            "training_samples": len(self.train_dataloader.dataset),
            "validation_samples": len(self.val_dataloader.dataset) if self.val_dataloader else 0,
            "epochs_per_round": self.epochs,
            "learning_rate": self.learning_rate,
            "rounds_completed": len(self.training_history)
        }

def create_client(client_id: int, 
                  train_dataloader: torch.utils.data.DataLoader,
                  val_dataloader: torch.utils.data.DataLoader = None,
                  epochs: int = 5,
                  learning_rate: float = 0.01,
                  device: str = None) -> MNISTFlowerClient:
    """
    Factory function to create a new Flower client.
    
    Args:
        client_id: Unique identifier for the client
        train_dataloader: DataLoader for training data
        val_dataloader: Optional DataLoader for validation data
        epochs: Number of local training epochs per round
        learning_rate: Learning rate for local training
        device: Device to use ("cpu" or "cuda")
        
    Returns:
        MNISTFlowerClient instance
    """
    return MNISTFlowerClient(
        client_id=client_id,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device
    )

# Client function for Flower simulation
# def client_fn(cid: str) -> MNISTFlowerClient:
#     """
#     Function to create a client instance for Flower simulation.
    
#     This function is used by Flower's simulation framework to create
#     client instances. It needs to be customized with actual data.
    
#     Args:
#         cid: Client ID as string
        
#     Returns:
#         MNISTFlowerClient instance
#     """
#     # This is a template - actual implementation should load real data
#     # based on the client ID
    
#     # Convert client ID to integer
#     client_id = int(cid)
    
#     # This would normally load client-specific data
#     # For now, we'll create a placeholder
    
#     print(f"Creating client {client_id} for simulation")
    
#     # Return a placeholder client
#     # In real implementation, this would load actual data partitions
#     return MNISTFlowerClient(
#         client_id=client_id,
#         train_dataloader=None,  # This should be loaded from data partitioner
#         val_dataloader=None,
#         epochs=5,
#         learning_rate=0.01
#     )



# Example usage and testing
if __name__ == "__main__":
    # Test client creation and functionality
    print("Testing MNISTFlowerClient...")
    
    # Create dummy data for testing
    from torch.utils.data import DataLoader, TensorDataset
    import torch
    
    # Create dummy training data
    dummy_train_data = torch.randn(1000, 1, 28, 28)
    dummy_train_targets = torch.randint(0, 10, (1000,))
    train_dataset = TensorDataset(dummy_train_data, dummy_train_targets)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Create dummy validation data
    dummy_val_data = torch.randn(200, 1, 28, 28)
    dummy_val_targets = torch.randint(0, 10, (200,))
    val_dataset = TensorDataset(dummy_val_data, dummy_val_targets)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create client
    client = create_client(
        client_id=0,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=2,
        learning_rate=0.01
    )
    
    # Test get_parameters
    params = client.get_parameters({})
    print(f"Number of parameter arrays: {len(params)}")
    
    # Test fit (training)
    config = {"epochs": 2, "learning_rate": 0.01}
    updated_params, num_examples, metrics = client.fit(params, config)
    print(f"Training completed. Metrics: {metrics}")
    
    # Test evaluate
    loss, num_examples, eval_metrics = client.evaluate(updated_params, {})
    print(f"Evaluation completed. Loss: {loss:.4f}, Metrics: {eval_metrics}")
    
    # Test client info
    info = client.get_client_info()
    print(f"Client info: {info}")
    
    print("Client testing completed!")