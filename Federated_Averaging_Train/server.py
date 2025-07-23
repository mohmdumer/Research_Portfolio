# server.py
# Purpose: Implements the federated learning server using Flower's FedAvg strategy
# Handles model aggregation, global evaluation, and coordination of federated training

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, Scalar
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import numpy as np
from model import create_model
from utils import (
    get_model_parameters, 
    set_model_parameters, 
    evaluate_model,
    check_cuda_availability,
    print_round_summary,
    save_experiment_results
)

class MNISTFedAvgStrategy(FedAvg):
    """
    Custom FedAvg strategy for MNIST federated learning.
    
    This strategy:
    1. Initializes the global model
    2. Aggregates client parameters using FedAvg
    3. Evaluates the global model after each round
    4. Logs training progress and metrics
    """
    
    def __init__(self, 
                 test_dataloader: torch.utils.data.DataLoader,
                 initial_parameters: Optional[Parameters] = None,
                 device: str = None,
                 config: Dict[str, Any] = None,
                 **kwargs):
        """
        Initialize the FedAvg strategy.
        
        Args:
            test_dataloader: DataLoader for global model evaluation
            initial_parameters: Initial model parameters
            device: Device to use for evaluation ("cpu" or "cuda")
            config: Experiment configuration dictionary
            **kwargs: Additional arguments for FedAvg
        """
        super().__init__(
            initial_parameters=initial_parameters,
            **kwargs
        )
        
        self.test_dataloader = test_dataloader
        self.device = device if device else check_cuda_availability()
        self.config = config or {}
        
        # Early stopping parameters
        self.early_stopping_patience = config.get('early_stopping_patience', 5)
        self.early_stopping_min_delta = config.get('early_stopping_min_delta', 0.001)
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False
        
        # Initialize global model for evaluation
        self.global_model = create_model()
        self.global_model.to(self.device)
        
        # Track global metrics
        self.global_metrics_history = []
        self.round_count = 0
        
        print(f"Server strategy initialized with device: {self.device}")
        print(f"Test dataset size: {len(test_dataloader.dataset)}")
        print(f"Early stopping: patience={self.early_stopping_patience}, min_delta={self.early_stopping_min_delta}")
        print(f"Configuration: {self.config}")
    
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """
        Initialize global model parameters.
        
        This method is called at the start of federated learning to
        initialize the global model parameters.
        
        Args:
            client_manager: Flower client manager
            
        Returns:
            Initial parameters for the global model
        """
        print("Initializing global model parameters...")
        
        # Get initial parameters from the global model
        initial_params = get_model_parameters(self.global_model)
        
        # Convert to Flower Parameters format
        parameters = fl.common.ndarrays_to_parameters(initial_params)
        
        print(f"Global model initialized with {len(initial_params)} parameter arrays")
        return parameters
    
    def aggregate_fit(self, 
                     server_round: int,
                     results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                     failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model parameters from clients after training.
        
        This method is called after each round of client training to
        aggregate the updated parameters using FedAvg.
        
        Args:
            server_round: Current round number
            results: List of (client, fit_result) tuples
            failures: List of failed clients
            
        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        print(f"\nServer Round {server_round}: Aggregating parameters from {len(results)} clients")
        
        # Handle failures
        if failures:
            print(f"Warning: {len(failures)} clients failed during training")
            for failure in failures:
                print(f"  Failed client: {failure}")
        
        # Extract client metrics for logging
        client_metrics = {}
        for client, fit_res in results:
            client_id = fit_res.metrics.get("client_id", "unknown")
            client_metrics[client_id] = {
                "train_loss": fit_res.metrics.get("train_loss", 0.0),
                "train_accuracy": fit_res.metrics.get("train_accuracy", 0.0),
                "num_examples": fit_res.num_examples
            }
        
        # Call parent class aggregation (FedAvg)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Log client training results
        print(f"Client training results for round {server_round}:")
        for client_id, metrics in client_metrics.items():
            print(f"  Client {client_id}: Loss={metrics['train_loss']:.4f}, "
                  f"Accuracy={metrics['train_accuracy']:.2f}%, "
                  f"Samples={metrics['num_examples']}")
        
        return aggregated_parameters, aggregated_metrics
    
    def evaluate(self, 
                server_round: int,
                parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Evaluate the global model on the test dataset.
        
        This method is called after parameter aggregation to evaluate
        the performance of the global model.
        
        Args:
            server_round: Current round number
            parameters: Aggregated global model parameters
            
        Returns:
            Tuple of (loss, metrics) or None if evaluation fails
        """
        print(f"\nEvaluating global model for round {server_round}...")
        
        try:
            # Convert parameters back to numpy arrays
            params_list = fl.common.parameters_to_ndarrays(parameters)
            
            # Set global model parameters
            set_model_parameters(self.global_model, params_list)
            
            # Evaluate the global model
            global_loss, global_accuracy = evaluate_model(
                model=self.global_model,
                dataloader=self.test_dataloader,
                device=self.device
            )
            
            # Store metrics
            round_metrics = {
                "round": server_round,
                "global_loss": global_loss,
                "global_accuracy": global_accuracy,
                "test_samples": len(self.test_dataloader.dataset)
            }
            
            self.global_metrics_history.append(round_metrics)
            self.round_count = server_round
            
            # Print results
            print(f"Round {server_round} - Global Test Accuracy: {global_accuracy:.2f}%")
            print(f"Round {server_round} - Global Test Loss: {global_loss:.4f}")
            
            # Early stopping check
            self._check_early_stopping(global_loss)
            
            # Return metrics for Flower
            metrics = {
                "accuracy": global_accuracy,
                "loss": global_loss,
                "round": server_round,
                "early_stop": self.should_stop
            }
            
            return global_loss, metrics
            
        except Exception as e:
            print(f"Error during global evaluation: {e}")
            return None
    
    def aggregate_evaluate(self, 
                          server_round: int,
                          results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
                          failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from clients.
        
        This method aggregates evaluation metrics from clients.
        In this implementation, we primarily use server-side evaluation.
        
        Args:
            server_round: Current round number
            results: List of (client, evaluate_result) tuples
            failures: List of failed evaluations
            
        Returns:
            Tuple of (aggregated_loss, metrics)
        """
        print(f"Aggregating evaluation results from {len(results)} clients")
        
        # Handle failures
        if failures:
            print(f"Warning: {len(failures)} clients failed during evaluation")
        
        # Call parent class aggregation
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Log client evaluation results
        if results:
            print(f"Client evaluation results for round {server_round}:")
            for client, eval_res in results:
                client_id = eval_res.metrics.get("client_id", "unknown")
                accuracy = eval_res.metrics.get("accuracy", 0.0)
                print(f"  Client {client_id}: Loss={eval_res.loss:.4f}, "
                      f"Accuracy={accuracy:.2f}%, Samples={eval_res.num_examples}")
        
        return aggregated_loss, aggregated_metrics
    
    def _check_early_stopping(self, current_loss: float) -> None:
        """
        Check if early stopping criteria are met.
        
        Args:
            current_loss: Current global loss value
        """
        if current_loss < self.best_loss - self.early_stopping_min_delta:
            # Loss improved significantly
            self.best_loss = current_loss
            self.patience_counter = 0
            print(f"📈 Loss improved to {current_loss:.4f} (best so far)")
        else:
            # Loss did not improve
            self.patience_counter += 1
            print(f"⏳ Loss did not improve. Patience: {self.patience_counter}/{self.early_stopping_patience}")
            
            if self.patience_counter >= self.early_stopping_patience:
                self.should_stop = True
                print(f"🛑 Early stopping triggered! No improvement for {self.early_stopping_patience} rounds.")
                print(f"   Best loss was: {self.best_loss:.4f}")
    
    def should_stop_early(self) -> bool:
        """
        Check if training should stop early.
        
        Returns:
            True if early stopping criteria are met
        """
        return self.should_stop
    
    def get_global_metrics_history(self) -> List[Dict[str, float]]:
        """
        Get the history of global model metrics.
        
        Returns:
            List of dictionaries containing global metrics for each round
        """
        return self.global_metrics_history
    
    def save_results(self, filepath: str = None):
        """
        Save the federated learning results.
        
        Args:
            filepath: Optional path to save results
        """
        results = {
            "experiment_type": "federated_learning_mnist_non_iid",
            "total_rounds": self.round_count,
            "global_metrics_history": self.global_metrics_history,
            "final_accuracy": self.global_metrics_history[-1]["global_accuracy"] if self.global_metrics_history else 0.0,
            "device": self.device
        }
        
        save_experiment_results(results, filepath)

def run_simulation_with_early_stopping(client_fn, num_clients, strategy, client_resources, max_rounds=20):
    """
    Run simulation with early stopping support.
    
    Args:
        client_fn: Client function
        num_clients: Number of clients
        strategy: Server strategy with early stopping
        client_resources: Client resources configuration
        max_rounds: Maximum number of rounds
        
    Returns:
        Simulation history
    """
    from flwr.simulation import start_simulation
    import flwr as fl
    
    print(f"🚀 Starting simulation with early stopping (max {max_rounds} rounds)...")
    print(f"   Early stopping patience: {strategy.early_stopping_patience}")
    print(f"   Early stopping min delta: {strategy.early_stopping_min_delta}")
    
    # Create a wrapper strategy that can terminate early
    class EarlyStoppingStrategy:
        def __init__(self, base_strategy):
            self.base_strategy = base_strategy
            self.current_round = 0
            
        def __getattr__(self, name):
            # Delegate all other methods to base strategy
            return getattr(self.base_strategy, name)
            
        def evaluate(self, server_round, parameters):
            # Call base evaluation
            result = self.base_strategy.evaluate(server_round, parameters)
            self.current_round = server_round
            
            # Check if we should stop early
            if hasattr(self.base_strategy, 'should_stop_early') and self.base_strategy.should_stop_early():
                print(f"\n🛑 EARLY STOPPING TRIGGERED at round {server_round}!")
                print(f"   No improvement for {self.base_strategy.early_stopping_patience} consecutive rounds")
                print(f"   Best loss achieved: {self.base_strategy.best_loss:.4f}")
                
                # Unfortunately, we can't actually stop Flower's simulation mid-flight
                # But we can log this information for analysis
                
            return result
    
    # Wrap the strategy
    wrapped_strategy = EarlyStoppingStrategy(strategy)
    
    try:
        history = start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=max_rounds),
            strategy=wrapped_strategy,
            client_resources=client_resources
        )
        
        # Post-simulation analysis
        if hasattr(strategy, 'should_stop_early') and strategy.should_stop_early():
            # Find when early stopping would have been optimal
            effective_rounds = 0
            for i, metrics in enumerate(strategy.global_metrics_history):
                if strategy.patience_counter >= strategy.early_stopping_patience:
                    effective_rounds = max(1, i - strategy.early_stopping_patience + 1)
                    break
            
            if effective_rounds > 0:
                print(f"\n📊 EARLY STOPPING ANALYSIS:")
                print(f"   Optimal stopping point: Round {effective_rounds}")
                print(f"   Rounds saved: {max_rounds - effective_rounds}")
                print(f"   Training efficiency: {(effective_rounds/max_rounds)*100:.1f}%")
        
        return history
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        raise

def create_server_strategy(test_dataloader: torch.utils.data.DataLoader,
                          config: Dict[str, Any],
                          device: str = None) -> MNISTFedAvgStrategy:
    """
    Create a configured FedAvg strategy for the server.
    
    Args:
        test_dataloader: DataLoader for global model evaluation
        config: Experiment configuration dictionary
        device: Device to use for evaluation (overrides config if specified)
        
    Returns:
        Configured MNISTFedAvgStrategy
    """
    # Extract values from config
    num_clients = config.get('num_clients', 3)
    min_fit_clients = num_clients  # All clients participate
    min_evaluate_clients = 0  # No client-side evaluation
    min_available_clients = num_clients
    fraction_fit = 1.0  # Use all available clients
    fraction_evaluate = 0.0  # No distributed evaluation
    
    # Use device from config if not specified
    if device is None:
        device = config.get('device', 'cpu')
    
    strategy = MNISTFedAvgStrategy(
        test_dataloader=test_dataloader,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        device=device,
        config=config  # Pass config to strategy
    )
    
    print(f"Server strategy created:")
    print(f"  Total clients: {num_clients}")
    print(f"  Min fit clients: {min_fit_clients}")
    print(f"  Min evaluate clients: {min_evaluate_clients}")
    print(f"  Fraction fit: {fraction_fit}")
    print(f"  Fraction evaluate: {fraction_evaluate}")
    
    return strategy

def run_server(strategy: MNISTFedAvgStrategy,
               num_rounds: int = 3,
               config: Dict[str, any] = None) -> None:
    """
    Run the federated learning server.
    
    Args:
        strategy: Configured FedAvg strategy
        num_rounds: Number of federated learning rounds
        config: Additional configuration for the server
    """
    print(f"\nStarting federated learning server for {num_rounds} rounds...")
    
    # Default server configuration
    if config is None:
        config = {}
    
    # Configure server
    server_config = fl.server.ServerConfig(num_rounds=num_rounds)
    
    # Start server
    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=server_config,
            strategy=strategy
        )
        
        print(f"Federated learning completed after {num_rounds} rounds")
        
        # Print final summary
        metrics_history = strategy.get_global_metrics_history()
        if metrics_history:
            final_accuracy = metrics_history[-1]["global_accuracy"]
            print(f"\nFinal global test accuracy: {final_accuracy:.2f}%")
            
            # Print accuracy progression
            print("\nAccuracy progression:")
            for metrics in metrics_history:
                round_num = metrics["round"]
                accuracy = metrics["global_accuracy"]
                print(f"  Round {round_num}: {accuracy:.2f}%")
        
    except Exception as e:
        print(f"Error running server: {e}")
        raise

# Example usage and testing
if __name__ == "__main__":
    # Test server strategy creation
    print("Testing server strategy...")
    
    # Create dummy test data
    from torch.utils.data import DataLoader, TensorDataset
    import torch
    
    dummy_test_data = torch.randn(1000, 1, 28, 28)
    dummy_test_targets = torch.randint(0, 10, (1000,))
    test_dataset = TensorDataset(dummy_test_data, dummy_test_targets)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create strategy
    strategy = create_server_strategy(
        test_dataloader=test_dataloader,
        num_clients=3,
        min_fit_clients=3,
        min_evaluate_clients=0,
        fraction_fit=1.0,
        fraction_evaluate=0.0
    )
    
    # Test parameter initialization
    initial_params = strategy.initialize_parameters(client_manager=None)
    print(f"Initial parameters created: {initial_params is not None}")
    
    # Test global evaluation
    if initial_params:
        result = strategy.evaluate(server_round=0, parameters=initial_params)
        if result:
            loss, metrics = result
            print(f"Initial evaluation - Loss: {loss:.4f}, Accuracy: {metrics['accuracy']:.2f}%")
    
    print("Server testing completed!")