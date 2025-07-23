# model.py
# Purpose: Defines the neural network model architecture for MNIST digit classification
# This is a simple feedforward neural network optimized for federated learning scenarios

import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    """
    Simple feedforward neural network for MNIST digit classification.
    
    Architecture:
    - Input: 784 neurons (28x28 flattened MNIST images)
    - Hidden Layer 1: 128 neurons with ReLU activation
    - Hidden Layer 2: 64 neurons with ReLU activation  
    - Output: 10 neurons (logits for 10 digit classes)
    
    This architecture is chosen to be:
    1. Simple enough for fast training on client devices
    2. Effective for MNIST classification
    3. Small enough to minimize communication overhead in federated learning
    """
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        # First hidden layer: 784 -> 128
        # 784 comes from flattening 28x28 MNIST images
        self.fc1 = nn.Linear(784, 128)
        
        # Second hidden layer: 128 -> 64
        # Gradually reducing dimensions for feature extraction
        self.fc2 = nn.Linear(128, 64)
        
        # Output layer: 64 -> 10
        # 10 outputs for 10 digit classes (0-9)
        self.fc3 = nn.Linear(64, 10)
        
        # Dropout for regularization (helps prevent overfitting)
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28) or (batch_size, 784)
        
        Returns:
            logits: Output tensor of shape (batch_size, 10) with class logits
        """
        # Flatten the input if it's still in image format (28x28)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten to (batch_size, 784)
        
        # First hidden layer with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout for regularization
        
        # Second hidden layer with ReLU activation
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout for regularization
        
        # Output layer (no activation - raw logits for CrossEntropyLoss)
        x = self.fc3(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def get_parameters(self):
        """
        Extract model parameters as a list of numpy arrays.
        Used for federated learning parameter sharing.
        
        Returns:
            List of numpy arrays containing model weights and biases
        """
        return [param.detach().cpu().numpy() for param in self.parameters()]
    
    def set_parameters(self, parameters):
        """
        Set model parameters from a list of numpy arrays.
        Used for federated learning parameter aggregation.
        
        Args:
            parameters: List of numpy arrays containing model weights and biases
        """
        params_dict = zip(self.parameters(), parameters)
        for param, new_param in params_dict:
            param.data.copy_(torch.tensor(new_param))
    
    def get_model_info(self):
        """
        Get information about the model architecture.
        Useful for debugging and logging.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layers': [
                {'name': 'fc1', 'input': 784, 'output': 128},
                {'name': 'fc2', 'input': 128, 'output': 64},
                {'name': 'fc3', 'input': 64, 'output': 10}
            ]
        }

def create_model():
    """
    Factory function to create a new model instance.
    
    Returns:
        MNISTNet: A new instance of the neural network model
    """
    return MNISTNet()

# Example usage and testing
if __name__ == "__main__":
    # Create model instance
    model = create_model()
    