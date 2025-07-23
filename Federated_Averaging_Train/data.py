# data.py
# Purpose: Loads the MNIST dataset and partitions it into non-IID subsets for each client.
# Non-IID means each client has a skewed distribution of digit classes, simulating real-world data heterogeneity.

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

class MNISTDataPartitioner:
    """
    Handles MNIST data loading and non-IID partitioning for federated learning.
    
    Non-IID (Non-Identically Distributed) data simulation:
    - Each client gets a skewed distribution of digit classes
    - This mimics real-world scenarios where different clients have different data patterns
    - Makes federated learning more challenging but realistic
    """
    
    def __init__(self, num_clients=3, batch_size=32, non_iid_alpha=0.5, data_path="./data"):
        """
        Initialize the data partitioner.
        
        Args:
            num_clients: Number of clients to partition data for
            batch_size: Default batch size for data loaders
            non_iid_alpha: Parameter controlling non-IID distribution (lower = more non-IID)
            data_path: Path to store/load MNIST data
        """
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.non_iid_alpha = non_iid_alpha
        self.data_path = data_path
        self.train_dataset = None
        self.test_dataset = None
        self.client_datasets = {}
        
        # Define transforms for MNIST data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])
    
    def load_data(self):
        """
        Load MNIST training and test datasets.
        
        Returns:
            tuple: (train_dataset, test_dataset)
        """
        print("Loading MNIST dataset...")
        
        # Load training data
        self.train_dataset = torchvision.datasets.MNIST(
            root=self.data_path,
            train=True,
            download=True,
            transform=self.transform
        )
        
        # Load test data
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.data_path,
            train=False,
            download=True,
            transform=self.transform
        )
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
        
        return self.train_dataset, self.test_dataset
    
    def partition_data(self):
        """
        Create non-IID data partitions for each client.
        This method calls create_non_iid_partition for compatibility.
        
        Returns:
            Dictionary of client datasets
        """
        return self.create_non_iid_partition()
    
    def create_non_iid_partition(self):
        """
        Create non-IID data partitions for each client.
        
        Strategy:
        - Client 1: 80% of samples from digits 0-3, 20% from others
        - Client 2: 80% of samples from digits 4-6, 20% from others  
        - Client 3: 80% of samples from digits 7-9, 20% from others
        
        Each client gets approximately equal total samples (~20,000 each)
        """
        if self.train_dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        print("Creating non-IID data partitions...")
        
        # Get all labels and create indices for each class
        all_labels = np.array([self.train_dataset[i][1] for i in range(len(self.train_dataset))])
        class_indices = {}
        
        for digit in range(10):
            class_indices[digit] = np.where(all_labels == digit)[0]
            print(f"Digit {digit}: {len(class_indices[digit])} samples")
        
        # Define client preferences (which digits each client prefers)
        client_preferences = {
            0: [0, 1, 2, 3],  # Client 1 prefers digits 0-3
            1: [4, 5, 6],     # Client 2 prefers digits 4-6
            2: [7, 8, 9]      # Client 3 prefers digits 7-9
        }
        
        # Create partitions for each client
        for client_id in range(self.num_clients):
            client_indices = []
            preferred_digits = client_preferences[client_id]
            
            # 80% of samples from preferred digits
            for digit in preferred_digits:
                digit_indices = class_indices[digit]
                num_samples = int(0.8 * len(digit_indices) / len(preferred_digits))
                selected_indices = np.random.choice(digit_indices, num_samples, replace=False)
                client_indices.extend(selected_indices)
                
                # Remove selected indices to avoid overlap
                class_indices[digit] = np.setdiff1d(class_indices[digit], selected_indices)
            
            # 20% of samples from other digits
            other_digits = [d for d in range(10) if d not in preferred_digits]
            remaining_samples_needed = 20000 - len(client_indices)  # Target ~20k samples per client
            
            for digit in other_digits:
                if len(class_indices[digit]) == 0:
                    continue
                    
                # Distribute remaining samples across other digits
                num_samples = min(
                    remaining_samples_needed // len(other_digits),
                    len(class_indices[digit])
                )
                
                if num_samples > 0:
                    selected_indices = np.random.choice(class_indices[digit], num_samples, replace=False)
                    client_indices.extend(selected_indices)
                    class_indices[digit] = np.setdiff1d(class_indices[digit], selected_indices)
            
            # Create subset for this client
            client_indices = np.array(client_indices)
            np.random.shuffle(client_indices)  # Shuffle to avoid ordering bias
            
            self.client_datasets[client_id] = Subset(self.train_dataset, client_indices)
            
            print(f"Client {client_id}: {len(client_indices)} samples")
        
        # Log the distribution for verification
        self.log_client_distributions()
        
        return self.client_datasets
    
    def log_data_distribution(self):
        """
        Log the data distribution across clients.
        This method calls log_client_distributions for compatibility.
        """
        self.log_client_distributions()
    
    def log_client_distributions(self):
        """
        Log and visualize the class distribution for each client.
        This helps verify that the non-IID partitioning worked correctly.
        """
        print("\n" + "="*50)
        print("CLIENT DATA DISTRIBUTION (Non-IID)")
        print("="*50)
        
        for client_id, dataset in self.client_datasets.items():
            # Count samples per class for this client
            labels = [dataset.dataset[idx][1] for idx in dataset.indices]
            class_counts = Counter(labels)
            
            print(f"\nClient {client_id}:")
            print(f"Total samples: {len(dataset)}")
            
            # Show distribution per digit
            for digit in range(10):
                count = class_counts.get(digit, 0)
                percentage = (count / len(dataset)) * 100
                print(f"  Digit {digit}: {count:4d} samples ({percentage:5.1f}%)")
            
            # Calculate and show dominant classes
            dominant_classes = [digit for digit, count in class_counts.items() 
                              if count > len(dataset) * 0.15]  # Classes with >15% of data
            print(f"  Dominant classes: {dominant_classes}")
    
    def get_client_dataloader(self, client_id, batch_size=32, shuffle=True):
        """
        Get a DataLoader for a specific client.
        
        Args:
            client_id: ID of the client (0, 1, 2, ...)
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader for the client's data
        """
        if client_id not in self.client_datasets:
            raise ValueError(f"Client {client_id} not found. Available clients: {list(self.client_datasets.keys())}")
        
            return DataLoader(
            self.client_datasets[client_id],
            batch_size=batch_size,
            shuffle=shuffle
        )
    
    def create_dataloaders(self, client_id):
        """
        Create train and validation dataloaders for a specific client.
        Splits the client's data into 80% train, 20% validation.
        
        Args:
            client_id: ID of the client
            
        Returns:
            tuple: (train_loader, val_loader)
        """
        if client_id not in self.client_datasets:
            raise ValueError(f"Client {client_id} not found. Available clients: {list(self.client_datasets.keys())}")
        
        client_dataset = self.client_datasets[client_id]
        dataset_size = len(client_dataset)
        
        # Create train/val split (80/20)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        
        # Get indices for train/val split
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create subsets
        train_subset = Subset(client_dataset, train_indices)
        val_subset = Subset(client_dataset, val_indices)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def get_test_dataloader(self, batch_size=None):
        """
        Get a DataLoader for the test dataset (used for global evaluation).
        
        Args:
            batch_size: Batch size for testing (uses default if None)
            
        Returns:
            DataLoader for the test data
        """
        if self.test_dataset is None:
            raise ValueError("Test dataset not loaded. Call load_data() first.")
        
        if batch_size is None:
            batch_size = self.batch_size
        
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    
    def visualize_distribution(self, save_path="client_distribution.png"):
        """
        Create a visualization of the data distribution across clients.
        
        Args:
            save_path: Path to save the visualization
        """
        if not self.client_datasets:
            print("No client datasets found. Run create_non_iid_partition() first.")
            return
        
        # Prepare data for visualization
        clients = list(self.client_datasets.keys())
        digits = list(range(10))
        
        # Create matrix of counts
        distribution_matrix = np.zeros((len(clients), 10))
        
        for i, client_id in enumerate(clients):
            dataset = self.client_datasets[client_id]
            labels = [dataset.dataset[idx][1] for idx in dataset.indices]
            class_counts = Counter(labels)
            
            for digit in digits:
                distribution_matrix[i, digit] = class_counts.get(digit, 0)
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot as heatmap
        plt.subplot(1, 2, 1)
        plt.imshow(distribution_matrix, cmap='Blues', aspect='auto')
        plt.colorbar(label='Number of Samples')
        plt.xlabel('Digit Class')
        plt.ylabel('Client ID')
        plt.title('Data Distribution Heatmap')
        plt.xticks(range(10))
        plt.yticks(range(len(clients)))
        
        # Plot as stacked bar chart
        plt.subplot(1, 2, 2)
        bottom = np.zeros(len(clients))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for digit in range(10):
            plt.bar(clients, distribution_matrix[:, digit], bottom=bottom, 
                   label=f'Digit {digit}', color=colors[digit])
            bottom += distribution_matrix[:, digit]
        
        plt.xlabel('Client ID')
        plt.ylabel('Number of Samples')
        plt.title('Data Distribution per Client')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Distribution visualization saved to {save_path}")

# Example usage and testing
if __name__ == "__main__":
    # Create data partitioner
    partitioner = MNISTDataPartitioner(num_clients=3)
    
    # Load data
    train_dataset, test_dataset = partitioner.load_data()
    
    # Create non-IID partitions
    client_datasets = partitioner.create_non_iid_partition()
    
    # Test client dataloaders
    print("\nTesting client dataloaders...")
    for client_id in range(3):
        dataloader = partitioner.get_client_dataloader(client_id, batch_size=32)
        print(f"Client {client_id} dataloader: {len(dataloader)} batches")
        
        # Test one batch
        for batch_idx, (data, target) in enumerate(dataloader):
            print(f"  Batch {batch_idx}: {data.shape}, {target.shape}")
            if batch_idx == 0:  # Only show first batch
                break
    
    # Test test dataloader
    test_dataloader = partitioner.get_test_dataloader(batch_size=32)
    print(f"\nTest dataloader: {len(test_dataloader)} batches")
    
    # Create visualization
    partitioner.visualize_distribution()