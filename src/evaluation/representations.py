"""
Evaluation functions for learned representations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
import numpy as np


class LinearClassifier(nn.Module):
    """Simple linear classifier for representation evaluation."""
    
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)


def compute_linear_eval(train_features, train_labels, test_features, test_labels, 
                       epochs=25, batch_size=256, lr=0.001, device='cpu'):
    """
    Trains a linear classifier on the learned representations and evaluates accuracy and NMI.
    
    Args:
        train_features (torch.Tensor): Training representations of shape (N, feature_dim)
        train_labels (torch.Tensor): Training labels of shape (N,)
        test_features (torch.Tensor): Test representations of shape (M, feature_dim)
        test_labels (torch.Tensor): Test labels of shape (M,)
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        lr (float): Learning rate
        device (str): Device to run evaluation on
        
    Returns:
        tuple: (accuracy, nmi) - accuracy and normalized mutual information scores
    """
    
    # Move tensors to device
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    
    # Get input dimension and number of classes
    input_dim = train_features.shape[1]
    num_classes = len(torch.unique(train_labels))
    
    print(f"Linear classifier setup: {input_dim} features -> {num_classes} classes")
    
    # Create linear classifier
    classifier = LinearClassifier(input_dim, num_classes).to(device)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Create data loaders
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    classifier.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = classifier(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Linear eval epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluation
    classifier.eval()
    with torch.no_grad():
        test_outputs = classifier(test_features)
        _, predicted = torch.max(test_outputs, 1)
        
        # Move to CPU for sklearn metrics
        predicted_np = predicted.cpu().numpy()
        test_labels_np = test_labels.cpu().numpy()
        
        # Calculate accuracy
        accuracy = accuracy_score(test_labels_np, predicted_np)
        
        # Calculate NMI
        nmi = normalized_mutual_info_score(test_labels_np, predicted_np)
        
    print(f"Linear Evaluation Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  NMI: {nmi:.4f}")
    
    return accuracy, nmi
