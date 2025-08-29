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


def compute_knn_accuracy(
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    k: int = 1,
    distance_metric: str = "euclidean"
) -> float:
    """
    Computes the KNN Top-N classification accuracy using either Euclidean or Cosine similarity.

    Args:
        test_features (torch.Tensor): A tensor of features for the test samples.
                                     Shape: (N_test, feature_dim).
        test_labels (torch.Tensor): A tensor of corresponding labels for the
                                   test samples. Shape: (N_test).
        train_features (torch.Tensor): A tensor of features for the training/database samples.
                                  Shape: (N_train, feature_dim).
        train_labels (torch.Tensor): A tensor of corresponding labels for the
                                database samples. Shape: (N_train).
        k (int): The number of nearest neighbors to consider for classification.
                 Must be a positive integer. Defaults to 1 (Top-1 accuracy).
        distance_metric (str): The metric to use for finding neighbors.
                               Accepts "euclidean" or "cosine".
                               Defaults to "euclidean".

    Returns:
        float: The computed Top-N accuracy as a decimal value (e.g., 0.85 for 85%).
    """
    # Ensure inputs are tensors and on the same device.
    if not all(isinstance(t, torch.Tensor) for t in [test_features, test_labels, train_features, train_labels]):
        raise TypeError("All inputs must be torch.Tensor objects.")

    # Validate k and distance_metric.
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    if distance_metric not in ["euclidean", "cosine"]:
        raise ValueError("distance_metric must be either 'euclidean' or 'cosine'.")

    if distance_metric == "euclidean":
        # Calculate the pairwise Euclidean distance matrix.
        # Shape: (N_test, N_train). Lower value means closer.
        distances = torch.cdist(test_features, train_features, p=2)
        # Find the k smallest distances and their corresponding indices.
        # The 'sorted=True' ensures that the indices are in ascending order of distance.
        topk_values, topk_indices = torch.topk(distances, k, largest=False, sorted=True)

    elif distance_metric == "cosine":
        # Cosine similarity requires normalized vectors for a correct interpretation.
        test_features = F.normalize(test_features, dim=1)
        train_features = F.normalize(train_features, dim=1)
        
        # Calculate the pairwise cosine similarity matrix.
        # Shape: (N_test, N_train). Higher value means closer similarity.
        similarities = F.cosine_similarity(test_features.unsqueeze(1), train_features.unsqueeze(0), dim=2)
        
        # Find the k largest similarities and their corresponding indices.
        # 'largest=True' is crucial here as we are looking for maximum similarity.
        topk_values, topk_indices = torch.topk(similarities, k, largest=True, sorted=True)

    # Use the top-k indices to look up the labels of the nearest neighbors.
    # This results in a tensor of shape (N_test, k).
    predicted_labels_topk = train_labels[topk_indices]

    # Reshape test labels to (N_test, 1) to enable a direct comparison.
    test_labels_reshaped = test_labels.unsqueeze(1)

    # Check if the true label is present in the top-k predicted labels for each query.
    # The comparison '==' broadcasts the test_labels_reshaped across the k dimension.
    # 'torch.any(..., dim=1)' checks if at least one of the k predictions is correct.
    correct_predictions = torch.any(predicted_labels_topk == test_labels_reshaped, dim=1)

    # Calculate the accuracy as the mean of correct predictions.
    accuracy = correct_predictions.float().mean().item()

    return accuracy
