"""
Loss functions used for representatation learning. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):

        # z1, z2 are assumed to be raw logits
        z1,z2 = F.normalize(z1,dim=-1),F.normalize(z2,dim=-1)
        N, _ = z1.shape  # Batch size

        # Stack all positive pairs and negative pairs
        z = torch.cat([z1, z2], dim=0)

        # Compute cosine similarity
        sim_matrix = torch.mm(z, z.T) / self.temperature

        # Mask to remove positive self-pairs and fill diagonal with very small values
        mask = torch.eye(2 * N, device=z.device).bool()
        sim_matrix.masked_fill_(mask, -1e9)

        # Labels: each example should match with its pair (positive example)
        labels = torch.arange(2*N, device=z.device)
        labels = (labels + N) % (2*N)

        # Compute cross-entropy loss, equivalent to softmax with log
        loss = F.cross_entropy(sim_matrix, labels, reduction='mean')

        return loss

class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_param=5e-3):
        super().__init__()
        self.lambda_param = lambda_param

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Computes the Barlow Twins loss.
        z1 and z2 are the augmented embeddings, shape: (batch_size, feature_dim).
        """
        assert z1.shape == z2.shape, "Input embeddings must have the same shape"
        batch_size, feature_dim = z1.shape

        # Normalize the embeddings along the batch dimension
        z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-5)
        z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-5)

        # Cross-correlation matrix
        c = torch.matmul(z1_norm.T, z2_norm) / batch_size

        # Invariance term (diagonal elements should be 1)
        on_diag = torch.diagonal(c)
        invariance_loss = ((on_diag - 1)**2).sum()

        # Redundancy reduction term (off-diagonal elements should be 0)
        off_diag = c.clone()
        off_diag.fill_diagonal_(0)
        redundancy_loss = (off_diag**2).sum()

        # Total loss
        loss = invariance_loss + self.lambda_param * redundancy_loss
        return loss
    

class mixedBarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_param=5e-3,):
        super().__init__()
        self.lambda_param = lambda_param

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Computes the Barlow Twins loss.
        z1 and z2 are the augmented embeddings, shape: (batch_size, feature_dim).
        """
        assert z1.shape == z2.shape, "Input embeddings must have the same shape"
        batch_size, feature_dim = z1.shape

        # Normalize the embeddings along the batch dimension
        z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-5)
        z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-5)

        # Cross-correlation matrix
        c = torch.matmul(z1_norm.T, z2_norm) / batch_size

        # Invariance term (diagonal elements should be 1)
        on_diag = torch.diagonal(c)
        invariance_loss = ((on_diag - 1)**2).sum()

        # Redundancy reduction term (off-diagonal elements should be 0)
        off_diag = c.clone()
        off_diag.fill_diagonal_(0)
        redundancy_loss = (off_diag**2).sum()

        # Total loss
        loss = invariance_loss + self.lambda_param * redundancy_loss
        return loss