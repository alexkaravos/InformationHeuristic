import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def iic_loss(p1,p2,lamb=1.0,eps=1e-12,scaling=True):
    """
    Args:
        p1 (batch_size, num_clusters): logits 1
        p2 (batch_size, num_clusters): logits 2
    """
    p1,p2 = F.softmax(p1,dim=1),F.softmax(p2,dim=1)
    
    #joint probability dist
    #P = (p1.unsqueeze(2) * p2.unsqueeze(1)).mean(dim=0)
    P = 1/p1.shape[0] * p1.T @ p2

    #marginal probabilities
    P1 = p1.mean(dim=0).unsqueeze(1)
    P2 = p2.mean(dim=0).unsqueeze(0)

    #compute the mutual information
    MI = (P * (torch.log(P + eps) - lamb * torch.log((P1+eps)*(P2+eps)))).sum()
    if scaling:
        k = p1.shape[1]
        MI -= np.log(k)*(2*lamb-2)
        MI /= (np.log(k))

    return -MI


class IIC_loss(nn.Module):
    def __init__(self,lamb=1.0,eps=1e-12,scaling=True):
        super(IIC_loss,self).__init__()
        self.lamb = lamb
        self.eps = eps
        self.scaling = scaling

    def forward(self,p1,p2,lamb=None):

        if lamb is None:
            lamb = self.lamb

        return iic_loss(p1,p2,lamb,self.eps,self.scaling)

def iic_loss_k(p1,p2,lamb=1.0,eps=1e-12,scaling=True):

    """
    Same implementation but now p2 is the logits of k neighbours
    args:
        p1 (batch_size, num_clusters): logits 1
        p2 (batch_size,k_neighbors,num_clusters): logits 2
        lamb regularization parameter
    """

    p1 = F.softmax(p1,dim=1) #bs,c
    p2 = F.softmax(p2,dim=2) #bs,k,c

    # we want to compute the joint distribution for each k so we get a k x c x c tensor
    P = p1.unsqueeze(1).unsqueeze(-1) * p2.unsqueeze(2) # (bs,1,c,1) * (bs,k,1,c) -> (bs,k,c,c)
    P = P.mean(dim=(0,1)) # (c x c)
    
    #marginal probabilities
    P1 = p1.mean(dim=0).unsqueeze(1)
    P2 = p2.mean(dim=0).unsqueeze(0)

    #compute the mutual information
    MI = (P * (torch.log(P + eps) - lamb * torch.log((P1+eps)*(P2+eps)))).sum()
    if scaling:
        k = p1.shape[1]
        MI -= np.log(k)*(2*lamb-2)
        MI /= (np.log(k))

    return -MI

class ScanLoss(nn.Module):
    def __init__(self,lamb=1.0):
        super(ScanLoss,self).__init__()
        self.lamb = lamb

    def forward(self,p1,p2,lamb=None):
        if lamb is None:
            lamb = self.lamb
        
        p2 = p2.squeeze(1)
        return contrastive_clustering_loss(p1,p2)




def scan_loss(p1,p2,lamb=1.0):
    """
    Args:
        p1 (batch_size, num_clusters): logits 1
        p2 (batch_size,k_neighbors,num_clusters): logits 2
        lamb regularization parameter
    """
    p1 = F.softmax(p1, dim=1)
    p2 = F.softmax(p2, dim=2)

    D,C = p1.shape
    
    dot_term = - 1/D * torch.sum(torch.log(torch.sum(p1.unsqueeze(1)*p2,dim=(2))+1e-10))
    #entropy term - get the average distribution across K
    
    P = p1.mean(dim=0)
    entropy = torch.sum(P * torch.log(P + 1e-10))

    #normalization factor 
    norm_factor = (lamb*torch.log(torch.tensor(C)))

    return (dot_term + lamb* entropy)/norm_factor       

import torch
import torch.nn.functional as F

def info_nce_loss_knn(p1, p2, temperature=0.1):
    """Calculates the InfoNCE loss for anchors and their k-neighbors using L2 normalized features."""
    p1 = F.normalize(p1, dim=1)
    p2 = F.normalize(p2, dim=-1)

    batch_size, k, feature_dim = p2.shape

    p2_flat = p2.reshape(batch_size * k, feature_dim)
    all_sims = p1 @ p2_flat.T
    grouped_sims = all_sims.reshape(batch_size, batch_size, k)
    similarity_matrix = grouped_sims.mean(dim=2)

    labels = torch.arange(batch_size, device=p1.device)
    logits = similarity_matrix / temperature
    loss = F.cross_entropy(logits, labels)

    return loss

def contrastive_clustering_loss(p1, p2, alpha=1.0, inst_temp=0.3, clust_temp=1.2):
    """
    Computes the Contrastive Clustering (CC) loss.
    This loss consists of an instance-level and a cluster-level contrastive objective.
    """
    # Instance-level loss
    p_i = F.softmax(p1, dim=1)
    p_j = F.softmax(p2, dim=1)
    p_cat = torch.cat([p_i, p_j], dim=0)
    
    sim_matrix_inst = F.cosine_similarity(p_cat.unsqueeze(1), p_cat.unsqueeze(0), dim=2)
    sim_matrix_inst = sim_matrix_inst / inst_temp
    
    batch_size = p1.shape[0]
    labels_inst = torch.arange(batch_size, device=p1.device)
    
    mask = torch.eye(batch_size * 2, device=p1.device).bool()
    sim_matrix_inst.masked_fill_(mask, -9e15)
    
    pos_mask = torch.zeros(batch_size * 2, batch_size * 2, device=p1.device, dtype=torch.bool)
    pos_mask[:batch_size, batch_size:] = torch.eye(batch_size)
    pos_mask[batch_size:, :batch_size] = torch.eye(batch_size)
    
    positives = sim_matrix_inst[pos_mask].view(batch_size * 2, 1)
    negatives = sim_matrix_inst[~pos_mask].view(batch_size * 2, -1)
    
    logits = torch.cat([positives, negatives], dim=1)
    labels_inst_ce = torch.zeros(batch_size * 2, device=p1.device, dtype=torch.long)
    loss_instance = F.cross_entropy(logits, labels_inst_ce)

    # Cluster-level loss
    c_i = F.normalize(p_i.t(), p=2, dim=1)
    c_j = F.normalize(p_j.t(), p=2, dim=1)
    c_cat = torch.cat([c_i, c_j], dim=0)
    
    sim_matrix_clust = F.cosine_similarity(c_cat.unsqueeze(1), c_cat.unsqueeze(0), dim=2)
    sim_matrix_clust = sim_matrix_clust / clust_temp
    
    num_clusters = p1.shape[1]
    
    mask_clust = torch.eye(num_clusters * 2, device=p1.device).bool()
    sim_matrix_clust.masked_fill_(mask_clust, -9e15)
    
    pos_mask_clust = torch.zeros(num_clusters * 2, num_clusters * 2, device=p1.device, dtype=torch.bool)
    pos_mask_clust[:num_clusters, num_clusters:] = torch.eye(num_clusters)
    pos_mask_clust[num_clusters:, :num_clusters] = torch.eye(num_clusters)
    
    positives_clust = sim_matrix_clust[pos_mask_clust].view(num_clusters * 2, 1)
    negatives_clust = sim_matrix_clust[~pos_mask_clust].view(num_clusters * 2, -1)
    
    logits_clust = torch.cat([positives_clust, negatives_clust], dim=1)
    labels_clust_ce = torch.zeros(num_clusters * 2, device=p1.device, dtype=torch.long)
    loss_cluster = F.cross_entropy(logits_clust, labels_clust_ce)

    return loss_instance + alpha * loss_cluster

def alex_loss(p1, p2, lamb_entropy=1.0, temperature=0.3):
    """
    A loss function for clustering that combines an InfoNCE-style contrastive
    objective with an entropy-based regularizer for cluster balance.

    This version explicitly calculates the InfoNCE loss without F.cross_entropy.

    Args:
        p1 (batch_size, num_clusters): logits for the anchor samples.
        p2 (batch_size, k_neighbors, num_clusters): logits for the neighbors.
        lamb_entropy (float): Weight for the entropy regularization term.
        temperature (float): Temperature scaling for the contrastive loss.
    """
    # Convert logits to probability distributions

    contrastive_loss = info_nce_loss_knn(p1, p2, temperature)

    p1 = F.softmax(p1, dim=1)
    p2 = F.softmax(p2, dim=2)

    D_p1, C = p1.shape
    D_p2, k, _ = p2.shape
    
    # Ensure batch sizes are consistent
    assert D_p1 == D_p2, "Batch sizes of p1 and p2 must be the same."
    D = D_p1

    dot_term = - 1/D * torch.sum(torch.log(torch.sum(p1.unsqueeze(1)*p2,dim=(2))+1e-10))


    # --- 2. Entropy Regularizer (for cluster balance) ---
    P_mean = p1.mean(dim=0)
    entropy_loss = torch.sum(P_mean * torch.log(P_mean + 1e-10))

    return dot_term + contrastive_loss + lamb_entropy * entropy_loss