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


        