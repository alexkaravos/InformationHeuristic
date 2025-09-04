"""
Wrappers for the dataset classes, mainly to support run-time augmentation.
"""
import torch
from torch.utils.data import Dataset
import torchaudio
from torch.nn import functional as F
class AugmentedDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for applying augmentation(s) to samples.
    
    Returns tuple of (aug1, aug2, ..., original_img, label)
    Optionally if there's normalization to be applied to the original x
    it can be stripped off as transform[-1], or add_normalize=False and pass the normalize function
    as normalize=normalize.
    """
    def __init__(self, dataset, augmentation, num_augs=2,normalize = None,add_normalize=True):
        self.dataset = dataset
        self.transform = augmentation
        self.num_augs = num_augs
        self.normalize = normalize
        
        if add_normalize:
            self.normalize = augmentation.transforms[-1]
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img, label = self.dataset[index]
        
        # Generate augmentations
        augs = [self.transform(img) for _ in range(self.num_augs)]
        
        # Apply normalization to original if specified
        if self.normalize is not None:
            img = self.normalize(img)
        
        # Return augs + original + label
        return (*augs, img, label)
    
class DoubleAugmentedDataset(torch.utils.data.Dataset):
    """
    Dataset class for running the augmnetations at the individual sample level
    returns a tuple of augmeted samples and labels. 
    """
    def __init__(self,dataset,augmentation,add_normalize=True):
        self.dataset = dataset
        self.transform = augmentation
        if add_normalize:
            #the last transform is the normalization on the augmentation 
            self.normalize = augmentation.transforms[-1]
        else:
            self.normalize = None

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,index):
        img,label = self.dataset[index]
        print(img.shape)
        x1 = self.transform(img)
        x2 = self.transform(img)
        if self.normalize is not None:
            img = self.normalize(img)
        return x1,x2,img,label
    

class DoubleAugmentedDataset(torch.utils.data.Dataset):
    """
    Dataset class for running the augmnetations at the individual sample level
    returns a tuple of augmeted samples and labels. 
    """
    def __init__(self,dataset,augmentation,add_normalize=True):
        self.dataset = dataset
        self.transform = augmentation
        if add_normalize:
            #the last transform is the normalization on the augmentation 
            self.normalize = augmentation.transforms[-1]
        else:
            self.normalize = None

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,index):
        img,label = self.dataset[index]
        x1 = self.transform(img)
        x2 = self.transform(img)
        if self.normalize is not None:
            img = self.normalize(img)
        return x1,x2,img,label
    

def find_knn(X: torch.Tensor, k: int, metric: str = 'euclidean'):
    """
    Finds the k-nearest neighbors for each row in a tensor.

    Args:
        X (torch.Tensor): The input tensor of shape (N, D).
        k (int): The number of nearest neighbors to find.
        metric (str): The distance metric to use, either 'euclidean' or 'cosine'.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - distances (torch.Tensor): The distances to the k-nearest neighbors, shape (N, k).
            - indices (torch.Tensor): The indices of the k-nearest neighbors, shape (N, k).
    """
    N = X.shape[0]
    
    if metric == 'euclidean':
        # Calculate pairwise Euclidean distances
        dist_matrix = torch.cdist(X, X, p=2)
    elif metric == 'cosine':
        # Normalize vectors to unit length for cosine similarity calculation
        X_norm = F.normalize(X, p=2, dim=1)
        # Calculate cosine similarity and convert to distance
        sim_matrix = torch.mm(X_norm, X_norm.t())
        dist_matrix = 1 - sim_matrix
    else:
        raise ValueError("Unsupported metric. Choose 'euclidean' or 'cosine'.")

    # To exclude self-matches, set the diagonal of the distance matrix to infinity
    dist_matrix.fill_diagonal_(float('inf'))

    # Find the k smallest distances and their corresponding indices for each row
    distances,indices = torch.topk(dist_matrix, k, dim=1, largest=False)

    return indices


class augmentedNearestNeighborDataset(torch.utils.data.Dataset):
    """
    Dataset class that returns the sample and k neighbours
    inputs are
    - X (N,D)
    - X_aug (N,num_augs,D)
    - neighbours (N,K)
    - labels (N)
    - k (number of neighbours to return each sample)
    returns:
    - X (D)
    - X_neighbours (k,D)
    - labels (N)
    """
    def __init__(self,X,X_aug,neighbours,labels,k=5):
        self.X = X # N,D
        self.X_aug = X_aug # N,A,D
        self.k = k 
        self.K = neighbours.shape[1]
        self.labels = labels # N
        self.neighbours = neighbours # N,K
        self.num_augs = X_aug.shape[1]        
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,index):
        aug_index = torch.randperm(self.num_augs)[:6]
        aug_index,neigh_index = aug_index[0],aug_index[1:]
        neigh_idx = self.neighbours[index,torch.randperm(self.K)[:self.k]]
        
        x = self.X[index]
        x_aug = self.X_aug[index,aug_index]
        x_neighbours = self.X_aug[index,neigh_index]
        label = self.labels[index]
        return x_aug,x_neighbours,x,label

class AugRepresentationDataset(torch.utils.data.Dataset):
    """
    Dataset class containing the augs, originals, and labels. Returns 
    a tuple of (num_augs,org,labels)
    """
    def __init__(self,augmentations,originals,labels,num_augs_return=2):
        self.augmentations = augmentations
        self.originals = originals
        self.labels = labels
        self.num_augs = augmentations.shape[1]
        self.num_augs_return = num_augs_return   
    def __len__(self):
        return len(self.originals)
    
    def __getitem__(self,index):
        #randomly samples num_augs_return from the num_augs
        aug_indices = torch.randperm(self.num_augs)[:self.num_augs_return]

        original_sample = self.originals[index]
        label = self.labels[index]
        augs = [self.augmentations[index,i] for i in aug_indices]
        
        return (*augs,original_sample,label)
    

