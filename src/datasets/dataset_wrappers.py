"""
Wrappers for the dataset classes, mainly to support run-time augmentation.
"""
import torch
from torch.utils.data import Dataset
import torchaudio

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
        x1 = self.transform(img)
        x2 = self.transform(img)
        if self.normalize is not None:
            img = self.normalize(img)
        return x1,x2,img,label
    