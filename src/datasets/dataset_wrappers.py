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