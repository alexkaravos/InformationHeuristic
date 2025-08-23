"""
Functions for fetching and processing the raw datasets. All datasets we use are loaded
throug this file. For those which can be downloaded if they are not at the path specified, they will be downloaded.

Currently supported datasets:
- MNIST (full, unbalanced)
- CIFAR-10 (full, unbalanced)
- CIFAR-100 (full, unbalanced)
- CIFAR-20 (full, unbalanced)
- STL-10

"""
import torch
import torchvision
import torchaudio
from torchvision import transforms


def mnist_dataset(data_folder, normalize=False):
    """
    Load MNIST dataset with optional normalization.
    
    Args:
        data_folder (str): Directory to download/store dataset
        normalize (bool): Apply MNIST normalization (mean=0.1307, std=0.3081)
    
    Returns:
        tuple: (train_dataset, test_dataset, combined_dataset)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) if normalize else transforms.Lambda(lambda x: x)
    ])
    if not normalize:
        transform = transforms.ToTensor()

    train_dataset = torchvision.datasets.MNIST(
        root=data_folder, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_folder, train=False, download=True, transform=transform
    )
    
    return train_dataset, test_dataset, torch.utils.data.ConcatDataset([train_dataset, test_dataset])

def cifar10_dataset(data_folder, transform=transforms.ToTensor(), normalize=False):
    """
    Load CIFAR-10 dataset with customizable transforms.
    
    Args:
        data_folder (str): Directory to download/store dataset
        transform: Base transform to apply (default: ToTensor)
        normalize (bool): Apply CIFAR-10 normalization
    
    Returns:
        tuple: (train_dataset, test_dataset, combined_dataset)
    """
    if normalize:
        transform = transforms.Compose([
            transform,
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_folder, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_folder, train=False, download=True, transform=transform
    )
    
    return train_dataset, test_dataset, torch.utils.data.ConcatDataset([train_dataset, test_dataset])




def cifar100_dataset(data_folder, transform=transforms.ToTensor(), normalize=False):
    """
    Load CIFAR-100 dataset with customizable transforms.
    
    Args:
        data_folder (str): Directory to download/store dataset  
        transform: Base transform to apply (default: ToTensor)
        normalize (bool): Apply CIFAR-100 normalization
    
    Returns:
        tuple: (train_dataset, test_dataset, combined_dataset)
    """
    if normalize:
        transform = transforms.Compose([
            transform,
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=data_folder, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_folder, train=False, download=True, transform=transform
    )
    
    return train_dataset, test_dataset, torch.utils.data.ConcatDataset([train_dataset, test_dataset])




def _cifar100_to_cifar20(target):
    """
    Convert CIFAR-100 fine-grained label to CIFAR-20 coarse-grained label.
    
    Maps each of the 100 CIFAR-100 classes to one of 20 superclasses
    based on semantic similarity.
    """
    mapping = {
        0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3, 
        10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 
        19: 11, 20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 
        28: 3, 29: 15, 30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 
        37: 9, 38: 11, 39: 5, 40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 
        46: 14, 47: 17, 48: 18, 49: 10, 50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 
        55: 0, 56: 17, 57: 4, 58: 18, 59: 17, 60: 10, 61: 3, 62: 2, 63: 12, 
        64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19, 70: 2, 71: 10, 72: 0, 
        73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13, 80: 16, 81: 19, 
        82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19, 90: 18, 
        91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13
    }

    return mapping[target]

def cifar20_dataset(data_folder, transform=transforms.ToTensor(), normalize=False):
    """
    Load CIFAR-20 dataset (CIFAR-100 with remapped labels to 20 superclasses).
    
    Maps the 100 fine-grained CIFAR-100 classes to 20 coarse-grained superclasses
    for experiments requiring fewer classes.
    
    Args:
        data_folder (str): Directory to download/store dataset
        transform: Base transform to apply (default: ToTensor)  
        normalize (bool): Apply CIFAR-100 normalization
    
    Returns:
        tuple: (train_dataset, test_dataset, combined_dataset)
    """
    train_dataset_100, test_dataset_100, _ = cifar100_dataset(data_folder, transform, normalize)
    
    class CIFAR20Dataset(torch.utils.data.Dataset):
        def __init__(self, cifar100_dataset):
            self.dataset = cifar100_dataset
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, index):
            img, label = self.dataset[index]
            return img, _cifar100_to_cifar20(label)
    
    train_dataset = CIFAR20Dataset(train_dataset_100)
    test_dataset = CIFAR20Dataset(test_dataset_100)
    
    return train_dataset, test_dataset, torch.utils.data.ConcatDataset([train_dataset, test_dataset])




def stl10_dataset(data_folder, transform=transforms.ToTensor(), normalize=False):
    """
    Load STL-10 dataset (labeled portion only).
    
    Only includes the 10-class labeled training and test data, excluding
    the large unlabeled portion of STL-10.
    
    Args:
        data_folder (str): Directory to download/store dataset
        transform: Base transform to apply (default: ToTensor)
        normalize (bool): Apply STL-10 normalization
    
    Returns:
        tuple: (train_dataset, test_dataset, combined_dataset)
    """
    if normalize:
        transform = transforms.Compose([
            transform,
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
        ])

    train_dataset = torchvision.datasets.STL10(
        root=data_folder, split='train', download=True, transform=transform
    )
    test_dataset = torchvision.datasets.STL10(
        root=data_folder, split='test', download=True, transform=transform
    )
    
    return train_dataset, test_dataset, torch.utils.data.ConcatDataset([train_dataset, test_dataset])



def mnist_unbalanced_dataset(data_folder, normalize=False, balance_settings=0,random_seed=0):
    """
    Load MNIST with class imbalance for robustness experiments.
    
    Creates artificial class imbalance by downsampling each digit class 
    according to predefined distributions.
    
    Args:
        data_folder (str): Directory to download/store dataset
        normalize (bool): Apply MNIST normalization
        balance_settings (int): Imbalance configuration (currently only 0 supported)
    
    Returns:
        tuple: (train_dataset, test_dataset, combined_dataset)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) if normalize else transforms.Lambda(lambda x: x)
    ])
    if not normalize:
        transform = transforms.ToTensor()
    
    train_dataset = torchvision.datasets.MNIST(
        root=data_folder, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_folder, train=False, download=True, transform=transform
    )

    if balance_settings == 0:
        distribution = {0: 0.9, 1: 0.4, 2: 0.6, 3: 0.3, 4: 0.7,
                       5: 0.7, 6: 0.5, 7: 0.15, 8: 0.9, 9: 1.0}

    torch.manual_seed(random_seed)
    
    train_imgs, train_labels = [], []
    test_imgs, test_labels = [], []
    
    for img, label in train_dataset:
        if label in distribution and torch.rand(1) < distribution[label]:
            train_imgs.append(img)
            train_labels.append(label)
    
    for img, label in test_dataset:
        if label in distribution and torch.rand(1) < distribution[label]:
            test_imgs.append(img)
            test_labels.append(label)

    train_dataset = torch.utils.data.TensorDataset(torch.stack(train_imgs), torch.tensor(train_labels))
    test_dataset = torch.utils.data.TensorDataset(torch.stack(test_imgs), torch.tensor(test_labels))
    
    return train_dataset, test_dataset, torch.utils.data.ConcatDataset([train_dataset, test_dataset])

def cifar100_unbalanced_dataset(data_folder, normalize=False, random_seed=0):
    """
    Load CIFAR-100 with class imbalance for robustness experiments.
    
    Creates artificial class imbalance by downsampling each class 
    according to predefined distributions.
    
    Args:
        data_folder (str): Directory to download/store dataset
        normalize (bool): Apply CIFAR-100 normalization
        random_seed (int): Seed for random downsampling
    
    Returns:
        tuple: (train_dataset, test_dataset, combined_dataset)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) if normalize else transforms.Lambda(lambda x: x)
    ])
    if not normalize:
        transform = transforms.ToTensor()
    
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_folder, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_folder, train=False, download=True, transform=transform
    )

    torch.manual_seed(random_seed)

    # return a random distribution of the classes between 0.2 and 1 of
    distribution = {i: torch.rand(1) for i in range(100)}
    
    train_imgs, train_labels = [], []
    test_imgs, test_labels = [], []
    
    for img, label in train_dataset:
        if label in distribution and torch.rand(1) < distribution[label]:
            train_imgs.append(img)
            train_labels.append(label)
    
    for img, label in test_dataset:
        if label in distribution and torch.rand(1) < distribution[label]:
            test_imgs.append(img)
            test_labels.append(label)

    train_dataset = torch.utils.data.TensorDataset(torch.stack(train_imgs), torch.tensor(train_labels))
    test_dataset = torch.utils.data.TensorDataset(torch.stack(test_imgs), torch.tensor(test_labels))
    
    return train_dataset, test_dataset, torch.utils.data.ConcatDataset([train_dataset, test_dataset])

def cifar10_unbalanced_dataset(data_folder, normalize=False, random_seed=0):
    """
    Load CIFAR-10 with class imbalance for robustness experiments.
    
    Creates artificial class imbalance by downsampling each digit class 
    according to predefined distributions.
    
    Args:
        data_folder (str): Directory to download/store dataset
        normalize (bool): Apply CIFAR-10 normalization
        random_seed (int): Seed for random downsampling
    
    Returns:
        tuple: (train_dataset, test_dataset, combined_dataset)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) if normalize else transforms.Lambda(lambda x: x)
    ])
    if not normalize:
        transform = transforms.ToTensor()
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_folder, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_folder, train=False, download=True, transform=transform
    )

    torch.manual_seed(random_seed)

    distribution = {0: 0.9, 1: 0.4, 2: 0.6, 3: 0.3, 4: 0.7,
                   5: 0.7, 6: 0.5, 7: 0.15, 8: 0.9, 9: 1.0}
    
    train_imgs, train_labels = [], []
    test_imgs, test_labels = [], []
    
    for img, label in train_dataset:
        if label in distribution and torch.rand(1) < distribution[label]:
            train_imgs.append(img)
            train_labels.append(label)
    
    for img, label in test_dataset:
        if label in distribution and torch.rand(1) < distribution[label]:
            test_imgs.append(img)
            test_labels.append(label)

    train_dataset = torch.utils.data.TensorDataset(torch.stack(train_imgs), torch.tensor(train_labels))
    test_dataset = torch.utils.data.TensorDataset(torch.stack(test_imgs), torch.tensor(test_labels))
    
    return train_dataset, test_dataset, torch.utils.data.ConcatDataset([train_dataset, test_dataset])


def cifar20_unbalanced_dataset(data_folder, transform=transforms.ToTensor(), normalize=False, random_seed=0):
    """
    Load CIFAR-20 unbalanced dataset (CIFAR-100 with remapped labels to 20 superclasses).
    
    Maps the 100 fine-grained CIFAR-100 classes to 20 coarse-grained superclasses
    for experiments requiring fewer classes. The imbalance is created at the CIFAR-100 level.
    
    Args:
        data_folder (str): Directory to download/store dataset
        transform: Base transform to apply (default: ToTensor)  
        normalize (bool): Apply CIFAR-100 normalization
        random_seed (int): Seed for random downsampling
    
    Returns:
        tuple: (train_dataset, test_dataset, combined_dataset)
    """
    train_dataset_100, test_dataset_100, _ = cifar100_unbalanced_dataset(data_folder, normalize, random_seed=random_seed)
    
    class CIFAR20Dataset(torch.utils.data.Dataset):
        def __init__(self, cifar100_dataset):
            self.dataset = cifar100_dataset
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, index):
            img, label = self.dataset[index]
            return img, _cifar100_to_cifar20(label)
    
    train_dataset = CIFAR20Dataset(train_dataset_100)
    test_dataset = CIFAR20Dataset(test_dataset_100)
     
    return train_dataset, test_dataset, torch.utils.data.ConcatDataset([train_dataset, test_dataset])


def speechcommands_dataset_raw(data_folder):

    
    test_dataset = torchaudio.datasets.SPEECHCOMMANDS(
        root=data_folder,
        download=True,
        subset='testing'
    )

    train_dataset = torchaudio.datasets.SPEECHCOMMANDS(
        root=data_folder,
        download=True,
        subset='training'
    )

    val_dataset = torchaudio.datasets.SPEECHCOMMANDS(
        root=data_folder,
        download=True,
        subset='validation'
    )

    train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

    return train_dataset,test_dataset,torch.utils.data.ConcatDataset([train_dataset,test_dataset])

def speechcommands_dataset(data_folder):

    speech_labels = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
    'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree',
    'two', 'up', 'visual', 'wow', 'yes', 'zero'
    ]

    train_dataset,test_dataset,_ = speechcommands_dataset_raw(data_folder)

    train_waveforms = []
    train_labels = []
    for sample in train_dataset:
        waveform,_,label,_,_ = sample
        if waveform.shape[1] < 16000:
            padding_needed = 16000 - waveform.shape[1]
            padding = torch.zeros((1, padding_needed))
            waveform = torch.cat([waveform, padding], dim=1)
        else:
            waveform = waveform[:, :16000]
        train_waveforms.append(waveform)
        train_labels.append(speech_labels.index(label))

    train_waveforms = torch.stack(train_waveforms)
    train_labels = torch.tensor(train_labels)

    test_waveforms = []
    test_labels = []
    for sample in test_dataset:
        waveform,_,label,_,_ = sample
        if waveform.shape[1] < 16000:
            padding_needed = 16000 - waveform.shape[1]
            padding = torch.zeros((1, padding_needed))
            waveform = torch.cat([waveform, padding], dim=1)
        else:
            waveform = waveform[:, :16000]
        test_waveforms.append(waveform)
        test_labels.append(speech_labels.index(label))

    test_waveforms = torch.stack(test_waveforms)
    test_labels = torch.tensor(test_labels)

    train_dataset = torch.utils.data.TensorDataset(train_waveforms,train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_waveforms,test_labels)

    return train_dataset,test_dataset,torch.utils.data.ConcatDataset([train_dataset,test_dataset])


