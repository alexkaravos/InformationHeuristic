from ast import LShift
import torchvision.transforms as transforms
import torch
import torchaudio.functional as F
import torch_audiomentations as audiomentations
import torch.nn as nn

mnist_aug1 = transforms.Compose([
    transforms.RandomRotation(degrees=35),
    transforms.RandomResizedCrop(28, scale=(0.8, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.4, p=0.7),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4),
                            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
                            ], p=0.5),
    transforms.Normalize((0.1307,), (0.3081,))
])

cifar_aug1 = transforms.Compose([
    transforms.RandomResizedCrop(32,scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

audio_transform = audiomentations.Compose([
        audiomentations.HighPassFilter(min_cutoff_freq=20.0, max_cutoff_freq=800.0, p=0.5, sample_rate=16000,output_type="tensor"),
        audiomentations.LowPassFilter(min_cutoff_freq=1200.0, max_cutoff_freq=8000.0, p=0.5, sample_rate=16000,output_type="tensor"),
        audiomentations.PitchShift(min_transpose_semitones=-2.0, max_transpose_semitones=2.0, p=0.5, sample_rate=16000,output_type="tensor"),
        audiomentations.Shift(min_shift=0.1, max_shift=0.1, p=0.5, rollover=True, sample_rate=16000,output_type="tensor"),
        audiomentations.Gain(min_gain_in_db=-15.0, max_gain_in_db=5.0, p=0.5, sample_rate=16000,output_type="tensor"),
        audiomentations.PolarityInversion(p=0.5, sample_rate=16000,output_type="tensor")
    ])  


class audio_transform_squeeze(nn.Module):
    """
    A wrapper to add unsqueeze and squeeze operations around the
    torch-audiomentations pipeline. This is useful for processing
    single audio files that lack a batch dimension.
    """
    def __init__(self, augmentations):
        super().__init__()
        self.augmentations = augmentations

    def forward(self, waveform, sample_rate=16000):
        # Add a batch dimension: (channels, samples) -> (batch, channels, samples)
        waveform = waveform.unsqueeze(0)
        
        # Apply the augmentations
        augmented_waveform = self.augmentations(samples=waveform, sample_rate=sample_rate)
        
        # Remove the batch dimension: (batch, channels, samples) -> (channels, samples)
        augmented_waveform = augmented_waveform.squeeze(0)
        
        return augmented_waveform

