import torchvision.transforms as transforms
import torch
import torchaudio.functional as F

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
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
])


#used for the speech commands dataset
class AudioTransform:
    """Standard audio augmentations 
    call applies:
    1. High-pass filter (20-800Hz cutoff) - p=0.5
    2. Low-pass filter (1.2-8kHz cutoff) - p=0.5
    3. Pitch shift (-2 to +2 semitones) - p=0.5
    4. Time shift (-10% to +10% with rollover) - p=0.5
    5. Volume/Gain change (-15dB to +5dB) - p=0.5
    6. Polarity inversion - p=0.5
    """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def __call__(self, waveform):
        """
        Apply standard contrastive learning augmentations
        Returns: augmented waveform
        """
        return self._apply_standard_augmentations(waveform.clone())
    
    def _apply_standard_augmentations(self, waveform):
        """Apply standard augmentations from audio contrastive learning papers"""
        
        # 1. High-pass filter (20-800Hz cutoff) - p=0.5
        if torch.rand(1).item() < 0.5:
            cutoff = torch.rand(1).item() * (800 - 20) + 20
            waveform = F.highpass_biquad(waveform, self.sample_rate, cutoff_freq=cutoff)
        
        # 2. Low-pass filter (1.2-8kHz cutoff) - p=0.5  
        if torch.rand(1).item() < 0.5:
            cutoff = torch.rand(1).item() * (8000 - 1200) + 1200
            waveform = F.lowpass_biquad(waveform, self.sample_rate, cutoff_freq=cutoff)
        
        # 3. Pitch shift (-2 to +2 semitones) - p=0.5
        if torch.rand(1).item() < 0.5:
            n_steps = torch.rand(1).item() * 4 - 2
            waveform = F.pitch_shift(waveform, self.sample_rate, n_steps)
        
        # 4. Time shift (-25% to +25% with rollover) - p=0.5
        if torch.rand(1).item() < 0.5:
            shift_percent = torch.rand(1).item() * 0.5 - 0.10
            shift_samples = int(shift_percent * waveform.shape[-1])
            waveform = torch.roll(waveform, shift_samples, dims=-1)
        
        # 5. Volume/Gain change (-15dB to +5dB) - p=0.5
        if torch.rand(1).item() < 0.5:
            gain_db = torch.rand(1).item() * 20 - 15
            gain_linear = 10 ** (gain_db / 20)
            waveform = waveform * gain_linear
        
        # 6. Polarity inversion - p=0.5
        if torch.rand(1).item() < 0.5:
            waveform = -waveform
            
        return waveform

