import torch
import torchaudio
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model
import os
import numpy as np
from src.datasets.datasets import speechcommands_dataset
from src.utils.transforms import AudioTransform
from tqdm import tqdm
#specs
#find the device if exist

    
def get_wav2vec_representations(model,dataloader,device = "cuda"):
    model.to(device)
    model.eval()
    representations = []
    labels = []
    for batch in tqdm(dataloader,desc="Getting wav2vec representations"):
        waveform,label = batch
        representations.append(model(waveform.to(device).squeeze(1)).last_hidden_state.cpu())
        labels.append(label)
    
    return torch.cat(representations,dim=0),torch.cat(labels,dim=0)

def main():
    train_dataset,test_dataset,dataset = speechcommands_dataset(data_folder)
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model.to(device)
    
    # Create output directory
    output_dir = f"{data_folder}/wav2vec"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    #for each dataset get the wav2vec representations    
    train_dataloader = DataLoader(train_dataset,batch_size=100,shuffle=False)
    test_dataloader = DataLoader(test_dataset,batch_size=100,shuffle=False)

    with torch.no_grad():
        # Get original (non-augmented) representations
        print("Getting original train representations...")
        train_representations,train_labels = get_wav2vec_representations(model,train_dataloader,device = device)
        
        # Save with compression
        print("Saving original train data...")
        np.savez_compressed(f"{output_dir}/train_wav.npz", 
                          features=train_representations.numpy())
        np.savez_compressed(f"{output_dir}/train_labels.npz", 
                          labels=train_labels.numpy())
        
        print("Getting original test representations...")
        test_representations,test_labels = get_wav2vec_representations(model,test_dataloader,device = device)
        
        print("Saving original test data...")
        np.savez_compressed(f"{output_dir}/test_wav.npz", 
                          features=test_representations.numpy())
        np.savez_compressed(f"{output_dir}/test_labels.npz", 
                          labels=test_labels.numpy())
        
        print(f"Original representations shape: {train_representations.shape}")
        
        # Generate and save augmented representations individually
        print(f"\nGenerating {num_transforms} augmented versions...")
        
        for i in range(num_transforms):
            print(f"Processing augmentation {i+1}/{num_transforms}")
            
            # Create fresh augmented datasets for each iteration (different random seeds)
            train_augment_dataset = AugmentedWaveformDataset(train_dataset,AudioTransform())
            test_augment_dataset = AugmentedWaveformDataset(test_dataset,AudioTransform())
            
            train_aug_loader = DataLoader(train_augment_dataset,batch_size=100,shuffle=False)
            test_aug_loader = DataLoader(test_augment_dataset,batch_size=100,shuffle=False)
            
            # Get augmented representations
            train_aug_repr,_ = get_wav2vec_representations(model,train_aug_loader,device = device)
            test_aug_repr,_ = get_wav2vec_representations(model,test_aug_loader,device = device)
            
            # Save each augmentation separately with compression
            np.savez_compressed(f"{output_dir}/train_aug_{i}.npz", 
                              features=train_aug_repr.numpy())
            np.savez_compressed(f"{output_dir}/test_aug_{i}.npz", 
                              features=test_aug_repr.numpy())
            
            print(f"Saved augmentation {i+1} - Shape: {train_aug_repr.shape}")
            
            # Clear memory
            del train_aug_repr, test_aug_repr
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"\nAll data saved successfully to: {output_dir}")
        print(f"\nFiles created:")
        print(f"- train_wav.npz & train_labels.npz (original)")
        print(f"- test_wav.npz & test_labels.npz (original)")
        print(f"- train_aug_0.npz through train_aug_{num_transforms-1}.npz")
        print(f"- test_aug_0.npz through test_aug_{num_transforms-1}.npz")
        print(f"\nTotal: {4 + 2*num_transforms} compressed files")
if __name__ == "__main__":
    main()