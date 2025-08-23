import os
import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.datasets import speechcommands_dataset
from src.datasets.dataset_wrappers import DoubleAugmentedDataset
from src.models.wav2vec import Wav2VecClsPooler
from src.representations.losses import BarlowTwinsLoss 
from src.representations.training import train_contrastive
from src.utils.transforms import AudioTransform
from src.evaluation.representations import compute_linear_eval


@hydra.main(config_path="../../../configs", config_name="experiments/speechcommands/speechcommands_barlow", version_base=None)
def main(cfg: DictConfig):

    # Print the composed config
    print("--- Composed Hydra Config ---")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------------------")

    # Set device and seed
    device = cfg.device
    torch.manual_seed(cfg.seed)
    
    # Load dataset
    train_dataset, test_dataset, full_dataset = speechcommands_dataset(
        data_folder=cfg.datasets.data_folder
    )

    # Get transform from transforms.py by string name
    transform = AudioTransform()
    
    augmented_dataset = DoubleAugmentedDataset(
        dataset=train_dataset, # Use the training split for training
        augmentation=transform,
        add_normalize=False
    )
    
    # Create dataloader
    train_loader = DataLoader(
        augmented_dataset,
        batch_size=cfg.representations.batch_size,
        shuffle=True,
        num_workers=10,
        pin_memory=True
    )
    
    # Create model, which now includes the encoder and projection head
    model = Wav2VecClsPooler(projection_dim=cfg.models.projection_dim)

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg.representations.lr)
    if cfg.representations.scheduler == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.representations.gamma)
    else:
        scheduler = None
    
    # Create loss function      
    criterion = BarlowTwinsLoss(lambda_param=cfg.representations.lambda_param)
    
    # Train model
    print(f"Starting training for {cfg.experiment_name}")
    print(f"Device: {device}")
    print(f"Epochs: {cfg.representations.epochs}")
    print(f"Batch size: {cfg.representations.batch_size}")
    print(f"Lambda Param: {cfg.representations.lambda_param}")
    print(f"Save directory: {cfg.save_dir}")
    
    # Make sure the saving directory exists
    save_dir = os.path.join(cfg.save_dir, f"barlow_{cfg.models.arch}_{cfg.representations.lambda_param*1000:.2f}")
    os.makedirs(save_dir, exist_ok=True)    
    model.to(device)
    
    # The train_contrastive function needs to handle the DoubleAugmentedDataset output format
    trained_model = train_contrastive(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=cfg.representations.epochs,
        device=device,
        save_dir=save_dir
    )
    
    # --- Linear Evaluation ---
    model.eval()
    
    # Get the feature representations from the trained model
    with torch.no_grad():
        train_loader_eval = DataLoader(
            train_dataset,
            batch_size=1000,
            shuffle=False,
        )
        test_loader_eval = DataLoader(
            test_dataset,
            batch_size=1000,
            shuffle=False,            
        )

        print("Extracting features for linear evaluation...")
        train_features, train_labels = [], []
        for batch in tqdm(train_loader_eval, "Train Features"):
            waveforms, labels = batch
            # Use the return_features=True flag to get representations before the projection head
            features, _ = model(waveforms.to(device), return_features=True)
            train_features.append(features.cpu())
            train_labels.append(labels)

        test_features, test_labels = [], []
        for batch in tqdm(test_loader_eval, "Test Features"):
            waveforms, labels = batch
            features, _ = model(waveforms.to(device), return_features=True)
            test_features.append(features.cpu())
            test_labels.append(labels)

    train_features = torch.cat(train_features, dim=0)
    test_features = torch.cat(test_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    # Save the features and labels
    torch.save(train_features, f"{save_dir}/train_features.pth")
    torch.save(test_features, f"{save_dir}/test_features.pth")
    torch.save(train_labels, f"{save_dir}/train_labels.pth")
    torch.save(test_labels, f"{save_dir}/test_labels.pth")
    
    # Perform linear evaluation
    print("\n" + "="*50)
    print("Starting Linear Evaluation")
    print("="*50)
    
    accuracy, nmi = compute_linear_eval(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        epochs=25,
        device=device
    )
    
    print(f"Final Results - Accuracy: {accuracy:.4f}, NMI: {nmi:.4f}")
    
    # Save evaluation results
    eval_results = {
        'accuracy': accuracy,
        'nmi': nmi
    }
    torch.save(eval_results, f"{save_dir}/linear_eval_results.pth")

if __name__ == "__main__":
    main()
