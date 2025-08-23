import os
import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.datasets.datasets import cifar10_dataset
from src.datasets.dataset_wrappers import DoubleAugmentedDataset
from src.models.resnet import ResNetSSL
from src.representations.losses import NTXentLoss
from src.representations.training import train_contrastive
from src.utils import transforms
from src.evaluation.representations import compute_linear_eval


@hydra.main(config_path="../../../configs", config_name="experiments/cifar10/cifar10_simclr", version_base=None)
def main(cfg: DictConfig):
    # Flatten the config
    cfg = cfg.experiments.cifar10

    # Set device and seed
    device = cfg.device
    torch.manual_seed(cfg.seed)
    
    # Load dataset
    train_dataset,test_dataset, full_dataset = cifar10_dataset(
        data_folder=cfg.datasets.data_folder,
        normalize=False 
    )

    # Get transform from transforms.py by string name
    transform = getattr(transforms, cfg.augmentation)
    
    augmented_dataset = DoubleAugmentedDataset(
        dataset=full_dataset,
        augmentation=transform,
        add_normalize=True
    )
    
    # Create dataloader
    train_loader = DataLoader(
        augmented_dataset,
        batch_size=cfg.representations.batch_size,
        shuffle=True,
        num_workers=10
    )
    
    # Create model
    model = ResNetSSL(
        arch=cfg.models.arch,
        in_channels=cfg.models.in_channels,
        proj_dim=cfg.models.proj_dim,
        small_images=cfg.models.small_images
    )

        # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg.representations.lr)
    if cfg.representations.scheduler == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.representations.gamma)
    else:
        scheduler = None
    
    # Create loss function
    criterion = NTXentLoss(temperature=cfg.representations.temperature)
    
    # Train model
    print(f"Starting training for {cfg.experiment_name}")
    print(f"Device: {device}")
    print(f"Epochs: {cfg.representations.epochs}")
    print(f"Batch size: {cfg.representations.batch_size}")
    print(f"Temperature: {cfg.representations.temperature}")
    print(f"Augmentation: {cfg.augmentation}")
    print(f"Save directory: {cfg.save_dir}")
    
    #make sure the saving directory exists
    #we need to make our own save directory within this based on the temperature,
    #and model type
    save_dir = os.path.join(cfg.save_dir, f"simclr_{cfg.models.arch}_{cfg.representations.temperature}")
    os.makedirs(save_dir, exist_ok=True)    
    model.to(device)
    model.train()
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
    
    # Save final model
    # get the full representations from the model
    
    #refetch the dataset but this time we want to return the normalized image
    model.eval()
    train_dataset,test_dataset, full_dataset = cifar10_dataset(
        data_folder=cfg.datasets.data_folder,
        normalize=True
    )
    #get the full representations from the model

    with torch.no_grad():
        train_dataloader = DataLoader(
            full_dataset,
            batch_size=1000,
            shuffle=False,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1000,
            shuffle=False,            
        )
        train_features = []
        test_features = []
        train_labels = []
        test_labels = []
        for batch in train_dataloader:
            features, _ = model(batch[0], return_features=True)
            train_features.append(features)
            train_labels.append(batch[1])
        for batch in test_dataloader:
            features, _ = model(batch[0], return_features=True)
            test_features.append(features)
            test_labels.append(batch[1])

    train_features = torch.cat(train_features, dim=0)
    test_features = torch.cat(test_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    #save the features and labels
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
    
    # Save evaluation results
    eval_results = {
        'accuracy': accuracy,
        'nmi': nmi
    }
    torch.save(eval_results, f"{save_dir}/linear_eval_results.pth")

if __name__ == "__main__":
    main()
