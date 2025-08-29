import os
import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.datasets.datasets import cifar100_dataset
from src.datasets.dataset_wrappers import DoubleAugmentedDataset
from src.models.resnet import ResNetSSL  
from src.representations.losses import BarlowTwinsLoss
from src.representations.training import train_contrastive
from src.utils import transforms
from src.evaluation.representations import compute_linear_eval,compute_knn_accuracy
from src.optimizers.optimizers import get_optimizer


@hydra.main(config_path="../../../configs", config_name="experiments/cifar100/cifar100_barlowtwin", version_base=None)

def main(cfg: DictConfig):

    # Print the composed config
    print("--- Composed Hydra Config ---")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------------------")

    # Set device and seed
    device = cfg.device
    torch.manual_seed(cfg.seed)
    
    # Load dataset
    train_dataset,test_dataset, full_dataset = cifar100_dataset(
        data_folder=cfg.datasets.data_folder,
        normalize=False 
    )

    # Get transform from transforms.py by string name
    transform = getattr(transforms, cfg.augmentation)
    
    augmented_dataset = DoubleAugmentedDataset(
        dataset=train_dataset,
        augmentation=transform,
        add_normalize=True
    )
    
    # Create dataloader
    train_loader = DataLoader(
        augmented_dataset,
        batch_size=cfg.representations.batch_size,
        shuffle=True,
        num_workers=8
    )


    #make sure the saving directory exists
    save_dir = os.path.join(cfg.save_dir, f"barlowtwin_{cfg.models.arch}_{cfg.representations.lambda_param*1000:.2f}")
    os.makedirs(save_dir, exist_ok=True)     
    
    model_args = {
        "arch": cfg.models.arch,
        "in_channels": cfg.models.in_channels,
        "proj_dim": cfg.models.proj_dim,
        "small_images": cfg.models.small_images
    }
    torch.save(model_args, f"{save_dir}/model_args.pt")
    
    
    # Create model
    model = ResNetSSL(
        arch=cfg.models.arch,
        in_channels=cfg.models.in_channels,
        proj_dim=cfg.models.proj_dim,
        small_images=cfg.models.small_images,
        final_batchnorm=cfg.models.final_batchnorm if hasattr(cfg.models, 'final_batchnorm') else False
    )

    # Create optimizer using new utility
    optimizer = get_optimizer(model, cfg.representations)
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
    print(f"Lambda param: {cfg.representations.lambda_param}")
    print(f"Augmentation: {cfg.augmentation}")
    print(f"Save directory: {cfg.save_dir}")
    
    model.to(device)
    model.train()
    model, loss_history = train_contrastive(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=cfg.representations.epochs,
        device=device,
        save_dir=save_dir,

    )
    
    # Save final model
    torch.save(model.state_dict(), f"{save_dir}/model.pt")
    torch.save(loss_history, f"{save_dir}/loss_history.pt")
    #save the model configs

    # get the full representations from the model
    
    #refetch the dataset but this time we want to return the normalized image
    model.eval()
    train_dataset,test_dataset, full_dataset = cifar100_dataset(
        data_folder=cfg.datasets.data_folder,
        normalize=True
    )
    #get the full representations from the model

    with torch.no_grad():
        train_dataloader = DataLoader(
            train_dataset,
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
            features, _ = model(batch[0].to(device), return_features=True)
            train_features.append(features)
            train_labels.append(batch[1])
        for batch in test_dataloader:
            features, _ = model(batch[0].to(device), return_features=True)
            test_features.append(features)
            test_labels.append(batch[1])

    train_features = torch.cat(train_features, dim=0).cpu()
    test_features = torch.cat(test_features, dim=0).cpu()
    train_labels = torch.cat(train_labels, dim=0).cpu()
    test_labels = torch.cat(test_labels, dim=0).cpu()
    #save the features and labels
    
    torch.save(train_features, f"{save_dir}/train_features.pt")
    torch.save(test_features, f"{save_dir}/test_features.pt")
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

    compute_knn_top1 = compute_knn_accuracy(
        test_features=test_features,
        test_labels=test_labels,
        train_features=train_features,
        train_labels=train_labels,
        k=1,
    )
    compute_knn_top5 = compute_knn_accuracy(
        test_features=test_features,
        test_labels=test_labels,
        train_features=train_features,
        train_labels=train_labels,
        k=5,
    )
    
    # Save evaluation results
    eval_results = {
        'accuracy': accuracy,
        'nmi': nmi,
        'knn_top1': compute_knn_top1,
        'knn_top5': compute_knn_top5
    }

    print(eval_results)
    torch.save(eval_results, f"{save_dir}/eval_results.pt")

if __name__ == "__main__":
    main()
