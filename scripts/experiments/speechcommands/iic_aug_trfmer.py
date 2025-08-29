import os
import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.datasets.datasets import speechcommands_dataset
from src.datasets.dataset_wrappers import DoubleAugmentedDataset,AugRepresentationDataset
from src.models import resnet
from src.utils import transforms
from src.models.wav2vec import Wav2VecEncoder
from src.models.clustering import Transformer_dict,Connected_Clusterer
from src.clustering.training import train_paired_aug
from src.clustering.losses import IIC_loss

@hydra.main(config_path="../../../configs", config_name="experiments/speechcommands/speechcommands_iic_aug", version_base=None)

def main(cfg: DictConfig):

    # Print the composed config
    print(OmegaConf.to_yaml(cfg))

    # Get the speech commands dataset
    _,_,full_dataset = speechcommands_dataset(data_folder=cfg.data_folder)

    transform = getattr(transforms, cfg.transform)

    augmented_dataset = DoubleAugmentedDataset(
        dataset = full_dataset,
        augmentation = transform,
        add_normalize=False
    )

    #setup the model
    augmented_dataloader = DataLoader(augmented_dataset,batch_size=cfg.clustering.batch_size,shuffle=True)

    #load our pretrained representation model (wav2vec2)
    backbone_model = Wav2VecEncoder()

    #setup our clustering networks
    clustering_model = Transformer_dict(
        K_range=cfg.clustering.k_range,
        lamb_range=cfg.clustering.lamb_range,
        num_copies=cfg.clustering.num_copies,
        input_dim=cfg.clustering.input_dim,
        hidden_dim=cfg.clustering.hidden_dim,
        dim_feedforward=cfg.clustering.dim_feedforward,
        num_layers=cfg.clustering.num_layers,
        dropout_rate=cfg.clustering.dropout_rate
    )

    #setup our clusterer
    model = Connected_Clusterer(
        backbone=backbone_model,
        clusterer=clustering_model
    )
    
    # Set device and seed
    device = cfg.device
    torch.manual_seed(cfg.seed)
    
    model.to(device)
    
    # Set up the saving directory
    save_dir = os.path.join(cfg.save_dir, f"{cfg.save_name}")
    os.makedirs(save_dir, exist_ok=True)

    # Set up the optimizer
    #only pass the clustering parameters (we dont want to train the backbone)
    optimizer = optim.Adam(model.clusterer.parameters(), lr=cfg.clustering.lr)
    if cfg.clustering.scheduler == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.clustering.gamma)
    else:
        scheduler = None

    # Train the clustering model
    criterion = IIC_loss()

    model, loss_history = train_paired_aug(
        model=model,
        dataloader=augmented_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=cfg.clustering.epochs,
        device=device,
        save_dir=save_dir,
        criterion=criterion
    )

    # Get the predictions for the model
    predictions = {key: [] for key in model.keys}
    labels = []
    from sklearn.metrics import normalized_mutual_info_score as nmi
    import numpy as np
    augmented_dataloader.shuffle = False

    with torch.no_grad():
        for (x1, x2, x, y) in augmented_dataloader:
    
            x = x.to(device)
            p_dict = model(x)
            
            for key in model.keys:
                predictions[key].append(p_dict[key].cpu().argmax(dim=1))
    
            labels.append(y)
    
    labels = torch.cat(labels, dim=0)
    predictions = {key: torch.cat(predictions[key], dim=0) for key in model.keys}
    
    # Save the predictions
    torch.save(predictions, f"{save_dir}/pred_dict.pt")
    
    # Calculate the nmi
    nmi_scores = {key: nmi(predictions[key], labels) for key in model.keys}
    torch.save(nmi_scores, f"{save_dir}/nmi_scores.pt")

    print(f"max NMI: {np.max(list(nmi_scores.values()))}")
    # Save the config
    OmegaConf.save(cfg, f"{save_dir}/config.yaml")
    


if __name__ == "__main__":
    main()  