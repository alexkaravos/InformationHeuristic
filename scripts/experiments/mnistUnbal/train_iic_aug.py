import os
import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.datasets.datasets import mnist_unbalanced_dataset
from src.datasets.dataset_wrappers import DoubleAugmentedDataset,AugRepresentationDataset
from src.models import resnet
from src.utils import transforms
from src.models.clustering import MLP_dict
from src.clustering.training import train_paired_aug
from src.clustering.losses import IIC_loss

@hydra.main(config_path="../../../configs", 
            config_name="experiments/mnistUnbal/mnistUnbal_iic_aug",
            version_base=None)

def main(cfg: DictConfig):

    # Print the composed config
    print("--- Composed Hydra Config ---")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------------------")

    # Set device and seed
    device = cfg.device
    torch.manual_seed(cfg.seed)

   
    # Load dataset
    _,_, full_dataset = mnist_unbalanced_dataset(
        data_folder=cfg.datasets.data_folder,
        normalize=False,
        balance_settings=0,
        random_seed=cfg.seed
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
        batch_size=cfg.clustering.batch_size,
        shuffle=False,
        num_workers=14
    )

    #get the directory to the representation model
    representation_model_dir = cfg.representation_model_dir

    #load the model args
    model_args = torch.load(f"{representation_model_dir}/model_args.pt")
    representation_model = resnet.ResNetSSL(**model_args)
    representation_model.load_state_dict(torch.load(f"{representation_model_dir}/model.pt"))

    backbone  = representation_model.backbone
    backbone.to(device)
    backbone.eval()
    #turn gradients off
    for param in backbone.parameters():
        param.requires_grad = False

    #grab 10 augmentated representations of each sample + originals
    
    augmentations = []


    print("Getting augmentation representations")
    with torch.no_grad():
        for i in tqdm(range(cfg.clustering.num_aug_copies)):
            originals = [] #gets overwritten but thats fine
            labels = []
            aug1 = []
            aug2 = []
            
            for (x1,x2,x,y) in train_loader:
                x1 = x1.to(device)
                x2 = x2.to(device)
                x = x.to(device)
                
                z1 = backbone(x1).cpu().reshape(x1.shape[0],-1)
                z2 = backbone(x2).cpu().reshape(x2.shape[0],-1)
                z = backbone(x).cpu().reshape(x.shape[0],-1)

                originals.append(z)
                labels.append(y)
                aug1.append(z1)
                aug2.append(z2)
                
            originals = torch.cat(originals,dim=0)
            labels = torch.cat(labels,dim=0)
            aug1 = torch.cat(aug1,dim=0)
            aug2 = torch.cat(aug2,dim=0)
            
            augmentations.append(aug1)
            augmentations.append(aug2)

    augmentations = torch.stack(augmentations,dim=1) # dataset_len,2,10,repr_dim
    print(augmentations.shape)
    #create new dataloader
    representation_dataset = AugRepresentationDataset(
        augmentations=augmentations,
        originals=originals,
        labels=labels,
        num_augs_return=2
    )
    representation_dataloader = DataLoader(
        representation_dataset,
        batch_size=cfg.clustering.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    

    k_range = range(cfg.models.k_start,cfg.models.k_end,cfg.models.k_step)
    lamb_range = range(cfg.models.lamb_start,cfg.models.lamb_end,cfg.models.lamb_step)
    clustering_config = {
        "K_range": k_range,  
        "lamb_range": lamb_range,
        "num_copies": cfg.models.num_copies,
        "lamb_factor": cfg.models.lamb_factor,
        "num_hidden_blocks": cfg.models.num_blocks,
        "input_dim": cfg.models.input_dim,
        "hidden_dim": cfg.models.hidden_dim,
        "dropout_rate": cfg.models.dropout_rate
    }

    #set up the clustering model
    clustering_mesh = MLP_dict(**clustering_config).to(device)

    #set up the savining directory
    save_dir = os.path.join(cfg.save_dir, f"{cfg.save_name}_{cfg.representation_model_id}")
    os.makedirs(save_dir,exist_ok=True)


    #set up the optimizer
    optimizer = optim.Adam(clustering_mesh.parameters(),lr=cfg.clustering.lr)
    if cfg.clustering.scheduler == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.clustering.gamma)
    else:
        scheduler = None

    #train the clustering model
    criterion = IIC_loss()

    clustering_mesh,loss_history = train_paired_aug(
        model=clustering_mesh,
        dataloader=representation_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=cfg.clustering.epochs,
        device=device,
        save_dir=save_dir,
        criterion=criterion
    )

    #get the predictions for the mode

    predictions = {key:[] for key in clustering_mesh.keys}
    labels = []
    from sklearn.metrics import normalized_mutual_info_score as nmi
    import numpy as np
    representation_dataloader.shuffle = False

    with torch.no_grad():
        for (x1,x2,x,y) in representation_dataloader:
    
            x = x.to(device)
            p_dict = clustering_mesh(x)
            
            for key in clustering_mesh.keys:
                predictions[key].append(p_dict[key].cpu().argmax(dim=1))

            labels.append(y)
    
    labels = torch.cat(labels,dim=0)
    predictions = {key:torch.cat(predictions[key],dim=0) for key in clustering_mesh.keys}
    
    #save the predictions
    torch.save(predictions, f"{save_dir}/pred_dict.pt")
    
    #calculate the nmi
    nmi_scores = {key:nmi(predictions[key],labels) for key in clustering_mesh.keys}
    torch.save(nmi_scores, f"{save_dir}/nmi_scores.pt")

    #print the key of the max nmi
    print(f"max NMI: {np.max(list(nmi_scores.values()))} for {list(nmi_scores.keys())[np.argmax(list(nmi_scores.values()))]}")
    #save the config
    OmegaConf.save(cfg, f"{save_dir}/config.yaml")

if __name__ == "__main__":
    main()


    
    
    
    