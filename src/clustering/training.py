from tqdm import tqdm
import torch

def train_paired_aug(model,
                     dataloader,
                     criterion,
                     optimizer,
                     scheduler,
                     epochs,
                     device,
                     save_dir,
                     internal_epoch_bar=False):
    """
    Train a clustering mesh with keys (lamb,K,idx)
    for an dataset with paired augmentations. The dataloader
    should return a tuple shape (xt1,xt2,x,y) where 
    xt1 and xt2 are paired augmentations of the same sample.
    Loss is then computed p1 = model(xt1), p2 = model(xt2) 
    loss +=criterion(p1[keys],p2[keys],lamb=model.lamb_factor**key[0])
    """

    #create the loss dictionary for each clustering head
    loss_dict = {key:[] for key in model.keys}

    progress_bar = tqdm(range(epochs))


    for epoch in progress_bar:

        temp_loss_dict = {key:[] for key in model.keys}

        if internal_epoch_bar:
            epoch_bar = tqdm(dataloader,desc=f"Epoch {epoch+1}/{epochs}")
        else:
            epoch_bar = dataloader

        for (x1,x2,x,y) in epoch_bar:

            x1 = x1.to(device)
            x2 = x2.to(device)
            
            p1_dict = model(x1)
            p2_dict = model(x2)

            net_loss = 0
            for key in model.keys:
                key_loss = criterion(p1_dict[key],p2_dict[key],lamb=model.lamb_factor**key[0])
                net_loss += key_loss
                temp_loss_dict[key].append(key_loss.item())

            optimizer.zero_grad()
            net_loss.backward()
            optimizer.step()

        
        

        for key in model.keys:
            loss_dict[key].append(torch.mean(torch.tensor(temp_loss_dict[key])))

        progress_bar.set_description(f"Epoch {epoch+1}/{epochs} - Loss: {torch.mean(torch.tensor([losses[-1] for losses in loss_dict.values()]))}")
        torch.save(model.state_dict(), f"{save_dir}/model.pt")
        torch.save(loss_dict, f"{save_dir}/loss_dict.pt")
        
    return model, loss_dict



def train_alex_loss(model,
                     dataloader,
                     criterion,
                     optimizer,
                     scheduler,
                     epochs,
                     device,
                     save_dir,
                     internal_epoch_bar=False):
    """
    Train a clustering mesh with keys (lamb,K,idx)
    for an dataset with paired augmentations. The dataloader
    should return a tuple shape (xt1,xt2,x,y) where 
    xt1 and xt2 are paired augmentations of the same sample.
    Loss is then computed p1 = model(xt1), p2 = model(xt2) 
    loss +=criterion(p1[keys],p2[keys],lamb=model.lamb_factor**key[0])
    """

    #create the loss dictionary for each clustering head
    loss_dict = {key:[] for key in model.keys}

    progress_bar = tqdm(range(epochs))


    for epoch in progress_bar:

        temp_loss_dict = {key:[] for key in model.keys}

        if internal_epoch_bar:
            epoch_bar = tqdm(dataloader,desc=f"Epoch {epoch+1}/{epochs}")
        else:
            epoch_bar = dataloader

        for (x1,x2,x,y) in epoch_bar:

            x1 = x1.to(device)
            x2 = x2.to(device)
            
            p1_dict = model(x1)
            p2_dict = model(x2)

            net_loss = 0
            
            for key in model.keys:
                key_loss = criterion(p1_dict[key],p2_dict[key])
                net_loss += key_loss
                temp_loss_dict[key].append(key_loss.item())

            optimizer.zero_grad()
            net_loss.backward()
            optimizer.step()

        
        

        for key in model.keys:
            loss_dict[key].append(torch.mean(torch.tensor(temp_loss_dict[key])))

        progress_bar.set_description(f"Epoch {epoch+1}/{epochs} - Loss: {torch.mean(torch.tensor([losses[-1] for losses in loss_dict.values()]))}")
        torch.save(model.state_dict(), f"{save_dir}/model.pt")
        torch.save(loss_dict, f"{save_dir}/loss_dict.pt")
        
    return model, loss_dict