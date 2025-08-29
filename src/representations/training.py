"""
Training functions for representation learning. 
"""

import torch
from tqdm import tqdm


def train_contrastive(model, dataloader, criterion, optimizer, epochs, device, save_dir, scheduler=None,
                      save_every=None):
    
    """
    Train a model where the loss is computed between two samples batch[0] and batch[1],
    example: simclr, barlow twins, etc. 

    Args:
    """

    model.to(device)
    model.train()
    loss_history = []
    epoch_bar = tqdm(range(epochs), desc='Epochs')
    
    for epoch in epoch_bar:
        total_loss = 0
        num_batches = 0
        
        # Create progress bar for batches
        
        
        for batch_idx, batch in enumerate(dataloader):
            x1,x2 = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
        
            loss = criterion(model(x1), model(x2))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            
        
        if scheduler:
            scheduler.step()
            
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        if save_every is not None and (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), f'{save_dir}/ssl_model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), f'{save_dir}/ssl_model.pth')
        torch.save(loss_history, f'{save_dir}/loss_history.pth')
        epoch_bar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Epoch': f'{epoch+1}/{epochs}'})
        
        
        
    
    return model, loss_history