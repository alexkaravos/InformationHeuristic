import torch
import torch.optim as optim
import math
from torch.optim.lr_scheduler import _LRScheduler

#Implementation of the LARS optimizer
#Fetched from the mx-bt repo
class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    A learning rate scheduler that combines a linear warmup phase with a
    cosine annealing decay phase.

    This scheduler is designed to be called at the end of each epoch.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        total_epochs (int): The total number of epochs for training.
        warmup_epochs (int): The number of epochs for the linear warmup phase.
        last_epoch (int, optional): The index of the last epoch. Defaults to -1.
    """
    def __init__(self, optimizer, total_epochs, warmup_epochs, last_epoch=-1, verbose=False):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """
        Calculates the learning rate for the current epoch based on the schedule.
        """
        # The self.last_epoch attribute is automatically incremented by the step() method
        # of the parent class. It starts at -1 and becomes 0 on the first call to step().
        current_epoch = self.last_epoch

        if current_epoch < self.warmup_epochs:
            # Linear warmup phase
            # Increase LR from a small value to the base LR.
            # The +1 ensures that the learning rate for epoch 0 is not zero.
            lr_scale = (current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing phase
            effective_epoch = current_epoch - self.warmup_epochs
            total_decay_epochs = self.total_epochs - self.warmup_epochs
            
            # Calculate the cosine decay factor
            cosine_decay = 0.5 * (1 + math.cos(math.pi * effective_epoch / total_decay_epochs))
            lr_scale = cosine_decay
        
        return [base_lr * lr_scale for base_lr in self.base_lrs]


"""
Optimizer utilities for flexible creation based on config.
"""

def get_optimizer(model, cfg):
    """
    Creates optimizer based on config parameters.
    """
    if cfg.optimizer == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == "adam":
        return optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay if hasattr(cfg, 'weight_decay') else 0
        )
    elif cfg.optimizer == "lars":
        return LARS(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            momentum=cfg.momentum,
            eta=cfg.eta
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")
