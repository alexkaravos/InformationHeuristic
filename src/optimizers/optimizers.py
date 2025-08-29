
"""
Optimizer utilities for flexible creation based on config.
"""

import torch.optim as optim

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
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")
