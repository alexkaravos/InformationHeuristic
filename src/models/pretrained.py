"""
Functions and classes for working from the pretrained models. We load in models from

MixedBarlow: github.com/facebookresearch/mixbarlowtwins
MV-MR: 

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50


class mixedBarlow(nn.Module):
    def __init__(self, feature_dim=128, dataset='cifar10', arch='resnet50'):
        super(mixedBarlow, self).__init__()

        self.f = []
        if arch == 'resnet18':
            temp_model = resnet18().named_children()
            embedding_size = 512
        elif arch == 'resnet50':
            temp_model = resnet50().named_children()
            embedding_size = 2048
        else:
            raise NotImplementedError
        
        for name, module in temp_model:
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if dataset == 'cifar10' or dataset == 'cifar100':
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.f.append(module)
            elif dataset == 'tiny_imagenet' or dataset == 'stl10':
                if not isinstance(module, nn.Linear):
                    self.f.append(module)
            else:
                raise NotImplementedError
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(embedding_size, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
    
def get_mixedbarlow_encoder(weight_dir, feature_dim=1024, arch='resnet50', dataset='cifar10'):
    # Initialize the model
    model = mixedBarlow(feature_dim=feature_dim, dataset=dataset, arch=arch)
    
    # Load the pretrained weights, ignoring unexpected keys
    state_dict = torch.load(weight_dir, map_location=torch.device('cpu'), weights_only=True)
    # Filter out unexpected keys (e.g., total_ops, total_params)
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)
    
    # Return the ResNet backbone
    # we return encoder f
    return model.f


