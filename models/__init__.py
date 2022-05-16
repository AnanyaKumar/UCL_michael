import os
import importlib
from .simsiam import SimSiam
from .barlowtwins import BarlowTwins
from torchvision.models import resnet50
import torch
from .backbones import resnet18, resnet152
# I'm guessing they might use the torch resnet50 for tinyimagenet, so not importing resnet50 from backbones.

def get_backbone(backbone, dataset, castrate=True):
    backbone = eval(f"{backbone}()")
    if dataset == 'seq-cifar100':
        backbone.n_classes = 100
    elif dataset == 'seq-cifar10':
        backbone.n_classes = 10
    backbone.output_dim = backbone.fc.in_features
    if not castrate:
        backbone.fc = torch.nn.Identity()

    return backbone


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('/sailhome/msun415/UCL/models')
            if not model.find('__') > -1 and 'py' in model]

def get_model(args, device, len_train_loader, transform):
    loss = torch.nn.CrossEntropyLoss()
    if args.model.name == 'simsiam':
        backbone =  SimSiam(get_backbone(args.model.backbone, args.dataset.name, args.cl_default)).to(device)
        if args.model.proj_layers is not None:
            backbone.projector.set_layers(args.model.proj_layers)
    elif args.model.name == 'barlowtwins':
        backbone = BarlowTwins(get_backbone(args.model.backbone, args.dataset.name, args.cl_default), device).to(device)
        if args.model.proj_layers is not None:
            backbone.projector.set_layers(args.model.proj_layers)

    names = {}
    for model in get_all_models():
        mod = importlib.import_module('models.' + model)
        class_name = {x.lower():x for x in mod.__dir__()}[model.replace('_', '')]
        names[model] = getattr(mod, class_name)
    
    return names[args.model.cl_model](backbone, loss, args, len_train_loader, transform)

