import os
import importlib
from .simsiam import SimSiam
from .barlowtwins import BarlowTwins
import torch
from types import FunctionType as ftype
from .backbones import resnet18, resnet50, resnet101, resnet152
import torch.nn as nn

swav = torch.hub.load('facebookresearch/swav:main', 'resnet50')
densenet121 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)

def get_head(backbone):
    if hasattr(backbone, "fc"):
        return backbone.fc 
    else:
        return backbone.classifier

def get_features(model, inputs):
    if hasattr(model, "embed"):
        return model.embed(inputs)[0] 
    else:
        return model(inputs, return_features=True)

def get_num_params(model, is_trainable = None):
    """Get number of parameters of the model, specified by 'None': all parameters;
    True: trainable parameters; False: non-trainable parameters.
    """
    num_params = 0
    for param in list(model.parameters()):
        nn=1
        if is_trainable is None \
            or (is_trainable is True and param.requires_grad is True) \
            or (is_trainable is False and param.requires_grad is False):
            for s in list(param.size()):
                nn = nn * s
            num_params += nn
    return num_params

def get_backbone(backbone, dataset, castrate=True, args=None):  
    if args is not None and 'use_group_norm' in args.model.__dict__:
        norm_layer = lambda c: torch.nn.GroupNorm(args.model.group_norm_num_groups, c)
        backbone = eval(f"{backbone}(norm_layer=norm_layer)")
    else:            
        backbone = eval(f"{backbone}")
        if type(backbone) == ftype:
            backbone = backbone()
    
    # reconcile head dimension from loading pretrained weights or default initialization
    head_attr_name = "fc" if hasattr(backbone, "fc") else "classifier"
    backbone.n_classes = dataset.HEAD_DIM
    if getattr(backbone, head_attr_name).out_features != dataset.HEAD_DIM:
        print(f"reinitializing head to dimension {dataset.HEAD_DIM}")
        in_features = getattr(backbone, head_attr_name).in_features
        setattr(backbone, head_attr_name, nn.Linear(in_features, dataset.HEAD_DIM))
    
    backbone.output_dim = (get_head(backbone)).in_features
    if not castrate:
        if hasattr(backbone, "fc"): backbone.fc = torch.nn.Identity()
        else: backbone.classifier = torch.nn.Identity()

    return backbone


def get_all_models():
    if os.path.exists('models'):
        models_dir = os.listdir('models')
    else:
        models_dir = os.listdir('../models')
    return [model.split('.')[0] for model in models_dir
            if not model.find('__') > -1 and 'py' in model and 'swp' not in model]

def get_model(args, device, len_train_loader, dataset, transform):
    loss = torch.nn.CrossEntropyLoss()
    if args.model.name == 'simsiam':
        backbone =  SimSiam(get_backbone(args.model.backbone, dataset, args.cl_default, args)).to(device)
        for class_ in [resnet18, resnet50, resnet101, resnet152, densenet121, swav]:            
            backbone_ = class_() if type(class_) == ftype else class_
            print(f"{backbone_.__class__} has {get_num_params(backbone_)} params")
        if args.model.proj_layers is not None:
            backbone.projector.set_layers(args.model.proj_layers)
    elif args.model.name == 'barlowtwins':
        backbone = BarlowTwins(get_backbone(args.model.backbone, dataset, args.cl_default, args), device).to(device)
        if args.model.proj_layers is not None:
            backbone.projector.set_layers(args.model.proj_layers)

    names = {}
    print(get_all_models())
    for model in get_all_models():
        mod = importlib.import_module('models.' + model)
        class_name = {x.lower():x for x in mod.__dir__()}[model.replace('_', '')]
        names[model] = getattr(mod, class_name)
    
    return names[args.model.cl_model](backbone, loss, args, len_train_loader, transform)

