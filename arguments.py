import argparse
import os
import torch
import copy

import numpy as np
import torch
import random

import re 
import yaml
import json

import shutil
import warnings
from copy import deepcopy

from datetime import datetime

import utils.io_utils as io_utils

class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value
    
    def __getattr__(self, attribute):

        raise AttributeError(f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")

def update_args(args, key, value):
    args_update = deepcopy(args)
    setattr(args_update, key, value)
    return args_update

def init_args(args_dict):
    """Init argparse from dictionary."""
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.__dict__ = args_dict
    return args

def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 

def fill_default_value(d, k, v):
    if k not in d:
        d[k] = v

def populate_defaults(config):
    fill_default_value(config, 'project_name', 'continual_learning')
    fill_default_value(config, 'debug', False)
    fill_default_value(config, 'debug_subset_size', 8)
    fill_default_value(config, 'download', False)
    fill_default_value(config, 'data_dir', os.getenv('DATA'))
    fill_default_value(config, 'log_dir', os.getenv('LOG'))
    # fill_default_value(config, 'ckpt_dir', os.getenv('CHECKPOINT'))
    # fill_default_value(config, 'ckpt_dir_1', os.getenv('CHECKPOINT'))
    fill_default_value(config, 'device', 'cuda'  if torch.cuda.is_available() else 'cpu')
    fill_default_value(config, 'eval_from', None)
    fill_default_value(config, 'hide_progress', False)
    fill_default_value(config, 'cl_default', False)
    fill_default_value(config, 'last', False)
    fill_default_value(config, 'debug_lpft', False)
    fill_default_value(config, 'lpft', False)
    fill_default_value(config, 'save_as_orig', False)
    fill_default_value(config, 'validation', False)
    fill_default_value(config, 'ood_eval', False)

def namespace_to_dict(obj):
    # Recursively transform dict of namespaces into a dict.
    cur_dict = obj.__dict__
    for k in cur_dict:
        if isinstance(cur_dict[k], Namespace):
            cur_dict[k] = namespace_to_dict(cur_dict[k])
    return cur_dict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', required=True, type=str, help="xxx.yaml")
    cl_args, unparsed = parser.parse_known_args()

    with open(cl_args.config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    populate_defaults(config)
    io_utils.update_config(unparsed, config)
    args = Namespace(config)
    def enforce_arg(args, arg_name):
        if arg_name not in args.__dict__:
            raise ValueError('Must specify {} in config or command line.'.format(arg_name))
    enforce_arg(args, 'group_name')
    enforce_arg(args, 'run_name')
    enforce_arg(args, 'log_dir')

        # for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
        #     vars(args)[key] = value

    if args.debug:
        if args.train: 
            args.train.batch_size = 256
            args.train.num_epochs = 2
            args.train.stop_at_epoch = 2
            if args.lpft:
                assert args.debug_lpft, "cover this case"
                args.train.num_lp_epochs = 1
        if args.eval: 
            args.eval.batch_size = 2
            args.eval.num_epochs = 1 # train only one epoch
        args.dataset.num_workers = 0


    assert not None in [args.log_dir, args.data_dir]

    # args.log_dir = os.path.join(args.log_dir, 'in-progress_'+datetime.now().strftime('%m%d%H%M%S_')+args.name)
    args.log_dir = os.path.join(args.log_dir, args.group_name, args.run_name)
    if os.path.isdir(args.log_dir):
        print("Removed old run directory at {}.".format(args.log_dir))
        shutil.rmtree(args.log_dir)
    os.makedirs(args.log_dir, exist_ok=False)
    print(f'creating file {args.log_dir}')
    args.ckpt_dir = os.path.join(args.log_dir, 'checkpoints')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # shutil.copy2(cl_args.config_file, args.log_dir)

    config_json = args.log_dir + '/config.json'
    args_dict = namespace_to_dict(copy.deepcopy(args))
    print(args_dict)
    with open(config_json, 'w') as f:
        json.dump(args_dict, f)
    
    set_deterministic(args.seed)


    vars(args)['aug_kwargs'] = {
        'name':args.model.name,
        'image_size': args.dataset.image_size,
        'cl_default': args.cl_default,
        'scale': 0.2,
    }
    vars(args)['dataset_kwargs'] = {
        # 'name':args.model.name,
        # 'image_size': args.dataset.image_size,
        'dataset':args.dataset.name,
        'data_dir': args.data_dir,
        'download':args.download,
        'debug_subset_size': args.debug_subset_size if args.debug else None,
        # 'drop_last': True,
        # 'pin_memory': True,
        # 'num_workers': args.dataset.num_workers,
    }
    vars(args)['dataloader_kwargs'] = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.dataset.num_workers,
    }

    return args
