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
import collections.abc

from datetime import datetime

import utils.io_utils as io_utils
from pathlib import Path


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

    @staticmethod
    def namespace_to_dict(namespace):
        is_namespace = lambda x: isinstance(x, Namespace) or isinstance(x, argparse.Namespace)
        return {
            k: namespace_to_dict(v) if is_namespace(v) else v
            for k, v in (vars(namespace) if is_namespace(namespace) else namespace).items()
        }
        

def update_args(args, key, value):
    args_update = deepcopy(args)
    setattr(args_update, key, value)
    return args_update

def init_args(args_dict):
    """Init argparse from dictionary."""
    return Namespace(args_dict)

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

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def fill_default_value(d, k, v):
    # v can be a dictionary, so need to update recursively
    new_d = {k: v}
    d = update(new_d, d) # this way, ensure config arguments isn't overridden
    return d

def make_checkpoints_dir(log_dir):
    checkpoints_dir = log_dir + '/checkpoints'
    checkpoints_dir = Path(checkpoints_dir).resolve().expanduser()
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)
    os.makedirs(checkpoints_dir)
    return checkpoints_dir

def populate_defaults(config):
    config = fill_default_value(config, 'project_name', 'continual_learning')
    config = fill_default_value(config, 'debug', False)
    config = fill_default_value(config, 'debug_subset_size', 8)
    config = fill_default_value(config, 'download', False)
    config = fill_default_value(config, 'data_dir', os.getenv('DATA'))
    config = fill_default_value(config, 'log_dir', os.getenv('LOG'))
    # config = # fill_default_value(config, 'ckpt_dir', os.getenv('CHECKPOINT'))
    # config = # fill_default_value(config, 'ckpt_dir_1', os.getenv('CHECKPOINT'))
    config = fill_default_value(config, 'device', 'cuda'  if torch.cuda.is_available() else 'cpu')
    config = fill_default_value(config, 'eval_from', None)
    config = fill_default_value(config, 'hide_progress', False)
    config = fill_default_value(config, 'cl_default', False)
    config = fill_default_value(config, 'last', False)
    config = fill_default_value(config, 'debug_lpft', False)
    config = fill_default_value(config, 'lpft', False)
    config = fill_default_value(config, 'rerun', True)
    config = fill_default_value(config, 'is_eval_script', False)
    config = fill_default_value(config, 'probe_train_frac', 1.0)
    config = fill_default_value(config, 'save_model', False)
    config = fill_default_value(config, 'save_as_orig', False)
    config = fill_default_value(config, 'validation', False)
    config = fill_default_value(config, 'ood_eval', False)
    config = fill_default_value(config, 'aug_kwargs', {
        'no_train_augs': False,
        'name': config['model']['name'],
        'image_size': config['dataset']['image_size'],
        'cl_default': config['cl_default'],
        'scale': 0.2,
    })
    config = fill_default_value(config, 'model', {
        'use_group_norm': False,
        'group_norm_num_groups': 32
    })
    config = fill_default_value(config, 'train', {
        'sklearn_lp_probe': False
    })
    return config


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
    config = populate_defaults(config)    
    io_utils.update_config(unparsed, config)
    args = Namespace(config)
    def enforce_arg(args, arg_name):
        if arg_name not in args.__dict__:
            raise ValueError('Must specify {} in config or command line.'.format(arg_name))
    enforce_arg(args, 'group_name')
    enforce_arg(args, 'run_name')
    enforce_arg(args, 'log_dir')
    enforce_arg(args, 'tmp_par_ckp_dir')

    args.aug_kwargs = vars(args.aug_kwargs)

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

    assert not None in [args.log_dir] # used to include args.data_dir too, but assume each dataset finds data location manually

    # args.log_dir = os.path.join(args.log_dir, 'in-progress_'+datetime.now().strftime('%m%d%H%M%S_')+args.name)
    args.log_dir = os.path.join(args.log_dir, args.group_name, args.run_name)
    if os.path.isdir(args.log_dir) and args.rerun and not args.is_eval_script:
        print("Removed old run directory at {}.".format(args.log_dir))
        shutil.rmtree(args.log_dir)

    if not args.is_eval_script:
        os.makedirs(args.log_dir, exist_ok=False)
        print(f'creating file {args.log_dir}')
        
    args.ckpt_dir = os.path.join(args.log_dir, 'checkpoints')

    if os.path.isdir(args.ckpt_dir) and args.rerun and not args.is_eval_script:
        shutil.rmtree(args.ckpt_dir)

    if not args.is_eval_script:
        if args.tmp_par_ckp_dir is not None:
            checkpoints_dir = make_checkpoints_dir(args.tmp_par_ckp_dir)
        else:
            checkpoints_dir = make_checkpoints_dir(log_dir)

        # shutil.copy2(cl_args.config_file, args.log_dir)
    else:
        if args.tmp_par_ckp_dir is not None:
            checkpoints_dir = args.tmp_par_ckp_dir + '/checkpoints'
        else:
            checkpoints_dir = log_dir + '/checkpoints'

    if not args.is_eval_script:
        config_json = args.log_dir + '/config.json'
        args_dict = namespace_to_dict(copy.deepcopy(args))
        print(args_dict)
        with open(config_json, 'w') as f:
            json.dump(args_dict, f)
    
    set_deterministic(args.seed)   
       
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

    return args, checkpoints_dir
