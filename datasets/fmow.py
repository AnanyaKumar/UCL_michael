# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch.nn.functional as F
from datasets.seq_tinyimagenet import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
import torch
from augmentations import *
from PIL import Image
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from copy import deepcopy
import socket

_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]

class FMOW(ContinualDataset):

    NAME = 'fmow'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 62    
    HEAD_DIM = 62

    REGION_ORDER = [1, 3, 0, 4, 2, 5]
    # REGION_ORDER = [5,2,4,0,3,1]
    YEAR_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    TASK_DEFINITION = "region"
    N_TASKS = 6

    def __init__(self, args):
        dataset = get_dataset(dataset="fmow", root_dir=f"/{socket.gethostname().split('.')[0]}/scr0/msun415/", download=False)
        transform = ImageBaseTransform()
        test_transform = ImageBaseTransformSingle()
        self.train_data = dataset.get_subset("train", transform=transform)
        self.memory_data = dataset.get_subset("train", transform=test_transform)
        self.test_data = dataset.get_subset("id_val", transform=test_transform)

        metadataset = dataset.metadata.iloc[np.argwhere((dataset.metadata['split'] != 'seq').values).flatten(), :]

        self.train_dataset = metadataset.iloc[self.train_data.indices]
        self.memory_dataset = metadataset.iloc[self.memory_data.indices]
        self.test_dataset = metadataset.iloc[self.test_data.indices]       

        self.train_data.targets = self.train_dataset.y.values
        self.memory_data.targets = self.memory_dataset.y.values
        self.test_data.targets = self.test_dataset.y.values

        super().__init__(args)


    def get_transform(self, args):
        not_aug_transform = transforms.Compose([transforms.ToTensor()])

        transform = transforms.Compose([
            transforms.ToTensor(),
            default_normalization
        ])
        return transform

   
    def get_data_loaders(self, args, divide_tasks=True):
        task_train_data = deepcopy(self.train_data)
        mask = self.train_dataset[self.TASK_DEFINITION] == (self.REGION_ORDER if self.TASK_DEFINITION == "region" else self.YEAR_ORDER)[self.i] if divide_tasks else np.full((len(self.train_dataset),), True)
        indices = self.train_dataset[mask].index.values
        # if args.debug: indices = np.random.permutation(indices)[:args.train.batch_size]
        task_train_data.indices = indices                
        task_train_data.targets = self.train_dataset[mask].y.values

        task_memory_data = deepcopy(self.memory_data)
        mask = self.memory_dataset[self.TASK_DEFINITION] == (self.REGION_ORDER if self.TASK_DEFINITION == "region" else self.YEAR_ORDER)[self.i]
        indices = self.memory_dataset[mask].index.values
        # if args.debug: indices = np.random.permutation(indices)[:args.train.batch_size]
        task_memory_data.indices = indices                
        task_memory_data.targets = self.memory_dataset[mask].y.values

        task_test_data = deepcopy(self.test_data)
        mask = self.test_dataset[self.TASK_DEFINITION] == (self.REGION_ORDER if self.TASK_DEFINITION == "region" else self.YEAR_ORDER)[self.i]
        indices = self.test_dataset[mask].index.values
        # if args.debug: indices = np.random.permutation(indices)[:args.train.batch_size]
        task_test_data.indices = indices                
        task_test_data.targets = self.test_dataset[mask].y.values

        num_workers = 0 if args.debug else 16

        train_loader = get_train_loader("standard", task_train_data, batch_size=self.args.train.batch_size, num_workers=num_workers)
        memory_loader = get_eval_loader("standard", task_memory_data, batch_size=self.args.train.batch_size//8, num_workers=num_workers)
        test_loader = get_eval_loader("standard", task_test_data, batch_size=self.args.train.batch_size//8, num_workers=num_workers)

        self.test_loaders.append(test_loader)
        self.train_loaders.append(train_loader)
        self.memory_loaders.append(memory_loader)
        self.train_loader = train_loader
        self.i += 1
        return train_loader, memory_loader, test_loader
    