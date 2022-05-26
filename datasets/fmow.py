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
from augmentations import get_aug
from PIL import Image
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from copy import deepcopy


class FMOW(ContinualDataset):

    NAME = 'fmow'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 62
    N_TASKS = 6

    REGION_ORDER = [1, 3, 0, 4, 2, 5]

    def __init__(self, args):
        dataset = get_dataset(dataset="fmow", root_dir="/u/scr/nlp/wilds/data/", download=False)
        transform = get_aug(train=True, **args.aug_kwargs)
        test_transform = get_aug(train=False, train_classifier=False, **args.aug_kwargs)
        self.train_data = dataset.get_subset("train", transform=transform)
        self.val_data = dataset.get_subset("id_val", transform=test_transform)
        self.test_data = dataset.get_subset("id_test", transform=test_transform)

        metadataset = dataset.metadata.iloc[np.argwhere((dataset.metadata['split'] != 'seq').values).flatten(), :]
        self.train_dataset = metadataset.iloc[self.train_data.indices]
        self.val_dataset = metadataset.iloc[self.val_data.indices]
        self.test_dataset = metadataset.iloc[self.test_data.indices]            

        super().__init__(args)

    def get_transform(self, args):
        cifar_norm = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]
        if args.cl_default:
            transform = transforms.Compose(
                [transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*cifar_norm)
                ])
        else:
            transform = transforms.Compose(
                [transforms.ToPILImage(),
                transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*cifar_norm)
                ])

        return transform


   
    def get_data_loaders(self, args):
        task_train_data = deepcopy(self.train_data)
        task_train_data.indices = self.train_dataset.where(self.train_dataset['region'] == self.REGION_ORDER[self.i]).index.values

        task_val_data = deepcopy(self.val_data)
        task_val_data.indices = self.val_dataset.where(self.val_dataset['region'] == self.REGION_ORDER[self.i]).index.values

        task_test_data = deepcopy(self.test_data)
        task_test_data.indices = self.test_dataset.where(self.test_dataset['region'] == self.REGION_ORDER[self.i]).index.values

        train_loader = get_train_loader("standard", task_train_data, batch_size=self.args.train.batch_size, num_workers=16)
        memory_loader = get_eval_loader("standard", task_val_data,batch_size=self.args.train.batch_size, num_workers=0)
        test_loader = get_eval_loader("standard", task_test_data, batch_size=self.args.train.batch_size, num_workers=0)

        self.test_loaders.append(test_loader)
        self.train_loaders.append(train_loader)
        self.memory_loaders.append(memory_loader)
        self.train_loader = train_loader
        self.i += 1
        return train_loader, memory_loader, test_loader
    