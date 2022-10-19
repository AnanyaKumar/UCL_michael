# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from datasets import get_dataset
from torch.optim import SGD

from models.utils.continual_model import ContinualModel
from utils.args import *


def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)


def modified_kl_div(old, new):
    return -torch.mean(torch.sum(old * torch.log(new), 1))


class Lwf(ContinualModel):
    NAME = 'lwf'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, len_train_loader, transform):
        super(Lwf, self).__init__(backbone, loss, args, len_train_loader, transform)
        self.old_net = None
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.dataset = get_dataset(args)
        self.current_task = 0
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK
        nc = get_dataset(args).N_TASKS * self.cpt
        self.eye = torch.tril(torch.ones((nc, nc))).bool().to(self.device)

    def begin_task(self, train_loader):
        self.net.eval()
        if self.current_task > 0:
            opt = SGD(self.net.backbone.fc.parameters(), lr=self.args.train.lp_lr)
            for epoch in range(self.args.train.num_lp_epochs):
                for idx, ((inputs, images2, notaug_images), labels) in enumerate(train_loader):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    opt.zero_grad()
                    with torch.no_grad():
                        feats = self.net.backbone(inputs, return_features=True)
                    mask = self.eye[(self.current_task + 1) * self.cpt - 1] ^ self.eye[self.current_task * self.cpt - 1]
                    outputs = self.net.backbone.fc(feats)[:, mask]
                    loss = self.loss(outputs, labels - self.current_task * self.cpt)
                    loss.backward()
                    opt.step()

            logits = []
            with torch.no_grad():
                for i in range(0, train_loader.dataset.data.shape[0], self.args.train.batch_size):
                    inputs = torch.stack([train_loader.dataset.__getitem__(j)[0][0]
                                          for j in range(i, min(i + self.args.train.batch_size,
                                                         len(train_loader.dataset)))])
                    log = self.net.backbone(inputs.to(self.device)).cpu()
                    logits.append(log)
            setattr(train_loader.dataset, 'logits', torch.cat(logits))
        self.net.train()

        self.current_task += 1

    def observe(self, inputs1, labels, inputs2, notaug_inputs, logits=None):
        self.opt.zero_grad()
        outputs = self.net.backbone(inputs1.to(self.device))

        mask = self.eye[self.current_task * self.cpt - 1]
        loss = self.loss(outputs[:, mask], labels.to(self.device))
        if logits is not None:
            mask = self.eye[(self.current_task - 1) * self.cpt - 1]
            loss += self.args.train.alpha * modified_kl_div(smooth(self.soft(logits[:, mask]).to(self.device), 2, 1),
                                                      smooth(self.soft(outputs[:, mask]), 2, 1))

        loss.backward()
        self.opt.step()
        data_dict = {'loss': loss}

        return data_dict
