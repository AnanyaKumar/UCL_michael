from torchvision.datasets import CIFAR10
import numpy as np
from sklearn.model_selection import train_test_split

class CIFAR10Subsampled(CIFAR10):

    def __init__(self, root, train, probe_train_frac, seed=0, transform=None, target_transform=None, download=False):
        super().__init__(
            root=root, train=train, transform=transform, target_transform=target_transform,
            download=download)
        cifar_len = super().__len__()
        if train:
            self.data, _, self.targets, _, self._indices, _ = train_test_split(self.data, self.targets, np.arange(cifar_len), 
            train_size=probe_train_frac, random_state=seed, stratify=self.targets)            
        else:
            self._indices = np.arange(cifar_len)

    def __getitem__(self, i):
        return super().__getitem__(i)
        # super_index = self._indices[i]
        # return super().__getitem__(super_index)

    def __len__(self) -> int:
        return len(self.data)

