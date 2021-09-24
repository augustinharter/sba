from typing import Union
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, dir, batch_size=32, num_workers=2):
        super().__init__()
        self.dir = dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        MNIST(self.dir, train=True, download=True)
        MNIST(self.dir, train=False, download=True)

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage: Union[str, None] = None):
        # transforms
        #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        transform = transforms.Compose([transforms.ToTensor()])
        # split dataset
        if stage in (None, "fit"):
            mnist_train = MNIST(self.dir, train=True, transform=transform)
            self.data_train, self.data_val = random_split(mnist_train, [55000, 5000])
        if stage == (None, "test"):
            self.data_test = MNIST(self.dir, train=False, transform=transform)

    # return the dataloader for each split
    def train_dataloader(self):
        data_train = DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers)
        return data_train

    def val_dataloader(self):
        data_val = DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)
        return data_val

    def test_dataloader(self):
        data_test = DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers)
        return data_test


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, dir, batch_size=32, num_workers=2):
        super().__init__()
        self.dir = dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        CIFAR10(self.dir, train=True, download=True)
        CIFAR10(self.dir, train=False, download=True)

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage: Union[str, None] = None):
        # transforms
        #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        transform = transforms.Compose([transforms.ToTensor()])
        # split dataset
        if stage in (None, "fit"):
            data_train = CIFAR10(self.dir, train=True, transform=transform)
            len = data_train.__len__()
            self.data_train, self.data_val = random_split(data_train, [int(len*0.8), int(len*0.2)])
        if stage == (None, "test"):
            self.data_test = CIFAR10(self.dir, train=False, transform=transform)

    # return the dataloader for each split
    def train_dataloader(self):
        data_train = DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers)
        return data_train

    def val_dataloader(self):
        data_val = DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)
        return data_val

    def test_dataloader(self):
        data_test = DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers)
        return data_test