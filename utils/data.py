from typing import Union

import torch as T
import torch
import torch.nn as nn
from einops import rearrange, repeat
import numpy as np
from torchvision import datasets
from pathlib import Path
import torch.nn.functional as F




def get_dataset_presets():

    dataset_presets = {
            'cifar10_2cls': {
                'input_dim': 3*32*32,
                'output_dim': 1
            },
            'cifar10': {
                'input_dim': 3*32*32,
                'output_dim': 10
            },
            'cifar10_ez': {
                'input_dim': 3*32*32,
                'output_dim': 10
            },
            'svhn': {
                'input_dim': 3*32*32,
                'output_dim': 10
            },
            'fmnist': {
                'input_dim': 1*28*28,
                'output_dim': 10
            },

        }
    
    return dataset_presets



def prepare_cifar10_2cls(dataset_folder: Path, num_data: int, classes: list, dataset_seed: int = 888):
    datafolder = dataset_folder / 'cifar10'
    trainset = datasets.CIFAR10(root=datafolder, train=True, download=False)
    testset = datasets.CIFAR10(root=datafolder, train=False, download=False)

    CLASS1, CLASS2 = classes

    train_size = num_data

    for partition in ['train', 'test']:
        partition_set = trainset if partition == 'train' else testset
        
        idx1 = [i for i, target in enumerate(partition_set.targets) if target == CLASS1]
        idx2 = [i for i, target in enumerate(partition_set.targets) if target == CLASS2]

        if partition == 'train':
            idx1 = idx1[:train_size // 2]
            idx2 = idx2[:train_size // 2]

        idx = idx1 + idx2
        idx.sort()

        partition_data = partition_set.data[idx]
        partition_target = np.array(partition_set.targets)[idx]
        partition_target = np.array([-1 if target == CLASS2 else 1 for target in partition_target])

        X = T.tensor(partition_data)
        Y = T.tensor(partition_target)

        # Normalize the images
        X = X / 255.0
        X = X.float()

        X = X - T.tensor([0.4914, 0.4822, 0.4465])
        X = X / T.tensor([0.2023, 0.1994, 0.2010])

        X = rearrange(X, 'b w h c -> b c w h')

        X = X.detach()

        # Now Y
        Y = Y.float()
        Y = Y.detach()

        if partition == 'train':
            X_train = X
            Y_train = Y
        else:
            X_test = X
            Y_test = Y
        
    return X_train, Y_train, X_test, Y_test


def prepare_cifar10(dataset_folder: Path, 
                    num_data: int, 
                    dataset_seed: int = 888,
                    loss_type: str = 'mse'
                    ):
    datafolder = dataset_folder / 'cifar10'
    trainset = datasets.CIFAR10(root=datafolder, train=True, download=False)
    testset = datasets.CIFAR10(root=datafolder, train=False, download=False)

    train_size = num_data

    for partition in ['train', 'test']:
        partition_set = trainset if partition == 'train' else testset
        
        if partition == 'train':
            rng = np.random.default_rng(dataset_seed)
            idx = rng.choice(len(partition_set), train_size, replace=False)

        else:
            idx = list(range(len(partition_set)))

        partition_data = partition_set.data[idx]
        partition_target = np.array(partition_set.targets)[idx]

        X = T.tensor(partition_data)
        Y = T.tensor(partition_target)

        # Normalize the images
        X = X / 255.0
        X = X.float()

        X = X - T.tensor([0.4914, 0.4822, 0.4465])
        X = X / T.tensor([0.2023, 0.1994, 0.2010])

        X = rearrange(X, 'b w h c -> b c w h')

        X = X.detach().float()

        # Now Y
        if loss_type == 'mse':
            Y = F.one_hot(Y, num_classes=10).float()
        else:
            Y = Y.long()


        if partition == 'train':
            X_train = X
            Y_train = Y
        else:
            X_test = X
            Y_test = Y
        
    return X_train, Y_train, X_test, Y_test

def prepare_fmnist(dataset_folder: Path, 
                            num_data: int, 
                            dataset_seed: int = 888,
                            loss_type: str = 'mse'
                            ):
    datafolder = dataset_folder / 'fmnist'
    trainset = datasets.FashionMNIST(root=datafolder, train=True, download=False)
    testset = datasets.FashionMNIST(root=datafolder, train=False, download=False)

    train_size = num_data

    for partition in ['train', 'test']:
        partition_set = trainset if partition == 'train' else testset
        
        if partition == 'train':
            rng = np.random.default_rng(dataset_seed)
            idx = rng.choice(len(partition_set), train_size, replace=False)
        else:
            idx = list(range(len(partition_set)))

        partition_data = partition_set.data[idx]
        partition_target = np.array(partition_set.targets)[idx]

        X = partition_data.unsqueeze(1)  # Add channel dimension
        Y = T.tensor(partition_target)

        # Normalize the images
        X = X / 255.0
        X = X.float()

        # Standard normalization for Fashion-MNIST
        X = X - 0.2860
        X = X / 0.3530

        X = X.detach().float()

        # Now Y
        if loss_type == 'mse':
            Y = F.one_hot(Y, num_classes=10).float()
        else:
            Y = Y.long()

        if partition == 'train':
            X_train = X
            Y_train = Y
        else:
            X_test = X
            Y_test = Y
        
    return X_train, Y_train, X_test, Y_test

def prepare_cifar100(num_data: int):
    pass

def prepare_svhn(dataset_folder: Path, 
                    num_data: int, 
                    dataset_seed: int = 888,
                    loss_type: str = 'mse'
                    ):
    import scipy.io as sio
    
    datafolder = dataset_folder / 'svhn'

    svhn_train_path = datafolder / 'train_32x32.mat'
    svhn_test_path = datafolder / 'test_32x32.mat'

    # Load training data
    train_data = sio.loadmat(svhn_train_path)
    X_svhn_train = np.transpose(train_data['X'], (3, 0, 1, 2))  # Convert to (N, H, W, C)
    Y_svhn_train = train_data['y'].reshape(-1)
    # SVHN labels are from 1-10, with 10 representing 0. Convert to 0-9
    Y_svhn_train[Y_svhn_train == 10] = 0

    # Load test data
    test_data = sio.loadmat(svhn_test_path)
    X_svhn_test = np.transpose(test_data['X'], (3, 0, 1, 2))
    Y_svhn_test = test_data['y'].reshape(-1)
    Y_svhn_test[Y_svhn_test == 10] = 0

    # Convert to torch tensors
    X_svhn_train = torch.from_numpy(X_svhn_train).float() / 255.0  # Normalize to [0, 1]
    Y_svhn_train = torch.from_numpy(Y_svhn_train).long()
    X_svhn_test = torch.from_numpy(X_svhn_test).float() / 255.0
    Y_svhn_test = torch.from_numpy(Y_svhn_test).long()


    for partition in ['train', 'test']:
        if partition == 'train':
            X = X_svhn_train
            Y = Y_svhn_train
            # If num_data is specified, limit the training data
            if num_data > 0 and num_data <= len(X_svhn_train):
                rng = np.random.default_rng(dataset_seed)
                idx = rng.choice(len(X_svhn_train), num_data, replace=False)
                X = X[idx]
                Y = Y[idx]
        else:
            X = X_svhn_test[:10_000]
            Y = Y_svhn_test[:10_000]
        


        # Normalize the images
        X = X # THE IMAGES ARE ALREADY NORMALIZED
        X = X.float()

        # Normalize using precomputed statistics
        X = X - T.tensor([0.4377, 0.4438, 0.4728])
        X = X / T.tensor([0.1980, 0.2010, 0.1970])

        X = rearrange(X, 'b w h c -> b c w h')

        X = X.detach().float()

        # Now Y
        if loss_type == 'mse':
            Y = F.one_hot(Y, num_classes=10).float()
        else:
            Y = Y.long()


        if partition == 'train':
            X_train = X
            Y_train = Y
        else:
            X_test = X
            Y_test = Y
    
    return X_train, Y_train, X_test, Y_test



def prepare_cifar10_ez(dataset_folder: Path, num_data: int, dataset_seed: int = 888, loss_type: str = 'mse'):
    # Define the file paths
    cifar_folder = dataset_folder / 'cifar10_ez'
    train_x_path = cifar_folder / 'X_train_pulled.npy'
    train_y_path = cifar_folder / 'Y_train.npy'
    test_x_path = cifar_folder / 'X_test_pulled.npy'
    test_y_path = cifar_folder / 'Y_test.npy'
    
    # Load the data
    X_train = torch.tensor(np.load(train_x_path)).float()
    Y_train = torch.tensor(np.load(train_y_path))
    X_test = torch.tensor(np.load(test_x_path)).float()
    Y_test = torch.tensor(np.load(test_y_path))
    
    # If num_data is specified, limit the training data
    if num_data > 0 and num_data <= len(X_train):
        rng = np.random.default_rng(dataset_seed)
        idx = rng.choice(len(X_train), num_data, replace=False)
        X_train = X_train[idx]
        Y_train = Y_train[idx]
    
    # Handle Y based on loss_type
    if loss_type == 'mse':
        Y_train = Y_train.float()
        Y_test = Y_test.float()
        #F.one_hot(Y_test.long(), num_classes=10).float()
    else:
        raise NotImplementedError("Cross-entropy loss is not supported for CIFAR-10 EZ dataset YET")
        Y_train = Y_train.long()
        Y_test = Y_test.long()
    
    return X_train, Y_train, X_test, Y_test


def prepare_dataset(dataset: str, dataset_folder: Union[str, Path], num_data: int, classes: list, dataset_seed: int = 888,
                    loss_type: str = 'mse'
                    ):
    dataset_folder = Path(dataset_folder)
    train_size = num_data
    if dataset == 'cifar10_2cls':
        if loss_type == 'ce':
            raise NotImplementedError("Cross-entropy loss is not supported for 2-class CIFAR-10 dataset YET")
        return prepare_cifar10_2cls(dataset_folder, num_data, classes, dataset_seed=dataset_seed)
    if dataset == 'cifar10':
        return prepare_cifar10(dataset_folder, num_data, dataset_seed=dataset_seed, loss_type=loss_type)
    if dataset == 'cifar10_ez':
        return prepare_cifar10_ez(dataset_folder, num_data, dataset_seed=dataset_seed, loss_type=loss_type)
    if dataset == 'svhn':
        return prepare_svhn(dataset_folder, num_data, dataset_seed=dataset_seed, loss_type=loss_type)
    if dataset == 'fmnist':
        return prepare_fmnist(dataset_folder, num_data, dataset_seed=dataset_seed, loss_type=loss_type)
    
