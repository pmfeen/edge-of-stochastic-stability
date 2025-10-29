from typing import Union, Tuple
import pickle
import os

import torch as T
import torch
import torch.nn as nn
from einops import rearrange, repeat
import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.nn.functional as F




class GaussianMixture2D(Dataset):
    """2D Gaussian Mixture dataset for testing DDPM on synthetic data."""
    
    def __init__(self, n_samples=10000, n_modes=8, radius=5.0, std=0.1):
        # equally spaced centers on a circle
        angles = np.linspace(0, 2*np.pi, n_modes, endpoint=False)
        self.centers = np.stack([radius*np.cos(angles), radius*np.sin(angles)], axis=1)
        self.n_samples = n_samples
        self.std = std

        # pick random centers and sample around them
        self.data = []
        for _ in range(n_samples):
            c = self.centers[np.random.choice(n_modes)]
            x, y = np.random.normal(c[0], std), np.random.normal(c[1], std)
            self.data.append([x, y])
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]


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
            'imagenet32': {
                'input_dim': 3*32*32,
                'output_dim': 1000
            },
            'cifar10_images': {
                'input_dim': (3, 32, 32),  # Image format for DDPM
                'output_dim': 1  # DDPM doesn't use output targets
            },
            'fmnist_images': {
                'input_dim': (1, 28, 28),  # Image format for DDPM
                'output_dim': 1  # DDPM doesn't use output targets
            },
            'gaussian_mixture_2d': {
                'input_dim': (1, 4, 4),  # Reshaped to 4x4 images for DDPM
                'output_dim': 1  # DDPM doesn't use output targets
            }

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


def load_batch(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read one ImageNet-32 pickle batch and return images (HWC uint8) & labels."""
    with open(path, "rb") as f:
        entry = pickle.load(f, encoding="bytes")

    data = entry["data"]          # (N, 3072) uint8
    labels = entry["labels"]

    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(labels, dtype=np.int64)
    return images, labels


def load_imagenet32(root: str, train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Load all train batches or the validation split into memory."""
    if train:
        batches = sorted(f for f in os.listdir(root) if f.startswith("train_data_batch_"))
        if not batches:
            raise FileNotFoundError("No 'train_data_batch_*' files found in " + root)
    else:
        batches = ["val_data"]

    imgs_list, lbls_list = [], []
    for bname in batches:
        imgs, lbls = load_batch(os.path.join(root, bname))
        imgs_list.append(imgs)
        lbls_list.append(lbls)

    return np.concatenate(imgs_list), np.concatenate(lbls_list)


class ImageNet32(Dataset):  # type: ignore[misc]
    """PyTorch-friendly wrapper (optional)."""

    def __init__(self, root: str, train: bool = True, transform=None, target_transform=None):
        self.images, self.labels = load_imagenet32(root, train)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):  # type: ignore[override]
        return self.images.shape[0]

    def __getitem__(self, idx):  # type: ignore[override]
        img, lbl = self.images[idx], int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        elif torch and not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.target_transform:
            lbl = self.target_transform(lbl)
        return img, lbl


def prepare_imagenet32(dataset_folder: Path, num_data: int, dataset_seed: int = 888, loss_type: str = 'mse'):
    """Prepare ImageNet32 dataset for training."""
    datafolder = dataset_folder / 'imagenet32'
    
    # Load train and test data
    train_images, train_labels = load_imagenet32(str(datafolder), train=True)
    test_images, test_labels = load_imagenet32(str(datafolder), train=False)
    
    train_size = num_data

    for partition in ['train', 'test']:
        if partition == 'train':
            partition_data = train_images
            partition_target = train_labels
            
            if num_data > 0 and num_data < len(partition_data):
                rng = np.random.default_rng(dataset_seed)
                idx = rng.choice(len(partition_data), train_size, replace=False)
                partition_data = partition_data[idx]
                partition_target = partition_target[idx]
        else:
            partition_data = test_images
            partition_target = test_labels

        X = T.tensor(partition_data)
        Y = T.tensor(partition_target)

        # Normalize the images
        X = X / 255.0
        X = X.float()

        # ImageNet normalization values
        X = X - T.tensor([0.485, 0.456, 0.406])
        X = X / T.tensor([0.229, 0.224, 0.225])

        X = rearrange(X, 'b w h c -> b c w h')

        X = X.detach().float()

        # Now Y
        Y = Y - 1 # ImageNet labels are from 1 to 1000, not 0 to 999
        if loss_type == 'mse':
            Y = F.one_hot(Y, num_classes=1000).float()  # ImageNet has 1000 classes
        else:
            Y = Y.long()

        if partition == 'train':
            X_train = X
            Y_train = Y
        else:
            X_test = X
            Y_test = Y
        
    return X_train, Y_train, X_test, Y_test


def prepare_cifar10_images(dataset_folder: Path, num_data: int, classes: list, dataset_seed: int = 888):
    """
    Prepare CIFAR-10 dataset in image format for DDPM training.
    
    Args:
        dataset_folder: Path to dataset folder
        num_data: Number of data points to use
        classes: List of classes to use (ignored for DDPM)
        dataset_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, Y_train, X_test, Y_test) where X are images
    """
    from torchvision import datasets, transforms
    
    # Set random seed for reproducibility
    torch.manual_seed(dataset_seed)
    
    # Transform to normalize images to [-1, 1] range for DDPM
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    # Load CIFAR-10 dataset
    datafolder = dataset_folder / 'cifar10'
    train_dataset = datasets.CIFAR10(
        root=str(datafolder),
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=str(datafolder),
        train=False,
        download=True,
        transform=transform
    )
    
    # Convert to tensors
    X_train = torch.stack([train_dataset[i][0] for i in range(min(len(train_dataset), num_data))])
    X_test = torch.stack([test_dataset[i][0] for i in range(min(len(test_dataset), num_data // 4))])
    
    # Create dummy targets (DDPM doesn't use targets)
    Y_train = torch.zeros(len(X_train), 1)
    Y_test = torch.zeros(len(X_test), 1)
    
    return X_train, Y_train, X_test, Y_test


def prepare_fmnist_images(dataset_folder: Path, num_data: int, classes: list, dataset_seed: int = 888):
    """
    Prepare Fashion-MNIST dataset in image format for DDPM training.
    
    Args:
        dataset_folder: Path to dataset folder
        num_data: Number of data points to use
        classes: List of classes to use (ignored for DDPM)
        dataset_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, Y_train, X_test, Y_test) where X are images
    """
    from torchvision import datasets, transforms
    
    # Set random seed for reproducibility
    torch.manual_seed(dataset_seed)
    
    # Transform to normalize images to [-1, 1] range for DDPM
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] for grayscale
    ])
    
    # Load Fashion-MNIST dataset
    datafolder = dataset_folder / 'fmnist'
    train_dataset = datasets.FashionMNIST(
        root=str(datafolder),
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.FashionMNIST(
        root=str(datafolder),
        train=False,
        download=True,
        transform=transform
    )
    
    # Convert to tensors
    X_train = torch.stack([train_dataset[i][0] for i in range(min(len(train_dataset), num_data))])
    X_test = torch.stack([test_dataset[i][0] for i in range(min(len(test_dataset), num_data // 4))])
    
    # Create dummy targets (DDPM doesn't use targets)
    Y_train = torch.zeros(len(X_train), 1)
    Y_test = torch.zeros(len(X_test), 1)
    
    return X_train, Y_train, X_test, Y_test


def prepare_gaussian_mixture_2d(dataset_folder: Path, num_data: int, classes: list, dataset_seed: int = 888):
    """
    Prepare GaussianMixture2D dataset for DDPM training.
    
    Args:
        dataset_folder: Path to dataset folder (not used for synthetic data)
        num_data: Number of data points to use
        classes: List of classes to use (ignored for DDPM)
        dataset_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, Y_train, X_test, Y_test) where X are 2D points reshaped for DDPM
    """
    # Set random seed for reproducibility
    torch.manual_seed(dataset_seed)
    np.random.seed(dataset_seed)
    
    # Create dataset
    dataset = GaussianMixture2D(n_samples=num_data, n_modes=8, radius=5.0, std=0.1)
    
    # Convert to tensors and normalize to [-1, 1] range for DDPM
    X_train = dataset.data
    
    # Normalize the 2D points to [-1, 1] range
    # Find the range of the data
    x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
    y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
    
    # Normalize to [-1, 1]
    X_train[:, 0] = 2 * (X_train[:, 0] - x_min) / (x_max - x_min) - 1
    X_train[:, 1] = 2 * (X_train[:, 1] - y_min) / (y_max - y_min) - 1
    
    # Reshape 2D data to work with DDPM UNet architecture
    # DDPM expects images of shape (batch, channels, height, width)
    # We'll reshape (N, 2) -> (N, 1, 4, 4) to meet UNet requirements
    batch_size = X_train.shape[0]
    
    # Pad the 2D data to create 4x4 "images"
    X_train_reshaped = torch.zeros(batch_size, 1, 4, 4)
    X_train_reshaped[:, 0, 0, 0] = X_train[:, 0]  # x coordinate
    X_train_reshaped[:, 0, 0, 1] = X_train[:, 1]  # y coordinate
    # Leave the rest as zeros (padding)
    
    # Create test set (use same data for simplicity)
    X_test = X_train_reshaped.clone()
    
    # Create dummy targets (DDPM doesn't use targets)
    Y_train = torch.zeros(len(X_train_reshaped), 1)
    Y_test = torch.zeros(len(X_test), 1)
    
    return X_train_reshaped, Y_train, X_test, Y_test


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
    if dataset == 'imagenet32':
        return prepare_imagenet32(dataset_folder, num_data, dataset_seed=dataset_seed, loss_type=loss_type)
    if dataset == 'cifar10_images':
        return prepare_cifar10_images(dataset_folder, num_data, classes, dataset_seed=dataset_seed)
    if dataset == 'fmnist_images':
        return prepare_fmnist_images(dataset_folder, num_data, classes, dataset_seed=dataset_seed)
    if dataset == 'gaussian_mixture_2d':
        return prepare_gaussian_mixture_2d(dataset_folder, num_data, classes, dataset_seed=dataset_seed)
    
