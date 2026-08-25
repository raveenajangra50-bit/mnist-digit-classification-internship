"""Data loader utilities for MNIST-like datasets (torchvision)

Provides: get_dataloaders(train_batch, test_batch, data_dir, download=True,
                           val_split=0.1, augment=False, seed=42)

Includes a small verify_dataset utility for quick checks.
"""
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np


def get_dataloaders(
    train_batch: int = 64,
    test_batch: int = 256,
    data_dir: str = './data',
    download: bool = True,
    val_split: float = 0.1,
    augment: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Return (train_loader, val_loader or None, test_loader).

    - val_split: fraction of training set to reserve for validation (0 -> no val set)
    - augment: enable simple data augmentations for training only
    """
    # transforms
    mean, std = (0.1307,), (0.3081,)
    train_transforms = [transforms.ToTensor()]
    if augment:
        # simple augmentations suitable for MNIST-like digits
        train_transforms = [
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.08, 0.08), shear=5),
            transforms.ToTensor(),
        ]
    train_transforms.append(transforms.Normalize(mean, std))

    basic_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_transform = transforms.Compose(train_transforms)

    # datasets
    train_ds = datasets.MNIST(data_dir, train=True, download=download, transform=train_transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=download, transform=basic_transform)

    # validation split
    val_loader = None
    if val_split and val_split > 0.0:
        total = len(train_ds)
        val_len = int(total * val_split)
        train_len = total - val_len
        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset = random_split(train_ds, [train_len, val_len], generator=generator)
        train_loader = DataLoader(train_subset, batch_size=train_batch, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=test_batch, shuffle=False, num_workers=2)
    else:
        train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True, num_workers=2)

    test_loader = DataLoader(test_ds, batch_size=test_batch, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def verify_dataset(data_dir: str = './data') -> None:
    """Download MNIST and print basic stats for quick verification."""
    ds_train = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
    ds_test = datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())

    print('Train size:', len(ds_train))
    print('Test size :', len(ds_test))

    labels = [y for _, y in ds_train]
    unique, counts = np.unique(labels, return_counts=True)
    print('Train label distribution:')
    for u, c in zip(unique, counts):
        print(f'  {u}: {c}')


if __name__ == '__main__':
    # quick smoke test / verification
    print('Downloading and verifying MNIST dataset...')
    verify_dataset('./data')
    t, v, te = get_dataloaders(64, 256, './data', download=True, val_split=0.1, augment=False)
    print('Train batches:', len(t), 'Val batches:', len(v) if v is not None else 0, 'Test batches:', len(te))
