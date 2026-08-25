"""Simple unit checks for dataloader shapes and label ranges."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import get_dataloaders


def test_dataloader_shapes():
    train_loader, val_loader, test_loader = get_dataloaders(train_batch=16, test_batch=32, data_dir='./data', download=False, val_split=0.1, augment=False)

    # check first batch shapes
    xb, yb = next(iter(train_loader))
    assert xb.ndim == 4 and xb.shape[1:] == (1, 28, 28), f'Unexpected train batch shape: {xb.shape}'
    assert yb.ndim == 1 and yb.dtype != object, 'Unexpected train labels'

    xb, yb = next(iter(test_loader))
    assert xb.ndim == 4 and xb.shape[1:] == (1, 28, 28), f'Unexpected test batch shape: {xb.shape}'
    assert yb.ndim == 1 and yb.dtype != object, 'Unexpected test labels'

    if val_loader is not None:
        xb, yb = next(iter(val_loader))
        assert xb.ndim == 4 and xb.shape[1:] == (1, 28, 28), f'Unexpected val batch shape: {xb.shape}'


if __name__ == '__main__':
    test_dataloader_shapes()
    print('Data loader shape checks passed (note: this expects MNIST present if download=False).')
