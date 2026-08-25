"""Small helper to download and verify datasets used by the project."""
import os
import sys
# Add project root so `from src.data import ...` works when running as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import verify_dataset, get_dataloaders


if __name__ == '__main__':
    # Download and print basic stats
    verify_dataset('./data')

    # Create dataloaders with a small validation split and augmentations disabled by default
    train_loader, val_loader, test_loader = get_dataloaders(train_batch=64, test_batch=256, data_dir='./data', download=True, val_split=0.1, augment=False)
    print('Created dataloaders:')
    print('  train batches:', len(train_loader))
    print('  val batches  :', len(val_loader) if val_loader else 0)
    print('  test batches :', len(test_loader))
