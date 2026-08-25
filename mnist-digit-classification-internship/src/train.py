"""Simple training script for the digit classifier.

Usage example:
    python src\train.py --epochs 2 --batch-size 128
"""
import argparse
import os
import sys
import json
import csv
from datetime import datetime
# Ensure project root is on sys.path so `from src.*` works when running as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from src.data import get_dataloaders
from src.model import SimpleCNN


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    # reproducibility
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch=args.batch_size,
        test_batch=args.batch_size,
        data_dir=args.data_dir,
        download=args.download,
        val_split=args.val_split,
        augment=args.augment,
        seed=args.seed,
    )

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logs_dir = os.path.join(args.checkpoint_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    metrics_csv = os.path.join(logs_dir, f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    metrics_json = os.path.join(logs_dir, f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

    # write CSV header
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        total = 0
        correct_train = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            preds = out.argmax(dim=1)
            correct_train += (preds == yb).sum().item()
            running += loss.item() * xb.size(0)
            total += xb.size(0)

        train_loss = running / total
        train_acc = correct_train / total if total > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - lr: {current_lr:.6f}')

        # validation (if available)
        val_loss = None
        val_acc = None
        if val_loader is not None:
            model.eval()
            v_running = 0.0
            v_total = 0
            correct = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    loss = criterion(out, yb)
                    preds = out.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    v_running += loss.item() * xb.size(0)
                    v_total += xb.size(0)
            val_loss = v_running / v_total if v_total > 0 else None
            val_acc = correct / v_total if v_total > 0 else None
            print(f'          val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')

        # step scheduler
        if scheduler is not None:
            scheduler.step()

        # save checkpoint each epoch with timestamp
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        ckpt_name = os.path.join(args.checkpoint_dir, f'model_epoch{epoch}_{ts}.pth')
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'args': vars(args),
        }
        torch.save(ckpt, ckpt_name)

        # save best model by val_loss when validation is present
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pth'))

        # append metrics to CSV and JSON summary
        with open(metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss if val_loss is not None else '', val_acc if val_acc is not None else '', current_lr])

        # update JSON summary
        summary = {}
        if os.path.exists(metrics_json):
            try:
                with open(metrics_json, 'r') as jh:
                    summary = json.load(jh)
            except Exception:
                summary = {}
        summary[epoch] = {'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc, 'lr': current_lr}
        with open(metrics_json, 'w') as jh:
            json.dump(summary, jh, indent=2)

    print('Training complete. Checkpoints saved to', args.checkpoint_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--scheduler', type=str, choices=['none', 'step'], default='none', help='LR scheduler to use')
    parser.add_argument('--step-size', type=int, default=1, help='StepLR step size')
    parser.add_argument('--gamma', type=float, default=0.9, help='StepLR gamma')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--val-split', type=float, default=0.1, help='Fraction of training set to use for validation')
    parser.add_argument('--augment', action='store_true', help='Enable training data augmentations')
    parser.add_argument('--download', action='store_true', help='Allow dataset download if not present')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--force-cpu', dest='force_cpu', action='store_true', help='Force CPU even if GPU available')
    args = parser.parse_args()
    train(args)
