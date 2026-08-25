"""Evaluation utilities: load checkpoint, compute accuracy and confusion matrix, and save metrics/artifacts."""
import argparse
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
import seaborn as sns

# Allow running as a script
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import get_dataloaders
from src.model import SimpleCNN


def save_image_grid(images, labels, preds, out_path, ncols=8, cmap='gray'):
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    plt.figure(figsize=(ncols * 1.5, nrows * 1.5))
    for i in range(n):
        ax = plt.subplot(nrows, ncols, i+1)
        plt.axis('off')
        plt.imshow(images[i].squeeze(), cmap=cmap)
        ax.set_title(f'T:{labels[i]} P:{preds[i]}')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def evaluate(checkpoint_path: str, batch_size: int = 256, data_dir: str = './data', output_dir: str = './outputs'):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_loader = get_dataloaders(train_batch=64, test_batch=batch_size, data_dir=data_dir)

    model = SimpleCNN().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    all_preds = []
    all_targets = []
    all_images = []
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1).cpu().numpy()
            targets = yb.numpy()
            imgs = xb.cpu().numpy()

            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())
            # store first channel images
            all_images.extend([imgs[i,0,:,:] for i in range(imgs.shape[0])])

            correct += int((preds == targets).sum())
            total += len(targets)

    acc = correct / total

    # compute precision, recall, f1
    precision, recall, f1, support = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)

    report = classification_report(all_targets, all_preds, digits=4, zero_division=0)

    # save summary JSON
    summary = {
        'accuracy': acc,
        'correct': int(correct),
        'total': int(total),
        'incorrect': int(total - correct),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_precision),
        'weighted_recall': float(weighted_recall),
        'weighted_f1': float(weighted_f1),
        'per_class': {str(i): {'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i]), 'support': int(support[i])} for i in range(len(precision))}
    }
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # save classification report
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    # confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    out_conf = os.path.join(output_dir, 'confusion_matrix.png')
    plt.title(f'Confusion matrix (acc={acc:.4f})')
    plt.savefig(out_conf)
    plt.close()

    # sample predictions: take first 32 examples
    n_samples = min(32, len(all_images))
    sample_images = all_images[:n_samples]
    sample_labels = all_targets[:n_samples]
    sample_preds = all_preds[:n_samples]
    sample_path = os.path.join(output_dir, 'sample_predictions.png')
    save_image_grid(sample_images, sample_labels, sample_preds, sample_path)

    # misclassified examples
    mis_images = []
    mis_labels = []
    mis_preds = []
    for img, t, p in zip(all_images, all_targets, all_preds):
        if t != p:
            mis_images.append(img)
            mis_labels.append(t)
            mis_preds.append(p)
            if len(mis_images) >= 32:
                break
    mis_path = None
    if len(mis_images) > 0:
        mis_path = os.path.join(output_dir, 'misclassified.png')
        save_image_grid(mis_images, mis_labels, mis_preds, mis_path)

    print('Evaluation complete. Summary saved to', summary_path)
    return summary, {'confusion_matrix': out_conf, 'sample_predictions': sample_path, 'misclassified': mis_path, 'classification_report': report_path}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./outputs')
    args = parser.parse_args()
    summary, artifacts = evaluate(args.checkpoint, args.batch_size, args.data_dir, args.output_dir)
    print('Artifacts:', artifacts)
