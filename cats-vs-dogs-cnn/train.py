"""Train the cat-vs-dog CNN.

Examples
    python train.py                                   # small CNN from scratch
    python train.py --epochs 60 --img-size 160        # longer / bigger pictures
    python train.py --arch resnet18 --pretrained --img-size 224 --epochs 15   # most accurate
"""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from catsdogs.data import class_weights, get_loaders
from catsdogs.model import ARCHS, build_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", default="data", help="folder containing train/ and val/")
    p.add_argument("--arch", default="simple", choices=ARCHS)
    p.add_argument("--pretrained", action="store_true", help="start from ImageNet weights (resnet18 only)")
    p.add_argument("--epochs", type=int, default=40, help="how many passes over the training set")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate (step size)")
    p.add_argument("--img-size", type=int, default=128, help="pictures are resized to NxN")
    p.add_argument("--patience", type=int, default=12, help="stop if val accuracy hasn't improved for N epochs")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="models/best_model.pt")
    p.add_argument("--plots-dir", default="outputs")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over the data. If an optimizer is given we learn (train); otherwise we only measure."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
    return total_loss / seen, correct / seen


def save_curves(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(epochs, history["train_loss"], label="train")
    a1.plot(epochs, history["val_loss"], label="validation")
    a1.set(title="Loss (lower is better)", xlabel="epoch")
    a1.legend()
    a2.plot(epochs, history["train_acc"], label="train")
    a2.plot(epochs, history["val_acc"], label="validation")
    a2.set(title="Accuracy (higher is better)", xlabel="epoch", ylim=(0, 1.02))
    a2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds, train_loader, val_loader = get_loaders(
        args.data_dir, args.img_size, args.batch_size, args.workers
    )
    classes = train_ds.classes
    print(f"Classes: {classes} | train images: {len(train_ds)} | val images: {len(val_ds)}")

    model = build_model(args.arch, len(classes), args.pretrained).to(device)
    print(f"Model: {args.arch} | parameters: {sum(p.numel() for p in model.parameters()):,}")

    weights = class_weights(train_ds.targets, len(classes)).to(device)
    print(f"Class weights (fix imbalance): {dict(zip(classes, [round(w, 2) for w in weights.tolist()]))}")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_loss, bad_epochs = -1.0, float("inf"), 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[k].append(v)

        improved = va_acc > best_acc or (va_acc == best_acc and va_loss < best_loss)
        if improved:
            best_acc, best_loss, bad_epochs = va_acc, va_loss, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arch": args.arch,
                    "classes": classes,
                    "img_size": args.img_size,
                    "val_acc": best_acc,
                    "epoch": epoch,
                },
                out_path,
            )
        else:
            bad_epochs += 1

        print(
            f"Epoch {epoch:3d}/{args.epochs} | train loss {tr_loss:.3f} acc {tr_acc:.1%} | "
            f"val loss {va_loss:.3f} acc {va_acc:.1%}" + ("  <- saved best" if improved else "")
        )
        if bad_epochs >= args.patience:
            print(f"No improvement for {args.patience} epochs, stopping early.")
            break

    print(f"\nDone in {time.time() - start:.0f}s. Best validation accuracy: {best_acc:.1%}")
    print(f"Model saved to: {out_path}")

    (Path(args.plots_dir) / "history.json").write_text(json.dumps(history, indent=2))
    save_curves(history, Path(args.plots_dir) / "training_curves.png")
    print(f"Training curves saved to: {Path(args.plots_dir) / 'training_curves.png'}")


if __name__ == "__main__":
    main()
