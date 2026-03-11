"""
ASL Alphabet Recognition - Training Script
==========================================
Model  : EfficientNetB0 (transfer learning via torchvision)
Device : Apple M4 MPS (GPU) → auto-falls back to CPU
Dataset: asl_alphabet_train/ with 29 class folders
Usage  : python train.py --data_dir /path/to/archive
"""

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Device selection (M4 MPS → CUDA → CPU) ───────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        print("✅ Using Apple MPS (M4 GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("⚠️  Using CPU (no GPU detected)")
        return torch.device("cpu")

# ── Data transforms ───────────────────────────────────────────────────────────
def get_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 20, img_size + 20)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf

# ── Model: EfficientNetB0 fine-tuned ─────────────────────────────────────────
def build_model(num_classes: int, unfreeze_blocks: int = 3) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    # Freeze all backbone layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last N feature blocks for fine-tuning
    features = list(model.features.children())
    for block in features[-unfreeze_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model

# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels

# ── Plot helpers ──────────────────────────────────────────────────────────────
def plot_curves(history, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in zip(axes,
                                  [("train_loss","val_loss"), ("train_acc","val_acc")],
                                  ["Loss", "Accuracy"]):
        ax.plot(history[metric[0]], label="Train")
        ax.plot(history[metric[1]], label="Val")
        ax.set_title(title); ax.set_xlabel("Epoch")
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"📊 Saved training_curves.png")

def plot_confusion(preds, labels, class_names, save_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"📊 Saved confusion_matrix.png")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ASL Alphabet Training")
    parser.add_argument("--data_dir",  default="./archive",      help="Path to archive/ folder")
    parser.add_argument("--save_dir",  default="./model_output", help="Where to save checkpoints & plots")
    parser.add_argument("--epochs",    type=int, default=30)
    parser.add_argument("--batch",     type=int, default=64)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--img_size",  type=int, default=224)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--workers",   type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = get_device()

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_tf, val_tf = get_transforms(args.img_size)
    train_root = os.path.join(args.data_dir, "asl_alphabet_train", "asl_alphabet_train")
    if not os.path.exists(train_root):
        train_root = os.path.join(args.data_dir, "asl_alphabet_train")
    print(f"\n📂 Loading dataset from: {train_root}")

    # Two ImageFolder instances with different transforms, same indices — no leakage
    train_full  = datasets.ImageFolder(train_root, transform=train_tf)
    val_full    = datasets.ImageFolder(train_root, transform=val_tf)
    class_names = train_full.classes
    num_classes = len(class_names)
    print(f"   Classes ({num_classes}): {class_names}")
    print(f"   Total images: {len(train_full)}")

    # Deterministic index split (seed=42 for reproducibility)
    n             = len(train_full)
    val_size      = int(n * args.val_split)
    indices       = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    train_indices = indices[val_size:]
    val_indices   = indices[:val_size]

    train_ds = Subset(train_full, train_indices)
    val_ds   = Subset(val_full,   val_indices)
    print(f"   Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # Class-balanced sampler
    targets        = [train_full.targets[i] for i in train_indices]
    class_counts   = np.bincount(targets, minlength=num_classes).astype(float)
    weights        = 1.0 / class_counts
    sample_weights = torch.tensor([weights[t] for t in targets], dtype=torch.float)
    sampler        = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # ── Model, loss, optimiser, scheduler ────────────────────────────────────
    model = build_model(num_classes, unfreeze_blocks=3).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 EfficientNetB0  |  trainable: {trainable:,} / {total:,} params")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training ──────────────────────────────────────────────────────────────
    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    best_val_acc, patience_count, patience = 0.0, 0, 7

    print(f"\n🚀 Training for {args.epochs} epochs  |  batch={args.batch}  |  device={device}\n")
    print(f"{'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}  {'LR':>9}  {'Time':>7}")
    print("─" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, val_preds, val_labels = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        lr_now = scheduler.get_last_lr()[0]
        marker = " ⭐" if va_acc > best_val_acc else ""
        print(f"{epoch:>3}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  {va_loss:>8.4f}  {va_acc:>7.4f}  {lr_now:>9.2e}  {elapsed:>5.1f}s{marker}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            patience_count = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": va_acc,
                "class_names": class_names,
            }, os.path.join(args.save_dir, "best_model.pth"))
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n⏹  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\n✅ Best val accuracy: {best_val_acc:.4f}")

    # ── Final evaluation on best model ────────────────────────────────────────
    ckpt = torch.load(os.path.join(args.save_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    _, _, final_preds, final_labels = eval_epoch(model, val_loader, criterion, device)

    report = classification_report(final_labels, final_preds,
                                   target_names=class_names, digits=4)
    print("\n📋 Classification Report:\n")
    print(report)

    with open(os.path.join(args.save_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(args.save_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    plot_curves(history, args.save_dir)
    plot_confusion(final_preds, final_labels, class_names, args.save_dir)

    print(f"\n🎉 All outputs saved to: {args.save_dir}/")
    print("   best_model.pth | class_names.json | training_curves.png | confusion_matrix.png")

if __name__ == "__main__":
    main()