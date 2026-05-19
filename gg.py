"""
VGGNet-16 on Blood Cell Images (BCCD) Dataset
Group: Dev Vardhan Bhamu (23BCS025) | Kavya Arora (23BCS041)
Assignment: Deep Learning - CNN Implementation
"""

import os
import random
from xml.parsers.expat import model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─── Config ─────────────────────────────────────────────────────────────────
DATA_DIR = "./dataset/BCCD_Dataset"  # root folder; subfolders = class names
NUM_CLASSES = 4
BATCH_SIZE = 16
EPOCHS = 12
LR = 3e-4
IMG_SIZE = 128
SAVE_PATH = "./models/vgg16_bccd.pth"

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

CLASS_NAMES = ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]


# ─── Transforms ─────────────────────────────────────────────────────────────
train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_test_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ─── Stratified Split ───────────────────────────────────────────────────────
def make_splits(data_dir):
    """Return train / val / test DataLoaders with stratified split (70/15/15)."""
    full_dataset = datasets.ImageFolder(data_dir)
    targets = np.array(full_dataset.targets)
    indices = np.arange(len(full_dataset))

    # First split: 70% train, 30% temp
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(sss1.split(indices, targets))

    # Second split: temp → 50% val, 50% test  (=> 15% / 15% of total)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_idx, test_idx = next(sss2.split(temp_idx, targets[temp_idx]))
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    print(
        f"Split sizes — Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}"
    )

    # Apply correct transforms by wrapping with a custom Dataset
    class SubsetWithTransform(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            # img is already a PIL image from ImageFolder
            img = self.transform(img)
            return img, label

    # ImageFolder caches transformed tensors; we need PIL images
    # Rebuild without transform so we always get PIL
    raw_dataset = datasets.ImageFolder(data_dir, transform=None)

    train_ds = SubsetWithTransform(raw_dataset, train_idx, train_transform)
    val_ds = SubsetWithTransform(raw_dataset, val_idx, val_test_transform)
    test_ds = SubsetWithTransform(raw_dataset, test_idx, val_test_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False
    )

    return train_loader, val_loader, test_loader


# ─── Model ──────────────────────────────────────────────────────────────────
def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """VGG-16 with ImageNet weights; replace classifier head for num_classes."""
    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg16(weights=weights)

    # Freeze conv features (transfer learning)
    for param in model.features[:24].parameters():
        param.requires_grad = False

    for param in model.features[24:].parameters():
        param.requires_grad = True

    # Replace classifier
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model.to(DEVICE)


# ─── Training Loop ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        if batch_idx % 50 == 0:
            print(f"Processing batch {batch_idx}/{len(loader)}")

        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


# ─── Full Test Evaluation (all metrics) ─────────────────────────────────────
def full_evaluation(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="weighted")
    cm = confusion_matrix(all_labels, all_preds)

    print("\n===== Test Set Metrics =====")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"AUC-ROC   : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    return acc, prec, rec, f1, auc, cm, all_labels, all_probs


# ─── Plotting ───────────────────────────────────────────────────────────────
def plot_curves(train_losses, val_losses, train_accs, val_accs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label="Train Loss")
    axes[0].plot(val_losses, label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(train_accs, label="Train Acc")
    axes[1].plot(val_accs, label="Val Acc")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("./plots/loss_accuracy_curves.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved: loss_accuracy_curves.png")


def plot_confusion_matrix(cm):
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — VGG-16 on BCCD")
    plt.tight_layout()
    plt.savefig("./plots/confusion_matrix.png", dpi=150)
    plt.show()
    print("Saved: confusion_matrix.png")


def plot_roc_curves(all_labels, all_probs):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    y_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
    plt.figure(figsize=(8, 6))
    colors = cycle(["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"])

    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves per Class — VGG-16 on BCCD")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./plots/roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: roc_curves.png")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    # 1. Data
    train_loader, val_loader, test_loader = make_splits(DATA_DIR)

    # 2. Model / Loss / Optimizer / Scheduler
    model = build_model(NUM_CLASSES, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [
            {"params": model.features[24:].parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 3e-4},
        ]
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # 3. Training
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}]  "
            f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f}  |  "
            f"Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc:.4f}"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Saved best model (val_acc={best_val_acc:.4f})")

    # 4. Load best model & evaluate on test set
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    acc, prec, rec, f1, auc, cm, all_labels, all_probs = full_evaluation(
        model, test_loader
    )

    # 5. Plots
    plot_curves(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(cm)
    plot_roc_curves(all_labels, all_probs)

    print(f"\nModel weights saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
