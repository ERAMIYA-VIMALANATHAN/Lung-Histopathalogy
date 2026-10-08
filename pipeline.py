"""
Lung Histopathology Image Classification Pipeline
=================================================
Project:
GhostNet-100 + Lightweight Transformer

Pipeline:
1. Load dataset
2. Preprocess and augment images
3. Split into train / validation / test
4. Build GhostNet-100 baseline
5. Build GhostNet-100 + Lightweight Transformer
6. Train with AdamW + Cosine Annealing
7. Apply early stopping
8. Evaluate using Accuracy, Precision, Recall and F1-score
9. Compare baseline and improved models
10. Save the best model

Note:
This file provides the project pipeline structure.
Model-specific classes such as GhostNet100Baseline and
GhostNetTransformerHybrid can be imported from your model file.
"""

import os
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

CONFIG = {
    "data_dir": "dataset",
    "image_size": 224,
    "batch_size": 32,

    "num_classes": 3,
    "num_workers": 2,

    "learning_rate": 2e-4,
    "weight_decay": 1e-2,

    "max_epochs": 12,
    "patience": 3,

    "transformer_embed_dim": 256,
    "transformer_heads": 4,
    "transformer_layers": 1,
    "transformer_ffn_dim": 512,
    "dropout": 0.1,

    "seed": 42,

    "baseline_checkpoint": "ghostnet100_baseline_best.pth",
    "hybrid_checkpoint": "ghostnet100_transformer_best.pth",
}


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def seed_everything(seed=42):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Makes experiments more reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. DEVICE
# ============================================================

def get_device():
    """Select GPU when available, otherwise CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================

def create_transforms(image_size=224):
    """
    Training:
        Resize -> Random Augmentation -> Tensor -> Normalize

    Validation/Test:
        Resize -> Tensor -> Normalize
    """

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, eval_transform


# ============================================================
# 5. DATASET AND DATA SPLITTING
# ============================================================

def create_dataloaders(data_dir, image_size=224, batch_size=32,
                       num_workers=2, seed=42):
    """
    Expected dataset structure:

    dataset/
    ├── Normal/
    ├── Adenocarcinoma/
    └── Squamous_Cell_Carcinoma/

    The dataset is split into:
        70% Training
        15% Validation
        15% Testing
    """

    train_transform, eval_transform = create_transforms(image_size)

    # First dataset is used to obtain the image list and class labels.
    full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=train_transform
    )

    print("Classes:", full_dataset.classes)
    print("Total images:", len(full_dataset))

    train_size = int(0.70 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # Validation and test images should not use augmentation.
    val_dataset.dataset = copy.deepcopy(full_dataset)
    val_dataset.dataset.transform = eval_transform

    test_dataset.dataset = copy.deepcopy(full_dataset)
    test_dataset.dataset.transform = eval_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Training images:   {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Testing images:    {len(test_dataset)}")

    return train_loader, val_loader, test_loader, full_dataset.classes


# ============================================================
# 6. MODEL IMPORTS
# ============================================================

def build_models(num_classes=3):
    """
    Import your actual model implementations here.

    Expected classes:
        GhostNet100Baseline
        GhostNetTransformerHybrid
    """

    # Change this import to match your project file.
    from models import GhostNet100Baseline, GhostNetTransformerHybrid

    baseline = GhostNet100Baseline(
        num_classes=num_classes
    )

    hybrid = GhostNetTransformerHybrid(
        num_classes=num_classes,
        embed_dim=CONFIG["transformer_embed_dim"],
        num_heads=CONFIG["transformer_heads"],
        num_layers=CONFIG["transformer_layers"],
        dim_feedforward=CONFIG["transformer_ffn_dim"],
        dropout=CONFIG["dropout"]
    )

    return baseline, hybrid


# ============================================================
# 7. TRAINING
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one complete epoch."""

    model.train()

    running_loss = 0.0
    predictions = []
    targets = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)

        predictions.extend(preds.detach().cpu().numpy())
        targets.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_accuracy = accuracy_score(targets, predictions)

    return epoch_loss, epoch_accuracy


# ============================================================
# 8. VALIDATION
# ============================================================

@torch.no_grad()
def validate(model, loader, criterion, device):
    """Evaluate the model on the validation set."""

    model.eval()

    running_loss = 0.0
    predictions = []
    targets = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)

        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_accuracy = accuracy_score(targets, predictions)

    return epoch_loss, epoch_accuracy


# ============================================================
# 9. MODEL TRAINING WITH EARLY STOPPING
# ============================================================

def train_model(model, train_loader, val_loader, device,
                checkpoint_path, model_name):
    """
    Training configuration:
        Optimizer: AdamW
        Learning rate: 2e-4
        Weight decay: 1e-2
        Scheduler: Cosine Annealing
        Loss: Cross Entropy
        Maximum epochs: 12
        Early stopping patience: 3
    """

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["max_epochs"],
        eta_min=1e-6
    )

    model = model.to(device)

    best_val_accuracy = 0.0
    best_state = None
    patience_counter = 0

    history = []

    print("\n" + "=" * 60)
    print(f"Training: {model_name}")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(CONFIG["max_epochs"]):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

        print(
            f"Epoch [{epoch + 1:02d}/{CONFIG['max_epochs']}] | "
            f"Train Loss: {train_loss:.5f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        # Save best model.
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

            torch.save(best_state, checkpoint_path)
            print("  -> Best model saved.")

        else:
            patience_counter += 1

        # Early stopping.
        if patience_counter >= CONFIG["patience"]:
            print("  -> Early stopping triggered.")
            break

    elapsed = time.time() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
    print(f"Training time: {elapsed / 60:.2f} minutes")

    return model, history


# ============================================================
# 10. TEST EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_model(model, test_loader, device, class_names):
    """Calculate final test metrics."""

    model.eval()

    predictions = []
    targets = []

    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.numpy())

    accuracy = accuracy_score(targets, predictions)

    precision = precision_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(targets, predictions)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1-Score : {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "class_names": class_names
    }


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

def compare_models(baseline_results, hybrid_results):
    """Print a simple comparison of the two models."""

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        f"{'Metric':<15}"
        f"{'GhostNet-100':<20}"
        f"{'GhostNet-100 + Transformer':<25}"
    )

    print("-" * 60)

    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-Score", "f1_score"),
    ]

    for display_name, key in metrics:
        print(
            f"{display_name:<15}"
            f"{baseline_results[key] * 100:<20.2f}"
            f"{hybrid_results[key] * 100:<25.2f}"
        )


# ============================================================
# 12. COMPLETE PIPELINE
# ============================================================

def main():

    # Step 1: Reproducibility
    seed_everything(CONFIG["seed"])

    # Step 2: Device
    device = get_device()

    # Step 3: Dataset
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=CONFIG["data_dir"],
        image_size=CONFIG["image_size"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        seed=CONFIG["seed"]
    )

    # Step 4: Build models
    baseline_model, hybrid_model = build_models(
        num_classes=CONFIG["num_classes"]
    )

    # Step 5: Train GhostNet-100 baseline
    baseline_model, baseline_history = train_model(
        baseline_model,
        train_loader,
        val_loader,
        device,
        CONFIG["baseline_checkpoint"],
        "GhostNet-100 Baseline"
    )

    # Step 6: Evaluate baseline
    baseline_results = evaluate_model(
        baseline_model,
        test_loader,
        device,
        class_names
    )

    # Step 7: Train improved hybrid model
    hybrid_model, hybrid_history = train_model(
        hybrid_model,
        train_loader,
        val_loader,
        device,
        CONFIG["hybrid_checkpoint"],
        "GhostNet-100 + Lightweight Transformer"
    )

    # Step 8: Evaluate improved model
    hybrid_results = evaluate_model(
        hybrid_model,
        test_loader,
        device,
        class_names
    )

    # Step 9: Compare models
    compare_models(
        baseline_results,
        hybrid_results
    )

    print("\nPipeline completed successfully.")


# ============================================================
# 13. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
