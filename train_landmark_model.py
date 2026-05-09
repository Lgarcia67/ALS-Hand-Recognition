import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH      = "landmark_data.csv"
MODEL_OUT     = "landmark_asl_best.pth"
ENCODER_OUT   = "label_encoder.npy"
PLOT_OUT      = "training_curve.png"
CONFUSION_OUT = "confusion_matrix.png"

EPOCHS        = 80
BATCH_SIZE    = 64
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
VAL_SPLIT     = 0.15
SEED          = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Dataset ───────────────────────────────────────────────────────────────────
class LandmarkDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels,   dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_csv(path):
    features, labels = [], []
    with open(path, "r") as f:
        for row in csv.reader(f):
            if len(row) == 64:
                features.append([float(v) for v in row[:63]])
                labels.append(row[63].upper())
    return np.array(features, dtype=np.float32), np.array(labels)


# ── Model ─────────────────────────────────────────────────────────────────────
class LandmarkASL(nn.Module):
    """
    Lightweight MLP: 63 landmark features → 26 ASL classes.
    Three hidden layers with BatchNorm + Dropout for regularisation.
    """
    def __init__(self, num_classes=26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out  = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * len(y)
        preds       = out.argmax(1)
        correct    += (preds == y).sum().item()
        total      += len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_curves(history, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"],   label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"],   label="Val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved training curves → {path}")


def plot_confusion(labels, preds, class_names, path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix"); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved confusion matrix → {path}")


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"'{CSV_PATH}' not found. Run collect_landmarks.py first.")

    print(f"Loading data from '{CSV_PATH}'...")
    features, raw_labels = load_csv(CSV_PATH)
    print(f"  {len(features)} samples, {len(np.unique(raw_labels))} classes")

    # encode labels
    le = LabelEncoder()
    labels = le.fit_transform(raw_labels)
    np.save(ENCODER_OUT, le.classes_)
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Saved label encoder → {ENCODER_OUT}")

    # augment: add small gaussian noise to features to improve robustness
    noise = np.random.normal(0, 0.005, features.shape).astype(np.float32)
    aug_features = np.concatenate([features, features + noise])
    aug_labels   = np.concatenate([labels, labels])

    dataset   = LandmarkDataset(aug_features, aug_labels)
    n_val     = int(len(dataset) * VAL_SPLIT)
    n_train   = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Train: {n_train}  Val: {n_val}")

    model     = LandmarkASL(num_classes=len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = 0.0

    print(f"\nTraining for {EPOCHS} epochs on {device}...\n")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, preds, true = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val:
            best_val = vl_acc
            torch.save(model.state_dict(), MODEL_OUT)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                  f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
                  f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.3f}"
                  + (" ← best" if vl_acc == best_val else ""))

    print(f"\nBest val accuracy: {best_val:.4f}")
    print(f"Saved best weights → {MODEL_OUT}")

    # Final evaluation on val set with best weights
    model.load_state_dict(torch.load(MODEL_OUT, map_location=device))
    _, _, final_preds, final_true = evaluate(model, val_loader, criterion)
    print("\nClassification Report:")
    print(classification_report(final_true, final_preds,
                                 target_names=le.classes_))

    plot_curves(history, PLOT_OUT)
    plot_confusion(final_true, final_preds, le.classes_, CONFUSION_OUT)


if __name__ == "__main__":
    main()