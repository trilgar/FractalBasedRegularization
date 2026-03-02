#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Torch & Lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

try:
    from lightning import LightningModule, LightningDataModule, Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import CSVLogger
except Exception:
    from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger

from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report
from PIL import Image

# =======================
# Config (Brain Tumor - BASELINE)
# =======================
DATA_PATH = r"F:/datasets/Brain Tumor"
TRAIN_DIR = os.path.join(DATA_PATH, "Training")
TEST_DIR = os.path.join(DATA_PATH, "Testing")

BATCH_SIZE = 128
IMG_SIZE = (96, 96)
NUM_CLASSES = None  # set after scanning dirs
NUM_EPOCHS = 30
LR = 3e-4
LABELED_FRACTION = 0.05
MODEL_NAME = "fd_brain_tumor_BASELINE_m005"  # Змінено

# Normalization (ImageNet)
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("saved_metrics", exist_ok=True)

seed_everything(10, workers=True)
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# =======================
# Dataset & transforms
# =======================
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

eval_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])


class BrainTumorClsDataset(Dataset):
    """
    Класифікаційний датасет: повертає (X, y) з абсолютних шляхів.
    Очікує df з колонками: image (abs path), label_idx (int), label (str)
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image"]
        X = Image.open(img_path).convert("RGB")
        y = int(row["label_idx"])
        if self.transform is not None:
            X = self.transform(X)
        return X, torch.tensor(y, dtype=torch.long)


# =======================
# DataModule (Brain Tumor)
# =======================
class BrainTumorDataModule(LightningDataModule):
    def __init__(self, batch_size=BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size
        self.ce_weights = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.num_classes = None

    def _scan_split(self, split_dir: str) -> pd.DataFrame:
        rows = []
        class_names = []
        for entry in sorted(os.listdir(split_dir)):
            full = os.path.join(split_dir, entry)
            if os.path.isdir(full):
                class_names.append(entry)

        if self.class_to_idx is None:
            self.class_to_idx = {c: i for i, c in enumerate(sorted(class_names))}
            self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
            self.num_classes = len(self.class_to_idx)

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        for cls in self.class_to_idx.keys():
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            files = []
            for e in exts:
                files.extend(glob.glob(os.path.join(cls_dir, e)))
            for f in files:
                rows.append({
                    "image": os.path.abspath(f),
                    "label": cls,
                    "label_idx": self.class_to_idx[cls],
                })
        return pd.DataFrame(rows)

    def prepare_data(self):
        train_df = self._scan_split(TRAIN_DIR)
        test_df = self._scan_split(TEST_DIR)

        assert len(train_df) > 0, f"No training images found in {TRAIN_DIR}"
        assert self.num_classes is not None and self.num_classes >= 2, "Need at least 2 classes"
        self.df_train_all = train_df.reset_index(drop=True)
        self.df_test_all = test_df.reset_index(drop=True)

    def setup(self, stage=None):
        global NUM_CLASSES
        NUM_CLASSES = self.num_classes

        train_split = 0.9
        valid_split = 0.025
        valid_split_adj = valid_split / (1 - train_split)

        y_all = self.df_train_all["label_idx"]
        train_df, val_test_df = train_test_split(
            self.df_train_all, train_size=train_split, random_state=62, stratify=y_all
        )
        val_df, _ = train_test_split(
            val_test_df, train_size=valid_split_adj, random_state=62, stratify=val_test_df["label_idx"]
        )

        self.train_full = BrainTumorClsDataset(train_df, transform=train_transform)
        self.val_set = BrainTumorClsDataset(val_df, transform=eval_transform)
        self.test_set = BrainTumorClsDataset(self.df_test_all, transform=eval_transform)

        # ---- Stratified labeled/unlabeled split
        n_labeled = max(1, int(LABELED_FRACTION * len(train_df)))
        idx_all = np.arange(len(train_df))
        y_all = train_df["label_idx"].to_numpy()

        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_labeled, random_state=62)
        labeled_idx, _ = next(sss.split(idx_all, y_all))
        # unlabeled_idx  = np.setdiff1d(idx_all, labeled_idx) # Не потрібен для baseline

        self.train_labeled = Subset(self.train_full, labeled_idx.tolist())
        # self.train_unlabeled = Subset(self.train_full, unlabeled_idx.tolist()) # Не потрібен

        self.ce_weights = None

        print(f"[BrainTumor] Classes: {self.class_to_idx}")
        print(f"[BrainTumor] Labeled size: {len(self.train_labeled)}")
        print(f"[BrainTumor] Train/Val/Test sizes: {len(train_df)}/{len(val_df)}/{len(self.df_test_all)}")

    def train_dataloader(self):
        # *** ЗМІНЕНО: Повертаємо ТІЛЬКИ розмічені дані ***
        return DataLoader(self.train_labeled, batch_size=self.batch_size, shuffle=True,
                          pin_memory=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          pin_memory=True, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                          pin_memory=True, num_workers=4, persistent_workers=True)


# =======================
# Model (Baseline)
# =======================

# *** НОВА МОДЕЛЬ: Тільки Енкодер + Класифікатор ***
class BaselineNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Encoder (скопійовано з вашого AutoencoderNet)
        self.conv1 = nn.Conv2d(3, 32, stride=1, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, stride=1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, stride=4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, stride=4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Classifier head (скопійовано з вашого AutoencoderNet)
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 256),
            nn.Linear(256, 64),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Прохід через енкодер
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        z = F.relu(self.bn4(self.conv4(h)))

        # Прохід через класифікатор
        logits = self.fc(z.view(z.size(0), -1))
        return logits


# *** НОВИЙ LIGHTNING MODULE: Тільки Класифікація ***
class LitBaselineModel(LightningModule):
    def __init__(self, num_classes, lr=LR):
        super().__init__()
        self.save_hyperparameters()
        self.net = BaselineNet(num_classes=num_classes)
        # ce_weights=None згідно вашого DataModule
        self.ce = nn.CrossEntropyLoss()

        # Для збереження графіків
        self.train_loss_hist, self.val_acc_hist, self.val_loss_hist = [], [], []
        # Для збереження звіту
        self.test_preds, self.test_targets = [], []

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch  # Батч тепер містить лише (x, y) з розміченого набору
        logits = self.forward(x)
        loss = self.ce(logits, y)

        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.ce(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("val/acc", acc, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return {"val_loss": loss.detach(), "val_acc": acc.detach()}

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val/loss")
        val_acc = self.trainer.callback_metrics.get("val/acc")
        if val_loss is not None: self.val_loss_hist.append(float(val_loss.cpu()))
        if val_acc is not None: self.val_acc_hist.append(float(val_acc.cpu()))
        # Змінено "train/loss_total" на "train/loss"
        train_loss = self.trainer.callback_metrics.get("train/loss")
        if train_loss is not None: self.train_loss_hist.append(float(train_loss.cpu()))

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x).argmax(dim=1)
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(y.detach().cpu())

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds).numpy()
        targs = torch.cat(self.test_targets).numpy()
        # Додано zero_division=0 для уникнення помилок на малих вибірках
        report = classification_report(targs, preds, digits=3, zero_division=0)
        print("\n=== TEST CLASSIFICATION REPORT ===\n", report)
        with open(os.path.join("saved_metrics", f"test_report_{MODEL_NAME}.txt"), "w", encoding="utf-8") as f:
            f.write(report)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# =======================
# Plot helpers
# =======================
def plot_and_save(history, title, ylabel, path_png):
    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_png, dpi=160)
    plt.close()


# =======================
# Main
# =======================
def main():
    dm = BrainTumorDataModule(batch_size=BATCH_SIZE)
    dm.prepare_data();
    dm.setup()

    # *** ЗМІНЕНО: Створюємо LitBaselineModel ***
    model = LitBaselineModel(
        num_classes=dm.num_classes,
        lr=LR
    )

    ckpt = ModelCheckpoint(
        dirpath="models",
        filename=MODEL_NAME + "-{epoch:02d}-{val_acc:.4f}",
        monitor="val/acc", mode="max", save_top_k=1
    )
    lrmon = LearningRateMonitor(logging_interval='epoch')
    logger = CSVLogger("models", name=f"lightning_logs_{MODEL_NAME}")

    trainer = Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="gpu",
        devices=1,
        callbacks=[ckpt, lrmon],
        logger=logger,
        precision=16,
        deterministic=True,
    )

    print("DATA_PATH:", DATA_PATH)
    print("Train dir exists:", os.path.isdir(TRAIN_DIR))
    print("Test  dir exists:", os.path.isdir(TEST_DIR))
    print("Classes:", dm.class_to_idx)

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm, ckpt_path=ckpt.best_model_path if ckpt.best_model_path else None)

    torch.save(model.state_dict(), os.path.join("models", f"{MODEL_NAME}.pt"))

    plot_and_save(model.train_loss_hist, "Train Loss (total)", "Loss",
                  os.path.join("plots", f"{MODEL_NAME}_train_loss.png"))
    plot_and_save(model.val_loss_hist, "Validation Loss", "Loss", os.path.join("plots", f"{MODEL_NAME}_val_loss.png"))
    plot_and_save(model.val_acc_hist, "Validation Acc", "Accuracy", os.path.join("plots", f"{MODEL_NAME}_val_acc.png"))

    print("\nSaved:")
    print(" - Best checkpoint:", ckpt.best_model_path if ckpt.best_model_path else "(none)")
    print(" - Latest state_dict: models/{}.pt".format(MODEL_NAME))
    print(
        " - Plots: plots/{}_train_loss.png, plots/{}_val_loss.png, plots/{}_val_acc.png".format(MODEL_NAME, MODEL_NAME,
                                                                                                MODEL_NAME))
    print(" - Test report: saved_metrics/test_report_{}.txt".format(MODEL_NAME))


if __name__ == "__main__":
    main()