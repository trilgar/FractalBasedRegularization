#!/usr/-bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Torch & Lightning
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, Subset

try:
    from lightning import LightningModule, LightningDataModule, Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import CSVLogger
except Exception:
    from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger

from torchvision import transforms, models
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, fbeta_score
from PIL import Image
from torchmetrics import FBetaScore


# =======================
# Utils
# =======================
def fmt_pct_three(val: float, prefix: str) -> str:
    """Formats a float (0-1) as a three-digit percentage string."""
    n = int(round(val * 100))
    return f"{prefix}{n:03d}"


# =======================
# Config (Chest X-ray)
# =======================
DATA_PATH = "F:/datasets/Chest X-Ray"
TRAIN_DIR = os.path.join(DATA_PATH, "train")
TEST_DIR = os.path.join(DATA_PATH, "test")
TRAIN_LIST = os.path.join(DATA_PATH, "train.txt")
TEST_LIST = os.path.join(DATA_PATH, "test.txt")

IMG_SIZE = (512, 512)
BATCH_SIZE = 16
NUM_CLASSES = 2
NUM_EPOCHS = 10
LR = 1e-3
LABELED_FRACTION = 0.05

m_str = fmt_pct_three(LABELED_FRACTION, "m")
MODEL_NAME = f"resnet18_aug_classifier_512_{m_str}"

# ImageNet normalization
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("saved_metrics", exist_ok=True)

seed_everything(10, workers=True)
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# =======================
# Transforms & dataset
# =======================
# ### НОВІ, БІЛЬШ АГРЕСИВНІ АУГМЕНТАЦІЇ ###
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

eval_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])


def _read_list_file(list_path: str) -> pd.DataFrame:
    """
    Reads whitespace-separated lines: <id> <filename> <label_str> <source>
    Returns DataFrame with columns: filename, label_str
    """
    rows = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            _, fname, label_str, _src = parts
            rows.append({"filename": fname, "label_str": label_str})
    return pd.DataFrame(rows)


_LABEL_MAP = {"positive": 1, "negative": 0}


class ChestXRayClsDataset(Dataset):
    """Returns (X, y) from folders and list files."""

    def __init__(self, df: pd.DataFrame, transform=None, root: str = TRAIN_DIR):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, row["filename"])
        X = Image.open(img_path)
        if X.mode != "RGB":
            X = X.convert("RGB")
        y = int(row["label_idx"])
        if self.transform is not None:
            X = self.transform(X)
        return X, torch.tensor(y, dtype=torch.long)


# =======================
# DataModule
# =======================
class ChestXRayDataModule(LightningDataModule):
    def __init__(self, batch_size=BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size
        self.ce_weights = None

    def prepare_data(self):
        tr = _read_list_file(TRAIN_LIST)
        ts = _read_list_file(TEST_LIST)
        assert len(tr) > 0, f"No training entries in {TRAIN_LIST}"
        tr["label_idx"] = tr["label_str"].str.lower().map(_LABEL_MAP).astype(int)
        ts["label_idx"] = ts["label_str"].str.lower().map(_LABEL_MAP).astype(int)
        self.df_train_all = tr[["filename", "label_idx"]].reset_index(drop=True)
        self.df_test_all = ts[["filename", "label_idx"]].reset_index(drop=True)

    def setup(self, stage=None):
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

        train_full_dataset = ChestXRayClsDataset(train_df, transform=train_transform, root=TRAIN_DIR)

        n_total_train = len(train_df)
        n_labeled = max(1, int(LABELED_FRACTION * n_total_train))

        if n_labeled < n_total_train:
            idx_all = np.arange(n_total_train)
            y_tr = train_df["label_idx"].to_numpy()
            sss = StratifiedShuffleSplit(n_splits=1, train_size=n_labeled, random_state=62)
            labeled_idx, _ = next(sss.split(idx_all, y_tr))

            self.train_set = Subset(train_full_dataset, labeled_idx.tolist())
            used_labels = y_tr[labeled_idx]
            print(f"[ChestXRay] Using {n_labeled}/{n_total_train} ({LABELED_FRACTION:.1%}) samples for training.")
        else:
            self.train_set = train_full_dataset
            used_labels = train_df["label_idx"].to_numpy()
            print(f"[ChestXRay] Using all {n_total_train} samples for training.")

        self.val_set = ChestXRayClsDataset(val_df, transform=eval_transform, root=TRAIN_DIR)
        self.test_set = ChestXRayClsDataset(self.df_test_all, transform=eval_transform, root=TEST_DIR)

        counts = np.bincount(used_labels, minlength=NUM_CLASSES).astype(np.float32)
        inv = 1.0 / (counts + 1e-9)
        inv = inv / inv.mean()
        self.ce_weights = torch.tensor(inv, dtype=torch.float32)
        print(f"[ChestXRay] Used subset counts: {counts.tolist()} -> CE weights: {inv.tolist()}")
        print(f"[ChestXRay] Train/Val/Test sizes: {len(self.train_set)}/{len(self.val_set)}/{len(self.test_set)}")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          pin_memory=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          pin_memory=True, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                          pin_memory=True, num_workers=4, persistent_workers=True)


# =======================
# Model: ResNet18 Classifier
# =======================
class ResNet18Classifier(nn.Module):
    """Encoder: torchvision resnet18 with a classification head."""

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(pretrained=pretrained)

        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone.fc.in_features, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        logits = self.cls_head(features)
        return logits


# =======================
# Lightning module
# =======================
class LitClassifier(LightningModule):
    def __init__(self, lr=LR, ce_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["ce_weights"])
        self.net = ResNet18Classifier(num_classes=NUM_CLASSES, pretrained=True)

        if ce_weights is None:
            ce_weights = torch.ones(NUM_CLASSES, dtype=torch.float32)
        elif not isinstance(ce_weights, torch.Tensor):
            ce_weights = torch.tensor(ce_weights, dtype=torch.float32)
        self.register_buffer("class_weights", ce_weights)
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)

        # Initialize the F2 Metric
        self.f2_metric = FBetaScore(task="multiclass", num_classes=NUM_CLASSES, beta=2.0, average="macro")

        self.train_loss_hist, self.val_f2_hist, self.val_loss_hist = [], [], []
        self.test_preds, self.test_targets = [], []

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.ce(logits, y)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.ce(logits, y)
        preds = logits.argmax(dim=1)

        # Calculate F2
        f2 = self.f2_metric(preds, y)

        self.log("val/f2", f2, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        return {"val_loss": loss.detach(), "val_f2": f2.detach()}

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val/loss")
        val_acc = self.trainer.callback_metrics.get("val/acc")
        if val_loss is not None: self.val_loss_hist.append(float(val_loss.cpu()))
        if val_acc is not None: self.val_f2_hist.append(float(val_acc.cpu()))
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

        # FIX: explicitly order names for Class 0, then Class 1
        report = classification_report(targs, preds, digits=3, target_names=["negative", "positive"])

        f2_macro = fbeta_score(targs, preds, beta=2.0, average='macro')
        f2_positive = fbeta_score(targs, preds, beta=2.0, average='binary', pos_label=1)

        custom_metrics = (
            f"\n--- Custom F2 Metrics ---\n"
            f"Macro F2-score    : {f2_macro:.3f}\n"
            f"Positive Class F2 : {f2_positive:.3f}\n"
        )
        final_report = report + custom_metrics

        print("\n=== TEST CLASSIFICATION REPORT ===\n", final_report)

    # ### ОНОВЛЕНИЙ ОПТИМІЗАТОР З ПЛАНУВАЛЬНИКОМ ###
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',      # Слідкуємо за метрикою, де більше - краще
            factor=0.2,      # Зменшуємо LR на 80% (1 - 0.2)
            patience=2,      # Чекаємо 2 епохи без покращення
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/f2",
            },
        }

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
    dm = ChestXRayDataModule(batch_size=BATCH_SIZE)
    dm.prepare_data();
    dm.setup()

    model = LitClassifier(lr=LR, ce_weights=getattr(dm, "ce_weights", None))

    ckpt = ModelCheckpoint(
        dirpath="models",
        filename=MODEL_NAME + "-{epoch:02d}-{val_f2:.4f}",
        monitor="val/f2",
        mode="max",
        save_top_k=1
    )
    lrmon = LearningRateMonitor(logging_interval='step') # змінив на 'step' для кращого моніторингу
    logger = CSVLogger("models", name=f"lightning_logs_{MODEL_NAME}")

    trainer = Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="gpu",
        devices=1,
        callbacks=[ckpt, lrmon],
        logger=logger,
        precision="16-mixed",
        deterministic=True,
    )

    print("DATA_PATH:", DATA_PATH)
    print("Train dir exists:", os.path.isdir(TRAIN_DIR))
    print("Test dir exists:", os.path.isdir(TEST_DIR))
    print("TRAIN_LIST exists:", os.path.isfile(TRAIN_LIST))
    print("TEST_LIST exists:", os.path.isfile(TEST_LIST))

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm, ckpt_path=ckpt.best_model_path if ckpt.best_model_path else "best")

    final_model_path = os.path.join("models", f"{MODEL_NAME}_final.pt")
    torch.save(model.state_dict(), final_model_path)

    plot_and_save(model.train_loss_hist, "Train Loss", "Loss", os.path.join("plots", f"{MODEL_NAME}_train_loss.png"))
    plot_and_save(model.val_loss_hist, "Validation Loss", "Loss", os.path.join("plots", f"{MODEL_NAME}_val_loss.png"))
    plot_and_save(model.val_f2_hist, "Validation Accuracy", "Accuracy",
                  os.path.join("plots", f"{MODEL_NAME}_val_acc.png"))

    print("\nSaved:")
    print(" - Best checkpoint:", ckpt.best_model_path if ckpt.best_model_path else "(none)")
    print(" - Final state_dict:", final_model_path)
    print(" - Plots:", f"plots/{MODEL_NAME}_train_loss.png, etc.")
    print(" - Test report:", f"saved_metrics/test_report_{MODEL_NAME}.txt")


if __name__ == "__main__":
    main()