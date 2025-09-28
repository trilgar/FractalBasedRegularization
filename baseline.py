# Baseline: Encoder-only ConvNet on ISIC2024 (train ONLY on labeled split)
# - Same encoder architecture as your AE
# - No decoder, no FD, no reconstruction losses
# - Same ISIC split logic (labeled/unlabeled), but unlabeled is unused

import os, warnings
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

warnings.filterwarnings("ignore")

# =======================
# Config (ISIC)
# =======================
DATA_PATH   = "F:/datasets/ISIC2024"
IM_DIR      = os.path.join(DATA_PATH, "train-image", "image")
CSV_PATH    = os.path.join(DATA_PATH, "train-metadata-balanced.csv")  # use your balanced csv or original

BATCH_SIZE       = 128
IMG_SIZE         = (96, 96)
NUM_CLASSES      = 2
NUM_EPOCHS       = 10
LR               = 1e-3
LABELED_FRACTION = 0.99
MODEL_NAME       = "cnn_isic_baseline_encoderonly_m100"

# Normalization (ImageNet)
norm_mean = [0.485, 0.456, 0.406]
norm_std  = [0.229, 0.224, 0.225]

# Optional outputs (disable if you want 100% no files)
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

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

class ISICClsDataset(Dataset):
    """Minimal classification dataset for ISIC that returns (X, y)."""
    def __init__(self, df: pd.DataFrame, transform=None, root=IM_DIR):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.root = root

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, row["image"])
        X = Image.open(img_path).convert("RGB")
        y = int(row["label_idx"])
        if self.transform is not None:
            X = self.transform(X)
        return X, torch.tensor(y, dtype=torch.long)

# =======================
# DataModule (ISIC)
#   - same split logic (creates labeled & unlabeled)
#   - BUT train_dataloader uses ONLY labeled subset
#   - class weights computed on labeled subset
# =======================
class ISICDataModule(LightningDataModule):
    def __init__(self, batch_size=BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size
        self.ce_weights = None  # filled in setup()

    def prepare_data(self):
        df = pd.read_csv(CSV_PATH)
        df["image"] = df["isic_id"].astype(str) + ".jpg"
        df["label_idx"] = df["target"].astype(int)
        df["label"] = np.where(df["label_idx"] == 1, "Skin cancer", "Benign")
        df.drop_duplicates(subset=["isic_id"], inplace=True)
        self.df = df[["image", "label_idx", "label"]].reset_index(drop=True)

    def setup(self, stage=None):
        # train / val / test (stratified)
        train_split = 0.9
        valid_split = 0.025
        valid_split_adj = valid_split / (1 - train_split)

        train_df, val_test_df = train_test_split(
            self.df, train_size=train_split, random_state=62, stratify=self.df["label_idx"]
        )
        val_df, test_df = train_test_split(
            val_test_df, train_size=valid_split_adj, random_state=62, stratify=val_test_df["label_idx"]
        )

        # Datasets
        self.train_full = ISICClsDataset(train_df, transform=train_transform, root=IM_DIR)
        self.val_set    = ISICClsDataset(val_df,   transform=eval_transform,  root=IM_DIR)
        self.test_set   = ISICClsDataset(test_df,  transform=eval_transform,  root=IM_DIR)

        # ---- Stratified labeled/unlabeled split (kept identical) ----
        n_labeled = int(LABELED_FRACTION * len(train_df))
        idx_all = np.arange(len(train_df))
        y_all   = train_df["label_idx"].to_numpy()

        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_labeled, random_state=62)
        labeled_idx, _ = next(sss.split(idx_all, y_all))
        unlabeled_idx  = np.setdiff1d(idx_all, labeled_idx)

        self.train_labeled   = Subset(self.train_full, labeled_idx.tolist())
        self.train_unlabeled = Subset(self.train_full, unlabeled_idx.tolist())  # not used in training

        # ---- Class weights computed on the labeled subset ----
        labeled_labels = y_all[labeled_idx]
        counts = np.bincount(labeled_labels, minlength=NUM_CLASSES).astype(np.float32)  # [N0, N1]
        inv = 1.0 / (counts + 1e-9)
        inv = inv / inv.mean()
        self.ce_weights = torch.tensor(inv, dtype=torch.float32)
        print(f"Labeled counts: {counts.tolist()}  -> CE weights: {inv.tolist()}")

    def train_dataloader(self):
        # Train only on labeled subset
        return DataLoader(self.train_labeled, batch_size=self.batch_size, shuffle=True,
                          pin_memory=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          pin_memory=True, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                          pin_memory=True, num_workers=4, persistent_workers=True)

# =======================
# Simple ConvNet = your Encoder + Classifier head
# (identical conv stack to your AE encoder)
# =======================
class SimpleEncoderClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # Encoder (same as your code)
        self.conv1 = nn.Conv2d(3, 32, stride=1, kernel_size=1, padding=0)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, stride=1, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, stride=4, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, stride=4, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)

        # Classifier head (same dims as before: 256 * 6 * 6 for 96x96 input)
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 256),
            nn.Linear(256, 64),
            nn.Linear(64, num_classes)
        )

    def encode(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        z = F.relu(self.bn4(self.conv4(h)))    # [B, 256, 6, 6] for 96x96 input
        return z

    def forward(self, x):
        z = self.encode(x)
        return self.fc(z.view(z.size(0), -1))

# =======================
# LightningModule (classification only)
# =======================
class LitCNN(LightningModule):
    def __init__(self, lr=LR, ce_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["ce_weights"])
        self.net = SimpleEncoderClassifier(num_classes=NUM_CLASSES)

        # class-weighted CE (register a buffer so Lightning moves it to device)
        if ce_weights is None:
            ce_weights = torch.ones(NUM_CLASSES, dtype=torch.float32)
        elif not isinstance(ce_weights, torch.Tensor):
            ce_weights = torch.tensor(ce_weights, dtype=torch.float32)
        self.register_buffer("class_weights", ce_weights)
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)

        self.train_loss_hist, self.val_acc_hist, self.val_loss_hist = [], [], []
        self.test_preds, self.test_targets = [], []

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.ce(logits, y)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.ce(logits, y)
        acc  = (logits.argmax(dim=1) == y).float().mean()
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc",  acc,  on_epoch=True, prog_bar=True)
        return {"val_loss": loss.detach(), "val_acc": acc.detach()}

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val/loss")
        val_acc  = self.trainer.callback_metrics.get("val/acc")
        if val_loss is not None: self.val_loss_hist.append(float(val_loss.cpu()))
        if val_acc  is not None: self.val_acc_hist.append(float(val_acc.cpu()))
        train_loss = self.trainer.callback_metrics.get("train/loss")
        if train_loss is not None: self.train_loss_hist.append(float(train_loss.cpu()))

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(y.detach().cpu())

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds).numpy()
        targs = torch.cat(self.test_targets).numpy()
        print("\n=== TEST CLASSIFICATION REPORT ===\n",
              classification_report(targs, preds, digits=3))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# =======================
# Plot helpers (inline)
# =======================
def plot_curves(train_loss, val_loss, val_acc, title_prefix="CNN Baseline"):
    plt.figure(figsize=(14,4))
    plt.subplot(1,2,1)
    plt.plot(train_loss, label="train loss")
    plt.plot(val_loss,   label="val loss")
    plt.title(f"{title_prefix} - Loss"); plt.grid(True); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(val_acc, label="val acc")
    plt.title(f"{title_prefix} - Val Acc"); plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.show()

# =======================
# Main
# =======================
def main():
    dm = ISICDataModule(batch_size=BATCH_SIZE)
    dm.prepare_data(); dm.setup()

    model = LitCNN(lr=LR, ce_weights=getattr(dm, "ce_weights", None))

    ckpt  = ModelCheckpoint(
        dirpath="models",
        filename=MODEL_NAME + "-{epoch:02d}-{val_acc:.4f}",
        monitor="val/acc", mode="max", save_top_k=1
    )
    lrmon = LearningRateMonitor(logging_interval='epoch')
    logger = CSVLogger("models", name=f"lightning_logs_{MODEL_NAME}")  # set logger=None if you want no files

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
    print("CSV:", CSV_PATH)
    print("Images dir exists:", os.path.isdir(IM_DIR))

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm, ckpt_path=ckpt.best_model_path if ckpt.best_model_path else None)

    # Inline plots only
    plot_curves(model.train_loss_hist, model.val_loss_hist, model.val_acc_hist, title_prefix="Encoder-only CNN")

if __name__ == "__main__":
    main()