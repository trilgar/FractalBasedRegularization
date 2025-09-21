#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from fd_functions import *  # your FD utils (box_counting etc.)

# =======================
# Config (ISIC)
# =======================
DATA_PATH   = "F:/datasets/ISIC2024"
IM_DIR      = os.path.join(DATA_PATH, "train-image", "image")
CSV_PATH    = os.path.join(DATA_PATH, "train-metadata-updated.csv")

BATCH_SIZE       = 128
IMG_SIZE         = (96, 96)
NUM_CLASSES      = 2
NUM_EPOCHS       = 10
LR               = 3e-4
LABELED_FRACTION = 0.05
MODEL_NAME       = "fd_isic_l10_m005_r_030"

# Fractal regularization
LAMBDA_FD = 10
RC_RATE   = 0.3   # intermediate recon MSE weight inside reconstruction loss

# Normalization (ImageNet)
norm_mean = [0.485, 0.456, 0.406]
norm_std  = [0.229, 0.224, 0.225]

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

class SemiSupervisedDataset(Dataset):
    """
    Returns (labeled_x, labeled_y, unlabeled_x).
    Labeled indices are drawn with a target minority proportion p_min.
    """
    def __init__(self, labeled_ds: Dataset, unlabeled_ds: Dataset, labeled_labels, p_min=0.5):
        self.labeled_ds = labeled_ds
        self.unlabeled_ds = unlabeled_ds
        self.labeled_size = len(labeled_ds)
        self.unlabeled_size = len(unlabeled_ds)

        labeled_labels = np.asarray(labeled_labels)
        self.min_idx = np.where(labeled_labels == 1)[0].astype(np.int64)
        self.maj_idx = np.where(labeled_labels == 0)[0].astype(np.int64)
        self.p_min = float(p_min)

    def __len__(self):
        return self.unlabeled_size  # drives batches

    def __getitem__(self, idx):
        # sample a labeled index with desired minority proportion
        if (len(self.min_idx) > 0) and (np.random.rand() < self.p_min):
            li = np.random.choice(self.min_idx)
        else:
            li = np.random.choice(self.maj_idx)

        lx, ly = self.labeled_ds[int(li)]
        ux, _  = self.unlabeled_ds[idx]
        return lx, ly, ux

# =======================
# DataModule (ISIC)
# =======================
class ISICDataModule(LightningDataModule):
    def __init__(self, batch_size=BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size
        self.ce_weights = None  # will be filled in setup()

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

        # ---- Stratified labeled/unlabeled split (instead of random_split)
        n_labeled = int(LABELED_FRACTION * len(train_df))
        idx_all = np.arange(len(train_df))
        y_all   = train_df["label_idx"].to_numpy()

        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_labeled, random_state=62)
        labeled_idx, _ = next(sss.split(idx_all, y_all))
        unlabeled_idx  = np.setdiff1d(idx_all, labeled_idx)

        self.train_labeled   = Subset(self.train_full, labeled_idx.tolist())
        self.train_unlabeled = Subset(self.train_full, unlabeled_idx.tolist())

        # >>> PASS labeled_labels + p_min to SemiSupervisedDataset (FIX)
        labeled_labels = y_all[labeled_idx]
        self.semi_train = SemiSupervisedDataset(
            self.train_labeled, self.train_unlabeled,
            labeled_labels=labeled_labels, p_min=0.5
        )

        # ---- Class weights computed on the labeled subset
        counts = np.bincount(labeled_labels, minlength=NUM_CLASSES).astype(np.float32)  # [N0, N1]
        inv = 1.0 / (counts + 1e-9)
        inv = inv / inv.mean()
        self.ce_weights = torch.tensor(inv, dtype=torch.float32)
        print(f"Labeled counts: {counts.tolist()}  -> CE weights: {inv.tolist()}")

    def train_dataloader(self):
        return DataLoader(self.semi_train, batch_size=self.batch_size, shuffle=True,
                          pin_memory=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          pin_memory=True, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                          pin_memory=True, num_workers=4, persistent_workers=True)

# =======================
# FD targets helper
# =======================
@torch.no_grad()
def batch_fd_targets_from_images(x: torch.Tensor) -> torch.Tensor:
    device = x.device
    mean = torch.tensor(norm_mean, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(norm_std,  device=device).view(1, 3, 1, 1)
    x01  = (x * std + mean).clamp(0, 1)

    fds = []
    for b in range(x01.size(0)):
        img = x01[b].detach().cpu()
        if box_counting is not None:
            fd = float(box_counting(img))
        else:
            g = 0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]
            thr = g.mean().item()
            binimg = (g > thr).float().unsqueeze(0).unsqueeze(0)
            H, W = g.shape
            max_pow = int(np.floor(np.log2(min(H, W))))
            sizes = [2 ** k for k in range(1, max_pow + 1)]
            if len(sizes) < 2:
                fd = 2.0
            else:
                Ns = []
                for s in sizes:
                    pooled = F.max_pool2d(binimg, kernel_size=s, stride=s, ceil_mode=True)
                    Ns.append((pooled > 0).sum().item() + 1e-6)
                logN = np.log(np.array(Ns))
                logr = np.log(np.array([1.0 / s for s in sizes]))
                A = np.vstack([logr, np.ones_like(logr)]).T
                slope = np.linalg.lstsq(A, logN, rcond=None)[0][0]
                fd = float(slope)
        fds.append(fd)
    return torch.tensor(fds, device=device, dtype=torch.float32)

# =======================
# Model (AE + FD head)
# =======================
class FDRegressor(nn.Module):
    def __init__(self, latent_channels=256, p_drop=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(latent_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(64, 1)
        )
    def forward(self, latent):  # latent: [B, C, H, W]
        return self.model(latent).squeeze(1)

class AutoencoderNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 32, stride=1, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, stride=1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, stride=4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, stride=4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        # Decoder
        self.deconv4 = nn.ConvTranspose2d(256, 128, stride=4, kernel_size=3, output_padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, stride=4, kernel_size=3, output_padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, stride=1, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.deconv1 = nn.ConvTranspose2d(32, 3, stride=1, kernel_size=1)
        self.bn8 = nn.BatchNorm2d(3)
        # Classifier head
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 256),
            nn.Linear(256, 64),
            nn.Linear(64, NUM_CLASSES)
        )
        # FD head
        self.fd_head = FDRegressor(latent_channels=256)

    def encode(self, x):
        h = F.relu(self.bn1(self.conv1(x))); out1 = h.detach()
        h = F.relu(self.bn2(self.conv2(h)));  out2 = h.detach()
        h = F.relu(self.bn3(self.conv3(h)));  out3 = h.detach()
        z = F.relu(self.bn4(self.conv4(h)))
        return z, (out1, out2, out3)

    def decode(self, z):
        u = F.relu(self.bn5(self.deconv4(z))); dout3 = u
        u = F.relu(self.bn6(self.deconv3(u))); dout2 = u
        u = F.relu(self.bn7(self.deconv2(u))); dout1 = u
        x_rec = torch.sigmoid(self.bn8(self.deconv1(u)))
        return x_rec, (dout1, dout2, dout3)

    def classify_from_latent(self, z):
        return self.fc(z.view(z.size(0), -1))

    def predict_fd_from_latent(self, z):
        return self.fd_head(z)

class LitFractalAE(LightningModule):
    def __init__(self, rc_rate=RC_RATE, lr=LR, lambda_fd=LAMBDA_FD, ce_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["ce_weights"])
        self.net = AutoencoderNet(num_classes=NUM_CLASSES)

        # ----- class-weighted CE (register buffer so Lightning moves it to GPU) (FIX)
        if ce_weights is None:
            ce_weights = torch.ones(NUM_CLASSES, dtype=torch.float32)
        elif not isinstance(ce_weights, torch.Tensor):
            ce_weights = torch.tensor(ce_weights, dtype=torch.float32)
        self.register_buffer("class_weights", ce_weights)
        self.ce  = nn.CrossEntropyLoss(weight=self.class_weights)

        self.mse = nn.MSELoss()
        self.train_loss_hist, self.val_acc_hist, self.val_loss_hist = [], [], []
        self.test_preds, self.test_targets = [], []

    def forward(self, x):
        z, _ = self.net.encode(x)
        return self.net.classify_from_latent(z)

    def reconstruction_loss(self, x_rec, x_in, out1, dout1, out2, dout2, out3, dout3):
        base = self.mse(x_rec, x_in)
        if self.hparams.rc_rate:
            base = base + self.hparams.rc_rate * (
                self.mse(out1, dout1) + self.mse(out2, dout2) + self.mse(out3, dout3)
            )
        return base

    def _fd_loss(self, x, z):
        fd_target = batch_fd_targets_from_images(x)   # (B,)
        fd_pred   = self.net.predict_fd_from_latent(z)  # (B,)
        return self.mse(fd_pred, fd_target)

    def training_step(self, batch, batch_idx):
        x_l, y_l, x_u = batch
        x_all = torch.cat([x_l, x_u], dim=0)

        z_all, (o1, o2, o3)    = self.net.encode(x_all)
        xrec_all, (d1, d2, d3) = self.net.decode(z_all)
        loss_rec = self.reconstruction_loss(xrec_all, x_all, o1, d1, o2, d2, o3, d3)
        loss_fd  = self._fd_loss(x_all, z_all)

        B_l      = x_l.size(0)
        logits_l = self.net.classify_from_latent(z_all[:B_l])
        loss_ce  = self.ce(logits_l, y_l)

        # >>> use self.hparams.lambda_fd (FIX)
        loss = loss_ce + loss_rec + self.hparams.lambda_fd * loss_fd

        self.log("train/loss_total", loss,    on_epoch=True, prog_bar=True)
        self.log("train/ce",         loss_ce, on_epoch=True)
        self.log("train/rec",        loss_rec,on_epoch=True)
        self.log("train/fd",         loss_fd, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z, (o1, o2, o3)    = self.net.encode(x)
        xrec, (d1, d2, d3) = self.net.decode(z)

        loss_rec = self.reconstruction_loss(xrec, x, o1, d1, o2, d2, o3, d3)
        loss_fd  = self._fd_loss(x, z)
        logits   = self.net.classify_from_latent(z)
        loss_ce  = self.ce(logits, y)

        loss = loss_ce + loss_rec + self.hparams.lambda_fd * loss_fd
        acc  = (logits.argmax(dim=1) == y).float().mean()

        self.log("val/acc",  acc,     on_epoch=True, prog_bar=True)
        self.log("val/ce",   loss_ce, on_epoch=True)
        self.log("val/rec",  loss_rec,on_epoch=True)
        self.log("val/fd",   loss_fd, on_epoch=True)
        self.log("val/loss", loss,    on_epoch=True, prog_bar=True)
        return {"val_loss": loss.detach(), "val_acc": acc.detach()}

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val/loss")
        val_acc  = self.trainer.callback_metrics.get("val/acc")
        if val_loss is not None: self.val_loss_hist.append(float(val_loss.cpu()))
        if val_acc  is not None: self.val_acc_hist.append(float(val_acc.cpu()))
        train_loss = self.trainer.callback_metrics.get("train/loss_total")
        if train_loss is not None: self.train_loss_hist.append(float(train_loss.cpu()))

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x).argmax(dim=1)
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(y.detach().cpu())

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds).numpy()
        targs = torch.cat(self.test_targets).numpy()
        report = classification_report(targs, preds, digits=3)
        print("\n=== TEST CLASSIFICATION REPORT ===\n", report)
        with open(os.path.join("saved_metrics", f"test_report_{MODEL_NAME}.txt"), "w", encoding="utf-8") as f:
            f.write(report)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

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
    dm = ISICDataModule(batch_size=BATCH_SIZE)
    dm.prepare_data(); dm.setup()

    model = LitFractalAE(rc_rate=RC_RATE, lr=LR, lambda_fd=LAMBDA_FD,
                         ce_weights=getattr(dm, "ce_weights", None))

    ckpt  = ModelCheckpoint(
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
    print("CSV:", CSV_PATH)
    print("Images dir exists:", os.path.isdir(IM_DIR))

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm, ckpt_path=ckpt.best_model_path if ckpt.best_model_path else None)

    torch.save(model.state_dict(), os.path.join("models", f"{MODEL_NAME}.pt"))

    plot_and_save(model.train_loss_hist, "Train Loss (total)", "Loss", os.path.join("plots", f"{MODEL_NAME}_train_loss.png"))
    plot_and_save(model.val_loss_hist,   "Validation Loss",   "Loss", os.path.join("plots", f"{MODEL_NAME}_val_loss.png"))
    plot_and_save(model.val_acc_hist,    "Validation Acc",    "Accuracy", os.path.join("plots", f"{MODEL_NAME}_val_acc.png"))

    print("\nSaved:")
    print(" - Best checkpoint:", ckpt.best_model_path if ckpt.best_model_path else "(none)")
    print(" - Latest state_dict: models/{}.pt".format(MODEL_NAME))
    print(" - Plots: plots/{}_train_loss.png, plots/{}_val_loss.png, plots/{}_val_acc.png".format(MODEL_NAME, MODEL_NAME, MODEL_NAME))
    print(" - Test report: models/test_report_{}.txt".format(MODEL_NAME))

if __name__ == "__main__":
    main()
