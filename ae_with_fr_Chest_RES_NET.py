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

from torchvision import transforms, models
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report
from PIL import Image

from fd_functions import *  # your FD utils (box_counting etc.)

# =======================
# Utils
# =======================
def fmt_pct_three(val: float, prefix: str) -> str:
    n = int(round(val * 100))
    return f"{prefix}{n:03d}"

# =======================
# Config (Chest X-ray)
# =======================
DATA_PATH   = "F:/datasets/Chest X-Ray"
TRAIN_DIR   = os.path.join(DATA_PATH, "train")
TEST_DIR    = os.path.join(DATA_PATH, "test")
TRAIN_LIST  = os.path.join(DATA_PATH, "train.txt")
TEST_LIST   = os.path.join(DATA_PATH, "test.txt")

IMG_SIZE         = (512, 512)    # <-- requested 512x512
BATCH_SIZE       = 16
NUM_CLASSES      = 2
NUM_EPOCHS       = 1
LR               = 2e-4

# Semi-supervised + regularization
LABELED_FRACTION = 0.05
LAMBDA_FD        = 10
RC_RATE          = 0.1

r_str = fmt_pct_three(RC_RATE, "r")
m_str = fmt_pct_three(LABELED_FRACTION, "m")
MODEL_NAME = f"fd_resnetAE_512_l{LAMBDA_FD}_{m_str}_{r_str}"

# ImageNet normalization
norm_mean = [0.485, 0.456, 0.406]
norm_std  = [0.229, 0.224, 0.225]

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("saved_metrics", exist_ok=True)

seed_everything(10, workers=True)
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# =======================
# Transforms & dataset
# =======================
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
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

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, row["filename"])
        X = Image.open(img_path)
        if X.mode != "RGB":  # many chest X-rays are single-channel
            X = X.convert("RGB")
        y = int(row["label_idx"])
        if self.transform is not None:
            X = self.transform(X)
        return X, torch.tensor(y, dtype=torch.long)

class SemiSupervisedDataset(Dataset):
    """Returns (labeled_x, labeled_y, unlabeled_x) per item."""
    def __init__(self, labeled_ds: Dataset, unlabeled_ds: Dataset, labeled_labels, p_min=0.5):
        self.labeled_ds = labeled_ds
        self.unlabeled_ds = unlabeled_ds
        labeled_labels = np.asarray(labeled_labels)
        self.min_idx = np.where(labeled_labels == 1)[0].astype(np.int64)
        self.maj_idx = np.where(labeled_labels == 0)[0].astype(np.int64)
        self.p_min = float(p_min)

    def __len__(self): return len(self.unlabeled_ds)

    def __getitem__(self, idx):
        if (len(self.min_idx) > 0) and (np.random.rand() < self.p_min):
            li = np.random.choice(self.min_idx)
        else:
            li = np.random.choice(self.maj_idx)
        lx, ly = self.labeled_ds[int(li)]
        ux, _  = self.unlabeled_ds[idx]
        return lx, ly, ux

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
        self.df_test_all  = ts[["filename", "label_idx"]].reset_index(drop=True)

    def setup(self, stage=None):
        # stratified train/val split from train list; test from test list
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

        self.train_full = ChestXRayClsDataset(train_df, transform=train_transform, root=TRAIN_DIR)
        self.val_set    = ChestXRayClsDataset(val_df,   transform=eval_transform,  root=TRAIN_DIR)
        self.test_set   = ChestXRayClsDataset(self.df_test_all, transform=eval_transform, root=TEST_DIR)

        # labeled/unlabeled split for semi-supervised training
        n_labeled = max(1, int(LABELED_FRACTION * len(train_df)))
        idx_all = np.arange(len(train_df))
        y_tr    = train_df["label_idx"].to_numpy()
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_labeled, random_state=62)
        labeled_idx, _ = next(sss.split(idx_all, y_tr))
        unlabeled_idx  = np.setdiff1d(idx_all, labeled_idx)

        self.train_labeled   = Subset(self.train_full, labeled_idx.tolist())
        self.train_unlabeled = Subset(self.train_full, unlabeled_idx.tolist())
        labeled_labels = y_tr[labeled_idx]

        self.semi_train = SemiSupervisedDataset(
            self.train_labeled, self.train_unlabeled, labeled_labels=labeled_labels, p_min=0.5
        )

        # class weights from labeled subset (to keep parity with your working setup)
        counts = np.bincount(labeled_labels, minlength=NUM_CLASSES).astype(np.float32)  # [N0, N1]
        inv = 1.0 / (counts + 1e-9)
        inv = inv / inv.mean()
        self.ce_weights = torch.tensor(inv, dtype=torch.float32)
        print(f"[ChestXRay] Labeled counts: {counts.tolist()} -> CE weights: {inv.tolist()}")
        print(f"[ChestXRay] Train/Val/Test sizes: {len(train_df)}/{len(val_df)}/{len(self.df_test_all)}")

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
# Fractal targets
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
# Model: ResNet18-based AE (unchanged core)
# =======================
class FDRegressor(nn.Module):
    def __init__(self, latent_channels=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(latent_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
    def forward(self, z): return self.model(z).squeeze(1)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class ResNet18AE(nn.Module):
    """Encoder: torchvision resnet18; Decoder: light pyramid to 512x512."""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            b = models.resnet18(weights=weights)
        except Exception:
            b = models.resnet18(pretrained=pretrained)

        # encoder
        self.enc_conv1 = b.conv1; self.enc_bn1 = b.bn1; self.enc_relu = b.relu; self.enc_maxp = b.maxpool
        self.l1 = b.layer1; self.l2 = b.layer2; self.l3 = b.layer3; self.l4 = b.layer4

        # decoder to 512x512
        self.dec4 = UpBlock(512, 256)  # 1/32 -> 1/16
        self.dec3 = UpBlock(256, 128)  # 1/16 -> 1/8
        self.dec2 = UpBlock(128, 64)   # 1/8  -> 1/4
        self.dec1 = UpBlock(64, 32)    # 1/4  -> 1/2
        self.dec0 = UpBlock(32, 16)    # 1/2  -> 1
        self.out  = nn.Conv2d(16, 3, 1)

        # adapters for RC loss
        self.ad3 = nn.Conv2d(256, 256, 1)
        self.ad2 = nn.Conv2d(128, 128, 1)
        self.ad1 = nn.Conv2d(64, 64, 1)

        # heads
        self.cls_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes))
        self.fd_head  = FDRegressor(512)

    def encode(self, x):
        x = self.enc_relu(self.enc_bn1(self.enc_conv1(x))); x = self.enc_maxp(x)
        o1 = self.l1(x)   # 64,  1/4
        o2 = self.l2(o1)  # 128, 1/8
        o3 = self.l3(o2)  # 256, 1/16
        z  = self.l4(o3)  # 512, 1/32
        return z, (o1.detach(), o2.detach(), o3.detach())

    def decode(self, z):
        d3 = self.dec4(z)
        d2 = self.dec3(d3)
        d1 = self.dec2(d2)
        dout3, dout2, dout1 = self.ad3(d3), self.ad2(d2), self.ad1(d1)
        u = self.dec1(d1); u = self.dec0(u)
        xrec = torch.sigmoid(self.out(u))
        return xrec, (dout1, dout2, dout3)

    def classify_from_latent(self, z): return self.cls_head(z)
    def predict_fd_from_latent(self, z): return self.fd_head(z)

# =======================
# Lightning module
# =======================
class LitFractalAE(LightningModule):
    def __init__(self, rc_rate=RC_RATE, lr=LR, lambda_fd=LAMBDA_FD, ce_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["ce_weights"])
        self.net = ResNet18AE(num_classes=NUM_CLASSES, pretrained=True)

        # class-weighted CE
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

    def reconstruction_loss(self, x_rec, x_in, o1, d1, o2, d2, o3, d3):
        base = self.mse(x_rec, x_in)
        if self.hparams.rc_rate:
            base = base + self.hparams.rc_rate * (
                self.mse(o1, d1) + self.mse(o2, d2) + self.mse(o3, d3)
            )
        return base

    def _fd_loss(self, x, z):
        fd_t = batch_fd_targets_from_images(x)
        fd_p = self.net.predict_fd_from_latent(z)
        return self.mse(fd_p, fd_t)

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
    dm = ChestXRayDataModule(batch_size=BATCH_SIZE)
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
    print("Train dir exists:", os.path.isdir(TRAIN_DIR))
    print("Test dir exists:", os.path.isdir(TEST_DIR))
    print("TRAIN_LIST exists:", os.path.isfile(TRAIN_LIST))
    print("TEST_LIST exists:", os.path.isfile(TEST_LIST))

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm, ckpt_path=ckpt.best_model_path if ckpt.best_model_path else None)

    torch.save(model.state_dict(), os.path.join("models", f"{MODEL_NAME}.pt"))

    plot_and_save(model.train_loss_hist, "Train Loss (total)", "Loss", os.path.join("plots", f"{MODEL_NAME}_train_loss.png"))
    plot_and_save(model.val_loss_hist,   "Validation Loss",   "Loss", os.path.join("plots", f"{MODEL_NAME}_val_loss.png"))
    plot_and_save(model.val_acc_hist,    "Validation Acc",    "Accuracy", os.path.join("plots", f"{MODEL_NAME}_val_acc.png"))

    print("\nSaved:")
    print(" - Best checkpoint:", ckpt.best_model_path if ckpt.best_model_path else "(none)")
    print(" - Latest state_dict:", f"models/{MODEL_NAME}.pt")
    print(" - Plots:", f"plots/{MODEL_NAME}_train_loss.png, plots/{MODEL_NAME}_val_loss.png, plots/{MODEL_NAME}_val_acc.png")
    print(" - Test report:", f"saved_metrics/test_report_{MODEL_NAME}.txt")

if __name__ == "__main__":
    main()
