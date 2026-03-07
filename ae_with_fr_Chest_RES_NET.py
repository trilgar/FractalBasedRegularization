#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, warnings, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Torch & Lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from torchmetrics import FBetaScore
from PIL import Image

# =======================
# Utils
# =======================
def fmt_pct_three(val: float, prefix: str) -> str:
    n = int(round(val * 100))
    return f"{prefix}{n:03d}"


# =======================
# Config (Kaggle Version)
# =======================
DATA_PATH = "F:/datasets/Chest X-Ray"
TRAIN_DIR = os.path.join(DATA_PATH, "train")
TEST_DIR = os.path.join(DATA_PATH, "test")

TRAIN_CSV = os.path.join(DATA_PATH, "train_with_fd.csv")
TEST_CSV = os.path.join(DATA_PATH, "test_with_fd.csv")

IMG_SIZE = (512, 512)
BATCH_SIZE = 16
NUM_CLASSES = 2
NUM_EPOCHS = 10
LR = 1e-3

# Semi-supervised + regularization static parameters
LABELED_FRACTION = 0.05
RC_RATE = 0.1

# ImageNet normalization
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

# Kaggle Output Directories (must be in /kaggle/working/)
OUTPUT_DIR = "."
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
METRICS_DIR = os.path.join(OUTPUT_DIR, "saved_metrics")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

seed_everything(10, workers=True)
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# =======================
# Transforms & dataset
# =======================
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

_LABEL_MAP = {"positive": 1, "negative": 0}


class ChestXRayClsDataset(Dataset):
    """Returns (X, y, fd) from folders and list files."""

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
        fd = float(row["fd_norm"])

        if self.transform is not None:
            X = self.transform(X)

        return X, torch.tensor(y, dtype=torch.long), torch.tensor(fd, dtype=torch.float32)


class SemiSupervisedDataset(Dataset):
    """Returns (labeled_x, labeled_y, labeled_fd, unlabeled_x, unlabeled_fd) per item."""

    def __init__(self, labeled_ds: Dataset, unlabeled_ds: Dataset, labeled_labels, p_min=0.5):
        self.labeled_ds = labeled_ds
        self.unlabeled_ds = unlabeled_ds
        labeled_labels = np.asarray(labeled_labels)
        self.min_idx = np.where(labeled_labels == 1)[0].astype(np.int64)
        self.maj_idx = np.where(labeled_labels == 0)[0].astype(np.int64)
        self.p_min = float(p_min)

    def __len__(self):
        return len(self.unlabeled_ds)

    def __getitem__(self, idx):
        if (len(self.min_idx) > 0) and (np.random.rand() < self.p_min):
            li = np.random.choice(self.min_idx)
        else:
            li = np.random.choice(self.maj_idx)

        lx, ly, lfd = self.labeled_ds[int(li)]
        ux, _, ufd = self.unlabeled_ds[idx]

        return lx, ly, lfd, ux, ufd


# =======================
# DataModule
# =======================
class ChestXRayDataModule(LightningDataModule):
    def __init__(self, batch_size=BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size
        self.ce_weights = None

    def prepare_data(self):
        tr = pd.read_csv(TRAIN_CSV)
        ts = pd.read_csv(TEST_CSV)

        assert len(tr) > 0, f"No training entries in {TRAIN_CSV}"

        tr["label_idx"] = tr["label_str"].str.lower().map(_LABEL_MAP).astype(int)
        ts["label_idx"] = ts["label_str"].str.lower().map(_LABEL_MAP).astype(int)

        self.df_train_all = tr[["filename", "label_idx", "fd_norm"]].reset_index(drop=True)
        self.df_test_all = ts[["filename", "label_idx", "fd_norm"]].reset_index(drop=True)

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

        self.train_full = ChestXRayClsDataset(train_df, transform=train_transform, root=TRAIN_DIR)
        self.val_set = ChestXRayClsDataset(val_df, transform=eval_transform, root=TRAIN_DIR)
        self.test_set = ChestXRayClsDataset(self.df_test_all, transform=eval_transform, root=TEST_DIR)

        n_labeled = max(1, int(LABELED_FRACTION * len(train_df)))
        idx_all = np.arange(len(train_df))
        y_tr = train_df["label_idx"].to_numpy()

        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_labeled, random_state=62)
        labeled_idx, _ = next(sss.split(idx_all, y_tr))
        unlabeled_idx = np.setdiff1d(idx_all, labeled_idx)

        self.train_labeled = Subset(self.train_full, labeled_idx.tolist())
        self.train_unlabeled = Subset(self.train_full, unlabeled_idx.tolist())
        labeled_labels = y_tr[labeled_idx]

        self.semi_train = SemiSupervisedDataset(
            self.train_labeled, self.train_unlabeled, labeled_labels=labeled_labels, p_min=0.5
        )

        counts = np.bincount(labeled_labels, minlength=NUM_CLASSES).astype(np.float32)
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
# Model: ResNet18-based AE
# =======================
class FDRegressor(nn.Module):
    def __init__(self, latent_channels=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
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

        self.enc_conv1 = b.conv1
        self.enc_bn1 = b.bn1
        self.enc_relu = b.relu
        self.enc_maxp = b.maxpool
        self.l1 = b.layer1
        self.l2 = b.layer2
        self.l3 = b.layer3
        self.l4 = b.layer4

        self.dec4 = UpBlock(512, 256)  # 1/32 -> 1/16
        self.dec3 = UpBlock(256, 128)  # 1/16 -> 1/8
        self.dec2 = UpBlock(128, 64)  # 1/8  -> 1/4
        self.dec1 = UpBlock(64, 32)  # 1/4  -> 1/2
        self.dec0 = UpBlock(32, 16)  # 1/2  -> 1
        self.out = nn.Conv2d(16, 3, 1)

        self.ad3 = nn.Conv2d(256, 256, 1)
        self.ad2 = nn.Conv2d(128, 128, 1)
        self.ad1 = nn.Conv2d(64, 64, 1)

        self.cls_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes))
        self.fd_head = FDRegressor(512)

    def encode(self, x):
        x = self.enc_relu(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_maxp(x)
        o1 = self.l1(x)  # 64,  1/4
        o2 = self.l2(o1)  # 128, 1/8
        o3 = self.l3(o2)  # 256, 1/16
        z = self.l4(o3)  # 512, 1/32
        return z, (o1.detach(), o2.detach(), o3.detach())

    def decode(self, z):
        d3 = self.dec4(z)
        d2 = self.dec3(d3)
        d1 = self.dec2(d2)
        dout3, dout2, dout1 = self.ad3(d3), self.ad2(d2), self.ad1(d1)
        u = self.dec1(d1)
        u = self.dec0(u)
        xrec = torch.sigmoid(self.out(u))
        return xrec, (dout1, dout2, dout3)

    def classify_from_latent(self, z):
        return self.cls_head(z)

    def predict_fd_from_latent(self, z):
        return self.fd_head(z)


# =======================
# Lightning module
# =======================
class LitFractalAE(LightningModule):
    def __init__(self, rc_rate=RC_RATE, lr=LR, lambda_fd=40, ce_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["ce_weights"])
        self.net = ResNet18AE(num_classes=NUM_CLASSES, pretrained=True)

        if ce_weights is None:
            ce_weights = torch.ones(NUM_CLASSES, dtype=torch.float32)
        elif not isinstance(ce_weights, torch.Tensor):
            ce_weights = torch.tensor(ce_weights, dtype=torch.float32)

        self.register_buffer("class_weights", ce_weights)
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)
        self.mse = nn.MSELoss()

        self.f2_metric = FBetaScore(task="multiclass", num_classes=NUM_CLASSES, beta=2.0, average="macro")

        self.train_loss_hist, self.val_f2_hist, self.val_loss_hist = [], [], []
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

    def training_step(self, batch, batch_idx):
        x_l, y_l, fd_l, x_u, fd_u = batch

        x_all = torch.cat([x_l, x_u], dim=0)
        fd_t_all = torch.cat([fd_l, fd_u], dim=0)

        z_all, (o1, o2, o3) = self.net.encode(x_all)
        xrec_all, (d1, d2, d3) = self.net.decode(z_all)

        loss_rec = self.reconstruction_loss(xrec_all, x_all, o1, d1, o2, d2, o3, d3)

        fd_p_all = self.net.predict_fd_from_latent(z_all)
        loss_fd = self.mse(fd_p_all, fd_t_all)

        B_l = x_l.size(0)
        logits_l = self.net.classify_from_latent(z_all[:B_l])
        loss_ce = self.ce(logits_l, y_l)

        loss = loss_ce + loss_rec + self.hparams.lambda_fd * loss_fd

        self.log("train/loss_total", loss, on_epoch=True, prog_bar=True)
        self.log("train/ce", loss_ce, on_epoch=True)
        self.log("train/rec", loss_rec, on_epoch=True)
        self.log("train/fd", loss_fd, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, fd_t = batch

        z, (o1, o2, o3) = self.net.encode(x)
        xrec, (d1, d2, d3) = self.net.decode(z)

        loss_rec = self.reconstruction_loss(xrec, x, o1, d1, o2, d2, o3, d3)

        fd_p = self.net.predict_fd_from_latent(z)
        loss_fd = self.mse(fd_p, fd_t)

        logits = self.net.classify_from_latent(z)
        loss_ce = self.ce(logits, y)

        loss = loss_ce + loss_rec + self.hparams.lambda_fd * loss_fd

        preds_class = logits.argmax(dim=1)
        f2 = self.f2_metric(preds_class, y)

        self.log("val/f2", f2, on_epoch=True, prog_bar=True)
        self.log("val/ce", loss_ce, on_epoch=True)
        self.log("val/rec", loss_rec, on_epoch=True)
        self.log("val/fd", loss_fd, on_epoch=True)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        return {"val_loss": loss.detach(), "val_f2": f2.detach()}

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val/loss")
        val_f2 = self.trainer.callback_metrics.get("val/f2")

        if val_loss is not None: self.val_loss_hist.append(float(val_loss.cpu()))
        if val_f2 is not None: self.val_f2_hist.append(float(val_f2.cpu()))

        train_loss = self.trainer.callback_metrics.get("train/loss_total")
        if train_loss is not None: self.train_loss_hist.append(float(train_loss.cpu()))

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        preds = self.forward(x).argmax(dim=1)
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(y.detach().cpu())

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds).numpy()
        targs = torch.cat(self.test_targets).numpy()

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

        # Save report text to the Kaggle METRICS_DIR
        with open(os.path.join(METRICS_DIR, f"test_report_{self.hparams.model_name}.txt"), "w",
                  encoding="utf-8") as f:
            f.write(final_report)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.2,
            patience=2,
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
# Experiment Runner
# =======================
def run_experiment(lambda_fd: int, dm: ChestXRayDataModule):
    print(f"\n{'=' * 60}")
    print(f"STARTING EXPERIMENT WITH LAMBDA_FD = {lambda_fd}")
    print(f"{'=' * 60}\n")

    # Generate dynamic model name
    r_str = fmt_pct_three(RC_RATE, "r")
    m_str = fmt_pct_three(LABELED_FRACTION, "m")
    current_model_name = f"fd_resnetAE_512_l{lambda_fd}_{m_str}_{r_str}"

    model = LitFractalAE(rc_rate=RC_RATE, lr=LR, lambda_fd=lambda_fd,
                         ce_weights=getattr(dm, "ce_weights", None))
    # Pass the name to hparams so the test step can use it to save the text report
    model.hparams.model_name = current_model_name

    # Save checkpoints to the Kaggle MODELS_DIR
    ckpt = ModelCheckpoint(
        dirpath=MODELS_DIR,
        filename=current_model_name + "-{epoch:02d}-{val_f2:.4f}",
        monitor="val/f2", mode="max", save_top_k=1
    )

    lrmon = LearningRateMonitor(logging_interval='epoch')
    # Save logs to the Kaggle MODELS_DIR
    logger = CSVLogger(MODELS_DIR, name=f"lightning_logs_{current_model_name}")

    trainer = Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="gpu",
        devices=1,
        callbacks=[ckpt, lrmon],
        logger=logger,
        precision="16-mixed",
        deterministic=True,
    )

    # Train and test
    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm, ckpt_path=ckpt.best_model_path if ckpt.best_model_path else "best")

    # Save model weights to Kaggle MODELS_DIR
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{current_model_name}.pt"))

    # Generate plots and save to Kaggle PLOTS_DIR
    plot_and_save(model.train_loss_hist, f"Train Loss (L_FD={lambda_fd})", "Loss",
                  os.path.join(PLOTS_DIR, f"{current_model_name}_train_loss.png"))
    plot_and_save(model.val_loss_hist, f"Validation Loss (L_FD={lambda_fd})", "Loss",
                  os.path.join(PLOTS_DIR, f"{current_model_name}_val_loss.png"))
    plot_and_save(model.val_f2_hist, f"Validation F2 Score (L_FD={lambda_fd})", "F2 Score",
                  os.path.join(PLOTS_DIR, f"{current_model_name}_val_f2.png"))

    print(f"\nSaved assets for LAMBDA_FD = {lambda_fd}:")
    print(f" - Best checkpoint: {ckpt.best_model_path if ckpt.best_model_path else '(none)'}")
    print(f" - Latest state_dict: {MODELS_DIR}/{current_model_name}.pt")

    # Cleanup memory for the next loop
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


# =======================
# Main
# =======================
def main():
    print("DATA_PATH:", DATA_PATH)
    print("Train dir exists:", os.path.isdir(TRAIN_DIR))
    print("Test dir exists:", os.path.isdir(TEST_DIR))
    print("TRAIN_CSV exists:", os.path.isfile(TRAIN_CSV))
    print("TEST_CSV exists:", os.path.isfile(TEST_CSV))

    # Initialize and prepare DataModule ONLY ONCE
    print("\nPreparing DataModule...")
    dm = ChestXRayDataModule(batch_size=BATCH_SIZE)
    dm.prepare_data()
    dm.setup()

    # Define the list of lambda parameters to test
    lambda_values = [5]

    # Run them sequentially
    for l_fd in lambda_values:
        run_experiment(l_fd, dm)

    print("\nAll experiments completed successfully!")


if __name__ == "__main__":
    main()