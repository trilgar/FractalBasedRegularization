#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, warnings

from datasets import HAM10000Dataset, SemiSupervisedDataset

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from fd_functions import *

# Torch & Lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

try:
    from lightning import LightningModule, LightningDataModule, Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import CSVLogger
except Exception:
    from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger

from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =======================
# Config
# =======================
DATA_PATH = "F:/datasets/SkinCancer"
IM_DIR = os.path.join(DATA_PATH, "images")
CSV_PATH = os.path.join(DATA_PATH, "GroundTruth.csv")

BATCH_SIZE = 256
IMG_SIZE = (96, 96)
NUM_CLASSES = 2
NUM_EPOCHS = 10
LR = 1e-3
RC_RATE = 0.3              # вага проміжних MSE (out<->dout) усередині reconstruction_loss
LABELED_FRACTION = 0.05
MODEL_PATH = "ae_fr_005.pt"

# Лише одна вага для FD (λ). Реконструкцію не масштабуємо додатково — як у базі.
LAMBDA_FD = 0.5

# нормалізація з базової моделі
norm_mean = [0.76303685, 0.54564613, 0.570045]
norm_std  = [0.14092818, 0.15261282, 0.16997021]

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

# =======================
# DataModule
# =======================
class HAMDataModule(LightningDataModule):
    def __init__(self, batch_size=BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        df = pd.read_csv(CSV_PATH)
        df["image"] = df["image"].astype(str) + ".jpg"

        labels_cols = list(df.columns[1:])
        label_list = []
        for i in range(len(df)):
            row = list(df.iloc[i])[1:]
            idx = int(np.argmax(row))
            label = "Melanocytic nevi" if labels_cols[idx] == "NV" else "Skin cancer"
            label_list.append(label)
        df["label"] = label_list
        df = df.drop(labels_cols, axis=1)
        df["label_idx"] = pd.Categorical(df["label"]).codes
        df.drop_duplicates(inplace=True)
        self.df = df

    def setup(self, stage=None):
        train_split = 0.9
        valid_split = 0.025
        valid_split_adj = valid_split / (1 - train_split)

        train_df, val_test_df = train_test_split(self.df, train_size=train_split, random_state=62,
                                                 stratify=self.df["label"])
        val_df, test_df = train_test_split(val_test_df, train_size=valid_split_adj, random_state=62,
                                           stratify=val_test_df["label"])

        # Балансування train
        max_size = 7000
        samples = []
        group = train_df.groupby('label')
        for label in train_df['label'].unique():
            Lgroup = group.get_group(label)
            count = int(Lgroup['label'].value_counts())
            if count >= max_size:
                sample = Lgroup.sample(max_size, random_state=62)
            else:
                sample = Lgroup.sample(frac=1.0, random_state=62)
            samples.append(sample)
        train_df = pd.concat(samples, axis=0).reset_index(drop=True)

        self.train_full = HAM10000Dataset(train_df, transform=train_transform, root=IM_DIR)
        self.val_set    = HAM10000Dataset(val_df,   transform=eval_transform, root=IM_DIR)
        self.test_set   = HAM10000Dataset(test_df,  transform=eval_transform, root=IM_DIR)

        n_labeled = int(LABELED_FRACTION * len(self.train_full))
        n_unlabeled = len(self.train_full) - n_labeled
        self.train_labeled, self.train_unlabeled = random_split(
            self.train_full, [n_labeled, n_unlabeled],
            generator=torch.Generator().manual_seed(62)
        )
        self.semi_train = SemiSupervisedDataset(self.train_labeled, self.train_unlabeled)

    def train_dataloader(self):
        return DataLoader(self.semi_train, batch_size=self.batch_size, shuffle=True,
                          pin_memory=True, num_workers=8, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          pin_memory=True, num_workers=8, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                          pin_memory=True, num_workers=8, persistent_workers=True)

@torch.no_grad()
def batch_fd_targets_from_images(x: torch.Tensor) -> torch.Tensor:
    """
    Обчислює FD для *кожного зображення в батчі* через твій box_counting.
    Очікує x нормалізований як у train (mean/std). Повертає FD на тому ж пристрої.
    """
    device = x.device
    mean = torch.tensor(norm_mean, device=device).view(1,3,1,1)
    std  = torch.tensor(norm_std,  device=device).view(1,3,1,1)
    x01 = (x*std + mean).clamp(0,1)

    fds = []
    for b in range(x01.size(0)):
        img = x01[b].detach().cpu()  # [C,H,W] CPU
        if box_counting is not None:
            fd = float(box_counting(img))  # функція всередині сама викличе .numpy() при потребі
        else:
            # резервний спрощений підрахунок (на випадок, якщо модуль відсутній)
            # бінаризація за середнім і box-counting через max-pool
            g = 0.2989*img[0] + 0.5870*img[1] + 0.1140*img[2]
            thr = g.mean().item()
            binimg = (g > thr).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            H, W = g.shape
            max_pow = int(np.floor(np.log2(min(H,W))))
            sizes = [2**k for k in range(1, max_pow+1)]
            if len(sizes) < 2:
                fd = 2.0
            else:
                Ns = []
                for s in sizes:
                    pooled = F.max_pool2d(binimg, kernel_size=s, stride=s, ceil_mode=True)
                    N = (pooled > 0).sum().item() + 1e-6
                    Ns.append(N)
                logN = np.log(np.array(Ns))
                logr = np.log(np.array([1.0/s for s in sizes]))
                # лінійна регресія
                A = np.vstack([logr, np.ones_like(logr)]).T
                slope = np.linalg.lstsq(A, logN, rcond=None)[0][0]
                fd = float(slope)
        fds.append(fd)
    return torch.tensor(fds, device=device, dtype=torch.float32)

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

# =======================
# Model (AE + FD head)
# =======================
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
        # FD head (скаляр)
        self.fd_head = FDRegressor(latent_channels=256)

    def encode(self, x):
        h = F.relu(self.bn1(self.conv1(x))); out1 = h.detach()
        h = F.relu(self.bn2(self.conv2(h))); out2 = h.detach()
        h = F.relu(self.bn3(self.conv3(h))); out3 = h.detach()
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
    def __init__(self, rc_rate=RC_RATE, lr=LR, lambda_fd=LAMBDA_FD):
        super().__init__()
        self.save_hyperparameters()
        self.net = AutoencoderNet(num_classes=NUM_CLASSES)
        self.ce  = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

        self.train_loss_hist, self.val_acc_hist, self.val_loss_hist = [], [], []
        self.test_preds, self.test_targets = [], []

    def forward(self, x):
        z, _ = self.net.encode(x)
        return self.net.classify_from_latent(z)

    def reconstruction_loss(self, x_rec, x_in, out1, dout1, out2, dout2, out3, dout3):
        # базова MSE по пікселях + (опційно) проміжні рівні з вагою rc_rate
        base = self.mse(x_rec, x_in)
        if self.hparams.rc_rate:
            base = base + self.hparams.rc_rate * (self.mse(out1, dout1) + self.mse(out2, dout2) + self.mse(out3, dout3))
        return base

    def _fd_loss(self, x, z):
        fd_target = batch_fd_targets_from_images(x)   # (B,)
        fd_pred   = self.net.predict_fd_from_latent(z)  # (B,)
        return self.mse(fd_pred, fd_target)

    def training_step(self, batch, batch_idx):
        x_l, y_l, x_u = batch
        # об'єднуємо в один батч для реконструкції та FD (без розділення)
        x_all = torch.cat([x_l, x_u], dim=0)

        z_all, (o1,o2,o3) = self.net.encode(x_all)
        xrec_all, (d1,d2,d3) = self.net.decode(z_all)

        loss_rec = self.reconstruction_loss(xrec_all, x_all, o1,d1, o2,d2, o3,d3)
        loss_fd  = self._fd_loss(x_all, z_all)

        # CE тільки по labeled частині
        B_l = x_l.size(0)
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
        z, (o1,o2,o3) = self.net.encode(x)
        xrec, (d1,d2,d3) = self.net.decode(z)

        loss_rec = self.reconstruction_loss(xrec, x, o1,d1, o2,d2, o3,d3)
        loss_fd  = self._fd_loss(x, z)
        logits   = self.net.classify_from_latent(z)
        loss_ce  = self.ce(logits, y)

        loss = loss_ce + loss_rec + self.hparams.lambda_fd * loss_fd
        acc  = (logits.argmax(dim=1) == y).float().mean()

        self.log("val/acc",  acc,      on_epoch=True, prog_bar=True)
        self.log("val/ce",   loss_ce,  on_epoch=True)
        self.log("val/rec",  loss_rec, on_epoch=True)
        self.log("val/fd",   loss_fd,  on_epoch=True)
        self.log("val/loss", loss,     on_epoch=True, prog_bar=True)
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
        with open(os.path.join("models", "test_report_{}.txt".format(MODEL_PATH)), "w", encoding="utf-8") as f:
            f.write(report)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# =======================
# Plot helpers
# =======================
def plot_and_save(history, title, ylabel, path_png):
    plt.figure(figsize=(8,5))
    plt.plot(history)
    plt.title(title); plt.xlabel("Epoch"); plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path_png, dpi=160); plt.close()

# =======================
# Main
# =======================
def main():
    dm = HAMDataModule(batch_size=BATCH_SIZE)
    dm.prepare_data(); dm.setup()

    model = LitFractalAE(rc_rate=RC_RATE, lr=LR, lambda_fd=LAMBDA_FD)

    ckpt = ModelCheckpoint(
        dirpath="models",
        filename="ae-fractal-unified-{epoch:02d}-{val_acc:.4f}",
        monitor="val/acc", mode="max", save_top_k=1
    )
    lrmon = LearningRateMonitor(logging_interval='epoch')
    logger = CSVLogger("models", name="lightning_logs_fractal_unified")

    trainer = Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="gpu",
        devices=1,
        callbacks=[ckpt, lrmon],
        logger=logger,
        precision=16,
        deterministic=True,
    )

    print("DATA_PATH:", DATA_PATH); print(os.listdir(DATA_PATH))
    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm, ckpt_path=ckpt.best_model_path if ckpt.best_model_path else None)

    torch.save(model.state_dict(), os.path.join("models", MODEL_PATH))

    plot_and_save(model.train_loss_hist, "Train Loss (total)", "Loss", os.path.join("plots", "fru_train_loss.png"))
    plot_and_save(model.val_loss_hist,   "Validation Loss",   "Loss", os.path.join("plots", "fru_val_loss.png"))
    plot_and_save(model.val_acc_hist,    "Validation Acc",    "Accuracy", os.path.join("plots", "fru_val_acc.png"))

    print("\nSaved:")
    print(" - Best checkpoint:", ckpt.best_model_path if ckpt.best_model_path else "(none)")
    print(" - Latest state_dict: models/{}".format(MODEL_PATH))
    print(" - Plots: plots/fru_train_loss.png, plots/fru_val_loss.png, plots/fru_val_acc.png")
    print(" - Test report: models/test_report_fractal_unified.txt")

if __name__ == "__main__":
    main()
