#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import warnings

from datasets import SemiSupervisedDataset, HAM10000Dataset

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Torch & Lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# Lightning import (v2 or v1 compatibility)
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

# ------------------------
# Config
# ------------------------
DATA_PATH = "F:/datasets/SkinCancer"
IM_DIR = os.path.join(DATA_PATH, "images")
CSV_PATH = os.path.join(DATA_PATH, "GroundTruth.csv")

BATCH_SIZE = 32
IMG_SIZE = (96, 96)
NUM_CLASSES = 2
NUM_EPOCHS = 10
LR = 1e-3
RC_RATE = 0.3
LABELED_FRACTION = 0.05
MODEL_PATH="ae_latest_state_dict.pt"

# нормалізація (твоя попередньо порахована)
norm_mean = [0.76303685, 0.54564613, 0.570045]
norm_std = [0.14092818, 0.15261282, 0.16997021]

# вихідні папки
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Репродюсибельність
seed_everything(10, workers=True)

# ------------------------
# Dataset & transforms
# ------------------------
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

torch.set_float32_matmul_precision('high')

# ------------------------
# Data Module
# ------------------------
class HAMDataModule(LightningDataModule):
    def __init__(self, batch_size=BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # CSV -> DataFrame, формування бінарних міток (NV -> benign, решта -> cancer)
        df = pd.read_csv(CSV_PATH)
        df["image"] = df["image"].astype(str) + ".jpg"

        labels_cols = list(df.columns[1:])
        # мапа у бінарну задачу: NV -> "Melanocytic nevi" (benign), решта -> "Skin cancer"
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
        # Split train/val/test як у тебе (90% / 2.5% / 7.5%)
        train_split = 0.9
        valid_split = 0.025
        valid_split_adj = valid_split / (1 - train_split)

        train_df, val_test_df = train_test_split(self.df, train_size=train_split, random_state=62,
                                                 stratify=self.df["label"])
        val_df, test_df = train_test_split(val_test_df, train_size=valid_split_adj, random_state=62,
                                           stratify=val_test_df["label"])

        # Балансування train (до max_size на клас) — як у тебе
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

        # Datasets
        self.train_full = HAM10000Dataset(train_df, transform=train_transform)
        self.val_set = HAM10000Dataset(val_df, transform=eval_transform)
        self.test_set = HAM10000Dataset(test_df, transform=eval_transform)

        # Розщеплення на labeled/unlabeled
        n_labeled = int(LABELED_FRACTION * len(self.train_full))
        n_unlabeled = len(self.train_full) - n_labeled
        self.train_labeled, self.train_unlabeled = random_split(
            self.train_full, [n_labeled, n_unlabeled],
            generator=torch.Generator().manual_seed(62)
        )

        # Обгортка під напівкероване тренування
        self.semi_train = SemiSupervisedDataset(self.train_labeled, self.train_unlabeled)

    def train_dataloader(self):
        return DataLoader(self.semi_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4)


# ------------------------
# Model (архітектура як у тебе)
# ------------------------
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

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 256),
            nn.Linear(256, 64),
            nn.Linear(64, num_classes)
        )

    def encode(self, x):
        h = F.relu(self.bn1(self.conv1(x)));
        out1 = h.detach()
        h = F.relu(self.bn2(self.conv2(h)));
        out2 = h.detach()
        h = F.relu(self.bn3(self.conv3(h)));
        out3 = h.detach()
        z = F.relu(self.bn4(self.conv4(h)))
        return z, (out1, out2, out3)

    def decode(self, z):
        u = F.relu(self.bn5(self.deconv4(z)));
        dout3 = u
        u = F.relu(self.bn6(self.deconv3(u)));
        dout2 = u
        u = F.relu(self.bn7(self.deconv2(u)));
        dout1 = u
        x_rec = torch.sigmoid(self.bn8(self.deconv1(u)))
        return x_rec, (dout1, dout2, dout3)

    def classify_from_latent(self, z):
        return self.fc(z.view(z.size(0), -1))


class LitAutoencoder(LightningModule):
    def __init__(self, rc_rate=RC_RATE, lr=LR, num_classes=NUM_CLASSES):
        super().__init__()
        self.save_hyperparameters()
        self.net = AutoencoderNet(num_classes=num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

        # історії для графіків
        self.train_loss_hist = []
        self.val_acc_hist = []
        self.val_loss_hist = []

        # для тестового звіту
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        # повертаємо logits для інференсу
        z, _ = self.net.encode(x)
        logits = self.net.classify_from_latent(z)
        return logits

    def reconstruction_loss(self, x_rec, x_in, out1, dout1, out2, dout2, out3, dout3):
        if self.hparams.rc_rate != 0:
            return self.mse(x_rec, x_in) + self.hparams.rc_rate * (self.mse(out1, dout1) + self.mse(out2, dout2) + self.mse(out3, dout3))
        else:
            return self.mse(x_rec, x_in)

    def training_step(self, batch, batch_idx):
        x_l, y_l, x_u = batch

        # Labeled forward
        z_l, (l_out1, l_out2, l_out3) = self.net.encode(x_l)
        xrec_l, (ld1, ld2, ld3) = self.net.decode(z_l)
        logits_l = self.net.classify_from_latent(z_l)

        loss_cls = self.ce(logits_l, y_l)
        loss_rec_l = self.reconstruction_loss(xrec_l, x_l, l_out1, ld1, l_out2, ld2, l_out3, ld3)

        # Unlabeled forward (тільки реконструкція)
        z_u, (u_out1, u_out2, u_out3) = self.net.encode(x_u)
        xrec_u, (ud1, ud2, ud3) = self.net.decode(z_u)
        loss_rec_u = self.reconstruction_loss(xrec_u, x_u, u_out1, ud1, u_out2, ud2, u_out3, ud3)

        loss = loss_cls + loss_rec_l + loss_rec_u

        self.log("train/loss_total", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_cls", loss_cls, on_step=False, on_epoch=True)
        self.log("train/loss_rec_l", loss_rec_l, on_step=False, on_epoch=True)
        self.log("train/loss_rec_u", loss_rec_u, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z, (o1, o2, o3) = self.net.encode(x)
        xrec, (d1, d2, d3) = self.net.decode(z)
        logits = self.net.classify_from_latent(z)

        loss_cls = self.ce(logits, y)
        loss_rec = self.reconstruction_loss(xrec, x, o1, d1, o2, d2, o3, d3)
        loss = loss_cls + loss_rec

        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss.detach(), "val_acc": acc.detach()}

    def on_validation_epoch_end(self):
        # зберігаємо історію для графіків
        val_loss = self.trainer.callback_metrics.get("val/loss")
        val_acc = self.trainer.callback_metrics.get("val/acc")
        if val_loss is not None: self.val_loss_hist.append(float(val_loss.cpu()))
        if val_acc is not None:  self.val_acc_hist.append(float(val_acc.cpu()))

        train_loss = self.trainer.callback_metrics.get("train/loss_total")
        if train_loss is not None: self.train_loss_hist.append(float(train_loss.cpu()))

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        preds = logits.argmax(dim=1)
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(y.detach().cpu())

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds).numpy()
        targs = torch.cat(self.test_targets).numpy()
        report = classification_report(targs, preds, digits=3)
        print("\n=== TEST CLASSIFICATION REPORT ===\n", report)

        with open(os.path.join("models", "test_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return opt


# ------------------------
# Plot helpers
# ------------------------
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


# ------------------------
# Main
# ------------------------
def main():
    # Data
    dm = HAMDataModule(batch_size=BATCH_SIZE)
    dm.prepare_data()
    dm.setup()

    # Model
    model = LitAutoencoder(rc_rate=RC_RATE, lr=LR, num_classes=NUM_CLASSES)

    # Callbacks & logger
    ckpt = ModelCheckpoint(
        dirpath="models",
        filename="ae-rc{rc_rate:.2f}-{epoch:02d}-{val_acc:.4f}",
        monitor="val/acc",
        mode="max",
        save_top_k=1
    )
    lrmon = LearningRateMonitor(logging_interval='epoch')
    logger = CSVLogger("models", name="lightning_logs")

    # Trainer
    trainer = Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="gpu",
        devices=1,
        callbacks=[ckpt, lrmon],
        logger=logger,
        precision=16,
        deterministic=True,
    )

    # Fit + Test
    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm, ckpt_path=ckpt.best_model_path if ckpt.best_model_path else None)

    # Save latest full state_dict as well (зручно для подальших експериментів)
    torch.save(model.state_dict(), os.path.join("models", MODEL_PATH))

    # Plots
    plot_and_save(model.train_loss_hist, "Train Loss (total)", "Loss", os.path.join("plots", "train_loss.png"))
    plot_and_save(model.val_loss_hist, "Validation Loss", "Loss", os.path.join("plots", "val_loss.png"))
    plot_and_save(model.val_acc_hist, "Validation Accuracy", "Accuracy", os.path.join("plots", "val_accuracy.png"))

    print("\nSaved:")
    print(" - Best checkpoint:", ckpt.best_model_path if ckpt.best_model_path else "(none)")
    print(" - Latest state_dict: models/{}".format(MODEL_PATH))
    print(" - Plots: plots/train_loss.png, plots/val_loss.png, plots/val_accuracy.png")
    print(" - Test report: models/{}_test_report.txt".format(MODEL_PATH))

if __name__ == "__main__":
    print("DATA_PATH:", DATA_PATH)
    print(os.listdir(DATA_PATH))
    main()
