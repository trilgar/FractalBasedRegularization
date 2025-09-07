import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import h5py
import numpy as np
from tqdm import tqdm


class ISICDataset(Dataset):
    def __init__(self, metadata, image_dir, transform=None, load_images_to_ram = False):
        self.metadata = metadata
        self.image_dir = image_dir
        self.transform = transform
        self.images = []  # list to store preloaded images
        self.labels = []  # list to store labels
        self.image_ids = []  # list to store image IDs
        self.load_images_to_ram = load_images_to_ram

        if load_images_to_ram:
            print("Preloading images into RAM...")
            for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
                isic_id = row['isic_id']
                label = int(row['target'])
                image_path = os.path.join(self.image_dir, f"{isic_id}.jpg")
                image = Image.open(image_path).convert('RGB')

                if self.transform:
                    image = self.transform(image)

                self.images.append(image)
                self.labels.append(label)
                self.image_ids.append(isic_id)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if self.load_images_to_ram:
            image = self.images[idx]
            label = self.labels[idx]
            isic_id = self.image_ids[idx]
        else:
            row = self.metadata.iloc[idx]
            isic_id = row['isic_id']
            label = int(row['target'])
            img_path = os.path.join(self.image_dir, f"{isic_id}.jpg")
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        return image, label, isic_id

# Простий клас Dataset для завантаження зображень з HDF5
class HDF5ISICDataset(Dataset):
    def __init__(self, hdf5_path, transform=None):
        self.hdf5_path = hdf5_path
        self.file = h5py.File(hdf5_path, 'r')
        self.images = self.file['images']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = np.array(image)

        if self.transform:
            image = self.transform(image)

        return image

class HAM10000Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, root="F:/datasets/SkinCancer/images"):
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
    """Повертає трійку: (labeled_x, labeled_y, unlabeled_x)"""
    def __init__(self, labeled_ds: Dataset, unlabeled_ds: Dataset):
        self.labeled_ds = labeled_ds
        self.unlabeled_ds = unlabeled_ds
        self.labeled_size = len(labeled_ds)
        self.unlabeled_size = len(unlabeled_ds)

    def __len__(self): return self.unlabeled_size

    def __getitem__(self, idx):
        lx, ly = self.labeled_ds[idx % self.labeled_size]
        ux, _  = self.unlabeled_ds[idx]
        return lx, ly, ux