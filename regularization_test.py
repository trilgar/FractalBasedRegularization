import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import umap
from autoencoders import Autoencoder
from datasets import ISICDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolorsw
from envs.current_env import CURRENT_DEVICE, IMAGE_DIR, METADATA_FILE

# === Конфігурація шляхів та параметрів ===
TEST_METADATA_FILE = METADATA_FILE
MODEL_PATH = "./models/AFLREG_005.pth"
BATCH_SIZE = 16

device = torch.device(CURRENT_DEVICE)
print(f'Device: {device}')

# === Завантажуємо модель ===
autoencoder = Autoencoder().to(device)
autoencoder.load_state_dict(torch.load(MODEL_PATH))
autoencoder.eval()

# === Трансформації для датасету ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Завантажуємо метадані ===
test_metadata = pd.read_csv(TEST_METADATA_FILE)

# Датасет та DataLoader
test_dataset = ISICDataset(test_metadata, IMAGE_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Отримуємо латентні вектори ===
latent_vectors = []
labels = []

autoencoder.eval()

with torch.no_grad():
    for images, batch_labels, _ in tqdm(test_loader):
        images = images.to(device)
        encoded = autoencoder.encoder(images)
        encoded = torch.flatten(encoded, start_dim=1)  # робимо вектор плоским
        latent_vectors.append(encoded.cpu().numpy())
        labels.extend(batch_labels.numpy())

# Конвертуємо у масиви numpy
latent_vectors = np.concatenate(latent_vectors, axis=0)
labels = np.array(labels)

print(f"Latent vectors shape: {latent_vectors.shape}")
print(f"Labels shape: {labels.shape}")

# === Застосовуємо UMAP ===
reducer = umap.UMAP(n_components=2,
                    random_state=42,
                    n_neighbors=15,
                    min_dist=0.01)
embedding = reducer.fit_transform(latent_vectors)

plt.figure(figsize=(12, 10))

# Вибираємо чітко розрізнювані кольори для класів 0 і 1
colors = ['#1f77b4', '#ff7f0e']  # синій та помаранчевий, наприклад
point_colors = [colors[label] for label in labels]

scatter = plt.scatter(
    embedding[:, 0], embedding[:, 1],
    c=point_colors,
    alpha=0.9,
    s=150,
    edgecolor='k'
)

# Створюємо власну легенду
for class_value in np.unique(labels):
    plt.scatter([], [], c=colors[class_value], label=f'Class {class_value}', edgecolor='k', s=60)

plt.legend(title='Класи', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title("UMAP-візуалізація латентного простору автоенкодера (чіткі кольори)", fontsize=16)
plt.xlabel("UMAP компонент 1", fontsize=12)
plt.ylabel("UMAP компонент 2", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
