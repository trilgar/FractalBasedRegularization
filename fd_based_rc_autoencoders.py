import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import fd_functions
from autoencoders import Autoencoder
from datasets import ISICDataset

# Конфігурація шляхів
IMAGE_DIR = 'F:/datasets/ISIC2024/train-image/image'
METADATA_FILE = 'F:/datasets/ISIC2024/train-metadata-updated.csv'
ALPHA = 0.05
MODEL_PATH = "models/AWFL_fourier_005.pth"
EPOCHS = 50

# Завантаження метаданих
metadata = pd.read_csv(METADATA_FILE)

# Трансформації зображень
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")

# Ініціалізація датасету та DataLoader
dataset = ISICDataset(metadata, IMAGE_DIR, transform=transform, load_images_to_ram=True)
# train_loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True, pin_memory_device="cuda", prefetch_factor=2, persistent_workers=True)
train_loader = DataLoader(dataset, batch_size=512, shuffle=True, pin_memory=True, pin_memory_device="cuda")
# Параметри тренування
learning_rate = 0.003

# Ініціалізація автоенкодера
autoencoder = Autoencoder().to(device)

# Оптимізатор
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)


# Функція втрат з FD

def fractal_mse_loss(original, reconstructed, fd_fn, alpha=0.03):
    mse = nn.functional.mse_loss(reconstructed, original)

    batch_size = original.size(0)
    fd_orig = torch.tensor([
        fd_fn(original[i].cpu().detach())
        for i in range(batch_size)
    ], device=original.device)

    fd_recon = torch.tensor([
        fd_fn(reconstructed[i].cpu().detach())
        for i in range(batch_size)
    ], device=original.device)

    fd_diff_squared = (fd_orig - fd_recon) ** 2
    fd_loss = fd_diff_squared.mean()

    loss = mse + alpha * fd_loss
    return loss


if __name__ == '__main__':
    # Тренувальний цикл
    losses = []
    mse_losses = []
    for epoch in range(EPOCHS):
        autoencoder.train()
        running_loss = 0.0
        mse_loss = 0.0
        for images, _, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            images = images.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(images)

            # Обчислюємо FD-based loss
            loss = fractal_mse_loss(
                original=images,
                reconstructed=outputs,
                fd_fn=fd_functions.fd_fourier2,  # ваша функція FD
                alpha=ALPHA
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            mse_loss += nn.functional.mse_loss(images, outputs).item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        losses.append(epoch_loss)
        epoch_mse_loss = mse_loss / len(train_loader.dataset)
        mse_losses.append(epoch_mse_loss)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss:.4f}, MSE: {epoch_mse_loss:.4f}")

    # Збереження моделі
    torch.save(autoencoder.state_dict(), MODEL_PATH)
    print(f"Модель збережено у файлі: {MODEL_PATH}")

    # Графік загальних втрат (FD + MSE)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title('Графік втрат (FD + MSE) протягом навчання')
    plt.xlabel('Епоха')
    plt.ylabel('Втрата (loss)')
    plt.grid(True)
    plt.show()

    # Графік MSE втрат (MSE)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(mse_losses) + 1), mse_losses, marker='o')
    plt.title('Графік втрат (MSE) протягом навчання')
    plt.xlabel('Епоха')
    plt.ylabel('Втрата (loss)')
    plt.grid(True)
    plt.show()
