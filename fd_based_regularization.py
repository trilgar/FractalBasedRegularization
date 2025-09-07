import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import fd_functions
from autoencoders import Autoencoder, FDRegressor
from datasets import ISICDataset
from envs.current_env import IMAGE_DIR, METADATA_FILE, CURRENT_DEVICE

# Конфігурація шляхів
ALPHA = 0.05
BETA = 0.1
MODEL_PATH = "models/AFLREG_005.pth"
EPOCHS = 100

# Завантаження метаданих
metadata = pd.read_csv(METADATA_FILE)

# Трансформації зображень
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
device = torch.device(CURRENT_DEVICE)
print(f"Current device: {device}")

# Ініціалізація датасету та DataLoader
dataset = ISICDataset(metadata, IMAGE_DIR, transform=transform, load_images_to_ram=True)
# train_loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True, pin_memory_device="cuda", prefetch_factor=2, persistent_workers=True)
train_loader = DataLoader(dataset, batch_size=512, shuffle=True, pin_memory=True)
# Параметри тренування
learning_rate = 0.003

# Ініціалізація автоенкодера
autoencoder = Autoencoder().to(device)
fd_regressor = FDRegressor(latent_channels=128).to(device)

# Оптимізатор
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
optimizer_fd = optim.Adam(fd_regressor.parameters(), lr=0.001)

# Втрата реконструкції з FD
def fractal_mse_loss(original, reconstructed, fd_fn, alpha=0.05):
    mse = nn.functional.mse_loss(reconstructed, original)
    batch_size = original.size(0)
    fd_orig = torch.tensor([
        fd_fn(original[i].cpu().detach())
        for i in range(batch_size)
    ], device=original.device, dtype=torch.float32)
    fd_recon = torch.tensor([
        fd_fn(reconstructed[i].cpu().detach())
        for i in range(batch_size)
    ], device=original.device, dtype=torch.float32)
    fd_loss = ((fd_orig - fd_recon) ** 2).mean()
    loss = mse + alpha * fd_loss
    return loss, fd_orig


if __name__ == '__main__':
    # Тренувальний цикл
    losses, mse_losses, fd_losses = [], [], []
    for epoch in range(EPOCHS):
        autoencoder.train()
        fd_regressor.train()
        running_loss, mse_loss, running_fd_loss = 0.0, 0.0, 0.0
        for images, _, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            images = images.to(device)
            optimizer.zero_grad()
            optimizer_fd.zero_grad()
            latent, outputs = autoencoder(images)

            # Обчислюємо FD-based loss
            recon_loss, fd_true = fractal_mse_loss(
                original=images,
                reconstructed=outputs,
                fd_fn=fd_functions.fd_fourier2,
                alpha=ALPHA
            )

            fd_pred = fd_regressor(latent)
            fd_reg_loss = nn.functional.mse_loss(fd_pred, fd_true)

            loss = recon_loss + BETA * fd_reg_loss
            loss.backward()

            optimizer.step()
            optimizer_fd.step()

            running_loss += loss.item() * images.size(0)
            mse_loss += nn.functional.mse_loss(images, outputs).item() * images.size(0)
            running_fd_loss += fd_reg_loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_mse_loss = mse_loss / len(train_loader.dataset)
        epoch_fd_loss = running_fd_loss / len(train_loader.dataset)
        losses.append(epoch_loss)
        mse_losses.append(epoch_mse_loss)
        fd_losses.append(epoch_fd_loss)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] Total: {epoch_loss:.4f}, MSE: {epoch_mse_loss:.4f}, FD: {epoch_fd_loss:.4f}")

    # Збереження моделі
    torch.save(autoencoder.state_dict(), MODEL_PATH)
    print(f"Модель збережено у файлі: {MODEL_PATH}")

    # Графіки втрат
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker='o', label='Total Loss')
    plt.plot(mse_losses, marker='x', label='MSE Loss')
    plt.plot(fd_losses, marker='^', label='FD Regression Loss')
    plt.title('Losses during training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()
