import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Енкодер
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # -> 16 x 112 x 112
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> 32 x 56 x 56
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> 64 x 28 x 28
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# -> 128 x 14 x 14
            nn.ReLU(),
        )
        # Декодер
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> 64 x 28 x 28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> 32 x 56 x 56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> 16 x 112 x 112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # -> 3 x 224 x 224
            nn.Sigmoid()  # Значення у [0,1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class FDRegressor(nn.Module):
    def __init__(self, latent_channels=128):
        super(FDRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(latent_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, latent):
        return self.model(latent).squeeze()

class BadAutoencoder(nn.Module):
    def __init__(self):
        super(BadAutoencoder, self).__init__()
        # Енкодер:
        self.encoder = nn.Sequential(
            # Зменшуємо 224x224 -> 112x112
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 112x112 -> 56x56
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 56x56 -> 28x28
            nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Декодер:
        self.decoder = nn.Sequential(
            # 4 x 28x28 -> 8 x 56x56
            nn.ConvTranspose2d(4, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 8 x 56x56 -> 16 x 112x112
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 16 x 112x112 -> 3 x 224x224
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded