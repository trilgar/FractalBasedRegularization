import numpy as np
import torch


###############################################################
# Обчислення фрактальної розмірності за допомогою перетворення Фур'є
# з використанням методу зі статті Florindo & Bruno (2018)
###############################################################
import numpy as np
import torch

def fd_fourier2(image):
    """
    Обчислює фрактальну розмірність зображення методом Фур'є за підходом зі статті Florindo & Bruno.

    Параметри:
        image - тензор PyTorch розмірності [C, H, W] (значення в діапазоні [0,1])

    Повертає:
        fd - оцінка фрактальної розмірності
    """
    # Перетворення в grayscale numpy-масив
    if torch.is_tensor(image):
        img_np = image.numpy()
        if img_np.shape[0] == 3:
            img_np = np.mean(img_np, axis=0)
    else:
        img_np = image
        if len(img_np.shape) == 3:
            img_np = np.mean(img_np, axis=2)

    # Розрахунок 2D-Фур'є перетворення та спектра потужності
    F = np.fft.fft2(img_np)
    F_shifted = np.fft.fftshift(F)
    power_spectrum = np.abs(F_shifted)**2

    # Розбиваємо спектр на кільця (кільцеві частотні діапазони)
    N = power_spectrum.shape[0]
    center = N // 2
    max_radius = center
    radii = np.arange(1, max_radius)
    radial_ps = np.zeros(len(radii))

    Y, X = np.indices((N, N))
    R = np.sqrt((X - center)**2 + (Y - center)**2)

    for i, r in enumerate(radii):
        mask = (R >= r - 1) & (R < r)
        if np.any(mask):
            radial_ps[i] = np.mean(power_spectrum[mask])

    # Видаляємо нульові значення для логарифмічного перетворення
    valid = radial_ps > 0
    radial_ps = radial_ps[valid]
    freqs = radii[valid]

    # Логарифмічна залежність спектра потужності від частоти
    log_freqs = np.log(freqs)
    log_ps = np.log(radial_ps)

    # Лінійна регресія для визначення нахилу
    slope, intercept = np.polyfit(log_freqs, log_ps, 1)

    # Обчислення фрактальної розмірності за нахилом
    fd = (slope + 6) / 2

    return fd
