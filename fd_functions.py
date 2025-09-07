import numpy as np
import torch


###############################################################
# Обчислення фрактальної розмірності за допомогою перетворення Фур'є
# з використанням методу зі статті Florindo & Bruno (2018)
###############################################################
import numpy as np
import torch
from skimage.color import rgb2lab


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

###############################################################
# Функція для обчислення фрактальної розмірності методом box-counting
###############################################################
def box_counting(image, threshold=0.5):
    """
    Обчислює фрактальну розмірність зображення за методом box-counting.

    Параметри:
      image - тензор PyTorch розмірності [C, H, W] (значення в діапазоні [0,1])
      threshold - поріг для бінаризації зображення

    Повертає:
      fd - оцінка фрактальної розмірності
    """
    # Перетворення зображення з тензора у numpy-масив і конвертація в відтінки сірого
    if torch.is_tensor(image):
        img_np = image.numpy()
        # Згладжуємо колірний простір: беремо середнє по каналах
        if img_np.shape[0] == 3:
            img_np = np.mean(img_np, axis=0)
    else:
        # Якщо зображення вже у вигляді масиву
        img_np = image
        if len(img_np.shape) == 3:
            img_np = np.mean(img_np, axis=2)

    # Бінаризація зображення: формуємо маску (True для "foreground")
    binary_image = img_np < threshold

    # Функція для підрахунку кількості заповнених боксу розміру k
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k), axis=1)
        # Рахуємо кількість боксу, де хоча б один піксель заповнений
        return np.sum(S > 0)

    # Визначаємо мінімальний розмір зображення
    p = min(binary_image.shape)
    # Вибираємо розміри боксу: степені двійки від максимальної можливої до 2
    sizes = 2 ** np.arange(int(np.floor(np.log2(p))), 1, -1).astype(int)

    counts = []
    for size in sizes:
        counts.append(boxcount(binary_image, size))

    # Лінійна регресія: log(counts) = -D * log(size) + C  => D = -нахил
    eps = 1e-6
    coeffs = np.polyfit(np.log(sizes), np.log(np.array(counts) + eps), 1)
    fd = -coeffs[0]
    return fd

def fd_RGB_Lab(image):
    """
    Обчислює фрактальну розмірність кольорового зображення за методом box-counting у Lab просторі.

    Параметри:
        image - тензор PyTorch розмірності [C, H, W], значення [0,1].

    Повертає:
        fd - оцінка фрактальної розмірності.
    """
    # Конвертація PyTorch тензора у numpy-масив
    if torch.is_tensor(image):
        img_np = image.numpy().transpose(1, 2, 0)
    else:
        img_np = image

    # Конвертація RGB в Lab простір
    lab_image = rgb2lab(img_np)

    # Мінімальний розмір зображення
    M = min(lab_image.shape[:2])

    # Вибір розмірів боксу (дільники розміру зображення)
    sizes = [size for size in range(2, M//2 + 1) if M % size == 0]

    Nr = []

    for size in sizes:
        n_boxes = 0

        for x in range(0, lab_image.shape[0], size):
            for y in range(0, lab_image.shape[1], size):
                # Обчислення максимальних та мінімальних значень у кожному каналі
                region = lab_image[x:x+size, y:y+size, :]
                max_val = np.max(lab_image[x:x+size, y:y+size, :], axis=(0,1))
                min_val = np.min(lab_image[x:x+size, y:y+size, :], axis=(0,1))

                # Евклідова відстань у Lab-просторі
                d = np.linalg.norm(max_val - min_val)

                # Висота боксу (h), де 100 — максимальне значення в Lab просторі
                h = (size * 100) / lab_image.shape[0]

                # Кількість боксів
                n_r = np.ceil((d + 1) / h)
                n_boxes += n_r

        Nr.append(n_boxes)

    # Лінійна регресія для обчислення нахилу
    coeffs = np.polyfit(np.log(1/np.array(sizes)), np.log(Nr), 1)
    fd = coeffs[0]  # Фрактальна розмірність – нахил прямої

    return fd

