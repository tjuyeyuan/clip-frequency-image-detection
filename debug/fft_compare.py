from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 加载数据
# ============================================================

dataset = load_dataset(
    "dragonintelligence/CIFAKE-image-dataset",
    split="train"
)


# ============================================================
# 2. 分别找一张 FAKE 和一张 REAL
# ============================================================

fake_sample = None
real_sample = None

for sample in dataset:

    if sample["label"] == 0 and fake_sample is None:
        fake_sample = sample

    if sample["label"] == 1 and real_sample is None:
        real_sample = sample

    if fake_sample is not None and real_sample is not None:
        break


# ============================================================
# 3. FFT 函数
# ============================================================

def get_spectrum(image):

    # 转灰度
    gray = image.convert("L")

    image_np = np.array(
        gray,
        dtype=np.float32
    )

    # 2D FFT
    freq = np.fft.fft2(
        image_np
    )

    # 低频移到中心
    freq_shift = np.fft.fftshift(
        freq
    )

    # 幅度
    magnitude = np.abs(
        freq_shift
    )

    # log 压缩
    spectrum = np.log1p(
        magnitude
    )

    return image_np, spectrum


# ============================================================
# 4. 分别处理 FAKE / REAL
# ============================================================

fake_image, fake_spectrum = get_spectrum(
    fake_sample["image"]
)

real_image, real_spectrum = get_spectrum(
    real_sample["image"]
)


# ============================================================
# 5. 打印 shape
# ============================================================

print(
    "Fake image:",
    fake_image.shape
)

print(
    "Fake spectrum:",
    fake_spectrum.shape
)

print(
    "Real image:",
    real_image.shape
)

print(
    "Real spectrum:",
    real_spectrum.shape
)


# ============================================================
# 6. 可视化
# ============================================================

plt.figure(
    figsize=(8, 8)
)


# FAKE 原图
plt.subplot(
    2,
    2,
    1
)

plt.imshow(
    fake_image,
    cmap="gray"
)

plt.title(
    "FAKE Image"
)

plt.axis("off")


# FAKE 频谱
plt.subplot(
    2,
    2,
    2
)

plt.imshow(
    fake_spectrum,
    cmap="gray"
)

plt.title(
    "FAKE FFT Spectrum"
)

plt.axis("off")


# REAL 原图
plt.subplot(
    2,
    2,
    3
)

plt.imshow(
    real_image,
    cmap="gray"
)

plt.title(
    "REAL Image"
)

plt.axis("off")


# REAL 频谱
plt.subplot(
    2,
    2,
    4
)

plt.imshow(
    real_spectrum,
    cmap="gray"
)

plt.title(
    "REAL FFT Spectrum"
)

plt.axis("off")


plt.tight_layout()

plt.show()