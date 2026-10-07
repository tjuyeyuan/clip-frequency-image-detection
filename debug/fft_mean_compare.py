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
# 2. 参数
# ============================================================

NUM_PER_CLASS = 500


# ============================================================
# 3. 频谱函数
# ============================================================

def get_log_spectrum(image):

    # 转灰度
    gray = image.convert("L")

    image_np = np.array(
        gray,
        dtype=np.float32
    )

    # FFT
    freq = np.fft.fft2(
        image_np
    )

    # 把低频移到中心
    freq_shift = np.fft.fftshift(
        freq
    )

    # 幅度谱
    magnitude = np.abs(
        freq_shift
    )

    # log 压缩
    spectrum = np.log1p(
        magnitude
    )

    return spectrum


# ============================================================
# 4. 分别收集 FAKE / REAL
# ============================================================

fake_spectra = []
real_spectra = []


for sample in dataset:

    label = sample["label"]

    if (
        label == 0
        and len(fake_spectra) < NUM_PER_CLASS
    ):

        spectrum = get_log_spectrum(
            sample["image"]
        )

        fake_spectra.append(
            spectrum
        )


    elif (
        label == 1
        and len(real_spectra) < NUM_PER_CLASS
    ):

        spectrum = get_log_spectrum(
            sample["image"]
        )

        real_spectra.append(
            spectrum
        )


    # 两类都够了就停止
    if (
        len(fake_spectra) == NUM_PER_CLASS
        and len(real_spectra) == NUM_PER_CLASS
    ):
        break


print(
    "Fake samples:",
    len(fake_spectra)
)

print(
    "Real samples:",
    len(real_spectra)
)


# ============================================================
# 5. 转成 NumPy
# ============================================================

fake_spectra = np.stack(
    fake_spectra,
    axis=0
)

real_spectra = np.stack(
    real_spectra,
    axis=0
)


print(
    "Fake spectra shape:",
    fake_spectra.shape
)

print(
    "Real spectra shape:",
    real_spectra.shape
)


# ============================================================
# 6. 求平均频谱
# ============================================================

mean_fake = np.mean(
    fake_spectra,
    axis=0
)

mean_real = np.mean(
    real_spectra,
    axis=0
)


# ============================================================
# 7. 差异图
# ============================================================

difference = (
    mean_fake
    - mean_real
)


# ============================================================
# 8. 打印一些统计信息
# ============================================================

print(
    "Mean fake spectrum:",
    mean_fake.mean()
)

print(
    "Mean real spectrum:",
    mean_real.mean()
)

print(
    "Mean absolute difference:",
    np.abs(difference).mean()
)


# ============================================================
# 9. 可视化
# ============================================================

plt.figure(
    figsize=(12, 4)
)


# FAKE mean spectrum
plt.subplot(
    1,
    3,
    1
)

plt.imshow(
    mean_fake,
    cmap="gray"
)

plt.title(
    "Mean FAKE Spectrum"
)

plt.axis("off")


# REAL mean spectrum
plt.subplot(
    1,
    3,
    2
)

plt.imshow(
    mean_real,
    cmap="gray"
)

plt.title(
    "Mean REAL Spectrum"
)

plt.axis("off")


# Difference
plt.subplot(
    1,
    3,
    3
)

plt.imshow(
    difference,
    cmap="bwr"
)

plt.title(
    "FAKE - REAL"
)

plt.axis("off")


plt.tight_layout()

plt.show()