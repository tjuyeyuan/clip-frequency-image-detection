from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt


# 1. 加载 CIFAKE
dataset = load_dataset(
    "dragonintelligence/CIFAKE-image-dataset",
    split="train"
)


# 2. 随便取一张图
sample = dataset[0]

image = sample["image"].convert("L")   # 转灰度图
label = sample["label"]

image_np = np.array(
    image,
    dtype=np.float32
)


# 3. 2D FFT
freq = np.fft.fft2(
    image_np
)


# 4. 把低频移动到中心
freq_shift = np.fft.fftshift(
    freq
)


# 5. 取幅度谱 + log
magnitude = np.abs(
    freq_shift
)

spectrum = np.log1p(
    magnitude
)


# 6. 看 shape
print(
    "Image shape:",
    image_np.shape
)

print(
    "Spectrum shape:",
    spectrum.shape
)

print(
    "Label:",
    label
)


# 7. 可视化
plt.figure(figsize=(5, 5))

plt.imshow(
    spectrum,
    cmap="gray"
)

plt.title(
    f"FFT Spectrum | label={label}"
)

plt.axis("off")

plt.show()