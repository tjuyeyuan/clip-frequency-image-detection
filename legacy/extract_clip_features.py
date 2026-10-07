from pathlib import Path

# Resolve local files relative to the project root, regardless of working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

from datasets import load_dataset, concatenate_datasets
from transformers import CLIPModel, CLIPProcessor

import torch
import numpy as np
from tqdm import tqdm


# ============================================================
# 1. 配置
# ============================================================

MODEL_NAME = "openai/clip-vit-base-patch32"

BATCH_SIZE = 64

# True：先跑小规模
# False：跑完整数据集
USE_SUBSET = True

TRAIN_SUBSET_SIZE = 1000
TEST_SUBSET_SIZE = 200

SAVE_DIR = PROJECT_ROOT


# ============================================================
# 2. 选择设备
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print("Device:", DEVICE)


# ============================================================
# 3. 加载 CIFAKE
# ============================================================

print("\nLoading CIFAKE...")

dataset = load_dataset(
    "dragonintelligence/CIFAKE-image-dataset"
)

print(dataset)


# ============================================================
# 4. 构造 balanced dataset
# ============================================================

if USE_SUBSET:

    print("\nCreating balanced subset...")

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_fake_all = dataset["train"].filter(
        lambda x: x["label"] == 0
    )

    train_real_all = dataset["train"].filter(
        lambda x: x["label"] == 1
    )

    train_half = TRAIN_SUBSET_SIZE // 2

    train_fake = train_fake_all.select(
        range(train_half)
    )

    train_real = train_real_all.select(
        range(train_half)
    )

    train_dataset = concatenate_datasets(
        [
            train_fake,
            train_real
        ]
    )

    train_dataset = train_dataset.shuffle(
        seed=42
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_fake_all = dataset["test"].filter(
        lambda x: x["label"] == 0
    )

    test_real_all = dataset["test"].filter(
        lambda x: x["label"] == 1
    )

    test_half = TEST_SUBSET_SIZE // 2

    test_fake = test_fake_all.select(
        range(test_half)
    )

    test_real = test_real_all.select(
        range(test_half)
    )

    test_dataset = concatenate_datasets(
        [
            test_fake,
            test_real
        ]
    )

    test_dataset = test_dataset.shuffle(
        seed=42
    )

else:

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]


print("\nTrain samples:", len(train_dataset))
print("Test samples:", len(test_dataset))


# ============================================================
# 5. 检查原始标签分布
# ============================================================

train_raw_labels = np.array(
    train_dataset["label"]
)

test_raw_labels = np.array(
    test_dataset["label"]
)


print("\nRaw train label distribution:")

unique, counts = np.unique(
    train_raw_labels,
    return_counts=True
)

for label, count in zip(unique, counts):
    print(
        f"label {label}: {count}"
    )


print("\nRaw test label distribution:")

unique, counts = np.unique(
    test_raw_labels,
    return_counts=True
)

for label, count in zip(unique, counts):
    print(
        f"label {label}: {count}"
    )


# ============================================================
# 6. 加载 CLIP
# ============================================================

print("\nLoading CLIP...")

model = CLIPModel.from_pretrained(
    MODEL_NAME
)

model = model.to(DEVICE)

processor = CLIPProcessor.from_pretrained(
    MODEL_NAME
)

model.eval()


# ============================================================
# 7. 提取 CLIP image embedding
# ============================================================

def extract_features(
    split,
    split_name="dataset"
):

    all_features = []
    all_labels = []

    print(
        f"\nExtracting {split_name} features..."
    )

    for start in tqdm(
        range(
            0,
            len(split),
            BATCH_SIZE
        )
    ):

        end = min(
            start + BATCH_SIZE,
            len(split)
        )

        # ----------------------------------------------------
        # 取一个 batch
        # ----------------------------------------------------

        batch = split[
            start:end
        ]

        images = [
            image.convert("RGB")
            for image in batch["image"]
        ]

        labels = batch["label"]

        # ----------------------------------------------------
        # CLIP 图片预处理
        # ----------------------------------------------------

        inputs = processor(
            images=images,
            return_tensors="pt"
        )

        pixel_values = inputs[
            "pixel_values"
        ].to(DEVICE)

        # ----------------------------------------------------
        # CLIP forward
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model.get_image_features(
                pixel_values=pixel_values
            )

            # 你当前 transformers 版本中
            # get_image_features 返回
            # BaseModelOutputWithPooling
            features = outputs.pooler_output

            # L2 normalization
            features = (
                features
                / features.norm(
                    dim=-1,
                    keepdim=True
                )
            )

        # ----------------------------------------------------
        # 第一批打印信息
        # ----------------------------------------------------

        if start == 0:

            print(
                "\nOutput type:",
                type(outputs)
            )

            print(
                "Feature shape:",
                features.shape
            )

            print(
                "Batch labels:",
                len(labels)
            )

        # ----------------------------------------------------
        # 保存当前 batch
        # ----------------------------------------------------

        all_features.append(
            features
            .cpu()
            .numpy()
        )

        all_labels.extend(
            labels
        )

    # --------------------------------------------------------
    # 合并全部 batch
    # --------------------------------------------------------

    all_features = np.concatenate(
        all_features,
        axis=0
    )

    all_labels = np.array(
        all_labels,
        dtype=np.int64
    )

    return (
        all_features,
        all_labels
    )


# ============================================================
# 8. 提取 train
# ============================================================

train_features, train_labels = extract_features(
    train_dataset,
    split_name="train"
)

print("\nTrain finished:")

print(
    "train_features shape:",
    train_features.shape
)

print(
    "train_labels shape:",
    train_labels.shape
)


# ============================================================
# 9. 提取 test
# ============================================================

test_features, test_labels = extract_features(
    test_dataset,
    split_name="test"
)

print("\nTest finished:")

print(
    "test_features shape:",
    test_features.shape
)

print(
    "test_labels shape:",
    test_labels.shape
)


# ============================================================
# 10. 再检查一次最终标签
# ============================================================

print("\nFinal train label distribution:")

unique, counts = np.unique(
    train_labels,
    return_counts=True
)

for label, count in zip(
    unique,
    counts
):
    print(
        f"label {label}: {count}"
    )


print("\nFinal test label distribution:")

unique, counts = np.unique(
    test_labels,
    return_counts=True
)

for label, count in zip(
    unique,
    counts
):
    print(
        f"label {label}: {count}"
    )


# ============================================================
# 11. 保存
# ============================================================

if USE_SUBSET:
    prefix = "small"
else:
    prefix = "full"


train_features_path = (
    f"{SAVE_DIR}/"
    f"{prefix}_train_features.npy"
)

train_labels_path = (
    f"{SAVE_DIR}/"
    f"{prefix}_train_labels.npy"
)

test_features_path = (
    f"{SAVE_DIR}/"
    f"{prefix}_test_features.npy"
)

test_labels_path = (
    f"{SAVE_DIR}/"
    f"{prefix}_test_labels.npy"
)


np.save(
    train_features_path,
    train_features
)

np.save(
    train_labels_path,
    train_labels
)

np.save(
    test_features_path,
    test_features
)

np.save(
    test_labels_path,
    test_labels
)


# ============================================================
# 12. 完成
# ============================================================

print(
    "\n=============================="
)

print(
    "Feature extraction finished!"
)

print(
    "=============================="
)

print("\nSaved files:")

print(train_features_path)
print(train_labels_path)
print(test_features_path)
print(test_labels_path)