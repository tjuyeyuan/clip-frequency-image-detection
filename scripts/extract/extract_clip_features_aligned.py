from pathlib import Path

# Resolve local files relative to the project root, regardless of working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

import os
import random
import logging
import numpy as np
import torch

from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor


# =========================================================
# 0. Config
# =========================================================

SEED = 42

TRAIN_FAKE = 500
TRAIN_REAL = 500

TEST_FAKE = 100
TEST_REAL = 100

BATCH_SIZE = 64

MODEL_NAME = "openai/clip-vit-base-patch32"

EXPERIMENT_NAME = "aligned_clip_feature_extraction"


# =========================================================
# 1. Output folders
# =========================================================

os.makedirs(
    (PROJECT_ROOT / "results"),
    exist_ok=True
)

os.makedirs(
    (PROJECT_ROOT / "features"),
    exist_ok=True
)


LOG_PATH = os.path.join(
    (PROJECT_ROOT / "results"),
    f"{EXPERIMENT_NAME}_seed{SEED}.log"
)


# =========================================================
# 2. Logging
# =========================================================

logger = logging.getLogger(
    EXPERIMENT_NAME
)

logger.setLevel(
    logging.INFO
)

logger.handlers.clear()


console_handler = logging.StreamHandler()

file_handler = logging.FileHandler(
    LOG_PATH,
    mode="w",
    encoding="utf-8"
)

formatter = logging.Formatter(
    "%(asctime)s | %(message)s"
)

console_handler.setFormatter(
    formatter
)

file_handler.setFormatter(
    formatter
)

logger.addHandler(
    console_handler
)

logger.addHandler(
    file_handler
)


# =========================================================
# 3. Seed
# =========================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


# =========================================================
# 4. Load CIFAKE
# =========================================================

logger.info(
    "Loading CIFAKE..."
)

dataset = load_dataset(
    "dragonintelligence/CIFAKE-image-dataset"
)

logger.info(
    f"Train size: {len(dataset['train'])}"
)

logger.info(
    f"Test size: {len(dataset['test'])}"
)

logger.info(
    "Label mapping: 0 = FAKE, 1 = REAL"
)


# =========================================================
# 5. Same balanced subset logic
# =========================================================

def make_balanced_subset(
    hf_dataset,
    num_fake,
    num_real,
    seed=42
):

    fake_indices = []
    real_indices = []

    for i in range(
        len(hf_dataset)
    ):

        label = hf_dataset[i]["label"]

        if label == 0:
            fake_indices.append(i)

        elif label == 1:
            real_indices.append(i)

    rng = random.Random(
        seed
    )

    rng.shuffle(
        fake_indices
    )

    rng.shuffle(
        real_indices
    )

    selected_indices = (
        fake_indices[:num_fake]
        +
        real_indices[:num_real]
    )

    rng.shuffle(
        selected_indices
    )

    subset = hf_dataset.select(
        selected_indices
    )

    return (
        subset,
        selected_indices
    )


small_train, train_indices = make_balanced_subset(
    dataset["train"],
    TRAIN_FAKE,
    TRAIN_REAL,
    SEED
)

small_test, test_indices = make_balanced_subset(
    dataset["test"],
    TEST_FAKE,
    TEST_REAL,
    SEED
)


logger.info(
    f"Selected train size: {len(small_train)}"
)

logger.info(
    f"Selected test size: {len(small_test)}"
)


# =========================================================
# 6. Save selected indices
# =========================================================

np.save(
    (PROJECT_ROOT / "features/train_indices.npy"),
    np.array(
        train_indices,
        dtype=np.int64
    )
)

np.save(
    (PROJECT_ROOT / "features/test_indices.npy"),
    np.array(
        test_indices,
        dtype=np.int64
    )
)

logger.info(
    "Saved train/test indices."
)


# =========================================================
# 7. Device
# =========================================================

if torch.backends.mps.is_available():

    device = torch.device(
        "mps"
    )

elif torch.cuda.is_available():

    device = torch.device(
        "cuda"
    )

else:

    device = torch.device(
        "cpu"
    )


logger.info(
    f"Using device: {device}"
)


# =========================================================
# 8. Load CLIP
# =========================================================

logger.info(
    f"Loading CLIP: {MODEL_NAME}"
)

processor = CLIPProcessor.from_pretrained(
    MODEL_NAME
)

model = CLIPModel.from_pretrained(
    MODEL_NAME
)

model = model.to(
    device
)

model.eval()

logger.info(
    "CLIP loaded."
)


# =========================================================
# 9. Extract function
# =========================================================

def extract_clip_features(
    hf_subset,
    batch_size,
    split_name
):

    all_features = []
    all_labels = []

    total = len(
        hf_subset
    )

    logger.info(
        f"Start extracting {split_name} features..."
    )

    with torch.no_grad():

        for start in range(
            0,
            total,
            batch_size
        ):

            end = min(
                start + batch_size,
                total
            )

            batch = hf_subset.select(
                range(
                    start,
                    end
                )
            )

            images = list(
    batch["image"]
)

            labels = list(
    batch["label"]
)

            inputs = processor(
                images=images,
                return_tensors="pt"
            )

            pixel_values = inputs[
                "pixel_values"
            ].to(
                device
            )

            outputs = model.get_image_features(
                pixel_values=pixel_values
            )

            # 兼容你当前 transformers 版本
            if hasattr(
                outputs,
                "pooler_output"
            ):

                features = outputs.pooler_output

            else:

                features = outputs

            # L2 normalize
            features = (
                features
                /
                features.norm(
                    dim=-1,
                    keepdim=True
                )
            )

            all_features.append(
                features
                .cpu()
                .numpy()
            )

            all_labels.extend(
                labels
            )

            logger.info(
                f"{split_name}: "
                f"{end}/{total}"
            )

    all_features = np.concatenate(
        all_features,
        axis=0
    )

    all_labels = np.array(
        all_labels,
        dtype=np.int64
    )

    logger.info(
        f"{split_name} feature shape: "
        f"{all_features.shape}"
    )

    logger.info(
        f"{split_name} label shape: "
        f"{all_labels.shape}"
    )

    return (
        all_features,
        all_labels
    )


# =========================================================
# 10. Train features
# =========================================================

train_features, train_labels = extract_clip_features(
    small_train,
    BATCH_SIZE,
    "train"
)


# =========================================================
# 11. Test features
# =========================================================

test_features, test_labels = extract_clip_features(
    small_test,
    BATCH_SIZE,
    "test"
)


# =========================================================
# 12. Save
# =========================================================

np.save(
    (PROJECT_ROOT / "features/aligned_train_clip_features.npy"),
    train_features
)

np.save(
    (PROJECT_ROOT / "features/aligned_train_labels.npy"),
    train_labels
)

np.save(
    (PROJECT_ROOT / "features/aligned_test_clip_features.npy"),
    test_features
)

np.save(
    (PROJECT_ROOT / "features/aligned_test_labels.npy"),
    test_labels
)


logger.info(
    "Saved aligned CLIP features."
)

logger.info(
    f"Train features: "
    f"{train_features.shape}"
)

logger.info(
    f"Test features: "
    f"{test_features.shape}"
)

logger.info(
    f"Log saved to: "
    f"{LOG_PATH}"
)