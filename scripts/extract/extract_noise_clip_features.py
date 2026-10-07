from pathlib import Path

# Resolve local files relative to the project root, regardless of working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

import os
import logging
from io import BytesIO

import numpy as np
import torch

from PIL import Image
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor


# =========================================================
# 0. Config
# =========================================================
SIGMAS = [5, 10, 20, 30]

BATCH_SIZE = 64

MODEL_NAME = "openai/clip-vit-base-patch32"

EXPERIMENT_NAME = "noise_clip_feature_extraction"


# =========================================================
# 1. Output folders
# =========================================================

os.makedirs(
    (PROJECT_ROOT / "features/noise"),
    exist_ok=True
)

os.makedirs(
    (PROJECT_ROOT / "results"),
    exist_ok=True
)

LOG_PATH = os.path.join(
    (PROJECT_ROOT / "results"),
    f"{EXPERIMENT_NAME}.log"
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
# 3. Gaussian noise
# =========================================================

def add_gaussian_noise(
    image,
    sigma=10,
    seed=0
):
    image = image.convert("RGB")

    image_np = np.array(
        image,
        dtype=np.float32
    )

    rng = np.random.default_rng(
        seed
    )

    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=image_np.shape
    )

    noisy = image_np + noise

    noisy = np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(
        noisy
    )


# =========================================================
# 4. Load CIFAKE
# =========================================================

logger.info(
    "Loading CIFAKE..."
)

dataset = load_dataset(
    "dragonintelligence/CIFAKE-image-dataset"
)


# =========================================================
# 5. Load fixed test indices
# =========================================================

test_indices = np.load(
    (PROJECT_ROOT / "features/test_indices.npy")
)

small_test = dataset["test"].select(
    test_indices.tolist()
)

logger.info(
    f"Aligned test size: {len(small_test)}"
)


# =========================================================
# 6. Device
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
# 7. Load CLIP
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
# 8. Extract one Gaussian noise level
# =========================================================

def extract_noise_features(
    hf_subset,
    sigma,
    batch_size
):

    all_features = []
    all_labels = []

    total = len(
        hf_subset
    )

    logger.info(
        f"Start Gaussian noise sigma={sigma}"
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

            # ---------------------------------------------
            # Gaussian noise
            # ---------------------------------------------

            noisy_images = [

            add_gaussian_noise(
                image,
                sigma=sigma,
                seed=start + i
            )

            for i, image in enumerate(images)
        ]


            # ---------------------------------------------
            # Gaussian noise images -> CLIP
            # ---------------------------------------------

            inputs = processor(
                images=noisy_images,
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
                f"Gaussian noise sigma={sigma}: "
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
        f"Gaussian noise sigma={sigma} feature shape: "
        f"{all_features.shape}"
    )

    return (
        all_features,
        all_labels
    )


# =========================================================
# 9. Run all Gaussian noise levels
# =========================================================


for sigma in SIGMAS:

    features, labels = extract_noise_features(
        small_test,
        sigma,
        BATCH_SIZE
    )

    feature_path = os.path.join(
        (PROJECT_ROOT / "features/noise"),
        f"test_clip_sigma{sigma}.npy"
    )

    label_path = os.path.join(
        (PROJECT_ROOT / "features/noise"),
        f"test_labels_sigma{sigma}.npy"
    )

    np.save(
        feature_path,
        features
    )

    np.save(
        label_path,
        labels
    )

    logger.info(
        f"Saved: {feature_path}"
    )

    logger.info(
        f"Saved: {label_path}"
    )
    logger.info(
    "All Gaussian noise feature extraction finished."
    )

    logger.info(
        f"Log saved to: {LOG_PATH}"
    )