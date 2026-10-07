from pathlib import Path

# Resolve local files relative to the project root, regardless of working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

import os
import csv
import random
import logging
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# =========================================================
# 0. Config
# =========================================================

SEEDS = [42, 123, 2026]

BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3

SIGMAS = [5, 10, 20, 30]

EXPERIMENT_NAME = "fusion_noise_robustness"


# =========================================================
# 1. Logging
# =========================================================

os.makedirs((PROJECT_ROOT / "results"), exist_ok=True)

LOG_PATH = (PROJECT_ROOT / f"results/{EXPERIMENT_NAME}.log")
CSV_PATH = (PROJECT_ROOT / "results/fusion_noise_robustness_results.csv")


logger = logging.getLogger(EXPERIMENT_NAME)

logger.setLevel(logging.INFO)
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

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# =========================================================
# 2. Seed
# =========================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
# 4. Image -> frequency spectrum
# =========================================================

def image_to_spectrum(image):

    image = image.convert("RGB")

    image_np = np.array(
        image,
        dtype=np.float32
    )

    # RGB -> gray
    gray = image_np.mean(
        axis=2
    )

    # FFT
    fft = np.fft.fft2(
        gray
    )

    fft = np.fft.fftshift(
        fft
    )

    magnitude = np.abs(
        fft
    )

    spectrum = np.log1p(
        magnitude
    )

    spectrum = torch.tensor(
        spectrum,
        dtype=torch.float32
    ).unsqueeze(0)

    return spectrum


# =========================================================
# 5. Device
# =========================================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


logger.info(
    f"Using device: {device}"
)


# =========================================================
# 6. Dataset
# =========================================================

class FusionDataset(Dataset):

    def __init__(
        self,
        hf_dataset,
        clip_features,
        labels,
        sigma=None
    ):

        self.hf_dataset = hf_dataset

        self.clip_features = clip_features

        self.labels = labels

        self.sigma = sigma


    def __len__(self):

        return len(
            self.labels
        )


    def __getitem__(
        self,
        idx
    ):

        image = self.hf_dataset[
            idx
        ]["image"]


        # 如果是 robustness test
        # 先对图片做 JPEG
        if self.sigma is not None:

            image = add_gaussian_noise(
                image,
                sigma=self.sigma,
                seed=idx
            )


        spectrum = image_to_spectrum(
            image
        )


        clip_feature = torch.tensor(
            self.clip_features[idx],
            dtype=torch.float32
        )


        label = torch.tensor(
            self.labels[idx],
            dtype=torch.float32
        )


        return (
            clip_feature,
            spectrum,
            label
        )


# =========================================================
# 7. Frequency CNN
# =========================================================

class TinyFreqCNN(nn.Module):

    def __init__(self):

        super().__init__()


        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        self.fc = nn.Linear(
            32,
            128
        )


    def forward(
        self,
        x
    ):

        x = self.features(
            x
        )

        x = x.flatten(
            1
        )

        x = self.fc(
            x
        )

        return x


# =========================================================
# 8. Fusion model
# =========================================================

class FusionModel(nn.Module):

    def __init__(self):

        super().__init__()


        self.freq_encoder = TinyFreqCNN()


        self.classifier = nn.Sequential(

            nn.Linear(
                512 + 128,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                1
            )
        )


    def forward(
        self,
        clip_feature,
        spectrum
    ):

        freq_feature = self.freq_encoder(
            spectrum
        )


        fusion_feature = torch.cat(
            [
                clip_feature,
                freq_feature
            ],
            dim=1
        )


        logits = self.classifier(
            fusion_feature
        )


        return logits


# =========================================================
# 9. Load data
# =========================================================

logger.info(
    "Loading CIFAKE..."
)

dataset = load_dataset(
    "dragonintelligence/CIFAKE-image-dataset"
)


train_indices = np.load(
    (PROJECT_ROOT / "features/train_indices.npy")
)

test_indices = np.load(
    (PROJECT_ROOT / "features/test_indices.npy")
)


train_hf = dataset["train"].select(
    train_indices.tolist()
)

test_hf = dataset["test"].select(
    test_indices.tolist()
)


train_clip = np.load(
    (PROJECT_ROOT / "features/aligned_train_clip_features.npy")
)

train_labels = np.load(
    (PROJECT_ROOT / "features/aligned_train_labels.npy")
)


clean_test_clip = np.load(
    (PROJECT_ROOT / "features/aligned_test_clip_features.npy")
)

test_labels = np.load(
    (PROJECT_ROOT / "features/aligned_test_labels.npy")
)


# =========================================================
# 10. Check label alignment
# =========================================================

train_hf_labels = np.array(
    train_hf["label"]
)

test_hf_labels = np.array(
    test_hf["label"]
)


assert np.array_equal(
    train_hf_labels,
    train_labels
)

assert np.array_equal(
    test_hf_labels,
    test_labels
)

logger.info(
    "Dataset alignment check passed."
)


# =========================================================
# 11. Test conditions
# =========================================================

test_conditions = {
    "clean": {
        "clip": clean_test_clip,
        "sigma": None
    }
}


for sigma in SIGMAS:

    noise_clip = np.load(
        (PROJECT_ROOT / f"features/noise/test_clip_sigma{sigma}.npy")
    )

    test_conditions[
        f"noise_sigma{sigma}"
    ] = {
        "clip": noise_clip,
        "sigma": sigma
        
    }


# =========================================================
# 12. Evaluation
# =========================================================

def evaluate(
    model,
    loader
):

    model.eval()


    all_probs = []
    all_preds = []
    all_labels = []


    with torch.no_grad():

        for (
            clip_feature,
            spectrum,
            label
        ) in loader:

            clip_feature = clip_feature.to(
                device
            )

            spectrum = spectrum.to(
                device
            )


            logits = model(
                clip_feature,
                spectrum
            ).squeeze(1)


            probs = torch.sigmoid(
                logits
            )


            preds = (
                probs > 0.5
            ).long()


            all_probs.extend(
                probs.cpu().numpy()
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                label.numpy()
            )


    return {

        "accuracy": accuracy_score(
            all_labels,
            all_preds
        ),

        "precision": precision_score(
            all_labels,
            all_preds
        ),

        "recall": recall_score(
            all_labels,
            all_preds
        ),

        "f1": f1_score(
            all_labels,
            all_preds
        ),

        "auc": roc_auc_score(
            all_labels,
            all_probs
        )
    }


# =========================================================
# 13. Multi-seed experiment
# =========================================================

all_results = []


for seed in SEEDS:

    logger.info(
        f"\n===== Seed {seed} Started ====="
    )


    set_seed(
        seed
    )


    train_dataset = FusionDataset(
        train_hf,
        train_clip,
        train_labels
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    model = FusionModel().to(
        device
    )


    criterion = nn.BCEWithLogitsLoss()


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )


    # =====================================================
    # Clean training
    # =====================================================

    for epoch in range(
        EPOCHS
    ):

        model.train()

        total_loss = 0.0


        for (
            clip_feature,
            spectrum,
            label
        ) in train_loader:

            clip_feature = clip_feature.to(
                device
            )

            spectrum = spectrum.to(
                device
            )

            label = label.to(
                device
            )


            optimizer.zero_grad()


            logits = model(
                clip_feature,
                spectrum
            ).squeeze(1)


            loss = criterion(
                logits,
                label
            )


            loss.backward()

            optimizer.step()


            total_loss += (
                loss.item()
            )


        avg_loss = (
            total_loss
            /
            len(train_loader)
        )


        logger.info(
            f"Seed {seed} | "
            f"Epoch {epoch + 1:03d}/{EPOCHS} "
            f"| Loss = {avg_loss:.4f}"
        )


    # =====================================================
    # 同一个模型测试所有 condition
    # =====================================================

    for condition, config in test_conditions.items():

        test_dataset = FusionDataset(
            test_hf,
            config["clip"],
            test_labels,
            sigma=config["sigma"]
        )


        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )


        result = evaluate(
            model,
            test_loader
        )


        logger.info(
            f"===== Seed {seed} | "
            f"{condition} ====="
        )


        logger.info(
            f"Accuracy  : "
            f"{result['accuracy'] * 100:.2f}%"
        )

        logger.info(
            f"Precision : "
            f"{result['precision'] * 100:.2f}%"
        )

        logger.info(
            f"Recall    : "
            f"{result['recall'] * 100:.2f}%"
        )

        logger.info(
            f"F1        : "
            f"{result['f1'] * 100:.2f}%"
        )

        logger.info(
            f"AUC       : "
            f"{result['auc'] * 100:.2f}%"
        )


        all_results.append({
            "seed": seed,
            "condition": condition,
            **result
        })


# =========================================================
# 14. Summary
# =========================================================

logger.info(
    "\n===== Fusion Gaussian Noise Robustness Summary ====="
)


summary_rows = []


for condition in test_conditions.keys():

    rows = [
        r
        for r in all_results
        if r["condition"] == condition
    ]


    summary = {
        "condition": condition
    }


    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc"
    ]:

        values = np.array([
            r[metric]
            for r in rows
        ])


        summary[
            f"{metric}_mean"
        ] = values.mean()


        summary[
            f"{metric}_std"
        ] = values.std()


    summary_rows.append(
        summary
    )


    logger.info(
        f"{condition}"
    )

    logger.info(
        f"  Accuracy : "
        f"{summary['accuracy_mean'] * 100:.2f} "
        f"± {summary['accuracy_std'] * 100:.2f}%"
    )

    logger.info(
        f"  F1       : "
        f"{summary['f1_mean'] * 100:.2f} "
        f"± {summary['f1_std'] * 100:.2f}%"
    )

    logger.info(
        f"  AUC      : "
        f"{summary['auc_mean'] * 100:.2f} "
        f"± {summary['auc_std'] * 100:.2f}%"
    )


# =========================================================
# 15. Drop from clean
# =========================================================

clean_summary = next(
    row
    for row in summary_rows
    if row["condition"] == "clean"
)


logger.info(
    "\n===== Drop From Clean ====="
)


for row in summary_rows:

    if row["condition"] == "clean":
        continue


    acc_drop = (
        clean_summary["accuracy_mean"]
        -
        row["accuracy_mean"]
    )


    f1_drop = (
        clean_summary["f1_mean"]
        -
        row["f1_mean"]
    )


    auc_drop = (
        clean_summary["auc_mean"]
        -
        row["auc_mean"]
    )


    logger.info(
        f"{row['condition']} | "
        f"Acc drop = {acc_drop * 100:.2f} pp | "
        f"F1 drop = {f1_drop * 100:.2f} pp | "
        f"AUC drop = {auc_drop * 100:.2f} pp"
    )


# =========================================================
# 16. CSV
# =========================================================

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(
        f
    )


    writer.writerow([
        "model",
        "condition",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "auc_mean",
        "auc_std",
    ])


    for row in summary_rows:

        writer.writerow([
            "fusion",
            row["condition"],

            row["accuracy_mean"],
            row["accuracy_std"],

            row["precision_mean"],
            row["precision_std"],

            row["recall_mean"],
            row["recall_std"],

            row["f1_mean"],
            row["f1_std"],

            row["auc_mean"],
            row["auc_std"],
        ])


logger.info(
    f"\nLog saved to: {LOG_PATH}"
)

logger.info(
    f"CSV saved to: {CSV_PATH}"
)