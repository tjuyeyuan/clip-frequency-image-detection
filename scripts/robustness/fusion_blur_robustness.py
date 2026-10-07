from pathlib import Path

# Resolve local files relative to the project root, regardless of working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

import os
import csv
import random
import logging

import numpy as np
import torch
import torch.nn as nn

from PIL import ImageFilter
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


SEEDS = [42, 123, 2026]

BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3

RADII = [0.5, 1.0, 1.5, 2.0]

EXPERIMENT_NAME = "fusion_blur_robustness"


os.makedirs((PROJECT_ROOT / "results"), exist_ok=True)

LOG_PATH = (PROJECT_ROOT / f"results/{EXPERIMENT_NAME}.log")
CSV_PATH = (PROJECT_ROOT / "results/fusion_blur_robustness_results.csv")


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


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def gaussian_blur(
    image,
    radius=1.0
):

    image = image.convert("RGB")

    return image.filter(
        ImageFilter.GaussianBlur(
            radius=radius
        )
    )


def image_to_spectrum(image):

    image = image.convert("RGB")

    image_np = np.array(
        image,
        dtype=np.float32
    )

    gray = image_np.mean(
        axis=2
    )

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


if torch.backends.mps.is_available():
    device = torch.device("mps")

elif torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")


logger.info(
    f"Using device: {device}"
)


class FusionDataset(Dataset):

    def __init__(
        self,
        hf_dataset,
        clip_features,
        labels,
        radius=None
    ):

        self.hf_dataset = hf_dataset
        self.clip_features = clip_features
        self.labels = labels
        self.radius = radius


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

        if self.radius is not None:

            image = gaussian_blur(
                image,
                radius=self.radius
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


assert np.array_equal(
    np.array(train_hf["label"]),
    train_labels
)

assert np.array_equal(
    np.array(test_hf["label"]),
    test_labels
)


test_conditions = {
    "clean": {
        "clip": clean_test_clip,
        "radius": None
    }
}


for radius in RADII:

    blur_clip = np.load(
        (PROJECT_ROOT / f"features/blur/test_clip_radius{radius}.npy")
    )

    test_conditions[
        f"blur_radius{radius}"
    ] = {
        "clip": blur_clip,
        "radius": radius
    }


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


all_results = []


for seed in SEEDS:

    logger.info(
        f"\n===== Seed {seed} Started ====="
    )

    set_seed(seed)

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

    for epoch in range(EPOCHS):

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

            total_loss += loss.item()

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

    for condition, config in test_conditions.items():

        test_dataset = FusionDataset(
            test_hf,
            config["clip"],
            test_labels,
            radius=config["radius"]
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
            f"===== Seed {seed} | {condition} ====="
        )

        logger.info(
            f"Accuracy : {result['accuracy'] * 100:.2f}%"
        )

        logger.info(
            f"F1       : {result['f1'] * 100:.2f}%"
        )

        logger.info(
            f"AUC      : {result['auc'] * 100:.2f}%"
        )

        all_results.append({
            "seed": seed,
            "condition": condition,
            **result
        })


logger.info(
    "\n===== Fusion Gaussian Blur Robustness Summary ====="
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

    summary_rows.append(summary)

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

    logger.info(
        f"{row['condition']} | "
        f"Acc drop = "
        f"{(clean_summary['accuracy_mean'] - row['accuracy_mean']) * 100:.2f} pp | "
        f"F1 drop = "
        f"{(clean_summary['f1_mean'] - row['f1_mean']) * 100:.2f} pp | "
        f"AUC drop = "
        f"{(clean_summary['auc_mean'] - row['auc_mean']) * 100:.2f} pp"
    )


with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

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