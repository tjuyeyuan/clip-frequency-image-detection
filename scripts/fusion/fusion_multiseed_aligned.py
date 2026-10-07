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

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# =========================================================
# 0. Config
# =========================================================

SEEDS = [42, 123, 2026]

BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3

EXPERIMENT_NAME = "fusion_multiseed_aligned"


# =========================================================
# 1. Paths
# =========================================================

os.makedirs(
    (PROJECT_ROOT / "results"),
    exist_ok=True
)

LOG_PATH = os.path.join(
    (PROJECT_ROOT / "results"),
    f"{EXPERIMENT_NAME}.log"
)

CSV_PATH = os.path.join(
    (PROJECT_ROOT / "results"),
    "experiment_results.csv"
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
# 3. Load CIFAKE
# =========================================================

logger.info(
    "Loading CIFAKE..."
)

dataset = load_dataset(
    "dragonintelligence/CIFAKE-image-dataset"
)


# =========================================================
# 4. Load fixed indices
# =========================================================

train_indices = np.load(
    (PROJECT_ROOT / "features/train_indices.npy")
)

test_indices = np.load(
    (PROJECT_ROOT / "features/test_indices.npy")
)

small_train = dataset["train"].select(
    train_indices.tolist()
)

small_test = dataset["test"].select(
    test_indices.tolist()
)


# =========================================================
# 5. Load aligned CLIP features
# =========================================================

train_clip_features = np.load(
    (PROJECT_ROOT / "features/aligned_train_clip_features.npy")
)

train_clip_labels = np.load(
    (PROJECT_ROOT / "features/aligned_train_labels.npy")
)

test_clip_features = np.load(
    (PROJECT_ROOT / "features/aligned_test_clip_features.npy")
)

test_clip_labels = np.load(
    (PROJECT_ROOT / "features/aligned_test_labels.npy")
)


logger.info(
    f"Train CLIP features: {train_clip_features.shape}"
)

logger.info(
    f"Test CLIP features: {test_clip_features.shape}"
)


# =========================================================
# 6. Alignment check
# =========================================================

train_hf_labels = np.array(
    small_train["label"]
)

test_hf_labels = np.array(
    small_test["label"]
)

assert np.array_equal(
    train_hf_labels,
    train_clip_labels
)

assert np.array_equal(
    test_hf_labels,
    test_clip_labels
)

logger.info(
    "Label alignment check passed."
)


# =========================================================
# 7. FFT
# =========================================================

def image_to_spectrum(image):

    image = np.array(
        image,
        dtype=np.float32
    )

    if image.ndim == 3:

        image = image.mean(
            axis=2
        )

    fft = np.fft.fft2(
        image
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
    )

    spectrum = spectrum.unsqueeze(
        0
    )

    return spectrum


# =========================================================
# 8. Dataset
# =========================================================

class FusionDataset(Dataset):

    def __init__(
        self,
        hf_dataset,
        clip_features
    ):

        self.data = hf_dataset

        self.clip_features = torch.tensor(
            clip_features,
            dtype=torch.float32
        )

    def __len__(
        self
    ):

        return len(
            self.data
        )

    def __getitem__(
        self,
        idx
    ):

        sample = self.data[idx]

        image = sample["image"]

        label = sample["label"]

        spectrum = image_to_spectrum(
            image
        )

        clip_feature = self.clip_features[
            idx
        ]

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return (
            clip_feature,
            spectrum,
            label
        )


# =========================================================
# 9. Dataset objects
# =========================================================

train_dataset = FusionDataset(
    small_train,
    train_clip_features
)

test_dataset = FusionDataset(
    small_test,
    test_clip_features
)


# =========================================================
# 10. Device
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
# 11. Seed helper
# =========================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# =========================================================
# 12. Frequency CNN
# =========================================================

class TinyFreqCNN(nn.Module):

    def __init__(
        self
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            1,
            16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(
            2
        )

        self.gap = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.fc = nn.Linear(
            32,
            128
        )

    def forward(
        self,
        x
    ):

        x = self.conv1(
            x
        )

        x = self.relu(
            x
        )

        x = self.pool(
            x
        )

        x = self.conv2(
            x
        )

        x = self.relu(
            x
        )

        x = self.pool(
            x
        )

        x = self.gap(
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
# 13. Fusion model
# =========================================================

class FusionClassifier(nn.Module):

    def __init__(
        self
    ):

        super().__init__()

        self.freq_encoder = TinyFreqCNN()

        self.classifier = nn.Sequential(

            nn.Linear(
                640,
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
# 14. Evaluation helper
# =========================================================

def evaluate_model(
    model,
    test_loader
):

    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():

        for (
            clip_feature,
            spectrum,
            labels
        ) in test_loader:

            clip_feature = clip_feature.to(
                device
            )

            spectrum = spectrum.to(
                device
            )

            logits = model(
                clip_feature,
                spectrum
            )

            logits = logits.squeeze(
                1
            )

            probs = torch.sigmoid(
                logits
            )

            preds = (
                probs > 0.5
            ).long()

            all_probs.extend(
                probs
                .cpu()
                .numpy()
            )

            all_preds.extend(
                preds
                .cpu()
                .numpy()
            )

            all_labels.extend(
                labels
                .cpu()
                .numpy()
            )


    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    precision = precision_score(
        all_labels,
        all_preds
    )

    recall = recall_score(
        all_labels,
        all_preds
    )

    f1 = f1_score(
        all_labels,
        all_preds
    )

    auc = roc_auc_score(
        all_labels,
        all_probs
    )

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        auc,
        cm
    )


# =========================================================
# 15. Experiment config
# =========================================================

logger.info(
    "========================================"
)

logger.info(
    f"Experiment: {EXPERIMENT_NAME}"
)

logger.info(
    f"Seeds: {SEEDS}"
)

logger.info(
    f"Batch size: {BATCH_SIZE}"
)

logger.info(
    f"Epochs: {EPOCHS}"
)

logger.info(
    f"Learning rate: {LR}"
)

logger.info(
    "Fusion: CLIP 512D + Frequency 128D"
)

logger.info(
    "========================================"
)


all_results = []


# =========================================================
# 16. Multi-seed training
# =========================================================

for seed in SEEDS:

    logger.info(
        ""
    )

    logger.info(
        "========================================"
    )

    logger.info(
        f"Seed {seed} Started"
    )

    logger.info(
        "========================================"
    )


    # -----------------------------------------------------
    # Set seed
    # -----------------------------------------------------

    set_seed(
        seed
    )


    # -----------------------------------------------------
    # Rebuild DataLoader
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # -----------------------------------------------------
    # Rebuild model
    # -----------------------------------------------------

    model = FusionClassifier()

    model = model.to(
        device
    )


    # -----------------------------------------------------
    # Loss + optimizer
    # -----------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )


    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    for epoch in range(
        EPOCHS
    ):

        model.train()

        total_loss = 0.0

        for (
            clip_feature,
            spectrum,
            labels
        ) in train_loader:

            clip_feature = clip_feature.to(
                device
            )

            spectrum = spectrum.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                clip_feature,
                spectrum
            )

            logits = logits.squeeze(
                1
            )

            loss = criterion(
                logits,
                labels
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


    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    (
        accuracy,
        precision,
        recall,
        f1,
        auc,
        cm
    ) = evaluate_model(
        model,
        test_loader
    )


    logger.info(
        f"===== Seed {seed} Results ====="
    )

    logger.info(
        f"Accuracy  : {accuracy * 100:.2f}%"
    )

    logger.info(
        f"Precision : {precision * 100:.2f}%"
    )

    logger.info(
        f"Recall    : {recall * 100:.2f}%"
    )

    logger.info(
        f"F1        : {f1 * 100:.2f}%"
    )

    logger.info(
        f"AUC       : {auc * 100:.2f}%"
    )

    logger.info(
        f"Confusion Matrix:\n{cm}"
    )


    all_results.append({

        "seed": seed,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "auc": auc,

        "cm": cm,
    })


# =========================================================
# 17. Mean ± std
# =========================================================

accuracies = np.array([
    r["accuracy"]
    for r in all_results
])

precisions = np.array([
    r["precision"]
    for r in all_results
])

recalls = np.array([
    r["recall"]
    for r in all_results
])

f1s = np.array([
    r["f1"]
    for r in all_results
])

aucs = np.array([
    r["auc"]
    for r in all_results
])


logger.info(
    ""
)

logger.info(
    "========================================"
)

logger.info(
    "===== Multi-Seed Final Results ====="
)

logger.info(
    "========================================"
)


logger.info(
    f"Accuracy  : "
    f"{accuracies.mean() * 100:.2f} "
    f"± {accuracies.std() * 100:.2f}%"
)

logger.info(
    f"Precision : "
    f"{precisions.mean() * 100:.2f} "
    f"± {precisions.std() * 100:.2f}%"
)

logger.info(
    f"Recall    : "
    f"{recalls.mean() * 100:.2f} "
    f"± {recalls.std() * 100:.2f}%"
)

logger.info(
    f"F1        : "
    f"{f1s.mean() * 100:.2f} "
    f"± {f1s.std() * 100:.2f}%"
)

logger.info(
    f"AUC       : "
    f"{aucs.mean() * 100:.2f} "
    f"± {aucs.std() * 100:.2f}%"
)


# =========================================================
# 18. Save CSV
# =========================================================

file_exists = os.path.exists(
    CSV_PATH
)


with open(
    CSV_PATH,
    "a",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(
        f
    )


    if not file_exists:

        writer.writerow([
            "experiment",
            "seed",
            "train_size",
            "test_size",
            "batch_size",
            "epochs",
            "lr",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "tn",
            "fp",
            "fn",
            "tp",
        ])


    for result in all_results:

        tn, fp, fn, tp = (
            result["cm"].ravel()
        )

        writer.writerow([
            EXPERIMENT_NAME,
            result["seed"],
            len(train_dataset),
            len(test_dataset),
            BATCH_SIZE,
            EPOCHS,
            LR,
            result["accuracy"],
            result["precision"],
            result["recall"],
            result["f1"],
            result["auc"],
            tn,
            fp,
            fn,
            tp,
        ])


logger.info(
    f"Log saved to: {LOG_PATH}"
)

logger.info(
    f"CSV saved to: {CSV_PATH}"
)