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

from torch.utils.data import TensorDataset, DataLoader

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

SEED = 42

BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3

EXPERIMENT_NAME = "clip_linear_aligned"


# =========================================================
# 1. Paths
# =========================================================

os.makedirs(
    (PROJECT_ROOT / "results"),
    exist_ok=True
)

LOG_PATH = os.path.join(
    (PROJECT_ROOT / "results"),
    f"{EXPERIMENT_NAME}_seed{SEED}.log"
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
# 3. Seed
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================================================
# 4. Experiment config log
# =========================================================

logger.info(
    "========================================"
)

logger.info(
    f"Experiment: {EXPERIMENT_NAME}"
)

logger.info(
    f"Seed: {SEED}"
)

logger.info(
    f"Batch Size: {BATCH_SIZE}"
)

logger.info(
    f"Epochs: {EPOCHS}"
)

logger.info(
    f"Learning Rate: {LR}"
)

logger.info(
    "Model: CLIP feature + Linear(512,1)"
)

logger.info(
    "========================================"
)


# =========================================================
# 5. Load aligned CLIP features
# =========================================================

train_features = np.load(
    (PROJECT_ROOT / "features/aligned_train_clip_features.npy")
)

train_labels = np.load(
    (PROJECT_ROOT / "features/aligned_train_labels.npy")
)

test_features = np.load(
    (PROJECT_ROOT / "features/aligned_test_clip_features.npy")
)

test_labels = np.load(
    (PROJECT_ROOT / "features/aligned_test_labels.npy")
)


logger.info(
    f"Train features shape: {train_features.shape}"
)

logger.info(
    f"Train labels shape: {train_labels.shape}"
)

logger.info(
    f"Test features shape: {test_features.shape}"
)

logger.info(
    f"Test labels shape: {test_labels.shape}"
)


# =========================================================
# 6. NumPy -> Tensor
# =========================================================

train_features = torch.tensor(
    train_features,
    dtype=torch.float32
)

train_labels = torch.tensor(
    train_labels,
    dtype=torch.float32
)

test_features = torch.tensor(
    test_features,
    dtype=torch.float32
)

test_labels = torch.tensor(
    test_labels,
    dtype=torch.float32
)


# =========================================================
# 7. TensorDataset
# =========================================================

train_dataset = TensorDataset(
    train_features,
    train_labels
)

test_dataset = TensorDataset(
    test_features,
    test_labels
)


# =========================================================
# 8. DataLoader
# =========================================================

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


# =========================================================
# 9. Device
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
# 10. Model
# =========================================================

model = nn.Linear(
    512,
    1
)

model = model.to(
    device
)

logger.info(
    f"Model:\n{model}"
)


total_params = sum(
    p.numel()
    for p in model.parameters()
)

logger.info(
    f"Total params: {total_params}"
)

logger.info(
    f"Trainable params: {total_params}"
)


# =========================================================
# 11. Loss + optimizer
# =========================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)


# =========================================================
# 12. Training
# =========================================================

logger.info(
    "===== Training Started ====="
)

for epoch in range(
    EPOCHS
):

    model.train()

    total_loss = 0.0

    for features, labels in train_loader:

        features = features.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad()

        logits = model(
            features
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
        f"Epoch "
        f"{epoch + 1:03d}/{EPOCHS} "
        f"| Loss = {avg_loss:.4f}"
    )

logger.info(
    "===== Training Finished ====="
)


# =========================================================
# 13. Evaluation
# =========================================================

model.eval()

all_probs = []
all_preds = []
all_labels = []

with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(
            device
        )

        logits = model(
            features
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


# =========================================================
# 14. Metrics
# =========================================================

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


# =========================================================
# 15. Log final results
# =========================================================

logger.info(
    "===== Final Test Results ====="
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
    "Confusion Matrix:"
)

logger.info(
    f"\n{cm}"
)


# =========================================================
# 16. Save CSV
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

    tn, fp, fn, tp = (
        cm.ravel()
    )

    writer.writerow([
        EXPERIMENT_NAME,
        SEED,
        len(train_dataset),
        len(test_dataset),
        BATCH_SIZE,
        EPOCHS,
        LR,
        accuracy,
        precision,
        recall,
        f1,
        auc,
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