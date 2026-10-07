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
)


SEEDS = [42, 123, 2026]

BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3

RADII = [0.5, 1.0, 1.5, 2.0]

EXPERIMENT_NAME = "clip_mlp_blur_robustness"


os.makedirs((PROJECT_ROOT / "results"), exist_ok=True)

LOG_PATH = (PROJECT_ROOT / f"results/{EXPERIMENT_NAME}.log")
CSV_PATH = (PROJECT_ROOT / "results/clip_mlp_blur_robustness_results.csv")


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


if torch.backends.mps.is_available():
    device = torch.device("mps")

elif torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")


logger.info(f"Using device: {device}")


train_features = np.load(
    (PROJECT_ROOT / "features/aligned_train_clip_features.npy")
)

train_labels = np.load(
    (PROJECT_ROOT / "features/aligned_train_labels.npy")
)

train_features = torch.tensor(
    train_features,
    dtype=torch.float32
)

train_labels = torch.tensor(
    train_labels,
    dtype=torch.float32
)

train_dataset = TensorDataset(
    train_features,
    train_labels
)


clean_test_features = np.load(
    (PROJECT_ROOT / "features/aligned_test_clip_features.npy")
)

clean_test_labels = np.load(
    (PROJECT_ROOT / "features/aligned_test_labels.npy")
)


test_conditions = {
    "clean": (
        clean_test_features,
        clean_test_labels
    )
}


for radius in RADII:

    features = np.load(
        (PROJECT_ROOT / f"features/blur/test_clip_radius{radius}.npy")
    )

    labels = np.load(
        (PROJECT_ROOT / f"features/blur/test_labels_radius{radius}.npy")
    )

    test_conditions[
        f"blur_radius{radius}"
    ] = (
        features,
        labels
    )


for condition, (_, labels) in test_conditions.items():

    assert np.array_equal(
        labels,
        clean_test_labels
    ), f"Label mismatch: {condition}"


def create_model():

    return nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )


def evaluate_model(
    model,
    features,
    labels
):

    features = torch.tensor(
        features,
        dtype=torch.float32
    )

    labels = torch.tensor(
        labels,
        dtype=torch.float32
    )

    dataset = TensorDataset(
        features,
        labels
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            logits = model(
                x
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
                y.numpy()
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = create_model().to(
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

        for features, labels in train_loader:

            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(
                features
            ).squeeze(1)

            loss = criterion(
                logits,
                labels
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

    for condition, (
        features,
        labels
    ) in test_conditions.items():

        result = evaluate_model(
            model,
            features,
            labels
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
    "\n===== CLIP+MLP Gaussian Blur Robustness Summary ====="
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
            "clip_mlp",
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