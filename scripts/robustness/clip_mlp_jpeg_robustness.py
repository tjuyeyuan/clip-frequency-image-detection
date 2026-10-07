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

SEEDS = [42, 123, 2026]

BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3

QUALITIES = [90, 70, 50, 30]

EXPERIMENT_NAME = "clip_mlp_jpeg_robustness"


# =========================================================
# 1. Paths + logging
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
    "jpeg_robustness_results.csv"
)


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
# 2. Seed
# =========================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# =========================================================
# 3. Device
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
# 4. Load clean train
# =========================================================

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


# =========================================================
# 5. Load clean test
# =========================================================

clean_test_features = np.load(
    (PROJECT_ROOT / "features/aligned_test_clip_features.npy")
)

clean_test_labels = np.load(
    (PROJECT_ROOT / "features/aligned_test_labels.npy")
)


# =========================================================
# 6. Load JPEG test features
# =========================================================

test_conditions = {
    "clean": (
        clean_test_features,
        clean_test_labels
    )
}


for quality in QUALITIES:

    features = np.load(
        (PROJECT_ROOT / f"features/jpeg/test_clip_q{quality}.npy")
    )

    labels = np.load(
        (PROJECT_ROOT / f"features/jpeg/test_labels_q{quality}.npy")
    )

    test_conditions[
        f"jpeg_q{quality}"
    ] = (
        features,
        labels
    )


# =========================================================
# 7. Label alignment check
# =========================================================

for condition, (_, labels) in test_conditions.items():

    assert np.array_equal(
        labels,
        clean_test_labels
    ), (
        f"Label mismatch: {condition}"
    )


logger.info(
    "All test label alignment checks passed."
)


# =========================================================
# 8. Create model
# =========================================================

def create_model():

    return nn.Sequential(

        nn.Linear(
            512,
            256
        ),

        nn.ReLU(),

        nn.Linear(
            256,
            1
        )
    )


# =========================================================
# 9. Evaluation
# =========================================================

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

            x = x.to(
                device
            )

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


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "cm": cm,
    }


# =========================================================
# 10. Experiment
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
    f"Epochs: {EPOCHS}"
)

logger.info(
    f"LR: {LR}"
)

logger.info(
    f"JPEG qualities: {QUALITIES}"
)

logger.info(
    "Train: clean only"
)

logger.info(
    "Test: clean + JPEG"
)

logger.info(
    "========================================"
)


all_results = []


# =========================================================
# 11. Multi-seed
# =========================================================

for seed in SEEDS:

    logger.info(
        f"\n===== Seed {seed} Started ====="
    )

    set_seed(
        seed
    )


    # 每个 seed 重建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    # 每个 seed 全新模型
    model = create_model().to(
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
            ).squeeze(1)


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


    # =====================================================
    # 同一个模型测试所有 condition
    # =====================================================

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
            **result,
        })


# =========================================================
# 12. Mean ± std
# =========================================================

logger.info(
    "\n========================================"
)

logger.info(
    "===== JPEG Robustness Summary ====="
)

logger.info(
    "========================================"
)


summary_rows = []


for condition in test_conditions.keys():

    condition_results = [
        r
        for r in all_results
        if r["condition"] == condition
    ]


    acc = np.array([
        r["accuracy"]
        for r in condition_results
    ])

    precision = np.array([
        r["precision"]
        for r in condition_results
    ])

    recall = np.array([
        r["recall"]
        for r in condition_results
    ])

    f1 = np.array([
        r["f1"]
        for r in condition_results
    ])

    auc = np.array([
        r["auc"]
        for r in condition_results
    ])


    logger.info(
        f"{condition}"
    )

    logger.info(
        f"  Accuracy : "
        f"{acc.mean() * 100:.2f} "
        f"± {acc.std() * 100:.2f}%"
    )

    logger.info(
        f"  F1       : "
        f"{f1.mean() * 100:.2f} "
        f"± {f1.std() * 100:.2f}%"
    )

    logger.info(
        f"  AUC      : "
        f"{auc.mean() * 100:.2f} "
        f"± {auc.std() * 100:.2f}%"
    )


    summary_rows.append({
        "condition": condition,

        "acc_mean": acc.mean(),
        "acc_std": acc.std(),

        "precision_mean": precision.mean(),
        "precision_std": precision.std(),

        "recall_mean": recall.mean(),
        "recall_std": recall.std(),

        "f1_mean": f1.mean(),
        "f1_std": f1.std(),

        "auc_mean": auc.mean(),
        "auc_std": auc.std(),
    })


# =========================================================
# 13. Performance drop from clean
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
        clean_summary["acc_mean"]
        -
        row["acc_mean"]
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
# 14. Save CSV
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
            "clip_mlp",
            row["condition"],
            row["acc_mean"],
            row["acc_std"],
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