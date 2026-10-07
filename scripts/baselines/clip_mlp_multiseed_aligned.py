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

EXPERIMENT_NAME = "clip_mlp_multiseed_aligned"


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
# 3. Load aligned CLIP features
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
    f"Train features: {train_features.shape}"
)

logger.info(
    f"Train labels: {train_labels.shape}"
)

logger.info(
    f"Test features: {test_features.shape}"
)

logger.info(
    f"Test labels: {test_labels.shape}"
)


# =========================================================
# 4. NumPy -> Tensor
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
# 5. Dataset
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
# 6. Device
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
# 7. Helper: set seed
# =========================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# =========================================================
# 8. Helper: create model
# =========================================================

def create_model():

    model = nn.Sequential(

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

    return model


# =========================================================
# 9. Helper: evaluate
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
# 10. Experiment start
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
    "Model: CLIP 512 -> 256 -> ReLU -> 1"
)

logger.info(
    "========================================"
)


all_results = []


# =========================================================
# 11. Multi-seed loop
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
    # 固定随机种子
    # -----------------------------------------------------

    set_seed(
        seed
    )


    # -----------------------------------------------------
    # 每个 seed 重新创建 DataLoader
    # shuffle 顺序会受 seed 影响
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
    # 每个 seed 都重新创建模型
    # 这一点非常重要
    # -----------------------------------------------------

    model = create_model()

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
            f"Seed {seed} | "
            f"Epoch {epoch + 1:03d}/{EPOCHS} "
            f"| Loss = {avg_loss:.4f}"
        )


    # -----------------------------------------------------
    # Evaluation
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
# 12. Mean ± std
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
# 13. Save CSV
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


    # 每个 seed 单独保存
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