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
# 0. 实验配置
# =========================================================

SEED = 42

TRAIN_FAKE = 500
TRAIN_REAL = 500

TEST_FAKE = 100
TEST_REAL = 100

BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3

EXPERIMENT_NAME = "frequency_only_v1"


# =========================================================
# 1. 创建 results 目录
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

# 防止重复 handler
logger.handlers.clear()


# 输出到终端
console_handler = logging.StreamHandler()

console_handler.setLevel(
    logging.INFO
)


# 输出到 log 文件
file_handler = logging.FileHandler(
    LOG_PATH,
    mode="w",
    encoding="utf-8"
)

file_handler.setLevel(
    logging.INFO
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
# 3. 固定随机种子
# =========================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


# =========================================================
# 4. 打印实验配置
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
    f"Train FAKE: {TRAIN_FAKE}"
)

logger.info(
    f"Train REAL: {TRAIN_REAL}"
)

logger.info(
    f"Test FAKE: {TEST_FAKE}"
)

logger.info(
    f"Test REAL: {TEST_REAL}"
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
    "========================================"
)


# =========================================================
# 5. 加载 CIFAKE
# =========================================================

logger.info(
    "Loading CIFAKE dataset..."
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
# 6. 平衡随机抽样
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

    return hf_dataset.select(
        selected_indices
    )


small_train = make_balanced_subset(
    dataset["train"],
    TRAIN_FAKE,
    TRAIN_REAL,
    SEED
)

small_test = make_balanced_subset(
    dataset["test"],
    TEST_FAKE,
    TEST_REAL,
    SEED
)


logger.info(
    f"Selected train subset: {len(small_train)}"
)

logger.info(
    f"Selected test subset: {len(small_test)}"
)


# =========================================================
# 7. 图像 -> FFT spectrum
# =========================================================

def image_to_spectrum(
    image
):

    image = np.array(
        image,
        dtype=np.float32
    )

    # RGB -> grayscale
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

    # [32,32] -> [1,32,32]
    spectrum = spectrum.unsqueeze(
        0
    )

    return spectrum


# =========================================================
# 8. Dataset
# =========================================================

class FrequencyDataset(
    Dataset
):

    def __init__(
        self,
        hf_dataset
    ):

        self.data = hf_dataset

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

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return (
            spectrum,
            label
        )


# =========================================================
# 9. Dataset + DataLoader
# =========================================================

train_dataset = FrequencyDataset(
    small_train
)

test_dataset = FrequencyDataset(
    small_test
)


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
# 10. 检查 batch shape
# =========================================================

spectra, labels = next(
    iter(train_loader)
)

logger.info(
    f"Batch spectra shape: {spectra.shape}"
)

logger.info(
    f"Batch labels shape: {labels.shape}"
)


# =========================================================
# 11. Tiny Frequency CNN
# =========================================================

class TinyFreqCNN(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
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
# 12. Frequency Classifier
# =========================================================

class FrequencyClassifier(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()

        self.encoder = TinyFreqCNN()

        self.classifier = nn.Linear(
            128,
            1
        )

    def forward(
        self,
        x
    ):

        features = self.encoder(
            x
        )

        logits = self.classifier(
            features
        )

        return logits


# =========================================================
# 13. Device
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
# 14. Model
# =========================================================

model = FrequencyClassifier()

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

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


logger.info(
    f"Total params: {total_params}"
)

logger.info(
    f"Trainable params: {trainable_params}"
)


# =========================================================
# 15. Loss + Optimizer
# =========================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)


# =========================================================
# 16. Training
# =========================================================

logger.info(
    "===== Training Started ====="
)


for epoch in range(
    EPOCHS
):

    model.train()

    total_loss = 0.0

    for spectra, labels in train_loader:

        spectra = spectra.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad()

        logits = model(
            spectra
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
# 17. Evaluation
# =========================================================

model.eval()

all_probs = []

all_preds = []

all_labels = []


with torch.no_grad():

    for spectra, labels in test_loader:

        spectra = spectra.to(
            device
        )

        logits = model(
            spectra
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
# 18. Metrics
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
# 19. Log Results
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
# 20. 保存 CSV
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
        len(small_train),
        len(small_test),
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