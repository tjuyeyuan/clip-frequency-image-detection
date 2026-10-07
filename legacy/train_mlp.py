from pathlib import Path

# Resolve local files relative to the project root, regardless of working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# 1. 配置
# ============================================================

SEED = 42
EPOCHS = 200
LR = 1e-3


# 固定随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 2. 读取 CLIP 特征
# ============================================================

X_train = np.load(
    (PROJECT_ROOT / "small_train_features.npy")
)

y_train = np.load(
    (PROJECT_ROOT / "small_train_labels.npy")
)

X_test = np.load(
    (PROJECT_ROOT / "small_test_features.npy")
)

y_test = np.load(
    (PROJECT_ROOT / "small_test_labels.npy")
)


# ============================================================
# 3. NumPy -> PyTorch Tensor
# ============================================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
).unsqueeze(1)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test,
    dtype=torch.float32
).unsqueeze(1)


print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# 4. 定义 MLP
# ============================================================

model = nn.Sequential(

    # 512维 CLIP feature
    # ↓
    # 256维隐藏特征
    nn.Linear(512, 256),

    # 非线性激活
    nn.ReLU(),

    # 256维
    # ↓
    # 1个二分类 logit
    nn.Linear(256, 1)
)


print("\nModel:")
print(model)


# ============================================================
# 5. Loss 和 Optimizer
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)


# ============================================================
# 6. Train
# ============================================================

print("\nTraining...")

for epoch in range(EPOCHS):

    # 清空上一轮梯度
    optimizer.zero_grad()

    # -------------------------
    # Forward
    # -------------------------

    logits = model(
        X_train
    )

    # -------------------------
    # Loss
    # -------------------------

    loss = criterion(
        logits,
        y_train
    )

    # -------------------------
    # Backward
    # -------------------------

    loss.backward()

    # -------------------------
    # 更新参数
    # -------------------------

    optimizer.step()


    # 每20轮打印一次
    if (
        epoch == 0
        or (epoch + 1) % 20 == 0
    ):

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {loss.item():.4f}"
        )


# ============================================================
# 7. Test
# ============================================================

model.eval()

with torch.no_grad():

    test_logits = model(
        X_test
    )

    # logit -> probability
    test_probs = torch.sigmoid(
        test_logits
    )

    # probability -> 0 / 1
    test_preds = (
        test_probs > 0.5
    ).float()


# ============================================================
# 8. Tensor -> NumPy
# ============================================================

y_true = (
    y_test
    .cpu()
    .numpy()
    .reshape(-1)
)

y_prob = (
    test_probs
    .cpu()
    .numpy()
    .reshape(-1)
)

y_pred = (
    test_preds
    .cpu()
    .numpy()
    .reshape(-1)
)


# ============================================================
# 9. Metrics
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

auc = roc_auc_score(
    y_true,
    y_prob
)

cm = confusion_matrix(
    y_true,
    y_pred
)


# ============================================================
# 10. 打印结果
# ============================================================

print("\n==============================")
print("MLP Test Results")
print("==============================")

print(
    f"Accuracy:  {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1:        {f1:.4f}"
)

print(
    f"AUC:       {auc:.4f}"
)

print("\nConfusion Matrix:")
print(cm)