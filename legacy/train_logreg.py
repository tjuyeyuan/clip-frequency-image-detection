from pathlib import Path

# Resolve local files relative to the project root, regardless of working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import torch
import torch.nn as nn
torch.manual_seed(42)
np.random.seed(42)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
# =========================
# 1. 数据
# =========================

X_train = np.load((PROJECT_ROOT / "small_train_features.npy"))
y_train = np.load((PROJECT_ROOT / "small_train_labels.npy"))

X_test = np.load((PROJECT_ROOT / "small_test_features.npy"))
y_test = np.load((PROJECT_ROOT / "small_test_labels.npy"))

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


# =========================
# 2. 模型
# =========================

model = nn.Linear(512, 1)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


# =========================
# 3. 训练
# =========================

for epoch in range(100):

    optimizer.zero_grad()

    logits = model(X_train)

    loss = criterion(
        logits,
        y_train
    )

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Loss: {loss.item():.4f}"
    )


# =========================
# 4. 测试
# =========================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

with torch.no_grad():

    test_logits = model(X_test)

    test_probs = torch.sigmoid(
        test_logits
    )

    test_preds = (
        test_probs > 0.5
    ).float()


# 转成 NumPy
y_true = y_test.cpu().numpy().reshape(-1)
y_prob = test_probs.cpu().numpy().reshape(-1)
y_pred = test_preds.cpu().numpy().reshape(-1)


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


print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
print(f"AUC:       {auc:.4f}")

print("\nConfusion Matrix:")
print(cm)
