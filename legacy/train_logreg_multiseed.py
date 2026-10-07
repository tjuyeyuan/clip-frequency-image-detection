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
# 1. 读取数据
# ============================================================

X_train = np.load((PROJECT_ROOT / "small_train_features.npy"))
y_train = np.load((PROJECT_ROOT / "small_train_labels.npy"))

X_test = np.load((PROJECT_ROOT / "small_test_features.npy"))
y_test = np.load((PROJECT_ROOT / "small_test_labels.npy"))


# NumPy -> Tensor
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


# ============================================================
# 2. 单次训练函数
# ============================================================

def run_one_seed(seed):

    print("\n==============================")
    print(f"Running seed = {seed}")
    print("==============================")

    # 固定当前实验的随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 每个 seed 都重新初始化一个新模型
    model = nn.Linear(512, 1)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    for epoch in range(200):

        optimizer.zero_grad()

        logits = model(
            X_train
        )

        loss = criterion(
            logits,
            y_train
        )

        loss.backward()

        optimizer.step()

        # 不需要100轮全部打印
        if (
            epoch == 0
            or (epoch + 1) % 20 == 0
        ):
            print(
                f"Epoch {epoch + 1:03d} | "
                f"Loss: {loss.item():.4f}"
            )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        test_logits = model(
            X_test
        )

        test_probs = torch.sigmoid(
            test_logits
        )

        test_preds = (
            test_probs > 0.5
        ).float()

    # Tensor -> NumPy
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

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    print("\nResult:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"AUC:       {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # 返回这一轮结果
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


# ============================================================
# 3. 多个 seed
# ============================================================

seeds = [
    42,
    123,
    2026
]

results = []


for seed in seeds:

    result = run_one_seed(
        seed
    )

    results.append(
        result
    )


# ============================================================
# 4. 计算 mean ± std
# ============================================================

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc"
]


print("\n")
print("====================================")
print("Final Results: Mean ± Std")
print("====================================")


for metric in metrics:

    values = [
        result[metric]
        for result in results
    ]

    mean_value = np.mean(
        values
    )

    std_value = np.std(
        values
    )

    print(
        f"{metric.capitalize():10s}: "
        f"{mean_value * 100:.2f} "
        f"± {std_value * 100:.2f}%"
    )