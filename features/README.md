# 特征与固定样本索引

`train_indices.npy` 和 `test_indices.npy` 保存正式实验的样本位置与顺序，纳入版本管理。其余 `.npy` 为本地生成缓存，通过 `.gitignore` 排除，磁盘内容不删除。

- clean：`aligned_train_clip_features.npy`、`aligned_train_labels.npy`、`aligned_test_clip_features.npy`、`aligned_test_labels.npy`。
- JPEG：`jpeg/test_clip_q{quality}.npy` 与 `test_labels_q{quality}.npy`。
- Noise：`noise/test_clip_sigma{sigma}.npy` 与 `test_labels_sigma{sigma}.npy`。
- Blur：`blur/test_clip_radius{radius}.npy` 与 `test_labels_radius{radius}.npy`。

生成顺序与索引备份要求见 [复现说明](../docs/reproduction.md)。根目录 `small_*.npy` 属于 legacy 流程，不可混用。
