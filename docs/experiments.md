# 实验登记与历史对应关系

这里的“正式”指本仓库的 aligned 比较流程；不是完整 CIFAKE 基准。所有名称、配置、模型实现和执行顺序均保留原样。原始实验 CSV 保留在 `results/`，不改写历史数值。

## 正式实验入口

下表路径相对于项目根目录；旧位置为根目录中的同名文件。

| 当前入口 | experiment 名称 / 用途 | seed | 输出 |
|---|---|---|---|
| `scripts/extract/extract_clip_features_aligned.py` | aligned_clip_feature_extraction | 抽样 42 | aligned 特征、标签、train/test indices；提取日志 |
| `scripts/baselines/clip_linear_aligned.py` | clip_linear_aligned | 42 | experiment_results.csv；对应日志 |
| `scripts/baselines/clip_mlp_aligned.py` | clip_mlp_aligned | 42 | experiment_results.csv；对应日志 |
| `scripts/baselines/clip_mlp_multiseed_aligned.py` | clip_mlp_multiseed_aligned | 42,123,2026 | experiment_results.csv 每 seed 一行；对应日志 |
| `scripts/baselines/frequency_baseline.py` | frequency_only_v1 | 42 | experiment_results.csv；对应日志 |
| `scripts/fusion/fusion_v1.py` | fusion_v1 | 42 | experiment_results.csv；对应日志 |
| `scripts/fusion/fusion_multiseed_aligned.py` | fusion_multiseed_aligned | 42,123,2026 | experiment_results.csv 每 seed 一行；对应日志 |

正式训练入口均使用 batch size 64、100 epochs、学习率 0.001、Adam、BCEWithLogitsLoss；均以 `sigmoid(logit) > 0.5` 分类。`0=FAKE, 1=REAL`，正类指标对应 REAL。

Frequency-only 仍使用自己的 `make_balanced_subset`，不会读取已保存 indices；其 seed=42 的采样算法和数量与 aligned 提取脚本对应。没有为了统一接口将它改写为索引读取。其样本对齐依赖相同的数据集版本及顺序。

`frequency_baseline.py` 与 `fusion_v1.py` 在模型初始化前调用 `next(iter(train_loader))`；多 seed Fusion 不做这一步。此顺序影响随机数状态，原样保留。不同脚本中的频谱函数、模型属性名称和设备选择也没有合并。

## 鲁棒性入口

所有路径分别位于 `scripts/extract/` 或 `scripts/robustness/`；每个评估入口均从 clean 数据训练三个 seed，然后评估 clean 和四个退化档位。

| 退化 | 提取脚本 | CLIP+MLP 评估脚本 → CSV | Fusion 评估脚本 → CSV |
|---|---|---|---|
| JPEG | extract_jpeg_clip_features.py | clip_mlp_jpeg_robustness.py → jpeg_robustness_results.csv | fusion_jpeg_robustness.py → fusion_jpeg_robustness_results.csv |
| Noise | extract_noise_clip_features.py | clip_mlp_noise_robustness.py → noise_robustness_results.csv | fusion_noise_robustness.py → fusion_noise_robustness_results.csv |
| Blur | extract_blur_clip_features.py | clip_mlp_blur_robustness.py → clip_mlp_blur_robustness_results.csv | fusion_blur_robustness.py → fusion_blur_robustness_results.csv |

缓存分别位于 `features/jpeg/`、`features/noise/`、`features/blur/`。JPEG quality 为 `[90,70,50,30]`，noise sigma 为 `[5,10,20,30]`，blur radius 为 `[0.5,1.0,1.5,2.0]`。所有评估结果 CSV 在 `results/`；每个脚本对应同名日志（特征提取日志按原 EXPERIMENT_NAME 命名）。

噪声缓存提取使用 `seed=start+i`，Fusion 频域分支使用 `seed=idx`，两者都是测试子集位置。不可改成原数据集索引或训练 seed。Fusion 的扰动频谱来自退化图像；不混用 clean 频谱。三类退化各自独立评估，没有组合退化实验。

## Legacy 与正式实验对应关系

legacy 文件旧位置均为根目录，现位于 `legacy/`，没有删除。

| Legacy 入口 | 对应正式入口 | 必须保留的差异 |
|---|---|---|
| extract_clip_features.py | scripts/extract/extract_clip_features_aligned.py | legacy 每类取前 500/100 个再 shuffle(42)；正式先按类别打乱候选索引再抽样，保存全局索引 |
| train_logreg.py | scripts/baselines/clip_linear_aligned.py | 都是 PyTorch Linear(512,1)；legacy 全批量、CPU、100 epochs，正式 mini-batch 64、设备选择 |
| train_logreg_multiseed.py | 无直接正式多 seed Linear 对应项 | legacy 全批量、CPU、200 epochs、三个 seed；不能当作 aligned Linear 多 seed 结果 |
| train_mlp.py | scripts/baselines/clip_mlp_aligned.py | legacy 全批量、CPU、200 epochs；正式 mini-batch、100 epochs |
| train_mlp_multiseed.py | scripts/baselines/clip_mlp_multiseed_aligned.py | 都使用三个 seed；legacy 全批量、CPU、200 epochs，数据也不同 |

Legacy 读取根目录的 `small_train_features.npy`、`small_train_labels.npy`、`small_test_features.npy`、`small_test_labels.npy`，不读取 aligned 缓存。`train_logreg*` 名字虽然含 logreg，实际实现是 PyTorch 线性分类头，不是 sklearn LogisticRegression。早期提取脚本的 CLIP 输出处理与设备选择也保持原样。

现有 `results/linear_baseline.log` 与 `results/mlp_baseline_200epoch.log` 分别记录早期多 seed Linear 和 MLP 的输出；旧脚本自身主要打印到终端，日志需要重定向。它们没有写入正式 `experiment_results.csv`，因此 README 主结果不混用这些日志。

## 调试与参考文件

`debug/` 的七个脚本保持原位置：CLIP 演示、CIFAKE 下载检查、三个 FFT 演示/对比、JPEG 尺寸检查和 blur 频谱统计诊断。这里的 FFT 演示采用 PIL `convert("L")`，正式频谱采用通道均值；没有统一这两种处理。

`data/real.jpg`、`data/ai.jpg` 是 CLIP 演示所用本地图片；`data/printZm.pdf` 是本地参考文件，没有被训练代码读取。素材不作为实验设定或重构指令来源，也不默认上传。

## 输出与报告约定

- 正式 clean 实验向 `experiment_results.csv` 追加记录；重复运行可能产生重复行。
- robustness CSV 覆盖写入，保存三个 seed 的 mean/std；每 seed 细节在日志中。
- 脚本内部日志以覆盖模式写入；历史日志保留本地，但默认不加入版本管理。
- README 数值来自整理前 CSV，百分数按两位小数展示；差值按原精度计算。单 seed 不伪装成 mean±std。
- 没有新增实验、checkpoint 保存、模型选择、阈值调优或结果清洗。
