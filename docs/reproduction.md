# 运行与复现说明

## 环境

从项目根目录创建虚拟环境，并安装 `requirements.txt`。直接依赖根据源码导入列出；历史训练环境没有可核实的版本锁定记录。本次整理未安装或升级依赖，也未验证全新环境中的训练结果。尤其 CLIP `get_image_features` 返回形式依赖 transformers 版本；保留原有处理代码，优先使用原实验环境。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

正式入口设备选择保留为 MPS → CUDA → CPU；legacy 提取为 MPS → CPU，legacy 分类器在 CPU 上运行。跨设备数值结果可能不同。不要导入实验脚本做工具函数复用：它们仍包含顶层执行代码。

## 数据和索引

脚本通过 Hugging Face datasets 加载 `dragonintelligence/CIFAKE-image-dataset`，CLIP 为 `openai/clip-vit-base-patch32`。首次运行需要下载资源；`data/` 只是本地示例素材目录。

随仓库保留 `features/train_indices.npy`（1000 项）、`features/test_indices.npy`（200 项）。它们是原始数据集 split 中的位置，顺序必须保留。标签约定为 `0=FAKE、1=REAL`。历史代码未固定远端 revision，因此索引文件本身无法保证在任意未来数据集版本上对应相同图像。

本地已有 aligned 缓存时，直接使用，不需要重新提取。Git 忽略特征缓存，新的 clone 需要自行生成。原 clean 提取脚本会按既有算法重新生成并覆盖 indices，因此在重建缓存前先备份：

```bash
# 以下从项目根目录执行。
# 使用一个尚不存在的备份目录，避免覆盖以前的备份。
mkdir ../cifake-indices-before-extraction
cp features/train_indices.npy features/test_indices.npy ../cifake-indices-before-extraction/
python scripts/extract/extract_clip_features_aligned.py
cmp features/train_indices.npy ../cifake-indices-before-extraction/train_indices.npy
cmp features/test_indices.npy ../cifake-indices-before-extraction/test_indices.npy
```

两个比较均应无差异。若不同，停止后续训练，检查数据集版本和环境；不要将新缓存与旧索引混用。为保留实验逻辑，本次未改写提取器为“只读取索引”。

clean 缓存：`aligned_train_clip_features.npy` 为 (1000,512)，`aligned_test_clip_features.npy` 为 (200,512)，特征为 float32；对应标签为 int64，形状 (1000,) / (200,)。

## 正式 clean 实验

运行前备份 `results/` 中要保留的历史输出：clean CSV 会追加，日志会覆盖。下列入口是独立实验，可按需执行。

```bash
python scripts/baselines/clip_linear_aligned.py
python scripts/baselines/clip_mlp_aligned.py
python scripts/baselines/frequency_baseline.py
python scripts/fusion/fusion_v1.py
python scripts/baselines/clip_mlp_multiseed_aligned.py
python scripts/fusion/fusion_multiseed_aligned.py
```

Frequency-only 自行加载数据并按原算法抽样。CLIP-only 分类器需要 aligned 缓存；Fusion 还需要数据集与保存的索引，并保留原有标签对齐断言。

## 鲁棒性实验

先具备 clean 特征、标签和索引，再提取扰动特征。每种退化缓存有四档，每档特征 (200,512)、标签 (200,)。之后执行对应评估入口：

```bash
python scripts/extract/extract_jpeg_clip_features.py
python scripts/robustness/clip_mlp_jpeg_robustness.py
python scripts/robustness/fusion_jpeg_robustness.py

python scripts/extract/extract_noise_clip_features.py
python scripts/robustness/clip_mlp_noise_robustness.py
python scripts/robustness/fusion_noise_robustness.py

python scripts/extract/extract_blur_clip_features.py
python scripts/robustness/clip_mlp_blur_robustness.py
python scripts/robustness/fusion_blur_robustness.py
```

提取脚本覆盖对应特征缓存，评估脚本覆盖各自 robustness CSV 和日志。每个评估脚本从 clean 数据重新训练三个 seed，而不是读取已保存 checkpoint。

## Legacy 和调试

```bash
python legacy/extract_clip_features.py
python legacy/train_logreg.py
python legacy/train_logreg_multiseed.py
python legacy/train_mlp.py
python legacy/train_mlp_multiseed.py
python debug/diagnose_blur_spectrum_stats.py
```

Legacy 提取默认生成根目录 `small_*.npy`；不覆盖 aligned 缓存。所有脚本名称和默认参数保留。CLIP 图片演示需要自行提供 `data/real.jpg` 和 `data/ai.jpg`，这些本地素材不随仓库上传。

## 路径与整理验收边界

有本地文件读写的脚本通过 `Path(__file__).resolve()` 定位项目根目录，文件位置仍写为 `features/...`、`results/...`、`data/...` 或根目录 `small_*.npy`。因此可从其他工作目录调用脚本；不需要修改电脑用户名或目录常量，也不改变全局工作目录。

本次整理验证语法、迁移前后核心 AST 等价、路径解析、索引与缓存/结果校验和，不以重新训练结果宣称运行等价。未重新抽样、重算特征、重训模型或覆盖实验输出。
