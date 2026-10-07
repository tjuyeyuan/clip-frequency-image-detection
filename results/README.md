# 已有实验结果

这里的七个 CSV 是工程整理前保留的原始结果，没有重新计算或修改。

- `experiment_results.csv`：正式 clean 实验逐次/逐 seed 结果，脚本追加写入。
- `jpeg_robustness_results.csv`、`noise_robustness_results.csv`、`clip_mlp_blur_robustness_results.csv`：CLIP+MLP 鲁棒性汇总。
- `fusion_jpeg_robustness_results.csv`、`fusion_noise_robustness_results.csv`、`fusion_blur_robustness_results.csv`：Fusion 鲁棒性汇总。

robustness CSV 覆盖写入，指标为三个训练 seed 的均值和总体标准差（ddof=0）。CSV 数值为 0–1 尺度，README 表格转换为百分数。正类是 REAL。

历史 `.log` 保留本地，Git 默认忽略；训练脚本会覆盖自己的日志。复跑前备份需要保留的 CSV 和日志。正式与 legacy 记录对应关系见 [实验登记](../docs/experiments.md)。
