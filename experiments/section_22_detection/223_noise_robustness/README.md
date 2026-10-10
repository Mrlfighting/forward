# 论文 2.2.3：噪声条件下的静态测点识别

本实验读取论文 2.1 已保存的双浅部体及无浅部体 TM 物理正演数据。两处浅部体中心位于 −5 km、+5 km，均为 10 Ω·m、宽 200 m、厚 5 m。输入为 41 个测点 × 20 个频率的视电阻率；不再次运行 SimPEG，也不向双浅部体数据乘人工 `Cs`。无浅部体数据构成配对阴性对照。

配置位于 `configs/section_22/noise_robustness.json`。0%、2%、5%、10%、15% 为视电阻率独立相对高斯噪声标准差；非零水平各做 30 次独立重复。每个重复使用同一标准正态数组作用于双浅部体和无浅部体数据，也在各非零噪声级别复用，以便配对比较。噪声倍率为 `1 + sigma * Z`；若有非正倍率，仅重新抽取该元素并记录次数。本次实际未出现需要重抽的元素。相位数组原样保留，相位噪声为 0。

识别入口直接调用 `src/mt_static/detection/detect_static_sites`，配置沿用 `configs/section_22/site_types.json`。没有改动 CWT、小波、尺度、模极大值、传播线、Lipschitz 指数或候选判据。运行前核对正演文件 SHA256 和已保存的 S0/S2 配置；0% 噪声的逐频候选矩阵必须与当前 S0/S2 完全相同，才会写入结果。已有结果目录非空时拒绝覆盖。

```powershell
$env:PYTHONIOENCODING = 'utf-8'
& 'D:\SoftwareInstallation\Anaconda\envs\mt-static-rebuild\python.exe' 'D:\MT_StaticShift_Inversion\experiments\section_22_detection\223_noise_robustness\run.py'
```

每次识别在 `results/section_22/223_noise_robustness/noise_XXpct/<arm>/rep_YYY/` 保存：完整输入及随机底样 `input.npz`，逐频剖面、小波系数、模极大值、候选矩阵和最小相邻尺度 α 诊断 `detection_diagnostics.npz`，包含四个 `alpha_j` 的全部传播线 `maxima_lines.json`，候选明细 `static_candidates.csv`，以及 `trial_summary.json`。0% 和每个非零噪声级别的第 0 次重复另存四幅中文诊断图。总目录保存 `manifest.json`、`summary.json`、`summary.csv`、`repetitions.csv` 与汇总图。

TP、FP、FN、Precision、Recall、F1 均按“频率×测点”统计。双浅部体组每次有 40 个真值阳性，其余 780 个位置为阴性；无浅部体组没有阳性真值，其 Recall/F1 留空。±500 m 区域命中和区域外候选另作空间定位诊断，不替代严格测点指标。两中心各至少在一个频率出现候选只是一项宽松的逐次描述，并非跨频最终测点分类规则。

本次结果及局限见 `results/section_22/223_noise_robustness/README.md`。本实验的视电阻率噪声与论文 2.3.1 的相位噪声反演分开。
