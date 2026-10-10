# 论文 2.2.2：静态位移强度识别

本实验只改变两个目标测点的频率无关视电阻率倍率 `Cs`。原始数据取自论文 2.1 已保存的无浅部体 TM 正演，目标位于 −5 km 和 +5 km；其余 39 个测点、20 个频率、地下模型、相位和噪声状态固定。施加公式为 `rho_shifted[:, target] = Cs * rho_base[:, target]`，`Delta_s = abs(log10(Cs))`。此处是**人工倍率注入**，与论文 2.1 浅部体物理正演的有／无对照不可混称。

`run.py` 从 `configs/section_22/shift_strength.json` 读取八组参数，核对现有 S0 结果记录的正演 SHA256 和 2.2.1 配置，并用 `Cs=1` 重算出完全相同的无候选矩阵后才运行其他组。识别直接调用 `src/mt_static/detection/detect_static_sites`；没有修改小波、尺度、传播线、相邻尺度 α 或阈值，也没有调用 SimPEG。若 `results/section_22/222_shift_strength/` 已含内容，入口拒绝覆盖。

```powershell
$env:PYTHONIOENCODING = 'utf-8'
& 'D:\SoftwareInstallation\Anaconda\envs\mt-static-rebuild\python.exe' 'D:\MT_StaticShift_Inversion\experiments\section_22_detection\222_shift_strength\run.py'
```

每组保存 `input.npz`（基础/施加倍率后的视电阻率、不变的相位、测点与频率）、`detection_diagnostics.npz`（空间剖面、五尺度系数、原始/已选模极大值、逐频候选及目标 α 诊断）、`maxima_lines.json`（所有传播线、淘汰原因及候选线四个 `alpha_j`）、`static_candidates.csv`、`case_summary.json` 和四幅中文诊断图。总表为 `summary.csv`、`summary.json` 和 `强度与识别指标.png`。来源、配置和哈希保存在每组摘要与总摘要。

## 指标口径和本次结果

每个非零位移组的真值为 20 个频率×2 个测点，共 40 个阳性；其他 780 个频率×测点为阴性。TP/FP/FN、Precision/Recall/F1 均按此粒度计算。`Cs=1` 为无阳性真值的阴性对照，其 Recall/F1 留空。“成功识别”仅指两个真值点各至少在一个频率被列为候选，不能当作跨频最终测点分类。`alpha_min_at_targets` 是目标候选线在所有命中频率的**最小相邻尺度 α**，并非单一回归 α；全部四个 `alpha_j` 仍在传播线 JSON 中。

| Cs | Δs | −5 km 命中频数 | +5 km 命中频数 | TP | FP | FN | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.3010 | 8 | 17 | 25 | 26 | 15 | 0.5495 |
| 0.7 | 0.1549 | 8 | 8 | 16 | 18 | 24 | 0.4324 |
| 0.8 | 0.0969 | 8 | 8 | 16 | 17 | 24 | 0.4384 |
| 1.2 | 0.0792 | 6 | 8 | 14 | 17 | 26 | 0.3944 |
| 1.5 | 0.1761 | 9 | 8 | 17 | 18 | 23 | 0.4533 |
| 2.0 | 0.3010 | 10 | 9 | 19 | 19 | 21 | 0.4872 |
| 3.0 | 0.4771 | 13 | 11 | 24 | 26 | 16 | 0.5333 |

`Cs=1` 与现有 S0 逐频候选矩阵一致，TP/FP/FN 均为 0。七个非零强度均满足“两处至少一次命中”，故本次测试点**没有围住检出／漏检边界**。最弱测试强度是 `Cs=1.2` 的 Δs≈0.0792 和 `Cs=0.8` 的 Δs≈0.0969；不能据此推断更弱位移必然检出。命中频数与 F1 的变化并不严格单调，也没有进行拟合或参数调优。

历史旧表 S0–S3 数值回放仍未判 PASS；本实验所依赖的验证限于当前正式模块、现有新 2.1 数据 S0 阴性对照及相关单元测试。
