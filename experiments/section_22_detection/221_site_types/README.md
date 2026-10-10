# 论文 2.2.1：基础静态测点识别

正式识别计算位于 `src/mt_static/detection/`，参数位于 `configs/section_22/site_types.json`。当前结果入口 `run_section21_forward_s2_check.py` 读取新 2.1 已保存的 TM 正演数组，将无浅部体和双浅部体结果分别写入 `results/section_22/221_site_types/table2_s0_no_static_negative/` 与 `table2_s2_multiple_discrete_static_sites/`。`run_site_types.py` 是旧表 2 的历史回放入口；其 `data/synthetic/section_22/221_site_types/` 输入副本已按用户要求删除，因此现在不能直接运行。两个入口均不运行 SimPEG 正演。旧识别摘要副本仍在 `data/reference/section_22/221_site_types/`。

本项目 S0–S3 配置：每个频率取 `ln(rho_a)` 空间剖面，线性插值至 512 点；采用一阶高斯导数反对称小波、`1/√a` 尺度归一化、`reflect` 边界和五个二进尺度 `2,4,8,16,32`。原文对单条空间曲线的传播线规则为：从最大尺度向最小尺度搜索同符号局部模极大值，同位置优先；没有同位置节点时，在邻近极大值限定区间内选择**模值差最小**的节点，而非空间距离最近者；无法传播的线剔除。本实验要求传播线贯穿五尺度，保存四个相邻尺度 `alpha_j=log2(|W_{j+1}|/|W_j|)`；用户指定至少一对 `alpha_j<0` 即列为逐频候选。单个回归斜率只是可选扩展，不参与当前判别。

于生宝等（2017）原文没有固定要求对 `rho_a` 取对数、使用哪种母小波或边界方式，也未规定五尺度线的单一 α 汇总。本项目将这些选择明确放在配置中；`signal_transform=raw` 可运行原始视电阻率剖面，S0–S3 则按本项目论文定义选择 `log`。当前只实现配置中的一阶高斯导数 CWT。

于生宝等（2017）第二章没有给 MT 多频的最终测点分类规则，因此本入口**不把逐频候选并成最终静态测点**。跨频出现次数与最近测点映射仅为诊断结果；不能将新结果的逐频指标与旧表 2 的测点级 Precision、Recall、F1 当作同一指标比较。该文第三章的阈值属于校正，不纳入本识别过程。

当前结果的 `npz` 含每频率空间剖面、小波系数、逐尺度模极大值及映射测点出现次数。完整传播线、四个 `alpha_j` 及拒绝原因保存在 JSON；CSV 列出逐频候选。所有图均为诊断图，尚非论文定稿图。

## 新 2.1 正演数据的 S0/S2 结果

`run_section21_forward_s2_check.py` 只读取 `results/section_21_forward/structured/section_21_forward.npz` 中已保存的 TM 有／无浅部体视电阻率。其 41 测点、20 频、两处 200 m×5 m 低阻体与旧表 2 的 S2 不同。按用户要求，结果和图件分别写入 `results/section_22/221_site_types/table2_s0_no_static_negative/` 与 `table2_s2_multiple_discrete_static_sites/`。此前旧回放结果、首次独立检查副本和 S1/S3 结果已经从该结果目录清理，仅保留最新 S0/S2。

```powershell
& 'D:\SoftwareInstallation\Anaconda\envs\mt-static-rebuild\python.exe' 'D:\MT_StaticShift_Inversion\experiments\section_22_detection\221_site_types\run_section21_forward_s2_check.py'
```

根结果目录的 `comparison.json` 只比较新 2.1 数据的 S0/S2；两组各自还有 `detection_summary.json`、`detection_diagnostics.npz`、`maxima_lines.json`、`static_candidates.csv` 和四幅中文诊断图。结果只表达逐频候选，不生成跨频最终测点判决。

这组已知真值、无噪声合成对照中，无浅部体时完全没有候选，有浅部体时在 −5 km 与 +5 km 附近反复出现候选，因此支持识别静态效应的有无及所在区域。候选落在目标两侧并映射至中心及邻站，反映当前空间定位分辨率；不应将其写成区域识别失败，也不凭此增加未定义的跨频测点分类规则。

## 独立正演场景的识别

`run_forward_scenario_detection.py` 读取指定场景的 `forward.npz`，沿用上述基础识别配置，分别处理有、无浅部体的视电阻率。正演源文件的 SHA256、逐频候选、传播线与诊断图写入 `results/section_22/221_site_types/scenarios/<scenario_id>/<mode>/`。默认模式是 TM；该入口不运行 SimPEG，也不设置跨频最终测点分类阈值。

```powershell
$env:PYTHONIOENCODING = 'utf-8'
& 'D:\SoftwareInstallation\Anaconda\envs\mt-static-rebuild\python.exe' 'D:\MT_StaticShift_Inversion\experiments\section_22_detection\221_site_types\run_forward_scenario_detection.py' --scenario-id four_low_200x5_v1 --mode TM
```

四体场景无浅部体对照在 20 个频率均无候选。有浅部体时，−5、−1.5、+1.5、+5 km 中心测点分别在 7、5、7、7 个频率出现候选，目标两侧邻站亦反复出现候选；另有 +9.5 km 一次离群候选。因此当前结果支持识别四处异常附近的位置，但**不能据此宣称唯一确定四个中心测点或给出最终测点级 Precision/Recall/F1**。详细出现次数见 `scenarios/four_low_200x5_v1/TM/comparison.json`。
