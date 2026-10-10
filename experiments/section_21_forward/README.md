# 论文 2.1：静态效应正演特征

本入口使用 `src/mt_static/forward/` 中分离的网格、模型、测量、正演、响应转换、绘图和流程模块，在完全相同的网格、深部结构、41 个测点和 20 个频率上，计算有、无浅部异常体的 TE、TM 四组二维物理正演。没有加入人工静态位移或随机噪声。旧脚本保留在 `reference_codes/forward/`，本程序不导入旧脚本，也不以旧结果作数值比较。

当前正式模型将左右两处浅部体均设为 **10 Ω·m 低阻体**，背景为 100 Ω·m。用户将浅部厚度从 200 m 改为 5 m，水平宽度仍为 200 m。下表同时列出代码输入坐标尺寸和网格实际表示尺寸；“宽×高”分别沿水平和垂向计算。

| 异常体 | 电阻率 | 输入坐标尺寸 | 当前网格实际尺寸 |
| --- | ---: | ---: | ---: |
| 左深部低阻体 | 10 Ω·m | 6000 m × 6000 m | 6000 m × 6100 m |
| 右深部高阻体 | 1000 Ω·m | 6000 m × 6000 m | 6000 m × 6100 m |
| 左浅部低阻体 | 10 Ω·m | 200 m × 5 m | 200 m × 5 m |
| 右浅部低阻体 | 10 Ω·m | 200 m × 5 m | 200 m × 5 m |

`section_21_summary.json` 的 `models.anomaly_regions` 保存各体的输入与实际边界、尺寸、电阻率及单元数。

仅改变浅部体时，可使用 `--reuse-unchanged-baseline`。程序会逐项核对已有结果中的频率、测点、网格节点和无浅部体模型，完全一致才复用 TE/TM 基线阻抗，并重新计算两组有浅部体响应；最后照常覆盖全部结果文件。

```powershell
& 'D:\SoftwareInstallation\Anaconda\envs\mt-static-rebuild\python.exe' 'D:\MT_StaticShift_Inversion\experiments\section_21_forward\run_forward.py'
```

只重绘现有结果，不执行物理正演：

```powershell
& 'D:\SoftwareInstallation\Anaconda\envs\mt-static-rebuild\python.exe' 'D:\MT_StaticShift_Inversion\experiments\section_21_forward\plot_forward.py'
```

新结构的结果写入 `results/section_21_forward/structured/`；此前版本的输出保留在其上级目录，不能与新结构的数据字段混用：

- `section_21_forward.npz`：网格节点、两套对数电导率模型、SimPEG 原始 `dpred`、复阻抗、视电阻率和相位。字段采用英文 `snake_case`；数组频率和测点均升序。每个频点的 `dpred` 先存全部测点的实部，再存全部测点的虚部。
- `section_21_summary.json`：模型、网格、四个异常体的输入及实际尺寸、模式定义及有无浅部体的定量差异，字段采用英文。
- `TE_视电阻率与相位拟断面.png`、`TM_视电阻率与相位拟断面.png`：参考示例采用红低蓝高的平滑拟断面，两模型使用一致色标。
- `TE_负五千米测点曲线.png` 等四张曲线图：横轴为对数周期（0.001～1000 秒），相位范围固定为 0～90°；视电阻率使用对数纵轴，并按实际数据选择下限以避免截断。图内模式标题同样使用 `TE`、`TM`。

## 模式及参考脚本注释

本二维剖面的走向垂直于网格平面。TE 为沿走向电场，使用 `Simulation2DMagneticField` 和 `Zyx`；TM 为跨走向电场，使用 `Simulation2DElectricField` 和 `Zxy`。参考目录中的 `plot_pseudosection.py` 把后一组标作“TE”，与此物理极化定义不一致。最初沿用该标签的一次运行保存在 `results/section_21_forward/diagnostic_initial_labels/`，仅作排查记录，不能作为当前结果。

参考模型注释称浅部体为“4 m × 4 m”，但其代码使用 x 范围约 302 m、z 范围 4 m；在旧的 200 m 水平、5 m 顶层网格中，每处实际占据 400 m × 5 m。用户之后指定左右浅部体均为 200 m 水平宽度、10 Ω·m，并将厚度由 200 m 改为 **5 m**。正式网格核心区水平单元为 100 m，地表至 200 m 深度的垂向单元为 5 m，使两处浅部体现在均准确占据 **2×1 个单元，200 m × 5 m**。200 m 以下节点沿用旧网格，因此深部体的实际垂向尺寸仍为 6100 m。

## 多浅部体独立场景

`configs/section_21/base_v1.json` 固定已验证的深部模型、网格版本、41 个测点和 20 个频率；`configs/section_21/scenarios/<scenario_id>.json` 仅列出浅部体和需要绘制曲线的测点。新场景必须使用新的文件名与 `scenario_id`，已有结果目录不会被覆盖。浅部体按现有网格单元中心选取；如果给定边界与网格实际边界不一致，程序会报错，不会悄悄改变几何。

已登记的 `two_low_200x5_v1` 逐项核对后直接使用现有双体正演数组。`four_low_200x5_v1` 的四个 10 Ω·m 浅部体分别以 −5、−1.5、+1.5、+5 km 为中心，每个宽 200 m、厚 5 m；前后两处位于深部异常体中心正上方，中间两处对称分布。四体场景实际进行了 TE/TM 物理正演，没有人为注入静态位移或噪声。

```powershell
$env:PYTHONIOENCODING = 'utf-8'
& 'D:\SoftwareInstallation\Anaconda\envs\mt-static-rebuild\python.exe' 'D:\MT_StaticShift_Inversion\experiments\section_21_forward\run_scenario.py' --scenario 'D:\MT_StaticShift_Inversion\configs\section_21\scenarios\four_low_200x5_v1.json'
```

场景完成后，`results/section_21_forward/scenarios/<scenario_id>/` 下有自包含的 `forward.npz`、配置哈希及模型几何记录 `manifest.json`、`figures/`。每个场景的图包括模型图、TE/TM 视电阻率与相位拟断面，以及配置所列测点的 TE/TM 曲线；四体场景还绘制 0 km 对照测点。无浅部体基线只在网格、深部模型、测点和频率数组完全相同时复用，缓存放在 `cache/forward/baselines/`。现有 `structured/` 结果不被新场景覆盖。

只重绘场景图，不重新正演：

```powershell
& 'D:\SoftwareInstallation\Anaconda\envs\mt-static-rebuild\python.exe' 'D:\MT_StaticShift_Inversion\experiments\section_21_forward\plot_scenario.py' --scenario 'D:\MT_StaticShift_Inversion\configs\section_21\scenarios\four_low_200x5_v1.json'
```

图件用于数值核对，论文定稿图仍应单独放入 `paper/figures/section_21/`。
