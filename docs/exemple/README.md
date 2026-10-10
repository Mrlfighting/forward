# 二维 MT 正演与反演入门：均匀背景中的单个低阻体

本案例在 `docs/exemple/` 中独立完成**建模 → 网格剖分 → TM 正演 → 正演曲线 → 视电阻率与相位联合反演 → 反演图和曲线拟合**。全部案例逻辑位于本目录，可把整个目录复制到其他位置使用；运行时只需安装 `requirements.txt` 中的 Python 依赖。

## 1. 模型与观测设置

所有可改参数集中在 [`config.json`](config.json)。示例只有一个地质异常体，不加入浅部静态体、人工 static shift 或随机噪声。

| 项目 | 设置 |
| --- | --- |
| 背景 | 均匀 100 Ω·m |
| 低阻体 | 10 Ω·m；水平 −1.5～+1.5 km；埋深 1.0～2.5 km |
| 测点 | −4～+4 km，每 1 km 一点，共 9 点 |
| 频率 | 0.001～320 Hz，对数等距 20 点；对应周期 0.003125～1000 秒 |
| 模式 | 项目约定的 TM：`Zxy=Ex/Hy`，`Simulation2DElectricField` |
| 正演网格 | 中心水平单元 250 m；地表至 3 km 的垂向单元 25 m；向深部扩展至 1120 km |
| 反演网格 | 中心水平单元 500 m；地表至 3 km 的垂向单元 50 m；同样扩展深边界，反演活动域限于 ±8 km、深 8 km |
| 反演观测 | 所有测点均使用 `rho_a + phase`；无噪声，误差权重为视电阻率 5%、相位 2° |

正演与反演使用**不同网格**，避免直接在完全相同离散模型上回放。程序逐项检查低阻体在两个网格中的实际边界，不允许静默地把异常体尺寸舍入。

## 2. 环境和最短运行方法

先创建独立环境并安装依赖。下面的示例使用 PowerShell 和清华镜像；已有包含这些依赖的环境也可以直接使用。

```powershell
conda create -n mt-static-example python=3.11 -y
conda activate mt-static-example
cd 'D:\MT_StaticShift_Inversion\docs\exemple'
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python run_example.py --step all
```

本机核验使用已安装 SimPEG 的 `mt-static-rebuild` 环境。数值文件、机器可读摘要和四张中文图会写入 `output/`。入口也支持 `--output-dir <新目录>`；所有输入路径根据脚本自身定位。安装后的 `simpeg` 从 Python 环境直接导入，案例代码不添加项目其他目录到搜索路径。当前测试版本为 SimPEG 0.25.2 开发版；`requirements.txt` 接受 0.25 系列的正式版本。

## 3. 按步骤学习和运行

### 第一步：建模与网格

```powershell
python run_example.py --step model
```

阅读 [`grid_model.py`](grid_model.py)：`build_mesh()` 分别创建正演细网格和反演粗网格；`build_true_model()` 先给所有地下单元赋值 `ln(1/100)`，再给矩形体内单元赋值 `ln(1/10)`；`active_cells()` 指定哪些反演网格单元可以更新。对数电导率 `m=ln σ=ln(1/ρ)` 保证经过 `ExpMap` 后电导率始终为正。此步生成 `output/model.npz`、`output/summary.json` 和 [`01_模型与网格.png`](output/01_模型与网格.png)。图中同时显示测点、真模型、两种网格和异常体真实边界。

### 第二步：二维 MT 正演和曲线

```powershell
python run_example.py --step forward
```

阅读 [`forward.py`](forward.py)：对每个频率，接收器按“所有测点的阻抗实部、所有测点的阻抗虚部”排列。SimPEG 计算复阻抗 `Zxy` 后，程序按下式求视电阻率：

\[
\rho_a=\frac{|Z_{xy}|^2}{\mu_0 2\pi f},\qquad \mu_0=4\pi\times10^{-7}\,\mathrm{H/m}.
\]

`phase_raw_deg` 是与 SimPEG `phase` 接收器一致的原始角度分支，供反演使用；`phase_display_deg` 是图上显示的 `arg(-Zxy) mod 180°`。程序另用 SimPEG 原生视电阻率与相位接收器计算同一真模型，核对公式结果和反演观测一致，避免 180° 相位分支错误。

此步分别正演均匀背景和含低阻体模型，生成 `output/forward.npz` 与 [`02_正演曲线.png`](output/02_正演曲线.png)。曲线展示低阻体正上方的 0 km 测点：横轴是周期 `T=1/f`，左侧为高频；上排为视电阻率，下排为相位。按照参考图固定绘图坐标：周期 `10⁻³～10³` 秒，视电阻率 `10¹～10⁴` Ω·m，相位 `0～90°`。蓝色空心标记及虚线为均匀背景，红色实线为含低阻体；其他测点的完整数值同样保存在 NPZ 中。均匀模型数值曲线接近但不一定严格等于 100 Ω·m，这是有限网格正演的离散误差。

### 第三步：反演及核对

```powershell
python run_example.py --step inversion
```

阅读 [`inverse.py`](inverse.py)：把正演保存的 `rho_a+phase_raw` 当作合成观测；无随机噪声，但给视电阻率 5%、相位 2° 的误差标准差，作为最小二乘目标函数的权重。`build_simulation()` 建立活动单元至全网格的对数电导率映射；`build_regularization()` 设置参考模型、方向长度尺度及边缘权重；`solve_inverse()` 从均匀 100 Ω·m 起始，使用 SimPEG 的 `ProjectedGNCG`、固定初始 β、β 冷却和灵敏度权重完成反演。所有数值参数集中在 [`config.json`](config.json)，无须读取本目录之外的项目代码。

此步生成 `output/inversion.npz`、[`03_反演模型.png`](output/03_反演模型.png) 和 [`04_反演曲线拟合.png`](output/04_反演曲线拟合.png)。前一图用相同色标对比真实和恢复的电阻率，虚线是真异常体范围；后一图用与正演图一致的坐标范围比较 0 km 测点的正演观测与反演预测。`output/summary.json` 保存均匀初始模型与恢复模型在同一范围内的 `log10(ρ)` RMSE，以及低阻体范围内的几何平均电阻率。当前最多 24 次迭代；需要看恢复模型和拟合曲线，不能仅凭迭代结束字样断言反演充分收敛。

本机本次宽频结果中，初始均匀模型的评价区 `log10(ρ)` RMSE 为 0.419，恢复模型为 0.205；目标范围的几何平均电阻率由 100 Ω·m 降至约 16.3 Ω·m（真值 10 Ω·m）。低阻位置恢复清楚，但边界被平滑，不能把恢复图理解为异常体真实形状的唯一解。没有注入随机噪声，因此很小的数据残差也不代表实测数据可达到同样精度。

从项目外工作目录重新运行完整案例后，按同名数组比较**抽离前与抽离后**的结果：恢复模型对恢复模型、预测对预测、观测对观测，三组最大绝对差均为 0。这只说明代码抽离没有改变数值行为。**反演预测与正演观测并不相同**：本次 360 项数据中，没有任何一项严格相等；视电阻率最大绝对残差约 1.912 Ω·m，相位最大绝对残差约 0.869°。

## 4. 文件职责与结果读取

| 文件 | 作用 |
| --- | --- |
| [`config.json`](config.json) | 模型、网格、测点、频率及误差/反演参数 |
| [`requirements.txt`](requirements.txt) | 独立安装所需的 Python 包 |
| [`grid_model.py`](grid_model.py) | 网格、真实电阻率模型、活动域和几何核对 |
| [`forward.py`](forward.py) | TM 阻抗 survey、物理正演、`rho_a`/相位公式和接收器核对 |
| [`inverse.py`](inverse.py) | SimPEG 模拟器、正则化、优化、观测误差和模型评价 |
| [`plotting.py`](plotting.py) | 四张中文图；不执行正演或反演 |
| [`run_example.py`](run_example.py) | 三个阶段的入口、文件保存及配置哈希核对 |
| `output/model.npz` | 网格节点、真模型、异常体单元、测点、活动域 |
| `output/forward.npz` | 频率、复阻抗、均匀与异常模型的视电阻率和相位、反演观测 |
| `output/inversion.npz` | 恢复模型、观测、预测、误差标准差及反演网格 |
| `output/summary.json` | 实际网格尺寸、接收器核对、数据量和评价指标 |

可以用 NumPy 查看数值：

```powershell
python -c "import numpy as np; d=np.load('output/forward.npz'); print(d.files); print(d['frequencies_hz'])"
```

修改 `config.json` 后请从 `--step model` 重新运行，或直接运行 `--step all`。每个阶段都检查配置文件 SHA256；如果上一阶段结果来自旧配置，后续阶段会明确报错，以免混用不一致的模型和数据。

## 5. 适用范围

这是帮助读者看清 MT 正反演数据流的**单目标、无噪声、TM 合成例子**，没有静态效应识别、校正、shape 约束或 L4 实测数据。它不用于证明论文各方法的优劣，也不代替正式实验的网格收敛、噪声和不确定性分析。
