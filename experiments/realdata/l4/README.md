# L4 实测剖面静态效应识别与反演

本目录只运行 L4 实测 XY 分量。原始 CSV 原样复制到 `data/realdata/l4/raw/`；旧 `D:\staticjiaozheng\论文整理\realdata`、L11 和合成实验均不作为运行时依赖。

## 数据和判定

- 15 个测点按文件名数值顺序排列，间距 500 m；保留 0.00114–320.0001 Hz 的 73 个原始频点。
- 数据准备沿用旧 L4：`MT_Rxy`、`MT_Pxy`，相位先折叠到 0–90°，进入 SimPEG 的相位减去 180°；不对频率插值。
- 逐频识别直接调用 `mt_static.detection.detect_static_sites`，参数复制自 `configs/section_22/site_types.json`。它只给逐频候选，并无跨频最终测点分类。
- S01、S05、S14、S15 是用户依据逐频结果人工确认的反演掩码；候选出现频数分别为 22、29、30、43。端点候选不代表静态效应已被实测证实。

三组反演为全点视电阻率＋相位、普通点完整数据且四候选点仅相位、在第二组基础上给候选点增加幅值不变的对数视电阻率形态。数据项分别为 2190、1898、2186。网格、误差、正则化、GLS shape、β 估计、迭代和边界均沿用旧 L4 正式实现；本次没有复用旧 L4 的 Ricker 小波识别器。

## 运行

使用项目的 `mt-static-rebuild` 环境，从仓库根目录执行：

```powershell
& 'D:\SoftwareInstallation\Anaconda\envs\mt-static-rebuild\python.exe' -B 'experiments/realdata/l4/run.py' --check-only
& 'D:\SoftwareInstallation\Anaconda\envs\mt-static-rebuild\python.exe' -B 'experiments/realdata/l4/run.py'
& 'D:\SoftwareInstallation\Anaconda\envs\mt-static-rebuild\python.exe' -B 'experiments/realdata/l4/report.py'
& 'D:\SoftwareInstallation\Anaconda\envs\mt-static-rebuild\python.exe' -B 'experiments/realdata/l4/plot.py'
```

为迁移核对，可对准备数组执行 `run.py --check-only --legacy-prepared-dir 'D:\staticjiaozheng\论文整理\realdata\MTDataTest\prepared_data'`，再用 `compare_legacy.py --legacy-script 'D:\staticjiaozheng\论文整理\realdata\MTDataTest\run_mt_realdata_inversion.py'` 核对同模型预测。正式运行无需旧目录。

`results/realdata/l4/identification/` 保存小波系数、模极大值、传播线及逐频候选；`inversion/` 保存观测、误差、网格、模型、预测和残差；`figures/` 保存中文识别图、三方法模型图与 15 个测点拟合图。不同方法的数据项数量不同，不用全局 WRMS 单独判断模型优劣；可参考 `common_data_fit.csv` 中共同数据子集的拟合，并结合模型和地质证据解释。

## 本次核对与结果

- 15 份复制的原始 CSV 与旧 L4 目录逐文件 SHA256 一致；8 类数据准备数组与旧保存数组逐项相同。
- 旧 L4 正式代码与新代码在同一起始模型上的网格、三组观测与标准差、形态算子及预测数据的最大绝对差均为 0。形态算子对跨频常数对数幅值的残差为 `3.55e-15`。
- 全数据、相位保留、形态约束三组均达到预设的 20 次迭代上限。共同的全测点相位 WRMS 分别为 0.904、1.029、0.977；共同的普通点视电阻率 WRMS 分别为 1.596、1.708、1.883。这些指标仅描述各自拟合，不能单独确定地下模型的真实性或证明 shape 优于相位保留法。
- 候选点视电阻率的绝对幅值只参与全数据组；另外两组的候选点视电阻率曲线和残差仅作诊断，不属于目标函数。四点掩码与旧正式五点掩码不同，因此不要求恢复模型或旧图一致。
- 本次实测数据缺少已知真值，结果状态记为 `completed_needs_interpretation`。端点 S01、S15 的候选需要结合数据质量、维性和地质信息进一步判断。
