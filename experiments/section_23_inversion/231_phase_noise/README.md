# 论文 2.3.1：随机频点相位噪声反演对比

## 当前四静态点矩形模型复现

用户提供的旧图由 `D:\staticjiaozheng\论文整理\04_static_site_phase_interference_downweighted\plot_static_site_phase_interference_publication.py` 绘制；其正演模型与反演入口分别是同目录的 `run_shape_advantage_rectangular_targets.py` 和 `run_static_site_random_phase_interference_static_phase_downweighted.py`。本轮将原正演数组按 SHA256 锁定后复制到 `data/synthetic/section_23/231_phase_noise/four_static_rectangles_v1/`，不修改旧程序、旧结果或 `external/simpeg/`。

背景为 100 Ω·m。深部低阻体中心 `(-5 km,-4 km)`，宽 3.2 km、厚 3.6 km、10 Ω·m；深部高阻体中心 `(+5 km,-4 km)`，宽 3.6 km、厚 4.0 km、1000 Ω·m。四个浅部体位于 −8、−5、+5、+8 km，均为 4 m×4 m，电阻率依次为 1000、10、1000、10 Ω·m。8 个测点、24 个频率与旧图完全一致。

旧数据在物理正演 `rho_a` 后，将静态场景的相位替换为无浅部体相位，然后两场景加入相同的基础随机噪声。这是旧图受控的理想静态效应设定。基础视电阻率噪声为 10%，基础相位噪声为 0.57°；有效 `rho_a` 误差为 10.198%，普通点相位误差为 0.8061°。随后仅在四个静态点的六个随机频点加入 σ=5° 相位干扰，实际 24 项 RMS 为 5.6735°；四个静态点的全部频率相位均降权至 5.0646°。shape 权重固定 0.7。

干净全数据、静态全数据、相位保留、相位加形态约束四方法的数据量分别为 384、384、288、380。相位保留法保留普通点 `rho_a+phase` 与静态点 `phase`；shape 法额外使用四个静态点各 23 个去常数幅值对数视电阻率形态分量。反演网格 80×64、活动单元 3100、初始电阻率 100 Ω·m、初始 β=`4.47×10⁻⁷`、五次迭代、`cg_maxiter=15`。不能仅用数据量不同的总 `phi_d/N` 比较方法。

配置为 `configs/section_23/phase_noise_four_static_rectangles.json`；入口为 `run_four_static_rectangles.py`，中文图由 `plot_four_static_rectangles.py` 生成。正式结果放在 `results/section_23/231_phase_noise/four_static_rectangles_v1/`。入口在目录非空时拒绝覆盖，可用 `--setup-only` 只核对数据，或通过 `--output-dir` 指定新的空目录。

## 先前双浅部体新模型结果（保留作对照）

本节读取新双浅部体场景 `results/section_21_forward/scenarios/two_low_200x5_v1/forward.npz` 的 TM 物理正演数据，按旧程序测点布局从 41 点中取 8 点：−10、−8、−5、−2、+2、+5、+8、+10 km。只有 ±5 km 为静态点；两处浅部低阻体各为 10 Ω·m、200 m×5 m。无浅部体响应是全数据参考，有浅部体响应用于其余三种方法。相位保留新浅部体的真实微小物理响应，不人为替换为无浅部体相位。

数据顺序是每个频率先列所有测点的 `rho_a`，再列所有相位。静态全数据使用八点 `rho_a + phase`；相位保留法对六个普通点使用 `rho_a + phase`，对两个静态点只使用 `phase`；shape 法在相位保留数据之外增加静态点的白化 Helmert 对数视电阻率形态。shape 算子各行之和为零，分别消除每个静态点跨频常数对数幅值，不含静态点绝对 `rho_a`。

20 个频率中，固定种子 `20260729` 选出索引 0、11、13、15、17、18。全部 8 个测点在这 6 频各自加入独立 σ=5° 的随机相位噪声；本次 48 个实际扰动的 RMS 为 4.727°。其余频点的相位、所有视电阻率均不加随机噪声。普通相位误差为 0.57°；被扰动的 48 项相位误差为 `sqrt(0.57²+5²)=5.032°`；视电阻率名义误差为 2%。这是用户确认后的**八测点共同干扰**，与旧图脚本仅在四个静态点加扰动且将其全部频率相位降权的设置不同。

反演网格沿用旧 80×64 网格，活动单元 3100 个；起始电阻率 100 Ω·m、固定初始 β=`4.47×10⁻⁷`、五次 GNCG、`cg_maxiter=15`，深度/侧向加权正则化及边界与旧代码一致。源场景保存的显示相位需减 180° 才与 SimPEG `phase` 接收器输出处于同一角度分支；入口对此逐点核验。

实验配置位于 `configs/section_23/phase_noise.json`，通用 survey、shape 和求解器模块位于 `src/mt_static/inversion/`。先在 `cache/section_23/231_phase_noise_validation/` 验证；通过后将数值结果迁至 `results/section_23/231_phase_noise/`。正式结果目录非空时，运行入口拒绝覆盖；`--setup-only` 仅检查输入而不运行反演。

```powershell
$env:PYTHONIOENCODING = 'utf-8'
& 'D:\SoftwareInstallation\Anaconda\envs\mt-static-rebuild\python.exe' 'D:\MT_StaticShift_Inversion\experiments\section_23_inversion\231_phase_noise\run.py' --setup-only
```

在空结果目录中，去掉 `--setup-only` 即运行四组正式反演；当前目录已有经验证结果，入口会拒绝覆盖。已有结果的图件可单独调用同目录的 `plot_results.py` 重绘。

正式结果包含 `input.npz`、四个恢复模型与预测的 NPZ、分项 shape 预测、`setup.json`、`summary.json` 和两幅中文图。`plot_results.py` 只从结果文件重绘，不重新执行反演。各方法的数据量分别为完整 320、相位保留 280、shape 额外 38、形态组合 318。评价以共同真模型的目标区对数电阻率 RMSE、背景误差与左右目标几何平均电阻率为主；各方法的数据拟合只在各自明示的数据子集内报告，不用不同数据量的总 `phi_d/N` 单独排序。实际数值和适用边界见结果目录 README。
