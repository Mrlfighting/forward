# 二维 MT 静态效应正演、识别与反演

本仓库包含论文 2.1、2.2 和 2.3.1 的程序、配置、必要的固定数值输入，以及 `external/simpeg/` 中的 SimPEG 源码。`results/` 中仅保存后续实验必须读取的少量已验证输入；运行产生的其余结果与缓存不纳入版本管理。

## 安装

在仓库根目录使用 Python 3.11。以下命令适用于 PowerShell；`python` 应指向新建环境。

```powershell
conda create -n mt-static-rebuild python=3.11 -y
conda activate mt-static-rebuild
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
$env:SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMPEG='0.25.2.dev9'
python -m pip install --no-deps -e ./external/simpeg -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m unittest discover -s tests -v
```

SimPEG 快照来自上游提交 `43b0120dd70a793e319a05b679459364b0461479`，保留其 `LICENSE`。本地快照另含 `simpeg/utils/pgi_utils.py` 中已有的延迟导入修改及根目录导入桥接文件。安装时设置版本环境变量，是因为发布仓库不附带 SimPEG 自身的 `.git` 历史；项目程序优先引用 `external/simpeg/` 的源码。不要把 `external/simpeg/build/`、`simpeg.egg-info/` 或独立 `.git/` 作为项目输入。

## 入口和固定输入

| 工作 | 入口 | 固定输入 |
| --- | --- | --- |
| 2.1 场景正演 | `experiments/section_21_forward/run_scenario.py` | `configs/section_21/` |
| 2.2.1 测点识别 | `experiments/section_22_detection/221_site_types/run_section21_forward_s2_check.py` | `results/section_21_forward/structured/section_21_forward.npz` |
| 2.2.2 位移强度 | `experiments/section_22_detection/222_shift_strength/run.py` | 上述正演文件及 S0 固定诊断 |
| 2.2.3 噪声鲁棒性 | `experiments/section_22_detection/223_noise_robustness/run.py` | 上述正演文件及 S0/S2 固定诊断 |
| 2.3.1 四静态点反演 | `experiments/section_23_inversion/231_phase_noise/run_four_static_rectangles.py` | `data/synthetic/section_23/231_phase_noise/` |
| 独立教学案例 | `docs/exemple/run_example.py` | `docs/exemple/config.json` |

从仓库根目录运行。重新正演时指定新的输出目录，避免覆盖仓库附带的固定输入：

```powershell
python experiments/section_21_forward/run_scenario.py --scenario configs/section_21/scenarios/four_low_200x5_v1.json --output-root results/section_21_forward/reproductions
python experiments/section_22_detection/221_site_types/run_forward_scenario_detection.py --scenario-id four_low_200x5_v1 --mode TM
python experiments/section_22_detection/222_shift_strength/run.py
python experiments/section_22_detection/223_noise_robustness/run.py
python experiments/section_23_inversion/231_phase_noise/run_four_static_rectangles.py --setup-only
python experiments/section_23_inversion/231_phase_noise/run.py --setup-only
python docs/exemple/run_example.py --step all
```

各实验的正式运行命令、配置和结果字段见对应 `experiments/` 下的 README。正反演耗时与机器性能有关；去掉 `--setup-only` 才运行 2.3.1 反演。2.2.1 的旧 S0–S3 回放入口 `run_site_types.py` 依赖已从当前项目清理的历史输入，仅用于代码溯源；当前数据对应的入口是上表所列程序。`compare_legacy.py` 需要另行提供原旧项目目录，不属于仓库内可独立复现的正式计算。

仓库不包含本地研究记忆文档、完整数值结果和过程缓存。已保存的固定输入保留原始 SHA256 校验；S0/S2 摘要中的绝对路径仅记录旧机器来源，实验运行时以哈希核对当前仓库文件。
