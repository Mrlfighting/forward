# 理论模拟：识别结果驱动的反演

本实验不修改原 2.3.1/2.3.2 使用已知静态点位置的反演结果。它读取相同的八测点合成观测，先调用正式 `mt_static.detection.detect_static_sites`，再以至少 20% 频率检出的测点构造相位保留及 shape 反演数据。检测使用的视电阻率和反演使用的观测来自同一份固定输入；真值测点索引只在检测完成后用于评价。

运行入口为 `run.py`，结果保存在 `results/section_23/detection_driven/`。从项目根目录运行：

```powershell
& 'D:\SoftwareInstallation\Anaconda\envs\mt-static-rebuild\python.exe' -B 'experiments/section_23_inversion/detection_driven/run.py' --detect-only
& 'D:\SoftwareInstallation\Anaconda\envs\mt-static-rebuild\python.exe' -B 'experiments/section_23_inversion/detection_driven/run.py'
& 'D:\SoftwareInstallation\Anaconda\envs\mt-static-rebuild\python.exe' -B 'experiments/section_23_inversion/detection_driven/plot.py'
& 'D:\SoftwareInstallation\Anaconda\envs\mt-static-rebuild\python.exe' -B 'experiments/section_23_inversion/detection_driven/report.py'
```

固定规则：20 频至少 4 票，24 频至少 5 票；不改变 CWT、尺度、小波或 α 判据。每个场景保存来源哈希、逐频候选矩阵、空间剖面、CWT 系数、模极大值、传播线及逐点评价。若没有测点达到票数门槛，相位保留会与全数据完全相同，而 shape 没有候选点可用；程序记录无法进入两种候选点反演，不用已知真值回填。全数据法不依赖识别掩码，直接引用原实验已经完成的结果与 SHA256。

若有候选，直接复用原场景的观测、逐项误差、二维网格、正则化及迭代设置，仅替换反演掩码。层状模型若有候选，必须先用检测为普通点的测点重算共同一维初值；不能沿用旧真值掩码产生的初值。shape 仍为 GLS 白化后去常数对数幅值。评价时同时报告识别 TP/FP/FN 和恢复模型、共同数据子集拟合，不用不同数据量方法的 WRMS 单独排序。

四点模型原始数据的随机相位噪声和逐项误差曾依据已知真值位置生成，本实验保留这些固定观测与误差以便与原结果对照，因此它验证的是**识别掩码进入反演后的影响**，不是完全未知噪声位置的全盲模拟。任何无检出或误检都是本次固定规则下的有效结论，不调整检测参数或静态体以追求成功反演。

## 固定规则的实际结果

| 场景 | 真值测点（km） | 检出测点（km） | 结论 |
| --- | --- | --- | --- |
| 2.3.1 双点 | −5、+5 | 无 | 20 频门槛 4 票，最高只有 3 票；不进入候选点反演 |
| 2.3.1 四点 | −8、−5、+5、+8 | −10、+8、+10 | 24 频门槛 5 票；TP=1、FP=2、FN=3，F1=0.286 |
| 2.3.2 层状 | −5、+5 | 无 | 24 频门槛 5 票，八点均为 0 票；不进入候选点反演 |

四点场景确实用检出的三个测点重新构造数据并完成两组反演。目标区 `log10` 电阻率 RMSE：已知真值掩码的相位保留/shape 为 0.211350/0.193098，检测掩码为 0.363588/0.344820。不能把检测掩码结果写成原论文的已知位置反演复现；三场景也不能宣称已通过自动识别到反演的完整流程。详细数值见 `results/section_23/detection_driven/summary.csv` 和各场景 `summary.json`。
