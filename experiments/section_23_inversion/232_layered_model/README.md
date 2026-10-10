# 论文 2.3.2 复杂层状组合模型

本目录复现旧目录 05_moderate_layered_static_data_selection 中的 moderate_layered_publication.png。旧程序只作数值基准，不作为新程序依赖。

## 模型和数据

- 0–1 km：50 Ω·m；1–8 km：500 Ω·m；8–12 km：30 Ω·m。
- 两个 10 Ω·m 椭圆目标：中心 x=±5 km、深度 5 km，水平半轴 2.2 km、垂向半轴 2.0 km。
- 八测点 x=−10、−8、−5、−2、2、5、8、10 km；静态点为 ±5 km。其正上方各有一个 4 m×4 m 浅部体，左侧 1000 Ω·m，右侧 10 Ω·m。
- TM，24 个对数间隔频率，范围 0.001–100 Hz。浅部体物理正演决定静态点视电阻率；按旧图规则使用无浅部体相位，并给两组数据加入相同随机噪声。
- 由六个普通测点的视电阻率和相位求一维共同初值，再以同一 80×64 反演网格、固定 β 和五次迭代运行四组方法。

## 运行顺序

在项目根目录的 mt-static-rebuild 环境中依次运行：

1. python experiments/section_23_inversion/232_layered_model/prepare_inputs.py
2. python experiments/section_23_inversion/232_layered_model/run.py --setup-only
3. python experiments/section_23_inversion/232_layered_model/run.py
4. python experiments/section_23_inversion/232_layered_model/plot_results.py
5. python experiments/section_23_inversion/232_layered_model/compare_legacy.py

若需要用新绘图代码对旧目录锁定的四组模型做来源明确的中文重绘，可另行运行：

python experiments/section_23_inversion/232_layered_model/plot_results.py --legacy-model-dir D:\staticjiaozheng\论文整理\05_moderate_layered_static_data_selection\moderate_layered_original_reproduced

固定输入保存于 data/synthetic/section_23/232_layered_model/，数值和诊断图保存于 results/section_23/232_layered_model/。只有全区域误差验收通过时，图件才另存于 paper/figures/section_23/；若未通过，则图名带“待核查”并保留各组数据供分析。历史比较脚本需要本机旧目录存在；其余入口不依赖旧目录。

普通点在相位保留法和 shape 方法中均使用视电阻率与相位；静态点只使用相位，shape 方法额外使用去常数对数幅值的跨频形态。不同方法的数据量不同，不按 WRMS 单独排序。

## 本次复现核查

新旧真值模型单元值完全相同；无浅部体和静态数据的最大绝对差分别约为 1.97×10⁻⁶ 和 7.93×10⁻⁶。共同一维初值与旧数组的对数电导率 RMSE 约为 0.00164。

无浅部体参考、相位保留、相位加 shape 三组恢复模型与旧数组的对数电导率 RMSE 均约为 0.001。静态全数据组为 0.403；即使用旧目录锁定的一维初值和观测重新计算，差异仍为 0.275，原因待确认。不得将直接读取旧恢复数组生成的“历史结果中文重绘”图误称为新反演结果。

将旧目录保存的静态全数据恢复模型代入当前正演算子，所得数据目标函数为 749.35556019；旧摘要为 749.35556011。旧模型与当前正演算子和观测相容，因此差异集中在反演求解路径，暂不能进一步确定具体版本或数值步骤。

已在当前 mt-static-rebuild 环境中直接调用旧目录现存的 controlled_fixed_iteration_inversion.py，使用旧初值和旧观测只读回放；得到 phi_d=791.26067167、相对旧保存模型 RMSE=0.27461，与新模块在相同输入下的诊断结果一致。因此静态全数据组的差异不是本次模块拆分导致。旧保存模型的生成环境或历史数值轨迹仍待确认。

共同一维初值的全区域 log10 电阻率 RMSE 为 0.49555，新无浅部体二维参考为 0.50512；该项验收未通过，故独立重算图件标注“待核查”，不放入最终论文图目录。目标体区域的误差从 1.49929 降为 1.29282，但这不能替代未通过的全区域指标。未为追求更接近旧图而调整物理模型、噪声、误差或反演参数。
