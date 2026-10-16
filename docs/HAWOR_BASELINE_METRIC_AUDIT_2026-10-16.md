# HaWoR baseline 与论文数值差距核查

日期：2026-10-16。用户询问为何当前 baseline 比论文大一个数量级。核对原论文 §4 / Table 1、作者 DexYCB evaluator，并在 `jchen57-fineturn-tl4zr` Pod CPU 上重新解码已有预测，补算 PA-MPJPE；没有重新运行视觉网络、训练或选择 checkpoint。

## 结论与实测

HaWoR 论文 Table 1 的表头简写 `MPJPE`，但 caption 和 §4 明确为 **PA-MPJPE**：逐帧对齐平移、旋转和统一尺度后衡量21点误差。论文 DexYCB All 为 **4.76 mm**。此前本项目报告的是 **未对齐的绝对 MPJPE21 70.532 mm**，两者不是同一个指标。所有值均为毫米，本次补算没有更改单位。

| 模型与集合 | 绝对 MPJPE21 mm | 仅腕平移对齐 mm | PA-MPJPE21 mm |
|---|---:|---:|---:|
| 论文 HaWoR / DexYCB test | 未在 Table 1 报告 | 未在 Table 1 报告 | 4.76 |
| 原官方权重 / 本项目 test1015 | 70.532 | 14.480 | **5.082** |
| 预选 joint best300 final / 同 test1015 | 48.194 | 11.414 | **4.689** |
| 预定终点 joint last1000 final / 同 test1015 | 47.238 | 10.742 | 4.633 |
| 原官方权重 / 本项目 val577 | 81.418 | 13.465 | 5.300 |
| 预选 joint best300 final / 同 val577 | 38.804 | 11.092 | 4.335 |

当前 baseline test PA 数值比论文4.76高 **6.756%**，没有十倍量级差距；此数值差不能当成同样本复现偏差。相同 test1015 上，best 相对官方 baseline 的 PA 降低 **7.719%**，绝对 MPJPE 降低31.67%。绝对定位与对齐后的局部姿态收益应分别报告；PA 依靠评测参考进行对齐，不能把5.082mm解释成部署时的绝对定位精度。70.532→14.480与14.480→5.082展示不同对齐条件下误差的变化，不是可相加的因果误差分解。

## 仍然不同的评测条件

- 本项目 test 是 s0 中16个预定录制、单相机 `836212060125`、raw frame_id<128、37 clips / 1015完整有效状态；不是作者默认全 s0_test 的160录制×8视角。val577也不是论文 test。
- 使用 GT joint crop、原16帧双向上下文、32帧 clip边界与padding；参考是 DexYCB 拟合 MANO / supplied joints，各分支使用自身预测 β。当前是固定相机 camera gauge，未运行完整头戴 SLAM / infiller benchmark。
- 当前所有21点均有效后才纳入，并已审计 reference FK；不能未经复核就声称作者逐帧筛选、crop、时序窗口和输入全部等同。
- 4.689 比4.76数值低不构成超过原论文的结论；需要相同完整集合、视角、输入和官方 evaluator 后才能作论文排名。
- HOT3D Table 3 的 PA4.79、W33.20与WA11.27也是不同数据／对齐协议，不能与本项目 DexYCB 绝对误差直接比较。

## 实现与可复现产物

新增 [evaluate_visual_geometry.py](../scripts/evaluate_visual_geometry.py) 的可选 `--include-pa`。默认原四项指标不变；新增PA对每帧21点进行 least-squares similarity fit，使用 float64 SVD，约束 proper rotation `det(R)=+1`。平移／旋转／尺度的合成不变性检查通过（最大误差<5e-13mm），镜像点云未被错误消去。

另用作者所引用的 FreiHAND `align_w_scale` 公式（中心化、归一化、SciPy orthogonal Procrustes、还原尺度，输入毫米）独立核对全部1015个baseline状态。与新增PA最大逐帧差 **1.389e-9mm**，通过；这1015帧的自由正交拟合均 `det(R)>0`，因此FreiHAND未限制反射与本次proper-rotation限制没有产生baseline数值差。验证入口 `code/scripts/verify_reference_alignment.py` 与 `reference_alignment_check.json` 保存远程。这是指标函数一致性，不是完整benchmark复现。

远程根：`/sciclone/data10/jchen57/Hand_Reconstruction/evaluations/hawor_paper_metric_audit_20261016/`。

- `geometry_test_pa.json`、`geometry_val_pa.json`与逐帧CSV：七分支完整重算、输入／模型资产／脚本SHA、reference FK与左右手分组。
- 新执行代码只新增评测脚本，`code/src`指向原 `visual_joint_final_report_20261016/src`（原e72f7ed执行快照），没有使用当前工作区另一个训练任务正在修改的codec。脚本SHA为 `4c1751566c7b45b82b8c1742c76ff8f8542726c9319630906b1cbc343250a774`。
- 同配对键逐帧核对原有四项指标；test 最大差 **0.00048828125mm**，val **0.000082016mm**，均<0.001mm（旧GPU与本次CPU解码的浮点差）。valid counts保持1015／577；reference FK审计通过。
- [Git小型摘要](../reports/hawor_paper_metric_audit_20261016.json)保存指标、manifest／资产／脚本SHA与输入产物fingerprint；完整预测、licensed MANO与逐帧原始产物留远程。原D40 best300／last1000身份不变，没有按PA指标重选。

来源：[HaWoR原论文 §4与Table1](https://arxiv.org/html/2501.02973v1)、[作者DexYCB HPEEvaluator](https://github.com/ThunderVVV/dex-ycb-toolkit/blob/ad97815f946f220f91fd70b1e75191d422ab85bf/dex_ycb_toolkit/hpe_eval.py)（`joint_3d *=1000`、`align_w_scale(kpt_gt,kpt_pred)`）、[FreiHAND对齐函数](https://github.com/lmb-freiburg/freihand/blob/master/eval.py)、[原联合终点评测协议](VISUAL_JOINT_FINAL_EVALUATION_2026-10-16.md)。

## 后续：AUC补算（2026-10-16）

按用户要求新增 `--include-auc`（自动启用PA），在同一Pod CPU重解码原三套保存预测，覆盖七个raw/candidate/final分支。阈值和积分沿作者HPEEvaluator引用的[FreiHAND EvalUtil](https://github.com/lmb-freiburg/freihand/blob/master/utils/eval_util.py)：0–50mm、100个含端点的等间隔阈值，逐joint判断 `error <= threshold`，PCK曲线梯形积分除以50，再对21个joint平均。计算使用全部完整有效状态；不是对逐帧MPJPE先取均值再阈值化。AUC归一化为[0,1]，下表乘100显示，越高越好；不是某个单阈值的正确率或模型置信度。

| 模型 / test1015 | 绝对 AUC % | 仅腕平移对齐 AUC % | PA-AUC % |
|---|---:|---:|---:|
| 官方HaWoR raw | 13.59279 | 71.10035 | **89.83933** |
| 预选best300 final | 33.08518 | 77.20214 | **90.62120** |
| 预定终点last1000 final | 38.42950 | 78.54020 | **90.73607** |

best同test PA-AUC提高 **0.78187个百分点**；绝对AUC提高19.49239个百分点。论文Table1 DexYCB All PA-AUC为 **90.5%**，但仍因当前单视角prefix128子集与论文完整协议不同，不能按90.62>90.5宣称超过论文。last仅终点对照，未重新选择best。

| 模型 / val577 | 绝对 AUC % | 仅腕平移对齐 AUC % | PA-AUC % |
|---|---:|---:|---:|
| 官方HaWoR raw | 5.16100 | 74.84914 | 89.40736 |
| 预选best300 final | 44.96625 | 78.16383 | 91.32986 |
| 预定终点last1000 final | 51.45567 | 79.44190 | 91.47516 |

验证：直接加载已审阅的官方EvalUtil源码，在远程对每个实际branch×alignment×all/left/right的结果及0/25/50/>50mm边界／mixed-joint用例独立核对。每个split共69项，AUC及每个PCK曲线点最大差均为0；对照作者源码常数0/50/100通过。原五项几何指标的所有分支均值与上轮CPU PA报告最大差8.882e-16mm（浮点舍入），test1015／val577计数与reference FK审计保持通过。无新视觉模型前向、训练、checkpoint调整、test筛选或遮挡分组；只新增FK／指标评测。

远程产物：`/sciclone/data10/jchen57/Hand_Reconstruction/evaluations/hawor_auc_audit_20261016/`，包括 `geometry_{test,val}_auc.json`、逐帧几何CSV、完整PCK曲线、参考源码及SHA、`reference_checks_{test,val}.json`、执行脚本和原日志。新脚本仍复用原e72f7ed的src/codec，当前工作区其它训练修改未进入本次评测。Git仅保存[小型AUC摘要](../reports/hawor_auc_audit_20261016.json)、源码与文档；完整预测及许可MANO留远程。
