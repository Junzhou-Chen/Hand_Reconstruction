# 联合微调完成后的模型与 HaWoR 对照

记录日期：2026-10-16 UTC。数据准备、RGB 推理、MANO 几何解码和数值汇总均在 SciClone 的 `jchen57-fineturn-tl4zr` Pod 执行；本机只维护源码、文档与小型摘要。

**独立录制 test 已完成：预先由训练 fixed8 选出的 best（step300）将腕位置误差从官方预训练 HaWoR 的 76.34 mm 降至 46.19 mm（下降39.49%），将绝对 MPJPE21 从 70.53 mm 降至 48.19 mm（下降31.67%）。** 同一组16录制／1,015状态上仍有27.68%的状态，最终 MPJPE 比 baseline 大超过1mm；best 的最后一轮 TWM 也使其输入候选的绝对 MPJPE 回升。方法有可测收益，同时存在局部误改与不确定性低估。下文分别报告完整 validation 和独立 test，保留预定终点 last1000 对照，不根据 test 重新挑权重。

## 1. 权重来源与选择规则

本轮训练于 **2026-10-16 03:49:16 UTC／纽约 10-15 23:49:16** 正常完成 **1,000 次 optimizer update**。训练使用源码 `eb8d0b9`，双卡每卡 8 段、每段32帧；从原每卡4段的第65步保存点继续至原总预算，并非另加1,000步。HaWoR／TWM／视觉插件目标学习率分别为 `1e-6 / 1e-5 / 1e-4`，两个主干均全量参与低学习率适配；TWM 初始化来自已完成10,000步的 D37 best。视觉最终 current NLL 与原四库 motion replay 保持各自原目标，本轮没有替换为 D38 讨论中的新 loss。

| 对照 | 使用权重 | 角色 |
|---|---|---|
| 官方 HaWoR baseline | 原始官方预训练 HaWoR | 独立加载，直接评估其 `raw` 输出 |
| best.raw | 联合微调 step300 内的 HaWoR | 主干适配后的诊断输出 |
| best.final | 同一个 step300 的 HaWoR、插件与 TWM | 预先由原 fixed8 validation 选定的正式 best |
| last.raw | 联合微调 step1000 内的 HaWoR | 最后权重的视觉侧诊断 |
| last.final | 同一个 step1000 的完整联合模型 | 预先指定的训练终点对照 |

best 是 **step300**，训练时原 fixed8 几何 selection score 为 `0.28206393683240527`。其保存时双 rank 模型 hash 一致：`b01749f60c92943f51440d454fb2046de116d2be63a966e9891286fbca713e3a`。last 为 **step1000**，模型 hash 前缀 `b2125098`，完整身份由评测 provenance 保存。

不会根据本轮 test 结果重新挑选或替换 best；last 即使某项测试更优，也仍按终点对照报告。这里的微调 HaWoR raw 与完整模型共享联合训练，**不是另一个独立训练预算的 HaWoR-only 微调实验**。`use_adapter=False` 仍会经过一轮 TWM，也不能将其称为官方 HaWoR baseline。

## 2. 数据、输入与配对约定

### 已完成的完整 validation

复用训练时固定的源池：

`/sciclone/data10/jchen57/Hand_Reconstruction/prepared/visual_feedback_dex_joint_sources_v2_20261016/source_manifest.json`

其 manifest SHA256 为 `67462dced45615c9e0ea8d761da58094d03f8b8cadc47370c06f126751259d3e`。本次完整评估 **24 clips／577 个有效完整状态／8 个录制**，来自 subject01／02 的官方 s0 val 录制。577 个状态对应8,655个有效手指局部旋转。它包含训练时用于模型选择的固定子集，因此仍是 validation，不称为未见过的独立 test。

原训练 fixed8 从24个 clips 中均匀取索引 **`[0, 3, 7, 10, 13, 16, 20, 23]`**，共210个状态。当前对所有24段只做一遍推理，再从结果中精确重算该固定子集。

### 已完成的独立 test

使用已下载的 DexYCB 原始 subject TAR，独立的 eval-only 脚本先读取已审计的100录制 metadata index，再固定 subject03–10 每人前2条官方 s0 test 录制，即 `sorted sequence_index % 5 == 4`。不根据模型预测、有效帧数量或测试误差替换录制。

每条取 **原始 `0 <= frame_id <128` 的前缀**，不是向后补足128个有效帧；原始序列短于128时以序列实际长度为限。最终实际得到 **16个录制／1,015个有效完整状态／37 clips**，其中 **9个左手、7个右手录制**，不是手帧数；拒绝录制为0。所有16个录制在任何 test label 解析前写入 `selection.json`，没有根据模型预测或有效帧数量补选。现有训练 preparation／loader 继续拒绝 test，新增 manifest 明确 `setup=s0_test_eval_only`、`split=test`、`training_allowed=false`。manifest SHA、逐文件 SHA、raw-prefix过滤与左右手分组细表保存于[最终小型报告](../reports/visual_joint_final_comparison_20261016.json)及其引用的远程原报告。

这16个 test 录制与所有已准备 train／val 录制不重叠。视觉联合训练只使用 subject01／02 的 RGB；但 DexYCB s0 是按录制划分，subject03–10 的其他录制已经在 D37 纯轨迹 s0 train 中使用，不能将这组 test 称为所有训练阶段均完全未见的主体测试。

### 所有分支共享的输入与时间策略

- 同一个固定 color camera `836212060125`，以其坐标系作为明确命名的 world gauge；`camera_to_world=I`，位置单位米。
- 使用参考21个 joints 投影得到 GT bounding box，再调用官方 HaWoR crop、square expansion 与归一化。保留真实 K，官方模型单 focal 使用 `mean(fx,fy)`；不输入 depth。
- 时间为 `frame_id /30` 的标称相对时间。按 frame-id gap 分段，不能让原16帧 HaWoR 块跨越缺帧。
- `clip_length=32`，每段独立 reset feedback history；保持官方16帧时间块及其双向上下文。尾部图像仅为前端 padding 重复，所有 padding 不计指标或历史提交。
- baseline／best／last 使用相同 source manifest、逐帧标识、crop、有效 mask、精度与 clip 边界；配对键为 `(recording, clip_index, reference_index)`，同时核对 frame_id、handedness、timestamp、target mask 和 raw availability。
- 参考状态与参考 β 只用于评测目标及上述公开的 GT crop；不会作为 frontend／TWM 的额外输入。TWM 仍只读数值状态与自身已完成历史。

## 3. 指标及 MANO 几何约定

所有误差越低越好。这里的 `wrist` 是解剖腕 J0，**不是 MANO `transl`**。

| 指标 | 定义 |
|---|---|
| Wrist mm | 预测与参考 J0 的三维欧氏距离 |
| Root ° | 全局 wrist/root rotation 的 SO(3) geodesic angle |
| Finger ° | 15个 parent-local finger rotations 的 SO(3) geodesic angle，按实际有效旋转数汇总 |
| MPJPE21 mm | DexYCB 对应21关节点的绝对位置误差，先每帧平均21点，再平均有效帧 |
| Wrist-aligned MPJPE21 mm | 分别减去预测／参考自己的 J0，只消除腕平移；不旋转、不缩放、不做 PA |
| MPVPE778 mm | MANO对应778顶点的绝对位置误差，参考 mesh 来自拟合 MANO，不是独立扫描 |

几何评测实际执行 MANO FK，不能从 rotation°或 wrist mm 换算出 MPJPE／MPVPE。prediction 使用**每个分支自身的预测 β**及官方 HaWoR 左手 shape 约定 `left_shapedirs_fix=True`。reference 使用 source calibration β 及 Dex decoder 约定 `left_shapedirs_fix=False`。二者独立解码，避免错误混用左手 shape basis。reference codec 从保存的99D状态和 source β 解码后，必须与原提供的21 joints 做逐帧对照，最大误差不超过 **1 mm** 才可纳入几何报告。

MANO16 加入 fingertips 后重排为 Dex21：tips为 `[745,317,444(right)/445(left),556,673]`，顺序为 `[0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]`。编码中的 absolute rotations 已含 hand mean，不再重复加入。

## 4. 完整 validation 结果

以下是同一 **577帧** 的配对结果，不是历史训练 batch 指标，也不是将多套集合的数值拼在一起。

| 输出 | Wrist mm | Root ° | Finger ° | MPJPE21 mm | Wrist-aligned MPJPE21 mm | MPVPE778 mm |
|---|---:|---:|---:|---:|---:|---:|
| 官方预训练 HaWoR raw | 85.2658 | 9.23988 | 8.67432 | 81.41795 | 13.46537 | 81.64146 |
| **best300.final** | **38.26487** | **7.75033** | **7.10760** | **38.80429** | **11.09192** | **38.73413** |
| last1000.final | 33.64612 | 7.26943 | 7.27604 | 34.12854 | 10.44994 | 34.05249 |

相对官方原始 HaWoR，预选 best 的腕误差下降 **55.12%**、绝对 MPJPE21 下降 **52.34%**、腕对齐 MPJPE21 下降 **17.63%**。last 的对应下降为 **60.54%／58.08%／22.39%**。last 的手指旋转误差比 best 稍高，不能说所有指标随训练单调改善。

这组差距主要体现在绝对位置。腕对齐的改善幅度小于绝对 MPJPE 的改善，说明不能将超过50%的绝对关节误差下降解释为同等幅度的手指局部结构改善。

### 完整联合模型相对其自身视觉主干

| 输出 | Wrist mm | MPJPE21 mm | Wrist-aligned MPJPE21 mm |
|---|---:|---:|---:|
| best300.raw | 56.0941 | 53.2572 | 11.00647 |
| best300.final | 38.26487 | 38.80429 | 11.09192 |
| last1000.raw | 47.0102 | 44.09795 | 11.91288 |
| last1000.final | 33.64612 | 34.12854 | 10.44994 |

best 的最终腕与绝对关节误差进一步低于同 checkpoint 的 raw；其腕对齐 MPJPE21 却从 **11.00647→11.09192 mm**，略退化 **0.776%**。因此，best 的完整校正并未在每一种局部几何指标上优于自身微调 HaWoR。该差分同时包含插件、TWM、帧内两轮与跨帧历史的联合效果，不能单独归因为某一个模块；还未运行等预算的独立 HaWoR-only 微调控制。

### 配对不确定性范围

按完整 recording 进行 **2,000次、seed20261016、95% percentile bootstrap**，每次有放回抽取8个录制。选中的录制贡献逐帧配对误差差值的总和与有效计数，再 pooled division；不是把577个相关视频帧独立重采样。正值表示 `官方 baseline误差 − final误差`，单位为 mm。

| 对照 | 误差下降均值 mm | 95% recording-bootstrap 区间 mm |
|---|---:|---:|
| best300.final vs 官方 baseline：wrist | 47.00094 | [15.6441, 69.7207] |
| best300.final vs 官方 baseline：MPJPE21 | 42.61366 | [10.29977, 66.50035] |
| last1000.final vs 官方 baseline：MPJPE21 | 47.2894 | [16.4366, 69.2921] |

这些区间只反映单次训练、当前有限8录制的重采样变化。val仅两名主体，recording 级抽样不能消除共同主体或采集条件相关性；不据此宣称跨训练seed稳定、总体统计显著或普遍因果优势。

### 旧 fixed8 的复现

新入口精确复现原 fixed8 的官方 baseline wrist **88.249596 mm**；best step300 重算 fixed8 geometry score 与训练保存值之差约 **1e-8**。这支持评测入口、checkpoint 与训练 fixedval 口径一致。完整24段的 baseline 为85.2658mm是集合扩大后的结果，与 fixed8 的88.249596mm不矛盾。

## 5. 独立 test 结果

独立 test 的 source preparation、三套权重 RGB 推理、MANO FK 与配对汇总已全部完成；driver 完成时间为 **2026-10-16 04:25:29.195702 UTC／纽约 10-16 00:25:29**。所有分支覆盖同一 **1,015个有效状态／37 clips／16录制**。独立几何评测中 reference FK 对原21 joints 的全体最大误差为 **0.00024620757699 mm**，通过1mm门槛；此项是拟合标注与 FK 的实现一致性检查，不代表独立传感器精度。

| 输出 | Wrist mm | Root ° | Finger ° | MPJPE21 mm | Wrist-aligned MPJPE21 mm | MPVPE778 mm |
|---|---:|---:|---:|---:|---:|---:|
| 官方预训练 HaWoR raw | 76.33502 | 9.84863 | 8.92100 | 70.53197 | 14.48009 | 70.86217 |
| **best300.final（预选模型）** | **46.18669** | **8.55237** | **8.40653** | **48.19357** | **11.41427** | **48.04749** |
| last1000.final（预定终点） | 45.09318 | 8.19426 | 8.43046 | 47.23770 | 10.74194 | 47.06342 |

best 相对官方 baseline 的腕／绝对 MPJPE21／腕对齐 MPJPE21 下降分别为 **39.49%／31.67%／21.17%**。last 的对应下降为 **40.93%／33.03%／25.82%**。test 提升幅度小于完整 validation，不能将 validation 的55%腕误差下降当作独立 test 的收益。

### 配对差值与录制间差异

下表使用与 validation 相同的2,000次 recording-cluster bootstrap，但每次有放回抽取 **16个 test录制**，seed与 pooled sum/count规则保持不变。正值仍为官方 baseline 减去 final 的误差下降。

| 对照 | 误差下降均值 mm | 95% recording-bootstrap 区间 mm |
|---|---:|---:|
| best300.final vs 官方 baseline：wrist | 30.14833 | [13.06343, 45.86822] |
| best300.final vs 官方 baseline：MPJPE21 | 22.33839 | [3.41530, 39.55497] |
| last1000.final vs 官方 baseline：wrist | 31.24184 | [7.53819, 50.50571] |
| last1000.final vs 官方 baseline：MPJPE21 | 23.29427 | [-2.27698, 43.76394] |

last 的 MPJPE 点均值略低于 best，但其相对官方 baseline 的该区间跨0；较好的点均值不等于跨录制更稳定。这些区间分别对比官方 baseline，**不是 best 与 last 之间差距的区间**，不据此认定两者互有显著优势，也不据此改变预选权重。

best 相对官方 baseline 的逐帧 MPJPE21 **恶化超过1mm为281／1,015状态，即27.6847%**。平均下降不意味着每帧都改善；按录制及左右手的完整细表留在小型报告与远程配对报告中。

**左右手收益不均衡。** 下表分别按各手的有效状态汇总；测试来源为9个左手和7个右手录制，左右手的主体／动作组成不同，不能把该差异直接解释成模型手性机制的因果证据。

| 测试分组 | 官方 HaWoR MPJPE21 mm | best300.final MPJPE21 mm | last1000.final MPJPE21 mm |
|---|---:|---:|---:|
| 左手 | 61.60197 | 63.73519 | 64.88233 |
| 右手 | 81.26346 | 29.51666 | 26.03351 |

两套联合权重的左手绝对 MPJPE21 均高于 baseline，整体均值改善主要来自右手。best 左手腕误差72.01253→62.51509mm、腕对齐 MPJPE21 17.83115→11.92296mm仍改善，但这不能抵消或隐藏绝对21点指标的退化，也不能认定所有左手局部结构均无收益。目前没有实验将这一左右手差异归因于特定代码、shape、相机或训练分布。

### 校正过程中的实际退化

| best300 分支 | Wrist mm | Root ° | Finger ° | MPJPE21 mm | Wrist-aligned MPJPE21 mm |
|---|---:|---:|---:|---:|---:|
| 微调 HaWoR raw | 75.66557 | 7.80388 | 8.79745 | 73.40106 | 10.73432 |
| 最后一轮输入 candidate | 43.79721 | 7.82366 | 8.52514 | 45.38363 | 10.39711 |
| 最终 final | 46.18669 | 8.55237 | 8.40653 | 48.19357 | 11.41427 |

best 最终绝对误差低于自身 raw；但腕对齐 MPJPE21 从自身 raw 的 **10.73432→11.41427 mm，退化6.3344%**。最后一轮 TWM 更将 candidate 的 **MPJPE21 45.38363→48.19357 mm**、腕误差43.79721→46.18669mm，局部手指旋转却略改善。这是实际的末轮误改证据，不能以输出更平滑或更“合理”掩盖。

这里的 candidate **已经包含首轮 TWM、第二轮视觉读出与此前已完成的最终历史**，因此末轮 candidate→final 的退化不能解释为整个 TWM 的总贡献为负。要独立判断 TWM、视觉插件、两轮取证及历史反馈的因果贡献，仍需保持训练预算与其它条件一致的消融。

last 的自身 raw 为腕 **59.64889 mm**、MPJPE21 **57.75577 mm**、腕对齐 MPJPE21 **12.03021 mm**；其 final 的三项均降低，其中腕对齐指标改善 **10.7086%**。这与 best 的对应局部退化不同，也说明不能只用一个平均绝对位置指标判断完整校正行为。

### 不确定性仍未校准

| 集合／模型 | 标称90%的 wrist xyz椭球实际覆盖率 | 平均预测逐轴 std mm |
|---|---:|---:|
| test／best300 | 60.4926% | 34.58047 |
| test／last1000 | 36.6502% | 14.81247 |
| val／best300 | 49.2201% | 见原始报告 |
| val／last1000 | 36.3951% | 见原始报告 |

此覆盖是 current wrist translation 的三维椭球，std为 anchor 坐标中逐轴预测标准差的平均；不是所有99D状态或旋转的覆盖保证。test 上 last 的 std 更小，但实际覆盖率更低，说明输出更集中不能直接当作更可信。两套权重均明显低于标称90%，仍不能把这些 σ／κ 描述当作已校准的最终质量 q。

## 6. 可复现产物与边界

远程根目录：

`/sciclone/data10/jchen57/Hand_Reconstruction/evaluations/visual_joint_final_20261016`

| 产物 | 内容／状态 |
|---|---|
| `val_baseline/`、`val_best/`、`val_last/` | 已完成 RGB 状态推理、provenance、逐帧指标、保存的 clip predictions |
| `geometry_val.json`及对应逐帧CSV | 已完成 MANO FK、reference audit、21点和778顶点指标 |
| `comparison_val.json` | 已完成逐帧身份／mask／计数核对、配对差距、分组与bootstrap |
| `test_baseline/`、`test_best/`、`test_last/` | 已完成独立 test 的相同 RGB 输入推理与逐帧导出 |
| `geometry_test.json`、`comparison_test.json` | 已完成 test FK、逐帧配对／分组／bootstrap；引用精确source与checkpoint身份 |

执行入口与版本：状态推理 [evaluate_visual_joint.py](../scripts/evaluate_visual_joint.py)，val baseline／best为 `4730ec1`、last为 `cbcbf51`；test source preparation [prepare_visual_eval_sources.py](../scripts/prepare_visual_eval_sources.py) 为 `bc4982a`；test三套RGB推理、完整val／test MANO几何评测 [evaluate_visual_geometry.py](../scripts/evaluate_visual_geometry.py) 与配对汇总 [summarize_visual_comparison.py](../scripts/summarize_visual_comparison.py) 均使用 `e72f7ed` checkout。几何脚本最初引入于 `cbcbf51`，后续文件内容不变。Pod内Git查询返回 `unavailable` 的原始 provenance 字段保持原样；登录节点另核四个实际checkout revision，并验证其脚本SHA与所有推理／几何／汇总原报告吻合，记录在 `external_revision_audit.json`，随最终小报告保存。具体文件 SHA、checkpoint SHA、saved-rank hash 与输入 manifest 身份保存于各报告，短 commit 仅作阅读索引。

汇总图与小型结果索引分别见 [visual_joint_final_comparison_20261016.png](../reports/figures/visual_joint_final_comparison_20261016.png) 和 [visual_joint_final_comparison_20261016.json](../reports/visual_joint_final_comparison_20261016.json)。后者保留 source manifest／checkpoint／文件SHA、配对口径及远程产物索引；大权重、RGB、licensed MANO、完整预测及原始日志仍只存远程。

本次是 **GT crop、单固定相机、拟合 MANO／joint 参考**下的受控比较。绝对 world 只是明确的固定camera gauge；不是移动头戴相机 SLAM、自动 detector、独立 glove／MoCap、严格因果在线模型或官方 HaWoR 全流程／全8视角 benchmark。上述指标没有 PA similarity alignment，不能与论文 PA-MPJPE 数字直接拼表。shape、腕位置、root/finger角度、局部几何与不确定性需分别判断；没有物理定律或长期遮挡唯一恢复的额外保证。
