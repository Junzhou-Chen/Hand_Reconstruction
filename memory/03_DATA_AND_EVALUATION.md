# HandLoop 数据与评测记忆

更新日期：2026-10-13。以下是数据角色、评测设计和待落地约定；本工作区尚无完整训练数据、已运行 loader 或实验结果。依据：[START_HERE §5、§7](../HandLoop_Project/START_HERE.md)、[实现清单 §2、§4、§6–8](../HandLoop_Project/03_References_And_Implementation.md)。数据资源以[专项调研](07_TWM_DATASET_SURVEY_2026-10-13.md)的公开文件、schema、规模及来源核验为准；方法引用见[写作与外部核验](05_WRITING_AND_REFERENCES.md)。

## 两个数据池

**用户最新决定D15：优先无需申请、可直接下载的数据，并参考论文实际使用。** 首轮建议GRAB/ARCTIC/HOT3D的路线涉及注册/登记，不再作为默认开工条件；Lightwheel等人工审批资源留在背景调查。直接公开的OakInk2/DexCap/DexWild可先验收，SynData需骨架映射、HumanTouch需许可核实；最终混合尚未决定。

论文用法已经分角色整理在[完整方法框架§9](../docs/METHOD_FRAMEWORK.md#9-数据选择采用用户最新的免申请原则)：HMP独立motion prior；HaWoR视觉估计数据与HOT3D infiller不同；StableHand八库主要服务quality network；PAD-Hand用DexYCB/HO3D训练评测、TACO跨域留出。DexYCB/HO3D有直接入口，适合参考paired恢复协议，但视觉拟合标签和第三视角不能替代头戴独立MoCap结论。

**Motion-only pool** 用于独立训练 TWM，不要求配对图像。需要高质量 joints/wrist trajectories、采样时间、骨架定义与质量来源。只有局部手指姿态而缺 wrist 6DoF 的记录，不能直接冒充完整 69D GT；如何做部分监督是待决接口。

**Video–reference paired pool** 用于训练/评估图像取证、可靠性与 Corrector。需要同步视频、相机参数、参考状态，以及真实 HaWoR 输出。纯人工 corruption 的表现不能代替真实视觉错误上的表现。

## 数据资源与使用边界

| 资源 | 项目中的角色 | 已知限制与下一步 |
|---|---|---|
| GRAB + ARCTIC | 文献对照/后备MoCap来源；不再默认首批 | Vicon→MANO/SMPL-X；需注册科研下载，与D15免申请偏好有获取成本。多视角不能重复计量 |
| OakInk2 | 推荐扩展双手复杂操作 | OptiTrack→SMPL-X/MANO；627段annotations约36.65GB独立可下；需FK、单位、wrist映射验收 |
| HOT3D | MoCap motion与头戴paired后备候选 | 完整版登记获取manifest，按D15暂非首批默认；纯motion可Aria/Quest3，RGB用Aria。TEST无公开GT |
| DexWild | 推荐扩展真实手套操作 | 公开human 7,917 episodes/2,924,588 transitions；MANUS手指+视觉/SLAM腕部。取原始human，不取LEAP action |
| HumanTouch | 优先验收的新手套候选 | 25 joints/手、world腕部、统一60Hz；parquet与视频分开。已核验两个分区meta约30.40小时，非全库统计；具体许可尚未读到 |
| DexCap | 小规模glove轨迹及adapter对照 | 公开30分钟wiping+60分钟packaging；Rokoko指关节、T265腕部；不能冒充论文全集或无漂移world GT |
| AMASS | 必须筛选子集的MoCap来源 | 原论文明确无hand markers时手姿为平均值；GRAB/TCDHands/SAMP有证据，不能整库当真实手指动态；与独立GRAB去重 |
| GigaHands / TACO | 单独的多视角视觉估计扩展池 | 可提供轨迹但非独立finger marker GT；TACO的MoCap主要用于物体/头相机；GigaHands检查重心化版本 |
| EgoDex / HoloAssist / SHOW3D | 规模、泛化或质量分层实验 | 设备/视觉估计，不能称glove-only；SHOW3D的world是移动背包rig。详见专项调研 |
| HandX | 快速短窗loader与自采MoCap小补充 | 大部分聚合自HoloAssist等；自采发布短窗约17.23分钟（推算）。按来源和实际NPZ keys拆分、去重 |
| TCDHands / SAMP / FPHA / HUMOTO / BigHand2.2M | 手势、局部姿态或腕运动补充 | 规模、公开子集、完整腕朝向和连续性条件不同，不能统一当完整69D监督 |
| DexYCB / InterHand / Re:InterHand / HO-3D；AssemblyHands等 | 视觉拟合补充或跨域评测候选 | 前几项已核验部分release与限制；AssemblyHands尚未专项审核。采用前需同输入、同坐标、标签可比 |

上述均为来源事实与使用建议，尚未选定最终训练混合。完整事实、许可入口与未验收事项见[2026-10-13专项调研](07_TWM_DATASET_SURVEY_2026-10-13.md)。预训练只需要运动状态，不必限制头戴相机或配对RGB。

用户指出数采团队覆盖不足后，[产业补充调查](09_INDUSTRY_HAND_DATASETS_2026-10-13.md)新增SynData（10fps/ITW qpos映射）、OpenGraph（glove局部关节）、Daimon DataDex（手部分量待查）、OOJU（Quest26joints）、EgoVerse（完整手字段可缺省）、Lightwheel（申请开放视觉手标注）和UniHand（已有数据格式聚合）。规模扩展不能只沿首轮学术清单；先列全来源，再根据完整性、质量、许可筛选。所有数据验收/处理在hongfei运行。

GT 来源必须按 **joint articulation 与 wrist/global pose** 分开记录，不能一个 `gt_source=glove` 字段掩盖腕部来自 SLAM。设备输出也可能有误差；对 GT 来源和可用性的描述，不等于估计出了传感器误差分布。

## 当前 schema 的状态

[03 文件 §4](../HandLoop_Project/03_References_And_Implementation.md) 有 `HandFrame` dataclass 示意和建议张量，不是运行实现：

| 字段组 | 已有示意 | 必须补齐的约定 |
|---|---|---|
| 时间和身份 | `timestamp_s, sequence_id, subject_id, handedness` | 时间源、同步误差、FPS/dt、左右手 track identity、切段/reset |
| 状态 | `wrist_pos_world (3,), wrist_rot_world (3,3), joints_local (20,3)` | 关节顺序、腕坐标定义、左右手、scale、单位、bone mapping |
| 原始观测 | `raw_detected, joint_valid_mask` | 手检测、joint visibility、状态可用性和 GT label availability 分开 |
| 相机 | `camera_K, camera_T_world` | 变换方向、坐标轴、crop/full-image/feature transform、畸变处理 |
| 参考状态 | `gt_wrist_pos_world, gt_wrist_rot_world, gt_joints_local` | 缺字段/缺帧 mask、joint 与 wrist 分别的标签来源 |
| 建议网络张量 | history `[B,K,69]`、current/raw `[B,69]`、`dt [B,K]` | padding、当前 dt、mask 的 20/21/22 粒度、quality 的 `num_parts` |

存储长度单位建议米；报告腕部/关节误差使用毫米；角度误差使用度。时间戳比较和向量差分必须以实际 `dt` 为准，不能默认不同数据同 FPS。

## 分割与可比性

在训练前记录 train/validation/test 划分及主体、序列、场景标识；避免同一轨迹相邻片段跨划分造成泄漏。先定义 OOD 是跨被试、场景、数据集还是视觉 backbone，不将同域随机切分叫跨域泛化。

相机世界坐标和参考世界坐标需有可复现的固定对齐协议。不能逐帧用评测 GT 优化对齐后，再声称获得绝对世界精度。是否允许一次序列级 gauge 对齐、尺度对齐及其数据来源仍需定案并单独报告。

## 指标及其含义

| 指标 | 用途与报告条件 |
|---|---|
| World-space MPJPE | 世界坐标下所有指定 joints 的真实位置误差；注明 root 是否包含、单位、对齐协议 |
| Camera-space MPJPE | 相机坐标手部误差；帮助区分世界轨迹与局部几何问题 |
| Wrist-local joint error | articulation 恢复；不能替代 global wrist 精度 |
| Wrist translation / rotation | mm 与 SO(3) 测地角度；腕局部 frame convention 必须一致 |
| Velocity error | 与 GT 速度比较，反映运动误差；不是越慢越好 |
| 遮挡分组误差 | 分 visible/occluded、缺失长度与程度；说明分组依据来自标注还是估计 |
| 高 confidence 错误子集 | 检验原始错误高置信观测能否被纠正；阈值与误差标准须预先记录 |
| 再可见后 error decay | 随恢复可见后的时间/帧数报告重锚定过程；定义事件起点与窗口 |
| 准确关节退化率 | 检查修复是否破坏原先准确测量；阈值须固定 |
| Runtime 与延迟 | 相同硬件、batch、图像尺寸、迭代数、chunk/lookahead 条件 |

PA-MPJPE 可作补充，不能代替 world-space accuracy。指标具体聚合方式、阈值、报告置信区间的方式均尚未决定，不预填数字。

## 能检验贡献的对照

| 对照 | 要隔离的作用 |
|---|---|
| HaWoR full vs pre-infiller | 分清现成 temporal/infiller 与自有模块收益 |
| Raw + smoothing / Kalman / GRU / temporal Transformer | 改进是否超越一般时间模型 |
| Prediction-only TWM vs restoration TWM | 当前受损状态条件恢复是否必要 |
| 不输入当前可信局部状态 vs 输入 | 是否确实利用当前观测约束缺失部分 |
| Fixed confidence gate vs learned reassessment | 是否改善 false-high-confidence 和突发动作 |
| One-shot vs N=2/3 | 迭代收益及计算代价 |
| 有无 candidate-guided independent visual re-observation | 收益是否来自新图像证据，而非重复计算 |
| 有无最终状态反馈 | 递归状态维护的作用 |
| 有无自身状态 rollout training | 是否改善训练/部署历史分布差异 |
| HMP、StableHand、PAD-Hand 等 | 与已有 motion prior/refinement 的真实差异；同数据/输入/GT条件 |

跨数据集/被试、突发动作、长遮挡和再可见诊断应覆盖上述关键版本。相同 frozen TWM 接不同视觉 backbone、预训练数据规模扩展、机器人 retargeting 是后续验证方向，不是首轮必须全做的工程。

## 开工 sanity checks

1. 检查视频中的真实可见关节与 world→camera→image 投影；包括左右手和 crop 映射。
2. 检查 local/world/anchor roundtrip、rotation 正交性与 `det≈+1`、rotation6D 解码。
3. 保留插值前 detection mask；确认独立视觉证据不是 HaWoR 自己的投影点。
4. 检查时间同步和骨架映射；不同来源 GT 不可按未说明的逐帧优化对齐。
5. Clean state 输入不被无故破坏；corruption restoration 和 last-value/interpolation/GRU 作可比较验证。
6. 对 missing 输入检查预测质量是否反映未知，而非默认高置信。
7. 覆盖错误高置信、突然转向和重新出现；检查历史只在最终帧输出后更新一次。
8. 记录 HaWoR 时序、infiller、坐标转换和缓存配置，避免 baseline 混淆。

## 实验记录约定

每次实验记录日期、run ID、问题、代码/第三方 commit、配置、随机种子、数据 split/hash、GT 来源与字段、输入模态、坐标/对齐协议、训练预算、指标定义、结果路径和失败原因。真实日志与产物优先于文字印象。

所有结果目前为**未运行**。旧讨论的 30/12/5 mm 等只是示意，不进入结果表。旧版 `GapClosed=(E_camera−E_ours)/(E_camera−E_sensor)` 是可选概念指标；若 sensor 本身作标签而取 `E_sensor=0`，只表示相对该标签的误差下降，不意味着传感器物理误差为零。[历史出处 §11.4](../HandLoop_Project/01_Conversation_Research_Archive.md)
