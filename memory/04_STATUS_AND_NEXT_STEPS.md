# HandLoop 进展与下一步

更新日期：2026-10-13。该状态仅适用于当前工作区 `/Users/junzhou/Project/Visual_Hand_Reconstruction`。

## 已有和未有的产物

| 项目 | 当前状态 | 证据 |
|---|---|---|
| 原始研究资料 | 已有 | `HandLoop_Project/` 中 5 个 Markdown 文件与 1 张 PNG |
| 核心 idea 与现行路线 | 已形成文字方案 | START_HERE 与对话归档 §15.23–16 |
| 中文/英文 Introduction | 已有草稿与认可首段 | `02_Introduction_CN_EN.md` |
| 方法图 | 已有早期概念图 | `04_Method_Overview.png`；TWM 仍是较早的 forecasting/prior 表达 |
| 项目记忆库 | 本次建立 | 根目录 MEMORY/AGENTS 与 `memory/` |
| 部分第三方论文/源码核验 | 本次完成只读核验 | [写作与外部核验](05_WRITING_AND_REFERENCES.md) |
| TWM公开数据集调研 | 已完成来源、获取、字段及部分release元数据核验 | [专项报告](07_TWM_DATASET_SURVEY_2026-10-13.md)；尚未做实际轨迹质量验收 |
| Git与远程工作方式 | 本地main已推送GitHub；hongfei已clone项目并建忽略的存储目录 | [.gitignore](../.gitignore)、[开发与远程约定](08_DEVELOPMENT_AND_REMOTE.md)；未启动实验 |
| 产业数采数据扩查 | 已补充首轮遗漏的公司/联盟/聚合资源 | [补充调查](09_INDUSTRY_HAND_DATASETS_2026-10-13.md)，按公开程度和人体字段筛选 |
| 完整方法框架 | 已整理，新增接口建议明确标注 | [完整框架](../docs/METHOD_FRAMEWORK.md)；无模型实现/结果 |
| MANO/69D表示与标签来源 | 已核验代表性论文与源码，用户随后采纳99D；含StableHand输出/quality/shape对照 | [专题调查](10_MANO_REPRESENTATION_AND_ANNOTATION.md)；adapter与效果未验，未生成fits |
| 99D与论文职责核验 | 用户采纳99D；12项论文数据事实已核；v1选库顺序随后被D18/v2替代 | [预训练方案](../docs/TWM_PRETRAIN_PLAN.md)、[论文依据](../docs/TWM_DATA_PAPER_EVIDENCE.md)；混合未定，无训练 |
| TWM完整99D输出 | 用户再次明确每手完整99D主输出；已区分统一输出、完整模型参考与部分几何监督 | [输出契约](../docs/TWM_PRETRAIN_PLAN.md#统一99d输出与标签完整性)；未实现转换或训练 |
| 传感器精度优先v2 | 用户纠正为glove/MoCap主导；HumanTouch许可/十份meta已核，新增UmeTrack | [传感器专题](11_SENSOR_FIRST_PRETRAIN_2026-10-13.md)；无实际轨迹质量/99D转换验收 |
| 模型源码、实验依赖、权重、训练数据 | 尚未建立 | 不因Git与SSH配置完成就视为实验已启动 |
| canonicalizer、HaWoR adapter、TWM、Corrector | 尚无实现 | 原 `03` 中仅有 schema/目录建议 |
| 实验和性能、投稿和录用 | 没有可核验完成证据 | 不将草稿贡献或历史示意当结果 |

## 第一项可反驳问题

在统一时间戳、骨架、坐标与 GT 条件下，**独立轨迹预训练的 current-state restoration 能否改善真实 HaWoR errors，并超过 smoothing/Kalman/普通 temporal model？**

若没有收益，先排查坐标与 GT 映射、噪声域差异、观测质量和 baseline 公平性，不立即增加网络复杂度。[依据：归档 §16](../HandLoop_Project/01_Conversation_Research_Archive.md)

## 推荐执行顺序与交付条件

以下是下一阶段计划，不是已经执行的任务。建库和首轮TWM数据调查已完成，随后用户明确GitHub同步与远程实验约定并要求扩查数采团队。尚未创建模型工程或启动完整数据下载、训练；所有后续实验在hongfei执行。

按D18与已定99D，v2共同验收**MoCap与glove**：公开Oak/HOT/UmeTrack的native→99D，以及HumanTouch/DexWild原人体骨架＋DexCap/OpenGraph小样本。GRAB/ARCTIC有授权即为高优先来源；免申请偏好仍有效，不作为手续前置。原MoCap split约束保留，glove按subject/session/sensor校准建split；通过第一条链即可训练，不等所有库。Re/EgoDex降后续视觉来源扩展，真实HaWoR适配不挡纯轨迹起步。以上尚未实施。[v2计划](../docs/TWM_PRETRAIN_PLAN.md)

| 阶段 | 具体工作 | 可检查的交付/验收 |
|---|---|---|
| P0 传感器split与状态契约 | MoCap原录制/split，glove source/session与标定；hongfei验native→99D及raw joint几何桥接 | 字段/形态/单位/dt、MANO转换残差、腕/finger质量来源与mask；不等全部loader |
| P1 视觉与参考对齐 | 导出 pre-infiller 原状态、raw mask、时间戳、相机/crop 变换；保留 world 转换；构造 canonical GT | 投影图与 roundtrip 通过；原始视觉误差可计算；full/pre-infiller 分开 |
| P2 纯轨迹恢复 baseline | 统一 TWM 训练/部署输入；corruption + current restoration，辅短预测 | clean/noisy/missing 验证；与 last-value、interpolation、GRU/temporal model 对比 |
| P3 最小真实视觉修正 | 冻结视觉和 TWM；先做一次状态修正 | 相同训练/GT 条件下真实几何误差变化，含准确点退化；此步不声称完成视觉重取证贡献 |
| P4 有效帧内迭代 | 实现真实 image evidence 读取与可靠性重估，候选改变查询位置 | 1/2/3 轮、有无独立重取证、固定 gate 对照；记录计算开销 |
| P5 时间反馈训练 | 最终状态与质量一次回灌，短序列自身输出 rollout | feedback/no-feedback 与 rollout/no-rollout；再可见恢复、突发动作、误差累积 |
| P6 强基线与迁移 | 匹配 HMP/StableHand/PAD-Hand 等协议，做独立主体/场景/数据集验证 | 同条件对比和可复现结果；再决定论文贡献与可选下游验证 |

优先级取自 [START_HERE §7](../HandLoop_Project/START_HERE.md)；详细阶段验收是本次整理的实施建议，数值达标阈值尚未约定。

## 实现前需要做出的决定

| ID | 未决问题 | 为什么会影响实现 |
|---|---|---|
| O01 | 验收统一H/Z/candidate/M/w预训练/部署接口提案 | 完整框架已给相关raw/candidate生成方案，尚无实现；冻结后不能临时加新输入 |
| O02 | 验收预训练质量提示与最终q的命名/监督 | 完整框架限定q为最终状态质量，需落到schema和loss |
| O03 | 具体骨架、wrist orientation、左右手/scale/bone length 规范 | 决定99D MANO或69D对照的编解码及跨传感器可比性 |
| O04 | mask 是 20、21 还是 22 个分量；root pos/rot/palm 如何分组 | 决定 token、quality head 和缺失监督；需分开 observation 与 GT availability |
| O05 | `camera_T_world` 的方向、轴与 crop/feature 变换 | 字段名不足以保证正确投影 |
| O06 | 旋转左/右复合、平移 residual frame、TWM 提议的编码 | wrist-local articulation 与 temporal anchor 不能混用 |
| O07 | 首帧、空历史、整手漏检、离屏与重入、左右手关联/reset | 目前公式依赖上一帧有效 wrist，边界条件空缺 |
| O08 | 只有手指无全局 wrist 的轨迹如何纳入 | 完整99D预测保持不变；缺失world腕GT不能补造，需部分监督 |
| O09 | 视觉 hook 层、feature shape、局部/全局读取和独立 evidence head | 决定是否真的有图像支持、是否引入未来信息以及吞吐量 |
| O10 | `w/q` 的 target、损失、校准与评价定义 | 不能先把“可靠性”默认当成概率、方差或二值标签 |
| O11 | corruption 分布、loss 权重、辅助预测 horizon、dt/padding | 合成错误与实际 HaWoR 错误可能分布不同 |
| O12 | rollout 的 teacher forcing schedule、detach/截断反传和微调策略 | 决定模型是否适应自身误差及训练成本 |
| O13 | split、指标聚合、序列/world 对齐、阈值、runtime 协议 | 防止指标不可比或 GT 泄漏 |
| O14 | 远程实验的软件环境、可用GPU、分辨率、FPS及是否严格causal | 已明确hongfei和项目根路径，但未验训练环境；后端反馈不能证明系统在线 |
| O15 | 最终预训练数据组合与目标评测保留 | 已有来源调研建议；仍需实际schema/质量检查、许可和split，HumanTouch完整上传量尚未核实 |
| O16 | 已选99D的adapter、β与异构监督契约 | 表示选择已关闭；源hand mean/root offset/shape估计与51D残差、native/joint-only mask尚待实现验收 |

这些是规格空缺，不是要求现在逐项询问用户。后续实现可先提出小而明确的契约，用数据和 sanity check 决定；会改变核心研究方向的选择应明确说明。

## 已识别的主要风险与验证路径

2026-10-13完成[WestWorld与视觉手重建方法精读](../docs/WESTWORLD_AND_HAND_RECONSTRUCTION_REVIEW.md)，含原论文方法/附录及关键源码。DIP补入直接条件恢复基线；DIR/WiLoR/GeoHand补入视觉迭代前作。确认HaWoR非因果及missing-only infiller、HMP局部prior、StableHand增强版差异和W/WA对齐口径。仅更新文献事实与实验约束，未改用户主线、运行模型或新增性能结果。

2026-10-13已补充核验HaWoR原生MANO字段与双手infiller，澄清69D/66D为编码长度；字段层面有可转换的公开数据。下一步应做adapter roundtrip及实际样本检查，尚不能将所有数据标为“69D验收通过”。

- **收益只是平滑。** 用 true geometric error、快速动作和准确点退化测量。
- **自我确认。** 原始 `Z` 固定；取证必须来自图像；mask 不冒充可靠性。
- **GT 质量混合。** joint/wrist 来源分开；辅助设备追踪不冒充独立高精度标签。
- **HaWoR 工程接入失误。** `hawor_infiller()` 含 world 转换；去掉补全时单独保留转换。
- **递归累积与合成噪声偏差。** 比较真实误差、自身历史 rollout 和再可见恢复。
- **创新性不足。** 三个候选贡献需在强基线下分别证明作用；DIP已有纯运动当前恢复和序列内反馈，DIR/WiLoR/GeoHand已有候选图像查询，不能只换术语。
- **评测同名异义。** 先固定HaWoR/StableHand版本、W/WA对齐、scale与camera/shape来源，再和明确world gauge的几何误差并列报告。

## 下次会话可直接使用的任务说明

> 先读MEMORY.md、docs/METHOD_FRAMEWORK.md与本文件。项目已决定头戴RGB+HaWoR+独立纯轨迹current-state-restoration TWM+evidence-guided Corrector+最终状态质量反馈。最新数据偏好是无需申请直接下载、参考论文实际用法；按下一步指令在hongfei验收直接公开样本与状态契约，保留raw detection和camera-to-world conversion。完整框架里的接口补充是建议，需样本/代码验收。不自动下载TB级全部视频，不把推荐混合或未跑性能写成已验证结果。
