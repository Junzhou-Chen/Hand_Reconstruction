# HandLoop 进展与下一步

更新日期：2026-10-14。该状态仅适用于当前工作区 `/Users/junzhou/Project/Visual_Hand_Reconstruction`。

## 已有和未有的产物

| 项目 | 当前状态 | 证据 |
|---|---|---|
| 原始研究资料 | 已有 | `HandLoop_Project/` 中 5 个 Markdown 文件与 1 张 PNG |
| 核心 idea 与现行路线 | 已形成文字方案 | START_HERE 与对话归档 §15.23–16 |
| 中文/英文 Introduction | 已有草稿与认可首段；10月14日按当前方法与前作复核在对话中提供新中文参考稿，待用户审阅 | 原档 `02_Introduction_CN_EN.md` 保留；[重写要点](05_WRITING_AND_REFERENCES.md#2026-10-14-中文-introduction-重写)，无新增结果或设计决定 |
| 方法图 | 已有早期概念图 | `04_Method_Overview.png`；TWM 仍是较早的 forecasting/prior 表达 |
| 项目记忆库 | 本次建立 | 根目录 MEMORY/AGENTS 与 `memory/` |
| 部分第三方论文/源码核验 | 本次完成只读核验 | [写作与外部核验](05_WRITING_AND_REFERENCES.md) |
| TWM公开数据集调研 | 已完成来源、获取、字段及部分release元数据核验 | [专项报告](07_TWM_DATASET_SURVEY_2026-10-13.md)；尚未做实际轨迹质量验收 |
| Git与远程工作方式 | main已推送GitHub；ds持久checkout已建立，数据在data10，hongfei旧checkout保留 | [.gitignore](../.gitignore)、[开发与远程约定](08_DEVELOPMENT_AND_REMOTE.md)；D23训练后由D24暂停 |
| 三库数据获取 | 用户报告后已核远程phase均complete；634/2461/1021文件全verified、failed0，合计404.25GB；进程正常结束 | [下载完成记录](../docs/TWM_DATA_DOWNLOADS.md)；D26推进全量99D准备，不重下载 |
| DexYCB / ARCTIC获取 | Dex13包127.61GB已complete，全verified/failed0；D22 ARCTIC暂缓 | [完成与安排](../docs/COMPARISON_DATA_DOWNLOADS.md)；ARCTIC保留清单、无数据包下载 |
| 四库实际标注与TWM适用检查 | 字段抽查后已完成Meta官方18帧decoder等价、Dex左右各6帧与bounded train/val FK对照、Oak full每序列32帧编码FK往返；真实四库3步smoke finite | [审计](../docs/TWM_FOUR_DATASETS_AUDIT_2026-10-14.md)、[adapter摘要](../reports/twm_four_source_adapters_20261014.json)；实现一致性不是独立GT精度，正式四库训练未启动 |
| D26四库全量准备 | Oak已complete：431 train/39 val录制、940手序列，状态1,414,552/113,382；Dex当前8/10，HOT/Ume full_v3 NPZ进行中 | [四库协议](../docs/TWM_FOUR_SOURCE_RUN_2026-10-14.md)；完整collection统计与正式启动待核，不把pending manifest写成已完成 |
| 合法MANO资产与shape约定 | 用户自行官网注册、确认许可并授权下载；原始/派生模型SHA与数值兼容通过。跨源左手shape约定审计已发现差异，本轮工程配置关闭β conditioning | [资产/adapter摘要](../reports/twm_four_source_adapters_20261014.json)；β及真实validity仍另存，99D动态状态不变，不是用户指定永久关闭 |
| HaWoR训练与对比数据复核 | 主定量HOT Aria27 validation与DexYCB test；四库视觉训练、单HOT infiller、EPIC定性已核 | [数据与协议](../docs/HAWOR_DATA_AND_COMPARISON_PROTOCOL.md)；GT crop/右手/100帧及Dex s0入口要固定，尚无baseline推理 |
| 产业数采数据扩查 | 已补充首轮遗漏的公司/联盟/聚合资源 | [补充调查](09_INDUSTRY_HAND_DATASETS_2026-10-13.md)，按公开程度和人体字段筛选 |
| 完整方法框架 | 已整理，新增接口建议明确标注；D23只实施纯轨迹部分 | [完整框架](../docs/METHOD_FRAMEWORK.md)；完整视觉闭环仍未实现 |
| TWM直接校正架构讨论 | 按用户质疑独立C的瓶颈，形成统一99D恢复、真实误差适配及数值几何取证方案 | [讨论与对照](../docs/TWM_DIRECT_REFINEMENT_PROPOSAL_2026-10-14.md)；待用户调整，未替代现行视觉架构；D23仅先跑pretrain |
| TWM物理数学保证与验证 | 已核WestWorld/OneWorld与Hamiltonian/Lagrangian/硬约束论文，写出条件性证明及独立验证协议 | [数学与验证方案](../docs/TWM_PHYSICAL_GUARANTEES_AND_VALIDATION_2026-10-14.md)；未选定额外模块或实现/实测 |
| MANO/69D表示与标签来源 | 已核验代表性论文与源码，用户随后采纳99D；含StableHand输出/quality/shape对照 | [专题调查](10_MANO_REPRESENTATION_AND_ANNOTATION.md)；D26源adapter已有有限范围验收，HaWoR adapter未验，未生成新fits |
| 99D与论文职责核验 | 用户采纳99D；12项论文数据事实已核；v1选库顺序随后被D18/v2替代，D26已定本轮四库联合执行 | [预训练方案](../docs/TWM_PRETRAIN_PLAN.md)、[论文依据](../docs/TWM_DATA_PAPER_EVIDENCE.md)；采样比例为工程起点，非论文结论 |
| v2.1后续数据建议 | 原建议Oak/HOT/UmeTrack/HumanTouch，DexWild可靠local可先接、world待核；当前执行已由D26选四库 | [历史归纳](../docs/TWM_PRETRAIN_PLAN.md#0-当前推荐的预训练组合)；glove优先原则保留，ARCTIC按D22暂缓 |
| 精度比较边界 | 已区分world腕、局部手指与MANO转换；建议更信外部MoCap完整位姿，保留主要glove候选 | [精度判断](../docs/DATASET_CAPTURE_AND_ANNOTATION.md#精度判断工程建议与实测排名分开)；没有同基准8库实测排名 |
| 八库采集与标注核验 | 已核官方论文/采集说明与固定源码，新增HumanTouch VIVE证据和DexWild world处理路径问题 | [来源报告](../docs/DATASET_CAPTURE_AND_ANNOTATION.md)；未处理轨迹、未验证标签精度 |
| 逐库完整适配判定 | 已区分完整字段主候选、待验映射/腕链的glove与局部补充；当前full-state trainer只接完整17组参考mask | [适配判定](../docs/TWM_PRETRAIN_PLAN.md#候选数据的完整适配判定)；D26四库adapter/loader已有检查，全库最终验收与独立GT质量仍待补 |
| TWM完整99D输出 | 用户再次明确每手完整99D主输出；已区分统一输出、完整模型参考与部分几何监督 | [输出契约](../docs/TWM_PRETRAIN_PLAN.md#统一99d输出与标签完整性)；D23 Oak99D转换与首轮训练已实现 |
| 传感器精度优先v2 | 用户纠正为glove/MoCap主导；HumanTouch许可/十份meta已核，新增UmeTrack | [传感器专题](11_SENSOR_FIRST_PRETRAIN_2026-10-13.md)；无实际轨迹质量/99D转换验收 |
| ds计算环境 | 已核用户Pod12CPU/128Gi/两张Blackwell；zhou CUDA两卡tiny check通过，默认镜像环境失败 | [操作记录](../docs/DS_SERVER_OPERATIONS.md)；D23已用zhou在GPU0训练TWM |
| 模型源码、训练依赖、权重、训练就绪数据 | 首版821,516参数TWM/Oak64train与8val子集/远程数值检查及训练已建立；旧run按D24暂停28,213步；D25新6.33M结构化Mamba实现并正常完成50k | [新模型最终记录](../docs/TWM_STRUCTURED_MAMBA_RUN_2026-10-14.md)；新378deda/旧4cb1da8，checkpoint仅data10 |
| anchor codec、HaWoR adapter、TWM、Corrector | Oak quaternion→99D、受损历史anchor与TWM已有pilot实现；HaWoR adapter/独立Corrector/视觉取证未实现 | [训练入口](../scripts/train_twm.py)、[模型](../src/handloop/models/twm.py)；不存在完整视觉闭环结果 |
| 实验和性能、投稿和录用 | 已有D25 Oak合成受损恢复50k最终结果；D26只有3步工程smoke，完整方法/论文尚未完成 | [D25结果](../docs/TWM_STRUCTURED_MAMBA_RUN_2026-10-14.md)同时报告clean退化；不当真实HaWoR或四库收益 |

## 第一项可反驳问题

在统一时间戳、骨架、坐标与 GT 条件下，**独立轨迹预训练的 current-state restoration 能否改善真实 HaWoR errors，并超过 smoothing/Kalman/普通 temporal model？**

若没有收益，先排查坐标与 GT 映射、噪声域差异、观测质量和 baseline 公平性，不立即增加网络复杂度。[依据：归档 §16](../HandLoop_Project/01_Conversation_Research_Archive.md)

## 推荐执行顺序与交付条件

D23先跑Oak64 train/8 val子集，D24主动暂停旧flat于28,213步，best27,500保留；D25结构化Mamba以378deda在纽约08:57正常完成50k，latest50k/best49.5k可读且finite。最终合成noisy腕4.594mm/finger1.254°、clean退化1.276mm/0.500°，仅参考历史＋合成错误恢复；旧run不恢复。[D25最终记录](../docs/TWM_STRUCTURED_MAMBA_RUN_2026-10-14.md)。

**当前执行是D26四库联合预训练。** 用户已明确Oak/HOT/Ume/Dex联合，并完成MANO官网注册/许可确认与下载授权。结构化Mamba、99D当前恢复主任务、β另存不变。最新已提交源码a9342a8；全量数据准备尚未全部完成，正式四库训练未启动。库间等概率、库内窗口均匀、分库固定验证与宏平均是工程起点，不能据此称最优混合。当前trainer要求完整17组参考GT，native/raw joints未补造成完整99D。

已验收的事实与范围：

- Meta官方decoder18个状态的mesh/joints/full pose等价，99D FK最大差1.19e-7 m；Dex左右各6帧、bounded train/val的21点检查通过，最大差约1.50e-7 m。它们验证PCA/mean/左右手资产/单位/腕位置的一致性，非native或独立MoCap精度。详见[adapter摘要](../reports/twm_four_source_adapters_20261014.json)。
- loader 9项远程检查通过；collection叶manifest SHA正确时可读、改变leaf后拒绝。Pod `RLIMIT_NOFILE` soft/hard均1,048,576；当时两个Dataset cache各256序列×5 maps的实际FD probe为4→2,564→4。该probe不等于后续新缓存配置的完整性能验收。
- 真实四库3步smoke代码2913f28，loss/梯度有限；24个训练样本Dex3/HOT6/Oak9/Ume6，仅证明混合数据流可运行，不作为精度结果。receipt位于下表路径。
- Oak full已complete：431 train/39 val录制，共940单手序列；train/val状态1,414,552/113,382。每序列32帧共30,080帧，FK往返最大joint1.49e-7 m、mesh1.88e-7 m。源quat与保存99D经同一MANO实现对照，不是独立source decoder或独立传感器GT验证。
- 本快照Dex进度8/10仍活跃，HOT/Ume正准备full_v3 NPZ；未记录它们全量完成。Meta NPY输出瓶颈后进行了lossless NPZ局部保存试验，约9.2×只限该局部测试，不代表全管线速度；Oak改8 workers另建full_v2，旧full_v1部分产物保留。

远程共同根目录为 `/sciclone/data10/jchen57/Hand_Reconstruction`：

| 产物 | 根目录下相对路径与状态 |
|---|---|
| Oak full | `prepared/oakink2_99d_full_v2/manifest.json`，已complete |
| HOT/Ume full | `prepared/hot3d_99d_full_v3/manifest.json`、`prepared/umetrack_99d_full_v3/manifest.json`，目标路径，准备中 |
| Dex full | `prepared/dexycb_99d_full_v1/manifest.json`，目标路径，准备中 |
| 四库smoke | `runs/twm_four_sources_smoke_20261014`；`audits/four_source_smoke_20261014/receipt.json` |
| β约定审计 | `audits/beta_convention_20261014/receipt.json` |

**β conditioning的工程决定：** zero＋7组实际β，左右手/两种pose的审计显示同腕左手joint差3.74–19.81mm、mesh差5.60–24.28mm；right/zeroβ差0，full-pose旋转参数差0。Meta左shapedirs修复与Dex/Oak原始约定没有统一shape语义，root据此将本轮`condition_on_beta=false`，模型输入`beta_valid=false`；保存的源β及真实validity不改，99D动态GT不改。此为工程选择，不是用户要求永久移除shape条件；开启前须明确adapter并做消融，正式启动的最终配置/SHA另行记录。

下一步是等待当前全量准备完成、验证四库collection/GT masks/划分与统计，然后以独立目录正式启动并核保存及训练步数；不得把3步smoke或D25的50k称为D26正式训练。之后仍需真实HaWoR误差、自身历史rollout、独立几何与基线验证。D26替代当前单库执行，不废D18 glove/MoCap优先；HumanTouch/DexWild映射与质量验收仍为后续，D22 ARCTIC暂缓。所有实验远程，原始档案/旧run不覆盖。[四库执行协议](../docs/TWM_FOUR_SOURCE_RUN_2026-10-14.md)

| 阶段 | 具体工作 | 可检查的交付/验收 |
|---|---|---|
| P0 传感器split与状态契约 | 完成D26四库full转换/collection最终验收；后续glove source/session与native/raw joint几何桥接 | 分组split、单位/dt、源解码/FK残差、完整GT mask与provenance；不混淆roundtrip和独立精度 |
| P1 视觉与参考对齐 | 导出 pre-infiller 原状态、raw mask、时间戳、相机/crop 变换；保留 world 转换；构造 canonical GT | 投影图与 roundtrip 通过；原始视觉误差可计算；full/pre-infiller 分开 |
| P2 纯轨迹恢复 baseline | D25已完成50k；D26四库正式run待full验收后启动，current restoration主任务＋0.1s短预测 | 分库clean/noisy/missing、宏平均与candidate baseline；真实前端误差/自身rollout、GRU等公平对照待补 |
| P3 最小真实视觉修正 | 冻结视觉和 TWM；先做一次状态修正 | 相同训练/GT 条件下真实几何误差变化，含准确点退化；此步不声称完成视觉重取证贡献 |
| P4 有效帧内迭代 | 实现真实 image evidence 读取与可靠性重估，候选改变查询位置 | 1/2/3 轮、有无独立重取证、固定 gate 对照；记录计算开销 |
| P5 时间反馈训练 | 最终状态与质量一次回灌，短序列自身输出 rollout | feedback/no-feedback 与 rollout/no-rollout；再可见恢复、突发动作、误差累积 |
| P6 强基线与迁移 | 匹配 HMP/StableHand/PAD-Hand 等协议，做独立主体/场景/数据集验证 | 同条件对比和可复现结果；再决定论文贡献与可选下游验证 |

优先级取自 [START_HERE §7](../HandLoop_Project/START_HERE.md)；详细阶段验收是本次整理的实施建议，数值达标阈值尚未约定。

## 实现前需要做出的决定

| ID | 未决问题 | 为什么会影响实现 |
|---|---|---|
| O01 | 验收统一H/Z/candidate/M/w预训练/部署接口提案 | pilot已分开H/Z/candidate并实现17组mask；是否作为最终部署契约仍待真实错误验证 |
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
| O14 | 远程实验的软件环境、可用GPU、分辨率、FPS及是否严格causal | ds Pod/zhou完整模型检查及持久checkpoint已落实；resume实测和未来Job续跑待验；pilot causal不证明整条视觉系统在线 |
| O15 | D26四库联合的完整验收、目标评测保留与后续glove接入 | 本轮四库已定；等库比例为工程假设，跨库去重/独立质量、HumanTouch有效规模与映射仍待核 |
| O16 | 已选99D的adapter、β与异构监督契约 | 四库源mean/root offset已有有限验收，root据shape差异关闭本轮β conditioning但保留源β；HaWoR adapter、统一shape与native/joint-only监督待验 |

这些是规格空缺，不是要求现在逐项询问用户。后续实现可先提出小而明确的契约，用数据和 sanity check 决定；会改变核心研究方向的选择应明确说明。

## 已识别的主要风险与验证路径

2026-10-13完成[WestWorld与视觉手重建方法精读](../docs/WESTWORLD_AND_HAND_RECONSTRUCTION_REVIEW.md)，含原论文方法/附录及关键源码。DIP补入直接条件恢复基线；DIR/WiLoR/GeoHand补入视觉迭代前作。确认HaWoR非因果及missing-only infiller、HMP局部prior、StableHand增强版差异和W/WA对齐口径。仅更新文献事实与实验约束，未改用户主线、运行模型或新增性能结果。

2026-10-13已核HaWoR原生MANO字段与双手infiller；D16后现行输出是99D。D26已有Oak/Meta/Dex源adapter与FK一致性检查；HaWoR adapter和跨源独立几何精度仍需验证，不能把本轮检查扩大为全部数据质量通过。

- **收益只是平滑。** 用 true geometric error、快速动作和准确点退化测量。
- **自我确认。** 原始 `Z` 固定；取证必须来自图像；mask 不冒充可靠性。
- **GT 质量混合。** joint/wrist 来源分开；辅助设备追踪不冒充独立高精度标签。
- **HaWoR 工程接入失误。** `hawor_infiller()` 含 world 转换；去掉补全时单独保留转换。
- **递归累积与合成噪声偏差。** 比较真实误差、自身历史 rollout 和再可见恢复。
- **创新性不足。** 三个候选贡献需在强基线下分别证明作用；DIP已有纯运动当前恢复和序列内反馈，DIR/WiLoR/GeoHand已有候选图像查询，不能只换术语。
- **评测同名异义。** 先固定HaWoR/StableHand版本、W/WA对齐、scale与camera/shape来源，再和明确world gauge的几何误差并列报告。

## 下次会话可直接使用的任务说明

**2026-10-14数学讨论新增：** 用户再次要求证明当前Mamba的物理/数学性质。已完成源码2b40aa9与官方Mamba scan核查及条件性推导，见[物理证明§8–12](../docs/TWM_PHYSICAL_GUARANTEES_AND_VALIDATION_2026-10-14.md#8-2026-10-14当前mamba-twm的证明范围复核)。SO(3)更新已实现；固定β内部FK与anchor-wrapper等变各有限定；条件恢复理想风险、旋转→内部joint误差界和pre-LN Mamba内部boundedness可推导。完整反馈收缩、真实HaWoR收益与独立物理正确性未证明。本轮没有远程数值实验或训练修改，四库执行状态仍以其真实运行记录为准。

> 先读MEMORY.md、docs/TWM_FOUR_SOURCE_RUN_2026-10-14.md与本文件D26快照。D25 Oak结构化Mamba50k已正常完成，旧flat stopped28,213，两个run/权重均保留且不自动重启。D26用户已授权Oak/HOT/Ume/Dex联合预训练，合法MANO已取得；adapter/loader/hash/FD与真实3步四库smoke通过，正式四库训练尚未启动。Oak full_v2已complete（431/39录制，940手序列），Dex8/10、HOT/Ume full_v3 NPZ仍在准备，先只读查现有任务，不重启。root因实际shape约定审计选择本轮关闭β conditioning，源β与99D不变；最终code/config/完整统计与启动须回查真实receipt。数据/资产/权重/raw logs仅远程data10，代码/小摘要同步Git；D18精度优先及D22 ARCTIC暂缓仍有效。
