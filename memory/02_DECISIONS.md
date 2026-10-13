# HandLoop 决策与演进记忆

更新日期：2026-10-13。来源是项目整理归档，不是官方逐字聊天记录。以下区分归档保留的用户表态、当前设计规范和未验证的实现建议；不为缺少单独日期的历史轮次补造时间。

## 决策效力

新用户指令优先于旧记忆。当前原始设计以 [START_HERE](../HandLoop_Project/START_HERE.md) 为准；[讨论归档](../HandLoop_Project/01_Conversation_Research_Archive.md) §15.23–16 提供对应完整框架。早期 §1–14 中的“最终”“当前”是当时语境，不具有现在的优先级。

已确定的是问题、边界与主路线，不是所有网络超参和 tensor schema。新事实若只涉及第三方实现，应作为勘误保存；不要据此自动更改研究选型。

## 可回溯的关键决定

| ID | 结论与理由 | 证据等级和来源 |
|---|---|---|
| D01 | 用高质量设备运动知识改善 camera-only 状态估计，降低大规模采集的设备依赖 | 原始 idea 整理；归档 §2.1，L89–99 |
| D02 | 保持三模块和两个闭环；不过度扩展 memory、contact、physics、robotics | 用户纠偏摘要；归档 §4.1，L183–191；START §1 |
| D03 | HaWoR + 自训练 TWM + 头戴相机 | 用户明确表态；归档 §15.8，L915 |
| D04 | TWM 纯轨迹学习，不接视觉，不复用 WestWorld 模型；WestWorld 只借鉴思想 | 用户明确表态；归档 §15.10，L942；START §1 |
| D05 | 主目标是当前视觉状态纠正；预测只是帮助学习运动过程 | 用户明确纠偏；归档 §15.11，L956–973 |
| D06 | 利用当前可信部分恢复不可信部分；高 confidence 仍可能错 | 同轮用户表态；归档 L956；条件恢复机制的展开 L962–971 |
| D07 | 第一版为 RGB、冻结 HaWoR、wrist 6DoF + hand joints；不做完整 arm/object/control 系统 | 当前设计规范；START §1；不是上述每项都有独立用户逐字指令 |
| D08 | 原始 `Z`、候选 `S`、历史 `H` 分开；候选引导图像取证；`w` 与 `q` 分开；最后状态只反馈一次 | 最新整合方案；归档 §15.23 B/E/F/G；START §3 |
| D09 | 先 pure-motion pretrain，再 paired correction，再自身状态 rollout | 稳定训练设计；START §4；细节是建议 |
| D10 | 方法研究参考完整 reconstruction pipeline；优先近年的 method papers，数据集不主导贡献定位 | 归档保留用户原话 §15.4，L864；不等于禁止用早期强基线 |
| D11 | 标题偏好问句 + 简洁技术副句；无手套仅指部署 | 用户标题偏好 §15.12–15.15，L979–1007；具体标题未最终定稿 |
| D12 | 保留已认可的中英文引言首段及其叙事顺序 | 用户纠正首段 §15.18–15.20，L1046–1079；完整文本见 02 引言档案 |
| D13 | 正常同步到`https://github.com/Junzhou-Chen/Hand_Reconstruction.git`的`main`；排除数据集和大文件，维护`.gitignore` | 2026-10-13用户直接指令；已采纳，详见[开发约定](08_DEVELOPMENT_AND_REMOTE.md) |
| D14 | 本机只写代码/论文等，所有实验远程；原初步用hongfei | 2026-10-13用户直接指令；远程分工仍有效，当前计算入口由D19改为ds；hongfei checkout保留备用 |
| D15 | 数据优先无需申请、可直接下载，参考当前相关论文实际使用的训练/评测数据；不把申请资源当开工前置 | 2026-10-13用户直接指令；已采纳，替代助手先GRAB/ARCTIC、优先申请产业库的默认顺序；不自动决定具体训练混合 |
| D16 | 采用每手99D MANO动态状态（p3＋root6＋15×finger6）；β另存；TWM主输出固定完整99D，异构标签不改变输出维数 | 2026-10-13用户明确“我们使用99D”，本轮再次强调“我希望我训练的TWM是能够预测99D的”；替代69D默认候选；部分监督/拟合不等于全参数GT，实现与效果未验证 |
| D17 | 以后plan广泛对照当前最新相关论文实际做法，适当参考后再规划，避免凭直觉 | 2026-10-13用户直接指令；已采纳；任务角色/GT/split/时间上下文分开比较，工作约定写入AGENTS |
| D18 | TWM不需要视觉；选库优先更准确的手姿态，恢复glove/MoCap导向；论文用库作参考，不必照搬 | 2026-10-13用户直接纠正；已采纳，限定D17用法并替代v1 glove支线/Re-EgoDex优先扩展建议；99D与免申请偏好未撤销 |
| D19 | hongfei GPU占用，当前改用ssh ds的Kubernetes资源；授权进入/使用用户预申请Pod jchen57-debug-mpd5p，先了解运维 | 2026-10-13用户直接指令及Pod具体授权；替代D14的当前计算位置，不改变方法。已进入、完成轻量CUDA验证；该运维核验阶段未迁移/训练，后续下载见D20。[操作记录](../docs/DS_SERVER_OPERATIONS.md) |
| D20 | 下载OakInk2、HOT3D train、UmeTrack；下载在用户计算Pod内而非ds登录入口，脚本确认正常后后台启动并结束对话，由用户后续检查 | 2026-10-13用户直接指令，最后纠正为登录入口CPU少。已实现续传/校验与查询脚本并在12CPU Pod启动。具体scope为Oak全标注保留split、HOT train先排HaWoR评测录制、HF Ume real train跳过synthetic；属于本轮获取实施选择，不固定最终训练混合/比例。[下载记录](../docs/TWM_DATA_DOWNLOADS.md) |

## 已替代的历史路线

| 历史方案 | 当前处理 | 替代依据 |
|---|---|---|
| RGB-D 优先，再拓展 RGB | 第一版头戴 RGB；推理不依赖 depth sensor | D03、D07；START §1 |
| EgoInfinity / Do as I Do 为最终主干 | 可留作背景或下游参考；视觉主干选 HaWoR | D03 |
| WiLoR 单帧模型定义整条视觉系统 | 可作为相关组件/对照；不能替代已选 world-space 视频链路 | START §1；归档 §15.7–15.8 |
| TWM 只读历史预测下一帧 | 恢复当前受损状态为主，短预测为辅 | D05 |
| 可见关节信视觉，不可见关节独立补全 | 可信局部观测通过结构和时序关系帮助恢复其他部分 | D06 |
| 固定 confidence gate，high confidence 不可修改 | 图像证据和候选支持下重估可靠性 | D06、D08 |
| 72D→69D关节坐标 | 历史上排除重复wrist得到69D；随后D16明确选99D MANO，69D只留可选对照 | 归档§15.2；START§2；2026-10-13 D16 |
| 直接 WestWorld checkpoint、action conditioning、MPPI、MoE | 仅借鉴 structured/kinematic trajectory modelling | D04；START §1 |
| 全身/物体接触图/长短记忆/完整机器人执行链 | 暂不进入第一版；只有明确研究必要性时再扩展 | D02、D07 |
| 旧方法图作为最终架构 | 保留作历史示意；新的概念图见 MEMORY，论文图以后按实现重画 | 归档 §15.1；START 开头说明 |

## 没有定案的事项

**99D状态已按D16选定**，69D/66D是被替代的原框架候选；Corrector的51D切向残差仍为实现建议，canonicalizer不存在。`K=8`、6层、`d=256`、4heads、`N=2`和16帧unroll是历史起步配置。数据、evidence head、可靠性目标、配比和loss权重未实验确定。

用户先要求核验主流表示与MANO标注，[调查](10_MANO_REPRESENTATION_AND_ANNOTATION.md)提出MANO输出＋joint监督；随后D16采纳99D。状态选择不改变D03–D09，参数转换和joint-only监督仍需验收。

数据建议演进：v0 HOT＋Re→v1 HOT＋Oak及Re/EgoDex扩展→D18用户纠正为传感器精度主导。[v2计划](../docs/TWM_PRETRAIN_PLAN.md)恢复GRAB/ARCTIC/Oak/HOT高优先，新增UmeTrack、提升HumanTouch/DexWild主要glove；Re/EgoDex后续单列。**精度优先原则已定，具体库/比例仍建议**；免申请资源先验，未新增用户采纳的混合。[专题证据](11_SENSOR_FIRST_PRETRAIN_2026-10-13.md)

HaWoR及本地/远程分工已选定；D19选ds用户Pod，D20开始三库获取。GPU和zhou基础CUDA已核；下载脚本有实际文件验收，数据native→99D、完整训练依赖与后续Job续跑仍需检查。完整缺口见[进展与下一步](04_STATUS_AND_NEXT_STEPS.md)。

[完整方法框架](../docs/METHOD_FRAMEWORK.md)给出了统一TWM输入、anchor残差、最终q、首帧缺测及单手共享参数的具体建议。它们是为消除歧义提出的候选契约，不因写入框架而升级为用户已定或实验已验证。

**2026-10-13本轮执行归纳仍是建议：** 首轮目标OakInk2/HOT3D有效train/UmeTrack train/HumanTouch，DexWild继续主要候选并先验local/world链，GRAB/ARCTIC有授权优先、DexCap小补充；没有新增用户已定混合或比例。v2.1将采集/精度审计结果落实到接入职责，不更改D18。首个checkpoint不要求四库loader全部就绪。[归纳](../docs/TWM_PRETRAIN_PLAN.md#0-当前推荐的预训练组合)

## 对后续协作有用的偏好

- 用户关心可实施、可解释的完整方法，反对为了概念完整堆叠模块。
- 当前优先收敛预训练的数据、任务、输入输出与验收；继续扩查需服务这些具体选择。
- 研究核心是物理状态关系与真实误差，不能把需求改写成远期预测、像素生成或纯平滑。
- 文献需认真吸收早期已讨论的相关工作，不能只围绕一篇 World Model 或数据集转圈。
- 有跨被试、场景、数据集、视觉系统迁移的研究愿望；目前属于验证目标，不是已有结果。
- 默认中文讨论，保留英文论文术语；引言已有中英文草稿。标题简洁、有问句，避免夸大能力。

## 后续决策记录格式

追加新条目时记录：日期、决定、触发问题、来源、替代哪项、影响哪些文件、状态（提案/采纳/验证/撤回）。设计提案未获采纳或未被实现证据支持时，不改写上面的现行结论。
