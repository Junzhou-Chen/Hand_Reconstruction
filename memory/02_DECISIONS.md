# HandLoop 决策与演进记忆

**2026-10-15损失调研请求与未决选择：** 用户质疑当前带权目标，要求参考world model与相近论文；这是研究/比较授权，不是已采纳新的loss或重训指令。12篇原文/实现对照后，建议低改动conditional NLL主线＋辅助项/初值梯度消融，保留DF/条件RFM作为第二路线；系数、任务采样、是否删项、是否换生成范式均未定，不新增已实施D编号。当前D37训练与D35单NLL视觉阶段保持。[事实/候选/验证边界](../docs/TWM_LOSS_LITERATURE_REVIEW_2026-10-15.md)。

更新日期：2026-10-15 UTC。来源是项目整理归档，不是官方逐字聊天记录。以下区分归档保留的用户表态、当前设计规范和未验证的实现建议；不为缺少单独日期的历史轮次补造时间。

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
| D20 | 下载OakInk2、HOT3D train、UmeTrack；下载在用户计算Pod内而非ds登录入口，脚本确认正常后后台启动并结束对话，由用户后续检查 | 2026-10-13用户直接指令，最后纠正为登录入口CPU少。已在12CPU Pod运行，后续用户报告完成且远程status全verified/failed0。scope为Oak全标注保留split、HOT train先排HaWoR评测录制、HF Ume real train跳过synthetic；不固定最终训练混合/比例。[下载记录](../docs/TWM_DATA_DOWNLOADS.md) |
| D21 | 在HaWoR数据复核后，再下载DexYCB与ARCTIC | 2026-10-13用户明确指令；ARCTIC虽需注册，用户本轮已选取，不能仅因旧免申请优先偏好拒绝处理。Dex13分包已后台运行；ARCTIC待实际官方账号，未绕注册。cropped+标注是本轮获取默认选择，非用户已定唯一图像版本或训练混合。[下载实况](../docs/COMPARISON_DATA_DOWNLOADS.md) |
| D22 | 尚未注册ARCTIC，当前先下载DexYCB；ARCTIC暂缓 | 2026-10-14用户回复账号问题“尚未注册，先下载DexYCB”；替代D21当前推进ARCTIC的安排，保留官方清单，不继续索要凭据、自动注册或启动。Dex已启动的下载继续；当时未授权新库、预处理或训练，随后D23已授权单库pretrain。 |
| D23 | 从已经下载的四库选一库，先实现并启动pretrain，争取睡前跑起来；数学讨论暂缓 | 2026-10-14用户直接指令，授权必要的模型、单库转换和远程训练。助手选OakInk2官方64 train/8 val子集，保持99D/纯轨迹/当前恢复主任务；4cb1da8正式run曾后台运行并保存checkpoint，随后D24暂停。单库、网络规模与噪声超参是本次pilot工程选择，不是最终混合或物理架构定案。[运行记录](../docs/TWM_OAKINK2_PILOT_RUN_2026-10-14.md) |
| D24 | 暂停当前tiny Transformer pilot；训练与模型构造要更充分参考WestWorld的架构和参数理由，输入输出仍按本项目任务设计 | 2026-10-14用户直接指令及后续澄清“不是完全参考”。替代D23继续本次run的执行安排；已SIGTERM保存退出，latest28,213/best27,500可读。99D、纯轨迹、当前恢复主任务不变；具体结构/超参待验证，不自动恢复或启动新run。[参考方案](../docs/TWM_WESTWORLD_GUIDED_DESIGN_2026-10-14.md) |
| D25 | 按WestWorld指导的结构适配方案重做，并启动训练；确认跑起来后再结束对话 | 2026-10-14用户直接指令“按照这样子，重新去做，然后跑起来”。撤销D24禁止另启新run的当前限制，但旧pilot保持暂停。采纳结构编码＋同帧关节attention＋时间Mamba/SSM方向和任务适配；实现6blocks/d256/4heads、连续99D当前恢复。Oak子集run已正常完成50,000步，latest50k/best49.5k可读，合成改善与clean退化同报；不证明结构优于所有对照。[实现与实测](../docs/TWM_STRUCTURED_MAMBA_RUN_2026-10-14.md) |
| D26 | 将已下载的OakInk2、HOT3D、UmeTrack、DexYCB统一预训练，保留结构化Mamba、纯数值输入、当前恢复主任务与完整99D；授权必要转换、验收和新run | 2026-10-14用户直接指令；用户自行注册、登录MANO并确认非商业研究许可后授权官网下载，合法资产已取得并核SHA。替代D23/D25单库作为当前执行范围，不覆盖旧run/原始档案，不撤销D18 glove/MoCap精度优先或D22 ARCTIC暂缓。四库full/collection已验收；2b40aa9于17:51:39 UTC从头启动的单卡阶段后来按D27保存停止于26,346步，旧数据/验证记录保留。库间等概率等细节为工程起点，非用户逐项指定或最优结论。[四库执行协议](../docs/TWM_FOUR_SOURCE_RUN_2026-10-14.md)、[单卡启动验收](../reports/twm_four_source_20261014_launch.json) |
| D27 | 当前四库TWM使用现有两张GPU一起训练，并提高batch | 2026-10-14用户直接要求“两张卡一起跑、batch调高”。仅替代D26单卡执行配置；任务、六层结构化Mamba、99D、β另存及本轮conditioning关闭、源GT/split保持。五档GPU1探针通过后，工程选择每卡512/global1024；验证仍64×16上限，LR与原50,000优化步不变。旧单卡26,346步模型/AdamW/绝对步数迁入独立run，rank RNG/采样重建，非逐样本精确续跑。双卡smoke/实际resume/仅rank1信号共同保存停机通过；10aa983正式run于19:47:53 UTC启动，19:52:33 UTC为26,450步。正式26,350 checkpoint双rank完整模型hash相等，CPU检查163个model及489个AdamW浮点张量finite；尚非50k完成。[双卡执行与边界](../docs/TWM_DUAL_GPU_RUN_2026-10-14.md)、[验收摘要](../reports/twm_dual_gpu_20261014.json) |
| D28 | 暂停当前运行，保留checkpoint，先验证目前效果 | 2026-10-14用户直接指令。实际在运行的是D27双卡训练，已通过共同STOP保存停止于32,783步，rank退出、旧run/权重保留。只授权冻结模型效果验证，不自动恢复训练；不改变模型/数据方向。固定协议/全量val的latest32,783、best28,000和历史保持基线，以及当前GT/带噪历史干预评测已完成；发现历史主导，当前观测利用不足。本轮不涉及其他并行模型。[实测报告](../docs/TWM_PAUSED_EVAL_2026-10-14.md)。 |
| D29 | 当前受损99D恢复同时预测较长未来轨迹分布；加入MoE与输出不确定性，时间编码由实现者选择；修改、启动训练并配置在线W&B | 纽约2026-10-14用户直接指令，UTC10-15实施。替代D25/D26旧短单点、无MoE/分布的设计及早期默认不加MoE约束；不恢复D28旧run。工程起点为6层d256/4dense experts、4整段modes×60步约2秒、物理秒Fourier、Gaussian＋Haar归一化isotropic matrix Fisher，13,530,032参数。四库GT/split、99D与β另存/conditioning暂关保留。4798b62新run双卡从头训练，每卡256/global512，W&B943f9c0c已在线且实际标量同步；权重/时域/数量不是最优结论，收益未验。用户随后提供新key，验证后仅存远程600私有文件、经child环境使用，不写Git/config。[设计与实测](../docs/TWM_MOE_LONG_HORIZON_RUN_2026-10-15.md) |
| D30 | 纠正已有视觉估计时，模拟噪声应尽可能贴近HaWoR实际误差 | 纽约2026-10-14用户在核对HaWoR误差后强调的目标；不等同指定把σ改成PA-MPJPE或所有错误都缩小。源码复核D29仍复用当前25mm/12°/8°、历史4mm/1.5°的corruption，尚未按HaWoR校准。具体分布、幅度/比例、真实前端适配和训练执行调整仍为建议，未改/暂停/恢复任何run。[复核与校准建议](../docs/TWM_DIRECT_REFINEMENT_PROPOSAL_2026-10-14.md#8-2026-10-14corruption应贴近hawor实际误差)。 |
| D31 | 先暂停目前训练；架构在另一聊天修改；混合高质量纯轨迹与受损条件，误差以轻中度为主、严重可有但少量 | 纽约2026-10-14用户直接指令与对D30的澄清。不必先完成真实HaWoR误差标定才能推进，后续真实视觉验证仍保留。D29已STOP共同保存停止814步，latest814/best500及额外快照通过finite/hash核验；不自动resume、不修改另一聊天架构。20/20/35/20/5与范数分档、unknown quality是本轮工程建议，尚未接入trainer、未获实验验证。[暂停摘要](../reports/twm_moe_pause_20261015.json)、[混合方案§9](../docs/TWM_DIRECT_REFINEMENT_PROPOSAL_2026-10-14.md#9-d31先暂停混合干净轨迹与适量受损条件)。 |
| D32 | 按D31当前混合设置实现并启动训练 | 纽约2026-10-14用户明确指令“很好，按照你目前的设置，跑起来”。采纳20/20/35/20/5、有界位移范数/总旋转角、quality unknown方向；0d83a54已接入独立active profile与12种验证，原架构/GT/split保持。工程选择新run从头，每卡256/global512，旧D29/D28保持暂停。4096窗语义检查＋双卡3更新/CPU hash通过，正式00:57:06UTC实际50步、checkpoint10 finite/两rank一致、W&B server step40同步；收益未验。[运行](../docs/TWM_MOE_LONG_HORIZON_RUN_2026-10-15.md#d32混合有界噪声训练已启动)、[摘要](../reports/twm_mixed_v2_20261015_launch.json)。 |
| D33 | 在HaWoR预训练权重上增加视觉侧校正反馈接口；重新梳理循环与实现逻辑，避免堆叠多项手工带权loss | 纽约2026-10-14用户直接要求。替代§10多loss菜单作为默认建议；保留TWM纯数值、raw/candidate/history分开。具体“PAM之后小型视觉读出插件＋最终当前单NLL＋冻结两主干”是本轮工程推荐，尚未实现或验证；不自动解冻HaWoR、不永久定案删除独立Corrector、不修改D32训练。[新建议§11](../docs/TWM_DIRECT_REFINEMENT_PROPOSAL_2026-10-14.md#11-d33预训练hawor读出侧反馈插件与单一视觉训练目标)。 |
| D34 | 检查当前约3k步收敛；batch翻倍为每卡512/global1024，独立从头训练10,000步 | 纽约2026-10-14用户明确要求，仅替代D32训练执行配置，不替代D33视觉方向。旧D32于02:02:24.032654UTC保存停止3,110步，latest与best均保留、额外快照SHA一致，模型/AdamW有限且CPU完整模型hash与两rank一致。旧1k→3k实际加权恢复score下降0.022255→0.012810，但noisy腕7.210mm仍高于input7.158mm，不能称全面收敛。7d53fa4a1b14f31e08f41ba545556e59adadd31f新配置保留模型、GT/split、mixed_v2、LR2e-4/warmup500，cosine终点改10k，目标10.24M样本抽取；02:12:13.931674UTC新run实际20步/20,480样本，独立检查checkpoint10有限且CPU与两rankhash一致，W&B12cc0781服务器step20同步。启动验收通过，不当10k完成或真实HaWoR收益。[执行记录](../docs/TWM_MOE_LONG_HORIZON_RUN_2026-10-15.md#d34每卡512从头训练10000步)、[本轮摘要](../reports/twm_batch512_10k_20261015.json)。 |
| D35 | 按D33视觉反馈方案搭建完整首版框架并作必要验收 | 纽约2026-10-14用户明确要求“尝试搭建出来这个完整的框架”。已实现冻结HaWoR读出插件→纯数值TWM→最终状态跨帧反馈、单一current NLL、缓存/训练/保存续跑/RGB track入口；原主干冻结、每帧一次。网络尺寸/clip32/20步为工程起点。真实127帧、官方等价、16项合同、3→20步及RGB/60步接口验收通过；GT crop/固定camera小样本中腕改善但旋转退化、coverage0，非正式性能结论。D34纯轨迹run不改，未永久决定删除所有Corrector对照或新增自动SLAM。[实现与记录](../docs/VISUAL_FEEDBACK_FRAMEWORK_2026-10-15.md)、[摘要](../reports/visual_feedback_framework_20261015.json)。 |

| D36 | 修复短时噪声生成及相关问题，重新运行；预算改为2个epoch，可中途保存停止 | 2026-10-15用户明确要求。D34已保存停止5,519步，旧checkpoint/STOP保留，替代继续10k安排。工程实现：window-local局部plateau/ramp/scattered/jitter＋因果缺失填充，encoder位置尺度0.05m，完整遍历636,610窗/epoch（用户于2026-10-15异步回复明确选择：每个窗口一次，先跑2个epoch），每卡512。比例/幅度/尺度为待验证工程假设；D35框架不改。25fee71正式两卡已实际0.080426/2epoch，语义/跨epoch续跑/尾批次/STOP/正式checkpoint finite及同步通过，仍非精度改善证明。[运行协议](../docs/TWM_TRANSIENT_EPOCH_RUN_2026-10-15.md)。 |

| D37 | 新Pod从头训练10,000 step；之前训练checkpoint仅保留best | 2026-10-15用户最新明确指令，替代D36两epoch预算与旧latest/暂停快照全保留策略。新Pod jchen57-fineturn-tl4zr；保留v3噪声、位置feature尺度、四库GT/split、full-coverage循环采样、每卡512/global1024。fresh10k/warmup500/eval500/save100为本次执行配置；87a2bcc正式已验收到40/10k，checkpoint finite/两rank同步及W&B服务器标量通过；历史18best保留/22非best删除，清理SHA通过。已确认D36两轮03:53纽约完成，但局部漂移/clean仍有失败，不保证加步数解决。[执行协议](../docs/TWM_TRANSIENT_10K_RUN_2026-10-15.md)。 |

**D38（2026-10-15，用户明确决定）：损失调整留到下个版本，本轮先检查目前效果。** D37继续原六项加权目标与10k预算；不将损失调研中的候选自动应用到当前run，也不据此启动下一版。下一版本的具体目标、系数、梯度隔离和是否采用flow尚未最终选择。[候选与边界](../docs/TWM_LOSS_LITERATURE_REVIEW_2026-10-15.md)。

## 已替代的历史路线

| 历史方案 | 当前处理 | 替代依据 |
|---|---|---|
| RGB-D 优先，再拓展 RGB | 第一版头戴 RGB；推理不依赖 depth sensor | D03、D07；START §1 |
| EgoInfinity / Do as I Do 为最终主干 | 可留作背景或下游参考；视觉主干选 HaWoR | D03 |
| WiLoR 单帧模型定义整条视觉系统 | 可作为相关组件/对照；不能替代已选 world-space 视频链路 | START §1；归档 §15.7–15.8 |
| TWM 只读历史预测下一帧 | 当前受损状态恢复＋D29较长未来轨迹分布 | D05、D29 |
| 可见关节信视觉，不可见关节独立补全 | 可信局部观测通过结构和时序关系帮助恢复其他部分 | D06 |
| 固定 confidence gate，high confidence 不可修改 | 图像证据和候选支持下重估可靠性 | D06、D08 |
| 72D→69D关节坐标 | 历史上排除重复wrist得到69D；随后D16明确选99D MANO，69D只留可选对照 | 归档§15.2；START§2；2026-10-13 D16 |
| 直接 WestWorld checkpoint、action conditioning、MPPI、MoE | 仅借鉴 structured/kinematic trajectory modelling | D04；START §1 |
| D23/D25单库作为当前执行范围 | 旧Oak pilot和50k结果保留；四库联合范围已采纳，现按D27双卡继续 | D26；不是否定glove/MoCap精度原则或自动恢复旧run |
| D26单GPU、batch64执行配置 | 旧四库run保存停止于26,346步；新run两卡、每卡512/global1024，源数据与模型任务保持 | D27；batch具体值为探针支持的工程折中，非用户逐项指定 |
| 全身/物体接触图/长短记忆/完整机器人执行链 | 暂不进入第一版；只有明确研究必要性时再扩展 | D02、D07 |
| 旧方法图作为最终架构 | 保留作历史示意；新的概念图见 MEMORY，论文图以后按实现重画 | 归档 §15.1；START 开头说明 |

## 没有定案的事项

**2026-10-15真实HaWoR条件适配：** 用户询问TWM局部/全量微调并认为全量可能合理，尚未指定执行范围。助手建议固定HaWoR，插件暖身后低LR全量TWM适配＋原motion恢复/长预测回放，局部作为对照；这补充D35冻结首版之后的研究候选，不把D35永久冻结或全量解冻写成用户已定。冻结future头不等于保持能力，仍需长forecast监督及独立保持集验收；现有trainer未实现新scope，本轮没有训练动作。[源码与官方依据](../docs/VISUAL_FEEDBACK_FRAMEWORK_2026-10-15.md#2026-10-15twm真实误差适配与先验保持讨论建议未实施)。

**2026-10-14用户发起架构重新评估：** 提出能否取消独立Corrector，让TWM直接校正视觉状态，并通过真实前端错误及增广样例联合微调，要求有依据地赞同或反驳。助手建议优先验证统一恢复TWM，保留数值几何证据入口与先验保持对照；[讨论方案](../docs/TWM_DIRECT_REFINEMENT_PROPOSAL_2026-10-14.md)。本轮不是用户已决定删除C或解冻HaWoR，D02/现行框架尚未被正式替代；D04无RGB/features、D16完整99D及D18传感器精度优先保持有效。

**同日用户要求数学物理依据：** 要求参考WestWorld/其他world model，找证明/验证及更符合物理的修改方式。助手提出分层的几何可行、参考系一致、固定目标下降和独立运动验证，真实动力学为有额外变量时的条件分支；[完整推导](../docs/TWM_PHYSICAL_GUARANTEES_AND_VALIDATION_2026-10-14.md)。此为研究讨论，未决定加入HNN、DeLaN、关节硬限位、物体物理或自动启动实验，99D与当前恢复主任务未改变。

**99D状态已按D16选定**，69D/66D是被替代的原框架候选；Corrector的51D切向残差仍为实现建议，canonicalizer不存在。`K=8`、6层、`d=256`、4heads、`N=2`和16帧unroll是历史起步配置。数据、evidence head、可靠性目标、配比和loss权重未实验确定。

用户先要求核验主流表示与MANO标注，[调查](10_MANO_REPRESENTATION_AND_ANNOTATION.md)提出MANO输出＋joint监督；随后D16采纳99D。状态选择不改变D03–D09。D26已推进Oak/Meta/Dex源解码和FK一致性检查；joint-only监督尚未进入当前full-state trainer，独立GT精度仍不能由roundtrip推出。

数据建议演进：v0 HOT＋Re→v1 HOT＋Oak及Re/EgoDex扩展→D18用户纠正为传感器精度主导。[v2计划](../docs/TWM_PRETRAIN_PLAN.md)恢复GRAB/ARCTIC/Oak/HOT高优先，新增UmeTrack、提升HumanTouch/DexWild主要glove；Re/EgoDex后续单列。随后D26明确本轮联合Oak/HOT/Ume/Dex四库，替代旧单库执行；**精度优先原则已定，等库采样比例仍是本轮工程假设**。后续glove验收及ARCTIC暂缓不变。[专题证据](11_SENSOR_FIRST_PRETRAIN_2026-10-13.md)

HaWoR及本地/远程分工已选定；D19选ds用户Pod，D20开始三库获取。D25已用独立Torch2.10/cu130 runtime完成50k；D26四库准备与单卡中途checkpoint验收通过，旧run保存停止于26,346步。D27双卡阶段随后按D28保存停止于32,783步，未完成其50k计划，旧run保持STOP；DDP实际resume与仅rank1信号共同停机已验，但sampler预取块/loader游标未完整保存，不宣称逐样本精确续跑。正式双卡checkpoint26350的CPU finite检查属于历史验收，不能代替当前D34新run的启动或最终效果验收。独立GT精度与后续Job安排仍按实际证据记录。完整缺口见[进展与下一步](04_STATUS_AND_NEXT_STEPS.md)。

[完整方法框架](../docs/METHOD_FRAMEWORK.md)给出了统一TWM输入、anchor残差、最终q、首帧缺测及单手共享参数的具体建议。它们是为消除歧义提出的候选契约，不因写入框架而升级为用户已定或实验已验证。

**2026-10-13本轮执行归纳仍是建议：** 首轮目标OakInk2/HOT3D有效train/UmeTrack train/HumanTouch，DexWild继续主要候选并先验local/world链，GRAB/ARCTIC有授权优先、DexCap小补充；没有新增用户已定混合或比例。v2.1将采集/精度审计结果落实到接入职责，不更改D18。首个checkpoint不要求四库loader全部就绪。[归纳](../docs/TWM_PRETRAIN_PLAN.md#0-当前推荐的预训练组合)

**后续用户任务：** 已要求核HaWoR论文训练/测试数据，以便在其上做对比。核验主定量HOT/Dex及代码协议后，建议HOT27原录制作world主、Dex test作camera补充；这不是用户已定全套实验split，也未授权自动重训HaWoR或下载其所有视觉训练库。GT crop、真实detector、双手、infiller与对齐需分别命名。[复核](../docs/HAWOR_DATA_AND_COMPARISON_PROTOCOL.md)

该复核之后D21用户又明确下载DexYCB/ARCTIC，已据此推进新两库获取；没有同时授权HO3D等其他训练库、重训HaWoR或运行完整实验。

**2026-10-14四库适用审计与后续D26：** 初始“MoCap三库核心＋Dex补充”是审计建议；随后用户明确要求四库统一训练，已登记D26。Meta/Dex PCA必须源解码，GT/test/缺标签与99D输出分开；有限decoder/FK检查和四库smoke通过不等于全库独立精度验收。[审计事实](../docs/TWM_FOUR_DATASETS_AUDIT_2026-10-14.md)

**D26工程选择与用户决定分开（2026-10-14）：** root选择库间等概率、库内窗口均匀采样，分别固定各库validation windows并宏平均；没有最优比例结论。Meta左手shapedirs修复与Dex/Oak原始约定不同，实际zero＋7组源β、左右两侧/两种pose审计显示同腕左手joint差3.74–19.81mm、mesh差5.60–24.28mm，right/zeroβ差0，full-pose旋转参数差0。root据此采用本轮`condition_on_beta=false`：只把模型输入`beta_valid`置false，源β及其真实有效性仍保存，99D动态状态不变。此为经检查的工程选择，不是用户指定的永久架构更改，也未统一shape语义；未来开启conditioning需adapter与消融。证据：data10项目根下`audits/beta_convention_20261014/receipt.json`；D26单卡阶段源码固定2b40aa9，D27双卡继续关闭本轮β conditioning；GPU1实际forward/backward也已核源mask true4、模型mask false4、shape projection输入4×11全零及有限梯度，receipt为同目录`model_conditioning_receipt.json`。详见[启动验收](../reports/twm_four_source_20261014_launch.json)。

**D27工程配置与预算边界（2026-10-14）：** 用户确定两卡和提高batch，没有指定512或要求LR缩放。GPU1的128/256/512/768/1024实际探针均finite；每卡512峰值allocated34.16GiB/reserved34.79GiB、240.7样本/s，768/1024为237.3/229.2样本/s。该测量排除loader和DDP，不当正式双卡稳态吞吐。采用512/卡、global1024是容量和吞吐的工程折中；验证仍64×16上限（Dex实际394窗），原LR与50,000优化步不变。剩余23,654次更新因此抽取24,221,696样本，相比batch64剩余预算16倍；不能以相同步数推断等样本公平比较，也没有证明更大batch改善精度。[双卡记录](../docs/TWM_DUAL_GPU_RUN_2026-10-14.md)。

## 对后续协作有用的偏好

**2026-10-14再次研究数学证明：** 用户认可当前Mamba设计后要求仔细思考并证明物理约束与数学推导，恢复此前暂缓的数学讨论。已将当前已实现结构、解码条件性质、理想恢复任务理论与未认证反馈/真实动力学分开推导，见[证明复核](../docs/TWM_PHYSICAL_GUARANTEES_AND_VALIDATION_2026-10-14.md#8-2026-10-14当前mamba-twm的证明范围复核)。该研究请求不自动采纳hard解剖限位、HNN、收缩正则或新训练设置；D26执行安排不变。

- 用户关心可实施、可解释的完整方法，反对为了概念完整堆叠模块。
- 当前优先收敛预训练的数据、任务、输入输出与验收；继续扩查需服务这些具体选择。
- 研究核心是物理状态关系与真实误差，不能把需求改写成远期预测、像素生成或纯平滑。
- 文献需认真吸收早期已讨论的相关工作，不能只围绕一篇 World Model 或数据集转圈。
- 有跨被试、场景、数据集、视觉系统迁移的研究愿望；目前属于验证目标，不是已有结果。
- 默认中文讨论，保留英文论文术语；引言已有中英文草稿。标题简洁、有问句，避免夸大能力。

## 后续决策记录格式

**2026-10-14（纽约）联合训练问题的研究建议：** 用户询问训练后如何接入HaWoR baseline并联合训练，未要求本轮启动视觉训练或决定删除Corrector。已按官方代码及D32接口形成[接入与分阶段训练协议§10](../docs/TWM_DIRECT_REFINEMENT_PROPOSAL_2026-10-14.md)：零适配baseline→真实前端误差适配→图像取证/校正与自身历史联合→可选HaWoR部分解冻。现TWM已条件当前Z，禁止将其解释为独立history-only prior后重复乘同观测；质量/uncertainty需区分。具体模块合并、损失权重、解冻范围、轮数和正式paired训练split仍为建议/未决，不新增已采纳D编号，不改变D32运行。

**后续D33替代说明：** 上段为早一轮建议记录；用户随后明确要求反馈进入视觉侧及简化loss。当前首版推荐以§11为准，§10接口/数据/公平评测核验保留，多loss/evidence/w/q/独立Corrector不再整套列为默认新增模块。D33记录用户方向，具体结构仍待实现与实验检验。

追加新条目时记录：日期、决定、触发问题、来源、替代哪项、影响哪些文件、状态（提案/采纳/验证/撤回）。设计提案未获采纳或未被实现证据支持时，不改写上面的现行结论。
