# HandLoop 进展与下一步

更新日期：2026-10-15 UTC（纽约10-14）。该状态仅适用于当前工作区 `/Users/junzhou/Project/Visual_Hand_Reconstruction`。

**D29当前执行：** 用户授权当前受损恢复＋较长未来分布、MoE与W&B。新13.53M模型4798b62已从头在GPU0/1训练，每卡256/global512；00:20:51 UTC验收40步、累计20,480样本四源均正，W&B服务器同step40可读。正式checkpoint10模型/AdamW finite且完整hash与两rank相等；双卡smoke3步、resume至5步通过。当前输出99D恢复、4个整段mode×60未来步、Gaussian/Matrix-Fisher不确定性，物理秒Fourier编码。D28旧run保持暂停，以下D28/D27为历史记录；新模型收益和是否解决旧模型当前观测利用不足仍待验证。[新运行记录](../docs/TWM_MOE_LONG_HORIZON_RUN_2026-10-15.md)、[小型摘要](../reports/twm_moe_long_horizon_20261015.json)、[W&B](https://wandb.ai/black-ant/handloop-twm/runs/943f9c0c)。

**D28旧模型历史状态：** 用户要求暂停并保留checkpoint、先验证效果。D27正式双卡已在32,783步共同保存退出（23:53:53 UTC/纽约19:53），STOP保留，不自动resume；冻结latest/best的固定＋全量61,370个val窗口及输入干预已完成；latest全量noisy腕5.435mm，去掉当前观测的固定窗5.451mm≈正常5.476mm，当前GT＋带噪历史仍5.317mm。证据见[暂停评测](../docs/TWM_PAUSED_EVAL_2026-10-14.md)。后文双卡启动/26,450快照属于历史，不表示当前仍在训练。

**D30训练分布要求：** 用户强调纠错噪声应尽可能贴近HaWoR实际错误。当前D29仍使用旧corruption，尚未做真实前端分布校准；建议先从train/calibration录制统计状态残差及时间/部件相关性，覆盖当前准而历史错，再检验仅减幅与分布校准的区别。论文PA/W均值不能直接充当每轴σ；未因本轮讨论修改/暂停/恢复训练，具体超参仍未定。[源码复核与建议](../docs/TWM_DIRECT_REFINEMENT_PROPOSAL_2026-10-14.md#8-2026-10-14corruption应贴近hawor实际误差)。

## 已有和未有的产物

| 项目 | 当前状态 | 证据 |
|---|---|---|
| D29当前MoE＋长轨迹分布 | 13.53M、当前99D恢复＋4modes×60步；正式两卡global512已运行，W&B在线；checkpoint有限性与保存恢复已验 | [新运行记录](../docs/TWM_MOE_LONG_HORIZON_RUN_2026-10-15.md)、[启动摘要](../reports/twm_moe_long_horizon_20261015.json)；尚无最终精度/真实视觉收益 |
| 原始研究资料 | 已有 | `HandLoop_Project/` 中 5 个 Markdown 文件与 1 张 PNG |
| 核心 idea 与现行路线 | 已形成文字方案 | START_HERE 与对话归档 §15.23–16 |
| 中文/英文 Introduction | 已有草稿与认可首段；10月14日按当前方法与前作复核在对话中提供新中文参考稿，待用户审阅 | 原档 `02_Introduction_CN_EN.md` 保留；[重写要点](05_WRITING_AND_REFERENCES.md#2026-10-14-中文-introduction-重写)，无新增结果或设计决定 |
| 方法图 | 已有早期概念图 | `04_Method_Overview.png`；TWM 仍是较早的 forecasting/prior 表达 |
| 项目记忆库 | 本次建立 | 根目录 MEMORY/AGENTS 与 `memory/` |
| 部分第三方论文/源码核验 | 本次完成只读核验 | [写作与外部核验](05_WRITING_AND_REFERENCES.md) |
| TWM公开数据集调研 | 已完成来源、获取、字段及部分release元数据核验 | [专项报告](07_TWM_DATASET_SURVEY_2026-10-13.md)；尚未做实际轨迹质量验收 |
| Git与远程工作方式 | main已推送GitHub；ds持久checkout已建立，数据在data10，hongfei旧checkout保留 | [.gitignore](../.gitignore)、[开发与远程约定](08_DEVELOPMENT_AND_REMOTE.md)；D23训练后由D24暂停 |
| 三库数据获取 | 用户报告后已核远程phase均complete；634/2461/1021文件全verified、failed0，合计404.25GB；进程正常结束 | [下载完成记录](../docs/TWM_DATA_DOWNLOADS.md)；D26全量99D准备已complete，不重下载 |
| DexYCB / ARCTIC获取 | Dex13包127.61GB已complete，全verified/failed0；D22 ARCTIC暂缓 | [完成与安排](../docs/COMPARISON_DATA_DOWNLOADS.md)；ARCTIC保留清单、无数据包下载 |
| 四库实际标注与TWM适用检查 | 字段抽查后已完成Meta官方18帧decoder等价、Dex左右各6帧与bounded train/val FK对照、Oak full每序列32帧编码FK往返；真实四库3步smoke finite | [审计](../docs/TWM_FOUR_DATASETS_AUDIT_2026-10-14.md)、[adapter摘要](../reports/twm_four_source_adapters_20261014.json)；实现一致性不是独立GT精度；四库正式训练已启动，尚非50k完成 |
| D26四库全量准备与单卡历史 | 四库均complete，8,418手序列，train/val状态2,788,135/271,100；旧单卡已SIGTERM保存停止于26,346步 | [四库协议](../docs/TWM_FOUR_SOURCE_RUN_2026-10-14.md)、[单卡启动验收](../reports/twm_four_source_20261014_launch.json)；19:07数据/验证快照保留为历史，不再是active run |
| D27双GPU、大batch训练历史（D28已暂停32,783步） | 10aa983于19:47:53 UTC迁移启动；每卡512/global1024，19:52:33 UTC实际26,450步，尚未完成50k | [双卡运行记录](../docs/TWM_DUAL_GPU_RUN_2026-10-14.md)、[验收摘要](../reports/twm_dual_gpu_20261014.json)；正式26,350 checkpoint两rank完整模型hash相等，CPU检查163个model及489个AdamW浮点张量finite；smoke/resume/单rank停机验收已过 |
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
| 逐库完整适配判定 | 已区分完整字段主候选、待验映射/腕链的glove与局部补充；当前full-state trainer只接完整17组参考mask | [适配判定](../docs/TWM_PRETRAIN_PLAN.md#候选数据的完整适配判定)；D26四库full/collection工程验收已完成，独立GT精度与跨库去重仍待补 |
| TWM完整99D输出 | 用户再次明确每手完整99D主输出；已区分统一输出、完整模型参考与部分几何监督 | [输出契约](../docs/TWM_PRETRAIN_PLAN.md#统一99d输出与标签完整性)；D23 Oak99D转换与首轮训练已实现 |
| 传感器精度优先v2 | 用户纠正为glove/MoCap主导；HumanTouch许可/十份meta已核，新增UmeTrack | [传感器专题](11_SENSOR_FIRST_PRETRAIN_2026-10-13.md)；无实际轨迹质量/99D转换验收 |
| ds计算环境 | 已核用户Pod12CPU/128Gi/两张Blackwell；zhou CUDA两卡tiny check通过，默认镜像环境失败 | [操作记录](../docs/DS_SERVER_OPERATIONS.md)；D23已用zhou在GPU0训练TWM |
| 模型源码、训练依赖、权重、训练就绪数据 | 首版821,516参数TWM/Oak64train与8val子集/远程数值检查及训练已建立；旧run按D24暂停28,213步；D25新6.33M结构化Mamba实现并正常完成50k | [新模型最终记录](../docs/TWM_STRUCTURED_MAMBA_RUN_2026-10-14.md)；新378deda/旧4cb1da8，checkpoint仅data10 |
| anchor codec、HaWoR adapter、TWM、Corrector | Oak quaternion→99D、受损历史anchor与TWM已有pilot实现；HaWoR adapter/独立Corrector/视觉取证未实现 | [训练入口](../scripts/train_twm.py)、[模型](../src/handloop/models/twm.py)；不存在完整视觉闭环结果 |
| 实验和性能、投稿和录用 | D25 Oak合成受损恢复50k最终结果已核；D26单卡已停止26,346步，D27双卡已按D28保存停止32,783步并完成冻结验证，完整方法/论文尚未完成 | [D25结果](../docs/TWM_STRUCTURED_MAMBA_RUN_2026-10-14.md)同时报告clean退化；双卡启动/恢复验收不当真实HaWoR收益或50k最终结果 |

## 第一项可反驳问题

**D28实测（旧dense结构化Mamba）：** latest32,783/best28,000都已冻结评测，3,466固定窗、61,370全量窗/模式。best原fixed选定指标复现差0；latest全量noisy腕48.703→5.435mm、root23.162→2.302°、finger15.444→1.808°，last-history hold为8.961mm/3.270°/2.839°。clean腕退化0.454mm、四库等权平均6.89%窗口增加>1mm腕误差。固定输入干预：normal5.476mm、整手当前缺失5.451mm、GT当前＋同一带噪历史5.317mm；后两者root2.280/2.233°、finger1.828/1.679°。这提示历史主导和可靠当前观测利用不足，而非完全零输入依赖（去掉当前预测仍变化1.123mm/0.446°/0.726°）。best full noisy腕5.389mm略小，但finger/clean更差，不是全指标赢家。原因假设来自history4mm/1.5° vs current25mm/12°/8°、clean同时去噪、quality未按真实误差标定；未改训练验证该假设。仅合成与拟合MANO参考，不是native传感器或真实HaWoR/闭环/物理保证。[报告与路径](../docs/TWM_PAUSED_EVAL_2026-10-14.md)。

在统一时间戳、骨架、坐标与 GT 条件下，**独立轨迹预训练的 current-state restoration 能否改善真实 HaWoR errors，并超过 smoothing/Kalman/普通 temporal model？**

若没有收益，先排查坐标与 GT 映射、噪声域差异、观测质量和 baseline 公平性，不立即增加网络复杂度。[依据：归档 §16](../HandLoop_Project/01_Conversation_Research_Archive.md)

## 推荐执行顺序与交付条件

D23先跑Oak64 train/8 val子集，D24主动暂停旧flat于28,213步，best27,500保留；D25结构化Mamba以378deda在纽约08:57正常完成50k，latest50k/best49.5k可读且finite。最终合成noisy腕4.594mm/finger1.254°、clean退化1.276mm/0.500°，仅参考历史＋合成错误恢复；旧run不恢复。[D25最终记录](../docs/TWM_STRUCTURED_MAMBA_RUN_2026-10-14.md)。

**当前D28已暂停D27旧dense run并完成冻结验证；以下D27为训练历史。** 用户要求“两张卡一起跑、batch调高”；它替代D26的单卡执行配置，不改变已授权的Oak/HOT/Ume/Dex范围、结构化Mamba、99D当前恢复、β另存/conditioning暂关、源GT及split。旧单卡源码2b40aa9790c1d18651d6ac810d23a72d3c7a5922从头训练阶段已保存停止于26,346步，run与历史验证保留；新双卡10aa98327eeaa3ff0afb49eef1e6ff73879c7d22迁移模型、AdamW和绝对步数，重建rank RNG/采样。库间等概率、库内窗口均匀、固定分库验证/宏平均和每卡512均为工程选择，非最优结论；trainer仍只接完整17组参考GT，native/raw joints未补造成完整99D。

已验收的事实与范围：

- Meta官方decoder18个状态的mesh/joints/full pose等价，99D FK最大差1.19e-7 m；Dex左右各6帧、bounded train/val的21点检查通过，最大差约1.50e-7 m。它们验证PCA/mean/左右手资产/单位/腕位置的一致性，非native或独立MoCap精度。详见[adapter摘要](../reports/twm_four_source_adapters_20261014.json)。
- loader 9项远程检查通过；collection叶manifest SHA正确时可读、改变leaf后拒绝。Pod `RLIMIT_NOFILE` soft/hard均1,048,576；当时两个Dataset cache各256序列×5 maps的实际FD probe为4→2,564→4。该probe不等于后续新缓存配置的完整性能验收。
- 真实四库3步smoke代码2913f28，loss/梯度有限；24个训练样本Dex3/HOT6/Oak9/Ume6，仅证明混合数据流可运行，不作为精度结果。receipt位于下表路径。
- Oak full已complete：431 train/39 val录制，共940单手序列；train/val状态1,414,552/113,382。每序列32帧共30,080帧，FK往返最大joint1.49e-7 m、mesh1.88e-7 m。源quat与保存99D经同一MANO实现对照，不是独立source decoder或独立传感器GT验证。
- 四库full均已complete，最后的HOT在17:45:27 UTC完成2455 TAR/4910手序列；Ume在17:31:36 UTC完成1020 TAR/1728手序列；Dex10 subjects/840序列已完成。Meta严格重读并重解码全部TAR：HOT已有1809个NPZ、Ume已有1481个NPZ经重算等价核对后保留，未以文件存在代替验收。历史Meta NPY输出瓶颈与lossless NPZ局部约9.2×保存测试保留，不代表全管线速度；Oak改8 workers另建full_v2，旧full_v1部分产物保留。
- D26单卡阶段历史：coordinator完成collection检查后正式launch；首条真实训练日志17:54:37 UTC step25，loss0.688073/梯度2.24077有限，Dex389/HOT421/Oak389/Ume401，首验证保存step1000。19:02:32 UTC step17,250；随后root在19:05:47 UTC核step18,075，loss0.0269482/梯度1.10944有限，累计四源Dex289582/HOT289120/Oak288354/Ume289744。19:02一次CPU只读验收latest17,000/best16,000可读，各163个浮点模型张量全finite；未另存远程验收receipt，以只读tool记录及[启动摘要](../reports/twm_four_source_20261014_launch.json)为据。不宣称50k完成。

远程共同根目录为 `/sciclone/data10/jchen57/Hand_Reconstruction`：

| 产物 | 根目录下相对路径与状态 |
|---|---|
| Oak full | `prepared/oakink2_99d_full_v2/manifest.json`，已complete |
| HOT/Ume full | `prepared/hot3d_99d_full_v3/manifest.json`、`prepared/umetrack_99d_full_v3/manifest.json`，均complete，NPZ |
| Dex full | `prepared/dexycb_99d_full_v1/manifest.json`，complete，NPY |
| 四库smoke | `runs/twm_four_sources_smoke_20261014`；`audits/four_source_smoke_20261014/receipt.json` |
| 四库collection | `prepared/twm_four_sources_99d_v1/manifest.json`，完整检查通过且四个leaf SHA匹配 |
| D26单卡历史run | `runs/twm_four_sources_structured_mamba_20261014`；已saved/stopped26346，旧coordinator为`audits/four_source_training_coordinator_20261014/status.json` |
| D27双卡run（D28已暂停） | `runs/twm_four_sources_structured_mamba_ddp_20261014`；launcher2963、ranks2968/2969、GPU0/1 |
| 双卡smoke/恢复验收 | `runs/twm_ddp_smoke_20261014_v1`；对应`audits/twm_ddp_smoke_20261014_v1` |
| GPU1 batch探针 | `audits/twm_batch_gpu1_20261014_v1/{receipt.json,process.json,benchmark.log}` |
| 正式双卡CPU验收receipt | `audits/twm_ddp_formal_20261014/launch_receipt.json`；checkpoint26350模型/AdamW finite，CPU完整模型hash与两rank一致 |
| β约定及输入审计 | `audits/beta_convention_20261014/receipt.json`、`model_conditioning_receipt.json`（同目录） |

**β conditioning的工程决定：** zero＋7组实际β，左右手/两种pose的审计显示同腕左手joint差3.74–19.81mm、mesh差5.60–24.28mm；right/zeroβ差0，full-pose旋转参数差0。Meta左shapedirs修复与Dex/Oak原始约定没有统一shape语义，root据此将本轮`condition_on_beta=false`，模型输入`beta_valid=false`；保存的源β及真实validity不改，99D动态GT不改。此为工程选择，不是用户要求永久移除shape条件；开启前须明确adapter并做消融。D26单卡run 2b40aa9已关闭β，D27保持该配置；GPU1实际forward/backward检查（3c0c）确认源mask true4/模型mask false4、shape projection输入4×11全零且梯度有限。

完整数据统计如下；Dex两条train源序列不能构成有效连续窗口，因此manifest共有8,418手序列，loader实际使用train7,608/val808，共8,416条。固定验证每库最多1024窗，Dex全部仅394窗，不写成每库都有1024窗。

| 来源 | 完整手序列 | train / val状态 | train / val可用窗口 |
|---|---:|---:|---:|
| OakInk2 | 940 | 1,414,552 / 113,382 | 346,860 / 27,728 |
| HOT3D | 4,910 | 660,300 / 76,200 | 132,008 / 15,240 |
| UmeTrack | 1,728 | 662,422 / 78,677 | 151,390 / 18,008 |
| DexYCB | 840 | 50,861 / 2,841 | 6,352 / 394 |
| 合计 | 8,418 | 2,788,135 / 271,100 | 636,610 / 61,370 |

Ume MANO有效741,099帧、native有效731,416帧；MANO有效但native absent/invalid各9,725，其中invalid包含absent，不能相加。保留独立mask，不把MANO拟合与native独立精度等同。

**D26单卡历史快照19:07:32 UTC：** step18,550/50,000、phase=training，约4.21step/s；742条training记录loss/梯度有限，四源累计样本Dex297154/HOT296754/Oak295896/Ume297396。step18,000固定val子集共3,466窗/模式，四库等权noisy腕49.542→6.155mm、root22.747→2.512°、finger15.481→2.153°；missing腕37.897→6.066mm。clean退化腕2.925mm、root1.456°、finger1.388°，必须同报。此为合成受损当前恢复的中途快照，不是最终50k、独立joint/mesh精度或真实HaWoR结果；分库与forecast数值在启动JSON。19:02 checkpoint CPU读取检查仍单独按当时17000/16000记录。

**D27双卡运行历史与验收（D28已暂停）：** GPU1 batch128/256/512/768/1024探针均finite；512/卡峰值allocated34.16GiB/reserved34.79GiB、240.7样本/s，768/1024为237.3/229.2样本/s。工程选512/卡、global1024；上述是排除loader/DDP的单卡探针，不是正式端到端吞吐。验证仍`eval_batch_size=64`、`eval_batches=16`，保留原窗口集合与corruption批次切分，Dex实际394窗。LR与原50,000优化步计划保持，剩余23,654×1024=24,221,696样本抽取，是旧batch64剩余预算的16倍。

双卡smoke实际26346→26446共100更新，随后真实DDP resume并仅对rank1发SIGTERM，两rank共同保存stopped26498；CPU检查163个model浮点张量及499个Adam递归浮点项（含超参，非499个张量）finite，两rank完整模型state的keys/dtype/shape/bytes SHA一致。正式迁移源是旧单卡latest26346，SHA256 `09a10b8f800cac5e8336da3a14cf69cfce9228ff8674be84182a84102e1bf27f`，不是smoke26498。正式run于19:47:53 UTC启动，launcher2963，rank0/1为2968/2969并使用GPU0/1；19:52:33 UTC actual step26450（迁移后104更新），四源实际累计抽取106,496=104×1024样本，各rank53,248且四源均正；该step loss0.0208302/梯度0.15867，早期435.9样本/s，不当稳态吞吐比较。updated checkpoint26350已独立CPU读取，163个model及489个AdamW浮点张量finite、Adam step26350；CPU重算完整模型hash与两rank及replica audit相等。19:52:47 UTC两卡瞬时util100%、显存38,172/36,686MiB，PID2968/2969；这不是长期利用率或峰值显存结论。严格逐样本resume未成立，sampler预取块/loader游标仍未完整保存。[运行与验收](../docs/TWM_DUAL_GPU_RUN_2026-10-14.md)、[摘要](../reports/twm_dual_gpu_20261014.json)。

下一步按D29验证新MoE/长轨迹模型，按D28保持旧dense run暂停，先复查当前观测/历史可靠性与corruption设计，再考虑真实HaWoR验证。D27未完成50k；不自动重启旧run、重复下载或准备；D29新run已另行授权启动。50k最终结果尚未核验，也不能把D25的50k挪作D26结果。之后仍需真实HaWoR误差、自身历史rollout、独立几何与基线验证。D26替代单库范围，D27替代单卡执行，不废D18 glove/MoCap优先；HumanTouch/DexWild映射与质量验收仍为后续，D22 ARCTIC暂缓。所有实验远程，原始档案/旧run不覆盖。[四库执行协议](../docs/TWM_FOUR_SOURCE_RUN_2026-10-14.md)

| 阶段 | 具体工作 | 可检查的交付/验收 |
|---|---|---|
| P0 传感器split与状态契约 | D26四库full/collection工程验收已完成；后续glove source/session与native/raw joint几何桥接 | 分组split、单位/dt、源解码/FK残差、完整GT mask与provenance；不混淆roundtrip和独立精度 |
| P1 视觉与参考对齐 | 导出 pre-infiller 原状态、raw mask、时间戳、相机/crop 变换；保留 world 转换；构造 canonical GT | 投影图与 roundtrip 通过；原始视觉误差可计算；full/pre-infiller 分开 |
| P2b D29长轨迹分布与MoE | 新模型已启动：13.53M、60步/4modes、当前恢复＋分布＋velocity；W&B在线，两卡global512 | 分库分时域MAP与hold/常速度、NLL/coverage、clean退化、current输入干预与公平消融待正式结果 |
| P2 纯轨迹恢复 baseline | D25已完成50k；D26单卡停止26,346步，D27已暂停32,783步，冻结评测完成，尚未完成50k；current restoration主任务＋0.1s短预测 | 后续checkpoint检查、分库clean/noisy/missing、宏平均与累计样本量；真实前端误差/自身rollout、GRU公平对照待补 |
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
| O14 | 远程实验的软件环境、可用GPU、分辨率、FPS及是否严格causal | ds Pod/独立runtime完整模型检查及持久checkpoint已落实；D27 DDP resume与单rank信号停机已验，逐样本精确续跑未证、未来Job续跑待验；pilot causal不证明整条视觉系统在线 |
| O15 | D27双卡最终训练评估、目标评测保留与后续glove接入 | 本轮四库已定；等库比例及batch为工程选择，跨库去重/独立质量、HumanTouch有效规模与映射仍待核 |
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

> 最新D29优先：先读MEMORY.md与docs/TWM_MOE_LONG_HORIZON_RUN_2026-10-15.md。新13.53M模型4798b62已在原Pod两卡从头训练，run=twm_four_sources_moe_long_horizon_20261015，每卡256/global512；W&B https://wandb.ai/black-ant/handloop-twm/runs/943f9c0c。当前99D恢复＋4整段modes×60步分布，当前/未来不确定性；新key仅存远程私有文件，勿读/打印/提交。16:25UTC共同保存早于Job16:37UTC到期；当前step读status.json，摘要快照不当实时状态。下文是D28旧模型移交背景。

> 先读MEMORY.md、docs/TWM_PAUSED_EVAL_2026-10-14.md及reports/twm_paused_eval_20261014.json。D28按用户要求暂停旧6.33M dense DDP，STOP保存停止32,783步，禁止自动resume；latest32,783和best28,000原件及评测快照hash相同，模型/AdamW finite。固定3,466窗、full61,370窗/模式与输入干预全部完成，原best fixed指标复现差0。latest full noisy腕48.703→5.435mm、finger15.444→1.808°，但固定normal5.476mm≈整手当前缺失5.451mm，当前GT＋带噪历史仍5.317mm，提示历史主导及可靠当前观测利用不足；原因假设需另行验证，未重新训练。全部模型评测已退出，不当真实HaWoR/独立GT/物理保证。远程产物在data10项目根evaluations/twm_paused_20261014_v1，旧run仍runs/twm_four_sources_structured_mamba_ddp_20261014。D25已完成50k、D26停止26,346及D27启动记录保留为历史；本条不涉及其他并行模型。
