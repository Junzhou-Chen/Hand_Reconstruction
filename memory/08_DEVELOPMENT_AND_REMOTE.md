# 开发、GitHub同步和远程实验约定

更新日期：2026-10-14。D13/D14为原GitHub与远程分工约定；最新D19因hongfei GPU占用改用ssh ds，并明确授权复用用户预申请Pod。已定约定与下文建议路径分别标注。

**当前D28：** 用户主动要求暂停并先验证效果。正式DDP run在32,783步通过STOP共同保存，23:53:53 UTC停止，rank2968/2969已退出，launcher僵尸不算活跃训练；checkpoint原目录保留。禁止当故障自动resume/删STOP/重启Job，效果验证单独在`data10/Hand_Reconstruction/evaluations/twm_paused_20261014_v1`保存快照与结果。

**D28验证完成：** `evaluations/twm_paused_20261014_v1/snapshots/{latest,best}.pt`分别32,783/28,000步，SHA43d789c5…/b2319b83…，模型与AdamW finite；00:12:06 UTC复核与原文件hash一致、STOP保留，4个评测PID已退出、GPU无活跃计算。`{latest,best}/{report.json,controls.json}`、`pause_and_launch.json`与`completion_verification.json`保存完整结果。冻结普通评测c8e9a8d、输入干预9b20493，原训练10aa983；项目Torch2.10 runtime推理，既有zhou只绘图，没有本地模型运算/optimizer更新。best原训练fixed选定指标复现差0，控制normal与fixed逐库一致；full每模式61,370窗。latest全量noisy腕5.435mm、clean退化0.454mm，GT当前＋带噪历史5.317mm暴露历史主导；这不是其他并行模型的效果。[暂停实测](../docs/TWM_PAUSED_EVAL_2026-10-14.md)、[小型汇总](../reports/twm_paused_eval_20261014.json)。

## GitHub与版本管理

- 远程：`https://github.com/Junzhou-Chen/Hand_Reconstruction.git`。
- remote名称：`origin`；主分支：`main`。
- 将代码、论文源文件、项目记忆、配置模板、数据获取/处理脚本和小型实验摘要及时commit并push；用户已经授权本项目的正常同步，无需每次重复请求。
- 不上传数据集、大型下载包、MANO/SMPL-X等受许可模型资产、权重/checkpoint、生成视频、缓存和完整运行日志；规则统一维护在根目录`.gitignore`。
- 不提交凭据、私钥、token和机器私有配置。Git不支持按文件大小自动忽略，提交前仍须检查暂存清单；`.gitignore`不会自动取消已跟踪文件。
- 小型手工整理的指标/表格放在`reports/`并跟踪，原始结果放在忽略目录；论文图和源文件不做全局图像/PDF忽略。
- 正常push，不force-push覆盖远程历史；有远程新提交时先核查并保留已有工作。

## 机器分工与当前入口

| 用途 | 当前位置/状态 |
|---|---|
| 编辑代码、论文、文档、Git同步、静态检查 | 本机 `/Users/junzhou/Project/Visual_Hand_Reconstruction` |
| 当前远程登录入口 | `ssh ds` → `cm.geo.sciclone.wm.edu`（Kubernetes登录节点） |
| 当前计算资源 | 用户明确授权的Pod `jchen57-debug-mpd5p`，namespace `jchen57` |
| 当前GPU/CPU/RAM | 已分配2×约96GiB Blackwell、12CPU、128Gi；D27正式DDP使用GPU0/1 |
| 当前ds代码checkout | `/sciclone/home/jchen57/project/Hand_Reconstruction`，D20已从GitHub clone/sync |
| 当前大数据目录 | `/sciclone/data10/jchen57/Hand_Reconstruction/datasets`，D20已建立三库子目录；其他产物以后按用途分目录 |
| 原hongfei checkout | `/data/hongfei/zhou/Hand_Reconstruction`保留备用，不再是当前唯一入口 |

训练、模型推理、数据预处理和性能评测全部远程运行；登录节点负责kubectl/Git等运维，GPU计算在Pod内。源文档、资源、环境和存储见[ds操作记录](../docs/DS_SERVER_OPERATIONS.md)。D19先学习运维，D20建立checkout与数据获取；之后D23/D25已运行模型，D26四库准备完成、单卡阶段保存停止于26,346步，当前D27已迁移到双GPU训练。本项目没有创建/删除用户Job或修改旧项目；新增依赖与数据均按下文隔离保存。

用户home和data10为Pod持久NFS挂载，`/dev/shm`为64Gi临时内存盘；scr清理政策及个人存储配额未核，不当永久唯一存储。Pod默认Python不等于zhou：默认Torch2.6/cu126在Blackwell上tiny CUDA失败，现有zhou Torch2.12/cu130两卡均通过；D25起项目官方Mamba使用独立Torch2.10/cu130 runtime且已验。不改旧项目环境/数据。

当前Job从2026-10-12 16:37:28 UTC起最多3天，按配置约10-15 16:37 UTC到期，需checkpoint持久化与续跑；退出exec不释放GPU。本轮复用用户现有Job，不按旧模板自动删除它。未来另建本项目Job时使用独立名称、label与本项目配置。

## 运行与复现

每次实验记录本地已推送commit、远程实际commit、配置、数据版本/split、命令、设备、随机种子与产物路径。当前通过Kubernetes Job/Pod运行，长期任务在持久挂载保存checkpoint/日志，并考虑Job时限；D25起使用独立Torch2.10/cu130 runtime，checkpoint持久保存。D27已实测DDP resume和仅rank1信号共同停机；sampler预取块/loader游标未完整保存，不宣称逐样本精确续跑。用户现有两卡已申请，不再为本轮学习新申请资源。

常规同步流程：本机整理并提交→push `origin/main`→远程确认工作区状态→fast-forward更新→远程运行。若远程修改了代码，先保存并提交这些改动，不用reset或rsync覆盖。

## 历史连接核验：hongfei（D19前）

- GitHub `git ls-remote`成功，初始化前远程无refs。
- `ssh hongfei`连接成功，`/data/hongfei/zhou`存在；可找到`git`和`nvidia-smi`。
- 初始提交`20fe231`已成功push到GitHub main并设置upstream；远程已从GitHub clone到`/data/hongfei/zhou/Hand_Reconstruction`，当时双方HEAD一致、远程干净。
- 远程`data/`、`checkpoints/`、`outputs/`、`logs/`已创建并受`.gitignore`保护。本轮后续文档提交继续正常push/pull；实际最新版本以Git HEAD为准。
- 初始提交20个文件、约1.78MB工作树内容，最大为约1.45MB历史方法图；没有训练数据/权重。已检查忽略规则与凭据特征。
- 以上是仓库/目录配置完成；未创建训练环境、占用GPU或启动实验。

## D19当前ds核验

ssh ds、kubectl context/namespace、用户Pod、持久挂载与两套Python已实测。极小CUDA检查见[操作记录](../docs/DS_SERVER_OPERATIONS.md#2-python环境与cuda验证)，不是项目模型实验。D20后续落实持久checkout/数据目录，旧VLM_KD只作运维参考。

## D20三库数据获取

用户要求OakInk2/HOT3D train/UmeTrack下载在计算Pod内，确认可运行后后台启动并结束，由他后续检查。脚本代码`853d2da`已推送并远程同步，Pod内4项HTTP续传/校验检查通过；22:57 UTC启动三库进程，各4workers。22:58:42 UTC快照均alive/failed=0，Oak约1.43GB、HOT约1.70GB、Ume约1.36GB已写，包含partial；不代表已校验/完成总量。实际范围约404.25GB、status/start命令与官方split约束见[下载检查](../docs/TWM_DATA_DOWNLOADS.md)。没有预处理、99D转换或训练结果。

后续完成核验：三个remote status均complete，4116文件全部verified、failed=0、字节等于manifest，进程已退出。Oak/Ume/HOT分别在纽约10月13日19:35/20:20/21:22完成；完整UTC时间见下载记录。本轮只是读取现有验收记录，没有全量再次hash或跑模型。用户最新任务是HaWoR对比数据核验，eval349 HOT clips/full27、DexYCB仍未下载。

## D21追加DexYCB与ARCTIC

用户随后明确请求新两库。代码20c64c2于UTC10-14 03:33（纽约10-13 23:33）在现有Pod启动；DexPID329/3workers，127.61GB官方13分包，03:36快照约10.66GB已写含partial，verified3/failed0。ARCTICPID330正常退出、waiting_credentials，343包仅manifest准备，无数据包获取；已问用户私有配置路径。gdown6.4.2在data10/download_runtime隔离安装，不修改zhou；校验脚本8f5d9f1的7项HTTP/CRC测试在Pod通过。源码/小文档同步Git，数据及receipts/原始日志只在data10。[详细检查与启动](../docs/COMPARISON_DATA_DOWNLOADS.md)

## D22先DexYCB，ARCTIC暂缓

纽约2026-10-14用户回复尚未注册ARCTIC，先DexYCB；不继续要求凭据或启动ARCTIC，清单保留。Dex只读快照04:00:17 UTC：PID329 alive/running，85.51/127.61GB已写含partial，verified7/checking2/downloading1/pending3/failed0。本轮前两次SSH返回UNKNOWN port65535连接关闭，后续显式ControlMaster=no/ControlPath=none的重试成功；有效配置本就没启用复用，不能据此认定stale socket，具体原因未知。未改SSH文件、任务、Job或数据。

随后核验Dex于04:15:58 UTC正常complete，13包全verified/failed0；四库合531.87GB，均不重启。用户要求检查四库能否TWM训练，06:08:55 UTC在Pod/zhou以CPU运行056122f标注审计：3Oak×20帧、4HOT、4Ume、6Dex与calibration；完整JSON存data10/Hand_Reconstruction/audits/twm_four_datasets_20261014.json，Git只同步脚本和摘要。未下载MANO资产、读取图像、全量预处理、运行网络或训练。[字段记录与边界](../docs/TWM_FOUR_DATASETS_AUDIT_2026-10-14.md)

## D23 纯轨迹首轮预训练已启动

2026-10-14用户授权从已下载四库先选一库pretrain，数学暂缓。本机只编辑/AST，Oak准备、数值8项检查、3步smoke和正式训练均在既有Pod的zhou中执行。正式源码4cb1da83cc86732b36cea76673bb1e703eae4102已同步，PID805自09:01:01UTC在GPU0后台运行，09:05 UTC step7275存活。数据在data10/prepared/oakink2_99d_pilot_v1，正式产物在data10/Hand_Reconstruction/runs/twm_oakink2_pilot_20261014；best/latest每500步保存持久盘，不需SSH不断线。50k自然完成，不自动重复启动、新建或续期Job。

运行状态由scripts/ds_train_twm.py status读取；只有进程停止且确需继续才start --resume。checkpoint保存主要训练状态，但DataLoader shuffle游标未存，实测中断重启尚未验。Smoke误写版本b04de13已在远程provenance_correction.json记录实际8593fcb，正式run自动捕获4cb1da8正确。所有大产物/原始日志留远程，Git仅源码与小摘要。[完整运行记录](../docs/TWM_OAKINK2_PILOT_RUN_2026-10-14.md)。

## D24 用户主动暂停

2026-10-14纽约05:16，已核PID805身份后SIGTERM，训练保存退出，phase=stopped。latest.pt可读step28,213、best.pt可读step27,500，代码均4cb1da8；原Job/Pod保留，没有新建/删除。pause_request.json/pause_verification.json在正式run目录。不能当故障自动resume，也不自动启动新架构run；用户后续要更充分参考WestWorld架构与参数依据且保持任务输入输出。[记录](../docs/TWM_OAKINK2_PILOT_RUN_2026-10-14.md)。

## D25 新主干与独立runtime

用户随后明确重新构造并跑起，旧D24暂停只保留旧run，不再禁止新run。新src主干359b88e及runtime源码378deda已同步。zhou Python建立完全隔离data10/runtime/twm_mamba_torch210_v1，Torch2.10.0+cu130/Triton3.6/Mamba2.3.2.post1/causal1.7，官方cp311 cu13 ABI TRUE两wheel核SHA，pip check/fused GPU/完整model12项数值检查/真实3步smoke均通过；不修改zhou、不新建Job。Pod无git导致初次数值记录包装器失败，实际checker已pass，后用登录节点核SHA补JSON；一次SSHUNKNOWN65535重试成功。

378deda47b123d64546404bf9780c19bcc48bb31于09:47:50UTC在既有Pod/GPU0启动新run PID1276，默认helper已指向新config/Python/output，旧run查询需显式output。新run数据路径与checkpoint细节见[运行记录](../docs/TWM_STRUCTURED_MAMBA_RUN_2026-10-14.md)。

D25完成核验：12:57:17UTC（纽约08:57）phase=complete50000，进程正常退出，耗时约3h9m。latest50k/best49.5k只读加载成功，SHA378deda、权重finite，100次val/2000training日志未发现非有限或OOM/Traceback。completion_audit.json保存在新run目录，Git仅同步curated final JSON和文档；未重启、延长训练或删除用户Job。

## D26 四库完整准备、合法资产与单卡历史

2026-10-14用户授权OakInk2/HOT3D/UmeTrack/DexYCB联合预训练，保留结构化Mamba与纯轨迹99D当前恢复。用户自行完成MANO官网注册、登录和非商业研究许可确认后授权下载；官方ZIP SHA256为`50976831790ea9657d8110e0c94e50e90eaf35cd76169f0b27e5d32f3fcd951f`，传到Pod后核一致。原始左右模型/三份license保留，NumPy/SciPy派生转换保持9个必要数值字段值与dtype不变；新进程验证无需Chumpy。受许可模型不进Git。资产及有限adapter来源见[报告](../reports/twm_four_source_adapters_20261014.json)。

沿用现有Pod和项目runtime `/sciclone/data10/jchen57/Hand_Reconstruction/runtime/twm_mamba_torch210_v1/bin/python`。新增smplx0.1.28、SciPy1.17.1、Chumpy0.70、six1.17.0均在项目runtime；Chumpy仅旧pickle转换兼容桥使用，未修改zhou。D26单卡阶段固定源码2b40aa9790c1d18651d6ac810d23a72d3c7a5922；后续数学/文档提交不追溯改变run provenance，早期2913f28 smoke仍单独记录。

远程共同根目录为 `/sciclone/data10/jchen57/Hand_Reconstruction`，源码仍在 `/sciclone/home/jchen57/project/Hand_Reconstruction`。以下相对路径均在该data10项目根下：

| 产物 | 路径/当前状态 |
|---|---|
| 合法原始ZIP | `model_assets/mano_v1_2/mano_v1_2.zip` |
| 原始/派生MANO与provenance | `model_assets/mano_v1_2/prepared_v1/{raw,derived}`、`model_assets/mano_v1_2/prepared_v1/provenance.json` |
| Oak full | `prepared/oakink2_99d_full_v2/manifest.json`；已complete，full_v1部分产物保留 |
| HOT/Ume full | `prepared/hot3d_99d_full_v3/manifest.json`、`prepared/umetrack_99d_full_v3/manifest.json`；NPZ，均complete |
| Dex full | `prepared/dexycb_99d_full_v1/manifest.json`；NPY，10subjects/840手序列complete |
| 真实四库smoke | `runs/twm_four_sources_smoke_20261014`、`audits/four_source_smoke_20261014/receipt.json`；2913f28，3步已通过 |
| β convention/输入审计 | `audits/beta_convention_20261014/receipt.json`及同目录`model_conditioning_receipt.json`；几何差异与实际GPU输入检查通过 |
| 四库collection | `prepared/twm_four_sources_99d_v1/manifest.json`；完整loader检查通过，四leaf SHA均匹配 |
| D26单卡历史run | `runs/twm_four_sources_structured_mamba_20261014`；2b40aa9，从头初始化后已SIGTERM saved/stopped26346，原产物保留 |
| 一次性coordinator | `audits/four_source_training_coordinator_20261014/status.json`；已完成检查并launch，不是循环重启器 |

已完成Meta官方decoder18帧等价、Dex左右各6帧及bounded train/val几何一致性检查；9项loader验收也通过。13ba665补collection叶manifest SHA防变更、resume config/manifest一致性与未知sampling拒绝。远程临时CPU检查确认leaf hash正确可读、改变leaf即拒绝；`RLIMIT_NOFILE` soft/hard均1,048,576，两个Dataset cache各256序列×5 mmap的实测FD为4→2,564→4。这是当时缓存上限的资源探针，新缓存配置/NPZ实际吞吐另验；没有运行模型或触碰真实数据来做该探针。

2913f28的真实四库3步GPU smoke loss/梯度有限，24个训练样本Dex3/HOT6/Oak9/Ume6；不是正式50k run或精度结论。Oak full改为8 workers重建full_v2，已完成431 train/39 val录制、940手序列，状态1,414,552/113,382；32帧/序列共30,080帧FK往返joint最大1.49e-7 m、mesh1.88e-7 m，只是同源实现/codec一致性。Meta原NPY写入出现性能瓶颈，改做lossless NPZ的full_v3；局部保存约9.2×不推断为全量端到端加速。随后Dex full_v1、Ume/HOT full_v3均完成，完整train/val状态合计2,788,135/271,100，可用窗口636,610/61,370。Meta最终严格重读HOT2455/Ume1020个TAR并重算全部状态，等价旧NPZ仅在核验后保留；HOT最后于17:45:27 UTC完成，详细分库统计见[启动摘要](../reports/twm_four_source_20261014_launch.json)。

β审计使用zero＋7组实际β、左右手和两种pose：同腕左手joint差3.74–19.81mm、mesh差5.60–24.28mm；right/zeroβ及full-pose旋转参数差均0。root选择本轮`condition_on_beta=false`（输入`beta_valid=false`），原始β及真实validity继续保存；未统一shape语义，99D动态状态不变。此是工程临时配置选择而非用户永久架构决定，未来开启需额外适配和消融。GPU1实际forward/backward检查也通过（3c0c）：源mask true4、模型mask false4、shape projection输入4×11全零、梯度有限，原始receipt保留在上述审计目录。

**D26单卡阶段启动与中途验收（历史）：** coordinator检查四个complete manifest及collection后，于17:51:39 UTC在GPU0从头启动PID2599，代码2b40aa9。首条training事件17:54:37 UTC step25，loss0.688073/梯度2.24077有限，四源样本Dex389/HOT421/Oak389/Ume401；首个保存验证step1000。19:02:30 UTC一次CPU只读检查latest17,000/best16,000可读，各163个浮点模型张量全finite，code/from-scratch/β关闭与manifest SHA一致；没有执行模型，也未另存远程验收receipt，事实来自只读tool返回。19:02:32状态step17,250；root后续19:05:47 UTC核step18,075，loss0.0269482、梯度1.10944，四源累计Dex289582/HOT289120/Oak288354/Ume289744。checkpoint验收时刻与后续步数分开，**未宣称50k完成**。

collection SHA256为`f733aa15cd18ed6f57ee0f16ebdadb3bd0d52924c7d99d08cea988544710b26f`，四个leaf hash逐一匹配；config SHA为`939260a27ef6a212dbc7ee85cafc6528ae7ab7c9b825e1452072d2e9e0c95302`。run_metadata记录6,330,892参数、equal_source、from scratch及β关闭。正式输出/旧smoke/旧pilot均独立，不覆盖旧产物、不重启已完成准备；D22 ARCTIC暂缓与D18 glove/MoCap优先不变。resume保存主要RNG/optimizer并核协议/manifest，但sampler预取chunk/loader游标未完整保存，不宣称逐样本精确续跑。[四库协议](../docs/TWM_FOUR_SOURCE_RUN_2026-10-14.md)、[正式启动摘要](../reports/twm_four_source_20261014_launch.json)。

## D27 当前双GPU与大batch执行

2026-10-14用户要求现有两卡一起训练并提高batch，只替代D26单卡执行设置。原run已SIGTERM保存停止于26,346步，原目录/权重/验证不覆盖。四库collection及split、完整99D、纯轨迹当前恢复、六层结构化Mamba、β另存且本轮conditioning关闭均保持。

GPU1实际batch128/256/512/768/1024探针均finite；512/卡峰值allocated34.16GiB/reserved34.79GiB、240.7样本/s，768/1024分别237.3/229.2样本/s。该探针排除loader/DDP，不当正式吞吐；选择512/卡、global1024是工程折中。LR与原50,000优化步不变，剩余23,654次更新对应24,221,696样本抽取，比batch64剩余预算16倍。验证保留64×16上限和corruption批次切分；Dex实际394窗。

| 产物/执行 | data10项目根下路径或事实 |
|---|---|
| 当前正式run | `runs/twm_four_sources_structured_mamba_ddp_20261014` |
| 正式代码 | `10aa98327eeaa3ff0afb49eef1e6ff73879c7d22` |
| 单卡迁移源 | `runs/twm_four_sources_structured_mamba_20261014/latest.pt`，step26346，SHA256 `09a10b8f800cac5e8336da3a14cf69cfce9228ff8674be84182a84102e1bf27f` |
| 正式启动 | 19:47:53 UTC；launcher2963、rank0 PID2968/GPU0、rank1 PID2969/GPU1 |
| 最新执行快照 | 19:52:33 UTC actualstep26450；新增104更新，四库实际累计106,496样本，rank各53,248且四源均正 |
| 正式checkpoint | updated step26350已保存，两rank完整模型hash相等；CPU检查163个model及489个AdamW浮点张量finite |
| 正式CPU验收receipt | `audits/twm_ddp_formal_20261014/launch_receipt.json`；文件SHA256 `7f8b72fb5972c7a9d0440b34b9fe9003f9c4159f63b844d321cfa7a2c70d3d59` |
| 被验收checkpoint文件SHA256 | step26350的`latest.pt`：`0cfe2189d21a3c875899ed0422c531699a235b794fadcfbb2ea332dac333fa89`；不与模型state hash或receipt文件hash混用 |
| 正式checkpoint模型state SHA256 | `90e1dda2f5905c473ab6f3134c7758fd6532d25ab408139fc4dfda3fff7d22b9`；CPU重算＝两rank＝replica audit |
| batch探针 | `audits/twm_batch_gpu1_20261014_v1/{receipt.json,process.json,benchmark.log}` |
| DDP smoke/恢复 | `runs/twm_ddp_smoke_20261014_v1`、`audits/twm_ddp_smoke_20261014_v1` |

双卡smoke26346→26446完成100更新；真实DDP resume后仅rank1 SIGTERM，两rank共同保存stopped26498，163个model浮点张量及499个Adam递归浮点项（含超参，非499个张量）经CPU检查finite，两rank完整model state keys/dtype/shape/bytes SHA相等。正式迁移使用旧单卡26346，不使用smoke26498；迁移保留模型、AdamW和绝对步数，重建rank RNG及采样。正式26350 checkpoint另经CPU检查，163个model及489个AdamW浮点张量finite，Adam step26350；CPU重算模型hash与两rank及replica audit一致。smoke的499是含超参的递归浮点项计数，不与正式489个AdamW浮点张量混用。

19:52:33 UTC step26450 loss0.0208302/梯度0.15867，早期435.9样本/s，不当稳态吞吐比较。19:52:47 UTC GPU0/1瞬时util均100%，显存38,172/36,686MiB，实际GPU进程为2968/2969；只记录当时占用，不推断长期利用率或峰值。

`train_twm_ddp.py`用torchrun两进程/NCCL，各rank独立seed的有放回等库采样；相等local batch的local mean loss由DDP平均梯度。rank0通过unwrapped model验证，其他rank等待广播；保存由所有rank收集RNG并核完整模型byte hash，rank0原子写plain model/optimizer。停止在共同collective边界归约flag，不在handler内保存。launcher显式resume会归档旧STOP，直接入口拒绝已有STOP，避免零步退出；同world resume核完整config，迁移只允许执行字段变化并核模型/任务/manifest。sampler块内未消费indices和loader cursor未完整保存，不能声称exact replay。沿用既有Pod与独立runtime，不修改zhou或创建/删除Job。[双卡运行记录](../docs/TWM_DUAL_GPU_RUN_2026-10-14.md)、[验收摘要](../reports/twm_dual_gpu_20261014.json)。
