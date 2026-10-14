# 开发、GitHub同步和远程实验约定

更新日期：2026-10-14。D13/D14为原GitHub与远程分工约定；最新D19因hongfei GPU占用改用ssh ds，并明确授权复用用户预申请Pod。已定约定与下文建议路径分别标注。

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
| 当前GPU/CPU/RAM | 已分配2×约96GiB Blackwell、12CPU、128Gi；GPU检查时空闲 |
| 当前ds代码checkout | `/sciclone/home/jchen57/project/Hand_Reconstruction`，D20已从GitHub clone/sync |
| 当前大数据目录 | `/sciclone/data10/jchen57/Hand_Reconstruction/datasets`，D20已建立三库子目录；其他产物以后按用途分目录 |
| 原hongfei checkout | `/data/hongfei/zhou/Hand_Reconstruction`保留备用，不再是当前唯一入口 |

训练、模型推理、数据预处理和性能评测全部远程运行；登录节点负责kubectl/Git等运维，GPU计算在Pod内。源文档、资源、环境和存储见[ds操作记录](../docs/DS_SERVER_OPERATIONS.md)。D19先学习运维，D20已建立ds本项目checkout并启动数据获取；没有创建/删除Job、改旧项目、安装依赖或训练。

用户home和data10为Pod持久NFS挂载，`/dev/shm`为64Gi临时内存盘；scr清理政策及个人存储配额未核，不当永久唯一存储。Pod默认Python不等于zhou：默认Torch2.6/cu126在Blackwell上tiny CUDA失败，现有zhou Torch2.12/cu130两卡均通过。正式TWM依赖仍待验；不改旧项目环境/数据。

当前Job从2026-10-12 16:37:28 UTC起最多3天，按配置约10-15 16:37 UTC到期，需checkpoint持久化与续跑；退出exec不释放GPU。本轮复用用户现有Job，不按旧模板自动删除它。未来另建本项目Job时使用独立名称、label与本项目配置。

## 运行与复现

每次实验记录本地已推送commit、远程实际commit、配置、数据版本/split、命令、设备、随机种子与产物路径。当前通过Kubernetes Job/Pod运行，长期任务在持久挂载保存checkpoint/日志，并考虑Job时限；D23已在zhou通过完整TWM数值检查并启动Oak训练；checkpoint持久保存，实测中断恢复尚未验。用户现有两卡已申请，不再为本轮学习新申请资源。

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
