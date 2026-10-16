# 已验收DexYCB＋HOT3D：几何监督联合训练5000步

日期：2026-10-16。用户最新决定优先看实际效果，不要求逐帧复现HaWoR，可在不泄漏测试数据的前提下增加来源；**现在先用此前已经验收并跑过的DexYCB＋HOT3D池重新跑5000步**，ARCTIC／HO3D下载与HOT更大范围预处理继续后台进行。

本条替代“必须等HaWoR四库准备完再开训”的启动条件及之前未启动的tiny-Dex-only 5k方案；数据扩展为后续版本，本run保持输入清单不变，不在运行中自动吸收新下载数据。旧混源100步和旧D40 best300／last1000保留。

## 初始化、目标与预算

- 从官方HaWoR＋D37 `best.pt`重新初始化，新插件同seed、全新AdamW；不从旧100步权重或optimizer恢复。5000步从头安排warmup与cosine schedule。
- 配置：[`visual_joint_dex_hot_geometry_5k.json`](../config/visual_joint_dex_hot_geometry_5k.json)。双GPU，每卡8×32 RGB帧／32 motion窗口；HaWoR／TWM／插件LR=1e-6／1e-5／1e-4，weight decay0.01，warmup20，cosine终点10%，各模块grad clip1，BF16及gradient checkpointing。
- 5000为总optimizer更新，原3:1顺序对应3750视觉＋1250纯轨迹回放；视觉DistributedSampler保持，motion原四库等概率有放回保持。不改噪声、raw／candidate条件、history30／两轮取证和跨帧detach。
- 两个主干低LR全量训练，TWM仍只接数值；每手99D，beta10另存。视觉目标仍NLL×1e-4＋raw J0绝对腕距离/0.05m＋final自身预测beta的有效绝对joint距离/0.05m；无PA或尺度／腕对齐，不加新loss。
- 原fixed8验证每50更新，best按原state geometry score选择；其4037状态的完整val池中只选143窗的固定8窗，不称全量val。保持与旧混源1000计划相同的验证身份／算法，方便后续看曲线；不依据test选best。

## 冻结的数据范围与参考

持久根：`/sciclone/data10/jchen57/Hand_Reconstruction`。

视觉清单：`prepared/visual_dex_hot_collection_v3_20261016/source_manifest.json`，SHA256 `6314ec2aea911709595c31e262a55ffa2831d14eb2aa5299810287c60b6a5db3`。

| 数据 | Train有效手状态 | Train32帧窗 | Val有效手状态 | Val32帧窗 |
|---|---:|---:|---:|---:|
| DexYCB | 50861 | 1980 | 577 | 24 |
| HOT3D | 22663 | 793 | 3460 | 119 |
| 合计 | **73524** | **2773** | **4037** | **143** |

Train独立RGB共**63652张**：Dex50861＋HOT12791，HOT双手共享RGB不重复计数。972条train／33条val单手record。global16 clips、每rank1387个sampler样本（DDP每轮最多补1窗），174视觉更新约遍历一轮；3750视觉更新约21.55轮，并非5000个epoch。

Dex源为官方拟合MANO及Dex21，在固定相机坐标作明确world gauge；HOT为源PCA／beta按Meta约定解码，Meta21缺拇指base槽位，按mask监督；单位均米。GT crop与已知相机条件保持，不声称独立MoCap精度、完整benchmark或真实检测／SLAM部署评估。

既有验收`evaluations/data_expansion_geometry_v3_20261016/visual_collection_acceptance.json`已覆盖67182个RGB／参考文件严格SHA、源与hardlink身份、真实crop和原Dex val577字节不变。本次重启核对完成manifest SHA与既有验收，并重查metadata物理录制边界／s0 test排除／HOT27评测原录制排除；不为改变训练步数重做未变的全部重编码和全模型测试。RGB loader依然在实际读取时核验各源参考与图像身份。

纯运动回放清单仍`prepared/twm_four_sources_99d_v1/manifest.json`，OakInk2／HOT3D／独立UmeTrack／DexYCB，使用D37内嵌原任务和受损配置。TWM资产SHA `9f500a283109fbad75f380281808b16781a335550b99853d66832cb120bef76c`。

## 运行位置与验收

授权Pod仍`jchen57-fineturn-tl4zr`，12CPU／200Gi／两张97887MiB Blackwell。沿已验证Torch2.10/cu130项目runtime，HaWoR extras与合法MANO复用。所有数据检查、训练、梯度、checkpoint及数值聚合远程执行；本机仅源码／文档／静态检查。

新不可变代码目录：`code/visual_feedback_dex_hot_geometry_5k_20261016`；新输出：`runs/visual_feedback_dex_hot_geometry_5k_20261016`；小型验收：`evaluations/visual_dex_hot_geometry_5k_20261016`。既有训练不覆盖，下载及预处理PID／路径保持。

当前为新协议准备；正式启动后应记录实际更新、两rank梯度／代表参数改变、checkpoint完整性与在线W&B。仅进程存在、初始化指标或train loss不构成新的平均纠正收益。
