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

已于**08:25:37UTC／纽约04:25**提交正式双卡任务，源码固定`a642e6eb55e320fa5cae89d5b8565dff9d43efe7`，torchrun PID4581、rank workers4594／4595。初始化阶段较慢，08:32日志已进入HaWoR／MANO构建，两rank仍存活；此时尚无参数更新，不把任务提交称为训练验收完成。

新启动源审计保存在`evaluations/visual_dex_hot_geometry_5k_20261016/prelaunch_source_audit.json`：视觉Dex800条train均为s0 train、与保存的test16物理录制不交；HOT train与27个eval父录制不交；整个视觉train／val物理组互斥。motion回放的Dex s0 test排除与HOT27排除均重查通过。原67182文件完整SHA验收保持，本轮未修改其任何源数据。

08:27后台快照：ARCTIC仍running／verified37／failed0；HO3D已转checking，尚未完成CRC／SHA；HOT全clips预处理960/1167。原进程、目录和配置均保持。观察到Pod接近200Gi内存账面上限主要为文件cache，实际匿名rss约2.5Gi（训练加载后上升），未因此停止下载或清理全局cache。

后续应记录实际更新、两rank梯度／代表参数改变、checkpoint完整性与在线W&B。仅进程存在、初始化指标或train loss不构成新的平均纠正收益。

## 正式启动验收通过

**08:40:39UTC／纽约04:40快照：training 30/5000**，新[W&B ff3108fd](https://wandb.ai/black-ant/handloop-twm/runs/ff3108fd)。模型708688515个可训练参数，两主干及插件均保持可训练；实际视觉来源仅DexYCB／HOT3D，2773 train窗／143 val窗，配置5000和每卡8已在本地run metadata及服务端确认。

- 两rank首8更新14组均累计出现非零梯度、所查gradient finite、14组代表参数均改变；这是跨视觉与motion任务累计覆盖，不是每个task同时训练所有分支。每卡peak allocated68173757952B＝63.4983GiB。
- 第4步checkpoint8510562607B已原子保存并硬链接留作验收；CPU加载检查1076个model／3135个Adam浮点tensor全finite，与保存的两rank模型hash均相同：`13141b18b9667f4b51e67e598d7359f83bffc9fac4fcaa8412a4ea0bd7917dd7`。未复制大型权重到本机或Git。
- W&B服务端08:37已收到step10、state=running，配置steps5000／rgb8／两来源正确；该时间早于本地step30快照，不能称同步瞬时进度一致。
- 源HaWoR权重SHA `4d1cc43853c190d6f2c10d9b6295c73109f0faf9ef41ac817a2b31d94b4823f2`，D37权重与motion清单SHA见小型报告；与旧run采用同一资产身份、全新Adam。
- 初始化fixed8结果仍完整保留（未训练的最终腕大于原raw），未据train batch或初始化作效果结论。正式周期验证从step50开始，后续再用同输入比较baseline。

启动与数据隔离检查全部通过，详见[小型摘要](../reports/visual_dex_hot_geometry_5k_launch_20261016.json)，远程完整检查在上述evaluations目录。训练退出SSH／本聊天后继续，按每50步既定验证／保存安排运行；旧best与100步旧run均未覆盖。

### 并行数据准备状态

08:40快照：ARCTIC仍running，verified51/343／failed0、已写约26.24GB。HO3D官方v2包已于08:30:22UTC完成37,044,700,091B获取、ZIP CRC与自算SHA；这是完整下载验收，不是该库99D／相机adapter已接入。HOT全部已下载1167 Aria clips的RGB准备08:37:57UTC完成，train275884／val38848个有效手状态、全split独立RGB174659张。它们继续留作下一版本准备，**没有改变正在训练的73524状态／63652独立RGB清单**。
