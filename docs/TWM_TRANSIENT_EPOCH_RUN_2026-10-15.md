# D36：短时错误增广修复与两轮训练

**后续状态（2026-10-15）：** 本轮已于纽约03:53正常完成2/2epochs；用户随后改为[D37从头10k及历史best-only](TWM_TRANSIENT_10K_RUN_2026-10-15.md)。此run的best.pt保留，latest/暂停快照已按授权清理。下文“已启动/旧入口/仅0.08epoch”保留为启动当时记录，非当前仍在训练；[完成审计](../reports/twm_transient_epoch2_completed_20261015.json)。

2026-10-15，用户明确要求修复噪声生成及相关问题，重新运行，以epoch而非任意万步为预算，先设两轮、允许中途保存停止。D34已于06:28:48 UTC保存停止在5,519步；原latest/best/STOP保留在原目录，CPU审计322个model/963个AdamW浮点张量finite、两rank与CPU hash一致（44a930a53cc5942a80f12cfa901b26bc055c560acef890249e17adb4a602f8d0）。这替代D34继续10k的执行安排，不修改D35视觉插件框架。

## 修改依据及范围

[冻结短时诊断](TWM_TRANSIENT_DRIFT_AUDIT_2026-10-15.md)已证实：64条参考轨迹上，旧5000步模型把连续四帧30mm腕偏移几乎原样留下（29.972mm）；单帧也29.913mm。旧sampler没有显式的正常→局部持续偏移→恢复时间线，而是整窗历史共用severity。>10mm阈值仍能偶然出现3/4帧run，不能声称所有短异常均未覆盖。位置输入尺度和future目标牵制是其他待验因素。

本轮组合修复：新的时间事件增广＋内部位置feature归一化＋真实epoch遍历。不是单因素消融，不用新旧总loss证明哪项有效。当前恢复/长预测目标及权重、13,530,032参数架构、99D接口、四库GT/split、关闭beta conditioning均保持。没有添加视觉特征或oracle异常标签。未来梯度隔离/current-only仍未实施。

## 事件与幅度

源码[temporal_corruption.py](../src/handloop/data/temporal_corruption.py)，[active profile](../config/twm_corruption_transient_v3.json)。保留五类窗口20%全净、20%历史异常后当前恢复、35%普通错误、20%缺失、5%少量局部强错误；这些比例仍是工程假设。

- 时间模式：30%逐帧jitter、45%固定方向和幅度plateau、15%ramp、10%分散错误；1/2/3/4帧时长概率15/15/35/35%。ordinary/missing/strong覆盖当前处于异常起始/中段/末尾，reanchor覆盖异常结束后1–4帧。jitter在最后10帧按0.3激活，与episode时长分布不同。
- 普通事件空间：腕translation-only /腕p＋root /一两个手指/全手，各25%。强事件只在腕/root或一两个手指；case probability为5%，不是所有候选误差超过阈值的比例。
- 普通峰值轻1–6mm或中6–15mm（中40%）；强20–50mm。root轻0–4°/中3–10°/强10–35°，finger轻0–3°/中2–8°/强8–25°。ramp实际瞬时幅度可低于峰值档下限；所有旋转在SO(3)复合，不对6D串直接加噪。
- 非事件帧/部分为全净或near≤1mm/1°；每窗各半。全净case精确无噪。新误差替换背景，不相加多个完整偏移。各group事件向量独立，plateau在事件内保持。
- 缺失沿时间事件mask作用，history availability也可false。候选和参考系只用最近可用过去观测因果填充，不能偷偷读缺失GT；首个历史状态各group可用提供初始种子。raw/history中的缺失值是占位符，candidate单独填充。
- available quality保持unknown，不给severity/event mask作为model输入。future和current源标签不加噪，只转换到共同的、由可用历史定义的参考系。

事件当前仅在独立窗口内连续；重叠训练窗口仍各自抽样，未实现按整录制持久化误差，也未进行自身输出rollout训练。验证包括固定single/burst3/burst4/position-only burst4/scattered3/ramp4、clean/reanchor/missing/history-only、legacy stress；后续须复跑相同冻结连续clip控制，不能把固定窗口改善替代闭环恢复。

## 位置尺度

新增`position_feature_scale=0.05`：仅将encoder token的p除以0.05m，外部99D/GT/anchor/output/integration及概率单位仍为米，rotation不变。默认1.0保证旧配置行为；没有新增参数。0.05m借用现有position loss的特征尺度，是工程起点，不是已证明最优值，不能保证这项改动单独解决腕部问题。

## epoch与运行协议

[两轮配置](../config/twm_four_sources_moe_transient_v3_epoch2.json)。启动时先采用已说明的完整遍历默认；用户随后于2026-10-15异步回复明确确认“每个窗口一次，先跑2个epoch”，与已启动配置一致：每轮四库窗口各一次，双卡共享同一shuffle permutation，disjoint shard、最后不足batch保留。不是旧equal-source有放回采样的换名。

| 来源 | train窗口/epoch |
|---|---:|
| OakInk2 | 346,860 |
| HOT3D | 132,008 |
| UmeTrack | 151,390 |
| DexYCB | 6,352 |
| 合计 | 636,610 |

这使Dex占约1%、Oak约54.5%，改变了旧等源25%训练权重；validation仍分源报告并等源宏平均。本轮定义为pilot，不能保证两轮收敛。每卡512/global1024、每轮622次optimizer更新、最后每卡353样本；两轮1,244更新/1,273,220样本使用，远少于旧10k的10.24M抽取，不能按相同步数判断公平优劣。

LR2e-4、warmup0.1epoch自动推导63次更新，cosine按两轮结束降至10%LR；每0.5epoch及epoch末验证，checkpoint每100更新/epoch边界/协同STOP。日志记录按实际窗口累计的epoch_progress、completed_epochs、windows_seen，不把不足整批次按1024多计。W&B仍保留optimizer_step并同时记录epoch轴。

Epoch sampler保存epoch/batch cursor，resume先恢复相同permutation再跳过已消费的index（跳过发生在dataset IO之前）；num_workers=0，rank RNG保存。一般奇数N时DistributedSampler至多补world−1索引，metadata记录；本轮N可被2整除，无padding/重复/丢弃。

远程仅复用已授权Pod与GPU0/1，Torch2.10/cu130＋官方Mamba runtime；不创建或删除Job。新输出`data10/.../runs/twm_four_sources_moe_transient_v3_epoch2_20261015`，旧run不resume。Pod16:37 UTC到期，16:25UTC保守截止仍有效。正式启动源码版本/语义检查/DDP更新、续跑和checkpoint数值同步审计见后续验收摘要；此处初版不宣称训练已启动或精度改善。


06:44UTC：SSH短暂被入口拒绝后恢复；原远程checkout的pull因另一聊天两份untracked视觉源码而安全拒绝。保留原目录，改用独立detached源码worktree `data10/.../code/twm_transient_v3_20261015`；训练与launcher显式指向这里。新验证的history-hold/constant-velocity基线对missing history只做因果carry-forward，避免把placeholder当有效运动。


## 远程验收（正式效果尚待训练）

- 13个v3模式各1,024抽样，用四源各一个真实train reference重复，位移范数/SO(3)角与profile一致。mixed有28个>6mm、两侧较准的历史三四帧run，reanchor有79个；相较旧同阈值的零覆盖，证明新sampler产生预期事件，不能说明已拟合实际HaWoR。
- 修改future数值不影响任何输入/当前标签；修改被隐藏的history/current GT不改变history/raw/candidate；available quality为unknown，beta conditioning关闭。公开状态仍是米，encoder position feature扩大20倍，旧默认行为保留。
- 对实际636,610个index验证两rank disjoint且并集完整，最后每卡353；permutation和batch cursor保存/恢复检查通过。
- 初次检查错误地用1.1mm数值容差同时比较position和rotation6D，reanchor的near旋转因此被拒绝；改为分别按位移/SO(3)实际角检查后通过，未为此放宽或修改噪声生成。失败记录保留在本报告。
- `smoke_temporal_epoch_training.py`从每源各一train/val录制取42连续参考帧，只写remote独立测试目录；每源3窗，12train窗，global8，最后global4。第一段1更新后保存，resume到4更新/2完整epoch/24窗口使用，实际经过尾批次及epoch重置；两段模型hash不同、AdamW更新到4，各source总计6。另一个fresh smoke在第1次更新后STOP，双rank共同保存退出。所有checkpoint模型/AdamW finite、CPU与两rank hash一致。没有将24样本smoke用作精度结论。

语义检查源码9e48955（后续25fee71未改对应runtime文件）；双卡smoke源码25fee71。远程产物在`data10/.../audits/transient_v3_epoch2_20261015/semantic_checks.json`、`ddp_smoke/summary.json`及各审计/日志/checkpoint；所有原始短片段仅远程保留。正式启动验收将在实际更新后补充。

运行入口（ds登录节点；训练实际在原Pod）：

```bash
python3 /sciclone/data10/jchen57/Hand_Reconstruction/code/twm_transient_v3_20261015/scripts/ds_train_probabilistic_twm.py status
python3 /sciclone/data10/jchen57/Hand_Reconstruction/code/twm_transient_v3_20261015/scripts/ds_train_probabilistic_twm.py stop
# 用户明确要继续已停止的新run时才执行；保留旧STOP归档后恢复同一run
python3 /sciclone/data10/jchen57/Hand_Reconstruction/code/twm_transient_v3_20261015/scripts/ds_train_probabilistic_twm.py start --resume
```


## 正式启动通过（06:56:30 UTC；纽约02:56）

源码25fee71a1fe1f0347054153b2135f0d018f63068，06:52:11 UTC启动，launcher6447、rank6452/6453。06:56验收实际50更新/0.080426epoch/51,200窗口，loss/grad finite，四源与五case均正；两卡peak allocated63.924/63.913GB。warmup自动63次、总1244次为两轮派生值，非用户另定步数预算。正式latest10模型322/AdamW963浮点张量finite，CPU与两rank完整hash=`238f3e37899da065883fe8070a9990eff383f7103792bc373f8343d3438e0f74`。metadata记录的源码文件SHA均与独立worktree匹配。

[W&B在线新run](https://wandb.ai/black-ant/handloop-twm/runs/58f012f8)服务端已实际返回10/20/30/40/50训练标量及epoch_progress，默认横轴设为epoch_progress；本地log存在不是唯一验收依据。完整小型[启动验收摘要](../reports/twm_transient_epoch2_20261015.json)含配置、数据/代码hash、初始验证几何、语义/续跑/停止及CPU数值审计。原始数据、短片段、权重和全日志均仅远程保存。Source GT仍为拟合MANO，坐标单位与原prepared数据及共同anchor契约不变。

注意日志temporal/spatial pattern计数是窗口模式抽签，含全净case的未施加模式；strong空间部分另有局部override，不能把这些计数直接当作实际坏关节/坏帧数。旧mixed_v2及既有checkpoint原样保留，后续相同连续clip与真实HaWoR验证仍待做；本次组合修复尚未构成单因素因果证明或最终性能提升证据。


用户随后确认其可登录ds。先前入口短暂拒绝已在正式启动前恢复，并非当前阻塞；本次确认无需重启或修改已运行训练。
