# D36：短时错误增广修复与两轮训练

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

[两轮配置](../config/twm_four_sources_moe_transient_v3_epoch2.json)。询问epoch口径后，未收到其他选择时采用已明确推荐的完整遍历默认：每轮四库窗口各一次，双卡共享同一shuffle permutation，disjoint shard、最后不足batch保留。不是旧equal-source有放回采样的换名。

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
