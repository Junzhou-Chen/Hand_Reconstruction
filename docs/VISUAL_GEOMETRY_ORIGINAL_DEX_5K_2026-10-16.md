# 原始Dex数据：直接几何监督，重新训练5000步

日期：2026-10-16。用户新指令替代上一轮“扩大数据＋直接几何、1000步”的执行安排：**先取消数据扩展，只先检验第2项几何监督，重新训练约4000／5000步。** 实现选择5000总optimizer更新作为上限，可按实际验证结果再讨论中途停止；不自动恢复已停止的混源run。

## 数据和初始化

- 原混源run `runs/visual_feedback_joint_v3_geometry_20261016`已于07:14:26UTC正常stopped100/1000；checkpoint100已保存，两rank模型hash `3fff5e8f8249faaa4949bddaac88eaa7328e26906614fc068846a0164f79ab81`相同。数据、best／latest和日志保留，不删除扩展资源。
- **视觉恢复原数据**：`prepared/visual_feedback_dex_joint_sources_v2_20261016/source_manifest.json`，SHA256 `67462dced45615c9e0ea8d761da58094d03f8b8cadc47370c06f126751259d3e`。train16录制／1128不同RGB帧／46个32帧名义窗口；val8录制／577帧／24窗口。两主体、左右手、原GT crop／固定camera gauge、米制及源拟合MANO参考均不改。此次既不使用扩大后的Dex train，也不使用HOT RGB或新native glove样本。
- 原四库motion预训练与回放保持：OakInk2／HOT3D／独立UmeTrack／DexYCB，D37 best10000不重训；它们是既有motion池，不属于本次撤回的新增视觉覆盖。原v3受损分布、现行3:1任务与库间等概率回放不改。
- 新run从**官方HaWoR＋原D37 best＋新插件**开始；seed20261016、全新AdamW和scheduler，不加载旧D40 best300，也不加载混源100步联合权重。

## 第2项如何监督

保留现有99D状态、预测β10、可微MANO和两轮视觉取证，不新增模块。视觉目标为：

\[
L_v=10^{-4}L_{final,NLL}
+ \operatorname{mean}\|p^{raw}_{J0}-p^{ref}_{J0}\|_2/(0.05m)
+ \operatorname{mean}_{valid\ joints}\|J(\hat s,\hat\beta)-J^{ref}\|_2/(0.05m).
\]

1. **raw腕**直接约束前端绝对定位，避免仅依赖后端补偿前端偏差。
2. **final关节**约束最终状态及其自身预测shape的真实解码几何，与最终绝对MPJPE目标一致。几何项不受预测σ／κ调节；不能通过增大方差降低该项。
3. **NLL**继续约束完整状态与概率输出；平移Gaussian、旋转SO(3)上的matrix Fisher，非冗余99D坐标的简单Gaussian。

没有腕平移、PA或尺度对齐；参考β不作为网络输入，也不强行对跨codec的原始β做L2回归。没有添加raw手指、2D、速度、shape参数、平滑或所有中间轮次监督。三个项分别按有效state／腕／joint求全局mean，padding和缺失参考不参与；GT仍为原source拟合参考，不称独立传感器真值。

**本轮保持上一版几何系数，不再同时调整权重。** 这些系数来自原Dex train清单左右各两段8帧的初始化梯度量级检查；对应的正是本轮恢复的允许train来源，不依赖新增HOT或扩大Dex的标签。`1e-4`与0.05m不是论文最优结论，也不是本次5000步收益的保证。旧probe／双卡归约／masked NaN／预测β／GT隔离与真实全主干backward验收见[上一版协议](VISUAL_DATA_GEOMETRY_V3_2026-10-16.md)。模型和loss代码不变，不重复跑已通过的相同合同测试。

## 训练预算和保持项

配置：[visual_joint_geometry_original_dex_5k.json](../config/visual_joint_geometry_original_dex_5k.json)。相对上一个geometry配置只改`steps=5000`、`eval_batches=24`及tracking；实际source由launcher显式指定原manifest。

| 项目 | 本轮实际设置 |
|---|---|
| 设备／视觉batch | 同授权Pod，GPU0／1；每卡8段×32名义帧；尾batch每卡7段 |
| 总更新 | 5000，含3750次视觉＋1250次motion；不是追加5000步 |
| motion batch | 每卡32窗／全球64，满预算80,000次窗口抽取；不当独立新增样本 |
| 原视觉池重复 | 每3个视觉更新完整走一轮，3750视觉更新约1250轮；独立RGB仍1128帧 |
| 可训练范围 | HaWoR与TWM全量低LR，插件全量；MANO固定几何资产 |
| 峰值LR | HaWoR1e-6／TWM1e-5／插件1e-4 |
| scheduler | warmup20个总更新；按新5000步从头cosine到10%峰值LR |
| 优化／精度 | AdamW、weight_decay0.01（沿用参数例外）、三模块分别clip norm1、BF16、gradient checkpointing |
| 历史／轮次 | history30、每clip空历史开始；跨帧detach、帧内两轮可微；仅末轮commit |
| 噪声与回放 | 原D37 v3 profile及原联合3:1更新顺序保持 |

长训练会重复访问原小池，不会凭更多step产生新姿态。5000上限是用户允许预算内的工程选择。新cosine分母与原1000不同，因此本轮“几何＋更长schedule”不是对loss的严格单因素消融；不能把后续所有差异只归因于新增几何项。

## 验证与保存

验证数据完全回到原577帧。**实现选择每50步完整检查原24个val窗口**，替代先前只抽8窗口；这是同一原val集合内的检查覆盖改变，不引入新训练标签，也不改GT／split。完整验证有助于观察小池长训练时的波动和分侧退化；仍不能证明跨主体、跨数据集或完整ego部署泛化。

best继续沿用原`state_geometry_loss`：位置SmoothL1（0.05m尺度）＋2×root rotation chordal＋平均finger chordal；final绝对MPJPE、raw绝对腕及NLL独立报告，没有默默换成按test挑best。原step0基线单独保存；每50步保存latest／改善时best，原第4步先保存供启动审计。保留原四库各16窗、noisy／clean的当前／未来小保持集。

训练结束或用户中途暂停后，在同一原Dex val577／test1015协议上与官方HaWoR和旧best300配对，报告整体、左右侧及尾部。取消扩展数据不等于取消后续独立评测；不按test选择权重。

## 执行状态

旧混源run已正常保存停止，原数据SHA、train46／val24窗口及双卡空闲已核验。新5000步配置已准备，等待固定源码部署与实际启动验收；尚未宣称新run已更新或效果改善。
