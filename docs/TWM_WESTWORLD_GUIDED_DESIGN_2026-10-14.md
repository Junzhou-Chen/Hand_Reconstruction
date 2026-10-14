# 参考 WestWorld 架构与配置、适配手部当前恢复

2026-10-14。用户先要求暂停原pilot，随后澄清：不是完整照搬WestWorld，重点参考架构和参数配置背后的依据，输入输出仍配合本项目任务。该方向是用户决定；下文具体结构与超参是可审阅的实施建议，尚未实现或启动新训练。

## 已完成的暂停

纽约05:16（09:16 UTC）核验PID805身份并发送SIGTERM，原训练保存退出，phase=stopped。只读加载检查点确认latest.pt为28,213步，best.pt为27,500步，代码均为4cb1da83cc86732b36cea76673bb1e703eae4102。原Pod/Job、数据与结果保留。用户主动暂停不当成故障，不自动resume或另启新run。[完整运行记录](TWM_OAKINK2_PILOT_RUN_2026-10-14.md)

## 架构参考的依据

复核[WestWorld v2 §4、§5.2、附录D](https://arxiv.org/html/2603.14392v2)及官方HEAD `0a90776aeaa43f47c97f3c05eb75418b377b73ae`。论文将同帧状态通道交互和逐通道时间建模分开，加入运动树结构编码；结构编码有删除消融。Mamba负责时间序列，Sys-MoE针对多机器人动力学。6blocks/d256/4heads是报告的配置，未找到各项独立最优性证明；bins/专家数有精度与成本比较，但不是手部恢复证据。

官方[block源码](https://github.com/511205787/WestWorld/blob/0a90776aeaa43f47c97f3c05eb75418b377b73ae/models/WestWorld/westworld.py#L263)实际顺序是同帧state-channel self-attention → action cross-attention → dense FFN → 每个state channel沿时间的Mamba → system/CLS路由与expert混合。不是将所有时间/通道展平的普通Transformer。原pilot正是展平T×16后联合attention，仅有joint-ID embedding，因此原pilot不能称作已采用该主干。

[结构实现](https://github.com/511205787/WestWorld/blob/0a90776aeaa43f47c97f3c05eb75418b377b73ae/models/WestWorld/westworld.py#L586)把运动树的索引embedding加入所属通道，当前spatial_mask=None；它不是硬约束求解器。[Mamba/MoE源码](https://github.com/511205787/WestWorld/blob/0a90776aeaa43f47c97f3c05eb75418b377b73ae/models/WestWorld/mamba_moe.py#L138)的末尾CLS汇总整段再路由，故Mamba扫描因果不证明整套block严格流式；迁移时应重新审计信息边界。

同时保留已有任务对照：[DIP §3](https://arxiv.org/html/2508.01835v1)提供受损运动条件恢复的训练先例；[StableHand §3](https://arxiv.org/html/2605.18553v1)提供quality条件恢复机制，但其生成器使用视觉条件。它们用于检查任务和证据设计，不把WestWorld未来预测或StableHand视觉生成任务替换成我们的任务。

## 建议保留与替换

| 部分 | 建议 | 理由与状态 |
|---|---|---|
| 输入接口 | 保留独立H/raw Z/candidate S、逐组availability与quality、dt、β、handedness | 符合现行状态/测量分离；quality不能当真值 |
| 状态与输出 | 每手完整99D，β另存；当前恢复主头、短预测辅头 | 用户已定任务；不采用机器人action预测 |
| 数值编码 | 保留连续旋转/位置编码，训练统计只在train计算 | 99D含16个SO(3)，不把96旋转坐标当独立自由变量 |
| token | 第一候选继续16解剖token，腕含p3/root6；17可用性组独立 | 是我们的任务适配，不是WestWorld原标量token复刻 |
| 结构编码 | 补实际MANO树对应的finger/depth/parent信息，保留joint identity | 先核源joint顺序；LCRS作为结构编码候选对照，不自动增加GCN或硬mask |
| 同帧交互 | 每帧joint self-attention，保持整手关节可交互 | 让当前可信局部信息影响其它部分；是否提高恢复必须实测 |
| 时间交互 | 每关节独立沿时间的共享dense causal SSM/Mamba | 更接近所参考的时空分工；不用跨机器人expert路由 |
| 状态读出 | 内部51D切向更新，SO(3)复合后输出99D | 延续已核数值接口；不是独立Corrector，也不因此自动删除最终视觉框架的C |

候选张量保持`[B,T,16,d]`，每个block在关节轴聚合，再转为`[B×16,T,d]`沿时间处理，最后恢复原布局。当前末帧输出恢复state，future head只读可用上下文与horizon。样本仅提供过去与当前信息；训练监督对齐同一时刻，future GT只做辅助目标。具体norm/residual顺序以新实现说明和数值验证为准，不称整套block照搬。

连续头与上述主干兼容，不要求使用256-bin CE。[官方数值编码](https://github.com/511205787/WestWorld/blob/0a90776aeaa43f47c97f3c05eb75418b377b73ae/models/WestWorld/westworld_utils.py#L14)输入为Gaussian soft bins、目标one-hot，输出取bin中心概率期望；不能错误声称其输出一定有一bin硬精度下限。对我们的6D旋转，独立分量分类不能自动保持旋转约束，需几何解码及角度/joint评价，因此第一候选继续几何回归。

## 容量与参数：有来源的起点，不是最优结论

| 参数 | 原pilot | 下一候选起点 | 依据及限制 |
|---|---:|---:|---|
| block数 | 4联合Transformer层 | 6时空分离block | WestWorld配置参照；block不同，不能只按层数比容量 |
| hidden dimension | 128 | 256 | 配置参照，增加表达容量；手部收益未验证 |
| attention heads | 4 | 4 | d256时每head64；不是head数越多越好 |
| dense FFN hidden | 512 | 512 | 配置参照，在d256下为2×expansion |
| dropout | 0.1 | 暂0.1 | 当前YAML与D.1一致；论文D.3写0.05，须明确实际运行值 |
| SSM state size / conv / expand | 无 | 64 / 4 / 2 | 仅Mamba候选配置；内部state64与手部99D无对应关系 |

上述值来自[固定官方配置](https://github.com/511205787/WestWorld/blob/0a90776aeaa43f47c97f3c05eb75418b377b73ae/configs/method/WestWorld.yaml)。它包含不同阶段设置，不能把整份YAML当论文pretrain协议；batch、lr、warmup、总步数和历史长度须按本项目数据量及验证选择，不预设百万步或50/100帧预测适合手部任务。lr=2e-4可列为候选，但本轮未设置新run。

实施前需在既有Pod/zhou验证Mamba依赖、Blackwell kernel兼容、BF16/float32数值与backward；没有运行证明前不宣称支持，也不静默换回展平Transformer后沿用名称。若用分离temporal attention作为对照，明确称为时空分离attention版本。

## 训练继续服务当前恢复任务

1. 第一轮结构对照沿已有Oak官方64train/8val子集、名义30Hz与同一corruption/完整参数loss，对比原展平模型、结构＋时空分离attention、结构＋dense SSM。控制训练预算与容量，避免把增大宽深的收益全归于结构。没有承诺必须全部组合穷举。
2. 验证新增结构需要与joint-ID、同容量无结构版本分开；SSM需要与同容量时间attention分开。保留clean/noisy/missing和未校正candidate，不只选有噪声最好的一列。
3. 结构跑通后再逐项加入连续多帧缺失、时间相关偏差、假高quality、污染历史与自身输出反馈；更换corruption单列版本。不是WestWorld遮住全部未来通道的原监督。
4. 位置与旋转loss继续主任务，短预测辅助权重先沿pilot记录但仍待验证。dt使用秒、anchor来自可用历史；embedding含dt不等于SSM已严格实现真实时间动力学。
5. MANO资产/FK具备后补joint几何验收。现有腕mm/旋转°相对拟合参考不冒充MPJPE或物理保证；真实HaWoR输出适配和glove部分监督另需实现。

当前只完成暂停、论文/源码复核和本建议记录。原模型代码保持，未把6/256/4、树embedding或Mamba写成用户逐项已选，也未开始新训练。最终视觉Corrector/直接TWM取证方案仍按之前讨论保留未决。
