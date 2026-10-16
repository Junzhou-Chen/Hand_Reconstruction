# D39：HaWoR／TWM 全量联合微调 pilot 协议

记录日期：2026-10-16 UTC（纽约 2026-10-15）。**状态：协议与实现准备中，尚未记录本轮启动或效果；下列预算、学习率和混合比例是工程 pilot 设置，不是已验证最优值。** 实际代码版本、数据规模、启动验收及结果由运行记录补充。

## 用户决定与保留的边界

用户本轮明确指出：只冻结两个主干并训练新插件不能充分协调三个模块，希望 **HaWoR 与 TWM 都以低学习率全量微调**，使用已经完成的 TWM 预训练结果，并把当前讨论的其余框架能力先实现、运行。此决定替代 D35 视觉 smoke 的双主干冻结执行范围，以及此前“先固定 HaWoR”的建议。它不等于修改 D37 预训练结果或认定原六项 loss 已解决。

保留每手 99D 动态状态与独立 shape β10；TWM 只接收数值状态，不读 RGB、图像 features 或相机 feature tokens。原始观测、当前迭代候选和已完成历史继续分槽。原始观测 `Z_t` 是本次 HaWoR forward 的结果，同一帧的迭代中不被覆盖；优化器更新后下个训练 forward 的 HaWoR 输出允许变化。**raw 不可覆盖不意味着 detach，也不意味着冻结 HaWoR。**

本轮允许两轮候选引导的视觉取证／恢复，同一帧内的历史与参考系固定，只有最后一轮的最终当前状态提交到历史。自身历史跨帧 detach，GT 不能写入历史。此设置不宣称已经实现独立可靠性 w／q 的训练与校准，也不增加物体物理或规划模块。

## 模型与梯度路径

训练必须读取 RGB，在线运行官方 HaWoR 的 ViT、IAM、MANO head 和 PAM；不再用冻结 patch features 缓存冒充 HaWoR 全量训练。原 D35 缓存可用于历史基线核对，不是本轮视觉特征来源。

每帧的执行意图为：

1. 原 HaWoR 16 帧路径输出真实 patch features、原始状态 `Z_t` 和预测 β。
2. 在固定的历史和参考系内，视觉插件利用候选几何读取图像证据，生成候选；TWM 结合原始观测、候选和已完成历史恢复当前状态。
3. 以上一轮恢复结果引导下一轮视觉查询，共两轮；只用最后一轮的当前分布计算视觉目标，只在两轮完成后提交一次历史。

具体候选查询、raw／candidate 不覆写、SO(3) 复合与逐轮输入依赖须以本轮源码和远程梯度验收为准，不能仅凭该流程将计划写成已实现事实。帧内保留到两个主干和插件的可微路径；MANO 非学习几何资产保持固定。相机轨迹与检测作为外部输入，本轮不反向训练 SLAM 或 detector。

HaWoR 保留 16 帧双向上下文和其 lookahead，不能称严格在线。末块复制只用于原网络的 padding；对应帧不增加监督或历史。未来输出落在视觉 lookahead 范围内时，不能解释为严格因果未来预测。

## 三组优化参数

| 参数组 | 初始学习率 | 本轮范围 |
|---|---:|---|
| 新视觉反馈插件 | `1e-4` | 全部可学习参数 |
| 预训练 TWM | `1e-5` | encoder、MoE、当前与未来读出等全部可学习参数 |
| 预训练 HaWoR | `1e-6` | ViT、IAM、MANO head、PAM 全部可学习参数 |

以上是参数组学习率，不应因为采用训练／评估模式切换再次冻结主干。HaWoR 官方资产版本、TWM checkpoint 和三组实际可训练参数数目须写入 run metadata。adapter 零初始化读出可能使其前级在第一步梯度为零，故不能只用第一步判断全部图像梯度链路失效；应在后续更新中分别验收。

shape β10 仍另存。当前完整 state NLL 不直接监督 β；若 TWM β conditioning 关闭，β head 主要通过 MANO rest-wrist／J0 转换得到间接梯度。该通路不能充分辨识真实 shape，必须记录 β 漂移，不宣称本轮已经训练出准确 shape。不得把 GT β 塞入预测 forward 以规避此问题。

## 视觉更新与运动回放

使用确定、两 rank 一致的任务序列：**每 4 次 optimizer update 中，3 次视觉、1 次纯轨迹 motion replay**。两类任务分别 backward／更新，不将两种 loss 相加。`3:1` 是工程起点，任务比例仍然影响训练权衡，不称没有权重。

视觉更新保持单一最终当前目标：

\[
\mathcal L_{\rm visual}
=-\frac{1}{N_{\rm valid}}\sum_{t\in\mathcal V}
\left[\log p_{\rm Gaussian}(p_t^{\rm GT})
+\sum_{j=0}^{15}\log p_{\rm MatrixFisher}(R_{t,j}^{\rm GT})\right].
\]

平移以米为单位，在该帧固定 anchor 中计算密度；旋转以 SO(3) 单位 Haar 测度归一化。按完整有效 state 平均，**不再除以 51**。GT 只进入外部 loss 和 metrics，不能进入模型输入、候选、参考系或历史。DDP 两 rank 的完整有效 state 数可能不同，必须按全局有效数归约，而非无条件平均两个 local mean。

motion replay 沿用原 D37 train 分区、受损条件与原六项 objective，保持原来当前恢复及最长 60 步未来分布的训练路径。它不读取 RGB，不把视觉标签混作原四库轨迹数据，也不使用 val／test 进行回放。

| D37 原 replay 项 | 系数 |
|---|---:|
| 当前几何恢复 | `1.0` |
| 当前 NLL／51 | `0.05` |
| 长轨迹 mixture NLL／有效时长／51 | `0.3` |
| 最佳轨迹几何项 | `0.2` |
| 速度项 | `0.05` |
| MoE balance | `0.01` |

精确定义以 [D37 配置](../config/twm_four_sources_moe_transient_v3_b512_10k.json)和 [objective 实现](../scripts/train_probabilistic_twm.py)为准。这里保留六项是明确的旧任务回放安排，**不是把六项改称新的视觉 loss**，也不提前实施用户已延后到下一版本的 loss 重设计。只做视觉 current NLL 时，future 专属路径没有相应监督；全量低 LR 本身也不保证未来能力保持。

## 数据与运行预算

准备使用已经下载的 DexYCB，沿官方 s0 recording split，在窗口化前固定训练／验证录制：

- subject-01 和 subject-02，各选 8 个 train、4 个 val 录制，即目标 16 train／8 val 录制。
- 每录制最多取 128 帧；总量上限 3,072 帧，实际可用帧与左右手分布待准备结果确认，不能把上限写成实测。
- 当前使用 GT 关节点投影产生 crop，使用该固定相机作为明确命名的 world gauge；时间戳为 frame index／30 的标称时间。
- GT 来源是官方 fitted MANO／joint annotation，不称独立 glove／MoCap 真值。源 β 与 HaWoR β 约定不得未经验证混用。
- 对帧间缺口分段，避免原 16 帧 HaWoR 块跨越不连续源时间；录制不能跨 train／val，测试录制排除。

该数据扩展仍是 **GT-crop、固定相机的联合训练 pilot**，不是头戴移动相机 SLAM、自动 detector 或完整 HaWoR benchmark 验证。D35 的 63 train／64 val 帧和 20 次更新只作为历史工程结果，不自动成为本轮基线。

初始预算为 **1,000 次 optimizer update**，按 `3:1` 序列约 750 次视觉、250 次 motion replay。使用双 GPU；视觉每卡 batch 1、clip 16 起，motion replay 每卡 batch 32。视觉 global batch 为 2 clips，motion global batch 为 64 windows；它们不是同一种样本计数。只有实际显存／梯度测试通过后才增加 batch，并在 run metadata 记录最终值及是否重启。

本轮 replay 初始实现使用四库等概率、库内窗口有放回的 `SourceBalancedSampler`；这与 D37 的 `full_coverage` 有限遍历不同。相同 train 分区、噪声和 objective 不等于相同采样协议；本轮不承诺遍历完整运动池。训练过程按 optimizer update 计数，实际视觉／运动样本暴露量分别记录。

D37 source 应取已完成 run 中固定、可核验的 best checkpoint，保留模型配置（包括 `position_feature_scale=0.05`），不能跟随另一个 run 的 `latest.pt` 变化。D35 的 combined checkpoint 内嵌旧 D32，不能把加载它当作已接入 D37。

## 远程启动验收

全部数值验收、RGB 推理、训练和数据准备在 `ssh ds` 的现有 Pod `jchen57-fineturn-tl4zr` 执行，本机仅代码、文档和轻量静态检查。沿用已验证的 CUDA／Mamba 环境；不得修改或删除用户 Job。启动前确认旧 D37 已结束与 GPU 可用，保存其 best 和来源。

必须检查以下可观察事实：

1. 在线 HaWoR 输出和 frozen 原实现的初始化等价；每个输入 RGB track 的时间、左右手、相机与 crop 坐标一致。
2. 同一帧两轮的 raw、history、anchor 固定；第二轮查询依赖第一轮候选；每帧最终状态只 commit 一次；padding 和跨录制 reset 正确；GT 改动不能改变 forward 输出。
3. 按组报告 finite 梯度及实际参数变化，包括 ViT、IAM、MANO head、PAM、adapter、TWM encoder／当前头；完成 replay 后另验 future 专属参数。只设置 `requires_grad=True` 不构成验收。
4. 两 rank 使用相同的任务 schedule，并对 visual／motion 的 unused 参数正确处理；motion forward 经过同一 DDP 封装；有效数归约正确。比较两个 rank 与已保存模型的完整参数 hash。
5. 正式 checkpoint 内模型／optimizer 浮点张量 finite，保存后恢复并实际再更新。保留两主干、插件、优化参数组、RNG、step／任务相位和数据／配置 SHA，不能在恢复时退回官方旧 HaWoR 权重。
6. 固定初始 HaWoR 与 D37 基线，区别于训练过程中已改变两主干的 adapter-bypass 诊断。评估真实当前腕／root／finger、准确输入退化、coverage／预测尺度、β 漂移；独立 motion 保持集检查当前恢复与 0.1／0.5／1／2 秒未来。

NLL 降低可能来自增大 σ 或减小 κ，不足以证明几何变准；预测 σ／κ 不是经过校准的状态质量 q。小样本训练运行、finite、SO(3) 合法性与整体物理正确性分别报告，不互相替代。

模型 checkpoint 应保存所有**已训练参数**，含更新后的 HaWoR、TWM 与 adapter，并记录官方 HaWoR 版本／配置、MANO 资产 hash 与预训练来源。受许可的 MANO 资产、模型权重、缓存、原始日志只存远程，不上传 GitHub。保存／恢复所需配置应随 checkpoint 保留，不以旧初始化文件仍在原路径为隐含条件；合法 MANO 运行资产仍需按明确约定提供。

## 参考机制与非等价处

[WHAM 官方 stage2 配置](https://github.com/yohanshin/WHAM/blob/main/configs/yamls/stage2.yaml)使用新模块 `1e-4`、预训练模块 `1e-5`；[optimizer 实现](https://github.com/yohanshin/WHAM/blob/main/lib/utils/utils.py)按 integrator 与其余模块分组，[stage2 数据实现](https://github.com/yohanshin/WHAM/blob/main/lib/data/datasets/mixed_dataset.py)仍包含 AMASS。这为低 LR 适配和 motion replay 提供机制参考；WHAM 的人体状态、视觉冻结范围和任务与本项目不同，不能直接外推保持保证或最优比例。

[HaWoR 原论文 §8.1](https://arxiv.org/html/2501.02973v1#S8.SS1)原训练采用 AdamW、`1e-5` 并冻结 WiLoR ViT。本轮包含 ViT 的双主干全量微调是用户最新选择的研究实验，**不是 HaWoR 官方训练协议复现**。相关旧版事实、保留与替代关系见 [D35 框架](VISUAL_FEEDBACK_FRAMEWORK_2026-10-15.md)与[原微调讨论](VISUAL_FEEDBACK_FRAMEWORK_2026-10-15.md#2026-10-15twm真实误差适配与先验保持讨论建议未实施)。

## 实际运行记录（待填）

- 代码入口：[联合 trainer](../scripts/train_visual_joint.py)、[在线 RGB loader](../src/handloop/data/visual_rgb.py)、[HaWoR bridge](../src/handloop/visual/hawor.py)、[联合 pipeline](../src/handloop/visual/pipeline.py)、[两轮反馈系统](../src/handloop/models/visual_feedback.py)。代码存在与静态审查不代表远程已成功运行。
- 代码 commit／源码 hash：待填。
- 已完成 D37 source step／SHA／配置与 best 选择依据：待填。
- RGB source manifest／SHA、实际 train／val 录制、帧数及左右手分布：待填。
- run 路径、W&B、启动 UTC、实际 batch／显存／吞吐：待填。
- 两轮输入依赖、全组梯度／参数更新、DDP／checkpoint／resume 验收：待填。
- 固定视觉 baseline、joint／mesh／state 几何与 motion 能力保持结果：待填；尚无本轮提升结论。
