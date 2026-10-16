# D39：HaWoR／TWM 全量联合微调 pilot 协议

记录日期：2026-10-16 UTC（纽约 2026-10-15）。**当前状态：每卡 4 段阶段已在 step65 共同保存停止，源码 `eb8d0b9` 从该断点完整恢复为双卡每卡 8 段 × 32 帧；00:59:46 UTC 快照已实际训练至 76／1,000 updates。新阶段 checkpoint69 已保存且两个 rank 参数 hash 一致；step73 的 14 组梯度与代表参数变化均已验，单卡 peak allocated 约 63.498 GiB。** 本次确认 batch 变更、两主干更新和恢复保存链路，不据此声称精度改善。下列预算、学习率和混合比例是工程 pilot 设置，不是已验证最优值。代码／数据事实、失败修复与效果边界见文末及[原启动摘要](../reports/visual_joint_launch_20261016.json)、[每卡 8 段恢复摘要](../reports/visual_joint_batch8_resume_20261016.json)。

**后续配置变更：** 用户随后要求视觉每卡由 4 段提高至 8 段。本文原每卡 4 段的启动快照保留为历史；新配置、受控断点恢复约定与已完成远程验收见文末“每卡 8 段配置变更”。

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

平移以米为单位，在该帧固定 anchor 中计算密度；旋转以 SO(3) 单位 Haar 测度归一化。按完整有效 state 平均，**不再除以 51**。当前／未来 99D 目标标签只进入外部 loss 和 metrics，不能作为状态填入模型输入、候选、参考系或历史。这里的标签隔离不消除本轮明确采用的 GT-derived crop 和 reference-valid 源帧筛选条件；这些输入条件仍按下一节披露。DDP 两 rank 的完整有效 state 数可能不同，必须按全局有效数归约，而非无条件平均两个 local mean。

motion replay 沿用原 D37 train 分区、受损条件与原六项 objective，保持原来当前恢复及最长 60 步未来分布的训练路径。它不读取 RGB，不把视觉标签混作原四库轨迹数据，也不使用 val／test 进行回放。

| D37 原 replay 项 | 系数 |
|---|---:|
| 当前几何恢复 | `1.0` |
| 当前 NLL／51 | `0.05` |
| 长轨迹 mixture NLL／有效未来步数／51 | `0.3` |
| 最佳轨迹几何项 | `0.2` |
| 速度项 | `0.05` |
| MoE balance | `0.01` |

精确定义以 [D37 配置](../config/twm_four_sources_moe_transient_v3_b512_10k.json)和 [objective 实现](../scripts/train_probabilistic_twm.py)为准。这里保留六项是明确的旧任务回放安排，**不是把六项改称新的视觉 loss**，也不提前实施用户已延后到下一版本的 loss 重设计。只做视觉 current NLL 时，future 专属路径没有相应监督；全量低 LR 本身也不保证未来能力保持。

## 数据与运行预算

使用已经下载的 DexYCB，沿官方 s0 recording split，在窗口化前固定训练／验证录制：

- subject-01 和 subject-02，各选 8 个 train、4 个 val 录制，即目标 16 train／8 val 录制。
- 每录制最多取 128 帧；总量上限 3,072 帧。源数据准备已经 complete，实际提取 **1,705 张 JPEG，其中 train 1,128、val 577；16 train／8 val 录制**。两主体分别为 8 train／4 val；全池 14 个右手、10 个左手录制。手性计数为录制数，不是左右手帧数，不能把上限写成实测。
- 当前使用 GT 关节点投影产生 crop，使用该固定相机作为明确命名的 world gauge；时间戳为 frame index／30 的标称时间。
- GT 来源是官方 fitted MANO／joint annotation，不称独立 glove／MoCap 真值。源 β 与 HaWoR β 约定不得未经验证混用。
- 对帧间缺口分段，避免原 16 帧 HaWoR 块跨越不连续源时间；录制不能跨 train／val，测试录制排除。

该数据扩展仍是 **GT-crop、固定相机的联合训练 pilot**，不是头戴移动相机 SLAM、自动 detector 或完整 HaWoR benchmark 验证。D35 的 63 train／64 val 帧和 20 次更新只作为历史工程结果，不自动成为本轮基线。

正式 pilot 预算为 **1,000 次 optimizer update**，按 `3:1` 序列约 750 次视觉、250 次 motion replay。双 GPU smoke 从视觉每卡 batch 1、clip 16 起，实测单卡 peak allocated 约 **17.77 GiB**。在完成该 smoke 后，正式配置曾增至**视觉每卡 batch 4、clip 32，motion replay 仍每卡 batch 32**，另开正式 run；实际 step10 快照与约 **37.742 GiB／卡**峰值已验收。该历史配置的视觉 global batch 为 8 clips，每 clip 32 个含 mask 的帧位置；motion global batch 为 64 windows，它们不是同一种样本计数。用户后续将视觉每卡改为 8 段，见文末；smoke 的 8 次更新不计为正式 1,000 次已完成。

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

当前实现已使用 checkpoint 内的 TWM 模型配置、source metadata 和 replay 配置构造恢复网络，再加载联合权重，**resume 不再读取原 D37 checkpoint 文件**。远程 smoke 已在 CLI 提供不存在的 D37 路径时，从 step4 恢复并实际完成至 step8，验证了这一点。仍需提供官方 HaWoR 结构／初始化资产、代码和合法 MANO 运行资产；目前不能称整个运行环境零外部依赖。

验证保存三种初始化基线：原 HaWoR 的 raw 指标、初始化两轮联合系统、初始化 adapter-bypass 系统。`best.pt` 只在训练 checkpoint 中按 validation geometry score 选择，**step0 基线单独保存、不参与 best 排序**；因此 best 表示已保存训练 checkpoint 中的最优，并不自动优于初始模型。初始与后续验证还使用四库每库最多 16 个固定窗口的纯轨迹保持集，分别记录 noisy／clean 当前几何、selected future ADE、恢复当前后 hold 对照，以及 0.1／0.5／1／2 秒误差与各指标有效 counts。该小保持集已实现，不等于其结果已通过；它不是真实 RGB 的未来预测评测。

## 参考机制与非等价处

[WHAM 官方 stage2 配置](https://github.com/yohanshin/WHAM/blob/main/configs/yamls/stage2.yaml)使用新模块 `1e-4`、预训练模块 `1e-5`；[optimizer 实现](https://github.com/yohanshin/WHAM/blob/main/lib/utils/utils.py)按 integrator 与其余模块分组，[stage2 数据实现](https://github.com/yohanshin/WHAM/blob/main/lib/data/datasets/mixed_dataset.py)仍包含 AMASS。这为低 LR 适配和 motion replay 提供机制参考；WHAM 的人体状态、视觉冻结范围和任务与本项目不同，不能直接外推保持保证或最优比例。

[HaWoR 原论文 §8.1](https://arxiv.org/html/2501.02973v1#S8.SS1)原训练采用 AdamW、`1e-5` 并冻结 WiLoR ViT。本轮包含 ViT 的双主干全量微调是用户最新选择的研究实验，**不是 HaWoR 官方训练协议复现**。相关旧版事实、保留与替代关系见 [D35 框架](VISUAL_FEEDBACK_FRAMEWORK_2026-10-15.md)与[原微调讨论](VISUAL_FEEDBACK_FRAMEWORK_2026-10-15.md#2026-10-15twm真实误差适配与先验保持讨论建议未实施)。

## 实际运行记录

- 代码入口：[联合 trainer](../scripts/train_visual_joint.py)、[在线 RGB loader](../src/handloop/data/visual_rgb.py)、[HaWoR bridge](../src/handloop/visual/hawor.py)、[联合 pipeline](../src/handloop/visual/pipeline.py)、[两轮反馈系统](../src/handloop/models/visual_feedback.py)。代码存在与静态审查不代表远程已成功运行。
- 代码演进：`63a471d` 初始联合训练实现；`bfd473e` 修复 AMP 下几何 FP32 边界；smoke_v2 从 **`836d201`** 开始运行；随后补充 mmap checkpoint、早期恢复点及相同 checkpoint 的硬链接复用，正式 pilot 发起源码为 **`eeac1fc`**。
- D37 于 **2026-10-15T23:55:39 UTC** 正常 complete 10,000 updates；本轮固定 source 为 best step10,000，SHA256=`9f500a283109fbad75f380281808b16781a335550b99853d66832cb120bef76c`，路径为 `/sciclone/data10/jchen57/Hand_Reconstruction/model_assets/twm_d37_final_20261016/best.pt`。保留其模型配置、四库 train／val 划分与原噪声／objective。best 的预训练指标依据沿 D37 原协议，不能用该身份推断真实视觉收益。
- RGB 准备：complete，实提 1,705 JPEG（train1,128／val577），16 train／8 val 录制；两主体分别 8 train／4 val，全池 14 right／10 left 录制。manifest 路径／SHA 与逐 split 手性细表待回收。
- **保留失败：** `63a471d` 的首次双卡执行在 step0 baseline 因 `camera_to_anchor` 的 BF16×FP32 矩阵乘法 dtype 不一致报 RuntimeError，尚无 optimizer 更新。该失败 run 的 logs／status 保留，自建测试进程已终止，未删除用户 Job。
- **修复与合同验收：** `bfd473e` 将相关几何计算限定为 FP32。远程 CPU 两轮梯度 fixture、真实 RGB FP32 的 crop／raw／GT 隔离／padding 合同已通过；修复后的 CUDA BF16 两轮梯度合同和真实 RGB AMP 合同也已通过。合同脚本见 [两轮检查](../scripts/check_visual_feedback_refinement.py)与[真实 RGB 检查](../scripts/check_visual_rgb.py)。这些测试不替代全量 HaWoR／TWM 联合更新、双卡同步与正式 checkpoint 验收。
- **smoke 更新与恢复：** 已实际更新到 step4，保存后恢复至 step8，于 **2026-10-16T00:37:16 UTC** 完成 smoke 执行；这不是正式 1,000 次 pilot 完成。恢复测试给 `--twm-checkpoint` 传入不存在路径仍成功完成更新，证明已训练的 TWM 不依赖旧 D37 source 文件重载。
- **checkpoint／DDP 验收：** step4 CPU audit 检查 1,076 个模型浮点张量、3,135 个 optimizer 浮点张量全部 finite；CPU 完整模型 hash 与两个 rank 相等。两 rank 的 14 个梯度组均记录 finite 非零梯度，且每组代表参数相对初始化已改变。该事实证明训练链路实际更新，不构成精度或全运动分布正确性结论。
- **训练后真实 RGB 推理：** 16 帧输出当前 `[1,16,99]`、未来 modes `[1,16,4,60,99]`、两轮 state trace `[1,16,2,99]`；finite、正定 translation covariance、mixture 归一化检查通过。这里只验接口和数值合同，尚未给出真实 RGB future 精度。
- **checkpoint I/O 修正：** 联合模型 checkpoint 约 8.5 GB，同 step 重复保存已改为复用硬链接。`93397b4` 曾残留与目标同 inode 的 `.tmp` 别名，不是 checkpoint 内容损坏；`eeac1fc` 已修复重复硬链接处理。历史同 inode 临时别名已安全清理，有效权重保留。
- **正式 pilot 启动：** 源码 `eeac1fc`，于 **2026-10-16T00:38:57 UTC** 发起，metadata 于 00:39:53 UTC 写出；**00:43:19 UTC 快照为 training step10／1,000**。视觉每卡4×clip32、motion每卡32，trainable 参数 **708,688,515**，train／val clips 为 **46／24**；每卡 peak allocated 约 **37.742 GiB**。run 为 `/sciclone/data10/jchen57/Hand_Reconstruction/runs/visual_feedback_joint_v2_20261016`，源码 checkout 为 `/sciclone/data10/jchen57/Hand_Reconstruction/code/visual_feedback_joint_v2_20261016`。
- **正式双卡与保存验收：** 两 rank 的 14 组非零梯度 seen 与代表参数变化均为 True；正式 `latest.pt` 在 step4，大小 **8,510,562,095 bytes**。两个 rank 完整模型 hash 一致，为 `37538aeafae4c1995e358f128e17c5de2852d2df16f57c0a7d4fa4522af1aa87`。smoke 的 CPU 1,076／3,135 finite 审计和不存在旧 D37 路径的续跑测试见上，不将其冒称为正式 step4 的额外 CPU 审计。
- **W&B：** [3abfc864](https://wandb.ai/black-ant/handloop-twm/runs/3abfc864)，服务端已验 running／optimizer_step8；服务端标量有异步滞后，与远程 step10 快照分别记录。
- **保留负面 step0 基线：** 固定 8 个 val clips、210 个有效 states，原 HaWoR raw 的腕／root／finger 为 **88.250 mm／11.729°／9.747°**；初始化两轮联合 final 为 **121.616 mm／27.489°／15.793°**，xyz 90% coverage 为 **0**。这是训练前明显失配的初始化结果，不能与随机训练 batch 的指标直接比较，也不能当成本轮已改善。首次训练后 fixed validation 在 step50，当前尚未到达。
- 训练后的固定视觉 state／joint／mesh 几何与 motion 能力保持结论：待验证；当前只确认正式训练、双主干更新和保存链路已运行，尚无本轮精度提升结论。小型记录见[启动摘要](../reports/visual_joint_launch_20261016.json)。

## 每卡 8 段配置变更（用户后续决定；2026-10-16 UTC）

用户观察 GPU 占用后，明确要求“每卡 8 段，重新跑”。本次以源码 **`eb8d0b9`** 将 [联合训练配置](../config/visual_joint_v2.json)的 `per_gpu_rgb_batch_size` 从 **4 改为 8**；clip length 仍为 **32**，双卡常规视觉 global batch 从 8 改为 **16 clips，即 512 个带 mask 的帧位置**。有效监督帧数由真实帧及标签 mask 决定，不能把 512 当成每步有效 GT 数。motion replay 仍每卡 **32 windows／全局 64 windows**，HaWoR／TWM／adapter 目标 LR 仍分别为 **`1e-6`／`1e-5`／`1e-4`**，任务次序仍每四次更新三次视觉、一次 motion；不因 batch 增大自动放大学习率。

预算继续为**总计 1,000 次 optimizer update**，保留已完成更新，从共同保存的 latest checkpoint 继续；不是从原始预训练重新初始化，也不是追加 1,000 次。保存停止后再更换执行代码与配置，保留旧阶段的配置、日志、验证与权重来源。总更新数不变而剩余视觉 batch 增大，剩余视觉样本暴露量会增加；更新比例 `3:1` 不等于视觉帧与 motion windows 的相同比例。现有 train 共 46 clips，DistributedSampler 每 rank 23 clips；未启用 `drop_last`，每 rank 的新一轮 batch 为 `8+8+7`，末批的全局样本数相应少于 16。

[trainer](../scripts/train_visual_joint.py)与 [ds launcher](../scripts/ds_train_visual_joint.py)新增显式 `--allow-rgb-batch-size-change`，只可与 `--resume` 联用。恢复时仅允许配置中的 `per_gpu_rgb_batch_size` 改变，旧／新值必须为正整数；其余配置、数据及资产 SHA、world size 仍须一致。模型、optimizer、scheduler、总 step、best score 与各 rank RNG 从 checkpoint 恢复，新 batch 由当前配置构建 DataLoader；schedule 与 `step % 4` 的任务相位延续，不能重新 warmup 或退回官方旧 HaWoR／原 D37 权重。

恢复记录写入 `run_metadata.json` 的 `resume` 字段及后续 checkpoint 的 `resume_transition`。旧 metadata 与新执行 metadata 存入 run 下的 `execution_metadata/`，并向 W&B 记录当前视觉／motion batch 和 `resume_start_step` 的 runtime scalar。由此区分同一 run 的 batch 4 与 batch 8 阶段，不能只依据服务端历史 config 判断本次 batch。既有实现的限制仍保留：**RGB epoch／loader cursor 与 motion sampler cursor 从头重建，恢复 RNG 不等于逐样本精确续跑**。

每次执行开始后的前 **8 次新增更新**重新记录两个 rank 的梯度与代表参数变化，并带 `execution_start_step`、当前 batch、peak allocated；相应逐步日志也按新增更新计数。早期 checkpoint 改为本次执行后的第 **4 次新增更新**（`start_step + 4`），正常每 50 步与停止时保存继续保留。验收需实际完成新 batch 的视觉前向、反向和 optimizer 更新，并确认新 checkpoint 与双卡参数一致；仅恢复加载成功不算 batch 8 跑起。

**远程实测验收（2026-10-16 UTC）：** 每卡 4 段阶段于 **00:52:37.602614 UTC** 共同保存停止在 **step65**，两个 rank 的模型 hash 为 `29b7ac7e457591eec434acf3a6ac2298d349df82d26a51c4f5c2763b6bba7d9a`。旧阶段的小型记录与权重硬链接归档在原 run 下 `batch4_before_batch8_20261016/`；原始结果保留。源码 **`eb8d0b98f5e2e27fae9ed6f664399d1eca6f2dae`** 于 **00:53:43.749509 UTC** 从 step65 在同一 run 完整恢复，每卡 batch8，未重置 optimizer／scheduler／总更新计数。

- **实际更新与显存：** **00:59:46.061830 UTC** 快照为 `training`，**76／1,000 updates**，即恢复后已完成 11 次新增更新。step73 两个 rank 的 14 个参数组均记录 finite 梯度、非零梯度 seen 与代表参数实际改变；每卡 peak allocated 为 **63.498240470886 GiB**。同次 `nvidia-smi` 两卡各 **79,446 MiB**，单时刻 utilization 为 100%／60%；这是采样瞬间，不能解释为稳态吞吐率或稳定利用率。
- **新阶段 checkpoint：** 本次第 4 次新增更新保存 **step69**，配置 batch8，大小 **8,510,565,551 bytes**；两个 rank 模型 hash 均为 `9455c4b976f5e1c5d58e044cdd5de0022f57b79646d8d5d96d9fd25d2e7ec4c9`，与旧阶段不同。保存的 scheduler `last_epoch=69`，第一个 HaWoR optimizer parameter 的 `step=52`；各参数只在参与其任务并有梯度时累计 Adam step，不能把该局部计数误当总 optimizer 更新数。
- **服务端进度：** 沿用 [W&B 3abfc864](https://wandb.ai/black-ant/handloop-twm/runs/3abfc864)，服务端已验 `running`／`optimizer_step=75`，runtime 记录 batch8／`resume_start_step=65`。服务端异步进度与远程 step76 快照分别记录。
- **产物：** run 仍为 `/sciclone/data10/jchen57/Hand_Reconstruction/runs/visual_feedback_joint_v2_20261016`；远程汇总为 `/sciclone/data10/jchen57/Hand_Reconstruction/evaluations/visual_joint_v2_20261016/batch8_resume_summary.json`，小型本地摘要见[每卡 8 段恢复记录](../reports/visual_joint_batch8_resume_20261016.json)。

以上证据确认新 batch 的前向、反向、参数更新和保存已运行；没有新增关于几何精度或能力保持的验证结论。每卡 4 段历史峰值 **37.742 GiB** 与本次 batch8 的 **63.498 GiB** 分别保留，不混用。
