# 联合训练梯度、目标与时序静态审计（2026-10-16）

本文针对已完成 D40 联合训练做源码审计，区分确定的实现事实、结构性优化风险和待验证因果。**本次不改模型、目标、数据划分或 checkpoint，不启动训练。** 用户要求先充分分析，再考虑修改。训练源码 `eb8d0b9`；所审文件中的训练逻辑与该次运行保持一致。数值 gradient probe 作为独立只读诊断脚本提供，其结果必须以远程执行产物为准，不能把脚本存在写成实验已完成。

## 1. 目前没有坐实“左手梯度反号”等具体实现错误

逐段检查了 trainer、在线 RGB loader、HaWoR bridge、99D codec、两轮视觉反馈与概率 TWM：

- `is_right=0/1` 来自实际录制，TWM 接收原侧别 embedding；没有右手过滤或强制将所有输入标为右手。
- 左图 crop 反射、相机 `tx` 反号、`S R S` 旋转还原和候选投影中的 crop 反射各承担不同坐标变换，没有发现同一处被重复还原的确定证据。
- raw、两轮候选、MANO 与视觉查询之间的 autograd 通路没有被意外 `no_grad` 或全局 `detach` 切断。
- motion replay 调用的是视觉分支使用的同一个 `pipeline.feedback.twm`，不是独立的未更新副本。每次更新前 `ddp.train()`，会在验证之后恢复两个主干的训练模式。

这些结论来自源码与既有合同记录，**不等于已证明每种输入、库间 shape 语义及所有自动微分数值均正确**。新增脚本包含可独立执行的 `S R S` 与相机 `tx` autograd sanity；真实左右手失配还需与分侧运动／真实视觉状态对照一起判断。

主要来源：[HaWoR bridge](../src/handloop/visual/hawor.py)、[codec](../src/handloop/visual/codec.py)、[pipeline](../src/handloop/visual/pipeline.py)、[visual feedback](../src/handloop/models/visual_feedback.py)。

## 2. 每 3 次视觉、1 次 motion 更新，具体意味着什么

[trainer `JointTaskModel`](../scripts/train_visual_joint.py#L119) 与 [更新循环](../scripts/train_visual_joint.py#L487) 的实际行为：

| 更新 | 被更新的模块 | objective | 归约 |
|---|---|---|---|
| 视觉，750／1000 次 | HaWoR、adapter、TWM 当前路径及共享表征 | 最后一次当前完整 state NLL | 对真实且完整标签的 state 平均；三个平移 log-density 相加、16个旋转 log-density 相加；不除51 |
| motion，250／1000 次 | 同一 TWM 的当前、未来与共享表征 | 原 D37 六项 objective | current NLL／51，trajectory NLL／有效未来步数／51，另有几何、速度、MoE 项 |

视觉每个 rank 的 local mean 乘 `world_size*local_valid/global_valid`；DDP 再平均梯度，得到全局有效 state mean。当前数据有效标签完整，不存在本轮某 rank 完全没有监督 state 的事实；如果未来出现全空 local batch，`feedback_current_nll` 仍会报错，这属于未覆盖的边界，不是本次左手失败证据。

motion 更新在 `zero_grad(set_to_none=True)` 后绕过 HaWoR／adapter。它们的 gradient 是 `None`，AdamW 不会在该次更新增加其 step 或施加 weight decay。共享 TWM 每次均可能更新；future 专属参数只在 motion 回放得到梯度。因此，总 optimizer step、每个参数的 Adam step 不同是正常行为，不能把 HaWoR 只有约750次参数更新误判为未解冻。统一 LR scheduler 按全部1000次 optimizer update 前进，future 参数也沿该全局时钟衰减 LR。

三组基础 LR 为 HaWoR `1e-6`、TWM `1e-5`、adapter `1e-4`，均由20步 warmup 和同一余弦倍率调整。实际是分别对 **整个 HaWoR、整个 adapter、整个 TWM** 做 norm=1 clipping，不是整个系统共用一个 norm，也不是每个小 head 独立裁剪。[参数分组](../scripts/train_visual_joint.py#L165)、[裁剪和更新](../scripts/train_visual_joint.py#L507)。

### 2.1 两类任务的 NLL 单位确实不同，3:1 不能解释成梯度权重3:1

视觉 current NLL 的前系数是1；回放 current NLL 的前系数是 `0.05/51`。**仅对同一份 current NLL 这一项比较，显式前系数比为1020。** 回放还包含当前几何、未来 NLL、未来几何、速度与 MoE；其数据误差、有效长度、梯度方向亦不同。因此不能说视觉的总训练影响是回放的1020倍，也不能把3:1任务频率理解为先验保留强度已被合理校准。

对高误差且 σ 很小的平移 Gaussian，均值梯度为 `(p−p_GT)/σ²`；旋转均值的梯度由各组 concentration κ 调节。原 motion 的小毫米级输入分布迁移到数十至百毫米级真实视觉误差时，同一 NLL 可以形成很大的梯度。norm clipping 会改变它们的绝对量级，Adam 的历史二阶矩又会改变参数更新。应检查原始 norm、裁剪比例、各组方向及实际性能，不能只比较 loss 标量或 LR。

## 3. 最终状态受监督，前端的几何语义未被单独约束

视觉 forward 保留 HaWoR ViT → IAM → MANO head → PAM，并将 camera、root、finger、β 经统一 codec 接到后续模块。最终 current NLL 会通过这条路径回传；但本次没有单独的 raw pose／camera／β 保持或几何目标。

| HaWoR 输出 | 到 final NLL 的已存在梯度路径 | 当前目标未直接要求什么 |
|---|---|---|
| camera | `get_trans` → raw J0 → candidate／TWM，以及 raw／candidate 投影 →视觉查询 | raw 腕必须独立准确、camera depth 本身保持原解 |
| root／finger pose | 6D→SO(3)→raw／candidate；MANO候选投影→视觉查询；TWM当前分布 | raw root／finger 必须各自接近参考 |
| β | shaped-rest J0 转换；MANO候选 joint 投影→视觉查询 | 正确 shape、mesh 或 bone length；β 不属于99D状态 NLL |
| IAM patches | adapter global／local attention→candidate→TWM | 中间特征保留原 HaWoR读出所需的语义 |

`condition_on_beta=false` 只意味着 TWM 不直接读取 β；**不意味着 β head 没梯度，也不意味着各模块对 β 不敏感。** 仅从 rest J0 的3维约束不能唯一识别10维 β。候选投影进一步提供间接梯度，但没有直接 shape 参考目标，仍不能声称已监督准确 shape。

一个局部的一维示意是 `p_final=p_raw+Δ_adapter+Δ_TWM`。若 `p_raw` 变化 δ 而后续修正补偿−δ，最终误差可以不变。实际网络是非线性且共享表征，不能由此构成完整不变性定理；但它说明**只约束最终输出，不保证中间 raw 仍具有独立几何准确性**。已观察到左 raw 位置退化、后续补偿，与此优化风险相容，不足以坐实该风险是唯一根因。

训练中 `beta_norm` 是 detached 日志，不是 loss；`geometry_score` 亦是测量／best 选择，不参与视觉 backward。[visual task](../scripts/train_visual_joint.py#L128)。

## 4. Detach 在哪里：帧内可微，跨时间截断

`visual_feedback.py` 的实际路径为：

1. 在帧开始构造 `raw_clean`；**raw 不 detach**。
2. 上一已完成历史 `.detach()`，同帧两个 refinement 共用这一份历史。
3. anchor 取上一个已完成腕，若空历史则取本帧 raw；**anchor 整体 detach**，同帧固定。
4. 首轮 adapter → TWM output 直接作为次轮 candidate，**轮间不 detach**。
5. 次轮 final NLL 回传到两轮调用及当帧 HaWoR；只有末轮 final 及分布描述 commit 一次。
6. commit state／uncertainty 继续 detach，后续物理帧的 loss 不通过已完成历史反向传回早前帧。

来源：[anchor／raw](../src/handloop/models/visual_feedback.py#L427)、[两轮](../src/handloop/models/visual_feedback.py#L498)、[commit](../src/handloop/models/visual_feedback.py#L594)。这是明确设计的 truncated temporal credit assignment，**不是意外断梯度 bug**。它会保留推理时的预测历史依赖，但不能直接训练“本帧改动如何影响后30帧”的完整闭环梯度。HaWoR16帧双向注意力仍可让同块帧通过视觉前端影响当前 raw，不能将整个系统误称为时间上完全独立。

每32帧训练 clip 从空反馈历史开始；30帧历史在大部分 clip 中仍处于填充期。纯 motion replay 则有固定30帧受损参考历史。真实自身历史与合成历史的分布、历史有效长度都不同。reset、gap和padding处理未发现将无效帧 commit 的确定错误，但这种训练上下文差别仍是适配风险。

## 5. 左右变换与投影为什么暂不能判为错误

HaWoR把左手图像镜像为规范 crop。bridge 对相机横向参数反号，并使用未翻转的原始 bbox center 恢复全图平移；codec 对左姿态做 `R_left=S R_canonical S`，`S=diag(-1,1,1)`。这保持 `det(R)=+1`，不需要不稳定的 axis-angle log/exp。[hawor.py](../src/handloop/visual/hawor.py#L231)、[codec.py](../src/handloop/visual/codec.py#L59)。

RGB loader 的 `crop_transform` 含左手 x 轴反射，且映射到 HaWoR移除两侧32像素后的192×256有效区域，与16×12 patch grid一致。候选 MANO joints 位于原始 camera 坐标，先经过真实 K，再经过该 crop transform；**此时包含反射是将真实几何投到规范化左 crop，不能直接删除该反射**。[visual_rgb.py](../src/handloop/data/visual_rgb.py#L157)、[candidate projection](../src/handloop/models/visual_feedback.py#L513)。

局部 attention 只使用正深度、有限且 crop 内的 joint；无效 projection 不提供局部更新，但 global image attention 仍存在。因此 raw 偏离 crop 可削弱候选局部取证，不是图像分支完全消失。候选可用性与 raw mask 分开维护。

HaWoR 用 `left_shapedirs_fix=True`，源参考可能不用这一 convention，这是已知语义差别。由于 source参考与prediction各自独立解码且不混用β，不能只凭“fix不同”认定评测代码错；它是否通过 β/J0 通路放大当前左定位偏差，需要 shape 与几何专项结果。

## 6. 回放保持能力仍有边界

joint replay 保留原 D37 train split、v3 corruption 与六项 objective，但 sampler 改成 **四库等概率／库内有放回**，不再是 D37 的完整窗口遍历。250个 motion 更新、每卡32窗、双卡约16,000次样本抽取，只相当于636,610窗池的约2.5%抽取次数，且可能重复，不能说每个运动样本在联合微调时都得到保持监督。左右手仍没有被过滤，主要问题是任务与错误分布迁移，不是覆盖数量自动保证恢复。

visual current-only 不调用 future decoder；motion 会调用，故 future 专属参数不是永久冻结。共享 encoder、current heads 和未来积分起点会同时受到视觉及回放影响，低LR或回放存在本身不能构成 future 能力保持证明。终点应使用同输入比较预训练与联合 TWM 的真实分侧当前／未来结果。

## 7. 新增只读梯度诊断的协议

[diagnose_joint_gradients.py](../scripts/diagnose_joint_gradients.py) 默认：

- 只从已用 train split 预先固定左、右各2条不同录制，取其首个完整16帧前缀，先保存 selection；不根据测试误差筛选样本，不训练。
- 加载 best300 或 last1000，eval／dropout关闭；同一固定seed产生四库各2窗的原 motion corruption。
- 对 final NLL、final geometry、raw geometry和raw wrist保持目标做自动微分；保存 HaWoR camera／pose／β／共享head／PAM、adapter以及TWM各组的 norm／cosine。
- 对视觉完整 NLL 及其 `/51` 版本与同一 TWM motion总objective比较。`/51` 仅用于单位诊断，不改变运行中的目标，也不构成修改建议已采用。visual与motion样本内容不同，cosine是局部诊断，不是严格匹配任务消融。
- 将左、右各2个train clip的完整NLL梯度分别平均，报告camera／pose／β读出、adapter与TWM各组及整个TWM的左右cosine；另按每个clip实际TWM／adapter moduleclip后再平均。只输出统计，不保存大梯度。左右录制没有动作镜像配对，不能由负cosine直接推导test左手失败因果。
- 精确计算整 TWM／adapter 的训练 module clipping factor，以及每小组的 squared-norm fraction；HaWoR只计算head／PAM子集，**不冒称整个HaWoR（含ViT/IAM）的真实裁剪比例**。
- 检查末帧位置对当帧raw与末轮candidate的位置局部Jacobian，并报告对之前raw状态张量的导数；该偏导不代表对之前RGB帧的总依赖。
- 远程执行镜像SRS、tx反号autograd sanity；前后hash验证所有被测参数和TWM未改变；无optimizer。

为了控制成本，ViT/IAM作为固定上游feature producer，其余被测参数保持可微。这不改变所测参数在该输入处的偏导数，但没有测全骨干梯度，也不将该probe当作实际训练更新的复现。BF16前向／FP32几何保持与训练一致；确定性dropout关闭与训练有差别，结论须保留此范围。

CLI：

```bash
python scripts/diagnose_joint_gradients.py \
  --checkpoint <joint_best_or_latest.pt> \
  --source-manifest <actual_visual_train_val_source_manifest.json> \
  --motion-manifest <actual_D37_collection.json> \
  --hawor-root <pinned_HaWoR> --hawor-checkpoint <official_hawor.ckpt> \
  --hawor-config <official_model_config.yaml> --mano-models <licensed_models_dir> \
  --device cuda:0 --output <new_report.json>
```

本文件不因局部梯度冲突就决定修改loss或删除TWM。修改优先级需综合：纯预训练TWM分侧恢复、相同真实HaWoR输入的替换／旁路控制、shape与绝对位置偏差，以及该只读梯度结果。
