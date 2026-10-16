# TWM 左手覆盖与联合模型左手退化诊断（2026-10-16）

## 1. 当前结论与证据边界

**TWM 预训练包含接近均衡的左右手运动，当前证据不支持“没有训练左手”这一解释。** D37 实际训练池有左手 **317,226** 个窗口、右手 **319,384** 个窗口，约为 49.83%／50.17%。当前联合模型在独立测试子集上的左手绝对 MPJPE 退化，首先出现在联合适配后的视觉 `raw` 输出；其后校正补偿了较大一部分，但最终仍未完全恢复。

这是对实际 manifest、窗口索引、训练采样实现和已保存评测产物的诊断。本次没有新增模型推理、修改权重、重训或改动数据划分。**目前仍未唯一确认根因。** 不能将整套联合模型的分侧结果直接归因为原始预训练 TWM，也不能把一个阶段的前后差值当作因果消融。

原始完整评测协议与指标定义见[最终对照报告](VISUAL_JOINT_FINAL_EVALUATION_2026-10-16.md)；本轮统计与配对诊断产物为 [left_hand_diagnosis_20261016.json](../reports/left_hand_diagnosis_20261016.json)。

## 2. 实际预训练左右手覆盖

以下按 D37 实际 loader 的基类 `TrajectoryWindows`，使用训练时相同 manifest、history30／min_future3／stride4 重新索引有效 **train window**；`ForecastTrajectoryWindows` 复用该最短未来窗口总体，再补充可变长度的未来目标。统计不是只数 manifest 条目或原始视频帧，窗口需满足既定历史长度、最少未来长度、完整99D标签和时间连续性条件。总数636,610与原训练记录一致。

| 数据源 | 左手窗口 | 右手窗口 |
|---|---:|---:|
| OakInk2 | 173,430 | 173,430 |
| HOT3D | 66,004 | 66,004 |
| UmeTrack | 74,605 | 76,785 |
| DexYCB | 3,187 | 3,165 |
| **合计** | **317,226** | **319,384** |

准备后的 train frame 总数为左手 **1,389,478**、右手 **1,398,657**；有效 val window 为左手 **30,509**、右手 **30,861**。这些数量表明左手既没有缺失，也没有严重数量不足；它们不证明左右两侧运动难度、姿态覆盖或标签质量完全相同。

D37 的 `source_sampling=full_coverage`，每 epoch 为 **636,610** 窗口、**622** 次双卡更新，无 DDP padding。10,000 步训练完成了 **16 个完整遍历，另加48批**。采样遍历整个窗口索引，不按右手筛选。完整覆盖也不等于对每个数据源、手别或动作重新等权；实际训练权重仍由有效窗口数决定。

执行配置为每卡512、双卡 global batch1024、30帧历史、最多60帧未来、最少3帧未来、window stride4，99D状态且 shape β 另存，`condition_on_beta=false`。训练源码为 `87a2bcc`。本次审计确认，相关 loader、forecast、v3 corruption、sampler、TWM 与 ProbabilisticTWM 文件相对该版本没有源码差异。

## 3. 手别在代码中如何传递

### 3.1 数据准备没有把左手删除或全部转成右手

- OakInk2 显式遍历 `("lh",0)` 和 `("rh",1)`，两侧分别保存状态与 `handedness`：[prepare_oakink2_twm.py:153](../scripts/prepare_oakink2_twm.py#L153)。
- Meta 数据准备显式遍历 `left/right`，读取该侧 MANO 标注，并用该侧模型解码：[prepare_meta_twm.py:319](../scripts/prepare_meta_twm.py#L319)、[prepare_meta_twm.py:372](../scripts/prepare_meta_twm.py#L372)。
- DexYCB 从 metadata 读取真实 `mano_sides`，按侧解码并用侧别相关的指尖索引做21点 FK 检查：[prepare_dexycb_twm.py:240](../scripts/prepare_dexycb_twm.py#L240)。
- MANO decoder 为左、右分别持有 PCA 基底和均值；展开时按 `sides` 选择模型：[meta_mano.py:184](../src/handloop/data/meta_mano.py#L184)。

### 3.2 Loader 与采样器不排除左手

`TrajectoryWindows` 按 split、状态维度、标签完整性、序列长度和时间连续性建立索引，没有 `handedness==right` 过滤；`0=left,1=right,2=unknown` 被校验后原样返回：[trajectory.py:222](../src/handloop/data/trajectory.py#L222)、[trajectory.py:255](../src/handloop/data/trajectory.py#L255)、[trajectory.py:366](../src/handloop/data/trajectory.py#L366)。`ForecastTrajectoryWindows` 复用同一历史、shape 和手别处理：[forecast_trajectory.py:114](../src/handloop/data/forecast_trajectory.py#L114)。

全覆盖采样由 `CoverageEpochSampler` 实现，使用 shuffle 的 `DistributedSampler`，不丢尾批次：[train_probabilistic_twm.py:320](../scripts/train_probabilistic_twm.py#L320)、[epoch_sampling.py:10](../src/handloop/data/epoch_sampling.py#L10)。

### 3.3 TWM 实际接收手别，而非仅保存 metadata

噪声包装将 `batch['handedness']` 原样放入模型 kwargs：[temporal_corruption.py:155](../src/handloop/data/temporal_corruption.py#L155)。TWM 定义了三类 `handedness_embedding`，并把它加到所有时刻／关节 token 上：[twm.py:80](../src/handloop/models/twm.py#L80)、[probabilistic_twm.py:335](../src/handloop/models/probabilistic_twm.py#L335)。

训练的 anchor 只是最近历史腕的刚体坐标变换：位置减去 anchor 后旋转，root rotation 变换到 anchor frame，finger local rotations 保持原样。它不包含反射矩阵或左右镜像：[temporal_corruption.py:138](../src/handloop/data/temporal_corruption.py#L138)。

视觉链也继续传递真实手别：[pipeline.py:88](../src/handloop/visual/pipeline.py#L88)、[visual_feedback.py:554](../src/handloop/models/visual_feedback.py#L554)。HaWoR 对左手图像做 canonical crop 后，进入统一状态前通过 `S R S` 还原左手旋转，`S=diag(-1,1,1)`：[codec.py:59](../src/handloop/visual/codec.py#L59)。这一视觉预处理不能与“纯轨迹 TWM 全部按右手训练”混为一谈。

## 4. 视觉适配数据覆盖

| 划分 | 左手录制／有效状态 | 右手录制／有效状态 |
|---|---:|---:|
| visual train | 8／575 | 8／553 |
| full val | 2／143 | 6／434 |
| 独立 bounded s0 test | 9／554 | 7／461 |

视觉 train 也包含近似均衡的左右手；但 full val 的左右手组成明显不同，右手占434／577状态。**原 checkpoint 选择实际使用的 fixed8 是左手2 clips／40状态、右手6 clips／170状态；按有效状态聚合时，左手只占19.05%。** 因此原模型选择指标更偏重右手表现，左手的代表性不足。这是已核实的协议局限，但不能据此证明它是左手退化的唯一原因；不得事后利用test重新选best。

测试为 subject03–10 各预先选定的2条官方 s0 test 录制、原始 `frame_id<128` 的有界子集，所选录制与 prepared train／val 不重叠。视觉微调只见过 subject01／02；TWM 已见过这些 test 主体的其它 train 运动，因此不能称其为所有训练阶段都未见过的主体。两侧来自不同录制／主体／动作组成，不能把它们当成同一动作的左右镜像配对。

## 5. 左手退化出现在哪个阶段

所有数值均为已保存输出上的真实 MANO 几何误差，单位 **mm，越低越好**。各分支使用自身预测 β；GT β 仅用于参考 FK。腕对齐 MPJPE 仅去除腕平移，没有旋转、缩放或 PA 对齐。

### 5.1 独立测试：左手554个有效状态

| 阶段 | 腕部误差 | 绝对 MPJPE21 | 腕对齐 MPJPE21 |
|---|---:|---:|---:|
| 原始 released HaWoR | 72.0125 | 61.6020 | 17.8312 |
| 联合 best300 的 `raw` | 114.5084 | 109.7539 | 11.5183 |
| best300 最后一轮 `candidate` | 59.5036 | 60.2089 | 10.6179 |
| best300 最终输出 | 62.5151 | 63.7352 | 11.9230 |

这组结果把问题定位得更具体：**联合适配后的视觉 raw 左手腕位置明显变差，但腕对齐后的关节几何反而改善。** 后续链条把腕部误差从114.51补偿到62.52mm，最终腕部误差和腕对齐 MPJPE 均优于原始 HaWoR；不过最终绝对 MPJPE 为63.74mm，仍略差于 baseline 的61.60mm。因此“左手退化”在此主要指绝对 MPJPE，不能写成左手所有指标都退化。

### 5.2 独立测试：右手461个有效状态

| 阶段 | 绝对 MPJPE21 |
|---|---:|
| 原始 released HaWoR | 81.2635 |
| 联合 best300 的 `raw` | 29.7145 |
| best300 最后一轮 `candidate` | 27.5676 |
| best300 最终输出 | 29.5167 |

右手的主要收益已经体现在联合适配后的 raw 中。左、右 raw 的方向相反，意味着仅用“纯 TWM 是否见过左手”解释整套联合模型结果并不充分。

### 5.3 Full val 上也观察到同方向的不对称

| 手别 | 原始 HaWoR MPJPE21 | best300 raw | best300 candidate | best300 final |
|---|---:|---:|---:|---:|
| 左手143状态 | 63.8116 | 149.8865 | 92.0075 | 86.4852 |
| 右手434状态 | 87.2191 | 21.4185 | — | 23.0937 |

左 raw 的位置退化不是只在 test 一组平均数上出现；val 上也存在。现有结果支持优先排查视觉联合适配与后续校正的协调，而非先假定预训练完全没有左手。

### 5.4 录制之间差异较大

test 左手9条录制中 **5条改善、4条退化**，录制级 `final−baseline` MPJPE 差值的中位数为 **−1.854mm**；右手7条录制全部改善。左手平均数受若干严重退化录制影响，不能写成“所有左手录制均失败”。最严重的一条为 subject06／`20200918_113405`，63状态的 MPJPE 从原始 **32.5804→raw130.8148→final99.9143mm**，最终退化 **67.3339mm**。

val 左手2条录制也是1条改善、1条退化。较差的一条 subject01／`142446` 为 **66.868→raw209.763→final118.616mm**。这些分散情况支持“左手泛化／定位不稳定”的诊断，但主体、动作与拍摄条件同时变化，尚不能单独指定因果。保留全部原定录制，不按结果删除样本或修改test。

### 5.5 不能据此作出的因果结论

- `best.raw` 是在最终联合目标下共同训练出来的 HaWoR 前端输出。它不是单独训练的 “HaWoR-only finetune” 实验；其变化同时受后续网络梯度与联合优化影响。
- 最后一轮 `candidate` 已经包含首轮 TWM 处理及此前最终状态的跨帧历史。因此 `candidate→final` 的退化只说明最后一次 TWM 更新在此子集上存在误改，不能当作整个 TWM 的独立贡献，也不能据此认定删去全部 TWM 会更好。
- 原始预训练 TWM 与联合微调后的 TWM 不是同一组权重。本轮没有新增分侧纯运动模型对照，不能声称已测明原始预训练 TWM 的左手恢复能力。

## 6. 当前候选解释与尚缺的验证

| 候选解释 | 当前证据 | 结论状态 |
|---|---|---|
| 预训练没有左手或数量严重不足 | train window 317,226左／319,384右，完整遍历16次以上 | **不支持** |
| loader／采样器把左手跳过 | 真实窗口统计、源码与 full-coverage 合同一致 | **不支持** |
| 联合适配后的视觉 raw 出现左手位置偏差 | test 与 val 的左 raw 腕／绝对MPJPE大幅变差，而 test 腕对齐MPJPE改善 | **已观察到现象，优化根因未定** |
| 最后一轮校正存在误改 | test 左 candidate60.21→final63.74mm，右27.57→29.52mm | **已观察到，非全部TWM消融** |
| 左手 shape／模型约定差异 | 项目已知跨源左 shapedirs 约定差异；当前 TWM β conditioning 关闭，视觉各分支仍预测β，缺少独立shape监督 | **候选风险，尚未坐实为当前bug** |
| 合成运动噪声不覆盖左手特有真实视觉误差 | 当前 v3 对两侧使用同一噪声机制，真实 raw 左右偏差不对称 | **分布差异假设，待配对对照** |
| checkpoint选择／主体动作组成影响侧别结果 | fixed8左手仅40／210状态；test左9录制5好4差，右7条全好，仍是小型有界集合 | **代表性不足已确认，因果贡献未验证** |

源码核对与覆盖统计已经排除了“完全没有左手”这一简单解释。更具体的下一步验证应保持原始评测协议与预选checkpoint，分开检查 raw 的全局位置、预测shape、真实左右坐标映射、纯预训练TWM与联合TWM在同一输入上的更新；需要新增模型实验的部分仍属于后续工作，不能写成本轮已验证结论。

## 7. 产物与复现范围

- D37 预训练源码：`87a2bcc`；[D37运行协议](TWM_TRANSIENT_10K_RUN_2026-10-15.md)、[启动与采样摘要](../reports/twm_transient_v3_10k_launch_20261015.json)。
- 联合训练源码：`eb8d0b98f5e2e27fae9ed6f664399d1eca6f2dae`；正式主比较为验证集事先选定的 `best.pt` step300，终点 `latest.pt` step1000另列；未按test更换best。
- 原始评测与几何汇总：远程项目根下 `evaluations/visual_joint_final_20261016/`；[最终小型汇总](../reports/visual_joint_final_comparison_20261016.json)。
- 本轮左右覆盖／配对诊断：[left_hand_diagnosis_20261016.json](../reports/left_hand_diagnosis_20261016.json)。详细绝对路径、manifest SHA、分侧统计及索引依据的训练源码版本以该产物为准；覆盖索引与派生汇总在Pod内临时脚本执行，未新增网络推理。
- GT 来源为数据源 fitted MANO 参考，不是独立扫描mesh或新增MoCap；视觉评测使用 GT crop、固定相机 world gauge、相同输入与validity。此处结果不是完整官方HaWoR检测／SLAM benchmark，也不证明长序列在线或未来预测性能。
