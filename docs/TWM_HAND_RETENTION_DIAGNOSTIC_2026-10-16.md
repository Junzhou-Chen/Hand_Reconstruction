# TWM左右手能力与联合微调后的保持性诊断

日期：2026-10-16。状态：**只读远程评测已完成；未改损失、模型结构、采样、checkpoint或训练任务，未重新训练。**

本次新增实测支持两个同时成立的判断：**D37预训练TWM具有相近的左右手当前恢复能力；联合微调后，部分合成运动恢复能力在两侧都下降，但联合TWM在真实视觉误差的固定上下文中又优于预训练TWM。** 因此，不能将左手视觉结果退化解释为“没有学过左手”，也不能把“直接换回预训练TWM”当成已经验证的修复。

完整联合诊断及相关几何、梯度证据以主报告和[数值汇总](../reports/joint_failure_deep_diagnosis_20261016.json)为准。本文只解释纯数值TWM配对测试及固定真实状态干预。先前的[左右手数据覆盖核查](LEFT_HAND_COVERAGE_AND_FAILURE_DIAGNOSIS_2026-10-16.md)与[完整视觉模型评测](VISUAL_JOINT_FINAL_EVALUATION_2026-10-16.md)保留各自证据边界。

## 1. 评测对象与可复现协议

### 1.1 三个固定权重

| 名称 | 权重身份 | 在本次诊断中的角色 |
|---|---|---|
| Pretrain | D37纯轨迹预训练best，step10,000 | 原先验能力参照 |
| Joint best | D40完整联合checkpoint中提取的TWM，step300 | 测试前由原验证协议选定的主比较 |
| Joint last | D40终点完整联合checkpoint中提取的TWM，step1,000 | 独立报告训练终点，不据本次结果替换best |

Joint内仅提取`pipeline.feedback.twm.`前缀的tensor，严格加载到原`ProbabilisticTWM`；未运行HaWoR或视觉adapter来构造纯motion测试输入。原TWM模型配置、`position_feature_scale=0.05 m`、99D状态定义、manifest及完整D37 replay config均与预训练一致。

- D37训练源码：`87a2bcce50eb01075e6f5e6959ec536e26f1be58`。
- 本次pure-motion及real-state干预执行源码：`88c7c00cdc5548aa2febddbb8366ce1b857a171b`。
- D37四库manifest：`/sciclone/data10/jchen57/Hand_Reconstruction/prepared/twm_four_sources_99d_v1/manifest.json`。
- Manifest SHA-256：`f733aa15cd18ed6f57ee0f16ebdadb3bd0d52924c7d99d08cea988544710b26f`。
- 预训练来源checkpoint SHA-256：`9f500a283109fbad75f380281808b16781a335550b99853d66832cb120bef76c`。

执行时核对D37数值模型、几何、loader、corruption及evaluator相关源码hash；joint中记录的预训练来源与实际文件SHA一致。三个模型的tensor及源checkpoint前后hash检查均通过；无optimizer、无反向更新，`eval()`、`no_grad()`，保持原CUDA bf16评测精度。

### 1.2 固定且配对的运动窗口

使用训练时同一`ForecastTrajectoryWindows`和既定recording-disjoint **val split**，不是新增独立motion test：

- OakInk2、HOT3D、UmeTrack、DexYCB各分left/right，共8格。
- 每格固定128窗，共**1,024个窗口**。先均衡分配到recording，再分散选取录制内窗口中心；选择seed为20261016。选择不读取姿态误差或模型结果。
- History30、min_future3、stride4、最多60个future target，与D37一致；未来按实际时间戳，尾部和时间间断用原mask，不补造GT。
- 五种原D37 v3模式：`clean`、`noisy`、`reanchor`、`position_burst4`、`missing`。同一1,024窗在五模式下重复测试，不是5,120个独立运动样本。
- Corruption沿用原`prepare_batch`，按每格每模式重置原eval seed `config.seed+123`；三个模型使用完全相同窗口、minibatch边界及输入。所有batch输入与目标hash配对检查通过。
- TWM只接收数值history、raw、candidate及原mask、dt、handedness等条件；**不接收图像、RGB features、未来状态或未来GT。** `forecast_dt`是查询时间。`condition_on_beta=false`，原sampler读取的源β始终被false validity屏蔽，不成为shape条件。

本文左右手表格是**每侧四源等权宏平均**；未来只平均该时域有eligible target的来源。并非按四库原始窗口占比加权，也不是把左右手的动作难度或运动速度配成一致。当前窗口可重叠；recording均衡不使窗口变成统计独立样本。

### 1.3 指标与参考标签

- Wrist：解剖腕位置的欧氏误差，单位mm；在原受损历史腕anchor中评估，刚体坐标变化不改变距离。
- Root：腕全局旋转SO(3) geodesic error，单位°。
- Finger：15个局部指关节SO(3) geodesic error的均值，单位°。
- 当前输入基线是原evaluator中的**因果填充candidate**；缺失raw的占位值不能直接当作有效完整观测。
- 退化率阈值为腕增加>1mm，root/finger各增加>0.1°；它们用于诊断保护性，不能当成任务成功率。
- 当前90% coverage是原anchor空间对角Gaussian的xyz联合椭球覆盖，阈值χ²(3)=6.251389；std为三个轴std的均值。未来coverage则是混合CDF的逐轴90%边缘覆盖，两者不可混用。
- 参考是四库已准备的**拟合MANO参数**；本轮不构成独立MoCap精度、关节/mesh几何精度或物理正确性的证明。

## 2. 原预训练TWM没有明显的左手能力缺口

普通`noisy`模式，左右手candidate输入腕误差均为4.55361mm。下表是输出误差，越低越好：

| 模型 | 腕L / R，mm | Root L / R，° | Finger L / R，° |
|---|---:|---:|---:|
| Pretrain | 2.78207 / 2.94546 | 1.28895 / 1.32576 | 0.64128 / 0.63321 |
| Joint best300 | 3.54833 / 3.68888 | 1.61034 / 1.72120 | 0.87148 / 0.87821 |
| Joint last1000 | 3.43998 / 3.52727 | 1.50017 / 1.57627 | 0.82404 / 0.81887 |

Pretrain在左右手都降低了输入误差，root/finger也相近。加上实际训练窗口左317,226、右319,384的覆盖记录，本轮证据不支持“预训练没见过左手”或“左手先验整体失效”。这不是左右严格等性能的统计证明：来源内动作、主体与参考标签差异仍可能与手别混杂。

Joint的普通恢复误差在两侧都高于Pretrain，但仍低于各自candidate输入。终点比best在此模式有所恢复，也没有回到Pretrain水平。**这是部分运动能力保持下降，不是专门发生于左手，也不等于完全丧失恢复能力。**

## 3. 下降具体发生在哪些条件

### 3.1 干净状态被更多地改动

`clean`中输入就是参考状态；非零输出误差反映模型引入了不必要的修改：

| 模型 | 腕L / R，mm | Finger L / R，° |
|---|---:|---:|
| Pretrain | 0.56654 / 0.59827 | 0.11612 / 0.11845 |
| Joint best300 | 0.88919 / 0.94331 | 0.22158 / 0.23852 |
| Joint last1000 | 0.80436 / 0.80534 | 0.18774 / 0.19501 |

两侧都有干净输入保护能力下降；同时，Pretrain原本也不是严格identity映射。均值不能替代>1mm等阈值的退化率，详细分布与counts保存在诊断数值中。

### 3.2 短时腕位置突变恢复变弱

`position_burst4`只注入局部腕位置事件，candidate输入腕误差为6.46452mm：

| 模型 | 输出腕L / R，mm |
|---|---:|
| Pretrain | 4.06003 / 4.51642 |
| Joint best300 | 5.03067 / 5.38296 |
| Joint last1000 | 4.82031 / 5.16931 |

这一条件比普通混合噪声更明确地显示了局部位置恢复能力下降，左右都存在。它是合成短事件，不代表已复现真实HaWoR的误差分布。

### 3.3 不是所有模式都退化

`reanchor`中当前输入近准、过去history受损，candidate腕误差为0.28250mm：

| 模型 | 输出腕L / R，mm |
|---|---:|
| Pretrain | 1.28087 / 1.39409 |
| Joint best300 | 1.34141 / 1.27851 |
| Joint last1000 | 1.24429 / 1.19569 |

右手best较Pretrain改善，终点两侧均较Pretrain改善。三个模型仍把近准输入改差，因此“错误历史压过当前观测”是仍存在的失败条件；不能将这张表解释成联合训练在所有模式上都更差。`missing`已执行同样配对测试，完整分格数据保留在JSON中，不以未在本文逐项展开代替结论。

## 4. Coverage提高，不能解释为估计更准确

普通`noisy`模式：

| 模型 | 当前90% xyz coverage L / R | 平均axis std L / R，mm |
|---|---:|---:|
| Pretrain | 79.8828% / 78.1250% | 1.689 / 1.708 |
| Joint best300 | 93.7500% / 91.9920% | 4.347 / 3.596 |
| Joint last1000 | 90.2340% / 88.0860% | 3.126 / 2.783 |

Joint误差均值变大，同时输出区间也变宽，所以coverage升高不能证明恢复更准，更不能称可靠性已校准。这里只检查一个合成模式、一个bounded val population；真实视觉test的coverage不能由这张表替代。当前矩阵Fisher rotation concentration同样只是分布参数，不是已验证的正确概率。

## 5. 未来能力：短期仍有效，长时优势较弱

未来输出沿用原evaluator的整段最高权重分支，不用GT选择分支。`restored_hold`将**该模型自身恢复后的当前状态**保持到未来；因此它是模型相关基线，不能把差值当成独立于当前恢复的纯预测贡献。

普通`noisy`模式的腕终点误差：

| 模型 | 1秒 L / R，mm | 2秒 L / R，mm |
|---|---:|---:|
| Pretrain | 99.1165 / 102.4817 | 128.394 / 156.626 |
| Joint best300 | 98.8492 / 101.9450 | 135.093 / 161.606 |
| Joint last1000 | 103.3747 / 101.5630 | 134.173 / 160.276 |

1秒的对应hold约121–122mm，模型预测仍有收益；联合best没有在这一时域整体退化。2秒只有OakInk2、HOT3D、UmeTrack三源具有有效目标，**不包含DexYCB**；对应hold约左134.8、右159.8mm，各模型具体hold与eligible counts见JSON。相较Pretrain，Joint长时预测优势变弱，部分格子不如hold；不能据此说所有未来预测失效。

时域资格由最近有效源时间戳决定，容差为最后观测dt的0.6倍。尚未验证长视频滚动预测、跨秒闭环、动作意图或未来分布的充分性。

## 6. 真实视觉状态干预：换回Pretrain不保证改善

### 6.1 固定joint best的真实上下文，只换数值TWM

使用完整视觉评测已保存的joint best raw、最后candidate及已完成历史，重建原anchor、mask和dt。所有模型接收同一固定上下文，以下只比较一次最后pass的腕输出：

| 替换的TWM | Test左腕，mm | Test右腕，mm |
|---|---:|---:|
| Pretrain | 70.849 | 29.069 |
| Joint best300 | 62.512 | 26.561 |
| Joint last1000 | 60.088 | 26.054 |

这里Joint TWM在真实联合状态上下文中优于Pretrain。因此，纯motion误差上升与真实误差适配收益可以同时发生，不能仅凭纯motion表就恢复旧权重。last1000这一行**不是完整last pipeline的效果**：raw/candidate/H仍固定来自best300。

原best replay的最大差为腕0.693mm、root0.200°、finger0.076°；在batch32 bf16重放的预定门槛1mm / 0.25° / 0.5°内。重放足以支持本轮bounded干预比较，不是逐bit相等。中间candidate已经包含前一轮TWM和历史影响，所以该干预不能给出整个TWM、adapter或历史分支的独立贡献。

### 6.2 历史会影响修正，但不能直接删掉

同一joint best raw/candidate，使用best TWM：

| 固定上下文控制 | Test左腕，mm | Test右腕，mm |
|---|---:|---:|
| 已保存的完成历史 | 62.512 | 26.561 |
| 仅将过去换成GT参考历史 | 46.710 | 19.269 |
| 清空过去历史 | 59.443 | 25.361 |

Oracle只使用当前之前的参考状态，没有把当前或未来GT作为TWM输入；但参考过去仍不可部署。Oracle改善说明本组raw/candidate中包含可被更好历史条件利用的信息，不能证明某个可部署history修复器已能获得同样收益。

Empty history在这个test固定上下文中有小幅收益，**val左手却从保存历史86.070mm变为93.212mm，反而变差**。因此不存在本轮证据支持的“去掉history一定更好”。历史可能提供信息，也可能传递偏差；其可靠性与适配需要分情况分析。

### 6.3 原始HaWoR误差不是合成扰动的简单替代

另一个控制直接令released HaWoR raw同时充当当前raw/candidate，过去也全部由raw组成，不经过视觉adapter：

| 控制 | Pretrain左 / 右腕，mm | Joint best左 / 右腕，mm |
|---|---:|---:|
| 原始raw当前＋原始raw过去 | 72.322 / 82.419 | 71.464 / 82.582 |
| 原始raw当前＋oracle参考过去 | 31.943 / 35.579 | 45.872 / 58.162 |

原始baseline腕误差为左72.013、右81.530mm。使用真实raw历史时，两个TWM都没有稳定明显优于原始raw；换成干净参考过去后，两者都能恢复很多，但Pretrain在该oracle控制下更强。

这支持“真实相关偏差历史难于本次合成扰动”和“联合适配存在能力取舍”的解释，也显示Pretrain并非没有处理真实当前误差的潜力。它仍不能唯一定位到history训练不足、loss、noise calibration或表示错误中的某一项：oracle同时改变历史数值与anchor，固定candidate又来自既有系统，尚无相应训练消融。

## 7. 目前能与不能据此决定的事

**已被本次实测支持：**

1. Pretrain对左右手均有当前恢复能力，左右在普通噪声下相近；“缺左手数据”不是当前主要解释。
2. 联合微调后的普通恢复、短时位置事件及clean保护能力在两侧均下降；这是真实测到的保持性问题。
3. 联合TWM又适配了现有视觉系统的真实固定上下文，直接换回Pretrain不是已验证修复。
4. 历史质量改变输出，GT过去帮助大；清空历史不跨split稳定有效。
5. 当前uncertainty参数、coverage和均值精度需要分开检查；2秒预测优势弱于1秒。

**仍需保留为未决问题：**

- 联合视觉loss与motion replay如何产生这种取舍，是否存在梯度冲突或监督尺度问题；需要主诊断中的实际梯度与几何证据，不能仅凭这些误差表推断因果。
- 真实左手绝对位置失败中，HaWoR前端、candidate几何、shape/相机转换与最终TWM各自占多少；本轮纯motion保持性测试本身不能分解。
- 使用哪种loss、回放权重、分侧采样、history监督或训练阶段安排更好；本次没有进行新的训练对照，不将候选修改记为已定设计。

## 8. 产物与审计位置

所有数据准备、模型推理和评测在授权Pod `jchen57-fineturn-tl4zr` 远程执行；本机仅保存源码、文档与小型数值汇总。远程产物根目录：

`/sciclone/data10/jchen57/Hand_Reconstruction/evaluations/joint_failure_deep_diagnosis_20261016/`

- `twm_paired/summary.json`：完整source×hand×mode指标、counts、配对hash、配置及只读检查。
- `twm_paired/selection.json`：全部1,024个窗口的来源、recording、中心及索引。
- `twm_paired/<label>_per_window.csv`：完整逐窗口原始指标，仅远程保留。
- `real_states_val/report.json`、`real_states_test/report.json`：冻结真实状态的数值TWM干预、replay parity及权重身份。
- 仓库汇总：[joint_failure_deep_diagnosis_20261016.json](../reports/joint_failure_deep_diagnosis_20261016.json)。模型、数据、逐窗口原始日志和干预tensor不上传Git。

脚本：[diagnose_twm_handedness.py](../scripts/diagnose_twm_handedness.py)、[diagnose_real_state_twm.py](../scripts/diagnose_real_state_twm.py)。

真实状态控制继承原视觉val/test的GT crop、固定camera gauge、拟合MANO参考、有限录制和短clip历史范围；不是完整HaWoR官方benchmark、长序列部署或独立主体泛化证明。本轮保持预先best300作为主比较，**未按诊断或test重新选择checkpoint，也未启动新训练**。
