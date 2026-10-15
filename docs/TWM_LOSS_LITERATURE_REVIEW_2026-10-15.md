# TWM损失设计：论文、实现与下一步对照建议

**用户后续决定（2026-10-15）：损失调整放到下个版本，当前先检查效果。** 本文候选不应用于D37本轮，不改变其六项目标；“下个版本”尚未指定最终公式、系数或训练启动。

日期：2026-10-15。用户质疑当前六项加权损失，要求广泛参考world model及相近恢复工作。本轮核读12篇原文/补充训练协议，并核查相关官方实现；这是研究建议，没有修改D37源码、配置、权重或正在运行的训练，也没有完成新消融。现行路线仍是motion-only当前恢复、辅助未来预测、99D动态状态与beta10另存。

判断：不能由“损失项多”断定设计错误。更具体的问题是，每项是否有任务依据、尺度和有效样本权重是否清楚、辅助任务是否改变主任务、训练与部署选择的输出是否一致。当前六项是工程起点，不是一篇参考论文的完整配方，也未通过逐项消融获得支持。

## 1. World model实际如何训练

| 工作与核读范围 | 实际目标/权重 | 可迁移部分与边界 |
|---|---|---|
| [WestWorld v2 §3–4.4、C/D](https://arxiv.org/html/2603.14392v2) | 状态分箱后的next-token CE；每通道训练集min/max归一，256 bins。官方主路径GT为onehot，Gaussian soft bins用于输入embedding；当前各通道support同为[0,1]，通道权重均匀；mask后时间/批次取mean。 | 统一概率预测、明确尺度/mask。任务给定历史和未来控制action预测机器人状态；无action的人手未来和当前受损恢复不同。没有附加MoE负载均衡或物理守恒loss。不能把6D旋转的6个坐标独立分箱直接当SO(3)分布。 |
| [DreamerV3，World model learning/Robust predictions](https://arxiv.org/html/2301.04104) | 观测/reward/continuation预测NLL＋动力学KL＋0.1表征KL；后两者分别对posterior/prior stop-gradient，KL有1nat freebits。当前官方rec/rew/con/dyn/rep系数为1/1/1/1/0.1。 | 不支持“world model都没有加权”。值得借鉴target分工、尺度处理和梯度方向；其latent posterior/prior、reward/critic不是我们现有模块，不应凭空增加。 |
| [TD-MPC2 §3.1、H Table8](https://arxiv.org/html/2310.16828) | 正文简式三项，附录与代码为20×latent consistency＋0.1×reward CE＋0.1×value CE，时间折扣0.5；future latent target stop-gradient。reward/value为symlog twohot。 | 目标是适合action控制的latent与return，刻意不重建原始未来。可借target隔离与rollout归一化，不能照搬reward/value目标代替精确手几何。 |

WestWorld的核心式是状态categorical likelihood；Dreamer/TD-MPC2明确保留多项目标。stop-gradient隔离指定target路径，不意味着共享表征不受其他损失影响，也不构成真实物理正确性保证。

官方关键代码（以2026-10-15读取的公开版本为准；main不是永久版本）：[WestWorld forward](https://github.com/511205787/WestWorld/blob/main/models/WestWorld/westworld.py#L669)、[CE归约](https://github.com/511205787/WestWorld/blob/main/models/WestWorld/westworld_utils.py#L117)、[dense MoE](https://github.com/511205787/WestWorld/blob/main/models/WestWorld/mamba_moe.py#L140)；[Dreamer配置](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml#L80)、[KL实现](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/rssm.py#L113)；[TD-MPC2配置](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/config.yaml#L14)、[update](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/tdmpc2.py#L239)。WestWorld可选KD默认关闭，不并入其默认目标；TD-MPC2当前代码可选episodic termination不混写成原论文三项。

可回查版本：Dreamer论文v2、配置固定commit `e3f02248693a79dc8b0ebd62c93683888ddaccfe`；TD-MPC2论文v2、配置固定commit `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`，上述两份固定SHA配置已核。WestWorld关键路径本轮读main，其history显示`0a90776aeaa43f47c97f3c05eb75418b377b73ae`，未冒称本轮全部SHA raw逐字核对成功。

## 2. 手部先验与受损恢复的直接参考

| 工作、任务与上下文 | 核实的训练目标 | 对我们意味着什么 |
|---|---|---|
| [StableHand v1 §3.2–3.3 Eq5–10](https://arxiv.org/html/2605.18553v1)：150帧双手视频条件恢复 | 生成器是FM＋0.1×二阶平滑＋5×腕位置MSE；QN另训log1p(error) MSE＋1.0×批内1D Wasserstein，之后冻结。 | 腕项专门处理3D平移被45D手指项淹没，说明FM仍有维度/尺度权衡。可借错误程度调制生成路径；不能把quality当真值硬锁，不能把其视觉/scene/camera条件称纯轨迹。 |
| [HMP完整论文/补充§3](https://arxiv.org/pdf/2312.16737)：MoCap先验＋视频拟合 | VAE重建、FK几何与KL；发布通用配置的rot/orient/position/KL系数1/1/10/1e-5，另有KL annealing。视频拟合是另一套七项加权能量、分阶段调权。 | 应分开先验训练与观测恢复。论文local hand prior排除global orientation；发布类含global encoder不等于学到world腕动力学。FK可作几何参考，不能直接搬七项test-time fitting energy当预训练目标。 |
| [HaWoR完整PDF §8.1–8.2](https://arxiv.org/pdf/2501.02973)：视觉重建＋缺段补全 | 视觉0.05×3D L1＋0.01×2D L1＋0.001×MANO MSE。infiller的translation/rootrot/fingerpose/shape L1系数为0.05/2/2/0.05。 | infiller是有后侧上下文的in-betweening；公开120帧、双手218D补全仅覆盖missing位置。保留valid预测避免改坏，但也不能修复valid-but-wrong。不是我们的causal当前恢复或未来分布预测。 |
| [DIP §3.2–4.1](https://arxiv.org/html/2508.01835v1)：纯运动受损手恢复 | diffusion clean-state重建＋motion-state CE＋状态相关kinetic/stability；系数50/500/1000。§4.1还说明加入几何和trajectory consistency项，不能只看Eq8就声称目标完整只有四项。 | 固定原始估计与当前去噪候选分开，前向过程用clean→observed残差。是直接前作；其contact/hand-object状态标签及local pose/shape范围，不直接满足world wrist任务。 |

StableHand版本必须分开：[Enhanced说明](https://github.com/huajian-zeng/stablehand/blob/main/docs/enhanced_version.md)将原腕辅助项替换为joint velocity/acceleration监督，仍保留smoothness，但[training code仍TODO](https://github.com/huajian-zeng/stablehand)。v1论文系数不能冒充增强版训练代码事实。其flow velocity是生成时间导数，二阶平滑是向零正则，均不同于我们匹配GT物理速度。

HMP的[通用配置](https://github.com/enesduran/HMP/blob/main/configs/generative.yaml)、[loss执行](https://github.com/enesduran/HMP/blob/main/src/nemf/generative.py)需连读：`output_trans:false`时配置中列出的其他velocity/contact分支没有因此自动执行。其[视频fitting](https://github.com/enesduran/HMP/blob/main/src/fitting_app.py)独立处理wrist并冻结global latent。HaWoR训练代码未完整公开，infiller系数依据PDF补充，不称代码复现。

纯轨迹预训练恢复不是尚无先例的问题，DIP/HMP/PoseBERT等已覆盖部分范式。本项目候选区别需要落在world wrist、受损当前、图像证据再取证及最终状态反馈的具体组合与公平实验，不能把“motion prior用于纠错”整体宣称首次。

## 3. 用一个生成目标覆盖缺失、噪声与未来

| 工作与核读范围 | 目标及实际实现 | 可迁移部分与限制 |
|---|---|---|
| [Diffusion Forcing v3 §3、D.2–D.6](https://arxiv.org/html/2407.01392v3) | 各token独立噪声级的denoising目标。prediction target随域不同：视频v、planning/imitation x0、time series v；视频另有history-fused min-SNR，非图像未用该boost。 | 最接近带噪历史、当前与完全未知未来的统一建模。已知生成噪声级不等于未经标定的HaWoR quality。原非视频50步DDIM、视频100步，不能当一次前向接口。 |
| [CSDI §4 Eq7](https://arxiv.org/pdf/2107.03502) | 条件noise-MSE，仅在人工隐藏的有标签target位置计算，除target元素数；condition原样保留，无独立几何/速度项。 | 可借mask与partial-label训练；原版missing imputation不等于修复错误condition，双向插补也不能直接称causal。默认50去噪步/样本。 |
| [MDM v2 Eq2–6、§4.1](https://arxiv.org/html/2209.14916v2) | clean-motion MSE；可加FK/GT帧差/contact。HumanML/KIT默认三辅助项0，action-to-motion官方命令三项1。 | 辅助项依表示和任务选择。冗余表示已含位置/速度/contact会改变是否需要额外项；不能用某任务single MSE证明所有辅助项应删。原1000步，后发50步模型需分开。 |
| [HumanMAC v3 §4、B](https://arxiv.org/html/2302.03665v3) | 截断DCT上的noise-MSE；DCT-completion反复注入观测prefix。当前官方代码还以0.8概率输入padded-prefix DCT modulation。 | 单目标forecast先例，但H36M处理丢弃root，不能作为world腕恢复证据；低频截断可能影响短事件，硬保留prefix也不学修复错误历史。1000训练噪声级/100步DDIM。 |
| [Riemannian Flow Matching v3 §3、A](https://arxiv.org/html/2302.03660v3) | 回归生成路径切向量，使用流形metric：E‖vθ(Xu,u)−u*(Xu,u)‖²g。 | 为R³×SO(3)^16上的生成提供数学结构；metric、先验和路径仍要设计。保证几何流形性质不等于人手关节限位/物理定律，也不是已验证手部运动方案。 |

关键官方代码：[DF paper分支](https://github.com/buoyancy99/diffusion-forcing/blob/paper/algorithms/diffusion_forcing/df_base.py)、[SNR/target](https://github.com/buoyancy99/diffusion-forcing/blob/paper/algorithms/diffusion_forcing/models/diffusion_transition.py)；[CSDI masked loss](https://github.com/ermongroup/CSDI/blob/main/main_model.py#L101-L164)；[MDM训练命令](https://github.com/GuyTevet/motion-diffusion-model#train-your-own-mdm)、[loss](https://github.com/GuyTevet/motion-diffusion-model/blob/main/diffusion/gaussian_diffusion.py)；[HumanMAC training](https://github.com/LinghaoChan/HumanMAC/blob/main/utils/training.py#L72-L88)、[completion](https://github.com/LinghaoChan/HumanMAC/blob/main/models/diffusion.py#L111-L150)。DF main已变更时序实现，原论文机制以paper分支为准。

## 4. 数据、GT和评测职责不混用

本轮是loss调研，不改变四库训练池与recording-disjoint split。以下记录参考工作的职责；不把身体/机器人标签或视觉拟合参数等同高精度手GT。

| 工作 | 训练/参考来源与评测角色 | 输入上下文、split迁移边界 |
|---|---|---|
| WestWorld | UniTraj的80模拟环境＋9真实机器人库预训练，未见系统/少样本/控制评测 | 已知future action；原文C/D划分不能替代本项目recording split |
| DreamerV3 / TD-MPC2 | 环境交互中的observation/action/reward监督；多环境控制评测 | RL数据分布与目标不同，无独立手姿态GT；不挪用其split或控制收益作重建证据 |
| StableHand | 八库约10M帧服务QN；生成器按HOT3D/ARCTIC训练与评测 | 150帧视频/几何条件，原官方任务split；八库不成为本项目pure-motion GT清单 |
| HMP | GRAB/TCDHands/SAMP等手运动MoCap先验，HO3D/DexYCB视频恢复评测 | 128帧局部运动，先验训练与视频test-time fitting分开；静止相机与world腕假设不同 |
| HaWoR | 视觉HOT3D/ARCTIC/DexYCB/HO3D；infiller主要HOT3D可见/缺失段 | 视觉16帧双向、补全120帧；源标签多为MANO拟合，按论文/官方划分，不能称glove轨迹 |
| DIP | DexYCB训练合成受损motion，HO3Dv2跨域及TACO补充评测 | 16帧观测序列可有后侧context；参考依源MANO拟合，虽论文称MoCap不扩称独立高精度world GT |
| Diffusion Forcing | 原文视频、控制轨迹、time-series域分别训练评测 | 使用域内固定生成噪声与任务划分，不是通用视觉误差模型或手部数据协议 |
| CSDI | Physio/空气质量等时间序列，人工隐藏有值项评插补 | 原始missing无监督；可利用后侧观测，不能让未来观测泄漏进当前恢复 |
| MDM | HumanML/KIT文本运动与action-conditioned身体数据分别训练评测 | 按原task标准split，不是手部参考质量排名；condition dropout非视觉错误模拟 |
| HumanMAC | H36M/HumanEva等身体MoCap的future prediction | 原任务split、root-relative，时间频域和world wrist不可混同 |
| RFM | 球面/环面/双曲/mesh等流形数据验证 | 是训练方法与数学机制依据，不是手部数据或current/future benchmark |

## 5. 对当前六项的具体审查

当前事实由[objective](../scripts/train_probabilistic_twm.py#L91)、[运行配置](../config/twm_four_sources_moe_transient_v3_b512_10k.json)和D37固定源码`87a2bcc`核验：

\[
L=G_t+.05N_t+.30N_F+.20G_F^{best}+.05V_F+.01B_{MoE}.
\]

保留的合理基础：当前/未来皆定义在R³×SO(3)^16；99D只是编码，NLL不用99个独立Gaussian；未来mask与实际timestamps已有，四mode对整段固定，不逐帧GT选不同分支。上述论文不构成推翻这些设计的证据。

具体未决问题：

1. **重复监督尚无消融。** 几何与NLL同时训练均值并非数学矛盾，但两组都要有收益依据；GT速度也是额外偏好，不能以“物理”自动保留。
2. **训练选择与推理选择不同。** best-trajectory几何只奖励GT最接近分支；推理按最大mixture weight选。NLL会训练混合概率，但尚未检验二者是否改善同一个部署决策。旧selected ADE与oracle minADE聚合权重不同，差值不能直接归因于选分支错误。
3. **未来能直接移动当前。** future NLL/geometry通过积分初值反传当前读出头；低运动/clean被改坏与此可能有关，但D37尚无因果消融。D34小样本梯度诊断不能挪作D37证明。
4. **平均与系数不等于实际任务优先级。** 当前1帧、未来最多60帧、51维密度归一、几何的root2/finger平均、每窗/每有效点聚合，都影响优化。只删显式系数不消除这些取舍。
5. **MoE正则也要证据。** WestWorld的dense MoE并未使用我们的负载均衡项，不能把0.01说成由它直接推导；routing均匀不证明专家有用。

目前合成fixed validation的clean/reanchor退化、2秒forecast接近hold等是真实记录；它们不单独证明六项loss是原因。[当前恢复](TWM_TRANSIENT_10K_PROGRESS_2026-10-15.md)、[未来预测](TWM_FORECAST_PROGRESS_2026-10-15.md)。

## 6. 推荐先比较的路线：保持接口，建立概率主目标

这是工程建议，尚非用户采纳或已实施的新方案。

优先保持99D/shape契约、当前架构、四库参考和v3输入过程；把恢复与预测各自的conditional NLL作为统计主线，逐项检验额外几何、best-of-K、速度和负载均衡。不是同时换loss、噪声、骨干和数据后再归因。

可把主目标写为任务采样下的归一化概率评分：

\[
\mathcal L_{prob}=\mathbb E_{a\sim\pi_{task}}
\left[-\frac1{d_a}\log p_\theta(X_a\mid C,a)\right],
\qquad a\in\{current\ restoration,\ future\ trajectory\}.
\]

其中C保存受损history、原始当前、candidate和availability，不能把任何高confidence输入当硬真值。当前d=51，future d=51×有效步数，是一个明确的归约选择；任务采样概率仍是权衡，不能包装成“完全没有权重”。它是两任务概率评分，不冒充对所有变量均未加权的严格联合最大似然。

现模型future从确定性current均值积分，未传播完整current分布到未来。因此若宣称完整联合后验，还要说明当前/未来条件独立近似或重新定义条件因子；不能仅把两项NLL相加就声称学到了任意joint uncertainty。

建议分开控制的问题：

- 以现D37为基线，保持其配置完成既有10k；使用可复现预算作辅助项消融，删除项的其余系数先保持，避免混淆“删项”和重新缩放。
- 另做当前NLL主项的诊断对照，判断只学习current restoration能否更好保护clean/reanchor；该对照不训练future，不能把其future变差当反证。
- 独立比较future积分起点使用`stop_gradient(restored_current)`；它只阻断初值直达current head，shared encoder仍受future梯度，不能保证完全隔离。
- 双NLL精简候选与逐项加回几何/GT速度比较。只有当辅助项对独立几何、校准或错误恢复有可复现增益才保留，不由论文先例直接决定最优系数。

不建议因这次调研立即在当前run上变更目标，也不把还未执行的消融写成已验证更好。

## 7. 第二路线：条件denoising/flow，而非只改loss名字

Diffusion Forcing的不同token噪声级、DIP的原始估计/候选分槽、StableHand的受损程度调制可为新协议提供机制参考；RFM支持在R³×SO(3)^16上构造合法旋转的生成路径。

必须区分两种扰动：

\[
Z=C(S,\xi_{obs}),\qquad X_u=\operatorname{GenerativePath}(\epsilon,S;u).
\]

前者模拟HaWoR/历史反馈错误，后者是训练生成模型的人工路径。Z可以作为固定的受损证据条件，干净S才是生成目标；若把Z对应坐标硬锁为输出真值，模型就无法修复这些错误。固定条件本身不妨碍纠错，但训练需要受损观测到干净目标的配对。物理时间t与生成时间u也不同。

条件RFM候选主项为

\[
\mathcal L_{CFM}=\mathbb E\left\|
v_\theta(X_u,u\mid C,M)-\dot X_u
\right\|_{g}^{2}.
\]

当前恢复和未来预测由target mask/query组织，历史只允许当前时刻以前的可用观测；高质量也不硬锁。位置尺度、SO(3)度量、部位/时间归约、任务与噪声级采样仍要明确；不承诺一条MSE自动解决所有权衡。尤其不能把99D直接平摊后宣称腕与手指自然平衡，StableHand的腕辅助项正提示该问题。

这条路线会改变输出头、训练噪声和采样：多次网络调用、样本决策与不确定性估计均要验收。最终99D仍可保持，现有Gaussian/matrix-Fisher解析NLL、四分支概率和D35单次current NLL训练接口不能自动保留。有限采样covariance也不等于已校准可靠性。故建议作为与概率主线并列的后续方案，暂不把它定为必须替代。

## 8. 任何候选都要通过的实验

固定独立recording split、GT与坐标/米制单位、同一合成扰动种子和训练窗口曝光量。至少同时报告：current腕/root/finger、clean与reanchor退化率、受控单帧/3–4帧异常逐帧恢复、运动突变保持、分源/分时域forecast与hold/CV、匹配权重的分支选择指标、coverage及区间宽度、推理调用数/延迟。2秒仅有有效标签的来源参与；指标与此前口径一致，不因更换目标或输出分布选择性报喜。

在这些通过后，再检验真实HaWoR错误和自身历史回灌。统计先验、合法SO(3)、训练loss较低或轨迹更平滑，都不能替代真实几何误差和独立物理验证。
