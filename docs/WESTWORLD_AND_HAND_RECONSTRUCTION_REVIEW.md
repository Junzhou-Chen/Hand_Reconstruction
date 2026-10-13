# WestWorld 与视觉三维手部重建方法精读

核验日期：2026-10-13。覆盖 WestWorld，以及单图网格重建、视频世界坐标恢复、运动先验和生成式恢复的代表方法。论文事实、公开实现和对 HandLoop 的研究判断分别陈述；没有运行模型或复现性能。现行研究方向仍以 [START_HERE](../HandLoop_Project/START_HERE.md) 和用户后续决定为准。

主要判断是：WestWorld 提供结构化轨迹建模的参考，但不直接训练视觉受损当前状态恢复。手部领域已经存在独立运动先验、质量引导恢复及候选引导图像查询。HandLoop 需要检验这些机制在真实测量错误和递归反馈下的作用，不能把任何单一机制的存在当成新颖性证明。

## WestWorld 的任务与模型

阅读版本为 [arXiv 2603.14392v2](https://arxiv.org/html/2603.14392v2)，2026-05-19 更新；[作者仓库](https://github.com/511205787/WestWorld)标注 ICML 2026 Spotlight。不是同名网页智能体 benchmark，也不以早期 workshop 短稿代替完整论文。

其目标可概括为已知历史状态、历史 action 与未来 action 序列，预测未来状态：

\[
p_\theta(s_{t+1:t+H}\mid s_{\le t},a_{\le t+H-1}).
\]

这里 action 是机器人控制量。§4 / Fig.1 的流程是：按标量通道归一化和离散化；把运动树经 LCRS 转换后的遍历序号编码到 token；用通道 self-attention 与 action cross-attention 交换信息；通过 Mamba 风格时序 SSM 和 system-conditioned MoE 建模，再预测未来状态的分箱分布。未来 query 支持一次前向产生多步输出，监督为状态 next-token cross-entropy。[原论文 §3–4](https://arxiv.org/html/2603.14392v2#S3)

论文在 80 个模拟任务环境与 9 个真实机器人数据集上预训练；zero-shot 测试包含与训练形态相似的未见环境。其指标首先针对预测误差与控制收益，v2 附录 E.1 另补物理单位误差。Go1 部署经过蒸馏和下游微调，不能称为原始大模型直接部署。[论文 §5、附录 E/G](https://arxiv.org/html/2603.14392v2#S5)、[作者实验页](https://westworldrobot.github.io/)

### 官方实现补充

源码核验版本为 `0a90776aeaa43f47c97f3c05eb75418b377b73ae`：

- [配置](https://github.com/511205787/WestWorld/blob/0a90776aeaa43f47c97f3c05eb75418b377b73ae/configs/method/WestWorld.yaml)明确为 256 bins、6 blocks、hidden 256、4 attention heads、4 experts。这些是 WestWorld 配置，不是本项目已经验证的超参。
- [westworld.py](https://github.com/511205787/WestWorld/blob/0a90776aeaa43f47c97f3c05eb75418b377b73ae/models/WestWorld/westworld.py)的 `forward` 保留 action 条件，遮住前缀后的 observation，并在 observation 维计算移位监督；随机 token masking 的默认值为 0。未来遮罩任务不能直接等同于假高置信度、局部偏移和错误历史下的当前状态恢复。
- 同文件的结构编码通过节点索引加入 embedding；当前调用设置 `spatial_mask=None`。不能把结构 embedding 描述成硬性关节约束或物理求解器。

论文正文 §4.4 / 附录 F 描述并行多步预测，而附录 D.3 有“所有方法 autoregressive rollout”的表述。公开 `forward` 支持前一种机制；若复现速度/rollout，应检查具体 evaluation 路径。这是协议核验点，不据此断言实验无效。

### 对 HandLoop 的含义

我们没有真实控制 action，需要学习的是受损状态条件恢复：

\[
U_t^{(k)}=W_\theta(\mathcal H_t,Z_t,S_t^{(k)},M_t,w_t^{(k)}).
\]

这里 TWM 输出恢复建议，最终状态仍由 Corrector 结合图像证据产生。这与 action-conditioned forecasting 是不同的条件问题。把 action 删掉、把机器人 token 换成人手 token，并不会自动获得当前恢复能力。可借鉴的是把骨架含义与时序信息显式组织起来；对固定人手骨架，joint identity、父子关系或图结构是否比 LCRS 编码更合适，应做简单对照。跨机器人 MoE、256-bin 输出与 MPPI 也没有因为论文有效就成为本项目必需模块。

纯轨迹模型可学习运动统计与结构关系，但不能由此声称识别了人的控制意图、接触力或完全遮挡时唯一的真实姿态。这与当前项目的任务边界一致。

## 单图重建如何使用图像和几何

| 方法 | 输入与输出 | 方法重点 | 与 HandLoop 的关系 |
|---|---|---|---|
| MobRecon，CVPR 2022 | 单图 → wrist-relative mesh | 2D joint pooling、joint-to-vertex lifting、轻量 mesh decoder | 图像局部证据与结构化解码已有；增强一致性不等于时间先验 |
| DIR，ICCV 2023 | 双手单图 → MANO、相对位移、camera | 多轮 joint 投影采样、GCN、双手 attention、图像特征更新 | 候选引导的多轮图像交互是明确前作 |
| HaMeR，CVPR 2024 | 单手 crop → MANO 与 camera | 大 ViT、Transformer decoder、大规模混合监督 | 强视觉基线，包含静态 pose/shape prior |
| Hamba，NeurIPS 2024 | 单手 crop → MANO 与 camera | joint-aligned sampling、GCN、双向 Mamba | SSM 扫描 joint 空间序列，不能误称时间 world model |
| WiLoR，CVPR 2025 | 整图检测 → 各手 MANO 与 camera | 多尺度 mesh-aligned refinement | 已有候选投影取特征；公开实现的一次更新须与多轮区分 |
| HandOS，CVPR 2025 | 整图 → box、2D joints、3D vertices、camera | 2D/3D query lifting、hierarchical attention | 说明主流也有直接顶点输出，MANO 参数不是唯一表示 |
| GeoHand，2026 预印本 | 单图 → MANO 与 camera | 单目几何先验与逐轮 joint-conditioned query | 新近、直接相关的候选取证方法，不能忽略 |

上述表格为下面原论文与实现证据的索引，并非同协议性能排名。

**HaMeR。** ViT-H 编码图像，decoder token 通过 cross-attention 回归 MANO pose/shape/camera；监督组合参数、3D joints、2D reprojection 与 adversarial pose/shape prior。官方 head 有可配置 IEF loop，`IEF_ITERS` fallback 为 1，具体轮数须查 checkpoint configuration；它使用缓存 image context，没有候选投影后的局部采样。不能把视觉方法描述成“完全没有先验”，也不能将 decoder 循环与显式重投影查询混为一谈。[论文 §3](https://arxiv.org/html/2312.05251v1#S3)、[官方 head](https://github.com/geopavlakos/hamer/blob/main/hamer/models/heads/mano_head.py)

**WiLoR。** 粗 MANO mesh 投影到多尺度图像 feature maps，采样后预测参数残差。官方 `RefineNet.forward` 对同一个粗 mesh 在多个尺度读取并合并特征，随后更新一次 pose/shape/camera；外层只调用一次 refiner。论文明确没有 temporal module。因而它已经覆盖“state-conditioned visual evidence”，但多尺度不是多轮更新查询位置。[论文 §4.2、§5.3](https://arxiv.org/html/2409.12259v1)、[refiner](https://github.com/rolpotamias/WiLoR/blob/main/wilor/models/heads/refinement_net.py)、[调用路径](https://github.com/rolpotamias/WiLoR/blob/main/wilor/models/wilor.py)

**DIR。** 每轮把上轮 3D joints 投影并 bilinear sample 图像特征，结合手内 GCN 与双手 Transformer；再把关节特征分 plane 投回图像并增强视觉特征。各阶段有几何监督。它已经实现 state-to-image-to-state 迭代，发表会议是 **ICCV 2023**。对本项目而言，原始 image evidence 与融入候选信息的特征应保留语义区别。[全文 §3.2、§4.4](https://arxiv.org/pdf/2302.02410)、[正式发表页](https://openaccess.thecvf.com/content/ICCV2023/html/Ren_Decoupled_Iterative_Refinement_Framework_for_Interacting_Hands_Reconstruction_from_a_ICCV_2023_paper.html)

**GeoHand。** 冻结 MoGe2 提供单目 geometry features，GeoAdapter 与 gated fusion 适配到手部；KQIR 每轮由当前 3D joints 和 2D 投影构造 query，对图像 tokens 做 cross-attention，更新 MANO/camera。公式是全图 attention，不能写成已实现局部投影窗口采样。论文默认两轮 KQIR；本次只确认预印本身份。[v1 §3.1–3.5、§4.2](https://arxiv.org/html/2605.17354v1#S3.S4)

另外三篇用于补齐技术谱系：[MobRecon §3](https://arxiv.org/pdf/2112.02753)通过预测 2D 点读取特征、提升为 coarse vertices 并解码 mesh；[Hamba §3.2](https://arxiv.org/html/2407.09646v2)在关节空间组织 Mamba；[HandOS §3](https://arxiv.org/html/2412.01537v2)把检测与 2D/3D query 解码整合。它们分别说明效率、结构扫描和系统集成的进展，不能因为“有 Transformer/SSM”就统称 trajectory world model。

## 视频与世界坐标恢复

**HaWoR，CVPR 2025。** camera-space 重建采用 Image Attention Module 和 Pose Attention Module；世界相机由屏蔽手区域的 DROID-SLAM 估计，Metric3D 与 AdaSM 提供尺度；另以 motion infiller 恢复缺失段。infiller 在 HOT3D 上训练，使用真实/合成连续 mask 与插值初始化。因此 HaWoR 已经包含视觉时序处理和运动补全。[论文 §3、补充 §8](https://arxiv.org/html/2501.02973v1)

在官方 `66c7d4108d58a716deccd192cb7645170cdc7bd7` 中，[视觉模型](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/lib/models/hawor.py)使用 16 帧 temporal attention，未设置 causal mask；[视频补全路径](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/scripts/scripts_test_video/hawor_video.py#L222)使用双手 218D、120 帧上下文，**只回填 missing slots，保留原来 valid 的输出**。补全后 valid 包含合成状态。这里的主要可检验区别是 HandLoop 能否修复仍 valid 但错误的状态，而不仅是填空；16/120 帧上下文也说明现成系统不能称严格在线。

**HMP，WACV 2024。** 用 MoCap 训练 VAE/NeMF，sequence latent 与时间解码局部手指姿态；视频恢复先优化 wrist 位姿与 shape，再加入 motion latent 调整局部姿态。目标结合重投影、confidence、初值与平滑约束；2D 条件来自 MediaPipe，缺检测时回退到 PyMAF-X 预测投影，不能把全部条件称为独立图像测量。它假定静止相机，学习先验排除全局 wrist orientation。纯 MoCap prior 用于视觉恢复已经是它的核心范式。[论文 §3 与补充](https://arxiv.org/pdf/2312.16737)

官方 HMP `35d799f76b2b2bc1d1e945117b021014b099e7e6` 的 [LocalEncoder](https://github.com/enesduran/HMP/blob/35d799f76b2b2bc1d1e945117b021014b099e7e6/src/nemf/prior.py)使用 skeleton-aware convolution/pooling；[实际 fitting](https://github.com/enesduran/HMP/blob/35d799f76b2b2bc1d1e945117b021014b099e7e6/src/fitting_app.py#L985)冻结 global latent、独立处理 root orientation/translation。不能因为通用类里存在 global encoder，就声称论文先验已经学习完整 world wrist 动力学。

**Dyn-HaMR，CVPR 2025。** 从视觉手估计、2D detections 和 SLAM 初始化，分阶段联合优化手运动、相机与尺度，再加入 HMP latent、生物力学及双手穿透目标。它复用每手 HMP，交互约束来自附加目标；并非仅凭“interacting prior”就能认定其另训练了双手联合生成模型。该工作证明世界位姿、先验与图像几何的联合优化也是已有路线。[论文 §3、补充 A](https://arxiv.org/html/2412.12861v1)、[CVPR 正式页](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Dyn-HaMR_Recovering_4D_Interacting_Hand_Motion_from_a_Dynamic_Camera_CVPR_2025_paper.html)

## 质量与生成式运动恢复

**DIP，ICCV 2025，是必须补入的直接前作。** 正式题名为 *Diffusion-based 3D Hand Motion Recovery with Intuitive Physics*。[论文 §3.2–3.4](https://arxiv.org/html/2508.01835v1)直接学习 `p(clean motion | initial motion)`，每轮同时读取固定初始估计与当前 diffusion candidate，经 MANO/MeshCNN/Transformer 恢复。训练随机扰动 GT 运动，不需要 RGB；主要数据为 DexYCB，不能因此自动称其标签来自 glove。它还自回归使用历史预测和预测的 motion-state；不能说已有恢复器完全没有反馈。物理部分是 reaching/grasping/manipulation/releasing 状态监督及相应运动约束，未描述候选重读 RGB 或观测质量重估。

其 motion-state 训练标签依据 contact 和 hand-object distance，因此完整物理分支的标注要求与只有关节轨迹的预训练并不相同。官方 `dbbff920aeab960db856a457b70a247ce1f9e041` 的 [demo](https://github.com/zhangy76/DIP-Hand/blob/dbbff920aeab960db856a457b70a247ce1f9e041/demo.py#L259)恢复含 root 的 rotation 和 shape，未见 world translation 修正；encoder 可读取整段，故 autoregressive decoder 不等于严格在线系统。**纯运动受损恢复、初始观测/候选分离、序列内预测反馈都已有先例。** HandLoop 应与 DIP 做相同输入、训练数据及可用时间上下文下的对照，不能只与 prediction-only 比较。

**StableHand。** 阅读 2026-05-18 的 [v1 §3、附录 F/G](https://arxiv.org/html/2605.18553v1)。每手状态为 54D：world translation、wrist rotation6D、MANO finger axis-angle；150 帧片段共同恢复双手。quality 分左右手 wrist/fingers 四路，从预测几何误差映射而来，既不是逐关节 visibility，也不是“正确概率”。QN 不接 image features，但生成 DiT 显式接 WiLoR 图像特征、scene 和 camera。quality 控制 flow path 与观测/噪声混合初始化，ODE 中的 quality 与视觉条件固定。高 q 对应较小目标更新，不宜夸成所有高 q 都被代码硬锁。

因此，StableHand 是质量引导恢复的直接竞争方法，不能称为纯轨迹生成器。HandLoop 的可检验差别应落到独立 motion-only pretraining、错误 quality 的再判断、变化候选的图像取证和最终状态反馈。

**StableHand 版本边界。** 官方 `0471a45e325ffa167cc6a8484d712a21befd3e47` 的 [Enhanced 说明](https://github.com/huajian-zeng/stablehand/blob/0471a45e325ffa167cc6a8484d712a21befd3e47/docs/enhanced_version.md)已改 geometry backbone 为 VGGT-Omega，加入 depth cues，预测相对平滑 wrist 轨迹的残差。[推理入口](https://github.com/huajian-zeng/stablehand/blob/0471a45e325ffa167cc6a8484d712a21befd3e47/sample/infer_clips.py)仍在 flow loop 前计算 quality。论文和增强版配置、指标不能混用；[README](https://github.com/huajian-zeng/stablehand/blob/0471a45e325ffa167cc6a8484d712a21befd3e47/README.md)记录 NeurIPS 2026 接收，同时保留训练/预处理发布待办。公开缓存评测路径不等于已复现 raw-video 全流程。

**PAD-Hand，CVPR 2026。** 阅读 [v2 §3、附录 C/E](https://arxiv.org/html/2603.26068v2)，更新日期 2026-03-30。它以真实视觉估计序列为条件，让 forward diffusion 把 GT 移向该估计附近，reverse 恢复姿态；原始与当前候选经 MANO 成 mesh，再用 MeshCNN/Transformer 建模。它不在每步重读 RGB，也不是独立 glove-only 预训练。训练加入 Euler–Lagrange residual likelihood，LLLA 将姿态后验不确定性传播到物理残差；这种 variance 不能直接解释成几何误差可靠性。

官方 `ca9ed97bc671199c25cf569d3b1de0e6f7937251` 的 [demo](https://github.com/DominoAI-Lab/PAD-Hand-CVPR-2026/blob/ca9ed97bc671199c25cf569d3b1de0e6f7937251/demo.py#L79)在 16 帧、48D Euler pose 上做 4 步恢复，保留初始 camera translation。encoder 可看全片段，decoder 的 causal mask 不使整条链路因果。当前发布树未提供完整训练与 LLLA/force variance 执行路径；这是公开复现范围的限定，不能说作者没有实现。

**JoHan，2026-10-07 预印本。** [v1 §2、附录 A](https://arxiv.org/html/2610.10512v1)用 video-conditioned flow 联合生成 local 2D 轨迹/visibility 与 MANO 3D；每个 flow step 更新 2D 引导，两分支均读取 DINOv3 图像特征，PnP 恢复 camera wrist 位姿；HOT3D 评测中再使用 HaWoR 相机轨迹转 world。它采用 128 帧片段与配对视频训练，不是独立纯轨迹或因果当前状态恢复。其评测 clip 约150帧，区别于 HaWoR 原始100帧协议；各几何指标并非全面胜过 HaWoR。

**两篇扩展近邻。** [AffHandGen v3 §3.3、Algorithm 1](https://arxiv.org/pdf/2510.00506v3)以 image/text affordance 条件 diffusion 恢复单帧手；2D guidance 的目标是初始 3D 预测自身投影，visibility 从初始 mesh 与 SAM2 获得。它覆盖遮挡生成与迭代引导，但该投影不是独立 keypoint 测量。[ACE-Ego-Hand v3 §3](https://arxiv.org/html/2608.20308v3)用预训练 video diffusion 作为特征 encoder，经双向时空解码恢复 MANO 和 visibility；属于 offline 视频恢复。其早期名 DreamHand，不应列成两篇独立工作。两者仅补充方法定位，不作为已完成的实验基线。

## 把不同的迭代与可靠性分开

| 机制 | 已核验的代表 | HandLoop 需要额外回答的问题 |
|---|---|---|
| 固定图像 context 上更新参数 | HaMeR IEF | 更新是否获得候选相关的额外图像信息 |
| 候选投影后读取局部特征 | WiLoR；DIR 多轮 | 多轮相较一次是否降低真实几何误差 |
| 当前 joint query 随轮次变化 | GeoHand | pure-motion 恢复是否提供不可替代的条件信息 |
| 优化 motion latent 匹配视觉 | HMP、Dyn-HaMR | learned current restoration 相较优化/普通temporal模型的收益 |
| 纯运动受损恢复与序列内自回归反馈 | DIP | 新的视觉取证与最终可靠性反馈相较已有恢复链是否有效 |
| quality-conditioned flow | StableHand | 错误高质量提示能否被图像反证并修正 |
| 固定初始姿态条件下反复去噪 | PAD-Hand | 状态迭代与重读图像的贡献是否不同 |
| 生成过程中更新2D/visibility引导3D | JoHan | 生成的引导与原始测量、可靠性目标有何区别 |
| 最终状态作为下一帧历史 | 本项目设计 | 是否减少恢复时间，或反而传播自身错误 |

表中只归纳被明确核验的机制；“未描述某机制”不是证明所有实现都没有。也不能将 visibility、detection validity、预测几何质量、flow 时间或 physics variance 都叫同一种 confidence。

## 世界误差与复现协议

HaWoR 官方 [HOT3D eval](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/scripts/scripts_eval/eval_hawor_hot3d.py#L337)按 100 帧分段；[first_align_joints](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/lib/eval_utils/eval_utils.py#L322)使用前两帧估计含 scale 的 similarity transform 后计算 W-MPJPE，WA 使用整段 similarity alignment。**论文中带 W 的名字不自动意味着未经对齐的绝对米制误差。**

StableHand 原论文附录 G 也描述首帧/全段 similarity alignment，但增强版 [metrics.py](https://github.com/huajian-zeng/stablehand/blob/0471a45e325ffa167cc6a8484d712a21befd3e47/eval/metrics.py#L18)将 W 定义为无对齐、WA 定义为逐帧减 wrist。坐标进入该函数之前另有处理；[数据 loader](https://github.com/huajian-zeng/stablehand/blob/0471a45e325ffa167cc6a8484d712a21befd3e47/data_loaders/clip_dataset.py)使用 benchmark 的首帧 GT camera 作坐标 anchor，shape 来自缓存。shape 来源及 raw-video 替代路径仍需核验，不能据缓存字段直接断言不当泄漏。

建议后续同时保存两套清晰指标：按论文协议计算的兼容 W/WA，以及注明 world gauge、是否允许一次刚体对齐、固定尺度规则的 wrist/joint 几何误差。若具备共同的度量世界参考，再报告无对齐误差。每份结果都要附 skeleton、单位、clip 长度、可用未来帧、camera/shape 来源；平滑、PA 和对齐后轨迹指标各回答不同问题。

## 对现行研究的判断

以下是文献约束下的研究判断，不是新用户决定或已验证贡献。

1. **问题仍然成立，但表述应更精确。** 研究“独立轨迹知识能否帮助判断并修复当前视觉测量错误”，比仅强调引入 world model 更容易检验。前向运动预测与视觉状态后验恢复应区分。
2. **原始观测固定，不表示准确观测被硬锁。** 保存 `Z_t` 是为了持续比较证据；候选仍可变化。这样才能研究 false-high-confidence，并衡量原本正确的观测是否被破坏。
3. **结构与重取证各自已有充分前作。** DIR/WiLoR/GeoHand 必须进入相关工作；HMP/DIP/StableHand/PAD-Hand 必须进入先验和恢复比较。尤其 DIP 已有当前受损运动条件恢复，不能把所有前作归类为未来预测。组合机制是否有效，要在相同前端和训练数据条件下拆开验证，不能靠模块命名证明。
4. **先证明几何恢复，再扩闭环。** 对照 no prior、static prior、prediction-only、current-state restoration；再比较固定视觉条件/变化候选查询、固定/重估可靠性、无/有最终反馈。核心诊断是有效但错误观测、真实突变、遮挡重现与正确观测退化。
5. **本轮没有理由自动更换主干或表示。** HaWoR 仍是用户选定前端；69D 是本项目建议而非行业标准。MANO/关节点及监督来源的另项核验见[表示与标注专题](../memory/10_MANO_REPRESENTATION_AND_ANNOTATION.md)。是否采用其建议，应另行通过接口和实验决定。

## 阅读覆盖与尚未覆盖的边界

核心精读包括 WestWorld v2、HaMeR、WiLoR、DIR、HaWoR、HMP、DIP v1、StableHand v1、PAD-Hand v2；扩展方法核验包括 MobRecon、Hamba、HandOS、Dyn-HaMR、GeoHand v1、JoHan v1、AffHandGen v3、ACE-Ego-Hand v3。前述章节均给出原论文、方法位置和必要的代码来源，第三方源码未在本机执行。

[EgoFound3R](https://arxiv.org/abs/2610.01210)等新近 hand/scene 联合工作本轮仅发现条目，未计入精读结论。本报告不是穷尽性 systematic review，不能依据未检出论文宣称任何组合“首次”。没有统一重跑各方法，因此不提供跨表拼接的 SOTA 排名。
