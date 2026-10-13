# 主流手部表示与MANO标注来源

核验日期：2026-10-13。用户要求查看代表性方法采用69D还是MANO，以及MANO数据标签如何产生。以下核对原论文和官方代码，没有运行模型、读取完整训练数据或验证论文指标。

## 结论与范围

与本项目接近的RGB手部网格重建/世界手运动恢复中，MANO参数输出很常见；直接关节/顶点坐标及混合表示也存在。**我们提出的3+6+20×3=69D不是这些论文共同采用的标准。** 本次是代表性方法核验，不是全领域使用比例统计。

必须分清：网络内部状态、最终输出、监督标签和评测表示。一篇方法输出MANO，不代表encoder只读取MANO参数；报告21点MPJPE也不能反推网络一定输出21点。使用MANO拓扑/训练mesh更不等于回归MANO参数。

## 代表性方法

| 方法 | 内部建模与输出 | 证据 |
|---|---|---|
| HaMeR，CVPR2024 | 回归MANO pose/shape/camera；默认16×6 rotation编码，shape10和camera3另头；MANO层输出mesh和joints | [论文§3](https://arxiv.org/html/2312.05251v1)、[MANO head](https://github.com/geopavlakos/hamer/blob/main/hamer/models/heads/mano_head.py)、[forward/loss](https://github.com/geopavlakos/hamer/blob/main/hamer/models/hamer.py) |
| WiLoR，CVPR2025 | MANO输出；论文pose48+shape10+camera，临时mesh支持图像对齐refinement | [论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.pdf)、[代码](https://github.com/rolpotamias/WiLoR/blob/main/wilor/models/wilor.py) |
| HaWoR，CVPR2025 | MANO姿态与世界位姿；pose attention处理16×6旋转，shape和camera另头 | [论文§3](https://arxiv.org/html/2501.02973v1)、[forward](https://github.com/ThunderVVV/HaWoR/blob/main/lib/models/hawor.py) |
| HMP，WACV2024 | encoder同时含position、velocity、rotation6D、angular velocity；decoder输出旋转并FK，最终用MANO轨迹恢复视频 | [论文§3](https://openaccess.thecvf.com/content/WACV2024/papers/Duran_HMP_Hand_Motion_Priors_for_Pose_and_Shape_Estimation_From_WACV_2024_paper.pdf)、[generative.py](https://github.com/enesduran/HMP/blob/main/src/nemf/generative.py) |
| StableHand，2026 | flow head预测108D状态空间的velocity，经ODE积分得到双手运动；每手54D为world root translation3+wrist rotation6D6+finger axis-angle45；shape另存，QN另预测四路观测误差再转换quality | [论文§3](https://arxiv.org/html/2605.18553v1)、[代码](https://github.com/huajian-zeng/stablehand) |
| PAD-Hand，CVPR2026 | 公开demo在16关节Euler48D姿态上扰动/恢复；MeshConv输入原始与当前MANO vertices，pose head预测16×6旋转并MANO解码 | [论文](https://arxiv.org/html/2603.26068v1)、[demo](https://github.com/DominoAI-Lab/PAD-Hand-CVPR-2026/blob/main/demo.py)、[model](https://github.com/DominoAI-Lab/PAD-Hand-CVPR-2026/blob/main/models/pad_hand.py) |
| HandX，CVPR2026 | diffusion双手42joints，每点xyz+rotation scalar；AR288D包含相对腕位置、速度、rot6D、局部关节与旋转补充；数据另存MANO | [论文§5](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_HandX_Scaling_Bimanual_Motion_and_Interaction_Generation_CVPR_2026_paper.pdf)、[README](https://github.com/handx-project/HandX) |
| METRO，CVPR2021 | 直接回归21×3 joints及195×3低分辨率vertices，上采样778vertices；MANO用于模板/拓扑，非最终pose回归 | [官方实现](https://github.com/microsoft/MeshTransformer/blob/main/metro/modeling/bert/modeling_metro.py) |

PAD-Hand公开demo保留初始WiLoR cam_t作render，未在该refinement函数更新world wrist translation；不能因其恢复MANO pose就认为与本项目完整world wrist恢复同任务。表中实现细节不代表全文所有训练模式已经复现。

HandX的rotation scalar补充也说明，关節位置不能唯一描述所有骨骼扭转和表面。69D位置恢复有数据适配优势，但不能直接替代完整mesh参数。

### MANO编码维数为什么不统一

- 手指局部姿态有15个模型关节旋转；axis-angle为45值。
- root axis-angle3+finger45=48值；加translation3是51；再加shape10是61。
- root换rotation6D时，3translation+6root+45finger=54；shape可单独作为静态条件。
- 全部16个旋转都用6D时为96；加translation为99，加shape为109。

这些是不同编码和字段范围，均不能直接解释成相应数量的独立人体物理自由度。我们的69D是腕位姿+局部关节坐标候选，而上述多种表示保留关节旋转。

### StableHand专项输出对照（2026-10-13补充）

用户明确要求将 *StableHand: Quality-Aware Flow Matching for World-Space Dual-Hand Motion Estimation from Egocentric Video* 加入对照。下面区分论文v1定义与增强版公开实现，不改变本项目尚未定案的状态表示。

| 输出层级 | StableHand的设计 | 与其他输出的区别 |
|---|---|---|
| 最终动态状态 | 每手 `(p_world:3, R_root_6D:6, theta_finger_aa:45)`，共54D；双手序列为 `T×108` | 与HaMeR的 `96 pose+10 shape+3 camera` 统计范围不同，不能按维数比较表达能力 |
| 单次flow网络输出 | `v_theta(x_tau,tau;conditions)`，与运动状态同维，经ODE积分生成最终状态 | 是生成过程的更新方向，不是一次前向直接输出最终MANO；也不是物理时间上的手部速度 |
| 质量网络输出 | 每帧左右手wrist/fingers共4个预测几何误差，转换为 `q=exp(-e_hat/sigma_hat)` | wrist组同时涉及平移和旋转；每手所有finger共用一组，不是21个关节各自的confidence |
| 形状与解码 | beta不在108D flow state内；增强版代码把左右手各10D beta另作条件并保存，MANO解码出mesh/joints | 最终mesh可用不等于flow同时估计了shape；缓存shape来源须在复现时登记 |

论文[§3、Eq.8–11](https://arxiv.org/html/2605.18553v1#S3)中的 `tau` 是flow积分时间，`t` 才是视频帧。quality通过噪声路径、velocity target、AdaLN和初始化影响恢复。其监督目标是**初始观测误差**；不是最终恢复结果误差的重新估计，也不能直接作为本项目最终状态质量 `q` 使用。本项目命名中它更接近观测质量 `w`，粒度和监督仍不相同。论文生成器显式接入图像/场景/相机条件，因此不能把该flow误称为纯轨迹预训练TWM。

固定实现 `0471a45e325ffa167cc6a8484d712a21befd3e47` 的[推理入口](https://github.com/huajian-zeng/stablehand/blob/0471a45e325ffa167cc6a8484d712a21befd3e47/sample/infer_clips.py)在ODE前预测quality并保持其条件，单手实际字段次序为 `root_rot6d, finger_aa45, trans3`。该次序与论文书写 `(p,r,theta)` 不同。该版还使用相对平滑腕轨迹的残差，不能把所有中间tensor都当成已还原的world state。

平移原点也需转换：[tcanon.py](https://github.com/huajian-zeng/stablehand/blob/0471a45e325ffa167cc6a8484d712a21befd3e47/data_loaders/tcanon.py)明确该MANO约定下 `wrist = J0_rest(beta) + trans`，推理FK把trans传给MANO `transl`。因此这里的root translation不能直接复制为69D中的实际wrist xyz；需要按同一MANO实现和shape解码/换算。这也是“同为3D translation仍未形成统一输出接口”的具体例子。

## MANO模型本身与数据集MANO标签是两件事

[MANO原项目](https://mano.is.tue.mpg.de/)说明模型从约1,000个高精度手扫描、31名被试学习shape/pose deformation。它提供一个可微的参数到手mesh/骨架的映射。

真实数据集再用这个模型去解释采集到的图像、深度、3D点或marker，拟合得到逐帧pose、整体位姿和subject shape。标注者一般标的是图像点、mask或少量质检对应，不是逐帧填写全部MANO角度和shape。

一个示意拟合目标为：

\[
\min_{\theta_{1:T},\beta,R_{1:T},t_{1:T}}
\lambda_{2D}\mathcal L_{reprojection}
+\lambda_{3D}\mathcal L_{points/markers}
+\lambda_D\mathcal L_{depth/silhouette}
+\lambda_P\mathcal L_{pose\ prior}
+\lambda_T\mathcal L_{temporal}.
\]

这里theta只指finger pose，R/t为整体位姿。各数据集实际使用的项不同；marker库不必有2D/depth loss。许多流程先标定subject shape并固定，再拟合逐帧运动。最终MANO标签的误差同时受原始测量、模型表达、遮挡、拟合优化和质量筛选影响。

## 代表性标注路径

| 数据集 | 采集/人工或自动标注 → MANO | 质量解释与来源 |
|---|---|---|
| DexYCB | 8台同步RGB-D；各视角21点人工标注/visibility → 联合拟合2D投影与多视角depth；subject shape先标定固定，逐帧优化 | [论文§2及附录B](https://arxiv.org/html/2104.04631)；它是RGB-D拟合reference，不是设备直接测MANO角度 |
| HO3D / HOnnotate | RGB-D → 自动2D关节/分割 → 拟合MANO和物体，使用depth/点云、silhouette、关键点与时间/物理约束 | [HOnnotate](https://arxiv.org/html/1907.01481)、[v3报告](https://arxiv.org/html/2107.00887)；v3的53帧人工五指尖验收不是每帧人工MANO标签 |
| FreiHAND | 8视角采集 → 少量人工关键点/mask启动 → MVNet/HandSegNet预测 → MANO拟合 → 人工验收与迭代重训 | [论文§4](https://arxiv.org/html/1909.04349v3)；human-in-the-loop的markerless拟合 |
| InterHand2.6M | 人工多视角2D→3D；自动detector+RANSAC扩规模；后续NeuralAnnot拟合MANO | [原论文](https://arxiv.org/html/2008.09309)、[官方更新](https://github.com/facebookresearch/InterHand2.6M)、[字段](https://mks0601.github.io/InterHand2.6M/)；约5mm是官方MANO fit误差，2.78mm是自动keypoint对人工的另一个环节 |
| Re:InterHand | 多视角/3D scan → V2V-PoseNet关节 → NeuralAnnot拟合scan与关节 → 重光照图像 | [论文§3](https://arxiv.org/html/2310.17768)、[发布](https://mks0601.github.io/ReInterHand/)；原关节、scan拟合和MANO拟合是不同产物 |
| GRAB | 皮肤marker、个人scan → personalized SMPL-X → MoSh++拟合 → 导出MANO | [论文§3](https://arxiv.org/html/2008.11200)；marker点不是关节中心，tracker精度不等于最终关节精度 |
| ARCTIC | Vicon手/身体marker、个人scan → SMPL-X/MoSh++ → MANO/身体标注 | [论文§3](https://arxiv.org/html/2204.13662)；全身模型辅助腕部global transform，raw marker与拟合值应分开 |
| OakInk2 | OptiTrack marker → 人工清理及遮挡marker多视角补齐 → 两阶段SMPL-X拟合 → MANO | [论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhan_OAKINK2_A_Dataset_of_Bimanual_Hands-Object_Manipulation_in_Complex_Task_CVPR_2024_paper.pdf)、[补充](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Zhan_OAKINK2_A_Dataset_CVPR_2024_supplemental.pdf)；T-pose定shape/marker位置，再拟合pose |
| HOT3D | 每手19枚marker+个人hand scan → OptiTrack → UmeTrack个人模型拟合 → 转换MANO | [论文附录C](https://arxiv.org/html/2411.19167)、[官方toolkit说明](https://github.com/facebookresearch/hand_tracking_toolkit#important-notes)；官方明确转换MANO精度略低，推荐用UmeTrack关键点监督 |

合成数据是另一路径：先选择MANO参数并渲染，标签来自生成过程；这与真实数据拟合的provenance不同。把已有单目预测器生成的MANO直接当标签则是伪标注，不能称独立高精度GT。

## 有joint标签也能训练MANO输出

可微MANO层能将网络预测的pose/shape解码为joints/mesh，因此有21点、2D或depth reference的样本可以用几何损失训练MANO输出，不要求每条样本都有完整theta/beta标签。

[HaMeR损失§3.4](https://arxiv.org/html/2312.05251v1)明确组合参数、3D joints、2D reprojection和先验。手套只有关节位置时，可以按有效点/骨架映射监督解码关节，或离线拟合MANO以增加参数监督；仅位置不能唯一确定某些rotation/shape，仍需先验及质量区分。

建议数据adapter同时保留：原始参数（若有）、原始测量/关键点、解码关节、wrist/root定义、shape、时间及fit来源。不能把MANO解码点与原始marker/joint标签混成同一类测量。

## 后续已定99D与实现边界

早先调查建议MANO输出＋joint监督；2026-10-13用户明确“我们使用99D”，现按D16采纳，取代69D默认候选。主线不变，转换/监督/效果仍待验收。

1. **已选99D动态MANO状态**：实际wrist p3＋root6＋15×finger6；shape10人物/序列条件另存。root translation与腕点正确换算，源45D先完整解码hand mean后统一旋转。
2. 保留69D joint-state作为有意义的对照，验证跨glove骨架接入和局部当前状态恢复的收益；不能宣传它是行业通用标准。
3. 99D主要以真实joint/wrist误差验收；mesh有reference时补充。69D不再首轮必做选型，后续比较仍需同数据/split/corruption及IK成本。

4. 不因表示变化而改变pure-trajectory TWM、current-state restoration、固定Z/变化candidate、独立图像取证、w/q或最终反馈的研究主线。

[v1预训练方案](../docs/TWM_PRETRAIN_PLAN.md)已广泛对照最新论文重新规划数据，β真实评测不暗用GT shape；Corrector51D切向残差是新增实现建议，非99D选择自动证明。暂无表示实验。

## HaWoR论文与代码的一个边界

[论文§9](https://arxiv.org/html/2501.02973v1)称各手独立建模、没有双手交互/不穿透约束；[当前filling_preprocess源码](https://github.com/ThunderVVV/HaWoR/blob/main/lib/eval_utils/filling_utils.py)却按手分别canonicalize后，把两手每帧109值拼为218输入向量。当前network用这个向量进入共享embedding。保留这两种证据，不把拼接输入等同于已验证的双手交互建模，也不拿main实现反推发表版本训练事实。复现实验应pin代码版本。

## 访问与现有记忆勘误

GRAB/ARCTIC等注册资源仅在本报告解释标注流程，不自动回到首批数据计划。免申请数据原则仍有效，数据下载与MANO模型asset许可独立。

HO3D的[Terms of use](https://github.com/shreyashampali/ho3d#terms-of-use)已核验为学术非商业用途，含禁止再分发、修改及商业使用的原文。本报告只登记发布条款，不解释其衍生使用边界；之前“未找到明确许可”的记录应更新。eval GT虽已发布，仍应遵守训练/test分割。
