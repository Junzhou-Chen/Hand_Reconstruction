# TWM 预训练公开数据集调研

核验日期：2026-10-13。目标是为 HandLoop 的 **wrist 6DoF + articulated fingers 当前状态恢复 TWM** 找到可获取的连续运动数据。结论基于作者论文、官方项目/仓库、下载说明和公开元数据；尚未完成数据本体质量验收或训练。

**后续补充：本报告首轮检索偏学术，并非完整全景。** 用户指出产业数采覆盖不足后，已新增[产业手部数据调查](09_INDUSTRY_HAND_DATASETS_2026-10-13.md)，包含SynData、Lightwheel、EgoVerse、OOJU、OpenGraph、Daimon和UniHand。选择数据时两份一起读，尤其不要由这里的短名单推断“大规模公开人手数据只有这些”。

**最新选择原则覆盖下面历史推荐顺序：** 用户随后明确优先无需申请、直接可下载且论文常用的数据；GRAB/ARCTIC与HOT3D full VRS的注册/登记路线不作为首批默认。**后续核验HOT3D-Clips官方train可HF直下，当前建议它与Re:InterHand 3D train建立首轮基线，不能排除所有HOT3D。** 详见[最新数据记忆](03_DATA_AND_EVALUATION.md)、[八库预训练方案](../docs/TWM_PRETRAIN_PLAN.md)。保留本报告的来源事实，不将旧推荐当新决定。

**建议先用 GRAB 与 ARCTIC 建立可复现基线，再加入 OakInk2；HOT3D 的公开训练标签适合补充目标域，DexWild 和 HumanTouch 值得用于扩展真实手套数据。** 这是本次研究建议，不是用户已选定的训练组合。

TWM 不看图像，所以预训练轨迹不必来自头戴 RGB。外部 MoCap、第三视角场景、没有视频的运动参数都可以适用。关键是手指确实在变化、标签来源可靠、时间连续、腕部参考系明确；视觉模型训练的数据需求与这里不同。

## 如何理解公开可用

- **直接公开文件**：看到了作者发布的数据目录、索引或文件；不等于已完整下载并验证。
- **注册或申请后获取**：有官方科研下载流程；本次没有代为注册、接受许可或提交申请。
- **只公开子集**：按实际 release 计量，不用论文宣称的全集规模。
- **待核验**：有发布声明，但文件、字段或许可未全部明确；不能当成已经可直接投入训练。

多数资源是有条件开放的科研数据，不是任意用途的“开源”。数据许可、代码许可和 MANO/SMPL-X 模型许可应分别记录。下文仅报告发布方标注和获取方式。

## 第一批主训练候选

| 数据集 | 已核验规模与采样 | 手指和腕部的来源 | 本项目用法 |
|---|---|---|---|
| GRAB | 1,334 序列、1,622,459 时间帧、120 Hz；约 3.76 小时为帧数换算 | Vicon marker MoCap → 模型拟合；有左右手独立 MANO 全局位姿与 articulation | 最适合先跑通统一表示和 corruption/restoration |
| ARCTIC | 339 序列、10 人、30 Hz；2.1M 是多视角图像数 | Vicon；双手/关节物体操作；世界系 MANO/SMPL-X | 易获取纯参数包，补充灵巧双手操作 |
| OakInk2 | 627 序列；同步 30 Hz；4.01M 为四相机图像数 | OptiTrack 双手/上身 marker → SMPL-X/MANO；部分遮挡 marker 人工多视角补齐 | 扩展复杂长任务与操作多样性 |
| HOT3D | 全集 833 分钟、19 人、30 Hz；不等于全部时长均有公开 GT | 专业 optical-marker MoCap，UmeTrack/MANO | 用公开 TRAIN reference；与 HaWoR/头戴应用接近 |
| DexWild human | 公开 release 人类子集 7,917 episodes、2,924,588 transitions；若按30Hz解释，约27.08小时 | MANUS 手套；ArUco tracker + ZED视觉/SLAM测腕部 | 扩展跨环境手套操作；不用重定向后的机器人 action 代替人手 |
| DexCap public subset | 官方公开 30 分钟 wiping + 60 分钟 packaging | Rokoko 手指 + T265 wrist SLAM | 数据量小但与原始 idea 很贴近，适合 adapter 验证和补充 |
| HumanTouch，条件性候选 | 官方称首批约100小时；实际已核验X001/X002元数据合计约30.40小时，非全库统计 | MANUS手套 + VIVE追踪；每手25 joints、wrist 7D，统一60Hz | 文件已发布，pose可单独取；完整上传量、许可、数值质量尚未验收 |

各项事实、资源入口和限制如下。有效可训练时长还需扣除 test split、缺标签、跳帧、坏拟合及重复来源；不能把上表简单求和当训练总量。

### GRAB

[官网及下载](https://grab.is.tue.mpg.de/)；[数据结构](https://github.com/otaheri/GRAB#contents-of-each-sequence)；[论文](https://arxiv.org/pdf/2008.11200)。10 人与 51 个物体，120 Hz；Table 1 的 1,622,459 帧除以采样率约为 3.756 小时。独立 `lhand/rhand` 提供 MANO translation、rotation/articulation 和模板，可直接解码手部运动，不需 RGB。

需注册接受[GRAB 科研许可](https://grab.is.tue.mpg.de/license.html)，模型另行获取。推荐先用独立 MANO release；AMASS 内的 GRAB 是重叠来源，不能再计一份。**MANO `transl` 不必等于实际 wrist joint 坐标**，应经模型解码后转换到本项目定义。未核验登录后的当前包体积。

### ARCTIC

[官方下载说明](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md)明确运动生成可只下载 `raw_seqs`（215 MB）和 `meta`（91 MB），不必下载数百 GB 图像。注册入口与[许可](https://arctic.is.tue.mpg.de/license.html)是科研用途流程。

[字段文档](https://github.com/zc-alexfan/arctic/blob/master/docs/data/data_doc.md)给出世界系 `rot[T,3]、pose[T,45]、trans[T,3]、shape[T,10]`，还有拟合误差和原始 marker 资源。[论文](https://download.is.tue.mpg.de/arctic/arctic_april_24.pdf)的 339 序列平均约 698 时间帧，估计约 23.7 万独立时刻、2.19 小时；这是平均数推算，非实际文件求和。9 视角不能给 TWM 增加 9 倍独立轨迹。

### OakInk2

[项目](https://oakink.net/v2)；[论文](https://arxiv.org/abs/2403.19417)；[官方发布数据](https://huggingface.co/datasets/kelvin34501/OakInk-v2)。12 台 OptiTrack 跟踪身体、手和物体 marker，拟合 SMPL-X 并导出 MANO；不是仅由头戴图像估计手。627 段包含 primitive 和 complex task sequences，适合学习跨动作的手部关系。

HF 文件索引中 `anno_preview` 有 627 个 pickle，总约 36.65 GB；可先取这些 annotations，不必下载约 2.13 TB 全库。数据卡标注 CC BY-SA 4.0，人体模型另外授权。**拟合值仍有误差，遮挡 marker 含人工视觉修复**；需要检查拟合、连续性和 root 坐标，不能称为所有关节直接传感器读数。

[官方schema](https://github.com/oakink/OakInk2)有 `frame_id_list`、`mocap_frame_id_list`、`raw_smplx` 和 `raw_mano`。后者的左右手 `pose_coeffs:[1,16,4]` 使用 wxyz quaternion，`tsl:[1,3]`、`betas:[1,10]`；SMPL-X还有world rotation/translation。固定MoCap参考系可支持全局运动，但长度单位、root定义与实际wrist点仍需通过样本和FK核验。

### HOT3D

[官网](https://facebookresearch.github.io/hot3d/)；[工具与下载说明](https://github.com/facebookresearch/hot3d)。有 optical-marker 手部参考和 UmeTrack/MANO，公开训练标注可用于纯轨迹学习。这些是marker驱动拟合出的手模型状态，不是所有关节的独立直接测量。官方 TEST 被试不提供 GT，833 分钟总规模不能全计入预训练。

**TWM 不需要 RGB，因此有标签的 Aria 与 Quest 3 轨迹都可考虑；仅后续 RGB paired correction 才需要挑 Aria。** 手部 annotation/模型、camera pose 和 timestamps 可按发布格式选取，避免先下全部视频；仍需实际验证对应下载分组。数据按 HOT3D agreement，MANO 另有要求。若用 HOT3D 做 OOD 目标集，则不应将它的 motion labels 偷放进预训练。

轨迹文件包括 `mano_hand_pose_trajectory.jsonl`、`umetrack_hand_pose_trajectory.jsonl`、user profile 和 `headset_trajectory.csv`，视频在VRS中；完整数据需登记邮箱取得manifest。独立的[HOT3D-Clips](https://huggingface.co/datasets/bop-benchmark/hot3d)官方HF可直下，不需该manifest；3,832个5秒片段中2,804为train，累计420,600帧/约3.89小时，非去重净时长。train tar有MANO/UmeTrack、shape、camera和原序列时间戳，图像与JSON混包，未确认官方pose-only轻包。Clips与full重叠，不能重复统计。手部数据依[发布协议](https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/main/hot3d_dataset_license_agreement.pdf)采用CC BY-NC-SA 4.0，代码Apache 2.0不等于数据许可。

### DexWild

[项目](https://dexwild.github.io/)、[论文](https://arxiv.org/html/2505.07813v1)、[代码](https://github.com/dexwild/dexwild)和[实际HF数据卡](https://huggingface.co/datasets/boardd/dexwild-dataset)都已核验。五任务human条目合计7,917 episodes、2,924,588 transitions；若按30Hz解释，约27.08小时。30Hz依据来自论文环境/策略频率，各段实际采样间隔仍须通过timesteps.txt验收。33+小时和9,505 episodes包含机器人，论文9,290 human demos也不是当前release数量。

原始包列出 `left/right_manus_full.pkl`、`left/right_manus.pkl`、`*tracker_cam_frame_abs.pkl` 与 `timesteps.txt`。**应从原始human字段构造状态，不能用LEAP机器人action替代完整人手。** 腕部须结合tracker与相机/SLAM变换，不能仅凭文件名当成固定world frame。论文含追踪筛选、插值和Gaussian smoothing，参考轨迹已有处理。

HF标注MIT，总文件约2.14TB；可按任务、人/机器人区分，但未找到官方轻量pose-only包。大包传输、原始骨架和坐标映射是比MANO数据更实际的接入成本。

### DexCap

[官方仓库](https://github.com/j96w/DexCap)链接的[公开数据卡](https://huggingface.co/datasets/chenwangj/DexCap-Data)只列 1.5 小时、两个任务，含 raw ZIP 与 processed HDF5，标注 CC BY 4.0。不要把论文系统的数据规模与公开子集混为一谈。

raw wiping约22.7GB、packaging约42GB。`pose_2.txt / pose_3.txt`为双腕6DoF，`left/right_hand_joint.txt`为palm-local人手关节；world参考为初始胸前T265。公开关节数量尚未通过文件确认。[论文](https://arxiv.org/html/2403.07788v2)采集60Hz，但长录制会降帧并可能出现SLAM jump。

TWM不需要胸前RGB-D。**glove指关节与SLAM全局腕位姿须分别记录质量来源**；processed HDF5偏机器人训练，不应使用还舍弃小指的LEAP 16D action代替人手。

### HumanTouch

[SparkLab官网](https://xsparkai.com/sparklab/humantouch/)及作者提供[Hugging Face](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch)、[ModelScope](https://www.modelscope.cn/datasets/chuqiaoLyu/Xspark-HumanTouch)入口；本次在HF核验到实际parquet、meta与视频文件，未成功读取ModelScope内容。官方称首批约100小时，HF目录X001–X010、文件约2.09TB；**计划中的1,000小时不能当成已发布量**。Viewer的文件级rows也不能当成动作或帧数。

本次成功读取的元数据：

| 分区 | Episodes | Frames | FPS | 元数据推算时长 |
|---|---:|---:|---:|---:|
| [X001](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch/blob/main/X001/meta/info.json) | 2,217 | 3,832,214 | 60 | 17.74小时 |
| [X002](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch/blob/main/X002/meta/info.json) | 3,836 | 2,733,581 | 60 | 12.66小时 |

两分区合计约30.40小时，不能写成全库总时长或质量筛选后的净时长。X003–X010因网页cache miss/访问限制未完成逐项核验；不能由旧缓存推断未发布。

字段包括head/left_wrist/right_wrist各7D、每手 `position[25,3]` 与 `orientation[25,4]`、joint/source validity和timestamp。[episode data contract](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch/blob/main/X001/meta/humantouch/episodes/episode_000001.json)声明米、xyzw quaternion、world coordinates与joint names。还需实际parquet确认骨架/腕部数值一致、坐标手性、tracker到解剖wrist偏置。官方23DoF描述与25-joint schema不是同一个概念。

这是位置插值、旋转SLERP后的统一60Hz时间轴，不代表所有硬件均原生60Hz。[format说明](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch/blob/main/X001/meta/humantouch/format.json)没有robot state/action；应做人体骨架映射，而非机器人retarget。

可只下载 `X*/data/**/*.parquet` 与 `X*/meta/**`，[真实episode文件目录](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch/tree/main/X001/data/chunk-000)与视频分开。HF将许可指向ModelScope，但本次未读到具体条款；**正式纳入训练前需核实许可**。因此推荐优先验收，不将它写成已可无条件训练的成熟数据源。

## 小规模传感器补充

| 数据 | 已核验能力 | 获取与建议 |
|---|---|---|
| TCDHands | AMASS 表中 1 人、62 序列、8.05 分钟；原采集每手 20 markers、120 Hz | [AMASS](https://amass.is.tue.mpg.de/)注册参数包；多种手势/打字等，适合非抓取 OOD，规模很小 |
| SAMP | 单人约 100 分钟、30 Hz；Vicon→SMPL-X，主要家具/全身动作 | [官方](https://samp.is.tue.mpg.de/)注册获取；可补 wrist/global motion，精细手指覆盖弱于抓取集 |
| FPHA / FPHAB | 1,175 动作视频、6 人、45 类、30 Hz；6 个磁传感器 + IK，21 joints | [官方仓库](https://github.com/guiggh/hand_pose_action)有申请表；学术非商业；世界系 `skeleton.txt` 可做轨迹，公开手腕朝向字段未确认 |
| HUMOTO | 官方 release 口径 735 序列；Rokoko suit+gloves，30 Hz，Mixamo rig | [公开仓库](https://github.com/adobe-research/humoto)仅约 10% 子集，完整数据需申请；需将 FK hand skeleton 转换到 canonical space |
| BigHand2.2M | 2.2M hand-pose 帧、10 人；磁传感器 + IK | [官方项目](https://kcvl-kaist.github.io/bighand2.2M/)和[实验室入口](https://labicvl.github.io/Datasets_Code.html)有获取流程；先作局部姿态恢复补充，时间戳/连续序列和腕朝向未验收 |

FPHA 的公开格式是 `frame_id + 21×3 world joints(mm)`，不能假定采集时传感器具备 6DoF 就等于 release 已给完整腕旋转。[论文](https://openaccess.thecvf.com/content_cvpr_2018/papers/Garcia-Hernando_First-Person_Hand_Action_CVPR_2018_paper.pdf)

HUMOTO 的全库时长约 7,875 秒，经过 artist cleanup/retargeting；这支持运动先验学习，但清理后的动画不能视为未处理的独立评测真值。原 arXiv v1 写 736 序列，现发布仓库写 735，采用 release 口径并保留差异。[采集方法](https://arxiv.org/html/2504.10414v1)；[工具](https://github.com/Jiaxin-Lu/humoto)

## AMASS 必须筛选子集

[AMASS 原论文](https://files.is.tue.mpg.de/black/papers/amass.pdf)明确：**没有 hand markers 的运动，手姿态使用 MANO 平均值。** 不能看见 SMPL-H/SMPL-X 的手参数字段，就认为存在真实手指动态。

已找到有用证据的白名单包括 GRAB、TCDHands、SAMP；不是声称 2026 年全集只有这些子集。其他序列只有 wrist/body 可靠时，可考虑部分监督，不能给不曾测量的手指贴上真值标签。

[HMP 论文](https://openaccess.thecvf.com/content/WACV2024/papers/Duran_HMP_Hand_Motion_Priors_for_Pose_and_Shape_Estimation_From_WACV_2024_paper.pdf)选择了 GRAB/TCDHands/SAMP；但[当前公开代码](https://github.com/enesduran/HMP/blob/main/src/datasets/amass.py#L1089)默认 `train=[GRAB,ARCTIC]、valid=[SAMP]、test=[TCDHands]`。论文用数与当前工程 split 要分开陈述。其[预处理](https://github.com/enesduran/HMP/blob/main/src/scripts/process_amass.py)可参考，不应照搬高 motion-variance 过滤而删除本项目需要的静止抓握、急停和真实转向。

## 可训练但应单独分层的视觉来源

这些数据的数值状态可以输入 TWM，因此技术上能用于 pretraining；但若混入主池，论文不能再把整个模型描述为仅从独立 glove/MoCap 学习。建议作为明确的扩展实验，并保留来源标签。

此处分层依据是测量来源与项目研究问题，并不是已证明glove/marker的每个标签都优于多视角拟合。手套标定、marker拟合、腕部SLAM和视觉重建各有误差，最终权重应由样本质量验收与对照实验决定。

| 数据集 | 可用内容和规模 | 对本项目的主要限制 |
|---|---|---|
| GigaHands | 34 小时、56 人、约 14K clips；51-view markerless hand reconstruction | 高覆盖多视角拟合，仍非 glove/marker GT；部分发布 joint 版本已 recenter/normalize |
| TACO | release v1 为 2,317 序列，精细双手操作 | **手由多视角关键点/三角化/MANO 优化获得**；MoCap 主要跟踪物体和头戴相机 |
| EgoDex | 官方 829 小时、30 Hz；Apple Vision Pro/ARKit episode transforms | 设备视觉估计；confidence为可选字段，低质量片段和episode frame需处理 |
| HoloAssist | 论文166小时、现官网169小时；HoloLens 2 hand tracking | 设备追踪，非独立 glove GT；姿态数据可单独下载 |
| SHOW3D | 当前官方 20 小时、2,137 recordings、60 Hz；多相机拟合 | calibration 的“world”是移动背包 rig，不是静止物理世界；不直接当真实 global wrist 运动 |
| DexYCB | 1,000 个约3秒的多视角抓取序列，30 Hz；MANO 与21 joints | 多视角 RGB-D/人工2D标注联合优化；582K为图像口径，非582K独立运动时刻 |
| InterHand2.6M / Re:InterHand | 有30fps版本、world joints/MANO；后者提供约17GB MANO fits | 多视角/学习型拟合；需按 frame index 重建连续段与去除多相机重复；不是物体操作主池 |
| HO-3D v3 / H2O-3D | 公开手物序列，MANO/3D标签 | HOnnotate 视觉优化标签；更适合额外重建评测/有质量筛选的补充 |

来源与下载细节：

- [GigaHands 官方仓库](https://github.com/brown-ivl/GigaHands)：手参数、keypoints与视频可分开下载。优先审查raw keypoints、MANO params和新版aligned joints，确认整段坐标变换固定且保留腕部位移；旧 `keypoints_3d_mano` 做过中心化。数据按CC BY-NC 4.0。
- [TACO官方下载说明](https://github.com/leolyliu/TACO-Instructions)及[论文](https://arxiv.org/html/2401.08399v1)：手来自多视角RGB关键点、三角化与MANO优化；物体和头戴相机由MoCap跟踪。Dropbox发布的v1有2,317序列、151 triplets、30Hz，`Hand_Poses`与视频分目录，CC BY 4.0。
- [EgoDex 官方仓库](https://github.com/apple-aiml-research/ml-egodex)：官方ZIP包含视频MP4与pose HDF5，未发现官方独立pose-only发布包；CC BY-NC-ND。约725小时train、7小时test、97小时extra，不全是训练量。设备估计可用于规模实验，不能作为与视觉独立的高精度金标准。
- [HoloAssist 官网](https://holoassist.github.io/)：hand pose 包单列约219.24GB，CDLA v2；“总录制169小时”不意味着每帧双手标签都有效。[论文](https://arxiv.org/abs/2309.17024)使用166小时口径。
- [SHOW3D 数据卡](https://huggingface.co/datasets/facebook/show3d-dataset)与[官方 loader](https://github.com/facebookresearch/show3d)：2026-10-03 hand pose v3 提供 UmeTrack/MANO/MHR，公开训练姿态、test held back，CC BY-NC 4.0。可只取 `hand_pose`、profiles、索引；跨时间 moving frame 不能用一个常量旋转消掉。
- [DexYCB 官网](https://dex-ycb.github.io/)与[工具字段](https://github.com/NVlabs/dex-ycb-toolkit)：CC BY-NC 4.0，公开119GB总包/按被试分包；`pose_m`为PCA MANO参数，`joint_3d`在camera frame，缺失有零/-1约定，需解码和选单一参考视角。
- [InterHand 官方](https://mks0601.github.io/InterHand2.6M/)提醒部分序列因导出缺帧低于名义fps；[Re:InterHand](https://mks0601.github.io/ReInterHand/)可独立下载MANO和帧索引、按CC BY-NC 4.0使用。渲染相机或重光照变体不是新运动。
- [HO-3D 官方仓库](https://github.com/shreyashampali/ho3d)给出下载与2024年发布评测GT的更新；后续已核验[Terms of use](https://github.com/shreyashampali/ho3d#terms-of-use)为学术非商业用途并有再分发、修改及商业使用限制，不能因repo公开而默认任意用途。

## 聚合数据 HandX 的特殊情况

[HandX 官方发布](https://huggingface.co/datasets/alexzhang598/HandX)已经有约8.91GB纯motion/参数文件。每条 `motion` 是 `[60,2,21,3]`，另有每手 MANO pose/trans；公开84,729训练条目、9,648测试条目。适合快速搭建短窗 loader，但并非54.2小时全新高精度 MoCap。

其[官方 source mapping](https://huggingface.co/datasets/alexzhang598/HandX/resolve/main/handx_to_dataset_source_release.json)经本次读取统计如下：

| 来源 | Train entries | Test entries |
|---|---:|---:|
| HoloAssist | 79,609 | 8,829 |
| GigaHands | 3,943 | 695 |
| HOT3D | 704 | 80 |
| HandX 自采 | 473 | 44 |
| ARCTIC，映射含但本release不分发 | 1,699 | 207 |
| H2O，映射含但本release不分发 | 763 | 88 |

据每条60帧/30Hz计算，映射中的自采517条短片合计约17.23分钟；这是**已发布处理短片的长度统计**，不代表原始采集全部只有这么长。聚合规模主要来自设备视觉追踪，应按来源拆分。

读取时以实际 NPZ keys 与 mapping 做对应，忽略没有数组的源条目；不能把映射总数当作发布文件数。源坐标已 canonicalize，适合短窗恢复；长序列反馈和真实世界腕轨迹应回到原始来源确认。ARCTIC/H2O需要自行获取。[官方许可和第三方说明](https://github.com/handx-project/HandX)中HandX作者发布部分采用Snap Inc. Non-Commercial Research License，各来源继续受各自条款约束，MANO另行。不要与原源数据重复计量或跨split泄漏。

本次通过HTTP Range读取test_mano.npz的ZIP目录，确认实际9,648 keys对应发布metadata前缀；未加载全部数组。自采部分采用36台OptiTrack、每手25 markers；聚合release的低活动过滤会减少静止抓握/单手活动，不能直接沿用为本项目唯一训练分布。[论文与预处理说明](https://arxiv.org/html/2603.28766v1)

## 暂不升级为核心候选的其他新资源

- [HRDexDB](https://huggingface.co/datasets/HRDexDB/HRDexDB)：人类手标签为多视角HaMeR/三角化/MANO；手套用于机器人遥操作。3.2K是人和机器人合计，不能当3.2K人手glove序列。[方法说明](https://arxiv.org/html/2604.14944v2)
- DexUMI / DEXOP：未核实完整未retarget的人手骨架序列，暂不将机器人/外骨骼actions作为人手关节GT。
- World In Your Hands：[官方issue #4](https://github.com/tars-robotics/World-In-Your-Hands/issues/4)有用户报告所获action约47分钟，这是尚未独立复验的下载反馈。不能据宣传小时数直接认定完整action全集已经可用，也不能据单条反馈断言全库一定只开放47分钟。

## 对本项目的建议训练路线

1. **先验证最干净、最易解释的来源。** 用 GRAB + ARCTIC 的训练主体做 TWM-Base；统一状态后先训练 corruption/restoration。只需要运动参数，不需要先运行 HaWoR。
2. **增加任务多样性。** 加 OakInk2；根据目标评测协议决定 HOT3D 用作训练域还是完全留作外部测试。评测保留不是因为它不能用于预训练，而是为了可解释泛化。
3. **增加真实手套操作。** 为 DexWild 原始human字段建adapter；HumanTouch在许可、坐标和实测样本校验完成后进入扩展。DexCap用于小规模对照。
4. **分开测试规模收益。** 比较 sensor/MoCap-only 与加GigaHands/TACO/SHOW3D，再与加EgoDex/HoloAssist。这样能回答收益来自高精度轨迹还是大规模设备估计。

不在没有实验的情况下给出最优混合比例或模型规模。小数据集应按source/subject/task采样，避免超大设备追踪库淹没高质量核心池。

## 数据接入时必须落地的契约

- 统一输出 `wrist_pos、wrist_rot、20 non-root local joints、timestamp/dt、handedness、quality/validity`；保留原字段和来源，避免不可追溯转换。
- MANO/SMPL-X的全局参数要解码，腕坐标由统一关节定义得到，不能直接复制root或pelvis translation。
- 分开 `finger_source` 与 `wrist_source`；字段缺失用 supervision mask，不填伪造GT。只给21点时可定义几何掌坐标，但应标为派生frame，不冒充传感器腕姿态。
- 确认 frame 是否固定于场景；移动相机/背包frame未经处理会混入头/身体运动，影响速度与短期预测目标。
- 以真实时间戳重采样。30/60/120Hz混合时保证窗口物理时长一致；旋转在SO(3)上处理，不逐分量线性平均。
- 沿连续序列切窗，遇到缺帧/身份变化/追踪reset要处理；多视角和同源聚合版本去重。
- train/validation/test按主体、原序列及source划分；不能用测试motion GT预训练后声称zero-shot OOD。
- 核查手骨长、速度尖峰、rotation合法性、缺失比例及干净输入退化；筛选不可把真实急停/快动作一律删掉。

## 当前核验边界

本轮检查了公开文档、部分源码、release目录和元数据，没有完成各库全部文件的下载/数值验收；注册数据没有获得访问权。推荐表的“可用”指有符合任务的数据与获取路线，不代表adapter已经跑通。最终许可和条款以使用时对应发布版本为准。
