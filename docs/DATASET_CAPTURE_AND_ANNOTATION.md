# 八个TWM候选数据集的采集与标注来源

核验日期：2026-10-13。依据官方论文、采集说明、发布schema与固定版本代码；仅文献和源码调查，未处理轨迹或验证实际误差。本报告解释已有数据怎么产生，不改变D16的99D输出或D18的glove/MoCap精度优先原则。

## 测量与标注的区别

光学MoCap测皮肤表面的反光marker位置；手套测传感信号并由厂商解算骨架。关节中心、骨段方向、模型旋转和MANO mesh通常在测量之后经过标定、拟合或转换得到。“数据带MANO”不等于仪器直接测了所有MANO角度。marker尺寸、设备指标、拟合残差、视觉预测误差是不同量，不能相互替代。

| 数据 | 手指参考主要来源 | 腕/整体位姿主要来源 | 发布姿态层 |
|---|---|---|---|
| GRAB | Vicon皮肤marker＋个体扫描＋MoSh++ | 全身模型在MoCap坐标中的拟合 | SMPL-X、独立MANO及个性化template |
| ARCTIC | Vicon皮肤marker＋个体扫描＋MoSh++ | 全身/腕关节拟合到MoCap world | world MANO/SMPL-X、raw markers |
| OakInk2 | OptiTrack marker＋人工清理/多视角补齐＋两阶段拟合 | 上身/手的全局模型拟合 | SMPL-X、MANO、MoCap/RGB索引 |
| HOT3D | OptiTrack marker＋个人手扫描＋native UmeTrack拟合 | marker驱动手模型到场景的变换 | native UmeTrack和转换MANO |
| UmeTrack独立集 | 光学marker＋个人扫描＋marker-based tracker | root/wrist全局变换 | native20轴模型、当前release另有MANO |
| HumanTouch | MANUS EMF/MetaGloves骨架解算＋触觉辅助标定 | 团队系统用VIVE tracker，公开episode配置待逐条核 | 每手25点/四元数方向、腕7D，未见原生MANO |
| DexWild | MANUS SDK解算骨架 | ArUco到相机；ZED另录相机运动，world组合待验 | raw人体25节点骨架与tracker；另有robot action |
| DexCap | Rokoko Studio解算骨架 | 手背T265视觉惯性SLAM，与胸前初始坐标对齐 | raw人体21节点/方向和腕；另有四指LEAP数据 |

## 1. GRAB：光学动捕与个性化数字人拟合

10名参与者与51种物体互动，包括抬起、使用、交给别人和左右手传递。54台Vicon红外相机以120Hz追踪marker；每人99个marker，其中36个用于两手手指，手部marker贴在手背皮肤，掌面保持自由。

标注链为：**marker轨迹→人工清理→个体扫描确定形态→MoSh++拟合SMPL-X运动→独立MANO手部参数/模板**。四位技术人员清理漏点、错ID和ghost markers。运动拟合固定个人形态，优化姿态与全局位置；拟合涉及手部pose空间，解码需遵循源PCA/mean配置。接触标签来自重建人/物表面距离和穿插处理，不是触觉传感器测压。

独立左右手发布中有translation、rotation/articulation与personalized `v_template`。转99D时保留个体模板和原解码几何；不能默认β10能无损表达该模板，也不能将MANO translation直接当解剖wrist。

来源：[论文§3.1–3.3](https://arxiv.org/html/2008.11200)、[发布结构](https://github.com/otaheri/GRAB#contents-of-each-sequence)、[官方MANO decoder](https://github.com/otaheri/GRAB/blob/master/grab/save_grab_vertices.py)。

## 2. ARCTIC：带同步视频的双手关节物体操作

10名参与者操作11类可开合/转动的物体，如剪刀、笔记本、盒子和笔记本电脑。54台Vicon追踪身体、双手、物体及头戴相机marker，另录8个固定RGB视角与1个第一视角；发布参考/视频按30Hz使用。

先用3dMD扫描得到个人身体和手形态，注册为个性化SMPL-X模板；MoSh++完善marker与模型表面的对应，再优化每帧姿态使表面marker吻合。全身骨架帮助估计手腕关节及每手global变换，之后发布MANO参数。物体旋转轴和开合状态由其刚体marker标定/解算；接触为几何派生标签。

world MANO含root rotation、pose、translation、shape和fitting error，另有raw marker及对应。对TWM有完整位姿参考，但仍保留测量与模型拟合的区别；手表面软组织变形并未由刚性骨架完全表达。

来源：[论文§3及附录2.2](https://arxiv.org/html/2204.13662)、[官方数据文档](https://github.com/zc-alexfan/arctic/blob/master/docs/data/data_doc.md)。

## 3. OakInk2：OptiTrack与人工补点的复杂双手任务

12台OptiTrack Prime13W红外相机追踪上身、双手和物体marker；4台RGB相机（3个外部、1个第一视角）记录操作。内容包含复杂任务及其primitive片段，如整理、容器操作等。

三位专业标注员先在Motive中清理ghost、错ID、合并marker及短缺口；极端遮挡时，在多视角编辑器中标marker的2D位置并三角化恢复3D。补的是**表面marker**，不是人工逐帧填写全部MANO角度。

采用受MoSh++启发的两阶段SMPL-X拟合：T-pose阶段估计个人shape与marker表面位置；固定这些量后拟合逐帧pose，结合marker误差、姿态/解剖先验、手物穿插与时间正则。release提供SMPL-X及MANO，MANO为16个wxyz旋转四元数、translation和β10。拟合及已有时间正则的误差须保留，MoCap/RGB帧索引不能混用。

来源：[论文§4.1](https://arxiv.org/pdf/2403.19417)、[官方补充A.1–A.3](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Zhan_OAKINK2_A_Dataset_CVPR_2024_supplemental.pdf)、[发布schema](https://github.com/oakink/OakInk2)。官方PDF本轮已只读提取文字核验；网页工具403/大文件限制不等于源文缺失。

## 4. HOT3D：独立marker参考配第一视角视频

19名参与者用Project Aria或Quest3在实验室搭建的厨房、办公室、客厅场景操作33种刚性物体。OptiTrack系统为数十台红外相机；官方未给本报告可确认的精确台数。每手19个直径3mm的皮肤marker，个人手形态由扫描获取。

标注链为：**个人扫描与marker注册→native UmeTrack模型拟合marker→转换MANO**。手指运动及腕到场景的位姿来自独立marker驱动拟合，不是UmeTrack视觉网络对头戴图像的预测。人工检查模型投影与视频的对齐并给valid mask；Clips还有标签完整性、可见性及曝光筛选。

手的场景位姿与相机定位来源应分开：Quest标定及采集时头显位姿均由动捕追踪，Aria的MPS/SLAM再对齐OptiTrack坐标。官方明确转换MANO整体略逊于native UmeTrack，建议用native关键点监督；3mm是marker直径，不是joint GT误差。

来源：[论文§3、Appendix A.3/B/C](https://arxiv.org/html/2411.19167v2)、[toolkit重要说明](https://github.com/facebookresearch/hand_tracking_toolkit#important-notes)、[Clips发布说明](https://github.com/facebookresearch/hot3d/blob/main/hot3d/clips/README.md)。

## 5. UmeTrack独立集：自由手势、双手互动和pinch

这是2022年的独立采集库，不是HOT3D的另一名称。原论文53人、1397段15秒/30Hz真实序列；36台MoCap相机追踪手上直径3mm的marker，头显另录广角单色视频。原文未确认这36台的品牌或每手marker数，不能套用HOT3D的配置。

每人扫描得到个人骨架与mesh，marker-based tracker求参考运动。native骨架是20个旋转轴的角度及root6DoF，不等于MANO15个完整SO(3)。人工pinch事件标签与自动marker姿态标注是不同产物。

Unity用同一批真实运动渲染不同背景/照明的合成视频，因此纯轨迹不新增一份motion。当前toolkit同时读取native `joint_angles/T_world_from_wrist`和转换MANO及shape；转换失败须mask，原论文统计与当前challenge发布量分开。其非物体自由运动是TWM分布补充。

来源：[原论文§3–4](https://arxiv.org/html/2211.00099v1#S4)、[当前发布](https://huggingface.co/datasets/facebook/hand_tracking_challenge_umetrack)、[decoder](https://github.com/facebookresearch/hand_tracking_toolkit/blob/main/hand_tracking_toolkit/dataset.py)。

## 6. HumanTouch：手套骨架、VIVE腕追踪与触觉辅助标定

SparkLab@Xspark AI系统由内层柔性压阻触觉手套、MANUS EMF/MetaGloves、VIVE tracker及头/腕相机组成。官方主页Figure5的设备说明及同团队DexTouch-WM v2 §III-C提供VIVE证据。精确产品代际和公开HF每个episode的采集配置仍未逐条核验；后者不再等于完全不知道腕追踪技术。

MANUS解算手指骨架，结合腕追踪和标定变换形成25点/方向。采集者用拇指触碰其它指尖；拇指与指尖触觉同时出现接触onset的事件约束数字手对应区域的重合，用于调整骨长、关节偏移及传感区域对齐。触觉是这条采集标定链的证据，本项目TWM不需新增触觉输入。

数据对齐到60Hz公共时间线：位置插值、旋转SLERP，图像选最近有效观测；大缺口/缺必需流拒收，episode人工审核。HF原人体字段为每手25×3位置、25×4方向、头/腕7D、时间与validity，米/xyzw/world；未见MANO pose/β。

版本边界：主页初版约100h/10任务，当前HF十份meta累计约132.75h非QC净量；DexTouch-WM论文约100h/50预训练任务＋另4任务，与HF的逐episode对应未核。不能把其67D机器人action当本项目99D人体GT，也没有全公开库独立毫米误差证明。

来源：[官方采集与标定主页](https://xsparkai.com/sparklab/humantouch/)、[DexTouch-WM §III-C](https://arxiv.org/html/2609.20649v2#S3.SS3)、[固定HF README](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch/blob/81822a10cac5def2638db546eaeced581d71a8b5/README.md)、[字段](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch/blob/81822a10cac5def2638db546eaeced581d71a8b5/X001/meta/info.json)。

## 7. DexWild：MANUS手指与ArUco腕标记

采集者在多种真实环境直接用人手执行倒水、叠衣、插花等操作。MANUS SDK的full输出含每手25个骨架节点的位置与四元数；short选择部分节点，机器人action另经IK retarget。TWM应保留raw human full，而非LEAP等机器人动作。

腕标记立方体通过ZED Mini图像/深度求到相机的位姿，ZED另记录相机运动。ROS近似时间同步及离线匹配不等于硬件严格同步。腕追踪丢失超过25%的episode剔除，缺失段插值/SLERP后平滑；已有处理需记录，不能视为未处理测量。

**本轮源码事实：** 固定commit `5fa34af5115bb6f54320d0071c3b1d4864cd224a`，`TrackerProcessor.single_tracker`计算相对首个tracker pose；`process_tracker`读取ZED pose但未显式组合到tracker，仍以`world_abs`命名保存。核查上游`CubeTracker/ZedProcessor/mass_process_data`后，同一处理路径未发现额外camera-to-world组合。因此不能凭文件名认定世界真值；该发现限此代码路径，公开样本/发布处理版本仍待验，不能推断整个release均无world。

来源：[论文采集/处理附录](https://arxiv.org/html/2505.07813v1)、[原始手套读取](https://github.com/dexwild/dexwild/blob/5fa34af5115bb6f54320d0071c3b1d4864cd224a/dexwild_ros2/src/glove/glove/read_and_send_zmq.py#L150)、[TrackerProcessor](https://github.com/dexwild/dexwild/blob/5fa34af5115bb6f54320d0071c3b1d4864cd224a/dexwild_utils/dexwild_utils/TrackerProcessor.py#L81)、[ZedProcessor](https://github.com/dexwild/dexwild/blob/5fa34af5115bb6f54320d0071c3b1d4864cd224a/dexwild_utils/dexwild_utils/ZedProcessor.py#L259)。

## 8. DexCap：Rokoko骨架与三台T265的腕/身体追踪

人体操作者戴Rokoko手套，左右手背各一台T265，胸前另有T265和L515 RGB-D相机。T265通过双鱼眼和IMU做SLAM。初始化将三台T265置于已知关系的胸前卡槽，再把两台移到手背，固定初始变换使它们共享胸前T265初始world。

Rokoko Studio给解算后的骨架；固定采集代码保存每手21节点位置/四元数，而不是人工逐帧标MANO。论文称手套为EMF；同期厂家资料有EMF数据和IMU融合，不能改称纯IMU。具体产品代际、传感点位置及采集固件未核。

raw人体骨架、腕SLAM和机器人训练文件不同：四指LEAP retarget明确舍弃小指，因此不能替代完整人手参考。人工主要整理示范起止、重置阶段和机器人空间对齐。论文60Hz记录会因长session降帧/SLAM跳变，采集代码读最新手套值也不证明严格同步；实际timestamp和腕/手指质量分别验。公开子集30min擦拭＋60min包装，不是全论文语料。

来源：[论文§III-A及采集附录](https://arxiv.org/html/2403.07788v2)、[固定人体骨架采集代码](https://github.com/j96w/DexCap/blob/4b0bed0966c87368f3cde4476aadb7585c3b94b5/STEP1_collect_data/redis_glove_server.py)、[固定记录脚本](https://github.com/j96w/DexCap/blob/4b0bed0966c87368f3cde4476aadb7585c3b94b5/STEP1_collect_data/data_recording.py#L230)、[Rokoko同期固件](https://support.rokoko.com/hc/en-us/articles/22103913523857-Smartgloves-firmware-v2-3-0)、[公开数据卡](https://huggingface.co/datasets/chenwangj/DexCap-Data)。

## 精度判断：工程建议与实测排名分开

2026-10-13用户进一步问哪种更准确。**工程判断：对完整世界系腕6DoF＋手指参考，更信通过标定、清理与有效性筛选的外部多相机MoCap；局部手指不能一概判所有glove低于marker。** 这是根据采集链作出的选择建议，不是已验证的八库精度排行榜。

- world腕：GRAB/ARCTIC/Oak/HOT/UmeTrack的外部marker参考可优先验收；形态拟合、tracker到解剖wrist偏移仍有误差。DexCap腕依赖T265 SLAM；DexWild需先核camera/world处理链。VIVE追踪也需设备/标定验收，未给出与这些库同基准的姿态误差排名。
- 局部手指：个人扫描/形态标定的MoCap可追踪片段提供可审计参考；EMF glove抗图像遮挡，保留HumanTouch/DexWild主要候选地位。分别检查marker遮挡与模型拟合、glove个体骨长/rest frame/佩戴和环境条件；不存在由相机台数或厂商名称直接推出的排序。
- 转换到MANO：Meta官方明确native UmeTrack优于转换MANO，支持保留原生几何监督。99D编码不增加原数据准确度；姿态参数误差与原生→MANO几何残差分别报告。

没有独立真值时，marker重建残差、自接触一致性、平滑/骨长稳定只能作QC，不能称真实joint毫米误差。GRAB主文RefineNet的4.4mm是网络重建指标；DexCap附录约0.4/0.8cm是闭环终点漂移，均不能拿来当全库逐帧GT精度。来源：[GRAB](https://arxiv.org/html/2008.11200)、[DexCap附录](https://arxiv.org/html/2403.07788v2)、[Meta转换说明](https://github.com/facebookresearch/hand_tracking_toolkit#important-notes)。

本轮建议按world腕位置/朝向、root-relative joints/指尖、可用方向、原生→MANO残差分项验收；MoCap与glove同时进行，不预设固定配比，也不将手套降成只补数据量。此为待验证建议，没有运行误差评测。

## 对99D预训练的处理要求

已有MANO按源decoder转换；glove/native skeleton先保留原几何，再标定/拟合或直接几何监督99D。分别保存measurement来源、个人形态、转换残差、腕world定义、人工/插值/平滑mask、时间与split。不能由接触或平滑外观宣称姿态精度，也不能把未测的twist、缺失world腕补成GT。实际标定、解码、样本处理和实验全部在hongfei执行。[训练契约](TWM_PRETRAIN_PLAN.md)
