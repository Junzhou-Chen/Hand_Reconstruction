# 可直接获取、面向99D的人手运动数据

日期：纽约2026-10-16夜间（UTC已10-17）。本轮用户明确暂缓DexWild，转而尽可能寻找可直接下载、适配99D的人手运动数据集。本表是官方发布／论文／源码／下载元数据调查；除延续HumanTouch外，没有启动新候选下载或新训练。

## 当前结论与判断标准

此前四库OakInk2／HOT3D／独立UmeTrack／DexYCB仍是唯一已入训motion池。**ARCTIC和HO-3D v2已经下载，但尚未加入motion pool**；它们应先完成转换，不需要重新下载。

新增候选中，**InterHand2.6M 30fps、TACO、GigaHands、Re:InterHand**已核得独立姿态标注入口，能够提供完整MANO拟合参数。**UNOC、CeTI-Age-Kinematics**是本轮新增的公开sensor连续运动，提供BVH旋转而非现成MANO。HOI4D、H₂O-3D的公开页面／完整参数条件成立，但获取包的实际验收仍有未决项。

这里把三件事分别记录：①公开无人工审批且实际文件可取；②连续时间、wrist与15指关节参数是否足够；③参考来源精度及现有adapter是否已验收。**格式贴合不能代替精度证据，完整MANO拟合参考也不是所有旋转的直接传感器GT。** 用户本轮没有取消既有glove/MoCap参考精度优先原则，也未选择新的混合比例。

## 完整MANO参数、较易接入现有trainer

| 来源 | 官方可获取范围与只取标注体积 | 参数／时序与监督来源 | 本项目判断 |
|---|---|---|---|
| **ARCTIC** | 已下载343/343文件verified；motion只需raw_seqs、meta、splits。267 train的QC前原始量384,780双手状态 | Vicon→MoSh++／MANO；pose45＋root3＋trans3＋β10，30Hz。官方MANO nonflat mean；subject_vtemplates属于SMPL-X | 优先完成已有数据的完整99D adapter。与原四库相比的13.8%只是原始状态量，不是新增有效窗量。[字段与源码](https://github.com/zc-alexfan/arctic/blob/master/docs/data/data_doc.md) |
| **HO-3D v2** | **已下载完整约37.04GB**，可直接取`train.txt`和`train/*/meta`，无需RGB解码；未核独立官方train-meta包 | HOnnotate RGB-D多帧／多视角优化；`handPose48, handTrans3, handBeta10`。官方flat mean路径直接赋fullpose，不能再次加mean；固定camera参考和录制去重须审计 | 较低获取成本的新增motion来源；尚未prepared。v3主要是同批运动的标注升级，不能与v2重复计独立motion。[官方发布与decoder](https://github.com/shreyashampali/ho3d) |
| **InterHand2.6M 30fps** | [官方30fps annotation Drive目录](https://drive.google.com/drive/folders/1MJ4ztwhMJ1RyFdsxzAprExvzr1cV6otS)，公开无人工审批。train MANO 142,194,147B＋train joint_3d 184,442,740B＝**326,636,887B／326.64MB**；三split的MANO总231,531,856B | 多视角人工／网络关键点→三角化→NeuralAnnot MANO；`pose48, shape10, trans3`，世界系拟合。joint_3d含seq及validity，可辅助建立capture／sequence／frame轨迹，无需3.97GB多视角data.json | **最值得优先取的小标注包**，补双手／自由手势。必须使用train；5fps与30fps同源不可叠加，2.6M是多视角图像口径。尚无本项目adapter／净连续时长验收。[官网与字段](https://mks0601.github.io/InterHand2.6M/) |
| **TACO V1** | [官方README](https://github.com/leolyliu/TACO-Instructions)链接[官方HF](https://huggingface.co/datasets/leolyliu/TACO)，public且非gated。`Hand_Poses.zip` **15,231,790,511B**；`Allocentric_Camera_Parameters.zip`7,960,127B。无需视频或物体模型 | V1实际发布2,317序列，逐帧pose48、trans3、shape10。YOLOv3／MMPose→多视角RANSAC→接触／时序MANO优化；**marker用于物体和标定，不是手部marker GT** | 双手操作与本任务贴合、数值包独立。官方`use_pca=False, flat_hand_mean=True, center_idx=0`，手trans已是解码J0，不可重复加rest-J0。按官方train/test_1～4列表取train，须验FPS及连续段。未下载／适配 |
| **GigaHands** | [官方README](https://github.com/brown-ivl/GigaHands)的[hand_poses.tar.gz](https://g-ad09a0.56197.5898.data.globus.org/hand_poses.tar.gz)，HEAD实测200、支持Range；**11,304,232,382B／11.30GB**，含params／3D点／camera，无视频 | HaMeR等2D检测→51视角三角化／滤波→MANO；全局变换和β可取。约34h、56人、14k clips是发布口径；视角和文本不增加独立motion | **完整参数池扩规模的重要候选**。使用原始`params`及frame/camera；旧`keypoints_3d_mano`已归一化和recenter，不能用作world腕。source EasyMoCap mean、单位、J0及β语义仍需首样本FK；论文16:3:1划分，公开ID清单尚待包内核实。未下载／适配 |
| **Re:InterHand** | [官方S3下载脚本](https://github.com/facebookresearch/InterWild/blob/main/tool/ReInterHand/download/download_mano_fits.py)，十capture HEAD均200。七train压缩包合**5,524,327,492B／5.52GB**；全部十个7,281,327,614B，官网17GB是解压量。包含参数和mesh，不含RGB | 多相机／3D扫描→V2V-PoseNet→NeuralAnnot；pose48、shape10、trans3，官方nonflat mean。7train／3test；train109,763独立时刻为官网帧表口径，不能以relighting／view重复计量 | 适合补双手交互和较可审计拟合参考，规模比图像宣传量小。训练只取七train包；真实时间基准见下节。未下载／适配。[官方说明](https://mks0601.github.io/ReInterHand/) |
| **HOI4D** | [官网](https://hoi4d.github.io/)2026-03-27起提供独立Hand Pose和Camera Parameters的OneDrive／百度链接；**不应继续概括必须申请**。本轮未成功核包大小或完成匿名文件传输 | poseCoeff48、β10、camera-frame trans3（米）；约20%帧人工关键点＋RGB-D拟合／传播／接触和时序优化。`output.log`有相机轨迹，须核变换方向／J0 | 格式与ego操作贴合；先完成真实获取验证，再适配。遵守release/test列表，不把4,000序列总量当已核完整train量。[官方字段与split](https://github.com/leolyliu/HOI4D-Instructions) |
| **H₂O-3D v1** | [Hampali官方仓库](https://github.com/shreyashampali/ho3d)直接给OneDrive分享，匿名页面可开；API401，独立标注包大小／程序化取得尚未核定。**与Kwon需注册的H2O不同** | HOnnotate双手扩展；左右pose48／Trans3／β10；5 RGB-D相机、17录制；60,998train／15,342test是图像而非独立时刻，官方FPS待核 | 完整拟合参数候选，获取验证后再排期；不能承诺现成免障碍自动下载或直接假定30Hz。[完整论文及补充](https://arxiv.org/html/2104.14639v2) |

TACO固定HF revision为`fa8ed8b4cf5d46949b90ef852128811a2854066a`，Hand_Poses.zip官方LFS SHA256为`db649d185ae818e63e9f14577067c1318d0751d35b213de8a06688637a1ece51`。InterHand train MANO Drive ID为`1pgPRPcL8usdb94aGOb-GF7CoQhy4kjOz`；train joint_3d ID为`1eqeYdJcm62gbuXxC_KQZba4F7cfPJ_Wi`。这些是后续获取依据，不表示本轮已下载。

## 新发现的sensor连续运动：有旋转但需骨架映射

| 来源 | 官方获取／采集 | 99D接口的实际差距 |
|---|---|---|
| **UNOC** | [官方仓库、MIT数据许可](https://github.com/facebookresearch/UNOC)，[combined body＋hands BVH](https://cloud.tugraz.at/index.php/s/ykqxA7HxYMnwbXr/download)匿名HEAD200；**918,229,742B**。23台OptiTrack／57个body marker＋NANSENSE每手16IMU；13人、211连续段、120Hz、论文约220min／1.6M净帧 | 推荐下一轮sensor适配审计。务必取**合并标定版**：单独hand BVH绑静态avatar、官方明确腕方向不可靠。四指各三级，thumb有额外一级且wrist含twist链，需合成局部旋转、MANO rest轴和J0／骨长映射，不能截取前三角度拼99D。[官方parser](https://github.com/facebookresearch/UNOC/blob/main/src/parsers/unoc_parser.py) |
| **CeTI-Age-Kinematics** | [2025论文](https://www.nature.com/articles/s41597-025-04818-y)、[官方Figshare](https://api.figshare.com/v2/articles/26983645)，CC BY4；[ZIP](https://ndownloader.figshare.com/files/49174882)匿名GET头200，**3,099,219,560B**，MD5`382273e83d5c6314e359a319cd9846ca`。Rokoko suit＋gloves，32人／1,563录制／100Hz | finger在BVH，TSV不提供完整finger。Maya IK骨架需要MANO rest／轴映射；IMU世界位置经start=end漂移校正，不能当独立光学world GT。适合日常活动／年龄覆盖的次级补充。HEAD403是签名GET方法限制，不应误判数据受限 |

HumanTouch正在下载，仍是native25点／方向＋腕参考，需输入桥接与partial监督。MySign、DexterHand、HandX自采等此前候选继续保留，来源与限制见[前轮总结](TWM_PRETRAIN_EXPANSION_SUMMARY_2026-10-16.md)；本轮没有把所有格式可转的BVH／FBX升级为已验收的MANO GT。

## 条件候选与排除项

- **Contact4D（3DV2026）**：[官方schema](https://github.com/jyuntins/Contact4D_Toolkit/blob/main/docs/annotations.md)有MANO root(3×3)、15finger(3×3)、β、trans、root_joint及world米制几何，格式很好。[官方HF](https://huggingface.co/datasets/Jyun-Ting/Contact4D)public／非gated，revision`6967ed84da133f57fdd99373ac58526d2468faa1`；448个未压缩TAR／34sessions共1,593,928,550,400B，混合媒体。没有已核独立标注包，未来可研究只取`processed_data`的Range方案；本轮未探包。手姿仍来自视觉重建，fingertip contact真值不等于所有手旋转的独立GT。
- **HIC**：[官方页面](https://hic.is.tue.mpg.de/download.html)提供17个指定序列的MANO fits、个人模板和shape。适合作小型手手／手物补充；本轮未核文件大小和源时间契约，不能直接写成大规模已就绪池。
- **AssemblyHands**：[官方annotation-only入口](https://github.com/facebookresearch/assemblyhands-toolkit)有双手42个3D joints／validity和标定，**没有完整MANO finger旋转**。适合部分几何监督，尚不能接现full17组trainer；时间稠密度也待审计。AssemblyHands-X未核到官方完整参数release。
- **CLUTCH / 3D-HIW（ICLR2026）**：[全文§3.2／§4](https://arxiv.org/html/2602.17770v1)正好每手99D，32k序列／12M hand poses；但由**HaWoR重建＋滤波**产生。[代码README](https://github.com/bala1144/CLUTCH)现有assets／checkpoints，完整motion与metadata是另一份release，本轮未核可直接下载入口。不能因99D一致就作为TWM修正HaWoR的独立干净GT。
- **EgoSteer-Egocentric**：[官方egosmith](https://github.com/egosteer/egosmith)发布annotations-only大规模轨迹，来自HaWoR／DPVO／Any4D伪标签。规模可观，属于后续弱监督池，不能用于证明独立姿态精度或无来源偏差。
- **FPHA、BigHand2.2M、TCDHands／AMASS**：FPHA官方仍要求表单；BigHand官方HF为gated=auto、匿名包401；AMASS明确注册。前两者主要21点参考而非MANO，不能绕过访问或骨架适配条件。FreiHAND／OakInk静态抓取不作为连续轨迹主池；5fps插值不制造30fps测量，多视角不制造新motion。

## 时间、99D和后续建议

完整MANO源可以提供全参数拟合监督，但仍需source decoder→FK→99D→FK往返验收：只加一次mean，区分trans和实际J0，按源β和左手模板约定解码，保存拟合质量／validity，不能仅检查数组长度。

**InterHand／Re:InterHand的frame编号不可统一除30。** 原采集有90Hz／30Hz设置；Re公开frame_list首capture正常步长3、后两个capture步长2，发布视频均标30fps。需要按capture官方采样设置／正常步长和segment链确认真实秒数，不能把每个编号跳跃当掉帧或随意串接。对约2秒future尤为重要。所有库先按原录制分train／val／test、跨motion／视觉共用排除表，再切窗口。

建议在已有高质量四库基础上，先完成**已有ARCTIC／HO3D适配**；新获取优先**InterHand30fps小标注、TACO、GigaHands、Re:InterHand train**，并另做**UNOC**的sensor映射小样本。此顺序同时考虑获取和工程成本，不是精度实测排名，也未替代用户的精度原则。新增数据究竟改善finger／未来预测，仍需分源固定验证和等预算对照；本轮没有新效果结论或新混合比例。
