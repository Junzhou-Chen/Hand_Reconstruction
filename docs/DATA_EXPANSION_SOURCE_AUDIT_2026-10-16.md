# 数据扩展来源核查：运动预训练与视觉联合

日期：2026-10-16。性质：官方资料／源码复核、来源选择建议与一个原生glove样本的远程结构验收；不是新增训练结果。

**同日夜间继续扩查：** 最新选库优先级、新增MySign／SynData glove子集、HOI4D直接下载入口及ARCTIC decoder勘误见[夜间重训前总结](TWM_PRETRAIN_EXPANSION_SUMMARY_2026-10-16.md)。ARCTIC的`subject_vtemplates`属于SMPL-X，不能误当其MANO必须注入的个性化手模板；完整99D输出与native部分监督边界保持。仍是选库建议，未启动新训练。

## 纽约10:54–11:00再次复核：当前TWM池与可扩充来源

用户要求核查TWM目前实际使用的数据、能否扩大以及改善方向；**本轮是只读调查与建议，没有启动新预训练、下载新大包、改架构／loss或改变正在运行的两库视觉5k清单**。本节取代下文历史状态中ARCTIC尚待账号／下载的描述。

实际motion manifest仍为`prepared/twm_four_sources_99d_v1/manifest.json`，SHA256 `f733aa15cd18ed6f57ee0f16ebdadb3bd0d52924c7d99d08cea988544710b26f`。D37完成10k后被当前联合训练载入，motion回放仍用同池；视觉训练的63652独立RGB不代表TWM预训练规模。

| 已入训来源 | Train已准备手状态 | Train重叠窗口 | 参考来源 |
|---|---:|---:|---|
| OakInk2 | 1414552 | 346860 | MoCap拟合MANO，world腕 |
| HOT3D | 660300 | 132008 | marker／native UmeTrack转换MANO，保留source world |
| 独立UmeTrack | 662422 | 151390 | marker／native骨架转换MANO，real train |
| DexYCB | 50861 | 6352 | 多视角RGB-D拟合MANO，固定一相机为参考系 |
| 合计 | **2788135** | **636610** | 两手分别计状态；窗口重叠，不等于同量独立运动 |

原val为61370窗。Oak原431 train录制、HOT已下载且排除评测来源的Aria＋Quest3、Ume real train、Dex官方s0 train均已按既有准备范围使用；更多RGB视角、Ume重渲染或重复切窗不算新的运动。D37 full coverage的窗量占比Oak54.49%、HOT20.74%、Ume23.78%、Dex1.00%；当前联合motion回放按旧配置等源有放回，二者不能混说。参数格式经工程验收不代表所有旋转均为传感器直接测量；Meta官方明确converted MANO略逊于native keypoints。

**ARCTIC现在已经可进入adapter验收。** 下载12:38:03UTC／纽约08:38完成，343/343文件verified、failed0。只读ZIP目录及官方train参数核实：`raw_seqs.zip`225334351B，301条含MANO标注录制；P1/P2的267 train身份完全相同，余34是val；两协议另列38 test，不能从其他格式补入训练。本轮只读取267 train参数作维度计数，得到192390个录制时间帧、左右合计384780个手状态，约为原池13.80%的额外prepared前原始量。**这是QC／适配／切窗前计数，不是已验收新增窗口量**。初样本有左右rot、pose45、trans、shape10、fitting_err；meta含个体模板。需核source mean、trans→J0、单位、模板／β语义、时序／有效性及跨任务统一排除表后入池。纯TWM只需raw＋meta，不依赖RGB处理完成。[官方数据职责与字段](https://github.com/zc-alexfan/arctic/blob/master/docs/data/data_doc.md)。

**glove仍是规模扩展的主要候选，尚未入训。** 重新读取官方固定发布：

- [HumanTouch](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch/blob/81822a10cac5def2638db546eaeced581d71a8b5/README.md)仍公开免申请，首版官方口径约100h、MANUS EMF每手25点、6DoF腕、60Hz参考时间线；1000h是计划，不能当现有量。原1597帧native结构验收与约9cm tracker／native根定义差仍有效，尚无完成的MANO标定与partial训练入口。
- [DexWild human](https://huggingface.co/datasets/boardd/dexwild-dataset/blob/df5cd775f985e79036ff23ad579abbe7752b8df1/README.md)仍公开免申请，按五个任务的人类行统计7917 episodes／2924588 transitions，排除robot行；总33+h混有人／机器人，不能全算人类运动。原人体MANUS及tracker可用于候选；既有tracker坐标处理风险未由本轮新样本解除。
- [GRAB](https://github.com/otaheri/GRAB)提供10主体、120Hz、左右MANO及个体模板；HMP官方motion prior训练使用GRAB等并另接ARCTIC，符合数值先验职责。需另取源访问许可，目前本项目没有该库数据，不作为公开glove开工前置。[HMP训练说明](https://github.com/enesduran/HMP#training-motion-prior-optional)。
- [Stanford DexCap](https://huggingface.co/datasets/chenwangj/DexCap-Data)的90分钟raw掌系人体点＋独立6DoF适合补充；[DexterHand](https://huggingface.co/datasets/pku-mocca/DexterHand)187MB公开参数可做小规模marker候选，其20／60Hz版本去重，完整数据仍在官方roadmap。二者不是同一库。

同时扩查了EgoEMG与ACE-Data-0等新资源：[EgoEMG](https://arxiv.org/abs/2605.05712)论文报告41参与者、60手势、10+h，但未在本轮核得可直接取得完整所需参考的发布清单；[ACE官方卡片](https://huggingface.co/datasets/ACERobotics/ACE-Data-0)仍写个人审批及数据未发布，HF显示大文件统计与该文字不一致、当前未获访问以核实，不能据宣传150h或平台体积直接纳入。暂不提高为优先可执行来源。

### 长未来与异构标签的实际约束

读取D37原run metadata的时间戳审计，没有重跑模型：

| Train源 | 有完整60步future的窗数／比例 | 实际达到至少2.0s的窗数 |
|---|---:|---:|
| OakInk2 | 334558／96.45% | 334558 |
| HOT3D | 65950／49.96% | 51528 |
| UmeTrack | 126567／83.60% | 126567 |
| DexYCB | 0／0% | 0 |

Dex这批窗口最长future仅1.5s。HOT采用实际时间戳，60帧可能略短于2.0s；上表两个计数定义不同，不据此称标签漏算。短future按mask训练，没有复制尾帧为真值。故新增长连续录制值得作为改善长预测覆盖的方向，但不能将现有预测局限唯一归因于数据不足。

现`ForecastTrajectoryWindows`显式要求完整17组MANO标签；不能只加HumanTouch路径就声称入训。建议保持99D输出，另实现校准后的native joint／方向部分监督：经MANO解码只比较确实对应的几何，源腕／局部姿态分别mask，未测twist不伪造旋转真值；骨长／个体形态和rest frame差异须先处理。输入端也需要桥接：99D历史／观测可由已标定骨架方向或带残差记录的IK得到，未知分量保留availability；这些派生输入参数不自动变成完整rotation GT，监督仍回到可靠native测量。60Hz数据统一真实秒数，不能沿用60步却把两秒任务变一秒。

建议下一版本先以ARCTIC完成完整参数适配，同时推进HumanTouch／DexWild小样本标定及partial监督，再作原四库、四库＋ARCTIC、再加glove的固定验证／等预算比较。来源权重尚无最优实验依据；大glove池不能按原始行数直接淹没参考更可审计的核心池。评测应包含真实视觉误差、干净输入误改、分源／左右／速度分层和1／2秒未来，预训练与视觉阶段共用原录制排除表。以上是建议，未实施新训练。

来源与远程审计：[实际池与ARCTIC计数](../reports/twm_data_pool_inventory_20261016.json)、[官方候选当前发布](../reports/twm_public_candidate_review_20261016.json)；完整产物在`evaluations/twm_data_expansion_review_20261016/`。既有传感器精度判断与源转换限制继续按下文解释。

## 本轮范围与实际状态

用户本轮授权推进此前建议的前两项：扩大运动预训练和视觉联合的数据，以及增加直接几何监督。此前第3项raw/candidate训练条件、第4项历史与motion回放、第5项增加训练步数暂不调整。此来源核查不改模型、受损样例构造、历史、任务比例、学习率或训练预算，也不自动把所有候选加入训练。

此前确实找过超过四个数据集，但“查到资源”“已经下载”“完成adapter验收”“已参与训练”是不同状态。

| 状态 | 截至本次核查的事实 |
|---|---|
| 已下载、完成完整99D接入并参与D37训练 | OakInk2、HOT3D、独立UmeTrack、DexYCB；具体split／来源和转换依据见[四库协议](TWM_FOUR_SOURCE_RUN_2026-10-14.md) |
| 新增下载并完成原生结构检查 | HumanTouch固定一个episode及两份metadata，共3文件、6,379,522 bytes、1,597帧；官方文件hash核验和结构检查通过，**ready_for_training=false** |
| 有来源和访问核查，尚未下载训练包／完成本项目adapter | DexWild、Stanford DexCap、GRAB、ARCTIC、DexterHand、HandX等候选；ARCTIC仍waiting_credentials |
| 本轮完成视觉多库接入 | DexYCB＋HOT3D合并train73,524个手部状态／63,652张不同RGB／2,773窗口，val4,037状态；完整source／split／SHA与loader验收通过，06:50UTC已启动新联合run，更新与checkpoint状态见[执行记录](VISUAL_DATA_GEOMETRY_V3_2026-10-16.md) |

新glove来源仍按native测量质量与标定可审计性优先，不因MANO字段方便优先选择视觉拟合标签。完整99D输出保持不变；统一输出不意味着每个来源都有全部99D参数真值。

## 主要来源与本轮可执行性

| 来源 | 官方开放与规模口径 | 参考来源、任务角色 | 本项目接入条件与当前判断 |
|---|---|---|---|
| **HumanTouch** | [HF公开免申请](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch)，2026-10-16复核仍为`81822a10cac5def2638db546eaeced581d71a8b5`；[ModelScope许可](https://modelscope.cn/datasets/chuqiaoLyu/Xspark-HumanTouch/summary)为CC BY-NC4.0。官方首版约100h／10任务；此前十份meta按声明60Hz累计约132.75h，均非去重QC净量；计划1000h不能当实际发布量 | MANUS EMF手部骨架＋6DoF wrist/head跟踪，原始用途是触觉、手运动、视觉联合采集；数值pose/meta可独立于视频下载。**本轮新增glove优先验收对象** | 25点／方向／valid、world腕契约与相机标定具备接入条件；native→MANO映射、传感器rest frame、腕偏移与主体划分未验。一个episode结构通过不等于全部99D或视觉投影通过 |
| **DexWild raw human** | [官方HF](https://huggingface.co/datasets/boardd/dexwild-dataset)公开MIT。人类7,917episodes／2,924,588transitions；仓库约2.14TB包含人／机器人、RGB和分包，不是纯pose下载量 | MANUS原人体骨架＋腕tracker；论文用于将人类示范转为机器人策略。人体raw与LEAP action必须分开 | local手指是高价值glove候选；[官方TrackerProcessor](https://github.com/dexwild/dexwild/blob/main/dexwild_utils/dexwild_utils/TrackerProcessor.py)在当前路径中相对首帧归一化，读ZED pose但未显式组合world，并有插值平滑。发布样本与版本核清前不把`world_abs`名字当world腕GT；这不证明整个release都无world |
| **Stanford DexCap raw** | [官方HF](https://huggingface.co/datasets/chenwangj/DexCap-Data)公开CC BY4.0；30min wiping＋60min packaging。原始zip与processed hdf5合计约108GB | Rokoko手部骨架与T265追踪；论文训练机器人模仿策略。raw含胸部RGB/depth、相机／左右手6DoF和掌系人体手点 | 可做较小glove adapter和native几何补充。SLAM腕与局部手指分开质检；优先raw五指而非retarget到机器人后的action。tracker→解剖腕、时间与相机同步须验 |
| **GRAB** | [官方仓库／下载说明](https://github.com/otaheri/GRAB)，注册取得数据与模型许可；10主体、120Hz | 外部marker→人体／手模型拟合；HMP等无图像运动先验工作使用其连续手运动，符合纯motion任务 | 高优先完整运动候选，但没有RGB可供视觉联合。必须保留个性化`v_template`及原参考几何；不能默认所有β=0仍无损。当前没有已授权数据副本，不阻塞公开数据工作 |
| **ARCTIC** | [官方说明](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md)要求注册；raw world序列约215MB、meta91MB；cropped RGB约116GB，原图649GB。实际账号／许可条件与MANO分开 | Vicon→双手MANO／world轨迹，并有个体模板和拟合质量；支持运动参考与配对视觉联合 | 高优先条件候选；本项目仍waiting_credentials。可只取参数做motion，也可按官方split接RGB；不能拿聚合包绕过源访问要求。保留fit error、参考模板、camera轨迹及源split |
| **DexterCap / DexterHand** | [官方HF](https://huggingface.co/datasets/pku-mocca/DexterHand)约187MB、Apache2.0；[官方代码](https://github.com/PKU-MoCCA/dextercap)仍注明full dataset coming soon。**不是Stanford DexCap** | 手部多视角marker三角化→MANO拟合；公开包只含最终参数，没有raw RGB或marker中间轨迹 | 新的小规模marker候选，不能因字段像MANO抢在glove前加入。20Hz源版与60Hz平滑插值版同源，不能重复计量；需验side／subject、动作边界及root约定。无RGB，当前不用于视觉联合 |
| **HandX独立自采部分** | [官方仓库](https://github.com/handx-project/HandX)公开base archive及`source_metadata.json`；自采／第三方许可分别保存，MANO条款仍适用 | 原自采MoCap＋GigaHands／HOT3D／HoloAssist等聚合，主要用于双手运动生成；ARCTIC／H2O不在可再分发base包 | 只筛可追溯的独立自采，按source和原录制去重。HOT3D不能重复当新motion；canonicalization和重新拟合可能改变global腕／原始参数，必须审计。不能整包称为同精度MoCap |
| **TACO** | [官方主页](https://taco2024.github.io/)提供Dropbox V1；[官方SDK许可](https://github.com/leolyliu/TACO-Instructions)为CC BY4.0。约2.5K序列／14参与者、ego＋12exo | [全文§3](https://arxiv.org/html/2401.08399v2#S3)区分：marker用于物体，手由MMPose多视角三角化＋contact-aware MANO优化。论文任务包括动作识别、运动预测与抓取生成 | 可以作为视觉联合扩展候选；**不能把物体MoCap误写成独立手部marker GT**。先验ego投影、真实dt、有效帧与源序列split，再决定是否使用；不是本轮sensor优先motion池的替代品 |

没有统一独立真值基准支持上述来源的毫米级精度排序。glove、marker和视觉标签也都可能包含骨架求解、拟合或后处理；来源与转换误差分别报告。

## 论文和数据职责不能混用

- **运动先验／受损状态恢复**：GRAB／本项目既有MoCap来源与原人体glove更接近TWM数值任务。HMP等论文提供motion prior先例，但不能据其用库推出各库所有99D字段同精度。既有论文职责表见[TWM数据依据](TWM_DATA_PAPER_EVIDENCE.md)。
- **机器人示范学习**：DexWild、DexCap原始目的是机器人策略训练。我们需要原人体运动参考，不读取机器人action，也不增加机器人规划模块。
- **质量学习**：StableHand八库用于quality network的角色不等于八库都是高精度motion GT，更不等于本项目已下载。[原核查](TWM_PRETRAIN_PLAN.md#1-截图的数据用途先核对清楚)保留版本与GT来源。
- **视觉重建／视觉联合**：有图像还不够；必须有相机标定、时间同步、可用标签与训练划分。HumanTouch/DexCap需投影验收；TACO属于视觉拟合参考；GRAB与当前DexterHand公开包不提供配对RGB训练入口。

本轮优先顺序是：已下载且可审计的数据扩大有效训练覆盖，同时验收HumanTouch native链；通过一个来源就记录其可监督字段，不等所有候选完成，也不把未验来源伪装为完整参数GT。

## HumanTouch固定小样本的已验收范围

选样在查看几何数值前固定为`X001 / episode_000001`，source revision为`81822a10cac5def2638db546eaeced581d71a8b5`。实际下载的三个文件：

| 文件 | 官方固定地址 |
|---|---|
| 信息与feature schema | [X001/meta/info.json](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch/resolve/81822a10cac5def2638db546eaeced581d71a8b5/X001/meta/info.json) |
| episode契约、关节名字、标定 | [X001/meta/humantouch/episodes/episode_000001.json](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch/resolve/81822a10cac5def2638db546eaeced581d71a8b5/X001/meta/humantouch/episodes/episode_000001.json) |
| 原生数值标签 | [X001/data/chunk-000/episode_000001.parquet](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch/resolve/81822a10cac5def2638db546eaeced581d71a8b5/X001/data/chunk-000/episode_000001.parquet) |

metadata声明：1,597帧／26.6167秒、统一60Hz、quality accepted、米、xyzw、world。它们是源契约，不是我们已证明了全部坐标链的结论。可用字段包括：

```text
observation.session_time_ns
observation.human.pose.head / wrist_left / wrist_right        [7]
observation.human.pose.valid                                 [3]
observation.human.hand_skeleton.{left,right}.position         [25,3]
observation.human.hand_skeleton.{left,right}.orientation      [25,4]
observation.human.hand_skeleton.{left,right}.valid            [25]
```

episode JSON还包含joint_names、头相机内参与畸变、`camera_from_head_tracker`偏移。所选metadata没有足以成立主体隔离划分的明确subject identity；不能把不同任务／session当成不同人。

[只读验收脚本](../scripts/audit_humantouch_source.py)在远程完成检查：文件和schema、真实session时间连续性、有效性、native骨段长度和四元数范数、腕位移与速度、native root与tracker的坐标假设、输入SHA前后不变。远程本次结果为`structural_pass=true`，**`ready_for_training=false`**。这证明样本结构可解析，并未验证仪器绝对精度、MANO转换或相机投影。

本次1,597行的真实相邻dt为约0.016666666秒；source和双腕全有效，双手25点方向四元数范数均约1。native root坐标范数量级约左0.90米／右1.07米，不是接近零的局部根坐标；它与declared wrist tracker位置的距离均值分别为**左91.45mm／右89.10mm**，范围约89.52–92.79／88.10–90.04mm。这个约9cm差值是**tracker与native根之间尚待核实的标定／定义差**，不能当测量误差，也不能据此称native根已对齐MANO解剖腕。证据保存在远程`/sciclone/data10/jchen57/Hand_Reconstruction/evaluations/data_expansion_geometry_v3_20261016/humantouch_native_audit.json`。

此脚本不拟合MANO、不造99D参数标签、不改训练数据或启动模型。原始parquet、metadata与数据文件不上传GitHub；仅小型聚合验收结果适合提交。

## 下一步验收与入池门槛

1. **原生骨架与腕链**：在远程检查左右native skeleton、tracker位姿及source有效段，明确坐标系、四元数顺序、rest frame和tracker→解剖腕偏移；骨长稳定和低抖动只能筛异常，不能当独立精度证据。
2. **native几何监督映射**：保留25点及validity作为源参考，标定到解码MANO可对应点／方向。未测twist和缺失world字段保持无监督；若做IK，记录其派生来源与残差，不能升级为传感器直接GT。
3. **视觉资格分开验收**：只对有配对图像的来源，检查head tracker到相机变换、畸变、手点重投影及同步偏移。投影未过验前，不因存在RGB文件就纳入视觉联合。
4. **划分与去重先于切窗**：再选不同任务／session的小样本确认结构可复用，按可识别subject、原始session、原recording划分。主体未知则明确限制；同轨迹多view、20／60Hz版本、重渲染和聚合转存不算新运动量。TWM训练与视觉评测共用同一排除表，不能从另一格式泄漏held-out轨迹。
5. **完整参数与部分参考分开接入**：现行motion trainer要求17组完整参数参考，不能把HumanTouch的缺失rotation填零／标valid后直接混入。native部分监督需要独立、可检查的数据与几何loss支持；当前仅样本验收，不代表该训练接口已完成。

以上是数据接入的验收步骤，不是本轮修改raw/candidate构造、历史或回放方案。来源混合比例、全库下载规模和性能收益仍待有效数据量与受控验证决定，不以宣传小时数直接分配权重。

**HumanTouch最小后续入口（建议，未实现）：** 可先验非拇指PIP／DIP的同指单位骨段夹角，作为不依赖global腕平移／旋转的局部几何参考；需先核25点名字对应的父子关系与解剖位置。metadata没有给出MANUS rest transforms、传感器骨段轴到MANO的固定变换或tracker→解剖J0，不能直接把25个源四元数当15组MANO local rotation。直接监督local点前还要验掌frame与个体骨长／shape；约9cm差不直接减常数。保持完整99D预测，仅对已核局部几何开放mask监督，需要独立完成现motion trainer的partial接口，不能沿用17组全GT入口冒充接入成功。
