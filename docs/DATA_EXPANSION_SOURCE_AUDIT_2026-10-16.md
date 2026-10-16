# 数据扩展来源核查：运动预训练与视觉联合

日期：2026-10-16。性质：官方资料／源码复核、来源选择建议与一个原生glove样本的远程结构验收；不是新增训练结果。

## 本轮范围与实际状态

用户本轮授权推进此前建议的前两项：扩大运动预训练和视觉联合的数据，以及增加直接几何监督。此前第3项raw/candidate训练条件、第4项历史与motion回放、第5项增加训练步数暂不调整。此来源核查不改模型、受损样例构造、历史、任务比例、学习率或训练预算，也不自动把所有候选加入训练。

此前确实找过超过四个数据集，但“查到资源”“已经下载”“完成adapter验收”“已参与训练”是不同状态。

| 状态 | 截至本次核查的事实 |
|---|---|
| 已下载、完成完整99D接入并参与D37训练 | OakInk2、HOT3D、独立UmeTrack、DexYCB；具体split／来源和转换依据见[四库协议](TWM_FOUR_SOURCE_RUN_2026-10-14.md) |
| 新增下载并完成原生结构检查 | HumanTouch固定一个episode及两份metadata，共3文件、6,379,522 bytes、1,597帧；官方文件hash核验和结构检查通过，**ready_for_training=false** |
| 有来源和访问核查，尚未下载训练包／完成本项目adapter | DexWild、Stanford DexCap、GRAB、ARCTIC、DexterHand、HandX等候选；ARCTIC仍waiting_credentials |
| 后续视觉多库准备 | DexYCB扩大范围与HOT3D实际准备进度由执行记录另记；本文不把资料可获得性写成已完成视觉训练接入 |

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
