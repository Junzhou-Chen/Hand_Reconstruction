# HandLoop 项目记忆变更记录

## 2026-10-13 初始建库

- 按用户“仔细理解完整项目细节和 idea，构建项目记忆库”的要求，阅读全部分文档、核对总档案精确重复关系，并检查旧方法图。
- 新建根目录 `MEMORY.md`、`AGENTS.md`，以及方法、决策、数据评测、进展、写作引用、来源索引六份专题记忆和来源 manifest。
- 保留核心纠偏：当前状态恢复优先；高 confidence 可错；可信局部观测约束其他关节；纯轨迹 TWM 与真实图像证据分工；帧内校正和跨帧反馈。
- 区分用户明确决定、现行整合设计、建议超参、历史废弃方案及未实现事项；登记 TWM 输入契约、mask、坐标方向、初始化、质量目标等开放问题。
- 核验部分官方论文与源码；记录 HaWoR infiller 内含 camera-to-world 转换，以及 HOT3D/DexCap/EgoDex 的关键数据限制。
- 全部原始材料保持不变。本轮没有训练、实测性能、投稿结论或模型实现。
- 完成独立方法复核与外部来源复核；检查 39 个本地链接有效、6 个原文件哈希未变、来源 manifest 一致。

## 2026-10-13 TWM公开数据专项调研

- 按用户“仔细调研有哪些开源数据集可以pretrain TWM”的要求，核验官方论文、项目、代码、许可入口和部分公开release元数据；新增[专项报告](07_TWM_DATASET_SURVEY_2026-10-13.md)。
- 建议GRAB+ARCTIC建立纯轨迹基线，OakInk2扩任务，HOT3D按评测协议使用公开train GT，DexWild扩手套规模；HumanTouch已有文件，许可与完整发布量待核实。建议未升级为用户决策。
- 区分marker/glove、视觉拟合和设备追踪；分开finger/wrist来源、world坐标和有效性；纠正整库AMASS真实手指、多视角图片当时间帧、TACO手MoCap、HandX全量自采等误读。
- 审计HandX source mapping/test NPZ索引；核验OakInk2 annotation文件索引、HumanTouch X001/X002元数据；均不等于全库数值质量验收。
- 更新MEMORY入口、数据评测、进展、引用边界；未改原始档案、未实现loader、未训练，也未启动TB级数据下载。
- 交叉复核重点数据集表述，检查本地链接、关键时长换算与原始六个文件哈希；未发现原档改动或无效本地链接。

## 2026-10-13 GitHub与远程实验约定

- 用户明确GitHub仓库、main分支、忽略大文件，并指定本机写代码/论文、hongfei远程执行全部实验；登记为D13/D14。
- 初始化Git，新增`.gitignore`、README及开发/远程专题；忽略数据、权重、视频、缓存、原始日志和私有配置，保留论文源文件和小型结果。
- 成功核验GitHub初始无refs和SSH连接，远程根目录、git与nvidia-smi存在。最终同步结果见[开发与远程约定](08_DEVELOPMENT_AND_REMOTE.md)。
- 初始提交`20fe231`已推送main，hongfei已clone且与初始提交一致；创建忽略的data/checkpoints/outputs/logs。原档的Markdown换行空格保留，原始六文件哈希不变。

## 2026-10-13 产业数采覆盖补充

- 接受用户指出首轮偏学术、漏查产业团队的纠正；不再将首轮短名单视为完整公开资源全景。
- 新建[产业补充调查](09_INDUSTRY_HAND_DATASETS_2026-10-13.md)，覆盖SynData、OOJU、Daimon、OpenGraph、EgoVerse、Lightwheel、BeingBeyond，以及厂商商业样本/机器人数据的边界。
- 读取UniHand官方Human_Data.md及9个人手TAR索引（约68.43GB）；确认world wrist/MANO字段和来源聚合，不下载大文件。
- 更新入口、数据专题和后续优先级；保留来源质量分层，也列出可申请的大规模视觉数据。没有自动申请数据、联系公司或开展训练。

## 2026-10-13 完整方法框架与免申请数据原则

- 用户明确免申请公开数据优先、参考论文实际用法，登记D15；替代原先需注册数据默认开工顺序，保留调查事实及后备资源。
- 新建`docs/METHOD_FRAMEWORK.md`，连贯整理任务、单手69D、三主体/两闭环、统一TWM接口提案、残差几何、w/q与三阶段训练；明确当前无实现或实验结果。
- 补充预训练raw/candidate槽位、anchor残差、最终q取证、缺测/首帧与单手共享参数等建议契约，不升级为用户决定。
- 核验HMP/HaWoR/StableHand/PAD-Hand实际数据角色；同步更新入口、方法、决策、数据和下一步，原始研究档案保留。

## 2026-10-13 69D与HaWoR原生输出澄清

- 按用户提问核查HaWoR实际源码：原生为MANO root/15关节旋转、translation、shape；视频支持左右手，原infiller为218维双手输入。69D是本项目每手统一表示，不是HaWoR原生张量。
- 明确69D/66D是编码长度，非独立物理DoF；记录MANO translation与wrist joint区别和adapter转换公式。
- 核查DexCap采集代码21点与orientation，区分源码字段、release样本和尺度/坐标验收；补充公开数据的完整/部分69D监督能力表。
- 更新方法和记忆；无数据下载、模型推理或实验结果。

## 2026-10-13 主流表示与MANO标注核验

- 按用户请求核验代表性重建/运动方法，确认MANO参数路线常见，但内部特征与输出不同，HandX/METRO提供坐标路线；69D为本项目候选而非行业标准。
- 新建表示与标注专题，记录视觉/depth拟合、marker→SMPL-X/MoSh++、HOT3D UmeTrack→MANO及FreiHAND human-in-the-loop流程；区分拟合reference与直接测量。
- 补充MANO参数输出可通过joint/2D/depth损失训练，提出MANO输出+joint监督baseline与69D对照建议；最终编码未由用户定案。
- 核验PAD-Hand公开代码、StableHand54D、HO3D发布条款；保留HaWoR论文“独立手”和当前218拼接实现的不同证据。
- 更新方法、状态和入口，原始档案保留；没有模型运行、数据下载或实验指标。

## 2026-10-13 WestWorld与视觉手重建方法精读

- 按用户要求阅读WestWorld v2和代表性单帧、视频、运动先验/恢复方法；新增[精读报告](../docs/WESTWORLD_AND_HAND_RECONSTRUCTION_REVIEW.md)，含方法、原文定位、关键源码SHA和研究判断。
- 补入DIP纯运动受损恢复/序列内反馈，以及DIR/WiLoR/GeoHand候选取证前作，收窄首次贡献表述；新增JoHan等2026方法，区分预印本与正式发表。
- 确认WestWorld 256-bin配置、HaWoR非因果及missing-only补全、HMP局部prior与2D fallback、StableHand原论文/增强版差异、PAD-Hand公开demo范围。
- 核验HaWoR W/WA含尺度对齐、StableHand同名指标版本变化；更新引用、评测、进展与入口，不改变用户选定路线和表示，不改写原始档案。没有模型运行、权重/实验数据下载或性能复现。

## 2026-10-13 StableHand输出对照补充

- 按用户明确要求扩充StableHand对照，区分54D/手最终动态状态、flow velocity直接输出、独立beta与四路观测误差/quality。
- 核验固定公开实现中MANO translation不等于解码wrist xyz，登记rest-root偏移和实际字段顺序；保留论文v1与增强版残差表示的边界。
- 将StableHand初始观测质量与本项目最终状态质量区分；更新表示专题、论文报告和状态，无模型实现或性能复现，不将表示建议升级为用户决定。

## 2026-10-13 八库核验与预训练方案收敛

- 按用户截图核验HOT3D、ARCTIC、Re:InterHand、H2O、HOI4D、EgoDex、EgoVerse、Ego-Exo4D，新增[首轮预训练方案](../docs/TWM_PRETRAIN_PLAN.md)，重点数据选择、输入输出、corruption、混合监督与验收。
- 纠正HOT3D full VRS与公开Clips访问混用；Clips train可直下，当前建议HOT3D-Clips＋Re:InterHand先建基线，HOI4D/OakInk2与EgoDex后扩。帧条目累计不当去重净量，split先按原sequence/subject/capture。
- 明确StableHand八库用于质量网络、joint-only不拟合MANO；公开缓存为benchmark评测，不作训练全集。记录H2O62字段、HOI4D15Hz、EgoDex25点与EE4D官方2D/3D数量口径。
- 具体化99D MANO动态状态＋序列β、joint-only可微几何监督、当前恢复主任务＋短预测辅助；编码、混合、窗口和配比为未验证建议，未升级为用户决定。
- 更新入口、方法及专题记忆，保留其他会话的论文精读/StableHand补充和原始档案。无完整数据下载、loader、模型运行或结果。

## 2026-10-13 99D采纳与论文驱动的数据重规划

- 用户明确使用99D，登记D16，替代69D默认候选；D17登记以后plan广泛对照最新论文，写入AGENTS。
- 新增[12项论文数据依据](../docs/TWM_DATA_PAPER_EVIDENCE.md)，补DIP/TaMF/UniHand/HandX/HandFlow/ACE/EgoFound3R/JoHan等实际训练职责，不沿单篇QN八库筛选。
- 将[预训练计划](../docs/TWM_PRETRAIN_PLAN.md)更新v1：建议HOT可用train＋OakInk2核心，Re/Giga拟合扩展、EgoDex规模、DexYCB恢复基线/真实HaWoR数值适配、HOI4D先留出。数据混合仍未被用户定案，删除默认50:50最优配方暗示。
- 发现HOT官方train含论文另留出的公开GT benchmark原recordings；先全源排除、内划val再切窗。Oak沿431/39/157序列split但验PT/CT父录制和subject属性；HO3D仅称TWM未见，HOI4D只GT-valid评测。
- 方法与接口改为99D＋β，21点/mesh从MANO解码；51D切向Corrector残差仍为建议，骨长loss不默认必需，窗口/噪声由pilot验证。原档保留，未下载完整训练数据或运行模型。

## 2026-10-13 首轮选库建议进一步收敛

- 用户再次要求判断TWM预训练最适合哪些数据；明确第一个checkpoint只要求HOT/Oak核心，Re为优先姿态扩展、EgoDex为映射后规模扩展，不把所有候选当开工前置。
- 澄清DexYCB可以纯轨迹预训练；先单独DexYCB-only合成恢复对照与真实HaWoR错误适配，再判断是否混入核心。第三视角不是纯TWM的排除条件。
- 将“能支持第一轮验证”和“已覆盖完整运动分布”分开；来源增量需同预算验证，未新增用户采纳的混合决定，无数据下载或模型运行。

## 2026-10-13 传感器精度主导的v2纠偏

- 用户强调TWM无视觉、优先更准确手姿态、初始glove/MoCap方向有价值；登记D18，限定论文用库只是参考，替代v1 glove支线/Re-EgoDex优先扩展建议。
- 新建[传感器精度专题](11_SENSOR_FIRST_PRETRAIN_2026-10-13.md)，恢复GRAB/ARCTIC高优先MoCap、保留Oak/HOT，补公开UmeTrack自由手势/pinch；HumanTouch/DexWild升主要glove验收，DexCap/OpenGraph做小样本与部分监督。
- 通过官方ModelScope预加载dataset字段复核HumanTouch许可CC BY-NC4.0；固定HF SHA读取十份小型meta，累计28,674,858帧/60Hz约132.75h非QC净量，meta/pose索引episode计数不同须验收。旧“许可待核”已解决。
- 明确source-native姿态与99D模型状态分存、glove几何/部分方向监督、腕/手指质量分开；不因非MANO拒绝原始骨架，不伪造未测twist。GRAB personalized template、UmeTrack转MANO残差纳入验收。
- StableHand v1 QN数值输入但误差标签源于视觉prediction/reference，Enhanced有depth cues；借异构标签/故障任务，不将10M当clean运动池。更新计划、方法、入口及专题，无完整数据下载、模型运行或实验结果。

## 2026-10-13 完整99D输出契约确认

- 用户再次强调训练后的TWM应预测99D，补充D16：每手完整99D主状态输出不随源标签完整性改变，β仍另存。
- 在[预训练计划](../docs/TWM_PRETRAIN_PLAN.md#统一99d输出与标签完整性)明确完整MANO参考、glove几何与局部部分标签三种监督；离线IK拟合不等于新增测量，未知world腕/twist不可补造GT。
- 同步方法、入口、传感器专题和下一步；架构、adapter及loss实现仍未验收。仅文档更新，没有数据处理或训练。

## 2026-10-13 逐库完整适配边界澄清

- 用户追问所列数据是否全部完全适配；澄清这是适配事实核验，不是新的选库决定：完整99D输出或参与部分loss不等于每库单独满足完整腕＋手指任务。
- [预训练计划](../docs/TWM_PRETRAIN_PLAN.md#候选数据的完整适配判定)分开记录完整字段MoCap候选、高价值但需映射/腕链验收的glove、OpenGraph局部补充及其它未验扩查资源；不因非MANO降低glove科学优先级。
- 只读复核官方ARCTIC/Meta字段说明、HumanTouch episode契约与OpenGraph README。没有库通过本项目adapter或数值质量验收，没有数据预处理或模型运行。

## 2026-10-13 八库采集与标注来源精读

- 按用户要求新建[采集来源报告](../docs/DATASET_CAPTURE_AND_ANNOTATION.md)，逐一核硬件、原测量、扫描/标定、模型拟合、人工清理/QC与发布标签层；不改99D或glove优先原则。
- 补HumanTouch官方主页和DexTouch-WM论文MANUS/VIVE证据，保留HF与论文任务版本及逐episode配置未核；精确设备代际仍待验。
- 固定DexWild `5fa34af5115bb6f54320d0071c3b1d4864cd224a`审tracker/camera上游：当前路径未显式组合ZED world，world_abs名不证明世界真值；结论限该路径，不断言全部发布无world。DexCap固定采集代码核每手21节点/方向、四指LEAP与原人体分离，Rokoko同期资料EMF/IMU并存。
- 核OakInk2官方论文/补充的人工marker补齐与两阶段拟合；论文PDF仅为静态文献阅读，临时文本不提交。同步专题、入口与进展，无实际数据处理、模型实现或训练结果。

## 2026-10-13 精度比较的工程判断

- 用户问哪类更准确；补入[来源报告](../docs/DATASET_CAPTURE_AND_ANNOTATION.md#精度判断工程建议与实测排名分开)的分项判断：完整world腕＋手指参考更信校准外部MoCap，局部手指不一概判glove较差，HumanTouch/DexWild仍主要候选。
- 保留无同基准八库排名的边界；Meta native→MANO有直接精度方向证据，GRAB网络误差和DexCap终点漂移不冒充GT标签逐帧误差。状态为建议，未新增用户选库决定，无数据处理/评测。

## 2026-10-13 预训练数据执行归纳v2.1

- 按用户要求基于既有12项论文职责、八库采集与精度核验重新归纳：首轮目标建议OakInk2/HOT3D有效train/独立UmeTrack train/HumanTouch，glove从首轮共同验收。
- DexWild仍主要glove候选，可靠local字段可早接；已核world代码路径问题解决前不做world腕GT。GRAB/ARCTIC有授权优先加入，DexCap小补充，局部/视觉来源按各自角色后续验。
- 在[预训练计划§0](../docs/TWM_PRETRAIN_PLAN.md#0-当前推荐的预训练组合)区分目标池与首checkpoint依赖；不等待所有库，不预设比例，保持99D与部分监督、split/去重及远程实验约定。此为建议，未登记为用户采纳的混合。
- 更新入口、决策边界、数据/状态及传感器专题；无数据下载、转换实现或训练结果。

## 2026-10-13 ds现有GPU Pod操作学习

- 用户说明hongfei GPU忙，指定ssh ds及旧项目运维参考，随后明确授权使用昨天申请的jchen57-debug-mpd5p；登记D19，当前入口替代D14的hongfei，旧checkout保留备用。
- 新建[ds操作记录](../docs/DS_SERVER_OPERATIONS.md)，只读核包、远程运维/非敏感设置、Kubernetes context/用户任务、NFS及Job时限。附件与live配置Debug CPU/Eval GPU不同，模板不等于实时资源。
- 已进入用户Pod，12CPU/128Gi/2×约96GiB Blackwell；默认Torch2.6/cu126两卡tiny CUDA失败，zhou Torch2.12/cu130两卡8元素平方和140通过。记录失败与成功，非模型实验。
- 不提交原包、凭据、内部NFS地址或原始日志；未创建/删除Job、改旧项目、clone ds本项目、安装依赖或训练。更新当前计划中的远程位置，保留历史hongfei记录；home/data10项目路径仍为建议未创建。

## 后续记录要求

每次记录日期、实质变化、依据和相关产物。实验失败和决定撤回同样记录；不要把计划完成、代码完成和实验验证完成混为一个状态。
