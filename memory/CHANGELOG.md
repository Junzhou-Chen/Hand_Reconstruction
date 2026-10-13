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

## 后续记录要求

每次记录日期、实质变化、依据和相关产物。实验失败和决定撤回同样记录；不要把计划完成、代码完成和实验验证完成混为一个状态。
