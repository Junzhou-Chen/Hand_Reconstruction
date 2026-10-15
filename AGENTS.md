# 项目工作约定

本工作区用于 HandLoop / Are Gloves Necessary? 研究。默认用中文交流，保留必要的英文术语。

## 开始工作时

1. 先读根目录 [MEMORY.md](MEMORY.md)，再按任务读取其中链接的专题记忆。
2. 设计依据是 [START_HERE.md](HandLoop_Project/START_HERE.md) 与此后用户明确作出的决定。历史归档和旧图不能覆盖现行方案。
3. 需要实现、引用或报告结果时，检查实际文件、官方来源和实验记录。记忆中的计划、建议超参、候选贡献不代表已实现或已验证。

## 保持研究方向

- 主路线：头戴 RGB → HaWoR → 纯轨迹预训练的 TWM → 视觉证据引导的当前状态迭代校正 → 最终状态及可靠性跨帧反馈。
- TWM 学习当前受损状态恢复；D29用户新增要求同时预测较长未来轨迹分布（本轮约2秒/60步）。TWM 不接收图像或 RGB features。
- 训练借助 glove/MoCap，部署只需相机。WestWorld 仅作结构思想参考。
- TWM选库优先手部姿态参考精度与可审计标定；glove/MoCap是主候选，不要求ego/RGB配对或原生MANO。99D统一输出可以用raw joints/可用方向做部分监督，不能因格式方便优先视觉伪标签。
- 第一版范围是 wrist 6DoF 与手指关节。D29用户已明确授权MoE、较长未来预测与输出不确定性；不要无依据切换视觉主干或增加机器人规划、物体物理、长短记忆等模块。
- 用户已选每手99D MANO动态状态：wrist p3＋root rotation6D6＋15个finger rotation6D90；shape beta10另存。69D只保留历史/可选对照，不再当现行接口。
- 原始观测、迭代候选、已完成的历史状态必须分开；高 confidence 不是不可修改的真值。
- 用真实几何误差衡量效果，不能把平滑、合理外观、预期收益写成实验结果。
- 数据优先无需申请、可直接下载的公开资源，参考论文实际训练/评测角色；需人工审批或联系作者的数据不作为开工前置。完整方法见[docs/METHOD_FRAMEWORK.md](docs/METHOD_FRAMEWORK.md)，其中新增实现契约仍标为建议。

## 研究规划依据

- 制定或实质修改plan前，广泛对照当前相关论文及最新公开方法，先读官方全文/训练协议，必要时核查源码；不要只围绕最近截图或单篇论文设计。
- 论文用于参考机制与协议，不按视觉论文用库频率决定pure-motion训练池。StableHand八库服务quality network，不能直接当高精度运动GT清单。
- 按任务接近程度比较motion prior、current restoration、visual reconstruction、quality learning，分别记录训练/评测数据、GT来源、输入上下文、split和可迁移部分。
- 回顾既有方案，明确哪些保留、哪些替代及其证据；论文先例、工程建议、用户决定和实验验证分开。超参或混合比例没有依据时写成待验证假设。

## 开发与实验位置

- GitHub为`https://github.com/Junzhou-Chen/Hand_Reconstruction.git`，`origin/main`；用户已授权正常commit/push，及时同步代码、论文、记忆及小型结果摘要。
- 本机只做代码、论文、文档及轻量静态检查；数据预处理、模型推理、训练和评测全部在远程执行。
- 当前计算优先`ssh ds`进入SciClone Kubernetes；用户已授权复用`jchen57-debug-mpd5p`，在Pod内运行实验，先显式使用已验CUDA的zhou环境。hongfei原checkout`/data/hongfei/zhou/Hand_Reconstruction`保留备用。不要为套用旧模板删除用户现有Job；实际入口/时限/存储见[服务器操作](docs/DS_SERVER_OPERATIONS.md)。
- 数据集、权重、大文件、缓存和原始日志不上传GitHub；维护`.gitignore`，提交前检查暂存清单。数据loader源码放在可跟踪源码目录，不与忽略的数据存储目录混用。
- 具体流程与连接状态见[memory/08_DEVELOPMENT_AND_REMOTE.md](memory/08_DEVELOPMENT_AND_REMOTE.md)。

## 更新记忆时

- 完成实质工作后，更新对应专题；进展写入 `memory/04_STATUS_AND_NEXT_STEPS.md`，决策写入 `memory/02_DECISIONS.md`，变更摘要写入 `memory/CHANGELOG.md`。
- 区分用户决定、现行设计、建议、未决问题、代码事实、实验结果；附日期、来源或产物路径。
- 新决定替代旧决定时保留替代关系；不要仅追加互相冲突的“最新方案”。用户的新指令优先。
- `HandLoop_Project/` 内的原始档案默认保留，避免整理记忆时改写历史。新源码事实或勘误写入记忆并链接来源。
- 实验记忆至少记录配置、代码版本、数据划分、GT 来源、坐标/单位、指标定义、产物位置；失败结果同样保存。
- 记忆条目和来源不一致时回查原文。开放问题应明确保留，不把实现者的推断写成用户已定选择。
