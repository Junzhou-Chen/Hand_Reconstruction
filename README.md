# Hand Reconstruction / HandLoop

从高质量手部轨迹学习运动先验，用于纠正头戴RGB的手部重建。当前路线是冻结HaWoR、独立训练纯轨迹TWM、通过真实图像证据迭代校正当前状态，并反馈最终状态与可靠性。

当前仓库处于研究设计、数据调研与论文草稿阶段；尚无模型实现或实验结果。

- [项目记忆入口](MEMORY.md)
- [完整方法框架](docs/METHOD_FRAMEWORK.md)
- [TWM首轮数据、训练与输出方案](docs/TWM_PRETRAIN_PLAN.md)
- [WestWorld与主流视觉手部重建论文精读](docs/WESTWORLD_AND_HAND_RECONSTRUCTION_REVIEW.md)
- [主流表示与MANO标注来源](memory/10_MANO_REPRESENTATION_AND_ANNOTATION.md)
- [现行原始方案](HandLoop_Project/START_HERE.md)
- [TWM数据集调研](memory/07_TWM_DATASET_SURVEY_2026-10-13.md)
- [产业数采团队与大规模手部数据补充](memory/09_INDUSTRY_HAND_DATASETS_2026-10-13.md)
- [开发、同步和远程实验约定](memory/08_DEVELOPMENT_AND_REMOTE.md)

## 工作方式

- GitHub：`https://github.com/Junzhou-Chen/Hand_Reconstruction.git`，主分支`main`。
- 本机：代码、论文、文档与轻量静态检查。
- 实验：`ssh hongfei`，项目路径`/data/hongfei/zhou/Hand_Reconstruction`。
- 数据集下载、预处理、模型推理、训练和评测全部在远程服务器执行。
- 数据优先无需申请、直接下载，并参考论文实际训练/评测用法；需审批资源不作为首轮开工前置。
- Git跟踪源码、配置、论文、记忆、下载说明及小型结果摘要。数据集、模型权重、缓存、原始日志和大文件按[.gitignore](.gitignore)排除。

数据文件和模型资产遵守各自许可，不随本仓库分发。仓库暂未指定统一开源许可证。
