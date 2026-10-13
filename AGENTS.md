# 项目工作约定

本工作区用于 HandLoop / Are Gloves Necessary? 研究。默认用中文交流，保留必要的英文术语。

## 开始工作时

1. 先读根目录 [MEMORY.md](MEMORY.md)，再按任务读取其中链接的专题记忆。
2. 设计依据是 [START_HERE.md](HandLoop_Project/START_HERE.md) 与此后用户明确作出的决定。历史归档和旧图不能覆盖现行方案。
3. 需要实现、引用或报告结果时，检查实际文件、官方来源和实验记录。记忆中的计划、建议超参、候选贡献不代表已实现或已验证。

## 保持研究方向

- 主路线：头戴 RGB → HaWoR → 纯轨迹预训练的 TWM → 视觉证据引导的当前状态迭代校正 → 最终状态及可靠性跨帧反馈。
- TWM 学习当前受损状态恢复，短期预测仅作辅助；TWM 不接收图像或 RGB features。
- 训练借助 glove/MoCap，部署只需相机。WestWorld 仅作结构思想参考。
- 第一版范围是 wrist 6DoF 与手指关节。不要无依据切换视觉主干或增加机器人规划、物体物理、MoE、长短记忆等模块。
- 原始观测、迭代候选、已完成的历史状态必须分开；高 confidence 不是不可修改的真值。
- 用真实几何误差衡量效果，不能把平滑、合理外观、预期收益写成实验结果。

## 开发与实验位置

- GitHub为`https://github.com/Junzhou-Chen/Hand_Reconstruction.git`，`origin/main`；用户已授权正常commit/push，及时同步代码、论文、记忆及小型结果摘要。
- 本机只做代码、论文、文档及轻量静态检查；数据预处理、模型推理、训练和评测全部在远程执行。
- 初步实验通过`ssh hongfei`，项目在`/data/hongfei/zhou/Hand_Reconstruction`（用户指定根目录`/data/hongfei/zhou`）。
- 数据集、权重、大文件、缓存和原始日志不上传GitHub；维护`.gitignore`，提交前检查暂存清单。数据loader源码放在可跟踪源码目录，不与忽略的数据存储目录混用。
- 具体流程与连接状态见[memory/08_DEVELOPMENT_AND_REMOTE.md](memory/08_DEVELOPMENT_AND_REMOTE.md)。

## 更新记忆时

- 完成实质工作后，更新对应专题；进展写入 `memory/04_STATUS_AND_NEXT_STEPS.md`，决策写入 `memory/02_DECISIONS.md`，变更摘要写入 `memory/CHANGELOG.md`。
- 区分用户决定、现行设计、建议、未决问题、代码事实、实验结果；附日期、来源或产物路径。
- 新决定替代旧决定时保留替代关系；不要仅追加互相冲突的“最新方案”。用户的新指令优先。
- `HandLoop_Project/` 内的原始档案默认保留，避免整理记忆时改写历史。新源码事实或勘误写入记忆并链接来源。
- 实验记忆至少记录配置、代码版本、数据划分、GT 来源、坐标/单位、指标定义、产物位置；失败结果同样保存。
- 记忆条目和来源不一致时回查原文。开放问题应明确保留，不把实现者的推断写成用户已定选择。
