# HandLoop 记忆来源索引

整理日期：2026-10-13。索引用于快速回查主张的出处，区分原始设计、历史讨论、本次实现审计和外部事实。

## 本地原始材料覆盖

| 文件 | 规模 | 性质与读取结果 |
|---|---|---|
| [START_HERE.md](../HandLoop_Project/START_HERE.md) | 243 行 | 全文阅读；现行设计规范，覆盖历史冲突 |
| [01_Conversation_Research_Archive.md](../HandLoop_Project/01_Conversation_Research_Archive.md) | 1,254 行 | 全文阅读；内容整理而非官方逐字 transcript；保留用户片段和助手方案的来源差异 |
| [02_Introduction_CN_EN.md](../HandLoop_Project/02_Introduction_CN_EN.md) | 81 行 | 中英文全文阅读；首段有用户认可，整篇仍是草稿 |
| [03_References_And_Implementation.md](../HandLoop_Project/03_References_And_Implementation.md) | 179 行 | 全文阅读；参考资料、示例 schema、拟议目录、sanity checks 与空结果模板 |
| [04_Method_Overview.png](../HandLoop_Project/04_Method_Overview.png) | PNG | 已视觉检查；旧 prior predictor、旧视觉状态记号及预测导向训练示意，不代表最新完整条件恢复设计 |
| [00_All_In_One_Research_Archive.md](../HandLoop_Project/00_All_In_One_Research_Archive.md) | 1,795 行 | 四份分文档的精确打包副本，外加导言与分隔标题；不构成独立的第二份证据 |

原文件 SHA-256、字节数和打包区间保存在 [source_manifest.json](source_manifest.json)。这是本次读取快照；后续文件发生变化需重新读受影响部分并更新快照。

## 全量打包的重复关系

通过完整字符串精确匹配核验，不只对比标题：

| 分文档全文 | 在 00 中的起始行 | 在 00 中的结束行 |
|---|---:|---:|
| START_HERE | 17 | 259 |
| 01 对话归档 | 267 | 1520 |
| 02 引言 | 1528 | 1608 |
| 03 参考与实现 | 1616 | 1794 |

除上述全文，00 仅添加了说明、分节标题与空行，没有隐藏的另一版方法或额外实验结果。无需每次同时加载 00 和全部分文档。

## 问题到原文的映射

下列行号对应本次 manifest 中的原文件；将来原文变更后优先用章节定位。

| 问题 | 文件与章节 | 起始行 |
|---|---|---:|
| 项目目前是否已实现 | START_HERE 开头状态 | 3 |
| 已定决策与旧版排除 | START_HERE §1 | 16 |
| 当前任务和科学边界 | START_HERE §2 | 33 |
| 69D/66D 与固定参考系 | START_HERE 状态表示 | 47 |
| HaWoR 输出与原始 mask/证据 | START_HERE §3.1 | 66 |
| TWM 任务和建议网络 | START_HERE §3.2 | 85 |
| 帧内校正 | START_HERE §3.3 | 116 |
| `w` 与 `q` 的分别定义 | START_HERE §3.4 | 130 |
| 最终状态一次回灌 | START_HERE §3.5 | 138 |
| 三阶段训练 | START_HERE §4 | 147 |
| 数据、指标、消融 | START_HERE §5 | 161 |
| 创新性和科学风险 | START_HERE §6 | 188 |
| 开工顺序 | START_HERE §7 | 199 |
| 用户为什么要求收缩范围 | 01 §4.1 | 183 |
| 为什么不是远期规划 | 01 §3 | 153 |
| 用户选择 HaWoR/头戴相机 | 01 §15.8，归档中的用户表态 | 915 |
| TWM 纯轨迹且不用 WestWorld 模型 | 01 §15.10，归档中的用户表态 | 942 |
| 最重要的 current-state 纠偏 | 01 §15.11，归档中的用户表态 | 956 |
| 标题风格偏好 | 01 §15.12–15.15 | 975 |
| 用户引言首段纠偏 | 01 §15.18 | 1042 |
| 最新完整方法整合 | 01 §15.23 | 1110 |
| 唯一现行方案与第一实验问题 | 01 §16 | 1231 |
| 示例误差和 GapClosed 定义 | 01 §11.4 | 675 |
| 认可的中英文引言首段 | 02 §A、§B | 9 |
| 引言写作边界 | 02 §E | 75 |
| HaWoR/WestWorld 源码入口 | 03 §3 | 32 |
| 尚未实现的 schema | 03 §4 | 49 |
| 尚未实现的目录建议 | 03 §5 | 101 |
| 首轮 sanity checks | 03 §6 | 144 |

## 本次新增的整理和审计

- `MEMORY.md`：快速上下文与按现行方案重建的 Mermaid 概念图。
- `01_METHOD.md`：保持原设计语义，补充状态、坐标和生命周期的可检索表达；没有宣称实现。
- `02_DECISIONS.md`：标记用户直接决定、助手整合规范、旧路线及建议的区别。
- `03_DATA_AND_EVALUATION.md`：整理数据/评测职责；实验记录规范是本次组织建议。
- `04_STATUS_AND_NEXT_STEPS.md`：以实际文件盘点为状态依据；O01–O14是初次文档审计待决事项，O15来自后续数据调研，阶段验收细化属于建议。
- `05_WRITING_AND_REFERENCES.md`：外部论文和代码的只读核验记录。HaWoR world conversion 与 infiller 耦合、HOT3D Aria RGB/TEST GT、数据来源限制是值得保留的新补充。
- [07_TWM_DATASET_SURVEY_2026-10-13.md](07_TWM_DATASET_SURVEY_2026-10-13.md)：后续用户请求的TWM公开数据专项调查，含官方论文/发布链接、字段/许可、部分文件索引与元数据检查；推荐组合不是用户决策。
- [08_DEVELOPMENT_AND_REMOTE.md](08_DEVELOPMENT_AND_REMOTE.md)：用户明确的GitHub同步、忽略大文件、本地编辑/远程实验约定及实际连接/clone记录。
- [09_INDUSTRY_HAND_DATASETS_2026-10-13.md](09_INDUSTRY_HAND_DATASETS_2026-10-13.md)：用户纠正首轮覆盖不足后的产业补查；含SynData、Lightwheel、EgoVerse等实际入口和访问限制。
- [docs/METHOD_FRAMEWORK.md](../docs/METHOD_FRAMEWORK.md)：随后用户要求的完整方法梳理；区别已定主线与接口提案，纳入D15免申请数据偏好及论文实际数据角色。
- [10_MANO_REPRESENTATION_AND_ANNOTATION.md](10_MANO_REPRESENTATION_AND_ANNOTATION.md)：用户要求的主流表示与MANO标注来源核验；含参数/坐标路线、HOT3D转换误差及MANO输出可用joint监督的证据。
- [docs/TWM_PRETRAIN_PLAN.md](../docs/TWM_PRETRAIN_PLAN.md)：2026-10-13针对用户截图的StableHand八库核验与首轮数据/训练/输出提案；纠正公开HOT3D-Clips访问、quality训练角色和帧数口径，99D等仍为建议。

已按后续用户要求创建Git历史、推送指定GitHub并在hongfei创建checkout；尚未训练、完整数据下载、安装实验依赖或改写原论文草稿。专项调查读取部分公开metadata和文件目录，不代表加载完整轨迹。新记忆不能替代相应代码/数据/论文证据。

## 维护来源的方式

- [WestWorld与视觉手重建精读](../docs/WESTWORLD_AND_HAND_RECONSTRUCTION_REVIEW.md)：2026-10-13按用户要求精读WestWorld与主流方法，含DIP直接前作、候选视觉查询谱系、HaWoR/HMP/StableHand/PAD-Hand固定SHA与世界指标口径。为文献及静态源码事实，不是实验复现或新用户选型。

对新用户决定保留原意、日期及影响；对代码事实记录 commit 与符号；对论文记录准确版本和 URL；对实验记录可访问的产物路径。外部网页随时间变化，标记核验日期，复现前再固定版本。

记忆错误可直接更正并在 CHANGELOG 说明；不要为了保持旧记忆而维护已被原始证据推翻的主张。
