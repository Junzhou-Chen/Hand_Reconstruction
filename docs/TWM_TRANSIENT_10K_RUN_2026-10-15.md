# D37：新Pod从头训练10,000步，历史checkpoint仅保留best

用户2026-10-15明确把D36两epoch预算改为10,000 step，从头重新训练，并要求“之前所有训练结果只保留best”。这替代D36的2epoch预算以及此前保留全部latest/暂停快照的策略；保留修复后的v3噪声、内部位置尺度0.05m、13.53M结构、完整99D/β另存、四库源GT/split及每卡512/global1024。无需把已完成checkpoint当resume起点。

## 已核实的前一轮

D36原任务06:52:11→07:53:55 UTC（纽约02:52→03:53）正常完成1,244更新/2epochs/1,273,220窗口使用；各源次数正好是其train窗口数×2。latest/best均为1244，model/AdamW finite，latest CPU hash与两rank相等，W&B58f012f8为finished。用户新Pod上能读共享存储，Torch2.10/cu130两卡tinyCUDA均140。历史两轮结果仅说明已训练完成；固定合成val ordinary腕4.652→4.217mm，但全关节burst4为5.042→5.237mm、position-only burst4为6.520→6.329mm，clean0→1.457mm（59.28%窗口腕退化>1mm），旋转多数略退化。尚未解决稳健短时恢复，更不能当真实HaWoR收益。小型[完成审计](../reports/twm_transient_epoch2_completed_20261015.json)保留这些失败；完整64clip对照因用户转为10k重训而暂缓，不能写成已完成。

## 新配置与采样

[10k配置](../config/twm_four_sources_moe_transient_v3_b512_10k.json)：`training_unit=step`，`steps=10000`，warmup500，LR2e-4，cosine终点10%LR，eval_every500、save_every100、log_every10；新W&B独立run。当前/未来目标及权重不改，不把更多步数保证成效果提升。

用户此前明确的full-coverage采样保留：共享shuffle、双rank disjoint、遍历完再打乱，每轮636,610窗/622更新、最后global706。训练预算和采样周期分开，不因步数模式暗中切回等源有放回采样。10,000更新对应16个完整遍历＋48个batch，合计10,234,912窗口使用；多次使用不是新独立样本。每源validation仍固定256窗、等源宏平均，14种v3/legacy模式与best选择权重同D36。step为主横轴，epoch_progress辅助。

## 新Pod与源码

用户指定`jchen57-fineturn-tl4zr`，namespace jchen57；旧debug Pod已不存在。新Job于10-15 05:24:05UTC开始计432,000秒，Pod实际容器15:51:02UTC才运行；到期按Job计算为10-20 05:24:05UTC，不从容器启动重算。新配置截止10-20 05:10UTC，留出保存余量；未改用户Job。12CPU/200Gi/2×RTX PRO6000 Blackwell约95GiB，数据/权重/runtime共享data10不重下载。

新独立源码目录`/sciclone/data10/jchen57/Hand_Reconstruction/code/twm_transient_v3_10k_20261015`，不修改旧25fee71源码及另一聊天原目录。新输出`runs/twm_four_sources_moe_transient_v3_b512_10k_20261015`。初版源码/配置已准备，待新双卡step-budget/full-coverage边界及resume检查和正式实际更新验收；此处不预写“已跑起”。

## 历史best-only范围

只处理本项目data10下`runs/`、`audits/`、`evaluations/`的历史训练checkpoint，排除新正式run、新smoke验收目录以及两份visual RGB inference预测`.pt`（它们是评测产物而非模型权重）。下载权重、MANO、原数据、源码、配置、日志和小型指标摘要保留。每目录优先canonical `best.pt`，没有canonical则保留明确best别名；两个无best的旧smoke目录经逐项审阅可删除其latest。先加载所有将保留best做CPU finite与SHA前后检查，dry-run完整manifest经审阅后再apply并逐文件记录。

清理新策略会让旧文档中的latest/暂停快照路径不再存在，历史数值与当时审计不改写；保留下来的best和本次清理清单成为后续可恢复来源。新run训练中保留latest用于中断恢复，用户本次“之前所有”不指向正在训练的新run。
