# DexYCB完整s0测试：baseline／best／last

日期：2026-10-16。用户要求安全暂停当前Dex＋HOT联合训练，随后启动三套权重的完整DexYCB测试，后台运行并实时同步W&B；不等测试全部完成才结束聊天。

## 训练暂停与模型身份

`runs/visual_feedback_dex_hot_geometry_5k_20261016`通过原`STOP`协议在 **4311步**正常停止（22:24:49UTC），保留optimizer／scheduler／两rank RNG、原日志和全部既有best。没有删除Job或杀死下载／预处理任务，也没有改训练配置。

- baseline：原始官方HaWoR预训练权重，直接输出raw，未经本轮联合微调。
- best：训练固定验证geometry score选择的 **step4200**完整联合模型。
- last：用户要求暂停时保存的 **step4311**完整联合模型。

best／last以硬链接保存到独立评测目录`models/`，后续训练若原子替换latest也不会改变测试权重。CPU审计模型／Adam浮点tensor有限性、saved-rank模型hash一致性及文件SHA后，写入`frozen_models.json`；不根据测试结果重选权重。

## 完整覆盖与输入协议

官方s0 test：subject03–10，每人sorted100序列中`index%5==4`的20个物理录制，**160录制×8个RGB相机＝1280条视角轨迹**。没有每人2段限制，没有raw frame_id<128前缀，也没有选有效帧补足长度。元数据选择在解析test labels前固定；selection指纹为`9cb1ae439a1c32c8b5ff6f3fb03bcb290bde4580e207554120b94d815cd15d82`。

官方HPE仅排除`np.all(joint_3d==-1)`的无效参考。其余参考必须有完整MANO状态、有限21点、可用RGB和有效GT crop，否则fail并记录原因，不静默降低测试分母。每个视角保存raw帧、有效帧、GT无效帧计数；逐个提供MANO FK≤1mm一致性审计，这不是独立MoCap精度保证。

三模型共享完全相同的RGB、参考投影GT crop、真实K、mean(fx,fy)、BF16与原16帧双向HaWoR块。32帧clip独立reset反馈历史，缺帧处分段，尾部padding不计指标或历史；保持训练实现，不在测试时改变成长录制连续历史。每个静态相机自身坐标作为identity world gauge；没有depth、自动检测或SLAM输入，不建立移动头戴相机的world精度结论。

录制身份同时存unique `subject/sequence/serial`与`physical_recording=subject/sequence`。按物理录制检查与当前视觉train／selection及原motion train的隔离；s0主体可重叠，不宣称跨主体泛化。

## 指标与在线汇总

报告wrist位置mm、root／15finger SO(3)角度、绝对／腕平移对齐／PA MPJPE21、绝对／腕对齐MPVPE778，以及绝对／腕对齐／PA的PCK-AUC。AUC沿作者所用FreiHAND EvalUtil，0–50mm、100个含端点阈值、逐joint `error<=threshold`、梯形积分归一化；跨clip合并真实hits／counts，不能直接平均clip AUC。

完整test主PA沿FreiHAND的自由O(3)相似变换，允许反射，并记录反射次数及proper-rotation PA作为诊断。此前1015小集使用proper rotation且baseline未出现反射，不能默认这一条件对所有新预测成立。所有分支均用自身预测β；参考使用源β和Dex无left-shape-fix解码，左手中指tip445／右444，GTβ不输入模型。

按全体、左右手、subject、camera分组汇总；raw／candidate／final诊断分支分别命名，正式比较是baseline.raw／best.final／last.final。此轮尚未增加论文的物体遮挡比例分组；完整指全s0 test轨迹和帧覆盖，不冒称逐项复现论文未公开的输入／选帧流程。

W&B项目为`black-ant/handloop-twm`，正式三run共用group **`dexycb-s0-full-20261016`**；独立真实RGB smoke使用单独group，不混作完整test。每完成一个视角原子保存数值预测／指标和累计状态，并记录：

- `progress/completed_shards`、`completed_frames`、`completed_clips`、prepared_shards。
- `test/raw/*`、`test/final/*`的累计指标和AUC百分制，分母／缺失预测计数。
- 失败状态显式记录，缺失有效预测不会作为成功零误差，也不静默跳过。

每视角提交一次结果；重启只重算未提交视角，completed集合与指标状态原子保存，避免重复累计。恢复重新核对源manifest SHA、帧数与model／code identity。最终不仅三模型互相比较完整逐帧键／hash，也核对全部1280个预定视角及全部manifest有效帧；完成后controller生成`comparison.json`。

## 代码与持久入口

所有数据处理、MANO、checkpoint检查和GPU推理在`jchen57-fineturn-tl4zr` Pod；本机仅代码、文档和小型摘要。

持久根`/sciclone/data10/jchen57/Hand_Reconstruction`：

- 代码：`code/dexycb_full_test_20261016`，以Git基底3b0574e＋本轮新增脚本SHA记录。
- 数据：`prepared/visual_dexycb_s0_full_test_20261016`，`shard_index.json`持续发布就绪视角；预处理独立后台PID与日志留在评测目录。
- 评测：`evaluations/dexycb_s0_full_three_models_20261016`，`controller_status.json`、三模型独立`status.json`／`running_metrics.json`／最终`report.json`／逐视角数值预测和W&B身份。

新源码：[完整源准备](../scripts/prepare_dexycb_full_test.py)、[流式指标](../scripts/dexycb_test_metrics.py)、[持久单模型worker](../scripts/evaluate_dexycb_full.py)、[三模型controller](../scripts/run_dexycb_full_test.py)、[独立数值验收](../scripts/check_dexycb_full_test_smoke.py)。数据、图片、权重、NPZ及原日志继续受`.gitignore`忽略，只留远程。

启动验收包括独立SciPy／FreiHAND PA与PCK公式、反射诊断、原官方EvalUtil、MANO自身β／左右手、masked NaN、恢复／合并分母，以及三模型真实RGB同输入推理。首次checker失败来自FP32工具生成的“精确旋转”fixture与NumPy2移除trapz；只修测试fixture精度及checker局部兼容别名，不放宽容差，不改正式指标。失败报告保留。

实际启动PID、W&B URL、CPU checkpoint／数值与真实RGB验收结果以后续小型启动报告为准；本页协议本身不代表已经得到完整测试结果。

## 已实际启动并在线验收

2026-10-16 **22:40UTC／纽约18:40**快照：完整测试controller PID5276为`testing`，三个worker命令身份核对均存活，W&B API查得三个正式run均`running`，服务端已有真实测试进度及MPJPE／PA／三种AUC标量。不是仅提交任务或初始化成功。

| 正式模型 | 本地已完成视角／有效状态 | W&B |
|---|---:|---|
| 官方baseline | 31／2091 | [85qboyf8](https://wandb.ai/black-ant/handloop-twm/runs/85qboyf8) |
| best4200 | 11／790 | [wb42qkka](https://wandb.ai/black-ant/handloop-twm/runs/wb42qkka) |
| last4311 | 11／790 | [f2fzfdu6](https://wandb.ai/black-ant/handloop-twm/runs/f2fzfdu6) |

W&B服务端在同一查询期间较本地略滞后，分别已收到24／8／8个视角的指标；三个分母不同，**不能把这次启动快照的累计均值当作配对模型优劣或最终结果**。结束后仍严格核对全1280视角及逐帧键集合。

完整源准备后台PID5200，快照1042／1280视角、64425有效状态，仍`preparing`且error为空；剩余视角继续在原任务中准备，已发布的immutable视角同时用于测试。当前计数不是完整test最终帧数，不以它替代1280视角目标。

best／last CPU模型与optimizer有限性及saved-rank hash一致性均通过。独立数值checker v3共14项通过，包含真实左右手参考；原两份失败报告保留。三模型真实RGB smoke分别完成相同2视角／6clips／142状态，逐视角manifest SHA、frame-ID hash与分母完全一致，保存`real_rgb_smoke_acceptance.json`；正式测试使用另外三个run，不把smoke数据累计两遍。

[小型启动报告](../reports/dexycb_full_test_launch_20261016.json)保存冻结权重SHA／step、代码基底3b0574e＋overlay488769b及脚本SHA、全部验收、源准备与进程／W&B服务端证据。三模型进程与controller均由Pod内脱离SSH的持久进程运行，结束当前聊天不会停止测试。未改训练权重或按测试结果选best；没有承诺自动追加新实验。
