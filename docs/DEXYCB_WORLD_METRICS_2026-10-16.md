# DexYCB 完整测试：W／WA／PA、加速度误差与 RTE 补算

记录日期：纽约 2026-10-16。用户要求统计截图中的五项指标。本轮复用已完成完整 s0 test 的固定官方 HaWoR、验证预选 best4200 和暂停保存 last4311 的预测；不是上一轮精选展示片段，也没有新神经推理或训练。

当前状态：**全量完成并通过独立核验**。UTC 2026-10-17 03:51:34 完成，远程补算耗时 101.717 秒；没有改变原冻结预测或 test 选择。

## 主结果

覆盖全部 **1,280 视角／78,768 有效手状态**，与原三模型完整测试完全一致。每项越小越好；best4200 是验证集预先选定，last4311 是暂停时保存权重。

| 指标 ↓ | 官方 HaWoR baseline | best4200 final | last4311 final | 有效计数 |
|---|---:|---:|---:|---:|
| W-MPJPE mm | 155.109 | 120.343 | 120.556 | 78,768 帧 |
| WA-MPJPE mm | 26.405 | 19.680 | 19.672 | 78,768 帧 |
| PA-MPJPE mm（Dex O(3)） | 5.867 | 5.161 | 5.167 | 78,768 帧 |
| 加速度误差 m/s² | 16.600 | 8.531 | 8.518 | 73,648 中心 |
| RTE % | 4.107 | 2.634 | 2.635 | 78,768 帧 |

best 相对 baseline 的误差降幅依次为 **22.41%、25.47%、12.02%、48.61%、35.86%**。这些是同覆盖的测量结果；当前完整模型改进不能全部单独归因于 TWM。

额外口径与边界敏感性：

| 诊断指标 | baseline | best4200 | last4311 | 计数 |
|---|---:|---:|---:|---:|
| proper SO(3) PA，mm | 5.866756 | 5.162860 | 5.168315 | 78,768 |
| 标准 T−2 加速度误差，m/s² | 16.989507 | 8.777088 | 8.764193 | 76,208 |
| 官方 T−4 加速度、排除 reset，m/s² | 16.341955 | 8.046236 | 8.012356 | 70,456 |

实际为 1,280 个长度 35–74 帧的连续段，均只形成一个不足 100 帧的对齐块，保留全部尾段；min16 子集与主范围完全相同。加速度主分母为 **78,768−4×1,280＝73,648**，其中 3,192 个中心跨原推理 reset；RTE 全部 1,280 段／78,768 帧有效，没有零路径或 <1 mm 非零路径。

W 只拟合开头两帧，其余帧不参与拟合，因而不保证全段误差小于未对齐误差。这里的 155.109 mm 不应与此前绝对 MPJPE 65.522 mm 混写；WA 的整段拟合也不等同于此前仅腕平移对齐。

## 指标定义与分母

主表沿用完整测试的全部有效状态，以每个固定相机为独立参考系。源 GT 是 DexYCB supplied 21 joints／fitted MANO，不是独立 MoCap。预测几何由各自保存的 99D 和自身 β 解码；参考 β 只用于恢复参考 native MANO transl。左右 shape convention 分别复用原评测的 prediction fix=True／reference fix=False。

关节集合仍为 Dex21。它与 HaWoR HOT3D wrapper 的 21 点指尖顶点索引不同，因此本轮是 **HaWoR 轨迹指标在固定相机 DexYCB 上的迁移统计**，不能与原论文 HOT3D 移动相机结果拼表。

- **W-MPJPE，mm：** 在每个连续有效段中按 100 帧分块，以块前两帧的全部关节点拟合 proper SO(3) 旋转、统一尺度和平移，再计算整块误差。不足两帧时使用实际可用帧。不是未经对齐的绝对位置误差。
- **WA-MPJPE，mm：** 每个同样的块用整块全部关节点拟合一次相似变换。不是此前表格中的逐帧腕平移对齐指标。
- **PA-MPJPE，mm：** 主表保留先前 DexYCB 标准的逐帧自由 O(3) 相似对齐，允许反射；另报与 HaWoR HOT 函数一致的 proper SO(3) PA，不能把两列混写。
- **加速度误差，m/s²：** 对绝对关节计算预测与 GT 的二阶差之差，取 21 点欧氏距离均值后乘 30²。按官方调用额外去掉二阶差结果首尾各一个中心，T 帧连续段贡献 max(T−4,0) 项；另存标准 T−2 中心诊断。时间来自 frame_id／名义 30 Hz，不是假称实测采集时间。
- **RTE，%：** 对完整连续段的 native MANO transl 轨迹作无尺度的刚体拟合，逐帧残差除以 GT 全段路径长度，再乘 100。native transl 由 `p3 − rest_wrist(own β)` 恢复，不直接拿 J0 代填；分母是累计路径而非起终点位移。GT 路径≤1e−12 m 时明确计为未定义，不加 epsilon 伪造有效值；非零但<1 mm 的短路径仅计数，不剔除。

以上 W／WA／proper PA／Accel／RTE 的函数和调用沿固定 [HaWoR eval_utils](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/lib/eval_utils/eval_utils.py)、[HOT evaluator](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/scripts/scripts_eval/eval_hawor_hot3d.py)。自由 O(3) PA 的既有来源与实现见 [完整测试报告](DEXYCB_FULL_TEST_RESULTS_2026-10-16.md)。

连续段按原 frame ID 差不为 1 时断开，禁止跨缺失 GT 的帧计算动态误差。主加速度保留相邻预测跨 32 帧 feedback reset 的跳变；另外排除中心位于段内 32k−1／32k 的三元组，作为边界敏感性诊断。不会因 reset 把主表误差删除。

W／WA／PA 按有效帧平均，加速度按有效差分中心平均，RTE 按已定义帧平均。100 帧块的尾部全部保留，包括单帧尾块。官方 HOT 还先删去不足 16 帧的连续段；为保持本项目完整测试覆盖，主表不作该删除，单独报告同样的 min16 子集。等块／等连续段权重另存诊断，不替换按帧主表。

## 数值核验与精度勘误

独立校验器 AST 提取固定官方源码的七个纯函数，核 W／WA／proper PA／Accel／RTE 的逐值结果及生产累计器的分子、分母。13 组合成案例与 24 段真实解码轨迹共 37 项对照通过，包含相似变换、刚体、反射、恒加速、静止 RTE、短段、尾块及缺帧／reset 边界。

初次 8 视角／568 状态预检因 PA 与旧报告差 0.000259278 mm 而被严格阈值拒绝。独立四配置探针确认原因：原完整测试显式启用 CUDA matmul TF32，补算初稿未启用。恢复 TF32 后大分组 PA 均值差降至 0.000001228 mm；按原连续段内 32 帧分组时，四项旧几何指标重现至约 1e−15 mm。初次失败与四配置结果均保留，没有静默放宽阈值。

正式全量计算显式启用原 TF32 设置，并按原 32 帧几何分组重放。`--decode-batch-size 512` 是分组内上限，实际分组仍≤32；这不是新的 RGB 推理 batch。补算后还重新核对全部三模型既有绝对／腕对齐／自由 PA／proper PA 均值，PA 容差 1e−5 mm、其余 1e−4 mm。

全量 12 项旧指标交叉核对全部通过，最大绝对差为 5.761e−9 mm；PA 两类最大差 3.553e−15 mm。独立复核 1,280 个唯一视角键、coverage SHA、脚本 SHA、各项 sum/count 及左右手回加均通过。官方参考函数保持 float32，生产累计使用更高精度；37 案例逐值比较的最大差及明确绝对／相对容差保存在小型报告，不要求不同浮点路径逐 bit 相同。

## 产物与复现

远程根：`/sciclone/data10/jchen57/Hand_Reconstruction/evaluations/dexycb_world_metrics_20261016/`。本机原始小报告位于忽略目录 `outputs/evaluations/dexycb_world_metrics_20261016/`；Git 只保存 [精选汇总与身份／验收](../reports/dexycb_world_metrics_20261016.json)。

远程 `full.json` 保存全部范围、左右手、权重、分母、模型与源码／资产 SHA；`full.coverage_audit.json` 保存逐视角的源／预测 SHA 和 frame-ID 身份。`self_test.json`、`reference_synthetic_checks.json`、`smoke_tf32.json`、`reference_real_checks.json`、`precision_probe.json` 及真实小型 fixture 保留独立核验与失败修正来源。许可 MANO、逐帧预测和 fixture 不入 Git。

实现：[完整补算](../scripts/evaluate_saved_world_metrics.py)、[官方函数独立对照](../scripts/check_saved_world_metrics.py)、[TF32／分组精度探针](../scripts/probe_world_metric_precision.py)。执行 snapshot 位于 `code/dexycb_saved_world_metrics_20261016/`，几何解码复用原 `code/dexycb_full_test_20261016/{src,scripts}`，已验 Torch2.10/cu130 环境、CPU 线程 2、I/O workers 6，在现有 `jchen57-fineturn-tl4zr` Pod 运行。
