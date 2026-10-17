# DexYCB完整测试最终结果：官方baseline、best4200、last4311

记录日期：纽约2026-10-16；查询时间UTC2026-10-17 00:57。以下来自已完成的冻结权重测试，不是此前test1015试验或训练fixed8小验证。

## 完成与覆盖验收

三个worker均complete，W&B服务端均finished、progress/complete=1。三者各覆盖 **160个物理录制×8相机＝1280个视角，2992个按32帧切分的独立clip（尾部padding不计指标），78768个有效手状态／1654128个joint观测**。source原始帧93264，按官方HPE规则排除joint_3d全-1的14496帧；未缺失任何应评测预测。左右手39160／39608状态。

controller于 **2026-10-17 00:28:05UTC／纽约10月16日20:28:05**完成。baseline于23:11:10UTC完成，best于次日00:26:31UTC完成，last于00:27:59UTC完成。独立重读1280个预定键、每视角frame count／source manifest SHA，以及三模型完整frame-ID hash映射，完全一致；comparison.json也标记exact_coverage_matches=true。W&B各分支所有标量与最终本地report最大差仅浮点JSON舍入量级（<1e-10）。

主比较：baseline.raw为原官方HaWoR；best.final为验证geometry score预先选择的4200步；last.final为用户暂停保存4311步。未根据test重选best。GT为DexYCB fitted MANO和supplied joints，不是独立MoCap。FK参考对supplied joints平均0.009400mm、最大0.221945mm，低于预设1mm容差；这仅验证解码一致性。

## 之前使用的指标

| 指标 | 官方baseline | best4200 final | last4311 final |
|---|---:|---:|---:|
| 腕位置误差 mm ↓ | 68.051 | 30.509 | 30.160 |
| 腕／根旋转误差 ° ↓ | 11.661 | 9.167 | 9.205 |
| 15个手指局部旋转误差 ° ↓ | 10.143 | 10.203 | 10.228 |
| 绝对 MPJPE21 mm ↓ | 65.522 | 30.558 | 30.365 |
| 仅腕平移对齐 MPJPE21 mm ↓ | 16.499 | 12.223 | 12.298 |
| 绝对 MPVPE778 mm ↓ | 65.631 | 30.537 | 30.340 |
| 仅腕平移对齐 MPVPE778 mm ↓ | 15.873 | 11.833 | 11.904 |
| 绝对 PCK-AUC % ↑ | 19.997 | 48.883 | 49.170 |
| 仅腕平移对齐 PCK-AUC % ↑ | 69.152 | 77.235 | 77.093 |

腕位置是99D的p3误差，角度是SO(3)测地角；finger对每状态15个局部旋转等权聚合。绝对MPJPE含21个关节点，MPVPE含778个顶点。腕对齐仅去掉每帧腕平移，不对齐旋转／尺度，不是HaWoR的WA-MPJPE。各分支用自身预测β解码，源β只用于参考；不将GT shape输入模型。MPVPE参考是拟合表面，不是独立扫描。

## HaWoR论文在DexYCB上使用的指标

[官方全文§4与Table1](https://arxiv.org/html/2501.02973v1)的MPJPE表头实际指 **PA-MPJPE**，AUC对应PA之后的PCK曲线面积；不可与本报告的绝对MPJPE／绝对AUC混用。

| 模型／来源 | PA-MPJPE mm ↓ | PA-AUC % ↑ |
|---|---:|---:|
| HaWoR论文Table1 All（仅供参考） | 4.760 | 90.500 |
| 本轮 baseline | 5.867 | 88.273 |
| 本轮 best | 5.161 | 89.682 |
| 本轮 last | 5.167 | 89.671 |

主PA按FreiHAND自由O(3)逐帧相似对齐（平移、正交旋转／反射、统一尺度）。0–50mm的100个含端点阈值，逐joint的error≤threshold算PCK，梯形积分／50后平均21joint；百分制AUC增益写“百分点”。[作者HPE evaluator](https://github.com/ThunderVVV/dex-ycb-toolkit/blob/ad97815f946f220f91fd70b1e75191d422ab85bf/dex_ycb_toolkit/hpe_eval.py)、[FreiHAND对齐](https://github.com/lmb-freiburg/freihand/blob/master/eval.py)、[EvalUtil](https://github.com/lmb-freiburg/freihand/blob/master/utils/eval_util.py)。

反射拟合计数：baseline23、best.final52、last.final56；对应proper-rotation PA分别5.866756／5.162860／5.168315mm，与主PA差异很小，未用这个差异解释模型提升。

论文还报告遮挡(50%,75%]的5.03mm／89.9%、(75%,100%]的5.07mm／89.9%；**当前没有物体遮挡比例分组，不能填入这些栏位或用手别分组替代**。本轮与paper对齐的是指标函数，crop／时序窗口及作者逐帧选择等未全部重现；论文数值只作参考。W-MPJPE、WA-MPJPE、RTE、Accel属于论文HOT3D世界轨迹评测，不能用DexYCB固定相机结果换名代填。

## 可支持的结论与局限

- best相对官方baseline：绝对MPJPE下降53.36%、腕位置误差下降55.17%、腕平移对齐MPJPE下降25.91%、PA-MPJPE下降12.02%；PA-AUC增加1.409个百分点。last相应绝对MPJPE下降53.66%、PA下降11.93%。
- best／last表现接近：last绝对MPJPE和腕位置略好，best的PA、腕对齐和手指角误差略好，未做统计显著性检验，不重新定义验证best。
- **手指局部旋转没有优于官方baseline：** 10.143°→best10.203°／last10.228°。不能把整体joint距离改善写成每个关节角都改善。
- 微调raw已达到约30.6mm绝对MPJPE，final绝对MPJPE与raw相近；PA和局部finger较微调raw改善，同时根角／腕对齐误差可能变差。总体收益包含前端联合微调、视觉adapter与TWM的作用，不能当作独立TWM消融收益。以下分支在同一联合模型运行条件下记录，不是独立重训或history反事实。

| 模型分支 | 腕位置 mm ↓ | 绝对MPJPE mm ↓ | PA-MPJPE mm ↓ | finger ° ↓ |
|---|---:|---:|---:|---:|
| best.raw | 30.458 | 30.604 | 5.423 | 11.341 |
| best.candidate | 30.122 | 30.128 | 5.189 | 10.509 |
| best.final | 30.509 | 30.558 | 5.161 | 10.203 |
| last.raw | 30.365 | 30.550 | 5.420 | 11.348 |
| last.candidate | 29.812 | 29.947 | 5.196 | 10.539 |
| last.final | 30.160 | 30.365 | 5.167 | 10.228 |

## 左右手配对核查

| 模型 | 手别 | 有效状态 | 绝对MPJPE mm ↓ | PA-MPJPE mm ↓ | finger ° ↓ |
|---|---|---:|---:|---:|---:|
| baseline | left | 39160 | 68.945 | 5.751 | 10.115 |
| baseline | right | 39608 | 62.137 | 5.981 | 10.171 |
| best | left | 39160 | 32.919 | 5.085 | 10.222 |
| best | right | 39608 | 28.224 | 5.237 | 10.184 |
| last | left | 39160 | 32.476 | 5.083 | 10.210 |
| last | right | 39608 | 28.278 | 5.249 | 10.245 |

本轮左右手的绝对与PA关节误差均改善；旧test1015的左手绝对误差退化是旧模型／旧子集结果，保留历史，但不覆盖本轮结果。新模型的finger角并非两侧都改善。

## W&B与产物

- [baseline](https://wandb.ai/black-ant/handloop-twm/runs/85qboyf8)
- [best4200](https://wandb.ai/black-ant/handloop-twm/runs/wb42qkka)
- [last4311](https://wandb.ai/black-ant/handloop-twm/runs/f2fzfdu6)

[机器可读小型汇总](../reports/dexycb_full_test_final_20261016.json)保存各分支精确均值、分母、左右手、coverage、代码／模型／report SHA和W&B终态。完整comparison.json、每模型report.json／accumulator.json、按subject／view的详细汇总和逐视角predictions.npz留远程：`/sciclone/data10/jchen57/Hand_Reconstruction/evaluations/dexycb_s0_full_three_models_20261016/`。

评测实现协议与启动验收见[完整测试协议](DEXYCB_FULL_TEST_2026-10-16.md)。代码3b0574e基底＋488769b新增评测脚本；冻结baseline／best／last身份在JSON中，未执行新推理、训练、bootstrap或遮挡分组。数据、权重、逐帧预测、原日志不入Git。
