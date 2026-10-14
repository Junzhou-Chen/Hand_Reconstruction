# HaWoR训练数据与对比实验协议

核验日期：2026-10-13。对照HaWoR论文§4、补充§6–8及当前官方源码。**论文主定量评测为HOT3D与DexYCB；ARCTIC和HO3D是其视觉训练来源，EPIC-KITCHENS用于定性展示。** 本次归纳服务HandLoop对比实验，不改变已定99D与pure-motion TWM训练方向。

## 论文数据角色

| 数据集 | HaWoR自身视觉网络训练 | 论文评测角色 |
|---|---:|---|
| HOT3D | 573K帧；另用于motion infiller训练 | Aria 110原视频train / 27 validation；相机、world手轨迹和不可见片段补全 |
| DexYCB | 169K帧 | test；camera-space姿态，All / 50%–75% / 75%–100%遮挡分组 |
| ARCTIC | 165K帧 | 未作为主表定量测试集 |
| HO3D | 66K帧 | 未作为主表定量测试集 |
| EPIC-KITCHENS | 不用于本体训练 | 补充Fig.6/7的野外定性展示 |

四库帧数按论文取整数相加为973K，作者概称约1M；它们是作者采用的训练帧数，不是四个完整数据库规模。**Infiller只用HOT3D**，训练包括真实不可见片段和人工连续mask，保留两端上下文。[论文§3.3、§4、§8](https://arxiv.org/html/2501.02973v1)

视觉ViT继承并冻结WiLoR pretrained backbone，所以四库清单不代表所有基础模型的预训练来源。官方README仍未发布完整training code；论文没明确DexYCB setup、HO3D版本、ARCTIC视角与逐帧筛选，不能仅按四库标准train恢复作者约1M清单。[官方README](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/README.md)

## HOT3D作为world主评测

论文使用的27个validation是作者从Aria原录制留出的集合，**不是Hugging Face的`test_aria`分区**。官方代码版本`66c7d4108d58a716deccd192cb7645170cdc7bd7`（本轮核main未变）在README列出27个ID，按完整VRS导出。`export_gt.py`会生成`val.json`，无须另寻找一份未发布的名单文件。[评测入口](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/README.md#evaluation-on-hot3d)、[名单生成](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/hot3d/export_gt.py)

完整复现数据需要RGB视频、手部MANO参考、相机标定/轨迹、时间对应。官方导出跳过前20个时间戳，从索引20起，使用RGB `214-1`流、去畸变与顺时针90°旋转，30fps；产物含RGB/mp4、`anno.pth`、`head_pose.pkl`、`ego_extrinsics.pkl`、`focal.txt`、`tt_list.pkl`。应沿同一导出实现，不能把别的view或旋转约定直接混入。[导出实现](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/hot3d/viewer_export_gt.py)、[视觉流处理](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/hot3d/Hot3DVisualizer.py)

公开主表评测脚本的条件是：

- 只评右手；使用GT生成bbox/visibility，至少两投影点在画面内为valid，关节点范围扩20%得到crop。
- 取`det & MANOvalid`帧，连续段至少16帧；world误差按100帧块计算。
- W用块前两帧similarity（包含scale）对齐，WA用整块similarity；SLAM相机还先作GT相机的无scale刚体对齐。
- 此脚本没有调用motion infiller，不能将其与demo的full infiller路径合称一个已复现协议。

来源：[GT crop与visibility](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/lib/pipeline/tools.py#L169)、[HOT3D evaluator](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/scripts/scripts_eval/eval_hawor_hot3d.py)、[对齐函数](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/lib/eval_utils/eval_utils.py#L322)。上述是发布源码条件；论文补全Table4b需要单独落实其invisible-segment协议。

**当前数据缺口：** 三库下载已完成，但HOT包是预训练用2455个train TAR，刻意没取27录制来源对应的349个Aria TAR。现有manifest记录这349段覆盖25个源录制、37,212,344,320 B；另两录制`P0011_72efb935`和`P0012_119de519`不在BOP版中。BOP每clip为150帧/5秒，短片和完整录制的SLAM上下文、片段边界及覆盖不同。[当前下载记录](TWM_DATA_DOWNLOADS.md)、[官方Clips定义](https://github.com/facebookresearch/hot3d/blob/main/hot3d/clips/README.md)

忠实复现优先准备27个完整Aria记录；官方获取需要填写邮箱领取14天有效URL manifest，并取得模型资产。若先用349个公开clips做便捷实验，应命名为“HaWoR来源的公开clip子集”，两方法同片段/输入，不直接拼接其数值到原论文full-recording表格。[完整数据获取](https://github.com/facebookresearch/hot3d/blob/main/README.md)

## DexYCB作为camera与遮挡补充评测

DexYCB是固定多视角RGB-D采集，适合评价局部手形与camera-space遮挡恢复。RGB用于推理，GT/calibration用于评价；不能将它代替头戴相机运动的world主结论。[数据官网与下载](https://dex-ycb.github.io/)

HaWoR官方链接的作者fork版本`ad97815f946f220f91fd70b1e75191d422ab85bf`中，`examples/evaluate_hpe.py`默认`s0_test`。其范围为subject03–10，每人sequence排序index `i%5==4`的20序列，8视角，共160原序列×8views。这个默认入口是复现依据，论文没有逐项声明其setup和额外筛选；S1跨主体、S2跨视角等属于另加协议。[作者入口](https://github.com/ThunderVVV/dex-ycb-toolkit/blob/ad97815f946f220f91fd70b1e75191d422ab85bf/examples/evaluate_hpe.py)、[split源码](https://github.com/ThunderVVV/dex-ycb-toolkit/blob/ad97815f946f220f91fd70b1e75191d422ab85bf/dex_ycb_toolkit/dex_ycb.py#L126)

作者HPE evaluator有遮挡分组实现：利用被抓物体mesh投影与depth关系估计遮挡比例；AUC阈值0–50mm、100步。需RGB、labels、intrinsics/calibration和对象mesh/pose，不能只取手轨迹。无效GT排除，但缺预测ID也会被跳过，故两个方法必须使用固定相同的GT-valid覆盖清单并报告失败率。[定制HPE evaluator](https://github.com/ThunderVVV/dex-ycb-toolkit/blob/ad97815f946f220f91fd70b1e75191d422ab85bf/dex_ycb_toolkit/hpe_eval.py)

该fork未提供HaWoR专用的DexYCB序列推理入口，且引用的根`freihand`依赖目录不在当前tree；需补齐依赖和预测格式后验收，不能称一键复现已完成。官方全包标称119G，Google Drive直接下载，也可分subject下载；本轮没有启动新数据下载。[官方包](https://dex-ycb.github.io/#data)

后续D21用户明确要求DexYCB与ARCTIC，Dex官方13分包已在ds后台启动；ARCTIC已准备官方清单，待用户账号。具体获取状态见[下载记录](COMPARISON_DATA_DOWNLOADS.md)，不代表baseline推理或新评测协议已实施。

## HandLoop对比安排建议

1. **主量化用HOT3D27录制，补充用DexYCB。** 先沿发布源码的crop/帧mask/右手/对齐条件复核baseline，再在相同条件下比较HaWoR与HandLoop；算法新增数据来源单独披露。
2. **另报真实检测与完整恢复。** 自动detector、双手、错误valid状态、完全出视野、再可见等是部署实验；HaWoR full infiller与pre-infiller分别命名，保留相同视觉权重和SLAM输出。
3. **评测集合持续排除。** HOT27源录制不进入TWM/Corrector训练；若用作最终比较，从train另划内部validation调参。DexYCB不用其test训练或调参，测试序列不可先拆短窗再随机分流。
4. **额外库单独定义任务。** ARCTIC/HO3D可增加评测，但这是本项目扩展，不能写成HaWoR原论文主测试。HaWoR前端见过其train，跨库结果不能据此称整个系统完全未见该库。EPIC只做定性，不凭外观声称几何精度。

这些是对比建议，尚未运行推理/evaluator、固定新实验split或训练Corrector。当前99D纯轨迹训练池保持原方向，未因视觉论文用库而重排。

相关获取事实：ARCTIC需注册；HO3D v2/v3官网自2024-11-03公开evaluation几何GT与本地eval，不能笼统称test GT未公开。公开joint/vertex GT不自动意味着完整MANO参数。[ARCTIC下载](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md)、[HO3D更新](https://github.com/shreyashampali/ho3d)
