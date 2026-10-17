# 九库 TWM 预训练数据准备

纽约日期2026-10-16夜间，UTC已2026-10-17。用户本轮最终决定：**保留上次全部四库，新增ARCTIC、InterHand30fps、TACO、GigaHands、HO-3D，九库一起pretrain**。先前七库／八库／暂不加HO-3D是本轮中间讨论，已被此明确决定替代。HumanTouch退出本轮并停止下载；DexWild继续暂缓。

## 确定的数据池

| 数据源 | 本轮角色 | 标签来源与准备边界 |
|---|---|---|
| OakInk2 | 原训练池完整保留 | 复用既有prepared manifest及固定SHA，不重新划分 |
| HOT3D | 原训练池完整保留 | 复用既有prepared manifest及固定SHA，既有HaWoR评测录制排除保持 |
| 独立UmeTrack | 原训练池完整保留 | 复用real train及原subject val，不并入synthetic/test |
| DexYCB | 原训练池完整保留 | 复用s0 train/val、固定单视角，不读test标签 |
| ARCTIC | 新增，原包已下载 | Vicon→MoSh++／MANO，原生参数加mean一次，经FK获取实际J0；原官方split隔离 |
| InterHand2.6M 30fps | 新增，数值标注下载 | 多视角关键点／三角化→NeuralAnnot MANO。只取train/val MANO与joint_3d，不取RGB／test／重复5fps |
| TACO V1 | 新增，数值标注下载 | 多视角视觉／接触／时序MANO拟合。官方train内按原sequence稳定SHA约10%工程val，官方test_1～4不读 |
| GigaHands | 新增，数值标注下载 | 多视角三角化→EasyMoCap MANO。采用原始params与chosen frame IDs；按subject约90/10工程划分，合并文本区间避免重复motion |
| HO-3D v2 | 新增，原包已下载 | HOnnotate多帧RGB-D拟合。核对完整MANO、实际腕中心、源asset、物理录制／多视角去重及名义时间后接入 |

这不是九种等精度传感器GT。视觉拟合参考的99D可用性、FK转换一致性和独立物理精度分别记录。没有本轮新增库提升最终精度的实验结果。

## 接口与保留规则

- 每手state99D＝实际wrist p3＋root rotation6D6＋15finger rotation6D90；beta10另存。旋转6D按矩阵前两列c0xyz,c1xyz排列，位置米、时间秒。
- TWM不输入图像；下载数值标注和必要标定／split／source decoder，不额外下载RGB。
- 先按原recording／subject确定split再切窗口；同一动作的多视角、文本、5fps/30fps或版本升级不当作独立motion。
- 新builder `scripts/prepare_twm_expanded_collection.py` 默认要求九库齐全；复核旧四库与前次collection中的**同一绝对路径和固定SHA**，不会为凑新库而替换、删减、重新划分旧库。
- 使用实际 `ForecastTrajectoryWindows` 验证train/val，报告0.1／0.5／1／2秒目标覆盖率。缺帧、源末尾、长度不足通过mask表达，不插值或复制尾帧制造2秒标签。

## 实际运行位置与当前证据

下载进程在ds的 `jchen57-fineturn-tl4zr` Pod内运行。数据根为 `/sciclone/data10/jchen57/Hand_Reconstruction/datasets/twm_expansion_20261017`；源下载工具与日志在 `tools/twm_expansion_20261017`。原四库collection仍为 `prepared/twm_four_sources_99d_v1/manifest.json`，未被修改。

下载器 `scripts/download_twm_expansion.py` 支持断点、独立源锁、大小／公开哈希或本地完整性回执、JSON结构检查、安全解压。TACO使用固定HF revision `fa8ed8b4cf5d46949b90ef852128811a2854066a`，Hand_Poses.zip官方LFS SHA256为 `db649d185ae818e63e9f14577067c1318d0751d35b213de8a06688637a1ece51`。InterHand与Giga没有获得作者公布SHA，不把本地SHA说成官方hash。

InterHand四个JSON已下载、解析和索引：142,194,147／184,442,740／29,427,783／38,809,095 bytes。非稀疏train Capture0–9主要frame-index step3，val Capture0全部step1；结合官方30fps release分别以frame_idx/90与/30建立名义时间，明确这是release-cadence推断而非硬件timestamp。train有1,088个非稀疏候选录制；1,744个稀疏录制暂不进入连续trajectory，未来有时基证据再恢复，不能把缺帧当连续动作。

远程InterHand bounded smoke实际通过：每split选4录制、每手最多128帧，8条hand sequences，train84／val768手状态；source FK→99D→FK通过默认1e-5m容差。此为转换验收，不代表完整池验收或独立精度。原始拟合与native joints误差另存。

HumanTouch PID6739已按本轮用户明确指令发送SIGTERM，已下载部分保留，停止回执写在其数据目录；不再等待HumanTouch作为本轮开训前置。

## 代码与训练状态

新增五库适配入口为 `prepare_interhand_twm.py`、`prepare_taco_twm.py`、`prepare_gigahands_twm.py`、`prepare_arctic_twm.py`、`prepare_ho3d_twm.py`。五库真实Pod bounded smoke均通过；静态通过、bounded验收、整库准备和模型精度分开记录。

`config/twm_nine_sources_moe_transient_v3_b512_10k.json` 只扩展required_datasets及W&B标识，其余沿用上次v3的模型、噪声、损失、global batch1024、10k步和2秒/60步目标。保持旧预算是工程起点，不声称在更大数据池上已经充分训练或混合最优。full_coverage下新旧库的自然窗口比例会变化，须记录每源实际samples。

新训练必须等待九库完整manifest、旧四库保留核验、实际loader及源解码验收；不得把下载中／队列等待／CPU准备称为GPU训练已启动。源码、文档和小型摘要同步GitHub；数据、模型资产、权重、完整日志留远程，沿用根`.gitignore`。

## 04:10 UTC实际队列与已完成产物

完整ARCTIC已准备602条hand sequences：train384,780／val51,766状态，全部保留帧FK往返通过。完整InterHand已准备1,024条hand sequences：train83,730／val14,210状态；稀疏时基、源无效joint和太短片段的拒绝原因保存，不插值补造运动。

TACO bounded smoke为train/val各192状态、4条hand sequences。GigaHands v2 smoke为train384／val376状态、8条hand sequences，独立固定EasyMoCap versus共享smplx的vertex最大误差约1.37e-7m。原始三角化wrist与fitted MANO J0另有约36mm小样本均值差异。独立CPU进一步核得virtual landmark0与J0仅约1.73／1.56mm，raw与virtual仍约36.05／35.31mm，故不能用两个wrist定义解释这项差异；带／不带extra regressor的完整mesh差为0，也无源码证据需要额外optim_params世界变换。这保留为源拟合／处理差异的未完全分解诊断，不调frame shift、筛样或宣称独立GT精度。

HO-3D v2 smoke为ABF10／SS2各96帧，source21 joint最大差2.17e-7m，roundtrip vertex最大1.25e-7m。首次失败源于后续可视化patch与原HOnnotate标注器的指尖顶点不同，已按真实meta及[原标注器](https://github.com/shreyashampali/HOnnotate/blob/d94f6b7e1c09a8c7229799a0d48fdc5f2b416780/optimization/manoTF/batch_mano.py#L230-L237)固定为`[728,353,442,576,694]`，不是放宽阈值。Giga首次边界失败按固定作者包含式切片`[start:end+1]`自然饱和修正，并保存原／有效bounds，未改成exclusive。

04:04首队列PID7413已完成TACO全量（train279,662／val35,192状态），但HO-3D full遇到BB10逐帧beta变化（span28.7003），触发工程上的固定shape检查。已保留失败产物并合作停止该队列，没有删库或训练权重。修复为完全保留逐帧beta[N,10]、不平均／不取首帧，源FK与roundtrip阈值均不变；这属于source拟合变量，不称人体真实瞬时形变。

04:10:17 UTC（纽约10-17 00:10）已启动替代后台队列PID7490：

1. 接纳已完成且已验收的InterHand、ARCTIC、TACO产物，保留其原producer未知／源码快照证据，不将它们冒称由新Git提交重新生成。
2. 重新准备HO-3D v2（保留逐帧shape），随后完整准备GigaHands；TACO／Giga使用未压缩NPZ减少海量小文件，sampled FK范围明确保存。TACO轨迹不依赖可选camera ZIP解压，独立`trajectory_ready.json`绑定官方手参数包receipt与完整9,268个pose/shape文件。
3. 九库真实Forecast loader及旧四固定SHA验收；任一失败则阻止聚合／训练，独立来源仍可继续准备。
4. 通过后由`launch_prepared_twm.py`在空闲GPU0,1启动全新训练，在线W&B使用原私有key文件，不把key写入recipe或日志。

代码为干净独立worktree `code/twm_nine_sources_20261017`，commit `6e294315c114a6dc65a56f2e9973974f59b2e8fd`。Pod无Git binary，因此登录端从clean HEAD导出141个源/配置文件SHA，Pod逐字节核验同一代码；最新aggregate `22b5646dafc7d2bd571513cfc51fe260de832deaceef7f1b1db9a18b08d2c04b`（旧cc0dcd5首次部署aggregate26ae330…与receipt296d64…保留在初次验收记录）。真实positive与隔离fixture的modified-bytes／unknown-file／symlink／wrong-commit四项拒绝检查通过；不要求修改训练runtime或安装Git。

当前队列根`/sciclone/data10/jchen57/Hand_Reconstruction/queues/twm_nine_sources_v2_20261017`（旧不带v2目录保留失败／停止记录），其中`queue_status.json`、`logs/`、`receipts/`和`recipe.json`用于检查。队列STOP只控制准备阶段；GPU训练若已启动，合作停止请求写入对应run的`STOP`。计划run为`runs/twm_nine_sources_moe_transient_v3_b512_10k_20261017`，使用trainer自身UTC10-20 05:10合作截止；队列准备截止04:30，避免临近Pod结束才开始。

**此处04:10快照仅证明替代队列已启动并进入HO-3D完整准备，尚不证明九库GPU训练已产生更新或W&B已建立新run。** 后续以run目录`status.json`和在线服务端实际step为准。

## 官方来源

- [InterHand2.6M官网](https://mks0601.github.io/InterHand2.6M/)，[30fps标注目录](https://drive.google.com/drive/folders/1MJ4ztwhMJ1RyFdsxzAprExvzr1cV6otS)。
- [TACO官方说明与split](https://github.com/leolyliu/TACO-Instructions)，[固定HF数据](https://huggingface.co/datasets/leolyliu/TACO/tree/fa8ed8b4cf5d46949b90ef852128811a2854066a)。
- [GigaHands官方仓库](https://github.com/brown-ivl/GigaHands)，[原始姿态包](https://g-ad09a0.56197.5898.data.globus.org/hand_poses.tar.gz)。
- [ARCTIC数据文档](https://github.com/zc-alexfan/arctic/blob/master/docs/data/data_doc.md)。
- [HO-3D官方代码与版本说明](https://github.com/shreyashampali/ho3d)，[HOnnotate论文](https://arxiv.org/abs/1907.01481)。v3是标注升级，不与v2重复计独立运动。

可跟踪的小型[状态摘要](../reports/twm_nine_source_preparation_20261017.json)与[实际队列recipe](../reports/twm_nine_source_queue_recipe_20261017.json)保存完整路径、版本和失败原因；状态快照不等于持续监控。
