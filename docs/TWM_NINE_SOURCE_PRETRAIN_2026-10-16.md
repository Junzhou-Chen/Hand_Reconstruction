# 九库 TWM 预训练：数据准备与真实启动验收

本轮最终范围是原OakInk2／HOT3D／独立UmeTrack／DexYCB全部保留，新增ARCTIC／InterHand30fps／TACO／GigaHands／HO-3D v2。七库、八库及暂不加HO-3D的讨论已被此用户决定替代。HumanTouch已按用户要求停止，DexWild继续暂缓。

## 当前实际状态

**2026-10-17T04:49:42.393400+00:00，九库全部准备完成，双卡已经实际训练到step 20，checkpoint step 10验收通过，在线W&B服务端已收到真实训练指标。** [打开W&B](https://wandb.ai/black-ant/handloop-twm/runs/fa17b1ef)。这不是仅有PID、step0验证或准备队列的启动记录。模型仍在后台继续运行，不要求本轮等待10k全部完成。

计算在ds的`jchen57-fineturn-tl4zr` Pod；代码commit `170c3473e1288d3d5a2877d0553185ce180af36a`，独立worktree `code/twm_nine_sources_20261017`。2×RTX PRO 6000 Blackwell，每卡512、global batch1024、从头训练10,000步，保持原D37 v3的模型／损失／噪声／优化器预算。使用更多数据后自然窗口比例改变，未声称此预算或混合比例已最优。

任务是纯轨迹当前状态恢复与未来60步分布预测；不输入RGB或视觉feature。每手99D＝实际MANO J0位置3＋root rotation6D6＋15finger rotation6D90；beta10独立保存，本轮关闭beta conditioning。旋转取矩阵前两列c0xyz,c1xyz，位置米、时间秒。

## 实际数据量与窗口

| 数据源 | train手状态 | val手状态 | train窗口 | val窗口 |
|---|---:|---:|---:|---:|
| OakInk2 | 1,414,552 | 113,382 | 346,860 | 27,728 |
| HOT3D | 660,300 | 76,200 | 132,008 | 15,240 |
| 独立UmeTrack | 662,422 | 78,677 | 151,390 | 18,008 |
| DexYCB | 50,861 | 2,841 | 6,352 | 394 |
| ARCTIC | 384,780 | 51,766 | 91,972 | 12,402 |
| InterHand2.6M 30fps | 83,730 | 14,210 | 1,215 | 54 |
| TACO V1 | 279,662 | 35,192 | 56,574 | 7,152 |
| GigaHands | 5,921,548 | 1,086,966 | 1,245,932 | 228,794 |
| HO-3D v2 | 23,011 | 1,744 | 5,588 | 420 |

状态合计train **9,480,866**／val **1,460,978**。这些是每手状态数，不是独立视频数或物理动作时长；左右手分别计数，窗口按stride4重叠。真实loader使用history30、minimum future3、最大future60；尾部／缺帧通过mask表示，不插值、桥接或复制尾帧制造2秒GT。各源0.1／0.5／1／2秒覆盖见[集合检查摘要](../reports/twm_nine_source_preparation_20261017.json)。部分时间源是有说明的名义cadence，不能称硬件timestamp精度。

原四库的prepared manifest绝对路径、固定SHA、原train／val及评测录制排除都与前次四库collection一致，已实际验证。未为了新增库替换或重新划分原数据。

## 五个新增源的处理协议

- **ARCTIC**：原官方train／val；301个可用原录制中train267、val34，602条双手序列。Vicon→MoSh++／MANO的非flat mean加一次；米制世界坐标，经FK取得实际J0。subject个性化SMPL-X模板不误当MANO模板；全436,546状态FK检查通过。
- **InterHand30fps**：只取train／val的MANO NeuralAnnot和native joint JSON，不取RGB／test／重复5fps。train Capture0–9主要frame-index step3，val Capture0 step1，以release cadence分别用frame_idx/90与/30建立名义时间；这是公开索引的时基推断，非硬件时间证明。1,744个稀疏录制因缺连续时基暂不使用；native joint无效及太短片段原因留存，不插值补造轨迹。源pose非flat mean加一次，按source左手shape约定与实际J0转换。
- **TACO V1**：固定官方train内按原sequence稳定SHA约10%工程val，官方test_1～4排除。full48D axis-angle按flat mean，center_idx0下hand_trans已是实际J0，不再加offset；左右手原beta保留，无补零。固定HF revision及官方LFS SHA验包；完整9,268个pose／shape文件已处理，optional camera包不是开训依赖。
- **GigaHands**：按完整已发布12,775个params录制建subject约90/10工程train／val，不使用官方caption-row随机划分。31个无caption录制同样进入数值候选；caption起止与None/Buggy文字不裁剪或筛除99D GT。原Rh／Th与MANO pose组合，实际J0＝R(Rh)@shaped_J0＋Th，root＝R(Rh)@R(pose[:3])，finger mean加一次。chosen frame IDs、原行数与时间对应必须可证，单手缺失不从另一手补。实际导出25,542条手序列。
- **HO-3D v2**：原HOnnotate完整MANO参数、right hand、flat mean；实际J0通过FK取得，保留OpenGL固定相机坐标。物理录制去重，只选固定单视角；train SB10/12/14与evaluation SB11/13属于同SB1物理组，整组从motion train排除。原标注器tips为[728,353,442,576,694]，没有沿用后续visualizer不同tips。逐帧beta[N,10]原样保留，绝不平均或用首帧替代。30Hz是公开HMP处理协议的名义约定，原capture fps未独立核实。

这些是来源不同的fitted参考，不是九种等精度的独立传感器真值。原始／拟合误差、参数转换一致性与模型最终精度必须分开。

## 真实失败及修复

1. HO-3D首次source21比较失败，按原HOnnotate实际tips修复；96帧ABF10与96帧SS2的source joint最大差约2.17e-7m。全量BB10触发静态shape假设后改存逐帧beta，阈值未放松。
2. Giga annotations引用7个未发布params。122.54秒完整TAR成员＋gzip EOF/CRC审计证实缺失，不是漏解压；完整12,775个已发布params与提取集合一致。7个缺失身份及其split／ranges保留为来源审计，不编造GT。
3. Giga早期工程适配依赖caption eligibility／范围，现已更正为`pure_motion_all_params`全原片段。所有已发布数值动作先入候选，再按真实数值与时间规则验收。
4. 两个录制`p008-boxing/014`、`p016-packing/042`的发布set-union行序无法可靠证明与时间对应，明确隔离；3个右手不足两帧不导出。
5. `p017-fastfood/049`左手source拟合发散：345行非有限，其余215个finite/chosen帧仍有约1e16的旋转量和4.67e17m平移，float32世界mesh坍缩为零直径。按已有3cm–50cm bbox guard逐手拒绝，原参数不修改；右手561帧保留，全帧FK最大1.46e-7m。此例也说明单纯source-FK一致不能证明参考合理。
6. ProcessPool在future异常时原先等全部已提交任务才回报失败，现立即记录recording／split／traceback并取消未执行任务。未知异常及FK mismatch仍hard fail，不静默吞掉。

早期队列与partial产物均保留。最终队列`queues/twm_nine_sources_v5_20261017`、Giga `prepared/gigahands_99d_full_v5`；初次／v2／v3／v4队列的失败或合作停止记录见状态摘要。全量Giga每手最多32帧做独立EasyMoCap source FK；TACO每手32帧做共享smplx的编码往返，未独立运行作者decoder。两者均不能称全帧验收；ARCTIC全帧及选中smoke／049诊断的全帧检查另有明确范围。Giga独立腕审计中raw triangulation与MANO fit仍有约36mm小样本差异，约1.7mm wrist语义差解释不了它；未调整frame shift／PA／尺度来掩盖，也未宣称独立GT精度。

## 启动验收与后续检查

[scripts/audit_twm_nine_source_launch.py](../scripts/audit_twm_nine_source_launch.py)已实际核验：

- 九库各有真实train／val窗口，collection、run与sampler人口一致，早期实际采样九源均大于0。
- checkpoint同一个打开fd计算SHA再CPU加载，model及Adam状态有限，Adam步数已证实更新；两rank模型hash与真实checkpoint tensor字节一致。
- W&B online API取得同一history row中的optimizer_step≥10与finite train/loss，未把本地tracking metadata当服务端证据。

[独立验收JSON](../reports/twm_nine_source_launch_acceptance_20261017.json)记录具体step、SHA、计数和在线记录；[准备摘要](../reports/twm_nine_source_preparation_20261017.json)记录九库路径、SHA、split、source与失败历史；[实际recipe](../reports/twm_nine_source_queue_recipe_20261017.json)可复现执行。

远程根`/sciclone/data10/jchen57/Hand_Reconstruction`。集合为`prepared/twm_nine_sources_99d_v1/manifest.json`；训练为`runs/twm_nine_sources_moe_transient_v3_b512_10k_20261017`，其中`status.json`、`metrics.jsonl`、`latest.pt`／`best.pt`与`train.log`留远程。save每100步、首次10步另存；val每500步，checkpoint按既有验证规则选best，不读test。训练合作截止2026-10-20T05:10Z，早于现Pod时限；写run/STOP可合作保存停止。主队列STOP仅控制准备阶段。

Pod无Git binary；登录端从clean HEAD导出142个源／配置文件SHA，Pod按同一完整commit逐字节核验。最终aggregate `ea2b9c57fa71eefd3daefe11812b0e5aa80997b4ca0866b3b00f61231b3c33e8`。无Git positive与modified／unknown-file／symlink／wrong-commit拒绝fixture已验证，不要求安装Git或修改CUDA runtime。

代码、文档和小型摘要同步GitHub；数据、MANO资产、权重、缓存和原始日志沿.gitignore留在远程。本轮验收证明训练真实运行，不代表新增五库已提升下游重建精度。

## 官方来源

- [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/)，[30fps标注目录](https://drive.google.com/drive/folders/1MJ4ztwhMJ1RyFdsxzAprExvzr1cV6otS)。
- [TACO官方split与说明](https://github.com/leolyliu/TACO-Instructions)，[固定HF数据](https://huggingface.co/datasets/leolyliu/TACO/tree/fa8ed8b4cf5d46949b90ef852128811a2854066a)。手参数包官方SHA256 db649d185ae818e63e9f14577067c1318d0751d35b213de8a06688637a1ece51。
- [GigaHands固定源码](https://github.com/brown-ivl/GigaHands/tree/8ad3ef9a50157f44a7eef5645b0766362110a961)，[原始参数包](https://g-ad09a0.56197.5898.data.globus.org/hand_poses.tar.gz)。下载local SHA不冒称作者公开hash。
- [ARCTIC官方数据文档](https://github.com/zc-alexfan/arctic/blob/master/docs/data/data_doc.md)。
- [HO-3D](https://github.com/shreyashampali/ho3d)，[原HOnnotate tips](https://github.com/shreyashampali/HOnnotate/blob/d94f6b7e1c09a8c7229799a0d48fdc5f2b416780/optimization/manoTF/batch_mano.py#L230-L237)。v3为升级标注，不与v2重复计独立motion。
