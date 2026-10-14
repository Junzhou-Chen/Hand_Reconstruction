# 四个已下载数据集的TWM适用性

日期：2026-10-14。**OakInk2、HOT3D、UmeTrack、DexYCB均已下载完成，均具备作为纯轨迹TWM训练来源的字段条件；前三个MoCap来源建议作为核心，DexYCB作为视觉拟合参考的补充/目标域适配。** 尚未完成99D adapter、FK几何一致性和全库有效帧验收，不能据此称四库已可直接合并训练。ARCTIC依用户D22暂缓。

## 下载完成

远程每库`download_status.json`均complete，process已退出，failed0，文件计数/字节与manifest一致：

| 数据 | 当前包范围 | 验收文件数 | 下载字节 / GB |
|---|---|---:|---:|
| OakInk2 | 627姿态标注＋program元数据＋官方split；无图像 | 634 | 36,673,901,351 / 36.67 |
| HOT3D | 排除HaWoR评测来源后的2455 train TAR＋元数据 | 2461 | 257,066,644,565 / 257.07 |
| UmeTrack | 1020 real train TAR＋README；不含synthetic/test | 1021 | 110,513,346,839 / 110.51 |
| DexYCB | 10 subject＋calibration/models/bop | 13 | 127,611,176,595 / 127.61 |

合计4,129文件、531,865,069,350 B，约531.87GB。前三库依据官方hash；Dex依据公开Drive大小、完整gzip CRC与TAR首header，自算SHA记录，不存在作者SHA匹配。Dex完成于2026-10-14 04:15:58 UTC（纽约00:15:58）。本轮只读完成记录，没有重新读取全库做hash。

## 实际标注抽查

抽查代码`056122f`，在ds的`jchen57-debug-mpd5p`、zhou Python以CPU读取，记录时间2026-10-14 06:08:55 UTC（纽约02:08）。命令：

```bash
/sciclone/home/jchen57/.conda/envs/zhou/bin/python \
  /sciclone/home/jchen57/project/Hand_Reconstruction/scripts/audit_twm_dataset_samples.py \
  --root /sciclone/data10/jchen57/Hand_Reconstruction/datasets \
  --output /sciclone/data10/jchen57/Hand_Reconstruction/audits/twm_four_datasets_20261014.json
```

只读标注，不读取图像、下载模型资产或运行网络。Oak以受限pickle解码，Torch storage限定CPU/weights_only；Meta JSON从TAR读取，Dex NPZ使用allow_pickle=False。详细字段记录仅在远程audit JSON，以下为小型摘要：

| 抽查范围 | 实际字段与时间信息 | 可确认/未确认 |
|---|---|---|
| Oak train索引0/215/430，每份前20帧 | 双手pose_coeffs `[1,16,4]`、tsl `[1,3]`、betas `[1,10]`；MoCap记录3268/2125/3792，id逐1递增；RGB id多为4、偶有3/5步 | 样本有限值，四元数norm最大偏差约1.2e-7；RGB与MoCap索引不混用。未做全帧FK/时间单位验收 |
| HOT Aria/Quest各首尾clip，共600时刻 | 每帧native角22项＋world变换；MANO thetas15、wrist_xform6；shape beta10，native profile；每clip150帧且连续 | 1200个手状态字段有限；Aria时间间隔约31.33–35.41ms，Quest约33.33ms，须读实际ref_timestamp_ns |
| Ume separate_hand/hand_hand各首尾，共1715时刻 | 同类native/MANO/shape字段；帧数450/449/367/449，frame IDs连续；这4份未带info.json绝对时间戳 | 样本有限；一份有1帧MANO存在而native缺失，另一份有1帧右手缺失，两类mask独立保留。相对dt可按官方30Hz标明nominal来源，不伪造精确绝对timestamp |
| Dex一个subject/sequence，6帧＋shape calibration | pose_m `[1,51]`，joint_3d `[1,21,3]`，元数据含hand side、shape校准、固定camera/calibration；beta在mano.yml | 样本有限、pose非全零、joint无-1 sentinel。TAR遇到frame3/45/31/5/19/27，需数字排序，不能当连续顺序 |

这些是schema/数值有限性/索引检查，不是GT真实精度排名、几何误差、全库有效帧统计或训练效果。只在标注存在的GT-valid片段造corruption；真实缺标签不补成GT。

## 各库怎样监督99D

固定输出仍为每手`p3 + root rotation6D6 + 15×finger rotation6D90`，beta10另存。源参数维数不必99；须依官方decoder恢复完整旋转/几何后再统一编码。

| 来源 | 原始表示与转换 | 建议训练职责 |
|---|---|---|
| OakInk2 | 16个wxyz quaternion可转换16个rotation6D；root-centered MANO的tsl按源约定处理，beta保留 | 复杂双手操作、腕与指关节恢复核心 |
| HOT3D | MANO thetas是15个PCA系数，不是15关节轴角；经15×45 basis与hand mean恢复finger轴角，再编码rotation6D。wrist_xform前三为root轴角、后三为MANO transl，不能直接把transl当解剖wrist | world腕/手物运动核心；native几何与MANO模型参考分开，优先保留较可靠native监督 |
| UmeTrack | 同类15-PCA MANO转换；native角与骨架不是MANO15关节，缺native/缺MANO独立mask。参考系统一后可做几何部分监督 | 自由手势、单手/双手互动，补操作库动作覆盖 |
| DexYCB | 51D pose_m为root轴角3＋finger PCA45＋transl3；官方use_pca=True、ncomps45、flat_hand_mean=False，不能把中间45数字直接当轴角。shape读subject校准；固定camera可作为序列固定参考，或用外参统一rig | train补充/匹配DIP恢复对照、HaWoR错误域适配；不当成独立glove/MoCap真值 |

固定源码依据：

- [Oak decoder](https://github.com/oakink/OakInk2/blob/502a02809b50b7f5d0037f92a1ccd2a479051ad1/src/oakink2_preview/launch/viz/seg_3d.py)：quat、use_pca=False、flat_hand_mean=True、center_idx0，FK后加tsl。
- [Meta MANO decoder](https://github.com/facebookresearch/hand_tracking_toolkit/blob/bc628e9286c18444b47a0971a89d075ea6f00747/hand_tracking_toolkit/hand_models/mano_hand_model.py)：use_pca=True、num_pca_comps15、beta10、root axis-angle/transl；[dataset读取](https://github.com/facebookresearch/hand_tracking_toolkit/blob/bc628e9286c18444b47a0971a89d075ea6f00747/hand_tracking_toolkit/dataset.py)分别定义native和MANO。
- [Dex官方示例](https://github.com/NVlabs/dex-ycb-toolkit/blob/64551b001d360ad83bc383157a559ec248fb9100/examples/visualize_pose.py)：PCA45＋非flat mean，输入pose[:,0:48]与pose[:,48:51]；[标注读取与beta](https://github.com/NVlabs/dex-ycb-toolkit/blob/64551b001d360ad83bc383157a559ec248fb9100/dex_ycb_toolkit/dex_ycb.py)。

MANO模型资产/PCA basis/mean、左右手shape修正和FK需在adapter验收阶段落实；本轮不新增资产下载或声称roundtrip已通过。native wrist tracking frame与MANO root/解剖wrist定义也不能直接混合，先核源解码几何、固定坐标、单位与骨架对应。Meta说明converted MANO整体略逊native，支持保留native几何而非只追求字段便利。[官方说明](https://github.com/facebookresearch/hand_tracking_toolkit#important-notes)

## 可以训练，但只取合格train

- Oak保留TaMF431 train/39val/157test；下载全627不表示全部训练。父任务/片段重复还需按原recording排查。
- HOT当前train已从2804 clips移除349个HaWoR评测来源，25原录制对应；完整27评测ID持续排除。clip边界按source/timestamp处理，不跨不同记录接轨迹。
- Ume当前只取官方real train，划内部validation按subject/sequence；不计多camera/render为多份motion，标签mask和非固定帧数保留。
- Dex下载了全10 subject，所以s0/s1等train/val/test均在包中。用于TWM时先固定评测setup，只训练对应train；若与HaWoR默认s0_test比较，s0_test绝不进入预训练、Corrector训练或调参。8view是同一运动，不计8倍独立轨迹。

本轮建议为“现有三个MoCap库核心＋Dex train补充、分别消融”，没有固定混合比、loss权重或全量训练启动。保留D18传感器精度原则、glove后续候选与已有方法框架；不是把Dex采集精度提升到MoCap，也不因为已有RGB就给TWM输入图像。当前恢复主任务与短期预测辅助不变。

## 同日新增：MANO资产与Meta/Dex小样本适配验收

以下为2026-10-14 16:02 UTC后新增事实，前文保留为较早的字段审计记录。小型汇总见[适配验收JSON](../reports/twm_four_source_adapters_20261014.json)，其中记录实际远程路径、源码hash、split、时间和原始receipt的hash。适配器代码为`1c002606d2ef325ca0f471d6d9e11490eb26e3d4`；资产转换脚本的实际执行内容与随后提交的`13ba665`完全一致。全部在ds Pod以CPU执行，专属runtime为Python3.11.15、NumPy2.4.6、Torch2.10.0+cu130、smplx0.1.28、SciPy1.17.1；没有在本机预处理或运行网络。

用户从官网提供的MANO v1.2 ZIP通过指定SHA256校验，原始左右模型及3份license保留，另存NumPy/SciPy派生模型；9个必要数值字段的值与dtype完全不变。Chumpy0.70兼容桥仅在转换子进程使用；禁止导入Chumpy的新进程中，左右smplx加载、mean与平移检查均通过（mean误差0，平移最大误差`1.86e-9 m`）。Meta遵循官方decoder的左手shapedirs修复，Dex不额外套用该修复。每手99D均使用实际解剖腕位置，beta10另存，不能把MANO translation直接当wrist。

| 本次范围 | 已验证结果 | 边界 |
|---|---|---|
| HOT3D 2个TAR，4条单手轨迹 | train/val各300个有效MANO状态；native也有600个有效字段状态；实际`ref_timestamp_ns`；mmap loader各60窗 | 官方train内部按原录制分组，排除27个HaWoR评测来源；此val不是官方validation |
| UmeTrack 2个real train TAR，2条单手轨迹 | train/val各450个有效MANO状态；native只有891帧有效，9帧MANO有而native缺独立保留；名义frame_id/30；loader各105窗 | 官方train内部按subject分组；native有效仅表示字段存在且有限，本次未消费native几何监督 |
| Meta上述6条手轨迹的首/中/末共18个状态 | 与固定官方Meta decoder的vertices、16 joints和full pose最大差均0；99D旋转解码后FK最大差`1.19e-7 m` | 这是官方实现与编码一致性，未测native→MANO独立几何精度 |
| DexYCB左右手各6帧几何抽查，另subject01官方s0各1条train/val | 双侧21 joints最大差`1.36e-7 m`以内；正式bounded产物最大差`1.50e-7 m`以内；固定单view；60帧上限得到51/57连续有效帧，history30＋forecast3、stride4得到5/6窗，全部真实mmap | 40帧上限的v1训练轨迹因前9帧空标注仅31帧，不满足该窗口；保留v1并另建v2。21 joints与MANO来自同一拟合标注，不能当独立MoCap精度 |

Meta/Dex均通过完整MANO状态mask、beta及99D loader契约检查，真实缺标注没有补成GT。Meta的native-invalid计数包含native-absent，Ume的9帧不能重复计为18帧。这里仅更新资产兼容、有限范围转换和几何一致性事实；不据此声明四库全量已验收、跨库运动已去重、native标注绝对准确或训练已完成，Oak本轮状态与正式启动记录另行维护。

## 同日后续：四库全量准备与正式run

截至19:07 UTC，四库完整train/val准备和collection工程验收已完成，正式训练从头运行到18,550步；精确统计、分库split/窗口、源revision、manifest hash及中途结果见[正式启动摘要](../reports/twm_four_source_20261014_launch.json)与[运行记录](TWM_FOUR_SOURCE_RUN_2026-10-14.md)。prepared train/val状态合计2,788,135/271,100，可用窗口636,610/61,370；Dex固定一个camera view，未解析test labels。全量Dex source-fitted 21 joints/FK最大差2.6281e-7m，Oak30,080帧共享实现往返及Meta全部TAR解码仍属于实现/编码一致性，不能升级为独立传感器精度结论。

用户自行注册、登录并明确确认MANO许可后，agent通过官方Chrome下载取得v1.2 ZIP，非未经许可第三方模型。跨源beta约定的左手shapedirs差异尚未统一，本轮`condition_on_beta=false`仅是工程隔离；源beta/有效性继续保存、实际wrist与完整旋转99D监督不变。上述较早字段审计和bounded验收范围保留，当前执行范围由D26四库决定取代当时三核心＋Dex补充的建议；无跨库去重、真实HaWoR收益或物理正确性保证。
