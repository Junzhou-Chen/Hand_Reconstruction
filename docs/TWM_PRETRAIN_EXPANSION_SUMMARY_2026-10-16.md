# TWM夜间重训前的数据扩充总结

日期：纽约2026-10-16夜间；远程最新只读检查为2026-10-17 02:59 UTC。性质：官方资料、发布文件索引、源码及现有实验产物核查；优先级为助手建议，尚未选定新混合比例或启动新训练。

用户本轮希望今晚重新训练，先总结可以扩充的pretrain数据。保持纯轨迹输入、当前受损状态恢复及约2秒未来分布任务、每手99D输出；不因候选含RGB改成视觉预训练。

## 1. 实际已用的数据

当前motion collection仍是远程`prepared/twm_four_sources_99d_v1/manifest.json`，SHA256为`f733aa15cd18ed6f57ee0f16ebdadb3bd0d52924c7d99d08cea988544710b26f`。已完成D37纯轨迹10k及后续联合motion回放使用的是同一个四库池。

| 已入训来源 | Train每手状态数 | 重叠train窗口数 | 标注来源与范围 |
|---|---:|---:|---|
| OakInk2 | 1,414,552 | 346,860 | MoCap参考拟合MANO；官方train |
| HOT3D | 660,300 | 132,008 | marker／native骨架到MANO；已排除HaWoR评测父录制 |
| 独立UmeTrack real | 662,422 | 151,390 | marker/native骨架到MANO；不重复计synthetic渲染 |
| DexYCB | 50,861 | 6,352 | 多视角RGB-D拟合MANO；s0 train固定一相机去重 |
| 合计 | **2,788,135** | **636,610** | 左右手分别计状态，窗口重叠 |

val为61,370窗。原始帧、双手状态、相机图像和重叠窗口不是同一计量；视觉联合的63,652张RGB也不是TWM数据量。四库工程转换已验收，不代表所有MANO旋转是传感器直接测量或独立精度相同。完整计数、源revision及split见[实际池摘要](../reports/twm_data_pool_inventory_20261016.json)、[原协议](TWM_FOUR_SOURCE_RUN_2026-10-14.md)。

## 2. 推荐首先接入的来源

| 来源／建议职责 | 原始参考与官方公开规模 | 本项目状态与接入限制 |
|---|---|---|
| **ARCTIC：下一版完整参数池的首个新增来源** | Vicon marker→MoSh++／MANO拟合，连续双手操作；30Hz。官方P1/P2相同267条train；我们只读统计192,390录制时间帧／384,780双手状态，均是QC前原始量 | **已下载343/343文件、failed=0**，仍未prepared／入训。纯motion只需raw_seqs、meta及splits。需完成source mean、trans→实际J0、单位、左右手、fit error和跨任务排除表验收。参考状态约为现prepared池13.8%的额外原始量，不能预报等量有效窗口。见[官方字段](https://github.com/zc-alexfan/arctic/blob/master/docs/data/data_doc.md) |
| **HumanTouch：大规模glove主候选** | MANUS EMF，每手25点及方向／validity，独立6DoF腕；统一60Hz。首版官方约100h／10任务；1000h是计划 | [公开免申请](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch)，可只取pose/meta。已取1个episode共1,597帧，native结构过验，**ready_for_training=false**。tracker与native根的约9cm定义／标定差尚未核清；不是已测手指误差。需native→MANO几何和输入桥接、腕标定、partial监督 |
| **DexWild raw human：大规模glove主候选** | MANUS人体原骨架＋腕tracker。五个任务human行合计7,917 episodes／2,924,588 transitions；总33+h混有人与机器人，不能全部计作人体运动 | [公开MIT发布](https://huggingface.co/datasets/boardd/dexwild-dataset)，尚未下载／适配。优先`*_manus_full.pkl`与人体tracker；LEAP动作不作人手GT。[TrackerProcessor](https://github.com/dexwild/dexwild/blob/main/dexwild_utils/dexwild_utils/TrackerProcessor.py)有相对首帧处理、插值和平滑；发布样本中的world组合仍需验证，文件名不能代替坐标证据 |
| **Stanford DexCap：小规模glove补充** | Rokoko手部骨架＋T265独立6DoF追踪；wiping30min＋packaging60min，合计90min | [公开CC BY4.0](https://huggingface.co/datasets/chenwangj/DexCap-Data)，未下载／适配。使用raw五指人体点与腕，分开审计SLAM漂移和手指；robot四指action不是目标。不是下行DexterCap |
| **DexterCap / DexterHand：小规模marker补充** | 多视角皮肤marker三角化→MANO；公开最终参数约187MB，full dataset仍未全部发布 | [公开Apache2.0参数包](https://huggingface.co/datasets/pku-mocca/DexterHand)，未下载／适配。20Hz源版与60Hz平滑插值版是同一运动；[生成脚本](https://github.com/PKU-MoCCA/dextercap/blob/main/Dataset/generate_dataset.py)含root转换，不能不核源码就重复平移 |
| **MySign：新增自由手势／手指覆盖候选** | 发布方说明OptiTrack＋MANUS Prime3同步采集、200Hz，SMPL-X retarget与ROM处理后发布FBX。索引5,000 FBX、5名参与者；任务是孤立手语 | [公开CC BY-NC-SA4.0](https://huggingface.co/datasets/mysigner/MySign)，未下载／适配。可补非物体操作手势；净时长、FBX rest frame及手／腕解码未验。官方1000 gloss均衡描述与README的重复take／词名归一化边界应进一步对照，暂不称5,000独立动作；wand标定<1mm也不是手指真值误差 |

**ARCTIC decoder勘误：** [官方`common/body_models.py`](https://github.com/zc-alexfan/arctic/blob/master/common/body_models.py)的MANO是`use_pca=False, flat_hand_mean=False`，pose45需按源hand mean契约解码且只加一次。meta中的`subject_vtemplates`用于SMPL-X；不能照搬GRAB的个性化手模板路径，误称ARCTIC MANO必须注入该模板。以源FK确认J0与β、手别约定后再转99D。

## 3. 保留在候选表，但暂不排在主池前面

| 来源 | 价值 | 暂后置的具体原因 |
|---|---|---|
| [GRAB](https://github.com/otaheri/GRAB) | Vicon，10主体／1,334序列／120Hz；适合连续手运动先验。[HMP训练](https://github.com/enesduran/HMP#training-motion-prior-optional)有使用先例 | 本项目没有合法取得的数据副本，需另走官方源访问。符合高精度路线，但不作为免申请资源开工前置；源手部个性化模板／PCA契约不能省略 |
| [HandX独立自采部分](https://github.com/handx-project/HandX) | 自采OptiTrack手部marker→骨架及MANO，适合双手运动 | 只计`source=HandX`新自采；聚合内HOT3D等不算新增。公开60帧clip未必容纳历史＋2秒未来，需原连续序列或future mask；canonicalization与源许可分别审计 |
| [SynData glove子集](https://huggingface.co/datasets/PsiBotAI/SynData) | 自研外骨骼glove来源的长尾操作；[glove-origin manifest](https://huggingface.co/datasets/PsiBotAI/SynData/blob/main/glove-origin/manifest.json)列95,383 clips／793 volumes | 公开的是重定向ITW每手30 qpos＋11点与腕，非原生MANO全旋转；10Hz不适合直接补高速手指真值。需[专用骨架映射](https://huggingface.co/datasets/PsiBotAI/SynData/blob/main/tools/syndata_render/README.md)，不把全站ego／robot等子集合计当glove量 |
| [OpenGraph MANUS sample](https://huggingface.co/datasets/OpenGraphLabs-Research/manus-egocentric-sample) | 20episodes、96,544行／30Hz；25点每手，可小样本检查native接口 | 只有约54min，独立world wrist SE3未证实，优先局部姿态补充；第三方转存同源样本不重复计量 |

本轮还核过PalmDex、DFKI hand_embodiment等小资源；前者本体关节／世界腕接口需核，后者局部marker及许可／净时长尚待核，不因易下载就提前扩主池。Lightwheel等需申请或完整参考未确认，暂不作为当晚公开可执行项。

## 4. 可以扩大数量，但属于另一类参考

以下资源可以作为后续规模／域多样性对照，不能与独立glove／marker参考不加区分地合并报告精度：

- [GigaHands](https://github.com/brown-ivl/GigaHands)：官方约34h、56人。手由视觉检测／多视角三角化与MANO拟合得到；51个相机不是51倍独立动作。公开参数可取，world／normalized版本要分清。
- [EgoDex](https://github.com/apple/ml-egodex)：约725h train＋97h extra，另7h test不可入训；Vision Pro／ARKit估计的手部SE3，episode内有固定参考系。大规模有吸引力，但不是独立MoCap GT；数值标注虽不依赖RGB训练，官方包未核得整库pose-only入口。
- [Re:InterHand](https://mks0601.github.io/ReInterHand/)：公开MANO fit，多视角视觉估计／拟合；多视角和重光照图像不能重复计为motion。
- [HOI4D](https://hoi4d.github.io/)：**官方2026-03-27已给独立HandPose和CameraParameters直接链接**，纠正旧“只能申请”的概括；本轮未成功实际下载核包。标注来自人工关键点、RGB-D拟合和时序／接触优化，非独立手部marker；需外参链才能转固定world轨迹。

StableHand八库服务quality network；HaWoR视觉训练的用库职责也不同。参考其划分／协议，不能据“论文用了”将上述视觉标签升级为glove同等级GT。更完整的论文职责区分见[既有论文依据](TWM_DATA_PAPER_EVIDENCE.md)。

## 5. 99D与约2秒任务的接入边界

输出保持每手`p3 + root rotation6D6 + 15×finger rotation6D90`，β10另存。ARCTIC等完整MANO fit可以适配完整参数监督；native glove可经可靠对应的joint／bone方向部分监督同一99D输出，但局部骨骼方向并不能唯一确定所有twist，不能把缺标签填零并标成GT。

现`ForecastTrajectoryWindows`要求完整17组参考，**native partial训练尚未实现**。HumanTouch／DexWild还需要输入桥接：经标定的骨架到99D历史／观测、或带残差记录的IK与availability；派生IK不是完整旋转真值。监督回到可靠native几何，shape／骨长、rest frame和腕链分别处理。

保持真实秒数，60Hz原数据的60步只有1秒；重采样不能制造新的测量精度。HandX短clip和10Hz SynData须专门审计可用预测时域。既有D37 train完整60未来窗口比例Oak96.45%、HOT49.96%、Ume83.60%、Dex0；统计与原因见[前轮审计](DATA_EXPANSION_SOURCE_AUDIT_2026-10-16.md#长未来与异构标签的实际约束)。新增长连续运动值得做，未证明它能单独解决当前finger误差或预测问题。

## 6. 今晚建议与本轮完成范围

1. 以原四库为对照，优先完成**ARCTIC**完整decoder／split／QC后形成五库完整参数版本；不是下载完成即可宣称入训。
2. 大规模扩充重点放在**HumanTouch＋DexWild原人体**，先实现并验收native partial链，再逐库加入；不得为了赶开训把IK参数伪装成完整99D GT。
3. **DexCap、DexterHand、MySign**用于补操作／自由手势覆盖，优先级次于前两项；GRAB待用户已有合法访问再接。
4. 跨motion与视觉任务共用held-out原录制排除表；多view／重渲染／聚合转存／插值版不重复计运动。源混合比例和是否增加视觉估计池由固定验证、等预算对照决定，目前没有最优比例实验结论。

这是重训前的选库建议，不是新数据验收完成或用户已选定的最终清单。本轮只读查询远程：ARCTIC complete、HumanTouch样本ready=false，没有新的候选prepared目录，D37状态complete／10,000步；没有启动新下载、预处理或训练。代码、权重、loss、现有评测结果均未改。来源访问状态与映射完成状态分别记录，后续执行再按当时文件复核。
