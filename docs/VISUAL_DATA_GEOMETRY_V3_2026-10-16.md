# 前两项修改：扩大真实数据与直接几何监督

日期：2026-10-16。用户本轮明确先推进前两项，原建议第3、4、5项暂缓。本文区分实现、验收与训练后结果；尚无本版本训练后的精度结论。

## 范围

保留每手99D、独立预测β、HaWoR与TWM低学习率全量适配、视觉插件两轮取证和纯轨迹TWM。只扩大可审计的数据覆盖，并增加raw解剖腕J0与final关节的直接绝对几何监督。

raw/candidate受损样例构造、历史长度与提交机制、motion回放及3:1任务顺序、学习率、每卡8段RGB／32个motion窗口和1000总更新均不调整。新实验从同一官方HaWoR＋D37 best初始化，不续用旧联合checkpoint或Adam状态。旧best300身份与旧评测结果保留。

配置见[visual_joint_v3_geometry.json](../config/visual_joint_v3_geometry.json)。与v2配置的差异仅为tracking和`visual_geometry_supervision`；1000步仍是原预算，不是本轮采纳了更长训练。

## 数据的实际职责

### 纯运动

当前已验收完整99D、可进入现有motion trainer的仍是OakInk2、HOT3D、独立UmeTrack、DexYCB。D37权重及四库回放manifest不变。

本轮新增HumanTouch一个固定native glove样本的下载和结构验收；1,597帧、3文件、官方hash核验通过，但`ready_for_training=false`。native根与tracker约9cm的定义／标定差尚未核清，不填造MANO旋转标签，也不称该来源已参与训练。新增候选、访问条件、参考来源、motion／视觉职责和接入门槛见[来源核查](DATA_EXPANSION_SOURCE_AUDIT_2026-10-16.md)。查到数据集不等于已完成训练接入。

### 视觉联合

- **DexYCB**：由原两主体16条train录制／1,128帧，扩大到官方s0允许的10主体800条train录制／50,861个reference-valid帧。使用已下载完整参考中实际有效帧，不声称覆盖所有原始RGB帧。原8条val录制／577帧的RGB和参考文件按字节原样保留；s0 test不用于训练。数据准备源码`d9550a3`。
- **HOT3D**：接入Aria真实RGB214-1；保留原motion parent split和HaWoR27评测录制排除表。按metadata在每个允许parent内均匀选一段clip，共99个物理clip，现已完整准备；选段不看模型误差。Quest3灰度相机未混入RGB。使用官方标定，FISHEYE624先重映射针孔，再对图像、K和camera-to-world一致做顺时针90°变换。
- Dex参考为源拟合MANO的Dex21；HOT参考为源PCA＋源β解码的Meta21，拇指base缺失slot1保持无监督。两者保留各自codec、左手shape约定、米制、world gauge、时间、有效性和来源；它们不因统一loader变成独立手部传感器GT。
- 合并时只hardlink不可变源文件，跨文件系统才逐字节复制；禁止symlink逃逸，核验原始与合并SHA，检查物理parent不跨train／val。manifest每条record保留dataset和参考契约。训练仍由现有clip sampler遍历合并池，不引入新的均衡采样方案；按实际clip数量形成的曝光比例需单独记录。

完整数据准备与合并于06:49:23UTC通过验收。训练池有63,652张不同RGB、73,524个手部状态，两个手共享一张HOT图像不重复算成两张RGB。具体数量：

| 来源／侧别 | Train状态 | Train窗口 | Val状态 | Val窗口 |
|---|---:|---:|---:|---:|
| Dex左 | 25,474 | 990 | 143 | 6 |
| Dex右 | 25,387 | 990 | 434 | 18 |
| HOT左 | 11,012 | 388 | 1,570 | 54 |
| HOT右 | 11,651 | 405 | 1,890 | 65 |
| 合计 | **73,524** | **2,773** | **4,037** | **143** |

单手record为train972／val33。HOT独立RGB为train12,791／val1,948；Dex为50,861／577。67,182个RGB／参考文件逐一严格SHA核验及hardlink发布通过，原Dex val的字节和顺序不变，test absent、物理parent互斥、来源manifest不变；8种source×side×split真实RGB crop／mask读取通过。HOT1条单手clip按预设正深度／图内交集规则无有效帧，拒绝原因保留；未按模型误差筛选。

正式manifest：`/sciclone/data10/jchen57/Hand_Reconstruction/prepared/visual_dex_hot_collection_v3_20261016/source_manifest.json`，SHA256=`6314ec2aea911709595c31e262a55ffa2831d14eb2aa5299810287c60b6a5db3`。完整验收`evaluations/data_expansion_geometry_v3_20261016/visual_collection_acceptance.json`。

## 目标与系数

视觉阶段的新目标为：

\[
L_{visual}=10^{-4}L_{final,99D\ NLL}
+\frac{\operatorname{mean}\|p^{raw}_{J0}-p^{ref}_{J0}\|_2}{0.05\,m}
+\frac{\operatorname{mean}_{valid\ joints}\|J(\hat s,\hat\beta)-J^{ref}\|_2}{0.05\,m}.
\]

三个项分别按真实有效state／腕／joint计数归约，DDP使用全局sum/count，空rank不稀释其他rank。没有Procrustes、尺度或腕平移对齐。final由预测自身β解码，参考β只服务来源参考解码／校验，不能作为模型输入；同样不对跨库原始β强行L2监督。未增加2D、速度、shape参数或中间轮次目标。motion阶段原D37目标保持。

`10^-4`与0.05m是**本轮工程初值，尚非最优或已验证收益的系数**。它们改变了NLL相对新增几何项的影响，不是将全部loss统一缩放。默认配置仍NLL=1／新增几何项=0，旧训练入口行为保留。

初值依据来自train-only、无optimizer的初始化探针：旧Dex train清单中左右各2段（也属于本轮允许train录制），每段8个真实帧。该probe清单为`prepared/visual_feedback_dex_joint_sources_v2_20261016/source_manifest.json`，SHA67462dce…，并非已经覆盖扩大后的整个train池。原NLL在完整TWM参数上的梯度norm约325k–468k，final米制几何约2.32–2.48；采用上述系数后分别约32.5–46.8与46.4–49.7。camera读出对应NLL约14–70、raw约105–379、final约173–309。HOT左右手探针也已通过。探针冻结ViT／IAM上游、eval关闭dropout，只查head／PAM与完整adapter／TWM，不能将这些norm当作全主干实际训练梯度、Adam贡献或最优平衡证据。随后真实mixed全主干检查和正式训练使用的是已完成的扩大Dex／HOT数据。原module clip=1保持，其饱和效应仍需观察。

源码：[losses.py](../src/handloop/visual/losses.py)、[RGB loader](../src/handloop/data/visual_rgb.py)、[MANO codec](../src/handloop/visual/codec.py)、[trainer](../scripts/train_visual_joint.py)。主体实现固定`41bf150`。

## 已完成的验收与失败保留

实验均在用户授权Pod `jchen57-fineturn-tl4zr`进行，本机仅代码、文档和静态检查。

- 双手Dex／Meta21映射、J0语义、预测β梯度、masked NaN与padding、默认NLL一致性、真实双卡不等count／空rank全局归约均通过。初版DDP测试fixture无输入触发Torch2.10 scatter异常；`7e1535c`只补测试输入，重跑两rank通过，未改变训练模型。旧失败日志保留。
- Dex真实train RGB的loss分项梯度探针通过；source checkpoint与所查参数前后不变，reference不进入forward。HOT真实train双手探针也通过，无optimizer。
- HOT烟测train／val各一个物理clip，双手共600参考状态、300独立RGB图像；源码PCA重解码与原prepared99D、源时间戳、相机和SHA检查通过。BF16真实RGB forward、官方crop parity、混合单位raw world还原、reference污染不改变模型输入、padding不提交历史／真实帧只提交一次均通过。
- 远程生成的train／val RGB参考点投影叠图已人工查看，位置与朝向相符；只作标定链QA，不当训练效果或独立精度证据。原图不上传Git。
- HOT99clip首次完整准备被投影自检拦截：某段旋转前后投影最大差1.431689px。失败目录与日志保留，排查修复并重新验收后才可入池。

**相机精度修复与新增集成验收：** 99clip只读审计覆盖29,591个positive handframe、491,156个图内landmark；旧检查有20帧超0.02px，失败点均为近相机平面的离屏点，不能解释为实际手距镜头只有0.14mm。官方工具首次float32量化后，`R.T`与`inv(R)`的差被投影`1/z`放大。`86f25ec`从原始JSON quaternion／translation构造float64 SE(3)，稳定计算`(p-t)@R`；所有positive点独立CW90恒等门槛收紧至1e-5px，实际float32导出图内点仍保持原0.02px门槛，不改depth／bbox／split／选段规则。只读审计float64最大差2.759e-7px，真实float32图内最大差0.00040670px；完整RGB准备需按修复后版本另跑。

`d634999`真实Dex左＋HOT右同一`default_collate` batch的全主干`train()`／BF16／checkpointing forward＋backward已通过：ViT、IAM、head、PAM、adapter及TWM encoder／current／rotation梯度均finite非零，camera与β读出也连通。每源8真实帧，T16带padding，峰值allocated约16.24GiB；无optimizer，所查全部trainable参数与源HaWoR／D37文件hash前后不变。它验证真实混合源的计算链，不是拟合收益；正式每卡8×32的容量仍以启动实测为准。

远程验收根目录：`/sciclone/data10/jchen57/Hand_Reconstruction/evaluations/data_expansion_geometry_v3_20261016/`。原始数据、每帧参考、模型权重和完整日志不上传；小型聚合报告可进入`reports/`。

## 评价边界

本轮同时改变数据覆盖和两个直接几何目标，只能验证这一组合，不能单独归因于其中一个改动。保留原`state_geometry_loss`选best、每50更新评测和fixed8等间距选clip算法；扩大val集合会改变被选clip的实际身份，**新旧fixed8 score不可当同一固定集直接比较**。训练后应在原Dex val577／独立test1015的同输入上与原生HaWoR、旧best300配对，并独立报告HOT held-out；不据test改选checkpoint。

平均绝对几何纠正收益为主，另报左右侧与尾部、腕平移对齐及PA指标。后两者不替代world绝对误差。GT crop、拟合MANO参考与筛选的少量HOT clips仍属于受控pilot，不宣称完整官方benchmark或完整ego部署验证。

## 执行记录

正式联合训练已于**2026-10-16 06:50:51.531197UTC／纽约02:50**启动；启动验收观察到**16/1000次实际更新**，训练继续。它不是训练后精度结论。

- Pod：`jchen57-fineturn-tl4zr`，GPU0／1；torchrun launcher PID3165。
- 源码：`86f25ec842043bce2bcbdab17dbe539bacc3eef3`；不可变Git工作树`code/visual_feedback_joint_v3_camera_20261016`。
- run：`/sciclone/data10/jchen57/Hand_Reconstruction/runs/visual_feedback_joint_v3_geometry_20261016`。
- 初始化：官方`model_assets/hawor_v1/weights/hawor.ckpt`＋D37 `model_assets/twm_d37_final_20261016/best.pt`，全新Adam；未从旧联合权重继续。
- 维持原1000更新、每卡RGB8×32／motion32、3:1、adapter／TWM／HaWoR学习率1e-4／1e-5／1e-6；新配置在同一固定源码中。
- 两rank首8更新审计：14/14组累计出现非零梯度、代表参数均改变、所有现存gradient finite。14组是跨visual／motion更新的累计覆盖，不是每个task同时调用全部分支。每卡peak allocated68,180,786,688bytes≈63.50GiB；瞬时nvidia-smi约78,000MiB仅为快照，不当稳态利用率。
- checkpoint4已落盘并硬链接保留原inode，8,510,562,479bytes；一次CPU检查1076个model／3135个Adam浮点张量全finite，1045个Adam参数状态。模型hash `b29bb2e589daa1316fe349bfd9da123a077f88a8414b0f2bad66d702ad3ee3dd` 与两rank保存一致。
- [W&B 98ff0737](https://wandb.ai/black-ant/handloop-twm/runs/98ff0737)本地／服务端均running；服务端检查收到step6及RGB batch8／配置1000。其时间早于本地step16验收，不冒称同步瞬时一致。
- 新fixed8初始验证有222个有效状态：官方raw腕46.412mm、未训练两轮final腕109.227mm、final绝对joint119.006mm；差的初始化完整保存。初始和首几步train batch不能当训练后平均收益。
- [启动验收小摘要](../reports/visual_joint_v3_launch_20261016.json)、[数据／几何验收](../reports/data_geometry_v3_acceptance_20261016.json)。远程同eval根下有`production_launch_acceptance.json`、`production_checkpoint4_audit.json`、原始日志和初始checkpoint硬链接；旧D40 run／best身份保持。
