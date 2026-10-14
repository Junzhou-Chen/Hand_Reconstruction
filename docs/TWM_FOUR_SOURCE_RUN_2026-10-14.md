# 四库联合 TWM 预训练

日期：2026-10-14。用户要求把已经下载的 OakInk2、HOT3D、UmeTrack、DexYCB 统一训练；这是 D25 单库结构化 Mamba pilot 之后的新实验。四库全量准备、collection 验收和真实混合 smoke 已通过，正式 run 于17:51:39 UTC（纽约13:51）启动。19:07:32 UTC只读快照为 **18,550 / 50,000步、phase=training**；计划尚未完成。可复现的小型摘要见 [正式启动与运行验收](../reports/twm_four_source_20261014_launch.json)。

后续D27：用户要求两张卡一起训练并提高batch，本文件所述单卡run已在26,346步保存退出并保留；新双卡阶段从同一模型/AdamW/绝对步数迁移，每卡512、global1024、验证batch仍64。当前正式双卡训练与验收见[双GPU运行记录](TWM_DUAL_GPU_RUN_2026-10-14.md)，不再自动恢复此旧单卡run。

保留纯轨迹输入、当前受损状态恢复主任务、0.1 秒预测辅助和每手 99D＋独立 beta10。结构沿用 6 blocks、d256、同帧关节 attention/FFN、逐关节官方 Mamba，以及 MANO 树 embedding。联合 run 从头初始化，旧 Oak 两次 pilot 与 checkpoint 保留。代码入口为 [train_twm.py](../scripts/train_twm.py)，配置 [twm_four_sources_structured_mamba.json](../config/twm_four_sources_structured_mamba.json)。

## 数据协议

| 来源 | 使用范围 / 划分 | 姿态与 wrist | 时间与坐标 |
|---|---|---|---|
| OakInk2 | 完整官方 431 train / 39 val；157 test 不读取 | raw_mano wxyz 全四元数，flat mean、center_idx0；wrist=tsl | 源 MoCap frame_id/120，stride4；world metres |
| HOT3D | 已下载的有效 official train Aria/Quest3；已排除 HaWoR 27 评测来源；父 sequence_id 约 10% 内部 val | PCA15＋手别 basis＋nonflat mean；wrist 为实际 FK J0 | 实际 ref_timestamp_ns；source recording world metres |
| UmeTrack | 已下载 real train，按 subject 约 10% 内部 val；不纳入 synthetic/test | PCA15，MANO 与 native availability 分开；wrist 为实际 FK J0 | 原 frame_id/30 名义相对时间；保留缺帧 |
| DexYCB | 官方 s0：800 train / 40 val / 160 test；固定 serial836212060125 单视角去重；不解析 test labels | PCA45 手别投影＋mean；FK 21 点与 joint_3d 对照，逐帧最大误差须≤1mm；state p3 为已验证的真实腕 | frame_id/30 名义相对时间；固定 camera frame，未冒称 world |

HOT/Ume 的内部验证划分是本轮工程选择，并非官方测试集。分组在任何 bounded sample 截断前建立；seed20261014。跨库原始动作去重尚未建立，不宣称各库之间完全独立。每个训练样本使用上一历史参考状态构建固定局部 anchor，从各源固定参考系转换；训练时历史来自源GT加合成噪声，尚未验自身预测历史回灌。这不恢复 SLAM gauge 或未知相机运动。

转换入口：[Oak](../scripts/prepare_oakink2_twm.py)、[Meta](../scripts/prepare_meta_twm.py)、[Dex](../scripts/prepare_dexycb_twm.py)、[collection](../scripts/prepare_twm_collection.py)。Oak/Dex 用 NPY 目录、真实 mmap；HOT/Ume 改用每手一个未压缩 NPZ，按需加载到有限 LRU。NPZ 不冒称 mmap。loader 按序列 lazy 加载、检查 recording split、保留时间 gap。完整 17 组 MANO GT 才进入本轮 full-state trainer；raw joints 和不完整标签没有被拟合/补造为完整 99D。

**Shape conditioning 的本轮工程处理：** 源 beta10/有效性/decoder provenance 全部保留；`condition_on_beta=false` 仅将模型 kwargs 的 beta_valid 设 false，旧单库配置默认仍 true。Meta 会翻转左 shapedirs 的 x 分量，原 Dex/Oak 路径不作同一修正。实际 zero＋7 个源 beta、左右手两种相同姿态、对齐实际腕后，非零 beta 左手 joints 最大点差3.74–19.81mm、mesh5.60–24.28mm，右手及零 beta 差0；full pose 旋转参数差0。这说明 shape 的共同语义尚未统一，不影响实际 wrist＋rotation 的99D动态监督。暂不将这些 beta 当共同条件是隔离混杂的工程选择；不是物理必要性证明，也未解决统一FK。以后重新启用 beta 须适配训练，joint/mesh loss须按源decoder或先统一shape。回执在 data10 `audits/beta_convention_20261014/receipt.json`。

## MANO 资产

用户已自行注册、登录并明确确认 MANO 非商业研究许可，随后授权下载。官方 Models & Code 下载成功：本机 `/Users/junzhou/Downloads/mano_v1_2.zip`，ZIP SHA256 `50976831790ea9657d8110e0c94e50e90eaf35cd76169f0b27e5d32f3fcd951f`，远程传输后 SHA 一致。来源为 [官方 MANO 下载页](https://mano.is.tue.mpg.de/download.php)，[许可](https://mano.is.tue.mpg.de/license.html)。模型原始/派生文件留在 data10，受 `.gitignore` 排除。

旧 Python2/chumpy pickle 转换为独立 NumPy/SciPy 模型，不覆盖原始文件。转换脚本 [prepare_mano_assets.py](../scripts/prepare_mano_assets.py) 保存 ZIP、原始与派生 hash，并用新进程检查不依赖 chumpy。参考解码 [meta_mano.py](../src/handloop/data/meta_mano.py) 始终显式展开 PCA，避免 smplx K45 自动关闭 PCA 的陷阱；hand mean 加一次。Meta 左 shapedirs 修复与 Dex 原始 convention 分开记录。

## 训练与验收

初始配置沿用 batch64、50,000 steps、lr2e-4、warmup500、weight decay0.01、BF16 网络、FP32 几何、forecast weight0.1；每500步保存，每1000步分库验证。库间等概率采样，库内按窗口均匀采样；不是按总帧数采样，也不宣称这是最优混合比例。每库每模式最多固定1024 validation windows，分别报告 noisy/missing/clean；checkpoint score 为四库等权 noisy/missing 当前恢复 loss。指标仍是 source wrist mm、root/finger geodesic degrees；没有将其写成 MPJPE、真实 HaWoR 收益或物理保证。训练和验证各自缓存至多8192序列/4GiB，仍lazy且有明确上限；NPZ只加载训练所需数组。

远程继续复用 `jchen57-debug-mpd5p`、项目独立 Torch2.10/cu130 runtime，不修改 zhou。新增准备依赖 smplx0.1.28、SciPy1.17.1、chumpy0.70、six1.17.0、requests2.34.2；chumpy 初次 PEP517 build isolation 因缺 pip 失败，`--no-build-isolation` 后安装成功。Oak 首次导入因 requests 缺失退出，安装后重启；失败记录保留。所有预处理、数值与网络运行均在 Pod。

已完成的远程 loader 验收：旧输出/缺帧窗口一致、lazy 初始化、mmap/LRU、tensor/mask schema、分库固定验证、20k 等库抽样及 RNG、partial GT fail-closed、recording 跨 split/重复路径拒绝、raw-joints-only 拒绝。人工数据的两来源抽样为 10002/9998；这是采样程序验收，不是训练精度。Dex 单遍 gzip TAR 的元数据排序、s0、固定 view、NPY/NPZ 写入也通过 bounded 工程检查。

真实适配验收见 [四库摘要](../reports/twm_four_source_adapters_20261014.json)。`2913f28` 的真实4库3步smoke在GPU1正常complete，24个样本为Dex3/HOT6/Oak9/Ume6；loss/梯度finite，latest/best step3权重finite，四库各自验证输出。产物 `runs/twm_four_sources_smoke_20261014`，回执 `audits/four_source_smoke_20261014/receipt.json`。这是功能检查，不是质量结果；正式run从头开始，不resume smoke。

全量 Oak 于16:31:57UTC complete：431train/39val，940手序列，1,414,552/113,382手帧；每序列最多32帧共30,080帧共享smplx FK roundtrip，joint max1.49e-7m、mesh1.88e-7m，非独立源GT精度。产物 `prepared/oakink2_99d_full_v2`，源码13ba665；2worker旧v1部分输出保留，依据CPU占用改8worker新v2。Dex于16:37:30、Ume于17:31:36、HOT于17:45:26 UTC manifest complete（各status写入可稍晚）。

I/O修正有实测依据：暖源HOT样本NPY保存占1.400/1.571秒；同数组未压缩NPZ保存0.155秒，相比NPY1.427秒约9.2倍，仅指该保存阶段。HOT/Ume各2TAR、86数组bitwise一致，12个loader窗口所有字段一致。冷TAR首次direct5.813秒（索引5.692秒），约93MB整包冷读＋RAM解析3.683秒，表明source NFS seek仍有耗时；没有把局部速度写成全库端到端提升。当前Meta新full_v3使用NPZ，旧NPY v2进程已核身份后替换，旧部分文件保留。初次v1因launcher日志位于fresh-output目录被拒绝，之后日志移到audits；不删除/更换用户Job。

提高Meta并发后使用`3e9b039`的`--reuse-equivalent-files`：HOT 12workers、Ume 8workers，重新读取全部TAR并完整重算FK与数组；现存NPZ须keys/dtype/shape/C-order bytes完全一致才能保留，包含正负零差异检查，不以文件存在代替验收。bounded 6文件86数组一致，6种改动拒绝；默认fresh和NPY路径仍不允许此复用。全量HOT重新解码2,455 TAR、重算4,910手序列/73,650数组，严格验后保留1,809旧NPZ、新写3,101；Ume重新解码1,020 TAR、重算1,728手序列/22,464数组，保留1,481、新写247。原进程经身份核对后退出，旧元数据/hash与失败记录保留，用户Job不变。

## 完整数据与实际启动

以下为每手状态数和可用连续窗口数；窗口长30帧历史、3帧后预测，stride4。训练mask为完整17组MANO参考，不将native标签有效性当作MANO精度。

| 来源 | prepared手序列 | train / val状态 | train / val可用窗口 |
|---|---:|---:|---:|
| OakInk2 | 940 | 1,414,552 / 113,382 | 346,860 / 27,728 |
| HOT3D | 4,910 | 660,300 / 76,200 | 132,008 / 15,240 |
| UmeTrack | 1,728 | 662,422 / 78,677 | 151,390 / 18,008 |
| DexYCB | 840 | 50,861 / 2,841 | 6,352 / 394 |
| 合计 | **8,418** | **2,788,135 / 271,100** | **636,610 / 61,370** |

HOT按232个父录制分209train/23val，13participants、2,201/254 TAR；27个HaWoR评测来源在解码前排除。Ume 41subjects分37train/4val、916/104 TAR；保留741,099个有效MANO状态。Ume的`mano_valid_native_invalid`包含`mano_valid_native_absent`，两者各9,725，不能相加；这仅说明availability不同，不证明两种标签误差等价。Dex固定一个view、全10subjects，逐个有效帧对照提供的21 fitted joints，实际最大差2.6281e-7m；这是source拟合/FK一致性。Dex prepared train800条中仅798条产生连续窗口，因此loader总计7,608train/808val可用序列，而非8,418条全部都有窗口。

四个叶manifest hash与collection逐一匹配，collection SHA256为`f733aa15cd18ed6f57ee0f16ebdadb3bd0d52924c7d99d08cea988544710b26f`。精确叶hash、源revision、decoder/assets、split统计与train/val窗口数保存在上述JSON。collection以17:46:11 UTC为开始时间，完成loader检查后由一次性协调器启动；不是以部分准备结果先训练。

所有远程路径以下列共同根目录为基准：`/sciclone/data10/jchen57/Hand_Reconstruction`。

| 产物 | 路径 |
|---|---|
| Oak全量 | `prepared/oakink2_99d_full_v2/manifest.json` |
| HOT全量 | `prepared/hot3d_99d_full_v3/manifest.json` |
| Ume全量 | `prepared/umetrack_99d_full_v3/manifest.json` |
| Dex全量 | `prepared/dexycb_99d_full_v1/manifest.json` |
| 四库collection | `prepared/twm_four_sources_99d_v1/manifest.json` |
| 正式run与checkpoint | `runs/twm_four_sources_structured_mamba_20261014/{run_metadata.json,status.json,metrics.jsonl,latest.pt,best.pt}` |
| 一次性协调器记录 | `audits/four_source_training_coordinator_20261014/{process.json,status.json,coordinator.log}` |

正式run源码`2b40aa9790c1d18651d6ac810d23a72d3c7a5922`，GPU0、PID2599，从头初始化，无resume旧pilot或smoke。协调器启动前核trainer/config SHA，完整命令在正式run的`process.json`及小型摘要。trainer配置SHA为`939260a27ef6a212dbc7ee85cafc6528ae7ab7c9b825e1452072d2e9e0c95302`。训练后文档提交不改变本次run的源码provenance。`run_metadata.phase=starting`是初始静态记录，运行状态读`status.json`。

首条真实训练事件17:54:37 UTC step25、loss0.688073、grad2.24077，样本Dex389/HOT421/Oak389/Ume401；四库已经实际参与梯度训练。19:07:32 UTC step18,550、loss0.024039、grad0.915672、约4.21step/s、峰值torch allocated4.73GB；累计样本Dex297,154/HOT296,754/Oak295,896/Ume297,396。已记录742条training事件和18次分库validation，训练日志中上述loss/grad字段均finite。19:02:30 UTC一次CPU只读checkpoint验收：latest17,000/best16,000可读，每份163个浮点model tensor全finite，code/从头初始化/beta=false符合协议；这是只读tool输出验收，未伪造额外receipt。checkpoint随后继续推进。

## 中途合成验证快照

19:05:29 UTC **step18,000**固定val子集，每模式Oak/HOT/Ume各1,024窗口、Dex394，共3,466；表为四库等权均值。仅评价人工noisy/missing和clean输入下的当前恢复，不是完整val全窗口评测。

| 模式 | wrist mm：输入→恢复 | root deg：输入→恢复 | finger deg：输入→恢复 |
|---|---:|---:|---:|
| noisy | 49.542→6.155 | 22.747→2.512 | 15.481→2.153 |
| missing | 37.897→6.066 | 16.776→2.471 | 11.070→2.176 |
| clean | 0→2.925 | 约0→1.456 | 约0→1.388 |

noisy/missing损失等权score0.01453065；分库和辅助forecast数值均在JSON中。clean输入仍出现退化，必须保留此结果，不能仅用受损输入的改善宣称质量保证。指标是source实际腕距离和MANO旋转测地角；没有独立模型joint/mesh GT指标，也未验证真实HaWoR误差或物理规律。

中断恢复限制：同config/manifest SHA协议才可resume，不能把完成的单Oak50k当四库初始化；保存主要RNG/optimizer/checkpoint，但不保证sampler chunk/loader游标逐样本精确复现。
