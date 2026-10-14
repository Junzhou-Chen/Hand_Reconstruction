# 四库联合 TWM 预训练

日期：2026-10-14。用户要求把已经下载的 OakInk2、HOT3D、UmeTrack、DexYCB 统一训练；这是 D25 单库结构化 Mamba pilot 之后的新实验。当前本文先记录执行协议，实际验收和启动状态将在远程检查后补充。

保留纯轨迹输入、当前受损状态恢复主任务、0.1 秒预测辅助和每手 99D＋独立 beta10。结构沿用 6 blocks、d256、同帧关节 attention/FFN、逐关节官方 Mamba，以及 MANO 树 embedding。联合 run 从头初始化，旧 Oak 两次 pilot 与 checkpoint 保留。代码入口为 [train_twm.py](../scripts/train_twm.py)，配置 [twm_four_sources_structured_mamba.json](../config/twm_four_sources_structured_mamba.json)。

## 数据协议

| 来源 | 使用范围 / 划分 | 姿态与 wrist | 时间与坐标 |
|---|---|---|---|
| OakInk2 | 完整官方 431 train / 39 val；157 test 不读取 | raw_mano wxyz 全四元数，flat mean、center_idx0；wrist=tsl | 源 MoCap frame_id/120，stride4；world metres |
| HOT3D | 已下载的有效 official train Aria/Quest3；已排除 HaWoR 27 评测来源；父 sequence_id 约 10% 内部 val | PCA15＋手别 basis＋nonflat mean；wrist 为实际 FK J0 | 实际 ref_timestamp_ns；source recording world metres |
| UmeTrack | 已下载 real train，按 subject 约 10% 内部 val；不纳入 synthetic/test | PCA15，MANO 与 native availability 分开；wrist 为实际 FK J0 | 原 frame_id/30 名义相对时间；保留缺帧 |
| DexYCB | 官方 s0：800 train / 40 val / 160 test；固定 serial836212060125 单视角去重；不解析 test labels | PCA45 手别投影＋mean；FK 21 点与 joint_3d 对照，逐帧最大误差须≤1mm；state p3 为已验证的真实腕 | frame_id/30 名义相对时间；固定 camera frame，未冒称 world |

HOT/Ume 的内部验证划分是本轮工程选择，并非官方测试集。分组在任何 bounded sample 截断前建立；seed20261014。跨库原始动作去重尚未建立，不宣称各库之间完全独立。每个训练样本使用上一已完成腕状态构建固定局部 anchor，从各源固定参考系转换；这不恢复 SLAM gauge 或未知相机运动。

转换入口：[Oak](../scripts/prepare_oakink2_twm.py)、[Meta](../scripts/prepare_meta_twm.py)、[Dex](../scripts/prepare_dexycb_twm.py)、[collection](../scripts/prepare_twm_collection.py)。状态以 NPY 目录存储，实现真实 mmap；loader 按序列 lazy 加载、检查 recording split、保留时间 gap。完整 17 组 MANO GT 才进入本轮 full-state trainer；raw joints 和不完整标签没有被拟合/补造为完整 99D。

## MANO 资产

用户已自行注册、登录并明确确认 MANO 非商业研究许可，随后授权下载。官方 Models & Code 下载成功：本机 `/Users/junzhou/Downloads/mano_v1_2.zip`，ZIP SHA256 `50976831790ea9657d8110e0c94e50e90eaf35cd76169f0b27e5d32f3fcd951f`，远程传输后 SHA 一致。来源为 [官方 MANO 下载页](https://mano.is.tue.mpg.de/download.php)，[许可](https://mano.is.tue.mpg.de/license.html)。模型原始/派生文件留在 data10，受 `.gitignore` 排除。

旧 Python2/chumpy pickle 转换为独立 NumPy/SciPy 模型，不覆盖原始文件。转换脚本 [prepare_mano_assets.py](../scripts/prepare_mano_assets.py) 保存 ZIP、原始与派生 hash，并用新进程检查不依赖 chumpy。参考解码 [meta_mano.py](../src/handloop/data/meta_mano.py) 始终显式展开 PCA，避免 smplx K45 自动关闭 PCA 的陷阱；hand mean 加一次。Meta 左 shapedirs 修复与 Dex 原始 convention 分开记录。

## 训练与验收

初始配置沿用 batch64、50,000 steps、lr2e-4、warmup500、weight decay0.01、BF16 网络、FP32 几何、forecast weight0.1；每500步保存，每1000步分库验证。库间等概率采样，库内按窗口均匀采样；不是按总帧数采样，也不宣称这是最优混合比例。每库每模式最多固定1024 validation windows，分别报告 noisy/missing/clean；checkpoint score 为四库等权 noisy/missing 当前恢复 loss。指标仍是 source wrist mm、root/finger geodesic degrees；没有将其写成 MPJPE、真实 HaWoR 收益或物理保证。

远程继续复用 `jchen57-debug-mpd5p`、项目独立 Torch2.10/cu130 runtime，不修改 zhou。新增准备依赖 smplx0.1.28、SciPy1.17.1、chumpy0.70、six1.17.0；chumpy 初次 PEP517 build isolation 因缺 pip 失败，`--no-build-isolation` 后安装成功。所有预处理、数值与网络运行均在 Pod。

已完成的远程 loader 验收：旧输出/缺帧窗口一致、lazy 初始化、mmap/LRU、tensor/mask schema、分库固定验证、20k 等库抽样及 RNG、partial GT fail-closed、recording 跨 split/重复路径拒绝、raw-joints-only 拒绝。人工数据的两来源抽样为 10002/9998；这是采样程序验收，不是训练精度。Dex 单遍 gzip TAR 的元数据排序、s0、固定 view、NPY/NPZ 写入也通过 bounded 工程检查。

完整 PCA/FK 来源对照、四库全量转换、真实四库 smoke 和正式启动状态仍待下一次远程验收补充。中断恢复保存主要 RNG/optimizer/checkpoint，但不保证 sampler chunk/loader 游标逐样本精确复现。
