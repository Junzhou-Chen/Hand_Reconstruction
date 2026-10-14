# OakInk2 单库 TWM 首轮预训练运行记录

2026-10-14。用户要求先从已下载的四库选一库跑起 pretrain，数学讨论暂缓。已实现纯轨迹当前状态恢复模型，在用户既有 ds Pod 后台训练；本记录是首轮 pilot 的配置与中途结果，不是完整 HandLoop 或真实视觉纠错结果。**用户随后要求暂停，已在纽约05:16（09:16 UTC）停止；不得自动恢复或另启训练。**

## 运行与产物

| 项目 | 实际值 |
|---|---|
| 正式训练代码 | `4cb1da83cc86732b36cea76673bb1e703eae4102`，已推送 `origin/main` |
| 环境 | `jchen57-debug-mpd5p` / namespace `jchen57`；现有 `zhou` Python，Torch 2.12.0+cu130 |
| GPU | GPU 0，RTX PRO 6000 Blackwell Max-Q；单卡 BF16 模型计算，几何和 loss 使用 float32 |
| 开始 | 2026-10-14 09:01:01 UTC（纽约 05:01） |
| 原计划 / 当前状态 | 原50,000步，batch64；每500步验证保存。用户在step28,213暂停，未完成原计划 |
| 最新核验 | 09:16:13 UTC，PID805已退出，phase=stopped，latest.pt可读且step28,213；best.pt可读且step27,500 |
| 代码目录 | `/sciclone/home/jchen57/project/Hand_Reconstruction` |
| 预处理目录 | `/sciclone/data10/jchen57/Hand_Reconstruction/prepared/oakink2_99d_pilot_v1` |
| 正式运行目录 | `/sciclone/data10/jchen57/Hand_Reconstruction/runs/twm_oakink2_pilot_20261014` |

后台进程独立于 SSH 会话。本轮没有创建、删除或续期 Kubernetes Job。现有 Job 配置到期约为 2026-10-15 16:37:28 UTC；checkpoint 在持久 data10，不在容器临时盘。原先预计50,000步自然结束，现已按用户指令提前停止；不自动重复启动或恢复。

在本机检查：

```sh
ssh ds 'python3 /sciclone/home/jchen57/project/Hand_Reconstruction/scripts/ds_train_twm.py status'
```

`status.json` 是训练进度，`latest_validation.json` 是最新固定验证，`metrics.jsonl` 与 `train.log` 保留原始过程；`latest.pt` 为最近可恢复状态，`best.pt` 按 noisy/missing 恢复 loss 均值选取。数据、权重、缓存和原始日志不提交 Git。技术上可用同一 helper 的 `start --resume`，但本次是用户主动暂停，必须待用户后续恢复指令；不要把stopped视为故障而重启。

## 数据、划分与 99D 编码

选 OakInk2 是为了在已经核过的 MoCap 拟合参考上建立可审计的完整腕与五指状态链，减少首轮 PCA/mean 适配依赖；这不代表它已在同基准上被证明比其它三库更准确，也不改变后续 glove/MoCap 精度优先原则。

- 数据 release revision：`21705616140d726607027e70d58b7837f442ffd8`。官方 split 为 431 train / 39 val / 157 test；seed `20261014` 抽取 64 train 与 8 val 原录制，test 标注未读取。64/8 的选择是开工规模，不是最优超参结论。
- 每个原录制的左右手独立成轨迹，共 128 train / 16 val；204,502 / 18,032 **单手帧状态**，不是相同数量的独立双手时刻。H=30、future offset=3、window stride=4 后得到 50,128 / 4,380 窗口。
- 使用 `raw_mano` 数值 frame ID；官方 mocap 120 Hz，以 `id/120` 建立名义时间，stride 4 得 30 Hz。缺帧不跨 gap 切窗，不以 RGB ID 或 TAR 顺序当时间。不是独立核验后的传感器真实时钟。
- 每手状态为 `world wrist p3 + root rotation6D6 + 15 local finger rotation6D90`；静态 `beta10` 单独条件输入，左/右手用 handedness 标识，共享模型权重。
- 原始 `pose_coeffs` 是 16 个 **wxyz quaternion**，不是 PCA。源 decoder 使用 `use_pca=False, flat_hand_mean=True, center_idx=0`，腕中心化后加 `tsl`（米），因此此源 `tsl` 对应实际 world wrist；该结论不可直接套到其它库。
- 本项目 6D 顺序为旋转矩阵 **第一列 xyz、第二列 xyz**，identity 为 `[1,0,0,0,1,0]`；不是其它库可能使用的行编码。原 quaternion 转矩阵后再编码。
- 所选文件已做有限性、quaternion norm、旋转正交性/determinant、beta 跨录制稳定性检查；manifest 保留录制 ID、源 SHA256、split 与数值误差。官方录制集合、basename/hash 交集已核；不宣称 subject-disjoint。

源语义依据：[OakInk2 固定 decoder](https://github.com/oakink/OakInk2/blob/502a02809b50b7f5d0037f92a1ccd2a479051ad1/src/oakink2_preview/launch/viz/seg_3d.py#L116)、[官方 FPS](https://github.com/oakink/OakInk2/blob/502a02809b50b7f5d0037f92a1ccd2a479051ad1/src/oakink2_toolkit/meta.py#L3)、[manotorch quaternion/center 用法](https://github.com/lixiny/manotorch#usage)。本轮没有 MANO 许可资产，未执行完整 FK/mesh 或 native marker→MANO 精度验收。

## 模型、任务与训练

模型为 821,516 参数的 4 层 causal Transformer，d=128、4 heads、FF=512、dropout=0.1。每帧 16 个解剖 token，当前 wrist xyz 与 root rotation 共同进入 wrist token，其余为 15 个手指旋转。availability/quality 有 17 组：xyz、root rotation、15 finger rotation。以上是本次工程配置，不是用户已经选定的最终网络规模。

历史 H、固定 raw Z 与 candidate S 分开输入，附 beta、handedness 和 dt；不读取 RGB、视觉 features、动作或物体信息。anchor 来自上一帧带小噪声的历史状态，不使用当前 GT 建参考系。缺失 raw 槽位用零/identity 与独立 mask 表示；candidate 缺失槽仅由上一历史状态填充，不用当前目标补造输入。

输出完整 99D 当前恢复状态，并预测 0.1 秒后的完整 99D 作为辅助任务。内部先预测 51D 平移/旋转切向更新，再用 SO(3) 左乘更新并编码回 99D；没有单独 Corrector。这一 pretrain 实现不等于最终视觉校正架构已决定删除 Corrector。首轮没有最终可靠性 q head、图像取证、帧内迭代或自身历史 rollout。

训练样例为 15% clean、40% missing、45% noisy；位置噪声标准差 25 mm，root/finger 轴分量噪声 12°/8°，10% 非 clean 样例放大三倍。missing 还包括独立组缺失、整指和整手缺失。历史在非 clean 样例上来自参考轨迹加 4 mm / 1.5° 小噪声，**没有使用模型自身输出递归生成历史**。quality 提示在 clean/noisy 下同分布，含假高值与 unknown，非 GT 误差或校准概率。

优化器 AdamW，lr=3e-4、weight decay=0.01，warmup=500，之后 cosine 到 0.1×lr，梯度裁剪 1。恢复 loss 为以 0.05 m 归一化的位置 SmoothL1 + 2×root chordal rotation loss + finger chordal rotation 平均；短预测权重 0.1。没有关节位置/mesh loss、关节角硬范围、接触或动力学 loss。

## 固定验证结果：中途 step 7,000

每种模式固定 1,024 个 held-out val 窗口（在全窗口索引上均匀取样）与固定 corruption seed，按窗口样本平均，非完整验证集或 recording 等权统计。baseline 为送入模型的未校正 candidate，missing 的不可用槽位采用上一历史状态。指标无 PA/scale/序列配准：腕为位置欧氏距离 mm，root 与 15 finger 为 SO(3) geodesic degree；参考是发布的 MoCap 拟合 MANO 参数，**不是 joint MPJPE、mesh error 或独立物理真值**。

| 模式 | 腕 baseline → restored（mm） | Root baseline → restored（°） | Finger baseline → restored（°） |
|---|---:|---:|---:|
| noisy | 49.090 → 8.657 | 22.538 → 2.252 | 15.421 → 2.259 |
| missing | 37.637 → 8.402 | 16.426 → 2.187 | 10.910 → 2.266 |
| clean | 0 → 4.551 | ≈0 → 0.932 | ≈0 → 1.642 |

step 7,000 的 best validation 恢复 loss 为 0.017812（noisy/missing 均值）。短预测腕误差 noisy/missing/clean 分别 14.254/13.944/10.890 mm，root 为 3.566/3.515/2.532°，finger 为 2.782/2.768/2.237°，不与缺少 future baseline 的结果宣称预测优越。

更早 step 1,000 noisy 腕为 49.090→48.102 mm、root 22.538→3.359°、finger 15.421→8.919°；clean 被改成 3.724 mm / 1.215° / 5.299°。随后训练继续改善，但准确输入退化仍存在，应保留而非只报告有噪声收益。表格是正在训练中的快照，不是最终 best checkpoint 全集评价；后续结果以远程产物为准。

09:10 UTC 再核 step 17,000 固定验证，noisy 的腕/root/finger 为 7.825 mm / 1.957° / 1.635°；missing 为 7.820 mm / 1.917° / 1.801°；clean 退化为 3.753 mm / 1.184° / 1.073°。所有 baseline 与上表相同；best noisy/missing 恢复 loss 历史最低为 0.013561。step 17,425 时进程仍存活，`best.pt` / `latest.pt` 分别约 9.95 MB，未出现非有限 loss/梯度导致的终止。

## 检查、勘误与下一步

远程 GPU 数值检查通过 8 组：quaternion 符号不变、quat/6D roundtrip（最大误差 4.17e-7）、零/共线 6D fallback、0/π 附近梯度、零残差恒等、完整模型 backward、forecast_dt 不改变 current 输出、空历史 padding。三步真实数据 smoke 完成，无非有限 loss/梯度；本机只做 AST 与 diff 等静态检查。

Smoke 保存在 `runs/twm_oakink2_smoke_20261014`。最初 smoke CLI 误写 code_version `b04de13`，实际代码为 `8593fcb`，原产物保留，并在远程 `provenance_correction.json` 明确勘误；正式 run 由 launcher 自动读取真实 Git HEAD，为 `4cb1da8…`。初稿 quality 在 clean 下恒为 1，存在判断 clean 的捷径风险，已在正式启动前改为 clean/noisy 同分布。没有把该 smoke 当正式验证结果。

Checkpoint 含 model、optimizer、step、配置、代码版本及 corruption/torch/CUDA RNG；保存采用临时文件原子替换，正常 SIGTERM/INT 尝试保存。已有 resume 路径，**尚未通过中断重启实验验证**，且未保存 DataLoader shuffle 游标，不能宣称逐 batch 精确续跑；resume 不自动核验数据/config 一致性。

`best.pt` 的选择没有约束 clean 退化。当前 history 槽位全部有效，没有连续多帧漏检、时间相关偏差或左右手身份错误训练；也没有 SLAM/world 漂移测试。当前 trainer 使用完整99D参考，尚无 partial-GT loss mask，不能直接称已支持 glove/raw-joints 部分监督。

当前按用户要求先核WestWorld架构与参数依据、适配当前恢复任务，暂停原pilot。已保存的checkpoint可供以后对照；不继续追50,000步。后续训练再补MANO FK的真实joint几何指标和合理基线、真实HaWoR误差与自身历史反馈验证。合成噪声收益不能证明实际视觉重建改善、物理正确或跨数据集泛化。其它三库暂未做统一转换/训练，数学模块暂缓。

源码入口：[Oak 预处理](../scripts/prepare_oakink2_twm.py)、[模型](../src/handloop/models/twm.py)、[几何](../src/handloop/geometry.py)、[轨迹切窗](../src/handloop/data/trajectory.py)、[训练](../scripts/train_twm.py)、[数值检查](../scripts/check_twm_numerics.py)、[Pod 启动/状态](../scripts/ds_train_twm.py)、[配置](../config/twm_oakink2_pilot.json)。

## 用户暂停与保存结果：2026-10-14 纽约05:16

用户要求暂停，并澄清更充分参考WestWorld的架构与参数理由，而非完整复刻；输入输出仍适配手部任务。09:16:13 UTC核验目标PID/cmdline/output后发送SIGTERM，进程保存并退出，phase=stopped。CPU只读torch.load验证latest.pt step28,213、best.pt step27,500，二者code_version均为4cb1da8；best noisy/missing loss历史最低0.0086143。用户Pod/Job保留，数据和已有结果保留。远程pause_request.json / pause_verification.json记录动作，不上传原始记录。

最后一次固定验证是step28,000，而非最新checkpoint重新评测：noisy恢复腕5.980mm/root1.724°/finger1.601°，missing为5.903mm/1.689°/1.759°，clean退化3.626mm/0.903°/0.998°。baseline、子集、GT与指标定义同上；不得把这组中途验证标成best.pt评价。
