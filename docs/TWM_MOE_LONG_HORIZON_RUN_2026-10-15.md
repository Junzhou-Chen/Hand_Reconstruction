# D29：当前恢复、MoE 与长轨迹概率预测

日期：2026-10-15（UTC）。用户明确要求修改模型并启动训练：同时恢复当前受损手状态、预测较长未来轨迹、加入 MoE、由实现者选择时间编码，并接入在线 W&B。这里区分用户决定、源码事实、工程起始配置与实际运行结果。

## 决定与替代范围

D29 替代旧模型“仅预测 +3 帧单点、无 MoE、无输出分布”的执行设计；此前不增加 MoE 的默认约束在本次明确授权范围内被替代。保留当前恢复主任务、纯轨迹输入、99D 接口、raw/candidate/history 分离、四库 split 与源标签约定。TWM 仍不接收图像、RGB features 或机器人 action；本次没有实现视觉纠错闭环。

D28 暂停的旧双卡 run 与 checkpoint 保留，不自动恢复。新架构从头初始化，使用独立目录；不是把旧 checkpoint 无声迁移到不同目标。用户提出长预测有助于学习动态、MoE 有助于性能，是本次设计动机，收益须通过实验验证。

| 项目 | 旧四库结构化 Mamba | 本次源码与配置 |
|---|---|---|
| 当前恢复 | 99D，确定性均值 | 保留 99D 均值，增加平移尺度与旋转 concentration |
| 未来 | +3 源帧的一个状态 | 接下来最多 60 个连续源帧，约 2 秒；4 条带概率的完整轨迹 |
| 时序结构 | 6 个关节 Mamba blocks | 保留；每 block 后增加 4 个 dense MLP experts；另有未来 latent Mamba decoder |
| 时间 | relative time、dt、log1p(dt) MLP | 保留三项并增加物理秒的多频 Fourier features |
| 参数 | 6,330,892 | 静态预计 13,530,032；实际远程 parameter sum 待填 |

源码：[模型](../src/handloop/models/probabilistic_twm.py)、[结构编码器](../src/handloop/models/structured_mamba.py)、[未来窗口](../src/handloop/data/forecast_trajectory.py)、[训练](../scripts/train_probabilistic_twm.py)、[配置](../config/twm_four_sources_moe_long_horizon.json)、[启动器](../scripts/ds_train_probabilistic_twm.py)、[W&B helper](../src/handloop/tracking.py)。

## 输入、主干与未来解码

每手状态为 wrist p3＋root rotation6D6＋15 个 finger rotation6D90；beta10 另存。6D 编码采用旋转矩阵前两列逐列展开。16 个 anatomical tokens 顺序为 wrist、index1–3、middle1–3、pinky1–3、ring1–3、thumb1–3；17 个 availability groups 分别对应平移与 16 个旋转。

历史 geometry9＋availability2＋quality2＋quality-known2 投影到 256D；当前 token 联合 raw9、candidate9、双方 availability 各 2、原始 quality2 与 known2，共 26D。joint ID、handedness、parent/finger/depth embedding 与时间信息加入特征。beta conditioning 本轮关闭，源 beta 与 validity 仍保留。

编码器为 6 blocks、d256、同帧 4-head attention、FFN512、官方 Mamba `d_state=64, d_conv=4, expand=2`；dropout0.1。每个 block 依次为 pre-LN 同帧关节 attention residual、pre-LN dense FFN residual、pre-LN 逐关节因果 Mamba residual，再接 dense expert mixture。不存在/补齐的历史帧在编码前移除，真实缺失状态以 availability 表示。每次调用重算窗口，不维护跨调用 SSM cache。

MoE 对每个 token 的归一化时序特征计算 4-way softmax，并加权 4 个独立的 256→512→256 GELU MLP，最后 residual 更新。**所有专家都计算，不是 sparse top-k**。路由不读取 dataset ID 或未来标签；它是否学到有用的动作专门化须检查负载与消融，不能由 gate 权重命名推断。每层分别计算负载均衡正则，防止相反的塌缩层在平均后看似均衡。

当前恢复头输出 p3 与 16×axis-angle3，共 51D 更新；平移缩放0.1m、旋转缩放0.5rad，再左乘 SO(3) 增量。恢复头零初始化，因此初始当前均值等于合法化 candidate。

未来解码先平均池化 16 个当前 joint features，叠加 4 个 mode embeddings 与每个未来物理时间的 encoding，形成 `[B×4,60,256]`。一层因果 latent Mamba＋FFN 解码时间，再通过 64D joint features 生成每关节速度与分布参数；不再为全部 `[B,4,60,16,256]` 建第二个大主干。未来 GT 不进入编码器或 decoder，不使用 future-state teacher forcing。

未来腕平移按 `p_f=p_(f-1)+v_f Δt_f` 累加；16 个旋转逐步左乘 `Exp(ω_f Δt_f)`，实现用归一化 quaternion products 保留不对易增量顺序。平移与 root 使用 noisy-history wrist anchor 坐标，finger 旋转保持局部 parent frame。速度头的0.5m/s、3rad/s是输出缩放，**不是物理速度上限**。

`forecast_modes` 为 `[B,4,60,99]`；`mixture_weights` 为 `[B,4]`，同一 mode 对整段固定。兼容输出 `forecast` 选概率最高 mode 的均值轨迹，不逐帧拼不同 mode；这不等同于求整个混合密度的全局 MAP。

## 物理秒时间编码与未来 mask

实际输入为 `[relative_seconds, dt, log1p(dt), sin(2πνt), cos(2πνt)]`，配置频率 ν 为0.25、0.5、1、2、4、8Hz，经 MLP 投影256D。历史相对当前为负时间，当前为0，未来为正 offset；源实际时间戳优先，不把帧号统一假设为同一 FPS。Fourier 提供多尺度特征，线性时间项避免仅用周期函数表达时间；该频带选择是待验证工程起点。

历史30帧，window stride4，最少保留3个有效未来帧；因此不因强制完整2秒未来而丢掉短 DexYCB clips。60帧是在现有约30Hz准备下的名义2秒，真实 offset 由源时间决定。

未来有效性是连续 prefix，在首个 timestamp gap 或序列边界截断；不跨 gap、不插值、不重复尾帧生成 GT。无效目标填 p0/rotation identity、GT mask false；仅 query 时间按最后有效间隔继续，使60个 query 有限且递增。NLL、几何、速度与指标均使用 `future_valid`。这些补齐 query 不代表新增观测。

loader 在初始化时只读 timestamps 审计每库每步有效窗口数，以及能到达0.1/0.5/1/2秒的窗口数，保存于 `run_metadata.json` 的 `train_forecast_contract` / `val_forecast_contract`。覆盖审计按末次有效 offset≥目标秒数计算；验证选时刻使用最近有效帧，容差为当前 dt 的0.6倍，二者定义不同，须分别报告。

## 输出概率与数学定义

状态分布定义在 `R³ × SO(3)^16`，局部维度51；没有把冗余99D编码当独立 Gaussian。给定均值状态 μ，每个 xyz 分量为独立 Normal，16个旋转分别使用 isotropic matrix Fisher：

\[
q(S\mid\mu,\sigma,\kappa)=\mathcal N(p;\mu_p,\mathrm{diag}(\sigma^2))
\prod_{j=0}^{15}\frac{\exp[-\kappa_j(3-\mathrm{tr}(M_j^\top R_j))]}{Z(\kappa_j)}.
\]

其中旋转以 unit Haar measure 为基准，`3−tr(MᵀR)=||M−R||²_F/2`。均匀旋转的 principal angle θ 密度为 `(1−cosθ)/π`，因此本项目各向同性特例有：

\[
Z(\kappa)=\frac1\pi\int_0^\pi e^{-2\kappa(1-\cos\theta)}(1-\cos\theta)d\theta
=e^{-2\kappa}[I_0(2\kappa)-I_1(2\kappa)]
=\mathrm{i0e}(2\kappa)-\mathrm{i1e}(2\kappa).
\]

代码以 float64 scaled Bessel difference 算 log normalizer，结果回 float32；避免直接指数溢出与高 concentration 下的抵消误差。一般 matrix Fisher 与 NLL 先例见 [Mohlin 等，NeurIPS2020 §3](https://proceedings.neurips.cc/paper/2020/file/33cc2b872dfe481abef0f61af181dfcf-Paper.pdf)；上述各向同性简化是本项目选择，不把原论文关于自由矩阵参数的凸性结论套到整个网络。

xyz σ clamp 至 `[0.001,1]m`，当前初值0.05m、未来0.10m；κ clamp `[0.1,10000]`、初值40。大κ时局部角度单轴 std 约 `1/sqrt(2κ)`，κ本身不是角度 std。输出 scales 是学到的 aleatoric uncertainty，没有 ensemble epistemic 保证，也未完成经验校准。

整段未来使用单一 categorical mode：

\[
p(Y_V\mid X)=\sum_{k=1}^{4}\pi_k(X)\prod_{f\in V}q(Y_f\mid\mu_{kf},\sigma_{kf},\kappa_{kf}),
\quad L_{future}=\frac1B\sum_i\frac{-\log p(Y_{V_i}\mid X_i)}{51|V_i|}.
\]

先按有效帧求和，再对 modes 做 `logsumexp(logsoftmax(logits)+component_log_prob)`，最后除51×有效帧数；不能先平均 component log-prob 再当作同一概率。给定 mode 后的时间/关节误差独立是明确近似，均值则由共享 causal decoder 和积分耦合。NLL 值依赖平移单位、Haar 基准与归一化，允许为负，不与旧回归 loss 直接比较。

## 数据、扰动与训练起始配置

沿用[四库执行协议](TWM_FOUR_SOURCE_RUN_2026-10-14.md)的 prepared collection、decoder provenance 与 recording-disjoint splits：

| 来源 | 保留的训练/验证边界与标签事实 |
|---|---|
| OakInk2 | 官方431 train/39 val；157 test不读取；源 MoCap MANO参考；frame_id/120 stride4 |
| HOT3D | 已下载有效 train Aria/Quest3；排除 HaWoR27评测来源；209/23父录制内部划分；真实ref_timestamp_ns |
| UmeTrack | real train，37/4 subjects；synthetic/test不纳入；frame_id/30名义时间，MANO/native availability分开 |
| DexYCB | 官方s0 800/40/160；固定serial836212060125；不读test labels；拟合MANO，camera frame |

仅完整17组 MANO参考进入本 trainer；native/raw joints 不被补造为99D GT。源解码/FK一致性不等于独立传感器精度；四库不能笼统宣称同等级 glove 真值。源 beta 约定仍有差异，`condition_on_beta=false` 是保留的工程隔离，不是已解决 shape harmonization。

采样为库间等概率、库内窗口均匀、有放回，两 rank 独立 seed；保留旧腐蚀与 anchor：mixed15% clean/40% missing/45% noisy；非clean历史加4mm/1.5°分量噪声，当前25mm/root12°/finger8°，10%非clean整体噪声×3；missing含20%逐组、50%整指及10%整手事件，允许重叠。quality为合成提示并含假高值/未知值，非已校准视觉误差。

总目标为：

\[
L=L_{restore}+0.05L_{current\ NLL}+0.3L_{future\ NLL}
+0.2L_{best\ trajectory\ geometry}+0.05L_{velocity}+0.01L_{MoE\ balance}.
\]

`L_restore` 保留 SmoothL1(p/0.05m,beta0.5)＋2×root chordal＋finger chordal均值。当前 NLL 除51。future geometry先对每条完整有效轨迹求平均，再选最佳 mode，不逐帧选mode。速度监督为相邻 GT 的平移差/Δt 与 SO(3) log差/Δt；按 detached trajectory responsibilities 加权，平移/角速度残差分别以0.5m/s、4rad/s归一化后 SmoothL1。均衡项为每层 `4Σ_e load_e²−1` 再平均。加入几何/速度等目标后，这是多任务优化，不能称纯 MLE。

以上 loss 权重、四个 modes/experts、频带、预测时域均是用户方向下的工程起点，尚未通过消融证明最佳。配置计划50,000步，AdamW lr2e-4、warmup500、cosine至0.1倍峰值、weight decay0.01；Mamba `_no_weight_decay` 参数保持0。BF16网络、FP32几何/损失、梯度clip1与非finite拒绝保留。

配置当前 global batch512 / 每卡256、两GPU；**正式采用的 batch 仍待新模型远程 smoke/显存检查确认**，不得复用旧模型每卡512的成本结论。验证batch32×最多8批/库；每500步验证、100步保存、10步记录。`stop_before_utc=2026-10-15T16:25:00Z` 触发共同保存退出，早于现有Job预期时限；50k是计划，不保证本次 allocation 内完成。

## 评价与 W&B

验证分别报告 noisy/missing/clean；当前比较 candidate、history hold、restored 的 wrist mm、root/finger geodesic degrees。future比较最高概率mode、restored-state hold、历史末两帧常速度外推，按0.1/0.5/1/2秒报告实际 eligible count；这些指标不是 MPJPE，也没有运行真实 HaWoR。

`forecast_ade` 按全部有效帧加权；`forecast_minade_wrist_mm` 先每个窗口对有效帧平均、再在4个完整mode中取最小、最后平均窗口。因此短窗口权重与 ADE 不同，不能混称相同聚合。跨库总表对有该时域目标的来源等权，缺目标不计零；同时保留分库结果。best checkpoint只按 noisy/missing 当前恢复 loss 的来源等权均值选择，clean仍单列，future未参与best score。

当前 xyz 90%椭球 coverage 用三维 Normal Mahalanobis²≤6.251389；未来用各轴混合 CDF 的 `[0.05,0.95]` coverage，后者是边缘90%而非三维联合90%。当前没有旋转 coverage。平移不确定性分别记录 within-mode与total：`Var_total=Σπ[σ²+(μ−Σπμ)²]`，包含mode均值分歧，不能以component std无权均值替代。

W&B配置 `enabled=true, mode=online, require_online=true, project=handloop-twm`，名称`twm-moe-long-horizon-v1`。仅rank0建run/上传标量，entity从已有账号解析，`wandb_run.json` 保存id/url用于resume。记录loss分项、NLL、velocity、MoE load/balance、mode entropy、分库/分时域几何误差、coverage/std、吞吐、学习率、梯度及GPU显存；train/val使用显式optimizer_step横轴，允许同一步分别记录。

不上传数据、MANO资产、checkpoint、源码diff或console；凭据使用正常环境/netrc配置，不进入公开config。在线初始化失败明确报错，不将offline伪称可实时check。checkpoint保存在持久远程目录，保存前对两rank完整模型hash一致性审计；resume要求config、manifest与world size一致。rank RNG保留，但prefetched sampler chunk/loader cursor未保证逐样本精确重放。

## 设计证据与收益边界

- [WestWorld v2 §4.3](https://arxiv.org/html/2603.14392v2)：参考结构化时空分解、MoE与多future queries；本项目不复制机器人action条件，其结果不证明手部增益。
- [MultiPath §3](https://arxiv.org/pdf/1910.05449)：参考整段固定mode的轨迹分布；原方法固定anchors，本项目learned mode embedding，两者训练机制不同。
- [DLow §5](https://arxiv.org/pdf/2003.08386)：人体MoCap采用0.5秒历史→2秒未来，提示多模态与mode averaging问题；不据此宣称手部2秒为最优。
- [SkeletonDiffusion，CVPR2025](https://arxiv.org/html/2501.06035v3)：多样性可能被骨长/jitter伪增益误导；保留几何评价，不能只看平滑/多样性。
- [Time2Vec §4](https://arxiv.org/pdf/1907.05321)：参考线性＋周期时间特征；本项目使用固定频带与dt，不是其learned-frequency实现复刻。
- [DIP §3–4](https://arxiv.org/html/2508.01835v1)：纯运动条件受损恢复已有手部先例，本次保留恢复主任务；不额外加入其依赖contact/物体距离的物理状态标签。

未来意图无观测，长期预测应被视为条件分布而非唯一真值。合法旋转与积分保持几何约束，不证明力/力矩、contact、能量守恒或闭环稳定；scale/NLL可训练不代表已经校准。长预测能否改善当前恢复、MoE能否提升精度、真实视觉校正是否受益，均需相同数据预算与独立GT的消融/闭环评测。

## 实际运行验收（启动后补充）

本节仅由实际远程产物填写，当前不把计划写成完成：

| 项目 | 待实测记录 |
|---|---|
| 训练代码commit、config SHA | 待填 |
| 实际参数sum、GPU/环境与batch | 待填 |
| 概率归一化/梯度/因果/padding/objective mask检查 | 待填 |
| 各库未来覆盖与固定验证样本数 | 待填run_metadata产物 |
| 两卡smoke、checkpoint finite与replica hash | 待填 |
| 新run目录、启动UTC/PID、已完成步数 | 待填 |
| 在线W&B run URL及实际metrics上传 | 待填 |

