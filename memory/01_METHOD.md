# HandLoop 方法与接口记忆

**D39当前实现方向（2026-10-16 UTC／纽约10-15）：** 用户明确两个主干均低LR全量微调，替代D35视觉双冻结的首版范围及此前优先固定HaWoR的建议。在线HaWoR patch/raw→2轮candidate-conditioned MANO16投影取证＋数值TWM→末轮最终分布→自身跨帧历史；帧内raw/H/anchor固定、仅最终commit，帧间detach。两主干参数、插件均在可微路径上，图像/相机features仍不进TWM；σ/κ仍非已校准w/q。视觉current NLL与原motion六项回放分开更新，D38 loss改版未提前实施；实际验收状态见[新协议](../docs/VISUAL_JOINT_FINETUNE_2026-10-16.md)。

**2026-10-15 loss对照建议，未替代现行六项：** 已读12篇world model/手部恢复/生成论文，WestWorld以CE为主但Dreamer/TD-MPC2/StableHand/HMP/HaWoR均有不同带权组合。建议先统一conditional probability主线、同架构逐项消融，future起点stopgrad单独验证；NLL＋几何不天然矛盾，尚无证据定位六项为失败原因。任务采样和有效帧归约不能隐藏为“无权重”；现future从确定性current均值出发，不能称完整current–future不确定性传播。DF/条件RFM是第二路线，会改变生成头/采样/可靠性与D35解析NLL接口，尚未实施。D37仍沿原六项；D35视觉阶段单一current NLL不改。[文献与精确建议](../docs/TWM_LOSS_LITERATURE_REVIEW_2026-10-15.md)。

更新日期：2026-10-15 UTC。本文整理现行设计，完整视觉闭环尚待验证；D29已按用户新指令实现当前恢复＋MoE＋较长未来分布并启动纯轨迹训练。主依据：[START_HERE §2–4](../HandLoop_Project/START_HERE.md)、后续用户决定和[讨论 §15.23、§16](../HandLoop_Project/01_Conversation_Research_Archive.md)。

完整叙述见[docs/METHOD_FRAMEWORK.md](../docs/METHOD_FRAMEWORK.md)。本次新增的统一预训练槽位、anchor残差、初始化与单手实例化均为建议契约；用户最新免申请数据偏好已单独登记，不把旧的注册数据开工建议视为当前要求。

## 任务与建模范围

估计当前真实手部状态：

\[
\hat S_t \approx \arg\max_{S_t}p(S_t\mid\text{当前视觉证据},\hat S_{<t},\text{高质量轨迹学得的运动知识}).
\]

第一版仅 wrist 6DoF 与 articulated hand joints；机器人学习是应用动机，下游 retargeting 是可选验证。没有真实人类控制 action 时，TWM 只能学习条件运动统计，不能唯一决定未来动作。

## 状态几何

**2026-10-13用户已选每手99D（D16）**，替代原69D坐标状态候选。[预训练方案](../docs/TWM_PRETRAIN_PLAN.md)给出固定raw、参数/joint mask和混合监督；Oak完整参数pilot已实现，异构部分监督和真实视觉效果尚未验证。

\[
S_t=(p_t^W,R_t^W,Q_{t,1:15}),\quad Q_{t,i}\in SO(3),\quad\beta\in\mathbb R^{10}\text{另存}.
\]

| 分量 | 物理定义 | 建议网络表示 |
|---|---|---|
| 腕部位置 | 世界坐标 3D translation，存储单位米 | 3D |
| 腕部朝向 | proper rotation，属于 SO(3) | rotation6D，6D |
| 手指姿态 | 按MANO/hand mean完整解码后的15个local rotations | 15×6=90D |
| 完整动态状态 | p3＋root6＋finger90，β人物/序列条件 | 99D，用户已选 |
| Corrector更新量 | 平移、root与finger旋转切向量 | 建议3+3+15×3=51D |

99D是编码长度，非99个物理DoF；16个SO(3)旋转各用6D冗余编码。HaWoR原生root/15joint rotation、β10、translation经adapter统一hand mean/腕偏移/坐标后组成99D；视频支持左右手，原infiller为双手输入。[外部核验](05_WRITING_AND_REFERENCES.md)

几何为`J_local=MANO(Q,beta)`腕中心化，再`J_world=p_world+R_world@J_local`。21点/778vertices由99D＋β解码；joint-only/glove观测不用造MANO角度GT，可以几何监督参数输出。wrist-local与时序anchor是不同坐标。

时序建模参考系建议取上一帧最终腕部位姿 `A_t = T̂_(t−1)^W`。对历史和当前候选统一变换，例如：

\[
\bar p_i=R_A^\top(p_i^W-p_A),\quad \bar R_i=R_A^\top R_i^W.
\]

**同一帧全部迭代保持 `A_t` 固定。** 腕局部关节不能又被当成世界点重复转换。首帧无历史的参考系、旋转增量的左/右乘及其坐标系、平移增量在哪个 frame 输出，尚待明确。

建议平移/root切向增量在anchor中，finger切向增量在MANO父关节frame中，各旋转左复合；51D残差需代码/数据验收，见[完整框架§3](../docs/METHOD_FRAMEWORK.md#3-任务输出与统一状态)。99D为单手；两手分别维护状态/历史、共享参数仍为建议，不宣称已实现双手交互。

旋转更新必须通过有效旋转复合，不能直接相加两个 rotation6D 向量。输入 6D rotation 编解码的行/列约定、左右手镜像、骨长和 hand scale 都需要固定，并做 roundtrip 验证。

## 符号与生命周期

| 符号 | 语义 | 帧内是否改变 |
|---|---|---|
| `Z_t` | 原始HaWoR MANO/腕位姿及可用joint数值观测 | 固定并保留 |
| `F_t` | 当前真实图像提取的特征缓存 | 缓存固定，读取位置可变 |
| `Γ_t` | intrinsics、camera pose、crop transform 等 | 固定 |
| `M_t` | 原始检测和各部分可用性 | 原始记录固定 |
| `S_t^(k)` | 第 k 轮当前候选 | 改变 |
| `H_t` | 已完成帧的状态与质量窗口 | 当前帧完成前固定 |
| `w_t^(k)` | 对原观测 `Z_t` 正确性的预测可靠性 | 可重估 |
| `U_t^(k), hW_t^(k)` | TWM 恢复建议和内部状态特征 | 可随候选改变 |
| `q_t` | 最终状态 `Ŝ_t` 的预测质量 | 完成后随状态反馈 |

原始 detector confidence 不是 `w`；mask 不是 confidence；`w` 不是 `q`。这一区分决定了误差监督、输入语义以及反馈能否可信。

## HaWoR 视觉侧

概念接口：`V(I_1:T) → (Z_t, F_t, Γ_t, M_t)`。先冻结视觉权重，复用 hand reconstruction 与 camera/world alignment，建议在 motion infiller 前接入自有后端，将 full HaWoR 与 pre-infiller 版本分别作为 baseline。

实现必须保留三件事：真实图像 feature、插值前 raw detection、相机与 crop 坐标变换。HaWoR 自身投影生成的 `pred_keypoints_2d` 不构成独立图像证据；插值后的有效 bbox 不证明实际检测存在。另据本次源码核验，`hawor_infiller()` 内也执行 SLAM 读取与 `cam2world_convert`，绕开补全网络时须单独保留世界坐标转换。[接口原始清单](../HandLoop_Project/03_References_And_Implementation.md)；[外部源码核验](05_WRITING_AND_REFERENCES.md)。

未来取证可复用缓存特征并训练轻量 evidence head；不要求每轮重跑 backbone。“独立”指证据来自图像，不是用候选自己的几何输出自我证实，不必另建完整视觉 backbone。

## 纯轨迹 TWM

**D29已替代本节早期短预测/Transformer建议：** 当前实现为13.53M、6层d256结构化Mamba＋每层4个dense experts，保持当前99D恢复；新增60步约2秒、4个整段mode的轨迹分布。时间采用物理秒relative time/dt/log1p(dt)＋0.25–8Hz固定Fourier特征。当前与未来分布定义在R³×SO(3)^16：对角Gaussian平移和Haar归一化的isotropic matrix Fisher旋转；未来先跨有效帧累计log density，再对整段mode做logsumexp。未来不作输入，tail/gap mask排除无标签帧，最少3未来帧保留短片段。用户决定方向；时域/专家/模式数及loss权重为工程起点。新目标为restore＋0.05 current NLL＋0.3 future NLL＋0.2 best-trajectory geometry＋0.05 GT velocity＋0.01 MoE balance，仍非力/接触物理定律或校准保证。[公式、源码与验收](../docs/TWM_MOE_LONG_HORIZON_RUN_2026-10-15.md)。下文早期建议保留演进背景。

主要预训练任务：对高质量状态做缺失或扰动，条件于历史恢复当前状态。

\[
\tilde S_t=\operatorname{Corrupt}(S_t^{GT}),\quad
\mathcal L_W=\mathcal L_{restore}+\lambda_f\mathcal L_{short\ forecast}+\lambda_v\mathcal L_{GT\ velocity}.
\]

扰动应覆盖关节缺失、整指缺失、局部错误但 mask 仍有效、连续偏差、历史误差、质量提示失真，以及无需修复的干净输入。噪声分布及强度尚未选定。完全正确的输入用于检验模型是否错误地强加运动先验。

建议结构是 wrist/joint structured tokens + temporal Transformer；起步 `K=8, layers=6, d=256, heads=4`。骨架 token 的确切映射、attention mask、loss 权重、预测辅助时域尚未确定。它们不是论文贡献本身。

99D后默认监督MANO解码几何＋可用root/finger旋转，MANO耦合骨架使bone loss不作必需项。历史K=8为旧建议；v1按真实时间比较0.5/1/2秒，不从不同论文的16至150帧平均出最优长度。GT来源与数据职责见[论文依据](../docs/TWM_DATA_PAPER_EVIDENCE.md)。

部署时的完整概念接口为：

\[
W_\theta(H_t,Z_t,S_t^{(k)},M_t,w_t^{(k)};\Delta t,\beta)\rightarrow(U_t^{(k)},h_t^{W,(k)}).
\]

TWM 可以看到视觉模型输出的低维数值状态，不看到 RGB、图像 feature 或 appearance。候选随 k 变化，所以 TWM 可以提出不同恢复建议；如果只读取固定历史，重复前向不会凭空产生新信息。

**统一接口提案：** 原文预训练四输入与部署五输入不一致。本次建议预训练也保留H/Z/candidate/M/w，raw和不同修复程度candidate由同一轨迹相关构造，包含candidate=raw、干净输入、自身恢复候选与错误高置信提示；U统一为当前完整状态恢复建议。最终q不再混用于预训练提示名称。[完整框架§5](../docs/METHOD_FRAMEWORK.md#5-twm统一的条件当前状态恢复模型)给出方案，尚未实现/验收。

## 帧内迭代

以下是逻辑流程，初始化和缺测 fallback 尚未实现：

1. 从 HaWoR 获得并保留 `Z,F,Γ,M`；有可用视觉估计时，可先试 `S^(0)=Z`，沿用早期归档 §7.2 的初始化建议。正式初始化及缺测策略仍待确定。
2. 将候选世界关节投影到图像/crop/feature 坐标，在 `F` 上读局部证据；同时保留原始观测附近和全手上下文，避免坏候选陷入局部错误区域。
3. 用图像证据校验原观测与候选，重新估计 `w`。高 confidence 可被纠正；偏离运动先验本身不是拒绝视觉的理由。
4. TWM 读取当前候选、原观测、固定历史与质量信息，提出 `U,hW`。
5. Corrector 联合这些信息预测结构化残差：

\[
S_t^{(k+1)}=S_t^{(k)}\oplus C_\psi(S_t^{(k)},Z_t,U_t^{(k)},E_t^{(k)},w_t^{(k)}).
\]

6. 起步建议共享权重迭代 `N=2` 轮，对比 1/2/3 轮；每轮监督真实几何误差。

迭代不是重复使用固定 confidence 加权混合；收益是否来自新增视觉查询，需要独立消融。不能把 `S^(k+1)` 当作下一轮新的原始 `Z`。

## 跨帧反馈

\[
\hat S_t=S_t^{(N)},\quad H_{t+1}=\operatorname{Update}(H_t,\hat S_t,q_t).
\]

每帧只在最终校正后写历史一次。历史窗口固定长度，不额外建立 long/short-memory 子系统。反馈质量提示告诉模型哪些过去估计可能有误，而非把它们视为传感器新测量。

需要实现序列边界 reset、左右手 identity、连续漏检、离开视野再进入的历史处理。上述策略尚未定案。

## 三阶段训练

| 阶段 | 数据与训练对象 | 要回答的问题 |
|---|---|---|
| I | 纯高质量轨迹训练 TWM；主 restoration、辅短预测和结构约束 | 是否学得有用的局部/时序条件恢复能力 |
| II | 同步 video/reference；冻结 HaWoR/TWM，训练 evidence、reliability、Corrector | 能否修复真实视觉错误而不破坏准确部分 |
| III | 模型自己的最终状态递归回灌；建议约 16 帧 unroll，必要时才微调 TWM | 是否抵抗自身误差累积，而非仅 teacher forcing 有效 |

Corrector 的概念损失为 state + image evidence + velocity，各轮有 state 监督。rotation 用 SO(3) 测地误差；velocity 要匹配 GT 速度，不是简单惩罚运动幅度。`w` 和 `q` 按各自对应状态的真实误差监督，GT 只用于训练/评测。误差到可靠性的具体映射、损失权重、截断反传方式均待定。

## 无法由当前设计自动保证的能力

- 完全遮挡时输出可行估计不等于唯一恢复实际动作。
- 无视觉锚点时，局部手运动规律不足以识别任意全局平移或 SLAM gauge 漂移。
- 轨迹 smooth 不代表几何准确，也不代表机器人执行成功。
- 后端递归不证明整个 HaWoR 视频处理链严格 causal 或 real time；需记录 lookahead、chunk、延迟。

详细未决项与验收条件见[进展与下一步](04_STATUS_AND_NEXT_STEPS.md)。

## 2026-10-14 统一TWM直接恢复的架构讨论

用户质疑独立Corrector成为瓶颈，要求评估TWM直接修正HaWoR输出，并通过多样样例和真实前端误差联合微调。助手建议优先验证统一恢复模型；这是讨论方案，尚未最终替代上文架构。[具体方案及前作证据](../docs/TWM_DIRECT_REFINEMENT_PROPOSAL_2026-10-14.md)

TWM可直接返回下一候选完整99D，状态head与融合职责移入同一模型；内部SO(3)残差仍可作为实现方式。先冻结HaWoR验证数值恢复及真实预测/reference适配，保留干净输入和原motion能力检查。删除独立C不自动删除必要的图像信息路径：若保留候选重取证，建议轻量图像头输出2D坐标/创新量、相机与mask/不确定性，由同一TWM更新；仅quality通常缺少校正方向，不能把任意RGB features改名数值证据。

维持Z/S/H、原观测w/最终q、同帧anchor/β固定及仅最终历史反馈。轨迹先验与增广微调能学习运动统计，不能保证力学定律；MANO合法旋转不等于关节限位、物体接触或动力学正确。是否正式删除C、几何输入schema与解冻范围待后续讨论，不能当作用户已采纳决定。

## 2026-10-14 物理约束与数学验证讨论

用户进一步要求参考world model，从数学上证明/验证物理合理性并考虑修改。已整理[条件性命题与验证方案](../docs/TWM_PHYSICAL_GUARANTEES_AND_VALIDATION_2026-10-14.md)：SO(3)合法解码/增量、固定beta的内部FK结构、明确解剖范围映射、anchor的全局SE(3)等变性；可选固定重建目标的下降验收。输出仍99D，硬解码可在统一TWM内部，不要求恢复独立learned Corrector。

关节范围、下降规则与辅助动力学均未采纳/实现。骨长保证限定16个内部FK中心，非全部21点/mesh；旋转6D须处理退化，生理角度不直接截断6D。每轮改quality/测量可能改变评价目标，下降证明须固定目标；代价下降不证明GT误差下降。物理速度沿timestamps计算，不沿帧内迭代。主动手不能套用无输入能量守恒，真实动力学需要有意义的驱动/惯性/外力与独立验证。

## 2026-10-14 D23 首轮代码事实

用户要求先单库跑起pretrain。4cb1da8的纯轨迹TWM是4层/d128/16 tokens/821,516参数，H30、17组mask，beta与左右手条件；raw/candidate/history分开，上一受损历史anchor，内部51D左乘更新输出完整99D，0.1s预测辅助。不是完整Corrector/取证/可靠性/rollout系统，也不等于用户定案删除C。[源码、配置、验证与限制](../docs/TWM_OAKINK2_PILOT_RUN_2026-10-14.md)。

## 2026-10-14 D24 架构参考方向

用户主动暂停tiny pilot，明确不是完整照搬WestWorld，而是参考架构与配置理由，输入输出配合手部当前恢复。已保留checkpoint，不自动恢复/另启。[结构、容量、归一化和训练适配建议](../docs/TWM_WESTWORLD_GUIDED_DESIGN_2026-10-14.md)区分源码事实、论文证据和待验证配置；不把6blocks/d256/4heads或Mamba视为已选最优。

## 2026-10-14 D25 新主干代码事实

用户按参考方案授权重做并跑起。新增StructuredMambaEncoder，6/256/4、同帧关节attention/FFN→逐关节官方Mamba64/4/2，无MoE/action；MANO parent/finger/depth embedding，padding全主干前压缩且无效dt置0，真实漏观测保留时间槽。6330892参数，99D接口/51D内部旋转更新及β条件保持。Mamba特殊初始化及A/D no-weight-decay标志保留；旧flat分支可加载旧权重。完整数值与smoke已通过，PID1276新run启动；完整视觉循环仍未实现。[实测与路径](../docs/TWM_STRUCTURED_MAMBA_RUN_2026-10-14.md)。

## 2026-10-14 当前Mamba数学证明复核

用户重新要求认真证明物理约束与数学性质，讨论恢复；本轮仅源码/官方原文与数学推导，不修改训练主干或运行实验。[详细证明§8–12](../docs/TWM_PHYSICAL_GUARANTEES_AND_VALIDATION_2026-10-14.md#8-2026-10-14当前mamba-twm的证明范围复核)按2b40aa9复核：SO(3)合法性为已实现结构；固定源decoder/β的16内部FK骨长是解码性质；SE(3)等变要求外部anchor规范化与逆变换，完整部署wrapper未验。

条件信息不增加理想最优恢复风险；局部MSE及高斯条件协方差可解释可信部分帮助隐藏部分，但现SmoothL1/chordal有限训练网络不等于MMSE或校准posterior。chordal与平方角误差有常数界，同β内部FK可导出腕/旋转到joint误差上界，不能替代MPJPE。

官方Mamba实际drive为ΔBu；pre-LN＋冻结有限参数可在精确运算下证明内部SSM有界/初始状态遗忘，不能推出完整head/candidate/anchor反馈收缩。允许常数translation bias的漂移反例；K30小增益条件尚未认证。关节限位、迭代下降与真实动力学仍未采纳/实现。

## 2026-10-14（纽约）视觉接入与联合训练复核（历史建议，后续D33调整）

用户询问TWM训练后如何接入视觉baseline与联合训练。新增[可实施协议§10](../docs/TWM_DIRECT_REFINEMENT_PROPOSAL_2026-10-14.md)：先冻结HaWoR/TWM，在pre-infiller数值输出与相同相机链之间验99D adapter及零适配baseline；再用train录制真实预测/reference适配TWM，保留motion replay；随后训练图像evidence/w/q与现行Corrector并选择性微调TWM，加入自身历史rollout；最后单列部分解冻HaWoR head/时序模块。独立C与统一TWM数值测量adapter仍为待比较选择，现checkpoint未有2D/相机观测输入。

重点边界：实际wrist J0/hand mean/列6D/左右手/β约定与投影往返先验收；GT质量/β/历史不进入部署输入。冻结参数与输入detach分开，旧no_grad/CPU inference脚本不可直接反传；解冻模块输出不复用旧缓存。长future训练可回传当前，过去forecast待新视觉到达才有可检验意义；当前TWM已读Z，不能再与同Z似然当独立Bayes相乘，σ/κ也不是最终q。HaWoR非因果上下文须匹配，HOT27/Dex test持续留出。本轮未改代码、现行模块决定或D32训练。

## D33：视觉侧实际读取校正反馈，首版只训练一个最终目标（建议）

用户要求在官方HaWoR权重上加反馈接口，拒绝多项手工带权loss。本轮[§11推荐](../docs/TWM_DIRECT_REFINEMENT_PROPOSAL_2026-10-14.md#11-d33预训练hawor读出侧反馈插件与单一视觉训练目标)：原ViT/IAM/MANO/PAM的16帧双向路径固定并缓存F/Z0；新增PAM之后的小视觉读出插件，逐帧读F、raw和已完成校正历史，产生51D切向残差candidate，再由冻结纯数值TWM输出最终状态及同一次读出的σ/κ。原始Z0不覆盖，当前输出只在下一帧成为历史；不把逐帧变化的反馈直接塞入原双向PAM。仍有原片段lookahead，不宣称在线。

视觉阶段首版只用最终current Gaussian×16个isotropic matrix-Fisher NLL；不再默认叠2D/joint/mesh/velocity/cycle/quality/replay各项。两主干冻结但保留TWM到candidate的输入梯度，先训练插件；若candidate通道梯度/利用不足，再联合少量current读出参数，同一目标。长60步预测保留为预训练能力并单独复测，不宣称冻结后新输入下能力不变。状态NLL要求完整reference可用，不伪造joint-only旋转标签；归一化似然不是无任何统计假设的“自动消除权重”。本段提出时尚未实现；后续D35已搭建该首版并完成接入验收，不改变pure-motion多目标训练，未永久决定删除Corrector对照。

## D35：视觉反馈首版实现与真实接入验收（2026-10-15 UTC）

用户明确要求搭建。代码以[框架文档](../docs/VISUAL_FEEDBACK_FRAMEWORK_2026-10-15.md)为当前实现依据：官方HaWoR冻结原16帧路径/192×1280真实patch、post-PAM raw99D；2层d256/2,458,118参数插件用16个状态query读取图像和已完成history，51D SO(3)残差candidate后交原13,530,032参数TWM最终读出。单一完整state NLL，TWM/HaWoR参数冻结但candidate输入梯度保留；默认clip32/历史30帧、帧间detach，每帧只提交最终状态一次。支持跨块initial/finalhistory、mask/reset/gap、current-only与可选60步分布。最后mean/std/κ同次读出，NLL在anchor内，另出world完整平移covariance；不含anchor/SLAM/epistemic传播或质量q保证。

官方比对、几何与16项核心合同、缺测NaN指标/GT隔离通过。真实Dex source127帧、s0 train/val各一录制，固定单相机、GT crop、nominal frame_id/30；用已停D32 step3110副本完成3→20步及RGB track组合推理，冻结hash不变。val64帧腕95.029→65.048mm同时root4.936→5.856°/finger6.779→8.493°退化，xyz90%coverage0；只证明框架能训练与接入，未证明整体效果、概率校准或future精度。检测/SLAM由caller提供，任意原始视频自动流程尚未打通。D34预训练未改，具体范围与产物见[摘要](../reports/visual_feedback_framework_20261015.json)。
