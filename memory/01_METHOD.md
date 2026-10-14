# HandLoop 方法与接口记忆

更新日期：2026-10-14。本文整理现行设计，完整视觉闭环尚待验证；D23首轮纯轨迹TWM已有实现与合成恢复训练。主依据：[START_HERE §2–4](../HandLoop_Project/START_HERE.md) 和[讨论 §15.23、§16](../HandLoop_Project/01_Conversation_Research_Archive.md)。

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
