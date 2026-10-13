# HandLoop / Are Gloves Necessary? — 全流程研究讨论档案（ChatGPT Work 项目版）

> **重要说明（真实性）**：这是基于当前聊天可见内容、前次已生成研究整理文件而编制的**内容完整性优先的研究归档和项目移交资料**，包含用户逐轮问题、每轮回应的关键技术讨论、详细公式、方法版本变化和完整 Introduction；**不是 ChatGPT 官方逐字逐句的原始聊天转录，也不保证每条助手回复的全部原文逐字包含**。如果需要具有原始消息顺序、角色与全部字词的严格记录，请在 ChatGPT 的数据导出中取得原始对话后再处理。
>
> 建议新项目优先读本文件第一章 `START_HERE`，所有历史上被替换的技术建议都不再代表现行方案。
>
> 早期画出的概念图位于单独文件 `04_Method_Overview.png`（同时包含于 ZIP），图中的 forecast/prior 细节不代表更新后的 state restoration 设计。

---



---

# 项目材料：项目当前方案及启动说明

# START HERE — Are Gloves Necessary? （ChatGPT 项目启动说明）

> 更新时间：2026-10-13  
> 当前状态：研究构思、方法设计与写作草稿阶段；**尚未完成所述方法的实验验证，也没有 CVPR 录用/投稿结论**。  
> 读者：新的 ChatGPT Work / Project 会话、共同研究者、未来的作者本人。  
> 其他材料：`01_Conversation_Research_Archive.md` 保存完整讨论过程的**内容整理**（不是平台逐字聊天记录）；`02_Introduction_CN_EN.md` 保存论文引言草稿；`03_References_And_Implementation.md` 保存相关工作及实现清单；`04_Method_Overview.png` 为早期概念图，**不代表最新完整方法**。

## 0. 项目一句话定义

**Glove-trained motion knowledge, camera-only state recovery.**

从高精度 glove/MoCap 手部轨迹中**独立**预训练一个低维、结构化的 Trajectory World Model（TWM），学习人手关节结构和短时运动过程；与现有第一视角视觉重建模型 HaWoR 结合，通过**可检验的视觉证据、可靠性重估和当前状态迭代恢复**，校正受到遮挡、深度误差、异常高置信度观测与时序累积偏差影响的手部姿态。完成校正的状态反馈给下一时刻，实现持续状态估计。**推理端只需头戴 RGB camera，不需 glove**。

核心：**纠正当前实际状态，不是预测遥远未来，也不以生成看起来合理的手为目标**。

## 1. 已经明确确定的设计（尽量不要重新争论）

| 决策 | 当前结论 |
|---|---|
| 实际采集与推理 | **Head-mounted / egocentric RGB camera**，第一版不依赖深度传感器、数据手套或外部 MoCap；图像分辨率、具体设备尚未确定 |
| 视觉前端 | **HaWoR，CVPR 2025**；先冻结权重，复用 camera-space hand reconstruction + camera SLAM/world alignment，保留视觉 feature 与原始检测信息 |
| TWM 训练 | **只用 glove / MoCap / 高质量传感器轨迹**，不看图像、不直接用 WestWorld 权重、不从机器人真实 control action 学习 |
| WestWorld 作用 | 仅借鉴 structured/kinematic embeddings、trajectory model 等思想；**不直接复刻 action conditioning、robot-system MoE、256-bin forecasting、MPPI** |
| TWM 核心能力 | **Corrupted/incomplete current-state restoration**；短时预测仅作辅助训练任务，提高对运动过程的学习 |
| Corrector 核心 | 当前可信观测帮助恢复不可信部分；即使高 confidence 也可被重新检验；不允许 motion prior 无条件覆盖突发真实动作 |
| 交互机制 | **帧内 candidate-conditioned visual re-observation / state refinement + 跨帧最终状态回灌** |
| 方法边界 | 第一版仅 wrist 6DoF + hand articulated joints；不引入 full-arm、object physics、contact-graph、MoE、long-term memory、robot planning |
| 目标验证 | 精确 3D/world-space wrist+joints；部分遮挡、false-high-confidence、动作突变、再可见后的恢复、跨数据集稳定性 |
| 投稿定位 | 计划尝试 CVPR，但不是已完成/已投稿/已录用；必须正面与 StableHand、HMP、PAD-Hand 等比较 |

**不要采用已被讨论排除的旧版本**：RGB-D 优先路线、以 EgoInfinity 为最终已定视觉主干、WiLoR 单帧作为唯一主干、只预测未来的 world model、固定 confidence 加权融合、直接使用 WestWorld robot checkpoint、不可更改的视觉高置信关节，都只是历史阶段的方案。

## 2. 科学问题与假设

### 科学问题

给定部分准确、部分缺失、部分带错误的视觉重建状态，能否从高质量轨迹中学到**不依赖具体成像设备**的状态关系，让当前可靠的局部观测帮助恢复完整手，并在持续递归估计中减少误差？

更精确：

\[
\hat S_t=\arg\max_{S_t} p(S_t \mid \text{visual evidence at }t,\ \hat S_{<t},\ \text{glove-trained motion knowledge}).
\]

不要承诺对于长期全遮挡、完全不可观测的指尖或纯粹全局 SLAM gauge drift 能唯一恢复 GT。

### 状态表示（建议方案，待代码验证）

单手：

\[
S_t=(p_t^W,R_t^W,J_t^{local})
\]

- \(p_t^W\)：world-space wrist translation，3D；
- \(R_t^W\)：wrist rotation，SO(3)，编码时用 6D representation；
- \(J_t^{local}\)：腕部局部坐标中的 20 个非根 hand joints，60D；
- 网络输入状态可表示为 **69D**；预测更新 \(3D\ translation+3D\ rotation\ increment+60D\ joints=\) **66D**。
- 腕部旋转应用旋转复合（Exp/Log），**不能直接将两个 6D rotation vectors 相加**。
- 左右手共享部分参数但需 handedness 标记；hand scale、bone lengths、joint conventions、camera/world transform 必须规范化。

从上一时刻腕部构建当前帧固定的局部参考坐标系 \(A_t=\hat T_{t-1}^W\)，将历史与候选转换到相同参考系后做时序建模。**同一帧所有迭代都使用同一个固定参考**。

## 3. 模块及明确接口

### 3.1 视觉侧：HaWoR

输入：头戴视频 \(I_{1:T}\)（HaWoR 自身可利用时序上下文，因此不能冒然宣称整个系统严格 causal / online）。

输出：

\[
\mathcal V(I)\rightarrow(Z_t,F_t,\Gamma_t,M_t)
\]

- \(Z_t\)：原始 3D wrist+hand estimate，所有 refinement 轮次保留；
- \(F_t\)：从真实图像提取、可重复局部采样的视觉特征；
- \(\Gamma_t\)：intrinsics, camera pose, crop transforms 等；
- \(M_t\)：**原始检测**和每部分观测有效性（不能把插值后的 bounding-box `valid` 误当成真正检测）。

**重要技术陷阱**：HaWoR 某些 2D keypoints 来自已有 3D 预测投影，不能拿其与本预测投影比较，称为独立图像证据。需要独立 image-feature/keypoint evidence head（可放在 Corrector 内）。

第一版建议用 HaWoR 视觉重建 + SLAM，但**不要**用已经通过 HaWoR motion infiller 合成的状态充当“视觉真值”；HaWoR full pipeline/infiller 应作为 baseline 单独比较。

### 3.2 纯轨迹 TWM

预训练输入是 glove/MoCap 轨迹，不需要 paired video：

\[
\widetilde S_t=\operatorname{Corrupt}(S_t^{GT}),\quad
W_\theta(\mathcal H_t,\widetilde S_t,M_t,q_t)
\rightarrow U_t\approx S_t^{GT}.
\]

Corruption：关节缺失、整根手指遮蔽、局部错位但标记仍有效、连续时序偏差、历史误差、完全正确无需修复。质量信息本身也可能不可靠，应做扰动。

辅助短期预测：

\[
\mathcal L_W=
\mathcal L_{\rm restore}
+\lambda_f\mathcal L_{\rm short\ forecast}
+\lambda_b\mathcal L_{\rm bone}.
\]

TWM 第一版推荐：structured joint/wrist tokens、temporal Transformer（起始 6 layers、d=256、4 heads、K=8 帧），不要起步就引入多专家模型和长时预测。

推理／校正阶段：

\[
W_\theta(\mathcal H_t,Z_t,S_t^{(k)},M_t,w_t^{(k)})\rightarrow(U_t^{(k)},h_t^{W,(k)}).
\]

**注意：TWM 看到来自 HaWoR 的低维状态，但绝不看到图像或 RGB features。** 模型的预训练与部署仍基于状态空间。

### 3.3 迭代校正器

当前 frame 的 \(Z_t\) 固定；候选 \(S_t^{(k)}\) 随轮次变；历史 \(\mathcal H_t\) 在当前 frame 完成前保持不变。

1. 将 \(S_t^{(k)}\) 的 3D joints 按 \(\Gamma_t\) 投影，读取 \(F_t\) 上局部 image evidence，且保留原始观测附近和全手上下文，避免错位 candidate 陷入错误局部搜索。
2. 校验原始观测及候选相对图像是否可信；**高视觉 confidence 仅是软锚点，不能硬锁死**。
3. TWM 以 current candidate 和历史的结构化状态提出恢复建议。
4. Corrector 联合视觉证据、原始观测、TWM 提示，预测结构化状态残差 \( \Delta S_t^{(k)}\)：
   \[
   S_t^{(k+1)}=S_t^{(k)}\oplus\Delta S_t^{(k)}.
   \]
5. 推荐第一次试验 **N=2 轮**，与 N=1、N=3 消融；各轮共享权重。
6. 视觉量和先验量不能只靠固定 gate 合并，因为二者可能同时不准。

### 3.4 可靠性（不引入冗长子系统）

区分：
- \(w_{t,j}\)：**原始视觉状态** \(Z_{t,j}\) 的预测可靠性，可迭代重评；
- \(q_{t,j}\)：**当前最终状态** \(\hat S_{t,j}\) 的预测可靠性，作为反馈质量提示。

二者误差分别用 paired video+GT 学习，推理不得使用真实 GT。质量提示可以辅助但不能成为唯一依据，尤其遇到 false-high-confidence 和真实动作突变。

### 3.5 跨帧反馈

\[
\hat S_t=S_t^{(N)},\quad
\mathcal H_{t+1}=\operatorname{Update}(\mathcal H_t,\hat S_t,q_t).
\]

反馈只在当前帧最终校正后发生一次；带质量标签而非把所有预测等同于新测量。历史长度 K 固定即可，**不要额外建 long/short memory**。

## 4. 建议训练过程（3 stages）

1. **Trajectory-only TWM pretraining**：数据仅需高质量 hand-state trajectories。主任务 masked/corrupted state restoration；辅助 short prediction / motion consistency；冻结后可复用于不同视觉重建器。
2. **Paired visual-correction training**：从同步 video + reference state 计算真实 HaWoR errors，冻结 HaWoR/TWM，训练候选状态 evidence head、可靠性预测和残差 Corrector。对每轮输出监督真实几何误差；区别人工噪声与真实视觉错误。
3. **Closed-loop rollout training**：逐步使用自己的 \(\hat S_{t-1}\) 做下一时刻历史（如 16 帧 unroll），使 Corrector 适应预测误差反馈；必要时才微调 TWM。保持 camera feature cache，注意长链反向传播的开销。

一组概念损失：

\[
\mathcal L_C=\mathcal L_{\rm state}+\lambda_e\mathcal L_{\rm image\ evidence}+\lambda_m\mathcal L_{\rm velocity}
\]

不要只罚运动幅度，防止真实快动作被抹平。rotation 用测地距离。

## 5. 数据与评测

### 训练数据池

- **Motion-only**：可使用只有 glove/MoCap 手部关节的轨迹；可缺视频；校准 joint order/handedness/scales、时间戳及采样率。
- **Video–GT paired**：用于训练 Corrector；HOT3D 等有头戴视频及外部高精度状态标注的资源；DexCap 有 glove 数据与 RGB-D，但其腕部世界轨迹包含 SLAM 等来源的误差，不能当作绝对无误差参考。
- EgoDex 等设备视觉追踪数据不能无条件混称为独立 glove/MoCap GT；可作辅助 pretraining、并做质量分层。
- 数据使用与许可、未公开 test GT 和 GT contamination 等需要在训练前核对。

### 核心量化指标

world-space MPJPE、camera-space hand MPJPE、local hand-joint error、wrist position (mm)、wrist rotation (degree)、velocity error、occlusion-conditioned error、恢复到可见后的 error decay、高 confidence 错误样本、跨场景/跨被试/跨数据集泛化、推理时间。

**必须同时间戳、同坐标系、同手骨架标注**；PA-MPJPE 无法替代真实 world-space accuracy。

### 必须做的消融

- HaWoR full / HaWoR minus infiller + original estimates；
- simple smoothing / Kalman / GRU / vanilla temporal model；
- prediction-only TWM vs restoration TWM；
- no current observed joints vs conditional recovery；
- confidence gate vs jointly learned reliability；
- one-shot vs repeated correction；
- with/without independent state-guided visual re-observation；
- with/without cross-frame feedback；with/without rollout training；
- OOD / occluded / sudden motion / drift-recovery diagnostic。

## 6. 风险与尚未解决的问题

1. **创新性**：HMP 已将 motion prior 用于 hand pose，StableHand 强调 quality-aware motion refinement，PAD-Hand 等也讨论不确定性；**单纯 HaWoR+TWM+gate 很可能不足以冲 CVPR**。可能的真实技术贡献：current partial-state-conditioned restoration + evidence-guided iterative re-observation + reliability-aware temporal feedback 的不可替代收益，实验必须逐一验证。
2. **TWM 无真实控制 action**：仅学习 motion statistics/kinematic regularities，不能预测任意未来意图、接触力或肌肉动力学。
3. **完全遮挡**：真实状态不可完全识别；只可输出合理估计与不确定性。
4. **SLAM 全局漂移**：运动先验无法唯一识别整体世界平移。需要 visual geometry/world anchors，不能将所有漂移都归功 TWM。
5. **视觉特征接口**：需要动 HaWoR 推理 hook 或使用额外轻量 image evidence head；不能用同源投影的 2D keypoints 自我确认。
6. **可复现性**：HaWoR 原代码、MANO license、相机内参、原始 bbox/mask、时序 chunk、FPS 必须记录。
7. **训练数据 GT**：大规模并不等于高精度，需核验 glove joint vs wrist global pose 的来源。
8. **论文定位**：只报告真实做出的实验，不将预期假设或示例误差写成结果。

## 7. 最小可行的开工顺序（不要同时攻克所有难题）

**阶段 0：统一数据**  
抽取一份 HOT3D 可用的头戴 RGB/GT 序列，跑 HaWoR；导出 \(Z_t\)、原始 detection mask、timestamp、相机内外参和 canonical GT。验证投影与坐标单位。

**阶段 1：纯轨迹 TWM**  
建立 69D joint/wrist canonicalizer，先训练 corrupted current-state restoration；检查遮挡、错位和无噪声样本的性能，与 last-value、linear interpolation、GRU 基线比较。

**阶段 2：最简单真实视觉校正**  
冻结 HaWoR 和 TWM，只用 state \(Z_t\)+修复提议+误差提示训练一次融合，先证明真实 GT 误差可下降（不是更平滑）。

**阶段 3：有效的帧内迭代**  
hook HaWoR feature 或训练独立二维证据小头；让候选状态改变下一轮读取位置，做 1/2/3 轮消融。

**阶段 4：跨帧反馈**  
反馈最终状态和质量提示，16 帧左右 unroll，验证长遮挡和重新可见的误差恢复。

**阶段 5：跨域及强基线**  
HMP、StableHand、PAD-Hand、HaWoR infiller 等同条件比较；至少一个独立主体/场景 OOD，视精力再添加下游机器人 retargeting。

## 8. 当前论文标题与写作定位

工作标题候选：
- **Are Gloves Necessary? Learning Motion Priors for Visual Hand Correction**
- **Are Gloves Necessary? Learning to Correct Visual Hand Reconstruction with Motion Priors**
- **Are Gloves Necessary? Learning to Recover Hand Motion from Imperfect Vision**

问号是可接受的 CVPR 风格；训练阶段仍用 glove，因此必须解释“no glove”仅指部署阶段。

建议 Method 目录：

3.1 Physical State Representation and Visual Observations  
3.2 Trajectory-Only Conditional State Restoration  
3.3 Evidence-Guided Iterative State Correction  
3.4 Reliability-Aware Temporal Feedback and Training

**CVPR 定位**：只有高置信错误纠正、遮挡恢复、state-driven re-observation 与 feedback 的扎实提升，才能证明不是 Kalman+motion prior 的已有简单组合。

## 9. 可直接复制到新 Work 项目的启动提示词

> 我在研究一个暂定标题为 “Are Gloves Necessary? Learning Motion Priors for Visual Hand Correction” 的 CVPR-oriented 研究项目。请首先阅读本项目全部材料，尤其 `START_HERE.md` 中的“已确定决策”和 `01_Conversation_Research_Archive.md` 的讨论纠偏。技术路线为头戴 RGB + 冻结 HaWoR + 只用 glove/MoCap 轨迹独立训练的低维状态 TWM + evidence-guided iterative Corrector + reliability-aware cross-frame feedback。核心任务是修复当前视觉状态，不是未来轨迹生成。不要建议重新换回 WiLoR/EgoInfinity，也不要引入 MoE、长短记忆、物体物理引擎或额外机器人控制模块。请从最小可行的代码工程入手：先为 HaWoR 与运动 GT 定义一致的状态/时间/坐标接口，再实现 pure-trajectory conditional-restoration baseline，并用真实 HaWoR errors 检查是否优于 smoothing。对任何数据集可用性、开源代码接口、论文创新性与实际指标，请核查资料，不要将我们的预期写成已经验证的结果。当前首个交付目标：具体的项目仓库目录、数据接口 schema、TWM PyTorch 实现计划、初始训练/评测脚本，以及每一步的 sanity checks。

## 10. 阅读提醒

**`01_Conversation_Research_Archive.md` 是对话内容的顺序整理，并非平台原始逐字聊天转录。** 同一个研究方案在历史上经历多次修正，存在相互冲突的早期建议；发生冲突时，以本 `START_HERE.md` 的当前决策和“最新完整方法框架”为准。不要将未复核的年份/量化指标用作正式论文引用。



---

# 项目材料：研究对话内容归档（前后完整演变）

# Camera-only × Glove-trained Trajectory World Model — 全部研究讨论的顺序内容归档

> 归档整理日期：2026-10-13。本文汇总从早期文献调研到最新闭环方法框架的全部**关键科研讨论内容与重要用户问题**，不等于 ChatGPT 平台逐字逐句原始聊天导出。  
> 前半部分保留第一次导出的历史版记录；下半部分补入其后的每轮主要交流和最终纠偏。  
> **新 ChatGPT Work 项目请先读 `START_HERE.md`，并将其定为唯一现行方法决策；本长文件包含已放弃的历史提案，切勿把历史建议误认成最终方案。**  
> 引言草稿在 `02_Introduction_CN_EN.md`，方法图在 `04_Method_Overview.png`，实现/参考在 `03_References_And_Implementation.md`。

---

# Camera-only Physical State Estimation × Trajectory World Model

> **对话完整内容整理（按讨论顺序）**  
> 整理日期：2026-10-08  
> 主题：利用 glove / MoCap / 专业数采设备的高质量状态监督，构建基于相机观测与 Trajectory World Model 的闭环状态估计系统，将 camera-only 轨迹尽可能校正到传感器真值精度。
>
> **说明**：本文保留本轮对话中全部核心讨论、用户提出的问题与重要纠偏、代表工作、方法定义、关键公式、设计选择及实验方案；为了方便后续研究和写作，将重复的回答合并重组，**并非逐字逐句的聊天导出**。论文指标和研究状态来自讨论过程中的查阅记录，正式引用、年份及数值应在写论文前再次核对原文。

---

# 一、原始研究动机：Camera-only 能否取代手套数采？

## 1.1 用户最初的问题

用户观察到机器人公司存在两种主要数据采集方案：

1. **穿戴式数采**：视觉眼镜 + 数据采集手套 / 遥操作手套，利用传感器或机器人自身编码器记录手、机械臂和机器人动作。
2. **Camera-based 数采**：直接拍摄人的操作过程，依靠视觉定位与三维重建恢复动作状态，无需佩戴遥操作手套。

最关心的问题是：相机路线最新、最强的方法是什么？与戴手套相比误差多大？工业和学术界如何克服这些精度差异？

## 1.2 第一轮文献调研的认识

讨论把相关技术划为四类：

| 路线 | 输入 | 主要输出 | 核心限制 |
|---|---|---|---|
| Wearable glove / MoCap | 传感器、tracker | 手部关节、手腕位姿、可能有触觉信息 | 设备成本、穿戴门槛、规模化难度 |
| Camera → Hand Pose | RGB / RGB-D | 关键点、MANO、手部 3D pose | 遮挡、尺度、世界坐标漂移 |
| Camera → 4D HOI → Robot | 视频 + 几何/接触 | 交互轨迹、可重定向机器人动作 | HOI reconstruction、接触/物理修正复杂 |
| Camera → Pseudo-action → VLA | 普通人类视频 | 相机坐标动作、伪标签 | 状态精度和机器人可执行性不直接保证 |

**关键区分**：

- *相对手部姿态误差*与*世界坐标绝对位置误差*不是一回事。
- *人手关键点误差*与*机器人训练数据质量/任务成功率*也不是一回事。
- 如果直接遥操目标机器人，机器人 encoder/控制日志提供的 action label 在定义上远比 camera 反推可靠。
- 普通 RGB 视频可能无法观测被物体完全遮住的手指、接触力和主动施力状态；模型可以做合理推断，但不保证唯一恢复真实状态。

举例：先前讨论引用 HaWoR 中对齐后 PA-MPJPE 与 world-space W-MPJPE 的量级差异，用来说明“手的相对结构很准”不等于“世界坐标可直接用于机器人”。讨论还引用了 camera-based joint-angle 与 CyberGlove 等手套测量比较，指出不同指标不能直接算统一倍数。

## 1.3 与普通 pose estimator 相比，近期 pipeline 的演变

越来越多系统采用：

```text
Human RGB / RGB-D video
       ↓
Depth / metric geometry / camera pose
       ↓
3D hand + object reconstruction
       ↓
Temporal alignment / tracking
       ↓
Hand-object interaction refinement
       ↓
Physical trajectory / robot retargeting
       ↓
Simulation verification / robot data
```

代表性工作（主要作为方法参考，而非所有工作都采用完全相同流程）：

- **Do as I Do**：普通 monocular RGB video → 手/物体 4D reconstruction → contact/physics-aware correction → 灵巧手机器人可执行轨迹。早期调研还讨论了 reconstructed visual reference 与更干净 MoCap reference 在相同 downstream retargeting 下的任务成功率差异。
- **RoboWheel**：RGB/RGB-D → hand-object interaction state → 物理约束/残差修正 → 不同 embodiment 的动作重定向。
- **VideoManip**：普通 RGB → hand/object trajectory → interaction/contact refinement → dexterous manipulation。
- **Human2Sim2Robot / OKAMI**：侧重 object-centric motion 和仿真/机器人迁移，而不是只恢复人体手指关节。
- **EgoInfinity**：metric geometry + hand reconstruction + occlusion handling + interaction-aware refinement。
- **C2Dex**：object-side contact consistency，使 noisy hand reconstruction 更适合机器人接触动作。
- **Phantom / DXTR**：RGB-D / 视觉+定位设备收集人类操作，再映射到 robot-compatible action。
- **UMI / FastUMI**：介于遥操作与纯相机之间的折中方案，使用手持 gripper 和定位/视觉辅助，减小 human-to-robot action 映射歧义。
- **HuRo / ACE-Ego-0 / EgoScale / IronMind**：人类视频的大规模动作伪标注、camera-space action pretraining，强调规模化而非每帧严格追求测量精度。

讨论总结：追求数采规模时，camera-only 有巨大优势；追求直接准确执行的机器人 action，设备采集仍有优势。现实中可以用少量高保真设备数据教会模型，再处理更大量纯视觉数据。

---

# 二、研究想法第一次形成：Glove GT 监督 Camera + State World Model

## 2.1 用户提出的原始 idea

用户提出：大量设备（包括数采手套、眼镜、MoCap 等）能提供高质量时序状态；另有普通 camera 视频。可以把**高质量设备动作作为 ground truth**，与 video 对齐，然后训练一个模型。相比已有每帧视觉估计 + 后处理的 pipeline，希望引入一个持续维护手臂、手部和物体空间关系的 **trajectory/state world model**，用过去状态辅助当前估计，在遮挡时也能继续追踪。

关键点：

- World model 不应是主要生成视觉像素的 video world model。
- World model 在**低维/浅层的结构化物理状态空间**中演化。
- 视觉提供 measurement，world model 提供 motion/state prior。
- 当前状态经过共同校正，再作为模型的下一次输入，形成闭环。
- 最终目标是 camera-only 预测接近 glove/MoCap ground truth。

## 2.2 最初的数学表述

令：

\[
S_t=\text{真实但不完全可观测的物理状态}, \quad
O_t=\text{camera observation}, \quad
G_t=\text{training-only sensor state}.
\]

Visual Observer 输出：

\[
V_t=V_\phi(O_t).
\]

Trajectory World Model 根据过去状态给预测：

\[
P_t=W_\theta(\hat S_{t-K:t-1}).
\]

校正后：

\[
\hat S_t=F_\psi(V_t,P_t).
\]

随后：

\[
\hat S_t \rightarrow W_\theta \rightarrow P_{t+1}.
\]

核心是**递归状态估计**，不是普通视频生成或单次手部姿态回归。

## 2.3 第一轮方案一度延伸的内容

早期讨论曾引入：

- hand/object/contact/scene/camera 组成的较大 interaction graph；
- 隐变量 belief state、uncertainty-aware fusion；
- 接触力/触觉、contact graph、physics refinement；
- causal filter 与 offline smoothing 两种版本；
- long-term interaction memory 与 short-term motion；
- 多阶段 foundation training、大规模 cross-embodiment adapter；
- latent motion/action token 以及 VLA downstream。

这些具有研究价值，但**用户随后多次强调：第一篇工作的 pipeline 不要泛化到什么都做，也不要被过度防御性讨论拖成大工程**。后续形成的最终方案明显收缩，不把上述扩展都作为必须实现的组件。

---

# 三、WestWorld 带来的启发与明确的边界

用户随后提到 **WestWorld**：它通过 Trajectory World Model 学习状态/动作的动态演化并进行规划。用户认为可以借鉴它在 trajectory space 学动态关系的思想，再用这个模型持续校正视觉估计的真实状态，期望 cross-domain / cross-benchmark 泛化。

讨论的借鉴点：

- 世界模型不一定要预测视频，可直接在 state/trajectory domain 学习。
- 同一类轨迹结构可能跨不同传感器、不同场景和不同运动过程迁移。
- 可以先独立预训练 trajectory prior，再接视觉分支。

**但用户明确指出：不要因为提到了 WestWorld 就将方法转成动作条件规划问题；它只是思想参考。**

区别：

\[
\text{规划型 TWM}:\quad p(S_{t+1}\mid S_t,A_t)
\]

\[
\text{当前任务}:\quad p(S_t\mid O_{1:t})
\]

对于纯被动人类视频通常没有真实 human motor command，因此第一版方案建议用**过去已校正 physical state**预测下一时刻 state/增量，不强行引入不可观测 action。

最终不把规划、MPPI、VLA control 放进核心主图；如要验证可执行性，可以使用现有 retargeter 作 downstream。

---

# 四、最终收敛：合理而简洁的 Closed-Loop Pipeline

## 4.1 用户的关键纠偏

用户明确提出：

1. 不要过度关注 long-term/short-term memory，状态关系转换的准确性才是重点。
2. 不要把所有可能因素都塞进一篇论文；pipeline 应当合理、能实现。
3. 最初画出的模块更像单向数据流水线，实际应该是 **loop**。
4. Visual 与 World Model 会相互校验；最终预测的 state 会反馈给 World Model，用于下一步预测。
5. 研究重点是“提出并实现有效闭环系统”，而不是为了概念完整而无限增加模块。

讨论据此统一为三个主要模块：

\[
\boxed{\text{Visual Observer} + \text{Trajectory World Model} + \text{Iterative State Corrector}}
\]

## 4.2 两个必要的 loop

**Loop A：当前时刻的 iterative state correction。**

- Camera 产生 visual measurement \(V_t\)。
- TWM 产生 trajectory-based prior \(P_t\)。
- 当前 candidate \(S_t^k\) 分别与 visual evidence 和 trajectory prior 比较。
- Corrector 产生小的 residual update \(\Delta S_t^k\)。
- 仅迭代 2–3 次，而不是无限展开。

\[
S_t^{k+1}=S_t^k\oplus\Delta S_t^k.
\]

**Loop B：时间上的 corrected-state feedback。**

\[
\hat S_t\rightarrow \text{corrected history}\rightarrow W_\theta\rightarrow P_{t+1}.
\]

注意：Loop A 是单个时间步内部的校正；Loop B 是随视频时间推进的递归状态估计。**不是 long-/short-memory 双分支。**

## 4.3 推荐的主体结构

```text
                  Camera RGB / RGB-D I_t
                           │
                           ▼
                  Visual Reconstruction
                      /         \
             metric geometry    hand pose
                      \         /
                       ▼       ▼
                  Visual State V_t
                         │
                         ▼
                  Initial State S_t^0
                         │
                         │      Corrected History
                         │          S_(t-K:t-1)
                         │                 │
                         │                 ▼
                         │         Trajectory World Model
                         │                 │
                         │            Prior P_t
                         │                 │
                         ▼                 ▼
                  ┌────────────────────────────┐
                  │  Iterative State Corrector │
                  │                            │
                  │  visual consistency       │
                  │  trajectory consistency   │
                  │             ↓              │
                  │  S_t^k → ΔS_t^k → S_t^(k+1)│
                  │               ↖____________│
                  │          2 iterations      │
                  └─────────────┬──────────────┘
                                │
                                ▼
                        Corrected S_t
                           /       \
                  Recovered         feedback to
                  trajectory        history / TWM

       TRAINING ONLY: glove / MoCap GT supervises
       TWM predictions and corrected state.
       INFERENCE: camera-based observations only.
```

## 4.4 State 设计：适度简单、与设备无关

建议第一版单手：

\[
S_t=[p_t^{wrist},R_t^{wrist},J_t^{rel}].
\]

- wrist translation \(p\in\mathbb R^3\)；
- wrist rotation 6D representation；
- 21 个 wrist-relative 3D joints，\(21\times 3\)；
- 总计 **72D/手**（3+6+63；是表示法选择，不是所有数据集自带的格式）。

双手可扩展为 144D；如果实验数据拥有可靠 object state、且证明有增益，可再加入 object 6DoF。**第一阶段不要强制加入 contact graph、tactile force 或复杂 scene graph。**

之所以不把 MANO 作为必需的统一空间：不同 glove、MoCap、视觉模型内部参数不兼容，而 canonical wrist + joints 更容易对齐。MANO 仍可用于几何恢复与可视化。

注意：wrist rotation/translation、world/camera 坐标之间必须用统一标定；直接比较不同坐标系或不同 skeleton conventions 的数字没有意义。

---

# 五、World Model 具体如何构造？

## 5.1 最初版本：Causal Trajectory Transformer

不是大规模视频 diffusion，不必一开始上 MoE、复杂 memory、planning。

**输入**：

\[
\hat S_{t-K},\ldots,\hat S_{t-1}.
\]

这里用的是前面闭环预测出的 **corrected states**，不是 raw camera 输出，也不是部署时不可能获得的 GT。

## 5.2 局部状态标准化

为了避免学到无意义的绝对房间坐标，对历史姿态做局部 canonicalization，例如以 \(t-1\) 时刻 wrist 坐标作为参考：

\[
\bar p_i=R_{t-1}^{-1}(p_i-p_{t-1}),\qquad
\bar R_i=R_{t-1}^{-1}R_i.
\]

可拼接局部 state 和 motion increment：

\[
x_i=[\bar S_i,\Delta\bar S_i].
\]

通过 MLP → state token embedding → causal Transformer。

## 5.3 推荐第一版超参数（待实验验证）

- History window \(K=8\)（也比较 4/16）。
- State embedding \(d=256\) 或 512。
- 6–8 层 causal Transformer 起步。
- 最初每一时刻一个 state token；后面才考虑细分为关节 token。
- 预测增量 \(\widehat{\Delta S_t}\) 而不是直接从零回归绝对位置。

\[
\widehat{\Delta S_t}=W_\theta(\hat S_{t-K:t-1}).
\]

\[
P_t=\hat S_{t-1}\oplus\widehat{\Delta S_t}.
\]

其中 \(\oplus\) 是状态的有效组合操作；旋转应使用 SO(3) composition，而不是直接把 rotation 6D 数组逐项相加。

## 5.4 World Model 的输出

最少：

\[
\boxed{W_\theta(\hat S_{t-K:t-1})\rightarrow(P_t,h_t^W)}
\]

- \(P_t\)：当前物理姿态的 dynamics prior，与视觉状态处在**相同 canonical physical state space**。
- \(h_t^W\)：内部 trajectory feature，可供 corrector 使用。

第一版不必同时预测 future image、force、contact graph、不同 motion experts 等。

## 5.5 训练

用高质量设备 trajectory 独立预训练：

\[
S_{t-K:t-1}^{GT}\rightarrow S_t^{GT}.
\]

可使用：

\[
L_{WM}=L_{pose}+\lambda_vL_{motion}.
\]

重点是学到可迁移的运动先验，而不是学习特定 camera 的视觉 bias。

---

# 六、传统视觉分支：从 WiLoR 转向现代 Metric Video Reconstruction

## 6.1 为什么不应只用 WiLoR 定义整条视觉 pipeline？

WiLoR 是强力 RGB hand pose / mesh 模型，但它的核心任务是 visible hand reconstruction，不等于完整的：

\[
\text{camera video}\rightarrow\text{world-space metric physical trajectory}.
\]

用户尤其提出：希望参考专业数采设备对齐的方案，以及 RGB 先估 depth、3D geometry 再预测 hand/action 的路线。于是后续建议是：**WiLoR 可以是 hand submodule，但整个传统分支要参考更现代的 2025–2026 metric reconstruction systems。**

## 6.2 现代视觉分支的建议输入输出

**输入**：

- RGB frame / video；
- 可选实际 RGB-D depth；
- 若无深度传感器，可使用估计 metric depth；
- camera intrinsics 与 camera pose（来自 SLAM/VIO 或固定标定）。

**分支 A：Hand reconstruction**

\[
I_t\rightarrow\text{hand crop/features}\rightarrow\text{3D wrist + joints / MANO}.
\]

**分支 B：Metric geometry**

\[
I_t\rightarrow\text{Depth/pointmap}\rightarrow\text{metric 3D geometry}.
\]

两路 metric alignment：

\[
\text{hand reconstruction}+\text{metric geometry}
\rightarrow V_t.
\]

最少输出：

\[
\boxed{V_t=[T^{wrist}_t,J^{hand}_t]}
\]

保留视觉特征图 \(F_t\) 以便在状态校正 loop 中重新读取当前 candidate 对应位置的视觉证据。

```text
                RGB frame I_t
                 /          \
                ▼            ▼
       Hand reconstruction   Metric geometry
         hand joints/MANO    depth / pointmap
                \            /
                 ▼          ▼
                 Metric Alignment
                        │
                        ▼
                   Visual State V_t
                  + features F_t
```

## 6.3 可参考的具体技术

- **MoGe-2 / metric depth / Flow3r 类 geometry**：获取 depth / metric scale / pointmaps。
- **WiLoR / HaWoR 组件**：恢复 hand joint / mesh / world-space trajectory。
- **Pseudo-Stereo Hand**：单目 RGB 提供深度预测与 pseudo-stereo 几何线索，再融合恢复 3D hand。
- **AnyHand**：RGB(-D) 手部重建，展示 depth fusion 的可行结构。
- **UmeTrack**：更早的 headset multi-view/multi-frame tracking 思想；虽然值得参考，但不是 2025+ 优先的主要方法贡献。
- **SHARP**：RGB pseudo-depth 用于 egocentric hand pose；作为历史启发，不做核心母体。

## 6.4 当前最推荐的传统母体

不是 `WiLoR + temporal correction`，而是：

\[
\boxed{\text{Modern Metric Video Reconstruction}+\text{TWM Closed-loop Correction}}
\]

尤其建议研究 **EgoInfinity、Do as I Do、VideoManip** 的视觉链路；**HaWoR** 为强 world-space hand baseline；**StableHand** 为强 temporal/trajectory refinement competitor。

---

# 七、Iterative Corrector：方法真正的核心

## 7.1 为什么不是简单拼接

单次：

\[
S_t=\mathrm{MLP}([V_t,P_t])
\]

研究区分度不够。最终讨论建议让它真正形成**状态驱动的视觉重观测 + trajectory-guided residual correction**。

## 7.2 初始化

\[
S_t^0=V_t.
\]

当前视觉提供可观测 evidence，trajectory world model 提供 prior，不能让 prior 不加判断地覆盖清晰视觉证据。

## 7.3 State-guided re-observation

用 candidate hand joints 投影回图像：

\[
u_{t,j}^k=\Pi(S_{t,j}^k;K,T_C).
\]

在先前预计算的 feature map \(F_t\) 上，以 joint/region 位置采样局部特征：

\[
f_{t,j}^{vis,k}=\mathrm{Sample}(F_t,u_{t,j}^k).
\]

可先用 bilinear sampling/ROIAlign，后续再尝试 deformable attention。**每一轮无需重跑昂贵的图像 backbone。**

若有 metric depth/pointmap，也可在候选 joint 附近采 3D geometry feature，这会更贴近真实物理空间，但不是第一版必须。

## 7.4 Residual correction

每个关节的候选 token 可包含：

\[
z_{t,j}^k=[S_{t,j}^k,P_{t,j},S_{t,j}^k-P_{t,j},f_{t,j}^{vis,k},h_t^W].
\]

一个较小的 Transformer/MLP corrector 输出状态 residual：

\[
\Delta S_t^k=C_\psi(S_t^k,P_t,h_t^W,f_t^{vis,k}).
\]

迭代：

\[
\boxed{S_t^{k+1}=S_t^k\oplus\Delta S_t^k.}
\]

建议第一版 \(N=2\) 次 refinement，对比一次/三次的收益和时间成本。

## 7.5 时间闭环

取最后一轮：

\[
\hat S_t=S_t^N.
\]

放入 corrected history，下一帧 TWM 使用：

\[
\hat S_{t-K+1:t}\rightarrow P_{t+1}.
\]

**这一步与当前帧的 iterative correction 同等重要**，因为它体现世界模型维护的是被不断视觉校正过的 state，而非从固定起点盲目生成的轨迹。

## 7.6 统一 Method 公式

\[
\begin{aligned}
(V_t,F_t)&=V_\phi(I_t,D_t,T^W_{C,t}),\\
(P_t,h_t^W)&=W_\theta(\hat S_{t-K:t-1}),\\
S_t^0&=V_t,\\
f_t^{vis,k}&=\operatorname{Sample}(F_t,\Pi(S_t^k)),\\
\Delta S_t^k&=C_\psi(S_t^k,P_t,h_t^W,f_t^{vis,k}),\\
S_t^{k+1}&=S_t^k\oplus\Delta S_t^k,\\
\hat S_t&=S_t^N.
\end{aligned}
\]

这一组公式就是整个算法的核心，不必为证明概念而增加大量旁支模块。

---

# 八、训练方案：只保留必要的三个阶段

## Stage I：Trajectory World Model Pretraining

- 输入：高质量 glove / mocap / robot-tracking 轨迹；
- 监督：下一个 state 或 state increment；
- 目标：TWM 学习 physical motion prior，而不依赖具体 camera appearance。

## Stage II：Camera–Sensor Paired Correction Training

- Visual Observer 给 \(V_t,F_t\)；
- 预训练 TWM 给 prior \(P_t\)；
- iterative corrector 预测残差；
- glove / mocap GT 监督每次 refinement、尤其最后一轮。

例如：

\[
L_{iter}=\sum_{k=1}^{N}w_k\,d(S_t^k,S_t^{GT}).
\]

## Stage III：Closed-loop Finetuning

- 训练时真正将预测 \(\hat S_{t-1}\) 回灌给世界模型，而不是始终喂 GT state；
- unroll 一小段序列，例如 8–16 frames；
- 避免 teacher forcing 下成功但 rollout 时累计漂移。

第一版 loss 不堆砌：

\[
L=L_{state}+\lambda_m L_{motion}+\lambda_iL_{iter}.
\]

其中 rotation 按 SO(3) 几何定义误差；world-space MPJPE 需要和 GT 做统一坐标、单位、关键点 convention 对齐。

---

# 九、主图设计讨论与已生成图

主图采用横向顶会论文风格。主信息是：

1. **蓝色**：Camera observation、visual observer、visual state。
2. **橙色**：Corrected history、Trajectory World Model、state prior。
3. **绿色**：Iterative state correction、corrected state、recovered trajectory。
4. **灰色虚线**：Training-only glove/MoCap GT supervision。

重点强调两个环：

- 当前帧内部的 `Visual evidence ↔ Candidate state ↔ Trajectory prior` 迭代校正。
- Final corrected state \(S_t\) 回灌 corrected history，进入下一个时刻 world-model prediction。

主图只保留必要标注，例如：

- `Camera Observation I_t`
- `Visual Observer`
- `Visual State V_t`
- `Corrected State History`
- `Trajectory World Model`
- `State Prior P_t`
- `Iterative State Correction`
- `State Corrector`
- `Visual residual`、`Trajectory residual`
- `Recovered Trajectory`
- `Training only: Glove / MoCap Ground Truth`

已在本次对话生成 PNG 主图，文件名：`闭环视觉状态估计与轨迹世界模型.png`。若 MD 与图放在同一目录，可以直接显示：

![Closed-loop method overview](闭环视觉状态估计与轨迹世界模型.png)

后续修图应优先解决：中央反馈箭头是否真的连通视觉重观测、corrector refinement 是否明确、状态回灌方向是否正确，以及训练 GT 的虚线监督是否清晰。**主图不应塞入 planning、MoE、long/short memory、全部 losses、dataset 名称等细节。**

---

# 十、关于 CVPR 价值的讨论

讨论结论：

- **CVPR scope 很适合** camera-based 3D physical-state reconstruction + learned dynamics prior。
- 单纯 `Vision + temporal module` 的方法 novelty 可能不够。
- 对已有 trajectory/motion prior 方法需要明确区别，尤其与 generative motion refinement、temporal infilling、HOI optimization 之间不能只换名字。
- 最核心 claim 是：**利用跨任务/跨设备学到的 trajectory prior，使低成本 camera-only state estimation 可以持续闭环校正，并在 unseen dataset/observation domain 上缩小与高质量传感器的误差。**

更谨慎的论文表述：

> We formulate camera-based motion capture as closed-loop physical-state estimation, in which visual evidence and a learned trajectory prior iteratively correct the estimated state toward high-fidelity sensor measurements.

并且：

> A trajectory model pretrained on high-quality motion measurements serves as a reusable physical prior for noisy visual state reconstruction.

避免不准确地宣称“以前工作完全没有时间建模”。已有 HaWoR、StableHand、Dyn-HaMR、WHOLE/EgoGrasp 等都做了时序、motion prior 或 HOI refinement。

---

# 十一、实验与消融设计

## 11.1 最小可行实验

先别做全机器人平台：

1. 找一套 paired video + instrumented high-quality 3D hand states。
2. 现有 Visual Reconstruction 给 \(V_t\)。
3. TWM 用设备 GT trajectory 预训练。
4. 建一个简单 corrector，形成递归预测、视觉观测校正与状态回灌。
5. 验证 \(V_t\) vs \(\hat S_t\) 相对于 GT 的误差变化。
6. 再逐步加入 metric geometry、state-guided re-observation、跨数据集测试。

## 11.2 核心 baselines

- Visual-only metric reconstruction。
- Visual + classical smoothing / Kalman。
- Visual + ordinary temporal Transformer。
- Visual + TWM one-shot correction。
- Visual + iterative correction。
- Visual + iterative correction + state-guided re-observation。
- Full closed-loop training。
- 强论文基线：HaWoR / StableHand / EgoGrasp 等（按可复现性与具体任务匹配筛选）。

## 11.3 主实验指标

- wrist translation error（mm）；
- wrist orientation error（degrees）；
- MPJPE / world-space MPJPE（统一 joint mapping 和对齐约定）；
- trajectory drift / temporal consistency；
- visibility-conditioned error；
- 受控遮挡长度与严重程度下的 state reconstruction error；
- downstream robot retargeting / policy utility（作为强验证，而不是必须成为方法主线）。

## 11.4 Camera–Sensor Gap Closing

讨论提出概念指标：

\[
\mathrm{GapClosed}=\frac{E_{camera}-E_{ours}}{E_{camera}-E_{sensor}}.
\]

注意：如果 sensor measurements 本身被当作 evaluation ground truth，通常应令 \(E_{sensor}=0\)，且要标明这种“gap closing”仅相对于所定义的标签/benchmark，不意味着真实物理误差为零。之前讨论中的 30/12/5 mm、occlusion 曲线数字都是**示意数据，不是真实实验结果**。

## 11.5 Cross-domain 与 scaling

用户希望做的不是只在一个 dataset 上取得微小增益。重点验证：

- cross-dataset；
- unseen subject / object / manipulation；
- RGB vs RGB-D vs multi-view 的不同 camera observation；
- 将同一 frozen TWM 接到不同 visual reconstruction backbones；
- 加大高质量 trajectory pretraining corpus 是否系统性提升 OOD reconstruction。

这些实验证明 world model 是**跨观察系统可迁移的 trajectory prior**，而不是某个视觉模型的后处理器。

---

# 十二、文献筛选思路的最终纠偏（最后几轮最重要）

用户强调：

> 请回看最早一批相关工作，而不是只盯 WiLoR、数采数据集和一篇 World Model。不要用 dataset paper 主导方法选择；尽量看 2025 年及以后（必要时参考 2024 年）的真正 method paper，并在完整 pipeline 上扩展。

据此作出以下区分：

## A. 最适合参考的现代视觉重建 / 视频到轨迹方法

| 方法 | 时间定位 | 关注点 | 在本研究中的用途 |
|---|---|---|---|
| **EgoInfinity** | 2026 | RGB → metric geometry → 4D hand/object → retargetable state | **优先参考视觉 front-end** |
| **Do as I Do** | 2026 | RGB → 4D HOI → physics/retargeting → robot trajectory | **完整系统母体 + downstream** |
| **VideoManip** | 2026 | RGB → hand/object → interaction/contact refinement → dexterous policy | camera-to-robot trajectory 参考 + downstream |
| **RoboWheel** | 2025–2026 研究线 | RGB/RGB-D → HOI → 物理修正 → cross-embodiment | 系统级对照 |
| **HaWoR** | 2025 | egocentric video → world-space hand reconstruction，已有 temporal refinement | 强视觉 baseline / hand submodule |
| **EgoGrasp** | 2026 | 视觉证据与 HOI motion prior 协同重建 | 强 reconstruction baseline |
| **StableHand** | 2026 | noisy visual trajectory + quality-aware generative motion prior | **必须严肃对照的 temporal prior competitor** |
| **C2Dex** | 2026 | object-centric contact refinement | 接触/机器人下游扩展参考 |

## B. 结构思路参考，不必成为视觉 front-end

| 方法 | 年份 | 借鉴点 |
|---|---|---|
| **ManipTrans** | 2025 | trajectory imitator + residual refinement，不替代观测、而是提供动力学和修正思想 |
| **DemoDiffusion** | 2025–2026 | noisy/rough trajectory → trajectory manifold correction |
| **Human2Sim2Robot** | 2025 | object-centric task transfer；更偏仿真 RL/robot grounding |
| **Phantom** | 2025 | camera-based human demonstrations → robot-compatible data |
| **Track2Act** | 2024 | visual tracks → structured motion → residual closed-loop robot control；保留为有启发性的较早工作 |
| **WestWorld** | 2026 | trajectory-level dynamics prior 的启发，**不是本研究的主导架构** |

## C. Dataset / 高质量数采资源：只作为训练和评测支持

- **DexCap**：同步 RGB-D、camera pose、手腕/手关节和 glove/mocap 信息，适合 paired sensor supervision。
- **HOT3D**：egocentric、多视角、hand/object 3D tracking GT，可做跨域评测。
- **EgoDex**：大规模 egocentric video + 设备估计 hand tracking，适合 trajectory pretraining；不应简单等同于独立 MoCap GT。
- **AssemblyHands**：多视角高质量标签，适合研究监督模式；不作为近期主要方法创新引用。
- 其他手物/触觉数据集按 license、同步字段和 ground-truth 质量决定是否纳入。

## D. 最终方法定位

之前过于简化：

\[
\text{WiLoR}+\text{Temporal World Model}.
\]

最终建议升级为：

\[
\boxed{\text{Metric Visual Reconstruction Pipeline}
\leftrightarrow\text{Trajectory World Model}
\rightarrow\text{Closed-loop Corrected Physical State}}
\]

即，**在 2025–2026 的现代 camera-to-physical-state pipeline 上，利用设备监督训练的 trajectory prior 做迭代校正与递归反馈**，而不是从零发明视觉 pose backbone。

---

# 十三、当前可以直接开工的第一版配置

| 组件 | 具体建议 | 状态 |
|---|---|---|
| Input | RGB + camera calibration / pose；先支持 RGB-D，后支持 pseudo-depth | 待确定采集设备 |
| Visual front-end | EgoInfinity / Do as I Do 启发的 metric geometry + hand reconstruction | 主参考 |
| Visual output | wrist SE(3) + canonical 3D joints，保留 image features | 已定义 |
| State representation | 72D/手的 wrist + joints | 初步设定 |
| TWM | 局部坐标标准化 + causal Transformer，K≈8，6–8 layers | 第一版 |
| TWM output | \(P_t, h_t^W\)，预测增量生成 prior | 已定义 |
| Corrector | 基于预测 3D state 重观测图像特征，输出 \(\Delta S_t^k\) | 核心创新候选 |
| Iteration | \(N=2\)，可对比 1/3 | 第一版 |
| Feedback | 最终 \(\hat S_t\) 回灌，进入下一时刻 TWM | 必须 |
| Training | clean trajectory pretrain → paired correction → closed-loop unroll | 已规划 |
| Main validation | Camera vs sensor GT、遮挡、跨 dataset/backbone | 必须 |
| Downstream | 现有 robot retargeter / action pipeline | 追加验证 |

## 最终一句话

> **Learning to recover sensor-quality human manipulation states from camera observations through closed-loop interactions between metric visual reconstruction and a pretrained trajectory world model.**

中文：

> **利用高精度数采设备学习物理轨迹先验，再将现代相机三维重建与轨迹世界模型组成持续迭代、持续反馈的闭环状态估计系统，在不佩戴手套的情况下尽可能恢复接近传感器真值的操作轨迹。**

---

# 十四、后续尚未最终定案的问题

这些属于实现决策，不是要求继续增加论文模块：

1. **Input hardware**：第一版究竟 RGB-only 还是 RGB-D 主实验？更建议先 RGB-D 建立精度上限，再提供 RGB-only 泛化结果。
2. **State convention**：单手/双手、wrist frame 定义、21-joint skeleton mapping、world/camera 标定。
3. **Base pipeline 选择**：EgoInfinity、Do as I Do 哪个代码链在目标数据集上能稳定跑起来？需要以仓库可复现性和许可证为准。
4. **Corrector 的 re-observation**：采样 RGB feature、depth feature 还是二者结合；先做最简单有效版本。
5. **论文最强区别点**：是否可以证明 iterative re-observation + TWM prior 比 one-shot fusion、generative trajectory denoising 和普通 temporal backbone 更准确、且更跨域。
6. **目标精度**：应通过实际 paired benchmark 测量，不提前承诺“与 glove 完全一样”。

---

## 研究结论

本轮对话从“Camera 和数采手套精度到底差多少”的行业调研出发，逐步收敛到一个可实现、适合 CVPR 叙述的研究方向：

**核心不在于更复杂的世界模型，也不在于重新发明视觉姿态估计器；而在于让一个从高质量设备轨迹学来的 World Model，成为现代 metric camera reconstruction 的动态先验，通过当前帧内部的迭代校正与跨帧状态反馈，持续提高 camera-only physical-state reconstruction 的准确性和稳定性。**



---

# 十五、2026-10-08 至 10-13 的后续会话顺序归档

> **归档边界说明**：下面沿消息顺序保留后续讨论中各轮用户问题（尽可能按当前可见原话）及对应答复的主要推导、公式、关键判断和重要纠偏；**“助手回应要点”是内容归档而非逐字转录**。不能把此文件冒充 ChatGPT 官方 JSON/HTML 原始对话导出。若确需逐字原文，需另行使用账户数据导出功能。此处优先保证新 Work 项目可以准确延续科研。

## 15.1 生成顶会风格主图

**用户：**

> 很好，现在参考定会论文的方式，把这个图生成出来

**助手回应 / 产物：**

- 生成《闭环视觉状态估计与轨迹世界模型》早期方法架构图；在项目包中保存为 `04_Method_Overview.png`。
- 图中蓝色是 Camera Observation、Visual Observer、Visual State；橙色是历史校正状态、Trajectory World Model、State Prior；绿色是 Iterative State Correction、Recovered Trajectory；灰色虚线是仅训练阶段的 Glove/MoCap GT。
- 展示两个闭环：当前帧内迭代纠正与跨帧 state feedback。
- 该主图反映**早期 prior predictor** 设计；后续 TWM 已从 forecasting-oriented 改为 **conditional current-state restoration**，并增加原始观测/候选状态分离与 reliability re-assessment，故此图后续应重画。

## 15.2 World Model 与传统视觉模型的输入输出

**用户：**

> 整体的模型架构是这样，但是其中譬如world model的构造，输入输出应该是什么。然后传统的model的又应该是什么样？传统的那一部分，你也可以参考近期做的比较好的工作

**助手回应要点：**

- 初始三模块：Visual Pose Estimator + Trajectory World Model + Iterative Corrector。
- 最初用 wrist translation 3 + rotation6D 6 + hand joints 21×3 = 72D/手；后来修正为**不重复 wrist root 的 20×3 joints，69D/手**。
- 最初提出 Visual Observer 参考 WiLoR：图像→hand detection→ViT→pose/wrist heads；输出 \(V_t,F_t\)，后端对图像 feature map 采样。
- 初始 TWM：causal Transformer，历史 8 帧、6–12 层、隐藏 256/512，输入 corrected states/motion delta，输出 prior \(P_t\) 与 dynamic feature \(h_t^W\)，主要预测下一帧的状态增量。
- 初始 iterative corrector：3D-to-2D projected joints → 在 image feature map 重新局部采样 → joint-wise residual correction，2轮。
- 初始三阶段训练：TWM GT next-state prediction → paired corrector → closed-loop rollout。
- **注意：以上均为历史中间架构，不是最新版本；最新 TWM 的重点已转为恢复当前受损状态**。

## 15.3 更接近 Glove/传感器对齐的视觉系统，及 RGB→Depth→Pose

**用户：**

> WiLoR-style Visual Observer这个手都是可见的，有没有跟我做的比较类似的work，就是也是拿手套的信息，等等来做。也不一定是手套，主要是靠齐数采设备， 你懂吧。你再多给我找一下可以在上面做拓展的方法。我记得不是还有那种吧camera先转换为深度等等信息，然后再预测。都可以看一下！

**助手回应要点：**

- 从单帧 WiLoR 拓展到 head-mounted tracking / sensor supervision：
  - UmeTrack：multi-view fisheye +时序 hand tracking，主要方法架构参考；
  - HOT3D：professional MoCap annotations 与头戴相机视频配对，主要是数据/评测；
  - DexCap：Rokoko glove + wrist 6DoF tracking + RGB-D，有高质量轨迹与视觉配对；
  - EgoDex：大规模 Vision Pro 视频/设备手部追踪，**不能当独立 glove GT**；
  - AssemblyHands：多视角高质量手部标注。
- 考虑了 RGB→monocular depth→3D state（MoGe-2等）、RGB-D真实 depth、multi-view UmeTrack 几何，不应把原始 RGB hand mesh 当成已经具备精确绝对 metric scale。
- 探索方案 `RGB → image feature + depth/point map → metric 3D geometry → wrist/joint visual state`。
- 后续明确主输入将是**头戴 RGB-only**，不应把中间提出的 RGB-D-first 当成最终定案。

## 15.4 方法论文优先；不要用 Dataset paper 主导框架

**用户：**

> 但是请你看我们最早期的文献记录，那个时候你先找到一些相关工作的，为什么你不参考那些相关工作？那些相关工作其实也不错。就是你再仔细思考一下，就是有些是 dataset paper，我们就不要考虑它。dataset paper，它的这个方法 contribution 很小，它没有它什么问题，我们这点先忽略掉。然后其他的话，我们尽可能用25年以后的方法，然后在上面再做改进，或者24年以后的，我觉得可能会更好。

**助手回应要点：**

- 重新筛选 2025–26 method papers：HaWoR（egocentric world-space hand）、EgoInfinity（metric visual reconstruction system）、Do as I Do（human video→HOI→robot）、VideoManip、RoboWheel、EgoGrasp、StableHand；ManipTrans、DemoDiffusion 作为 trajectory correction 思路；Phantom、Human2Sim2Robot 等偏下游。
- WestWorld 只应被视为 trajectory modelling 的结构启发。
- 数据集 DexCap/HOT3D/EgoDex/AssemblyHands 是 supervision/benchmark，不是主方法引用支柱。
- 强调 visual reconstruction 的进展：RGB→metric geometry + hand reconstruction→world-space state；不能将 WiLoR 单个 network 当整个 SOTA reconstruction system。
- 最终论文贡献应集中到 “sensor-trained reusable motion prior for closed-loop camera-state correction”，而不是重新设计整个机器人 pipeline。

## 15.5 第一次导出旧文档

**用户：**

> 你把上述所有的完整对话整理到一个md文件中，发我

**助手回应 / 产物：**

- 生成 `Camera_Trajectory_World_Model_完整对话整理.md`（约 795 行、33KB）。
- 该文件为前期讨论**按主题重组的内容版**，明确标明不是逐字聊天导出。
- 本项目包的第 1–14 节原样保留此文件的内容，之后又增加了第 15 节和新一轮完整 Method/Introduction 内容。

## 15.6 重述核心 Idea

**用户：**

> 请你总结我的核心idea

**助手回应要点：**

- 研究定位：从 glove/MoCap 的高精度数据学习 motion knowledge，降低 camera-only 采集过程中物理状态估计误差。
- 三模块：metric visual reconstruction、独立 trajectory prior、state corrector。
- 当时概括为预测+修复+时间反馈；随后用户进一步强调：“纠正当前状态”为主、预测为辅。

## 15.7 再检索适合直接改造的 Visual Reconstruction

**用户：**

> 很好，首先请你尽可能帮我找一些我可以使用的visual reconstruction的方案，最好是很好拓展，但是效果又最好的

**助手回应要点：**

- 按开源、输出接口、性能与可修改性比较 EgoInfinity、HaWoR、Do as I Do、ACE-Ego-Hand、MINT、OmniHands、EgoFound3R、ViDiHand。
- EgoInfinity：完整 metric camera→hand-object 4D pipeline，外部静态 camera更适合；Do as I Do：整套 camera→robot retargeting；HaWoR：head-mounted egocentric hand world-space tracking；MINT：视频多任务统一模型；ACE-Ego-Hand：偏遮挡恢复/强基线；OmniHands：双手和多视角。
- 额外查过 MoGe-2/MoGe-3、Flow3R、VGGT、MegaSaM 等几何工具。
- **最终用户从这些候选中明确选择 HaWoR，不应在后续工作中无端更换主干**。

## 15.8 最终确定头戴 Camera + HaWoR + 自训练 TWM

**用户：**

> 我打算使用HaWoR了，然后自己训练一个trajectory world model。打算使用头戴 Camera

**助手回应要点：**

- 研究第一版正式锁定：Head-mounted RGB → HaWoR → world-space wrist/hand states → TWM / Corrector → corrected trajectory。
- HaWoR 的工程接口可拆为 hand estimator、相机 SLAM / world conversion、infiller。
- 初始接入在 SLAM 转换之后、HaWoR motion infiller 之前，后者作 baseline；必须保存原始检测 mask，不能将插值帧视为真实检测。
- 首选测试 HOT3D 头戴/外部 MoCap 对齐数据；DexCap 的胸前视角作为其他数据源。
- 当时建议 history 8 frames、6 layers、hidden 256、N=2、69D wrist+finger；这些是起步参数，不是固定科研创新点。

## 15.9 WestWorld 灵感，但不能直接复用机器人预测任务

**用户：**

> 对，然后world model打算使用west world那个范式，但是有可能有点问题，请你仔细梳理一下

**助手回应要点：**

- WestWorld 原始任务：robot observations + known actions → future robot state；用于 robot planning。用户则没有真实 control action，也不打算训练 89 个系统的通用机器人模型。
- 可借鉴 structured/kinematic embeddings、trajectory modelling、state tokenization。暂不需要 robot action cross attention、System-aware MoE、MPPI planning 或原始机器人 checkpoint。
- 官方代码 `models/WestWorld/westworld.py` 有 observation/action channels、按通道 soft discretization/分类目标及 robot structure configs，不能简单换成 69D human-state head 直接使用。
- 首先提议小规模 structured temporal Transformer，局部 wrist frame 标准化，预测 state increments；但这一 forecast-oriented 设计在后续被用户明确调整。

## 15.10 TWM 纯轨迹学习，视觉仍由 HaWoR 负责

**用户：**

> 本思路是这个 Word Model，它是能够学会人手部的动力学，但是不通过视觉的方式，就只学浅部的动力学。至于这种视觉信息改变啊等等的，就是再通过那个视觉估计模型进行。然后训练数据,就是说现在有很多的通过手套采集的数据集，对吧？我们就用这些数据集，然后就是拿手套数据再重新训练出一个 world model。完全不用它这个，只是说是我们有没有可能里面借鉴一些思想呀之类的。还有就是你其他的漂移误差等等的，都可能会影响它的积累。所以就是说你再仔细想一下，我们还有什么地方能够改进这个方案

**助手回应要点：**

- 完全独立在结构化状态空间中以 glove/MoCap 轨迹训练 TWM，不看 image。
- 最初建议 local wrist motion (SE3 relative) + finger articulation；预测分布/uncertainty 而不是确定下一点；adapative correction gate，检测可靠性和 prior 的不确定性共同决定权重。
- 防累积误差：noisy history training、闭环 unroll、visual re-anchor。需区分局部运动误差与 head camera SLAM 漂移。
- 两数据池：trajectory-only pretraining 与 paired camera+GT corrector training；EgoDex 不等同 glove GT。
- 此时仍主要采用 next-state motion prior，用户下一轮进一步澄清：**预测只是教学手段，真正的目标是当前状态修复**。

## 15.11 最重要的一轮纠偏：纠正当前状态，不是预测未来

**用户原话：**

> 其实也并不是完全地去预测，就是有点类似于是现有卡尔曼滤波一样，就是说我不知道怎么去描述，就是说我并不是一定要去预测未来很远很远的东西，而重点是要把现有视觉的信息给它纠正过来，而且就是说预测未来的信息的方式来去增强它模型对于物理现象的理解。就重点并不是预测，而是说对于物理过程的理解。你能理解吗？就是说我再去重新思考一下。就是譬如视觉估计，它可能在一些能看到的一些 position，它的精度其实挺不错的了，但是一些看不见的地方，它效果不好，或者是头部发生偏移，或者是等等误差漂移。所以就是说我们重点是要对它 high confidence 的信息作为加以利用，然后对 low confidence 的信息进行，就是类似于是纠正呀等等的，而且并不一定 high confidence 一定是对的。所以重点就是说怎么让它纠正过来。

**助手回应要点：**

这成为**最终科学问题的关键定义**：

1. 与 Kalman 滤波相似的是 state estimation：model provides process prior, observations update current state；不是 long-horizon forecasting。
2. 不等价于 `visible=vision, invisible=prediction`；正确机制是**用当前可信局部观测的结构关系恢复当前不可见或错误的其他部分**，如掌心旋转变化直接约束被遮挡手指。
3. 即使 HaWoR 声称高 confidence 也可能错；应区分原始检测 confidence 与最终 calibrated reliability；与先验相冲突不意味着视觉一定错，真实手可突然转向。
4. 把 TWM 预训练目标从 next-state forecasting 改成 **motion-history-conditioned corrupted state recovery**：
   \[
   W_\theta(\mathcal H_t,\widetilde S_t,M_t)\rightarrow S_t^{GT}
   \]
   加短时预测辅助项。
5. TWM deployment 时可读取当前结构化候选 state，仍完全不接触 RGB 像素；Corrector 使用 image evidence 对候选状态进行测试时重检查。
6. 两闭环：当前帧内 candidate→视觉取证→TWM建议→state correction；帧间最终 state→history。当前帧原始 \(Z_t\) 必须始终保存，不能把迭代输出伪装成新观察。

**强制保留的单句**：预测是为了学习运动过程，**当前状态校正才是最终任务**。

## 15.12–15.15 论文标题系列讨论

**用户问题（同义重述连续两轮）：**

> 你觉得论文的名字怎么起好听，我的意思是前面是一个有意思问句，就是譬如手套一定是需要的吗？然后是我们的方法的title

> 你觉得论文的名字怎么起好听？我的意思是，前面是一个有意思的问句，譬如“手套一定是需要的吗？然后是我们的方法的title

**用户明确偏好的风格：**

> 我很喜欢06 · 更简洁的顶会风格  
> Are Gloves Necessary? Learning to Correct Visual Hand Motion with Dynamics Priors  
> 较短、直接，也非常准确地概括了你真正的研究贡献：纠正而非单纯预测。这个的风格，你能多想想吗

**用户继续问：**

> 可以用其他符号吗，或者近几年的论文title有带问号的吗

**助手回应要点：**

- 用户偏好 **问句 + 精炼技术性标题** 的风格，不要很长方法名/花哨修辞。
- 标题候选：
  - `Are Gloves Necessary? Learning to Correct Visual Hand Motion with Dynamics Priors`
  - `Are Gloves Necessary? Learning Motion Priors for Visual Hand Correction`
  - `Are Gloves Necessary? Learning to Correct Visual Hand Reconstruction with Motion Priors`
  - `Are Gloves Necessary? Learning to Recover Hand Motion from Imperfect Vision`
  - `Are Gloves Necessary? Closed-Loop Visual Hand Reconstruction with Learned Motion Priors`
  - `Can Motion Correct Vision? Learning Hand Dynamics from Glove Trajectories`
  - `When Vision Fails, Can Motion Help? Closed-Loop Hand State Reconstruction from Egocentric Video`
  - 方法名备选 HandLoop、KineLoop、MotionLoop 等，未检索排重。
- Motion Priors 更稳妥，Dynamics Priors 更容易引起 torque/contact physics 误解；用 `?` 本身没有 CVPR 格式问题，问号后不再添冒号是自然写法。
- 举过 CVPR 2025 `MambaOut: Do We Really Need Mamba for Vision?`、ICCV 2025 `What Changed and What Could Have Changed? ...` 等正式问句标题为例，写论文前仍可复核源头。
- 一定要解释：Glove-free 指 inference，训练仍借助 glove data。

## 15.16 再次完整概述方法

**用户：**

> 很好，然后请你完整的理解和总结一下我的 方法

**助手回应要点：**

- 更明确提出 `Glove/MoCap trajectory → learn physical-motion knowledge → correct imperfect camera-derived hand state`。
- 三模块与两个闭环。
- 当前 partial observed joints 通过学到的手骨架和时序关系帮助恢复其余 joints，而不是关节间彼此独立地补全。
- High confidence 有误时允许覆盖，但不因真实动作偏离 prior 就强制认为视觉错误。
- TWM 使用 `corrupted current-state restoration + auxiliary short prediction`。
- 建议训练层级：pure motion → camera/GT paired → rollout；现实边界：camera SLAM global drift 无法由 motion knowledge 自行解除。
- 贡献为 trajectory-only restoration；reliability-aware visual update；双循环 feedback。

## 15.17 引言第一版：中文 → 英文

**用户：**

> 如果让你来写introduction，你来思考一下怎么写，中文的就可以

随后：

> 你把这个intro翻译成英文发我

**助手回应要点：**

- 第一版完整 Introduction 已形成：robot imitation data capture 的需要 → instrumented glove deployment overhead → egocentric RGB occlusion/drift → incomplete/hybrid visual reliability → learned physical motion prior → reliable-observation-conditioned state restoration → iterative correction and temporal feedback。
- 随后译成英语，风格 CVPR/ICCV，而非逐字直译。
- 结尾三项贡献：trajectory physical state restoration；reliability-aware iterative correction；camera-only closed-loop reconstruction。
- 写作原则：避免暗示已有 HaWoR/HMP/StableHand 完全不使用时序、motion prior 或 quality-aware modeling。

## 15.18 用户亲自确定 Introduction 首段的新叙事

**用户原话（保留核心完整内容）：**

> 准确恢复人手的三维运动，是将人类操作演示转化为机器人学习数据的重要基础。相比于仅记录操作过程的视频，腕部的空间轨迹和手指的精细姿态能够提供更明确的运动描述。以 DexCap 为代表的采集系统通过数据手套、空间跟踪与视觉传感器的组合，获得对视觉遮挡更鲁棒的手部运动数据，为灵巧操作学习提供监督。与此同时，这类方案仍要求在每次采集时配置相应的穿戴设备和跟踪系统。... 这部分能不能这么转折？就是第一句话不变，说明手部运动数据的重要性。后面先跟视觉方案，然后说他的不好，所以行业现状还是主流使用数采设备的这一套方案，因为可以miss纯视觉方案的问题。但是仍要求在每次采集时配置相应的穿戴设备和跟踪系统。导致数据采集非常麻烦，十分依赖于设备，很难大规模采集。  
> 这促使我们思考：**手套是否必须始终作为采集过程的一部分，还是可以将其提供的高质量运动知识迁移到无需佩戴手套的视觉采集系统中？**

**助手回应要点：**

- 采用用户建议的新叙事顺序：3D hand motion 的数据价值→camera-only 的低侵入/便捷→camera occlusion/depth/viewpoint/drift 精度问题→高精度场景仍使用 glove+tracker→规模化部署负担→提出 question。
- 避免没有来源的“所有工业采集仍以 glove 为绝对主流”断言，使用更窄准确表达：“对数据精度与鲁棒性要求较高的机器人操作数据采集中，研究者仍广泛采用……”。
- 保留中文最后一句的研究问句，与 `Are Gloves Necessary?` 标题形成呼应。
- **中文和英语最终确定的第一段已在 `02_Introduction_CN_EN.md` 原文收录。**

## 15.19 固定第一段英文翻译

**用户：**

> 很好，把这段给我翻译成英文

**助手回应要点：**

- 完整翻译首段，重点 `high-fidelity motion knowledge` 而非仅 `motion data`。
- 最后一问：
  > Are instrumented gloves truly necessary throughout the data collection process, or can the high-fidelity motion knowledge they provide be transferred to a glove-free visual reconstruction system?
- 完整首段见 `02_Introduction_CN_EN.md`。

## 15.20 重写英文全篇 Introduction

**用户：**

> 很好，那你再想想后面的部分该怎么续写，重新写一下这个introduction

**助手回应要点：**

- 保留已确定第一段；
- 后续段落：视觉信息可靠性非均匀且可能假高信心 → HaWoR/HMP/StableHand 等已有进展 → 核心缺口不是没有 temporal/motion prior，而是“可靠局部观测通过学到的物理过程持续修复当前错误状态”→TWM 独立 glove training → reliability-aware iterative corrector → frame-local + temporal feedback → 训练与实验目标；
- 避免把尚未实现的结果写成事实。
- 对应英文全文见 `02_Introduction_CN_EN.md`。

## 15.21 完整中文版 Introduction

**用户：**

> 中文

**助手回应要点：**

- 完整中文版本与英语结构对应，包含 8 段正文和 3 条贡献，写入 `02_Introduction_CN_EN.md`。
- 主线为：Glove 高质量知识→独立 TWM→恢复当前 state 而非 long-term prediction→视觉证据 soft constraints→双闭环。

## 15.22 CVPR 潜力和创新性压力

**用户：**

> 你觉得这个idea投CVPR如何？

**助手回应要点：**

- CVPR 任务适配度高（egocentric reconstruction、human motion、robot demonstration）。
- **当前研究问题有意义，不等于当前设计已经具备录用创新性**。普通 `HaWoR + TWM + confidence gate` 可能被认为是 temporal prior + filtering 的工程拼接。
- 正面比较：HMP（独立 MoCap motion prior）、StableHand（quality-aware generative motion refinement）、PAD-Hand（physics-aware uncertainty/motion restoration）、WHOLE 等。
- 候选可辩护的新方法点：`current partial-state conditional restoration` + `candidate-conditioned independent visual re-observation` + `reliability recalibration and temporal feedback`。
- 关键 CVPR 实验证据：已准确的关节不被破坏、遮挡部位真误差下降、高 confidence 但错误的观测可被修正、突发真实动作允许改变先验、重新可见后能有效重锚定、多数据集严格泛化。
- 对 SLAM global drift 的界限应明确；不能宣称已实现 camera-global drift 消除。
- 曾查询 CVPR 2027 论文注册/正文/补充材料的日期；**具体日期、政策以 CVPR 官方当前页面为准**。
- 评分只是当时讨论时的主观判断，不是接收率或实际实验结果。

## 15.23 最近一轮：完整严谨的方法设计

**用户：**

> 请你好好完整的 帮我构思方法的完整框架，要非常有逻辑，好好思考一下！

**助手最终提出的完整方案（以下为高保真技术整合，不是逐字抄录）：**

### A. 核心任务

\[
\min_{\theta,\psi} \sum_t d(\hat S_t,S_t^{GT}),\qquad
\text{camera input at inference, glove/MoCap at train}.
\]

**不将 World Model 任务等同 future video generation 或 long-horizon forecasting。**

### B. 同时保留的三个量

1. \(Z_t\)：不变的原始 HaWoR 测量（所有帧内迭代留存）。
2. \(S_t^{(k)}\)：当前迭代候选状态。
3. \(\mathcal H_t\)：完成校正、附 reliability 的历史窗口。

混用三者会造成 measurement leakage / self-confirming feedback。

### C. Wrist + local joints state

\[
S_t=(p_t^W,R_t^W,J_t^{local}).
\]

position 3、rotation 6D、20×3 joints = **69D**；rotation 在 Lie group 中复合，修正量可用 **66D**（3+3+60）。左/右手需保留 handedness；固定局部参考 \(A_t=\hat T_{t-1}^W\) 在整帧迭代不动；手套、MoCap、HaWoR bone mapping 必须规范化。

### D. HaWoR image evidence 的重要代码事实

已查源码：
- `HAWOR.forward_step` 先 `backbone` 提取视觉特征，再 `temporal_attention`、`MANOTransformerDecoderHead`，返回 MANO 旋转与 3D joints 等。
- `pred_keypoints_2d` 由 3D predicted joints 投影得到，**不是独立检测真值**。不能将同一个 3D pose 的投影作为支持该 pose 正确的证据。
- 视频脚本对 missing detection 的 bbox 插值，并改变 `valid`；正确 training mask 应来自插值前原始 detection。
- 必须 hook/cache frame visual features 或建立轻量、独立的 2D image-evidence head；不需要重复完整 ViT inference。
- HaWoR 原始 temporal/infiller 要作 baseline，不能重复宣称自己首先解决 temporal smoothing。

### E. TWM 与 Corrector 的最终分工

\[
W_\theta(\mathcal H_t,Z_t,S_t^{(k)},M_t,w_t^{(k)})
\rightarrow (U_t^{(k)},h_t^{W,(k)}).
\]

只用 state/validity/reliability 数值；glove motion only pretrain；TWM 不看图像。

Corrector：
\[
u_{t,j}^{(k)}=\Pi_{\Gamma_t}(J_{t,j}^{W,(k)}),\quad
f_{t,j}^{(k)}=\operatorname{Read}(F_t,u_{t,j}^{(k)})
\]
然后联合原始视觉观测残差、TWM state restoration proposal、图像支持、reliability 预测 \(\Delta S_t^{(k)}\)。
\[
S_t^{(k+1)}=S_t^{(k)}\oplus \Delta S_t^{(k)}.
\]

TWM 读取了当前变化中的 candidate，所以重新运行可产生不同修复建议；如果它只读固定历史，每轮重新运行毫无意义。

### F. Reliability 的两个不同语义

- \(w_t\)：原始观测 \(Z_t\) 的可靠性，可在新证据下被重新估计；
- \(q_t\)：最终校正状态 \(\hat S_t\) 的可靠性，在反馈历史时传递。

分别用与 GT 的真实误差监督；推理绝不提供 GT，不应把任意高视觉 confidence 锁死。

### G. 两闭环

**帧内：** `candidate state → projected feature query → image evidence check → TWM conditioned state restoration → corrector residual → updated candidate`。

**帧间：** \(\hat S_t=S_t^{(N)}\)，然后 \(\mathcal H_{t+1}=\operatorname{Update}(\mathcal H_t,\hat S_t,q_t)\)。

原始观测、当前图像、该帧历史在同一帧内部不变；只有当前候选和预测 reliability 改变。

### H. 三阶段训练

1. 仅 glove/MoCap trajectories 预训练 TWM：masked/corrupted hand state recovery + auxiliary short forecast + structure/bone consistency。
2. 冻结 HaWoR/TWM：paired video+GT train independent visual evidence head、original-observation reliability and candidate reliability heads、iterative Corrector；每轮 state GT supervise，motion loss 匹配 GT velocity，不惩罚合理的快动作。
3. Closed-loop rollout training：将模型自身输出反馈，建议短窗口如 16 frames，必要时 TWM 轻调，前端 feature cache，避免反复重跑整个 vision network。

### I. 数据与实验

- HOT3D：头戴视频 + 高质量 motion capture reference，主要 paired training/evaluation；注意不同 split GT availability。
- DexCap：Rokoko gloves + RGB-D + wrist 6DoF source（腕部全局 tracking 包含 SLAM 误差，不可当作无误差 absolute ground truth）。
- EgoDex：可用于大量 trajectories，但为设备视觉 hand tracking，与 independent glove/MoCap quality 不同。
- 多种数据可能存在 skeleton、reference wrist frame、单位和采样差异。
- Metrics：world/camera/local MPJPE、wrist pos/orientation、occlusion, visibility, high-confidence wrong subsets、error after reappearance、velocity, runtime、cross-dataset.
- Ablation：HaWoR full、simple smoothing/Kalman、temporal Transformer、prediction-only TWM、current-state restoration TWM、one-shot corrector、state-guided re-observation、feedback/no feedback、reliability calibration。

### J. Scientific novelty

不能仅声称 first glove→motion prior / first confidence-aware refinement。最有希望的是**给定当前部分观测的状态恢复与候选引导证据查询同步迭代，同时维持可靠性标记的跨帧反馈**。该组合是否具备 CVPR 新意，要靠强基线下的显著、真实几何误差改进验证。

### K. Method 小节建议

3.1 State Representation and Visual Observations  
3.2 Trajectory-Only Conditional State Restoration  
3.3 Evidence-Guided Iterative State Correction  
3.4 Reliability-Aware Temporal Feedback and Training

详细且最新的工程规格请以 `START_HERE.md` 为准。

## 15.24 第二次完整导出（本次任务）

**用户：**

> 你帮我把上面的完整聊天记录导出，我需要创建chatgpt工作项目，开始尝试这个idea

**归档处理：**

- 保留以前 795 行早期研究讨论归档；
- 追加本节，覆盖以后每一轮研究问题、重要修正、方法接口、写作和投稿讨论；
- 单独提供 `START_HERE.md`（新 Work 项目最重要的当前方案）、`02_Introduction_CN_EN.md`、`03_References_And_Implementation.md`、方法主图及 ZIP；
- **再次强调：这是一份详尽的、按对话进程整理的项目级内容存档，不是逐字逐句从 ChatGPT 服务器下载的官方 transcript。**

---

# 十六、截至 2026-10-13 的唯一现行方案（覆盖历史上已被放弃的选择）

> **Head-mounted RGB → frozen HaWoR → original observation \(Z_t\) and actual image features → trajectory-only pretrained conditional-state-restoration TWM → evidence-guided reliability-aware iterative corrector → final state + reliability feedback into history.**

\[
\begin{aligned}
(Z_t,F_t,\Gamma_t,M_t)&=\operatorname{HaWoR}(I_{1:T})\\
(E_t^{(k)},w_t^{(k)})&=\operatorname{VisualCheck}(Z_t,F_t,\Gamma_t,S_t^{(k)})\\
(U_t^{(k)},h_t^{W,(k)})&=W_\theta(\mathcal H_t,Z_t,S_t^{(k)},M_t,w_t^{(k)})\\
S_t^{(k+1)}&=S_t^{(k)}\oplus C_\psi(S_t^{(k)},Z_t,U_t^{(k)},E_t^{(k)},w_t^{(k)})\\
\mathcal H_{t+1}&=\operatorname{Update}(\mathcal H_t,\hat S_t,q_t).
\end{aligned}
\]

TWM 预训练：

\[
\widetilde S_t=\operatorname{Corrupt}(S_t^{GT}),\quad
\mathcal L_W=\mathcal L_{\mathrm{restore}}+\lambda_f\mathcal L_{\mathrm{forecast}}+\lambda_b\mathcal L_{\mathrm{bone}}.
\]

**当前真正需要动手回答的第一个问题：** 在真实 HaWoR 输出上，相同训练和 GT 条件下，`glove-trained conditional state restoration` 是否能优于 `vision-only + Kalman/smoothing + ordinary temporal model`？如果没有明显收益，先检查状态坐标、GT 骨架匹配、观测可靠性，勿立即增加更复杂的模型。

**要点：先做可反驳的消融，再宣称具有理解物理过程的能力。**



---

# 项目材料：论文 Introduction 中英文草稿

# Introduction 草稿存档（中文版 / 英文版）

> 研究暂定标题：**Are Gloves Necessary? Learning to Correct Visual Hand Reconstruction with Motion Priors**  
> 状态：讨论期间的 **writing draft**，不是最终定稿。相关论文名称（DexCap / HaWoR / HMP / StableHand）需在 Overleaf 插入准确 BibTeX；尚未有实验结果，不可当成已验证结论。  
> 本文记录这段对话中逐步确认的**首段最终版本**，以及讨论后的中文/英文完整 Introduction 主稿。此前多个中间版本和修改理由见 `01_Conversation_Research_Archive.md`。

---

# A. 开头段落：确定保留的中文版

准确恢复人手的三维运动，是将人类操作演示转化为机器人学习数据的重要基础。近年来，基于视觉的手部运动重建方法为低成本、非侵入式的数据采集提供了新的可能，使得仅通过相机记录人类操作并恢复三维手部轨迹成为可能。然而，纯视觉方法仍然容易受到手物遮挡、深度歧义、视角变化及运动漂移等因素的影响，难以在复杂操作场景中持续获得高精度、稳定的手部运动状态。因此，在对数据精度与鲁棒性要求较高的机器人操作数据采集中，研究者仍广泛采用数据手套、空间跟踪设备等专用传感器。以 DexCap 为代表的采集系统结合数据手套、空间跟踪与视觉传感器，能够在一定程度上克服纯视觉方法的局限，获得更加可靠的手部运动数据。然而，这种对专用采集设备的依赖也带来了额外的硬件成本、设备标定与部署负担，使大规模、跨场景的人类操作数据采集面临挑战。这促使我们思考：**手套是否必须始终作为采集过程的一部分，还是可以将其提供的高质量运动知识迁移到无需佩戴手套的视觉采集系统中？**

# B. 开头段落：确定保留的英文版

Accurate reconstruction of three-dimensional hand motion is fundamental to transforming human demonstrations into valuable training data for robotic learning. In recent years, vision-based hand motion reconstruction methods have opened up new possibilities for low-cost, non-intrusive data collection, enabling the recovery of 3D hand trajectories directly from camera observations. However, purely vision-based approaches remain vulnerable to hand–object occlusions, depth ambiguities, viewpoint variations, and motion drift, making it challenging to consistently recover accurate and stable hand motion in complex manipulation scenarios. Consequently, researchers continue to rely on specialized sensing devices, such as instrumented gloves and spatial tracking systems, for robotic manipulation data collection where accuracy and robustness are critical. Representative systems, such as DexCap, integrate instrumented gloves, spatial trackers, and visual sensors to overcome some of the limitations of purely visual approaches and capture more reliable hand motion data. Nevertheless, this reliance on specialized hardware introduces additional costs, calibration requirements, and deployment overhead, posing significant challenges to large-scale human demonstration collection across diverse environments. This naturally raises a fundamental question: **Are instrumented gloves truly necessary throughout the data collection process, or can the high-fidelity motion knowledge they provide be transferred to a glove-free visual reconstruction system?**

---

# C. 完整中文版（讨论的最后一版）

## 1. Introduction

准确恢复人手的三维运动，是将人类操作演示转化为机器人学习数据的重要基础。近年来，基于视觉的手部运动重建方法为低成本、非侵入式的数据采集提供了新的可能，使得仅通过相机记录人类操作并恢复三维手部轨迹成为可能。然而，纯视觉方法仍然容易受到手物遮挡、深度歧义、视角变化及运动漂移等因素的影响，难以在复杂操作场景中持续获得高精度、稳定的手部运动状态。因此，在对数据精度与鲁棒性要求较高的机器人操作数据采集中，研究者仍广泛采用数据手套、空间跟踪设备等专用传感器。以 DexCap 为代表的采集系统结合数据手套、空间跟踪与视觉传感器，能够在一定程度上克服纯视觉方法的局限，获得更加可靠的手部运动数据。然而，这种对专用采集设备的依赖也带来了额外的硬件成本、设备标定与部署负担，使大规模、跨场景的人类操作数据采集面临挑战。这促使我们思考：**手套是否必须始终作为采集过程的一部分，还是可以将其提供的高质量运动知识迁移到无需佩戴手套的视觉采集系统中？**

实现这一目标的关键挑战之一，在于视觉手部重建结果具有不均匀的可靠性。尽管三维手部姿态估计已经取得了显著进展，重建误差通常并不会均匀分布在整只手上。在典型的操作场景中，手腕与掌心可能已经被准确定位，而部分手指由于物体遮挡或视觉歧义，仍然存在较大的估计误差。此外，视觉模型给出的高置信度并不一定意味着几何估计准确，相机运动估计中的误差也可能进一步影响世界坐标系下的手部轨迹。因此，无条件信任视觉预测，或对所有状态分量施加统一的时间连续性约束，都难以充分解决这一问题。更有效的重建方法应当能够区分可靠与错误的视觉观测，尽可能保留已有的准确估计，并利用手部运动中的结构关系与时序规律恢复不确定的部分。更重要的是，**可靠的视觉信息不应仅仅被保留，还应能够帮助修正那些无法直接观测到的手部状态。**

近年来，第一视角手部重建与学习型运动先验的研究为解决上述问题提供了重要基础。HaWoR 将时序手部重建与相机轨迹估计相结合，并通过运动补全恢复第一视角视频中的世界坐标手部运动。在视觉重建之外，HMP 表明，从高质量运动捕捉数据中学习的手部运动先验，能够有效辅助遮挡条件下的姿态恢复。更近期的 StableHand 进一步引入质量感知的生成式重建机制，在保留可靠观测的同时恢复不可靠的手部运动。这些研究共同说明，时序运动知识、运动先验以及观测可靠性建模，对于准确恢复手部运动具有重要价值。在此基础上，我们进一步关注一个互补的研究问题：**能否将独立学习的高质量手部运动知识转化为持续更新的状态空间先验，使可靠的视觉观测不仅能够恢复不确定的手部姿态，还能够帮助纠正不断积累的状态估计误差？** 与主要利用运动知识进行轨迹外推或生成不同，我们希望将其直接融入当前物理状态的估计与校正过程。

我们的核心观察是，高精度运动捕捉数据的价值并不仅限于提供准确的逐帧监督信号。在大量不同的操作序列中，这些数据还隐含着手腕运动、手指关节变化以及连续状态演化之间相对稳定的关联规律。这些结构与时序关系可以独立于视觉观测进行学习，并在图像信息缺失或不准确时，为手部状态恢复提供额外约束。然而，这种运动知识不应直接取代当前的视觉测量。人手运动具有复杂性与不确定性，即使符合历史运动规律的预测，也未必对应真实发生的动作。例如，人的手可能突然停止或改变运动方向，此时新的视觉证据应当能够推翻原有的运动预期。基于这一认识，我们将手部运动重建重新表述为一个**运动先验引导的状态校正问题（Motion-Guided State Correction）**，其中当前视觉证据与学习到的运动规律共同约束潜在的真实物理状态。可靠的视觉观测为重建提供锚点，而运动先验则用于恢复与当前证据不一致或无法直接观测的状态分量。

为实现这一目标，我们提出一个在低维物理状态空间中学习的**轨迹世界模型（Trajectory World Model, TWM）**。该模型完全基于高精度数据手套或运动捕捉轨迹进行独立训练，不依赖图像输入或配对的视觉观测。具体而言，我们通过手腕位姿与手指关节构型表示人手的物理状态，使模型能够学习关节之间的空间依赖关系以及连续状态之间的时序关联。不同于主要关注未来状态预测的传统轨迹模型，我们将当前状态恢复作为 TWM 的主要学习目标：通过人为构造不完整或受到扰动的运动状态，使模型学习如何结合历史轨迹与当前部分状态，恢复更加准确的手部构型。同时，我们引入短期状态预测作为辅助训练目标，促进模型学习运动演化与时间连续性规律。通过这一设计，TWM 能够在不依赖视觉输入的情况下获得可迁移的运动知识。更重要的是，这些知识最终用于**恢复当前真实的手部状态，而不仅仅是预测下一时刻手可能移动到的位置。**

进一步地，我们通过一个**可靠性感知的迭代状态校正机制（Reliability-Aware Iterative State Correction）**，将学习到的 TWM 与视觉重建模型相结合。对于输入的第一视角视频，我们采用 HaWoR 获取初始手部状态估计，以及用于后续校验的视觉与几何信息。在此基础上，状态校正器同时从视觉证据和 TWM 学习到的运动关系两个方面评估当前候选状态。一方面，视觉分支判断当前姿态是否受到图像证据的支持；另一方面，TWM 根据手部结构与历史运动过程，为当前状态提供修复建议。校正器综合这些信息，动态重新评估不同状态分量的可靠性，并预测对当前手部构型的残差修正。与直接将视觉置信度作为固定权重的方法不同，我们将视觉观测视为可以被重新检验的软约束，从而允许可靠的可见关节通过结构与时序关联，指导遮挡或错误关节的恢复。同时，系统不会仅仅因为新的视觉观测与历史运动趋势不一致就将其拒绝，从而能够适应真实运动中的突然变化，避免错误的运动先验对准确视觉信息造成过度约束。

我们方法的一个核心特点，是**视觉证据与学习型运动知识之间的闭环交互机制**。这一机制包含两个相互配合的反馈过程。在当前帧内部，经过校正的候选状态被用于重新定位相关视觉证据，并再次评估其与已学习运动规律的一致性，从而逐步优化当前手部构型。在连续时间维度上，最终校正后的物理状态被反馈至 TWM，作为后续状态恢复的运动上下文。通过这两个闭环，模型能够不断将新的视觉观测与已有的运动状态相协调，而不是仅依赖固定的时序平滑或彼此独立的逐帧估计。训练过程中，我们首先利用纯运动轨迹数据学习独立的运动先验，再利用同步的视觉观测与高质量运动捕捉数据监督状态校正过程。此外，我们通过闭环展开训练，使模型能够适应其自身递归估计过程中产生的状态误差。在推理阶段，整个框架仅依赖头戴相机采集的视觉信息，无需使用数据手套或其他穿戴式运动捕捉设备。

通过上述设计，我们希望缩小低成本视觉采集与高精度传感器采集之间的性能差距。我们将重点研究，从高质量轨迹中学习的运动知识，能否在部分遮挡、视觉估计不可靠以及复杂相机运动等条件下，有效提升当前手部状态的恢复精度。除了整体姿态重建误差，我们还关注模型能否保留原本准确的视觉测量、利用可靠的可见关节恢复不可见部分，以及在连续估计过程中维持准确且一致的运动状态。同时，我们将考察独立训练的 TWM 能否跨越不同被试、采集环境及视觉观测条件发挥作用。我们的最终目标并非仅仅生成更加平滑或符合运动先验的手部轨迹，而是**尽可能准确地恢复真实发生的手部运动过程，从而缩小纯视觉重建与高精度运动捕捉之间的精度差距。**

本文的主要贡献总结如下：

1. **基于轨迹学习的物理状态恢复。** 我们提出一种独立于视觉观测的轨迹世界模型，通过高精度运动捕捉数据学习人手的空间结构与时序运动规律。不同于以未来运动预测为核心的建模方式，我们将受损状态恢复作为主要学习目标，并利用短期预测辅助学习运动过程，从而获得面向不完整物理状态校正的运动先验。
2. **可靠性感知的视觉状态迭代校正。** 我们设计一种结合视觉证据与学习型运动先验的迭代状态校正机制，能够动态评估视觉观测的可靠性，并利用可靠的局部状态及其结构与时序依赖关系，修复缺失或错误的手部状态分量。
3. **闭环第一视角手部运动重建。** 我们构建一个结合帧内迭代校正与跨帧状态反馈的闭环重建框架，使从高精度传感器轨迹中学习到的运动知识能够持续参与视觉状态估计，实现推理阶段无需穿戴式运动捕捉设备的手部运动恢复。

---

# D. 完整英文版（讨论的对应完整稿）

## 1. Introduction

Accurate reconstruction of three-dimensional hand motion is fundamental to transforming human demonstrations into valuable training data for robotic learning. In recent years, vision-based hand motion reconstruction methods have opened up new possibilities for low-cost, non-intrusive data collection, enabling the recovery of 3D hand trajectories directly from camera observations. However, purely vision-based approaches remain vulnerable to hand–object occlusions, depth ambiguities, viewpoint variations, and motion drift, making it challenging to consistently recover accurate and stable hand motion in complex manipulation scenarios. Consequently, researchers continue to rely on specialized sensing devices, such as instrumented gloves and spatial tracking systems, for robotic manipulation data collection where accuracy and robustness are critical. Representative systems, such as DexCap, integrate instrumented gloves, spatial trackers, and visual sensors to overcome some of the limitations of purely visual approaches and capture more reliable hand motion data. Nevertheless, this reliance on specialized hardware introduces additional costs, calibration requirements, and deployment overhead, posing significant challenges to large-scale human demonstration collection across diverse environments. This naturally raises a fundamental question: **Are instrumented gloves truly necessary throughout the data collection process, or can the high-fidelity motion knowledge they provide be transferred to a glove-free visual reconstruction system?**

A key challenge lies in the heterogeneous reliability of visual hand reconstruction. Despite substantial progress in 3D hand pose estimation, reconstruction errors are rarely distributed uniformly across the hand. In a typical manipulation scenario, the wrist and palm may be accurately localized, while individual fingers are partially occluded or incorrectly estimated. Moreover, high prediction confidence does not necessarily guarantee geometric accuracy, and errors in camera motion estimation can further affect the reconstructed hand trajectory in world coordinates. These observations suggest that neither blindly trusting visual predictions nor uniformly enforcing temporal consistency is sufficient. Instead, an effective reconstruction framework should distinguish reliable observations from erroneous ones, preserve accurate measurements, and exploit the structural and temporal relationships of hand motion to recover uncertain components. Crucially, **reliable visual evidence should not merely be preserved; it should also help correct other parts of the hand that cannot be directly observed.**

Recent advances in egocentric hand reconstruction and learned motion priors have made important progress toward this goal. HaWoR integrates temporal hand reconstruction with camera trajectory estimation and motion infilling to recover world-space hand motion from egocentric videos. Beyond visual reconstruction, HMP demonstrates that motion priors learned from high-quality motion capture data can improve hand pose recovery, particularly under occlusion. More recently, StableHand introduces quality-aware generative refinement to preserve reliable observations while reconstructing unreliable hand motion. Together, these approaches highlight the potential of temporal reasoning, motion priors, and observation reliability modeling. Building upon these advances, we investigate a complementary direction: **Can motion knowledge learned independently from high-fidelity hand trajectories serve as a continuously updated state-space prior, enabling reliable visual observations to correct both uncertain hand configurations and accumulated estimation errors?** Rather than treating motion knowledge primarily as a means of extrapolating or generating trajectories, we seek to integrate it directly into the estimation of the current physical state.

Our central insight is that high-quality motion capture data provide more than accurate frame-wise supervision. Across diverse manipulation sequences, they encode recurring relationships among wrist motion, finger articulation, and temporal state evolution. Such relationships can be learned independently of visual observations and subsequently used to constrain hand reconstruction when image evidence is incomplete or inaccurate. Importantly, this knowledge should not replace visual measurements with a predetermined motion prediction. Human movements are inherently variable, and even a plausible motion prior may disagree with an actual abrupt movement. We therefore formulate hand motion reconstruction as a **motion-guided state correction problem**, in which current visual evidence and learned motion regularities provide complementary constraints on the underlying physical state. Reliable measurements anchor the reconstruction, while the learned motion model helps recover inconsistent or unobserved components.

To realize this idea, we introduce a **Trajectory World Model (TWM)** trained exclusively on high-fidelity hand motion trajectories in a low-dimensional physical state space. The model represents hand motion through wrist poses and articulated finger configurations, capturing both spatial dependencies among joints and temporal relationships across consecutive states. Unlike conventional trajectory models that primarily emphasize future prediction, our TWM is trained to recover corrupted or partially observed states conditioned on their motion history. Short-horizon prediction is incorporated as an auxiliary objective to encourage the learning of motion evolution and temporal consistency. This design allows the model to acquire motion knowledge without requiring image inputs or paired visual observations during its pretraining stage. More importantly, the resulting representation is intended to support the recovery of the **current** hand configuration, rather than simply predicting where the hand is likely to move next.

We further integrate the learned TWM with a visual reconstruction backbone through a **reliability-aware iterative state correction mechanism**. Given an egocentric video, HaWoR provides an initial estimate of the hand motion together with the visual and geometric information needed for subsequent verification. The corrector then jointly evaluates the current state hypothesis against visual evidence and the motion relationships captured by the TWM. Instead of treating confidence scores as fixed indicators of correctness, it progressively reassesses the reliability of individual state components and predicts corrections to the reconstructed hand configuration. In this process, reliable observations of visible joints can constrain the recovery of occluded joints through their learned structural and temporal dependencies. At the same time, observations that are inconsistent with historical motion are not automatically rejected, allowing genuine changes in hand movement to override inaccurate motion assumptions.

A defining characteristic of our framework is its **closed-loop integration of visual evidence and learned motion knowledge**. Within each frame, the updated state hypothesis is used to revisit relevant visual evidence and reevaluate motion consistency, enabling successive refinements of the current configuration. Across frames, the final corrected state is fed back into the trajectory model, providing updated context for subsequent reconstruction. These two feedback mechanisms allow the system to continuously reconcile new observations with previously estimated motion, rather than relying solely on fixed temporal smoothing or independent frame-wise predictions. During training, motion-only datasets are used to learn the trajectory model, while synchronized visual observations and high-quality motion capture measurements supervise the correction mechanism. The framework is further trained under recursive state feedback to improve robustness to estimation errors accumulated over time. At inference time, it operates solely on egocentric camera observations, without requiring instrumented gloves or other wearable motion capture devices.

Our framework aims to bridge the gap between the accessibility of vision-based data collection and the accuracy of instrumented motion capture. We investigate whether motion knowledge acquired from high-fidelity trajectories can improve current-state reconstruction under partial occlusion, unreliable visual predictions, and challenging camera movements. Beyond overall reconstruction accuracy, we focus on the ability to preserve accurate visual measurements, recover unobserved hand components using reliable evidence, and maintain consistent state estimates across time. We also examine whether a trajectory model trained independently of the visual backbone can generalize across different subjects, environments, and acquisition settings. Ultimately, our objective is not merely to produce smoother or more plausible hand trajectories, but to reconstruct hand motion that more faithfully reflects the underlying physical process.

Our main contributions are summarized as follows:

1. **Trajectory-driven physical state recovery.** We develop a trajectory world model that learns structural and temporal hand motion relationships independently from high-fidelity motion capture data. By prioritizing corrupted-state recovery and using short-horizon prediction as auxiliary supervision, the model learns motion priors tailored to correcting incomplete physical states.
2. **Reliability-aware visual state correction.** We introduce an iterative correction mechanism that integrates learned motion knowledge with visual evidence, allowing reliable observations to guide the recovery of missing or inaccurate hand components while dynamically reassessing observation reliability.
3. **Closed-loop egocentric hand motion reconstruction.** We establish a framework that combines within-frame iterative correction and cross-frame state feedback, enabling motion priors learned from instrumented trajectories to support camera-only hand reconstruction without requiring wearable motion capture devices during inference.

---

# E. 后续写作注意

- 标题中的“without gloves”严格指推理/实际大规模采集阶段；**训练仍需 glove/MoCap**。
- “学到手部动力学”应更准确表达为 kinematic motion regularities / learned motion priors；位置轨迹不保证包含接触力、肌肉动作、真实 torque。
- HMP / StableHand 等已覆盖部分动机；**不要声称首次提出 motion prior 或 quality-aware refinement**。
- 没有实证前，避免 “achieves / significantly outperforms / proves”；当前草稿用 “we aim / we investigate”。
- 最终方法中“State-guided Visual Re-observation”与“Confidence Re-assessment”还需明确网络实现和消融；如果未实现或没有效果，引言中的相应 claim 要降级。



---

# 项目材料：代码入口、参考文献与实验计划

# References, Code Interfaces, and First Experiments

> 用于新的 Work 项目开始实际 code/reproduction。名称、license、权重下载方式、数据使用协议、GT 标签定义以及 2026-10 之后的更新应进入对应仓库再次核对；**这里只列讨论中已出现的资源及能明确识别的接口，不将研究中的推测当成已验证实现**。

## 1. 相关工作（方法优先）

| 论文 / 系统 | 初步用途 | 官方/论文链接 | 注意事项 |
|---|---|---|---|
| **HaWoR (CVPR 2025)** | **已决定的 egocentric visual backbone**；camera-space hand、world-space alignment、temporal processing | [GitHub](https://github.com/ThunderVVV/HaWoR)、[arXiv](https://arxiv.org/abs/2501.02973) | 记录代码版本、权重、相机参数和 temporal lookahead |
| **WestWorld (2026)** | 借鉴 structured trajectory modeling，**不用机器人 checkpoint 或原始动作条件训练任务** | [GitHub](https://github.com/511205787/WestWorld) | robot actions + multi-system structure + discretized channels，不是直接 69D wrist/hand model |
| **HMP (2024)** | 早期独立 MoCap motion prior + visual hand recovery | [arXiv](https://arxiv.org/abs/2312.16737) | 重要 novelty 基线 |
| **StableHand (2026)** | 视觉质量感知的运动重建及 flow/trajectory prior | [arXiv](https://arxiv.org/abs/2605.18553) | 与本工作很相近；研究差异要通过强实验 |
| **PAD-Hand** | 物理/不确定性感知的手部运动恢复 | 查 CVPR 2026 官方文献 | 与 confidence+state recovery 相关，引用时核实 |
| **EgoInfinity (2026)** | metric geometry + RGB hand-object reconstruction；历史替代视觉方案 | [GitHub](https://github.com/Rice-RobotPI-Lab/EgoInfinity) | 非最终决定的主 backbone |
| **Do as I Do (2026)** | 下游 reconstruction→robot retargeting / validation | 按论文标题搜索官方项目 | 不必完整复刻 |
| **WiLoR (2025)** | single-frame RGB hand baseline；HaWoR 原有相关组件 | CVPR OpenAccess 查核 | 不取代已决定的 HaWoR |
| **MINT / OmniHands / ACE-Ego-Hand / ViDiHand / EgoFound3R** | 后续 backbone / occlusion 强基线候选 | 以 2026 官方仓库为准 | code/train checkpoints availability 随时变化 |
| **ManipTrans / DemoDiffusion** | residual trajectory refinement 的邻近思想 | 按论文名检索 | 更偏 retargeting，与 hand visual estimation 不能混同 |

请不要在论文中未经验证就使用归档中曾出现的精度数字、SOTA 排名或“首次提出”断言。

## 2. 数据资源（训练和评测用，数据集论文不是主要 Method）

| 数据来源 | 讨论中的用法 | 链接 | 核查事项 |
|---|---|---|---|
| **HOT3D** | 第一视角 RGB + professional MoCap/reference；paired training/eval | [GitHub](https://github.com/facebookresearch/hot3d)、[项目站](https://facebookresearch.github.io/hot3d/) | 哪些 split 有公开 3D GT；camera stream 的单/多视角；时间对齐；hand shape/topology |
| **DexCap** | instrumented glove + RGB-D + wrist pose，motion-only 和 paired | [GitHub](https://github.com/j96w/DexCap)、[论文](https://arxiv.org/abs/2403.07788) | Rokoko joint / T265 wrist SLAM 的不同误差来源、标定变换、数据访问协议 |
| **EgoDex** | 大规模运动预训练候选 | [GitHub](https://github.com/apple-aiml-research/ml-egodex) | Vision Pro 设备视觉 tracking **不等于独立 glove GT**，应质量分层 |
| **AMASS** | 额外高质量人体 MoCap motion pool | [项目主页](https://amass.is.tue.mpg.de/) | 哪些子集有 hand articulation、license、骨架映射 |
| **ARCTIC** | occlusion / HOI 验证候选 | 原论文官方 | 标签空间、相机模态、与 HaWoR 是否同条件 |

## 3. 官方 HaWoR 代码的真实接口（优先阅读）

1. `https://github.com/ThunderVVV/HaWoR/blob/main/scripts/scripts_test_video/hawor_video.py`
   - `hawor_motion_estimation(...)` 读取 tracklets，bbox 缺帧可做插值，再运行模型 inference。
   - 输出中看到 `init_root_orient`, `init_hand_pose`, `init_trans`, `init_betas` 等，并保存 `cam_space/.../*.json`。
   - **原始 `valid = ...['det']` 会在 bbox interpolation 后改变**；需要在插值前保存 raw `det` 才能区分 observed/inferred。
   - `hawor_infiller(...)` 可独立调用，第一版 **建议绕开而作为 baseline**。

2. `https://github.com/ThunderVVV/HaWoR/blob/main/lib/models/hawor.py`
   - `HAWOR.forward_step` 通过 `self.backbone`、`self.st_module`、`self.mano_head`、`self.motion_module` 输出 pose/shape/trans。
   - 代码里 2D joints 可以通过 `self.project(j3d, ...)` 生成，即 `pred_keypoints_2d` 有可能只是模型自己的重投影，不可与同源投影比较并宣称得到了图像检验。
   - 图像特征中间张量可以通过明确的 hook/返回值接口进行缓存，先从最小跑通版本开始，不必重新训练 HaWoR。

3. `https://github.com/511205787/WestWorld/blob/main/models/WestWorld/westworld.py`
   - 使用 observation/action channels、structural embeddings、bin discretization next-token loss 等。
   - 模型代码复杂度与人体手 current-state restoration 任务并不完全匹配，**只借鉴相应结构思想**。

## 4. 数据接口 schema（我们自己定义，不是某数据集的原始格式）

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class HandFrame:
    timestamp_s: float
    handedness: str                 # "left" or "right"

    # Standardized world-space wrist pose.
    wrist_pos_world: np.ndarray     # (3,), meters
    wrist_rot_world: np.ndarray     # (3,3), proper rotation matrix

    # Local 3D hand joints excluding wrist/root.
    joints_local: np.ndarray        # (20,3), meters

    # Original visual measurement availability.
    raw_detected: bool              # before bbox interpolation
    joint_valid_mask: np.ndarray    # (20,), observed or missing, may be estimated
    camera_K: Optional[np.ndarray]  # (3,3)
    camera_T_world: Optional[np.ndarray]  # (4,4)

    # Optional training-only reference with separate provenance.
    gt_wrist_pos_world: Optional[np.ndarray]
    gt_wrist_rot_world: Optional[np.ndarray]
    gt_joints_local: Optional[np.ndarray]

    # Metadata for coordinate/time/GT provenance.
    sequence_id: str
    subject_id: str
    dataset_name: str
    gt_source: str                  # optical mocap / glove / device tracking / none
```

这是约定 schema 而不是已运行的 loader；不同数据集可能缺部分字段。`joint_valid_mask` 表示标注的可用性，不自动等同“模型准确度”；可靠性由训练的误差头预测。

最终训练张量可以例如：

- `state`: `[B,K,69]`（保存 6D wrist rotation + 20×3 wrist-local joints）；
- `current_state`: `[B,69]`；
- `raw_observation`: `[B,69]`；
- `obs_mask`: `[B,21]` 或 `[B,22]`，需要明确定义 root/各关节；
- `dt`: `[B,K]`；
- `history_quality`: `[B,K,num_parts]`；
- `visual_feature`: 特征图 `[B,C,H,W]` 或相应视频特征；
- `gt_state`: `[B,69]`（只在训练/评测时读）。

注意：采用 root+20 关节时，是否把 palm 作为单独 token 以及有无重叠关节，需要通过具体 skeleton mapping 确认，不要直接把“21 joints”与“21 non-root joints”混淆。

## 5. PyTorch 目录建议

```text
handloop/
  README.md
  configs/
    data.yaml
    twm.yaml
    corrector.yaml
  third_party/
    HaWoR/                  # Git submodule or pinned commit
  datasets/
    gt_adapters/
      hot3d.py
      dexcap.py
      egodex.py
    schemas.py
    canonical_hand.py
    timestamps.py
  models/
    twm.py
    corruption.py
    image_evidence.py
    corrector.py
    state_geometry.py
  scripts/
    extract_hawor.py
    verify_coordinates.py
    train_twm.py
    evaluate_twm.py
    train_corrector.py
    rollout_eval.py
  metrics/
    mpjpe.py
    wrist_pose.py
    visibility.py
  tests/
    test_geometry.py
    test_hand_flip.py
    test_pose_roundtrip.py
    test_timestamp_alignment.py
```

## 6. 第一轮必须跑的 Sanity Checks

- 单手 3D points 经 world→camera 投影，与视频图像真实可见位置接近；左右手 flip 对齐。
- wrist-relative state→world→local roundtrip numerical error 足够小；rotation matrices 正交、det 接近 +1。
- glove/mocap GT 与 HaWoR 无 GT 泄漏对齐；严格按照同步采集时间戳，不做每帧 GT 优化对齐。
- `clean GT -> TWM -> GT` 输出稳定；缺失/扰动数据修复的效果必须明显胜过简单最后一帧复制（在实际有区别的数据上）。
- 对所有输入都被标记 missing 时预测 uncertainty 应上升，不能产生虚假的 100% confidence。
- 视觉原始正确的点不能在修复后大量退化；覆盖“真 high confidence 错误”和突发转向测试。
- 短时回放测试：若关闭帧内 re-observation，N=2 与 N=1 不能轻率宣称区别来自新证据。
- 监督 vs 评测 GT 来源不同、训练前处理与推理前处理一致。
- 检查 legacy HaWoR infiller/temporal component 的 overlap；定量比较 `HaWoR original/full` 版本。

## 7. 最重要的初始评测表（实验前空模板）

| Model | W-MPJPE↓ | Wrist translation↓ | Visible joints↓ | Occluded joints↓ | Recovery after reappearance↓ |
|---|---:|---:|---:|---:|---:|
| HaWoR full | TBD | TBD | TBD | TBD | TBD |
| HaWoR raw (before infilling) | TBD | TBD | TBD | TBD | TBD |
| Raw + Kalman | TBD | TBD | TBD | TBD | TBD |
| Raw + Temporal Transformer | TBD | TBD | TBD | TBD | TBD |
| Raw + prediction-only TWM | TBD | TBD | TBD | TBD | TBD |
| Raw + restoration TWM | TBD | TBD | TBD | TBD | TBD |
| + evidence-guided iterative corrector | TBD | TBD | TBD | TBD | TBD |
| + confidence re-assessment / feedback (full) | TBD | TBD | TBD | TBD | TBD |

**所有数值待实验，绝不能从历史讨论里随意引用示例数字。**

## 8. 第一周交付目标（建议，而非已开始工作）

- 下载/pin HaWoR 代码与 checkpoint，核对 license、dependencies、最低 GPU；
- 核验一小份 HOT3D reference+camera 与 HaWoR 输出能否同 timestamp 对齐；
- 实现 skeleton state adapter + plotting sanity check；
- 构造 pure trajectory corruption synthetic batch 并训练一个简单 restoration baseline；
- 用真实 HaWoR prediction errors 试验恢复模型的有效性，和 Kalman/temporal baseline 对比。

只有得到真实结果，才开始给 CVPR Intro 和 abstract 写任何 numerical claim。

