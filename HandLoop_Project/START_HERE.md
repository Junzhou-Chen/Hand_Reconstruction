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
