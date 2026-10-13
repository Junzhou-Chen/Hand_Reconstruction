# HandLoop 写作与外部核验记忆

更新与外部核验日期：2026-10-13。项目已有中英文 Introduction 草稿，尚无结果。下列外部核验采用原论文、作者项目页和官方源码，范围是引用与接口，不是实验复现；网页和 main 分支会变化，实际实现仍需 pin commit。

## 标题与叙事

“HandLoop”是工作名，尚未做正式命名排重。标题偏好是 **Are Gloves Necessary? + 简洁的技术表达**；候选包括：

- Are Gloves Necessary? Learning Motion Priors for Visual Hand Correction
- Are Gloves Necessary? Learning to Correct Visual Hand Reconstruction with Motion Priors
- Are Gloves Necessary? Learning to Recover Hand Motion from Imperfect Vision

用户曾明确喜欢 “Learning to Correct Visual Hand Motion with Dynamics Priors” 的风格；使用 Motion Priors 更克制，是此前助手提出的术语建议，不冒充用户已最终选定词语。[标题讨论 §15.12–15.15](../HandLoop_Project/01_Conversation_Research_Archive.md)

已认可的首段按这个顺序展开：3D 手运动对机器人演示学习的价值 → 视觉采集便捷、低侵入 → 遮挡/深度/视角/漂移不足 → 高精度场景使用 glove 与 tracker → 部署负担和扩展困难 → 能否迁移 glove 的高质量运动知识，令实际采集免手套。

中英文首段全文和完整 Introduction 保存在 [02_Introduction_CN_EN.md](../HandLoop_Project/02_Introduction_CN_EN.md)。修改后文时保留用户认可的首段，除非有新的改写要求。不要将“高精度场景采用专用设备”扩大成没有证据的“整个行业绝对主流”。

## 候选论文贡献与所需证据

| 候选贡献 | 要证明什么 |
|---|---|
| Trajectory-only conditional current-state restoration | 当前可靠状态能帮助恢复错误/缺失状态，并优于 prediction-only 或普通时序模型 |
| Evidence-guided iterative correction with reliability reassessment | 候选改变真实图像查询，能修正假高置信错误，同时保护准确观测与真实动作突变 |
| Reliability-aware temporal feedback | 自身状态反馈和 rollout 能改善连续恢复与跨域鲁棒性，而非放大错误 |

“传感器数据学运动先验”“引入 confidence”“加入 temporal model”本身不是新贡献。不能仅凭相关工作的摘要判断其缺少本项目机制；需进一步逐项比较训练任务、条件输入、独立证据和反馈。

Method 建议四节：State Representation and Visual Observations；Trajectory-Only Conditional State Restoration；Evidence-Guided Iterative State Correction；Reliability-Aware Temporal Feedback and Training。最终写法应服从真正实现的模块。

## 关键方法来源核验

### HaWoR

[HaWoR 论文](https://arxiv.org/abs/2501.02973)与[官方仓库](https://github.com/ThunderVVV/HaWoR)支持其 camera-space 手重建、camera SLAM/world-space 转换及 motion infilling 的分工。

本次实际读取 [hawor.py](https://raw.githubusercontent.com/ThunderVVV/HaWoR/main/lib/models/hawor.py) 和 [hawor_video.py](https://raw.githubusercontent.com/ThunderVVV/HaWoR/main/scripts/scripts_test_video/hawor_video.py)，确认：

- `pred_keypoints_2d` 由 predicted 3D joints 经 `self.project` 生成，不能当独立观测自证该 pose。
- bbox 插值后局部 `valid` 区间会设为 True；需保存插值前 raw detection。没有据此确认原始 tracklet 中各条目的 `t['det']` 被永久修改。
- **`hawor_infiller()` 内也读取 SLAM 结果并调用 `cam2world_convert`。绕开补全网络时必须保留世界坐标转换，不宜直接跳过整个函数。**

未验证 feature hook 运行、权重下载、延迟和因果性；目前没有本地 pinned checkout。

### HMP

[原论文](https://arxiv.org/abs/2312.16737)、[作者项目](https://hmp.is.tue.mpg.de/)和[官方代码](https://github.com/enesduran/HMP)对应 WACV 2024。项目说明以 AMASS 学习手部生成运动先验，通过 latent optimization 辅助视频手运动估计。它是“独立 MoCap prior + visual recovery”的已有工作，必须纳入 novelty 分析。未全面审查模型实现或复现效果。

数据专项调研进一步核验了其split和预处理：论文选GRAB/TCDHands/SAMP，但当前代码默认train为GRAB+ARCTIC、valid为SAMP、test为TCDHands。没有因此复现模型效果；细节和源码链接见[专项报告](07_TWM_DATASET_SURVEY_2026-10-13.md#amass-必须筛选子集)。[HaWoR论文](https://arxiv.org/html/2501.02973v1)的motion infiller在HOT3D训练，是配对/先验对照中需要单独记录的训练来源。

### StableHand

[论文](https://arxiv.org/abs/2605.18553)、[作者项目](https://huajian-zeng.github.io/projects/stablehand/)和[官方仓库](https://github.com/huajian-zeng/stablehand)描述质量信号、质量网络与 quality-aware flow matching。与保留可靠观测、恢复不可靠运动的目标接近。

核验时仓库标注 NeurIPS 2026，并记录 2026-09-07 发布推理、评测、权重和缓存数据；训练及预处理仍列 TODO。这是 README 发布状态，未审模型源码或验证可运行性，也未据此证明本项目的新颖性。

### PAD-Hand

准确题名为 **PAD-Hand: Physics-Aware Diffusion for Hand Motion Recovery**，作者 Elkhan Ismayilzada、Yufei Zhang、Zijun Cui。[arXiv 2603.26068](https://arxiv.org/abs/2603.26068)注明 Accepted to CVPR 2026；[论文正文](https://arxiv.org/html/2603.26068v1)描述条件 diffusion、Euler–Lagrange residual virtual observables 与 last-layer Laplace uncertainty。

其 physics-consistency variance 不等同本项目按 GT 几何误差学习的 observation reliability。CVF 条目搜索可见但本次直接抓取失败；官方代码可用性尚未核实。不将题名中的 physics 当作本项目必须添加物理引擎的理由。

### WestWorld

[官方仓库](https://github.com/511205787/WestWorld)和实际读取的 [westworld.py](https://raw.githubusercontent.com/511205787/WestWorld/main/models/WestWorld/westworld.py)包含 observation/action channels、action conditioning、结构编码、离散通道与 observation next-token loss。

它支持轨迹空间建模的启发，不提供可直接拿来做人手 69D 当前状态恢复的依据。只借鉴思想是本项目决定；并非 WestWorld 官方对用途的限制。本次未核验具体 bin 数，旧文档里的 “256-bin” 不作为新增验证事实。

## 数据来源核验

### HOT3D

[官方项目](https://facebookresearch.github.io/hot3d/)说明 professional optical-marker MoCap 与 UmeTrack/MANO 标注。Aria 有 RGB 流，Quest 3 为双灰度流，所以本项目 RGB 首轮候选应选 Aria。[官方 README](https://github.com/facebookresearch/hot3d)明确其列出的 TEST 被试（P0004、P0005、P0006、P0008、P0016、P0020）没有公开 GT，MANO annotations 另有协议要求。未下载序列验证同步和映射。

### DexCap

[原论文](https://arxiv.org/html/2403.07788v1)中手指局部位置来自 Rokoko glove，腕部 6DoF 来自 T265 SLAM，视觉为胸前 RGB-D。[官方仓库](https://github.com/j96w/DexCap)包含初始 SLAM 漂移修正工具。[官方仓库链接的数据卡](https://huggingface.co/datasets/chenwangj/DexCap-Data)仅列 30 分钟 wiping 和 60 分钟 packaging 的公开集。不能将公开集等同全部论文数据，也不能把腕轨迹视为无误差光学 MoCap。未验下载文件。

### EgoDex

[官方 README](https://github.com/apple-aiml-research/ml-egodex)说明轨迹由 Apple Vision Pro 的 ARKit/visionOS 估计，SE(3) 使用 episode 的 ARKit origin，confidence 不是每个文件都有；合成 RGB 与 tracking 的 perspective mismatch 可能影响重投影。它不是独立 glove GT，不能把所有投影偏差归于手状态几何错误。未验证具体文件。

## 尚未完成的文献工作

AMASS hand-marker限制、GRAB/ARCTIC/OakInk2及多种手套库已在[2026-10-13数据专项调研](07_TWM_DATASET_SURVEY_2026-10-13.md)完成文档和部分元数据核验，尚无数据本体质量验收或复现。AssemblyHands及历史候选EgoInfinity、Do as I Do、MINT、OmniHands等未全面复核，不升级为已确认可用基线。

正式写作仍需准确 BibTeX、paper version、评测协议和可复现条件。会议日期、投稿规则和资源发布状态要临近使用时再查官方来源，不把旧会话讨论当作当前承诺。

## 写作边界

- “Glove-free”只指部署；预训练和 paired 监督仍使用高质量参考轨迹。
- 使用 motion regularities / learned motion priors；位置轨迹不自动包含 force、torque、contact 或 muscle dynamics。
- 保留假设语气；结果产生前不写 achieves、significantly outperforms、proves 或首次/SOTA 断言。
- 无法宣称运动先验唯一恢复长期全遮挡 GT，或独自消除纯全局 SLAM 漂移。
- 再取证与可靠性重估若未实现或未体现效果，论文对应贡献应降级。
