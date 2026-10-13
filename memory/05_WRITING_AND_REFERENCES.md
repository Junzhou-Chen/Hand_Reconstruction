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

“传感器数据学运动先验”“引入 confidence”“加入 temporal model”本身不是新贡献。2026-10-13全文精读进一步确认：DIP已有纯运动受损条件恢复和序列内预测反馈；DIR/WiLoR/GeoHand已有候选引导图像查询。以上候选贡献需要相对具体前作证明区别和收益，不能当成已确认的新颖点。完整版本、方法和源码证据见[论文精读](../docs/WESTWORLD_AND_HAND_RECONSTRUCTION_REVIEW.md)。

Method 建议四节：State Representation and Visual Observations；Trajectory-Only Conditional State Restoration；Evidence-Guided Iterative State Correction；Reliability-Aware Temporal Feedback and Training。最终写法应服从真正实现的模块。

## 关键方法来源核验

同日新增[表示与MANO标注调查](10_MANO_REPRESENTATION_AND_ANNOTATION.md)，核查HaMeR/WiLoR/HaWoR、HMP、StableHand54D、PAD-Hand公开源码及HandX/METRO坐标路线；同时记录DexYCB/HO3D/FreiHAND、marker库与HOT3D标签拟合链。本文较早的“未审实现/未核实”限定到当时范围，后续明确更新如下。

### HaWoR

[HaWoR 论文](https://arxiv.org/abs/2501.02973)与[官方仓库](https://github.com/ThunderVVV/HaWoR)支持其 camera-space 手重建、camera SLAM/world-space 转换及 motion infilling 的分工。

本次实际读取 [hawor.py](https://raw.githubusercontent.com/ThunderVVV/HaWoR/main/lib/models/hawor.py) 和 [hawor_video.py](https://raw.githubusercontent.com/ThunderVVV/HaWoR/main/scripts/scripts_test_video/hawor_video.py)，确认：

- `pred_keypoints_2d` 由 predicted 3D joints 经 `self.project` 生成，不能当独立观测自证该 pose。
- bbox 插值后局部 `valid` 区间会设为 True；需保存插值前 raw detection。没有据此确认原始 tracklet 中各条目的 `t['det']` 被永久修改。
- **`hawor_infiller()` 内也读取 SLAM 结果并调用 `cam2world_convert`。绕开补全网络时必须保留世界坐标转换，不宜直接跳过整个函数。**

后续已静态核验官方 `66c7d4108d58a716deccd192cb7645170cdc7bd7`：视觉侧16帧与infiller 120帧上下文均非严格因果；infiller仅替换missing slots，保留valid输出。feature hook、模型运行与延迟未实测。[固定源码和评测口径](../docs/WESTWORLD_AND_HAND_RECONSTRUCTION_REVIEW.md)

2026-10-13回答69D问题时进一步读[视频脚本](https://github.com/ThunderVVV/HaWoR/blob/main/scripts/scripts_test_video/hawor_video.py)、[MANO wrapper](https://github.com/ThunderVVV/HaWoR/blob/main/lib/models/mano_wrapper.py)、[world转换](https://github.com/ThunderVVV/HaWoR/blob/main/lib/eval_utils/custom_utils.py)。原生按手保存root rotation、15个关节rotation、3D translation、10D shape；wrapper有21点映射（可另加extra regressor）。视频脚本分左右手track，原infiller用`2*(3+10+16*6)=218`维双手表示。故“HaWoR原生单手69D”或“整个HaWoR只处理单手”均不准确；69D是我们经adapter构建的每手状态。world转换显式处理MANO translation与wrist joint offset，直接复制trans为wrist位置不正确。均为源码核验，无数值复现。

进一步查[filling_preprocess](https://github.com/ThunderVVV/HaWoR/blob/main/lib/eval_utils/filling_utils.py)确认逐手canonicalize后拼接两手218输入；但[论文§9](https://arxiv.org/html/2501.02973v1)称各手独立、无交互/不穿透约束。拼接是当前实现事实，不等于证明已学到双手交互；main代码与发表版本事实分开，复现前pin版本。

### HMP

[原论文](https://arxiv.org/abs/2312.16737)、[作者项目](https://hmp.is.tue.mpg.de/)和[官方代码](https://github.com/enesduran/HMP)对应 WACV 2024。以 AMASS 学习局部手指 VAE/NeMF prior，通过 latent optimization 辅助视觉恢复；原设定static camera，wrist位姿独立优化。后续已审 `35d799f76b2b2bc1d1e945117b021014b099e7e6` 的实际fitting路径；2D条件含MediaPipe与缺测时的初始3D投影fallback，并非全部独立图像测量。未复现效果。[精读](../docs/WESTWORLD_AND_HAND_RECONSTRUCTION_REVIEW.md)

数据专项调研进一步核验了其split和预处理：论文选GRAB/TCDHands/SAMP，但当前代码默认train为GRAB+ARCTIC、valid为SAMP、test为TCDHands。没有因此复现模型效果；细节和源码链接见[专项报告](07_TWM_DATASET_SURVEY_2026-10-13.md#amass-必须筛选子集)。[HaWoR论文](https://arxiv.org/html/2501.02973v1)的motion infiller在HOT3D训练，是配对/先验对照中需要单独记录的训练来源。

### StableHand

[论文](https://arxiv.org/abs/2605.18553)、[作者项目](https://huajian-zeng.github.io/projects/stablehand/)和[官方仓库](https://github.com/huajian-zeng/stablehand)描述质量信号、质量网络与 quality-aware flow matching。与保留可靠观测、恢复不可靠运动的目标接近。

仓库标注 NeurIPS 2026，并记录 2026-09-07 发布Enhanced推理、评测、权重和缓存数据；训练及预处理仍列 TODO。后续已审 `0471a45e325ffa167cc6a8484d712a21befd3e47`：增强版backbone、QN输入、wrist residual与指标定义均有变化，不能与v1表格混用。原论文QN不读image feature，但生成DiT读图像/scene/camera，因此整体不是纯轨迹先验；quality在flow循环内固定。模型未运行。[版本对照](../docs/WESTWORLD_AND_HAND_RECONSTRUCTION_REVIEW.md)

后续读取[论文§3](https://arxiv.org/html/2605.18553v1)确认每手54D=3trans+6wrist rotation+45finger axis-angle，双手108D；shape不含在该动态state中。未据此复现生成/质量模型。

本轮按用户截图核验[Appendix F/Table10](https://arxiv.org/html/2605.18553v1#A6)：八库约10M用于QN，joint-only三库只计算视觉prediction与reference MPJPE，不拟合MANO；生成器另在HOT3D/ARCTIC训练。Table10/Re:InterHand provenance与官方独立捕获不一致，EE4D“21M3D”与官方4.3M3D/21M2D不一致，采用官方数据事实。公开399段`stablehand-data`是评测缓存，不是八库训练全集。[逐库核验与TWM提案](../docs/TWM_PRETRAIN_PLAN.md)

### PAD-Hand

准确题名为 **PAD-Hand: Physics-Aware Diffusion for Hand Motion Recovery**，作者 Elkhan Ismayilzada、Yufei Zhang、Zijun Cui。[arXiv 2603.26068](https://arxiv.org/abs/2603.26068)注明 Accepted to CVPR 2026；[论文正文](https://arxiv.org/html/2603.26068v1)描述条件 diffusion、Euler–Lagrange residual virtual observables 与 last-layer Laplace uncertainty。

其 physics-consistency variance 不等同本项目按 GT 几何误差学习的 observation reliability。不将题名中的 physics 当作本项目必须添加物理引擎的理由。后续已精读2026-03-30的v2及官方 `ca9ed97bc671199c25cf569d3b1de0e6f7937251`，训练用真实视觉估计→GT配对；公开demo存在，但发布树未提供完整训练/LLLA执行路径。[精读](../docs/WESTWORLD_AND_HAND_RECONSTRUCTION_REVIEW.md)

后续已找到并审查[官方demo](https://github.com/DominoAI-Lab/PAD-Hand-CVPR-2026/blob/main/demo.py)和[model](https://github.com/DominoAI-Lab/PAD-Hand-CVPR-2026/blob/main/models/pad_hand.py)：Euler姿态48D扰动、MANO vertices为MeshConv输入、16×6旋转输出；demo在refinement中沿用初始cam_t，不据此认为在恢复world wrist translation。源码存在已核验，模型未运行。

### WestWorld

[v2全文](https://arxiv.org/html/2603.14392v2)与官方 `0a90776aeaa43f47c97f3c05eb75418b377b73ae` 已精读核验。任务为action-conditioned future-state预测；结构编码、通道attention、Mamba-style SSM与Sys-MoE结合。官方配置确认256 bins、6 blocks、hidden256、4 heads/experts，替代此前“bin数未核验”状态。它支持结构化轨迹建模的启发，不证明已具有人手受损当前状态恢复能力；只借鉴思想仍是项目决定。[详细方法与协议限制](../docs/WESTWORLD_AND_HAND_RECONSTRUCTION_REVIEW.md)

### 新增直接前作与图像取证

[DIP，ICCV2025](https://arxiv.org/html/2508.01835v1)的准确题名为 *Diffusion-based 3D Hand Motion Recovery with Intuitive Physics*：随机扰动GT训练纯运动条件恢复，固定初始估计与变化candidate分开，并有序列内自回归预测反馈。它是TWM最直接的恢复基线之一，不能将其归为仅未来预测。

[DIR，ICCV2023](https://arxiv.org/pdf/2302.02410)、WiLoR和GeoHand已核验候选引导视觉查询；WiLoR公开实现一次多尺度读取后更新，DIR/GeoHand明确多轮。另核验Dyn-HaMR、JoHan与AffHandGen/ACE-Ego-Hand的相邻机制，正式发表与预印本分开。所有方法位置和链接集中在[精读报告](../docs/WESTWORLD_AND_HAND_RECONSTRUCTION_REVIEW.md)，尚无同协议实验比较。

## 数据来源核验

同日方法整理新增[论文数据角色对照](../docs/METHOD_FRAMEWORK.md#9-数据选择采用用户最新的免申请原则)：明确区分HMP先验/恢复评测、HaWoR视觉/infiller、StableHand quality/generative模型、PAD-Hand训练/TACO跨域评测。用户已明确免申请数据优先，论文常用不自动意味着照搬其需注册数据组合。新判断来自原论文，不代表已复现。

### HOT3D

[官方项目](https://facebookresearch.github.io/hot3d/)说明 professional optical-marker MoCap 与 UmeTrack/MANO 标注。Aria 有 RGB 流，Quest 3 为双灰度流，所以本项目 RGB 首轮候选应选 Aria。[官方 README](https://github.com/facebookresearch/hot3d)明确其列出的 TEST 被试（P0004、P0005、P0006、P0008、P0016、P0020）没有公开 GT，MANO annotations 另有协议要求。未下载序列验证同步和映射。

新增标注链核验：[官方hand toolkit](https://github.com/facebookresearch/hand_tracking_toolkit#important-notes)说明MANO从UmeTrack求解，整体精度略低，建议用UmeTrack keypoints监督；[论文附录C](https://arxiv.org/html/2411.19167)给出19手marker+personal scan拟合UmeTrack流程。

同日访问修正：[HOT3D-Clips](https://github.com/facebookresearch/hot3d/blob/main/hot3d/clips/README.md)的官方HF train可直接获取，区别于full VRS manifest。2,804段150帧为累计约3.89小时，按原sequence/timestamp去重后才是净量；MANO/UmeTrack、shape、camera齐备但各帧hand有效性需筛选。

### DexCap

[原论文](https://arxiv.org/html/2403.07788v1)中手指局部位置来自 Rokoko glove，腕部 6DoF 来自 T265 SLAM，视觉为胸前 RGB-D。[官方仓库](https://github.com/j96w/DexCap)包含初始 SLAM 漂移修正工具。[官方仓库链接的数据卡](https://huggingface.co/datasets/chenwangj/DexCap-Data)仅列 30 分钟 wiping 和 60 分钟 packaging 的公开集。不能将公开集等同全部论文数据，也不能把腕轨迹视为无误差光学 MoCap。未验下载文件。

同日进一步读取[glove采集源码](https://github.com/j96w/DexCap/blob/main/STEP1_collect_data/redis_glove_server.py)：每手明确21个关节名、`(21,3)`位置和`(21,4)`orientation。源码同时产生middle-proximal中心化/骨长归一化和raw字段；实际release保存哪种及恢复尺度/标定仍须检查，不能因采集代码完整而声称发布样本已验收。

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
