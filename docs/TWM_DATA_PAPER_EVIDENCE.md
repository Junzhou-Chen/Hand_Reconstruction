# TWM数据规划的论文依据

核验日期：2026-10-13。用户已选99D，要求以后制定plan广泛对照最新相关论文。本次重新检查纯运动先验、受损恢复、视频恢复和生成任务的训练数据；论文事实与本项目建议分开。现行执行提案见[预训练方案](TWM_PRETRAIN_PLAN.md)，尚无模型运行或完整数据验收。

## 1. 论文实际训练了什么

| 论文与版本 | 训练数据及任务 | 评测/留出 | 对TWM可借鉴的部分 |
|---|---|---|---|
| [HMP，WACV2024，§4.1](https://arxiv.org/html/2312.16737v1) | GRAB/TCDHands/SAMP训练motion prior；当前代码train GRAB/ARCTIC、valid SAMP、test TCDHands，与原论文分开 | HO3D/DexYCB恢复评测 | 高质量运动先验与视觉恢复分离；主要局部手先验，不能当完整world腕部99D方案 |
| [TaMF/OakInk2，CVPR2024](https://github.com/oakink/OakInk2-TaMF) | OakInk2的运动参数做任务/物体轨迹条件扩散生成 | 官方431/39/157段train/val/test；未证实subject-disjoint | OakInk2确有运动建模训练先例；只借运动数据/解码/split，不添加物体与任务条件 |
| [HaWoR，CVPR2025，§8](https://arxiv.org/html/2501.02973v1#S8) | 视觉网络用HOT3D/ARCTIC/DexYCB/HO3D；**infiller只用HOT3D** | HOT3D原录制110train/27val及DexYCB等 | 目标域轨迹与连续缺测；infiller两端条件/120帧不能等同past-only当前恢复 |
| [DIP，ICCV2025，§3.4/4.1](https://arxiv.org/html/2508.01835v1) | DexYCB train GT合成扰动训练纯运动恢复；另有真实HaMeR配对的model-specific版本 | HO3Dv2 zero-shot | 最直接的当前受损恢复先例；先合成、再真实前端适配；16帧与序列内反馈需公平对照 |
| [PAD-Hand，CVPR2026，v2](https://arxiv.org/html/2603.26068v2) | DexYCB/HO3D官方train，真实WiLoR估计作为恢复条件 | 两库eval，TACO未训练跨域 | 数值恢复也需真实上游错误分布；不是独立glove-only预训练；16帧 |
| [UniHand，ICLR2026，B.1](https://arxiv.org/html/2602.21631v1#A2.SS1) | DexYCB＋HOT3D train训练联合VAE/条件扩散；单checkpoint，含视觉/结构条件，48帧 | HO3D域外；HOT3D Aria 1272train/244test由官方train再分 | 支持MANO与不同结构条件；不能称纯轨迹TWM。此Sun等UniHand与BeingBeyond同名聚合/VLA资源不同 |
| [HandX，CVPR2026，§3/A](https://arxiv.org/html/2603.28766v1) | HoloAssist/GigaHands/HOT3D/ARCTIC/H2O＋自采，统一30Hz/60帧；生成训练 | 自建划分；公开base不含ARCTIC/H2O | 多源与来源标注；54.2h并非全MoCap。HoloAssist约49.3h占主体，不能照搬两手强动态过滤 |
| [StableHand，2026-05 v1，A/F](https://arxiv.org/html/2605.18553v1) | **八库约10M用于QN**；生成器按HOT3D/ARCTIC训练并接visual/scene/camera | HOT3D公开GT clips、ARCTIC s05；QN也排除s05 | 缺标签mask与质量训练角色；不能把八库QN规模当motion prior证据；150帧 |
| [HandFlow，2026-07 v1，B](https://arxiv.org/html/2607.11221v1#A2) | DexYCB/HOT3D train paired视觉flow，16帧 | DexYCB s0；HOT3D自行recording split，与其他论文不相同 | geometric/temporal监督与失效mask；非motion-only，也非统一公开HOT split |
| [ACE-Ego-Hand，2026-08 v3，§4.1/A/B](https://arxiv.org/html/2608.20308v3) | ARCTIC/HOT3D/H2O/OakInk2＋Re:InterHand＋FreiHAND/RHD，视觉恢复81RGB帧 | **HOI4D完全held-out** | OakInk2/Re可补真实操作/姿态；静态图复制只供姿态/外观，不能计真实motion时间 |
| [EgoFound3R，2026-10 v1，§4.1/B/D](https://arxiv.org/html/2610.01210v1) | H2O/HOT3D Aria/OakInk2/TACO/HOI4D/ARCTIC/STERA-10M；端到端视觉/scene多任务 | 内部held-out windows；STERA只训练 | 最新多库与masked监督先例；HOI4D在该协议是in-domain，与ACE的OOD不同 |
| [JoHan，2026-10-07 v1，A.1/A.3/A.4](https://arxiv.org/html/2610.10512v1) | InterHand2.6M/DexYCB/ARCTIC/HO3Dv3/Re/HOT3D；video-conditioned flow | HO3Dv2/v3；HOT3D沿HaWoR held-out原recordings | 统一MANO、腕偏移、有效连续片段；先split后切窗。InterHand单帧资源不能自动当连续运动 |

表中主协议来自相应论文。除特别标明外，窗口长度是该方法的训练/运行设置，不是本项目最优超参。最新不等于更接近TWM：HMP/DIP的数据分工仍比端到端视觉训练更直接。

## 2. 从这些先例能得出的规划依据

1. **高质量纯运动与视觉错误适配分开。** HMP提供独立prior范式；DIP同时提供synthetic与model-specific训练；PAD-Hand说明真实前端偏差需要适应。TWM读数值状态即可，不因适配真实HaWoR而加入RGB。
2. **完整MANO不等于高精度GT。** marker拟合、设备追踪、视觉三角化、单目预测都可以包含MANO，但监督来源与误差不同。按finger与wrist分别保留provenance。
3. **多源扩展有先例，但规模必须按有效独立轨迹计。** HandX多数是设备追踪来源，StableHand10M是QN，最新STERA等是伪标签；不能直接求和当clean99D监督量。
4. **joint-only可以几何监督参数输出。** ACE的RHD部分屏蔽MANO参数loss提供异构监督先例；对我们的连续EgoDex/传感器joint流，仍需MANO骨架/shape/方向适配，不认为所有twist都有独立GT。
5. **先固定eval再决定训练。** ACE保留HOI4D、PAD保留TACO、DIP/UniHand保留HO3D，说明“用于训练”与“检验迁移”是两种选择。第一轮先保留HOI4D，后续加入时改名为扩展in-domain版本。

## 3. HOT3D最需要修正的split

官方Clips train 2,804段只是发布分区。[JoHan A.4](https://arxiv.org/html/2610.10512v1#A1.S4)明确其公开GT评测沿用HaWoR从官方train留出的27个Aria原recordings；StableHand/UniHand公开GT benchmark也不是官方无GT test。各论文使用clips、stride与过滤不同，不能用官方train全量预训练后沿用这些留出评价。

本项目建议以HaWoR原recording划分为主协议，核对StableHand/UniHand/JoHan衍生clips；在full、Clips、HandX聚合和缓存之间共享`source + original_recording_id + timestamp`排除表。精确清单尚未在本仓库取得/实现；取得并固定后才统计可用train量，不能先声明42万条全部可训练。

HaWoR的110/27是Aria协议，Quest motion扩展沿官方train并单独留val/披露；subject-disjoint评测同时排同subject的Quest。增加来源时各基线同训练池，或把数据收益与方法收益分开报告。

这属于潜在协议重叠风险，不代表已经发生实验泄漏：本项目没有训练。自身新建split可以作为另一协议，但必须让基线同split，并单独报告。

HaWoR称27段为val；本项目若用其作最终held-out，早停/超参需用110 train内另划validation。HOI4D只在GT-valid帧评测，缺完整out-of-sight参考的窗口不当作真值恢复结果。

## 4. 之前忽略的来源与当前角色

- **OakInk2提升为核心。** [官方annotations](https://huggingface.co/datasets/kelvin34501/OakInk-v2/tree/main/anno_preview)627段可独立下载；[schema](https://github.com/oakink/OakInk2)包含16个quaternion、tsl和β，12台OptiTrack采集。TaMF已有运动训练和[431/39/157序列split](https://github.com/oakink/OakInk2-TaMF/tree/dev/asset/split)。这支持其优先级高于仅因格式便利而先选Re。
- **GigaHands/HoloAssist是扩展候选。** [GigaHands](https://github.com/brown-ivl/GigaHands)多视角拟合，HoloAssist设备手流；[HandX](https://github.com/handx-project/HandX)证明它们在运动生成中实际使用。不等同独立marker参考，EgoDex优先之后再比较新增价值。
- **HandX只先考虑自采部分。** 公开合并包[来源映射](https://huggingface.co/datasets/alexzhang598/HandX)与原split一致性尚未证实；[loader](https://github.com/handx-project/HandX/blob/main/diffusion/src/diffusion/data_loader/handx.py)的val复用test，caption重复不增加motion。必须单建validation；短窗canonical坐标也不能直接替代长时world腕轨迹。
- **SHOW3D作为local/relative扩展。** [v3数据卡](https://huggingface.co/datasets/facebook/show3d-dataset)2026-10-03发布MANO/MHR；[论文§3.2](https://arxiv.org/html/2603.28760v1#S3.SS2)手标注来自10视图三角化＋UmeTrack IK，MoCap跟的是headset。world随背包rig移动，不直接混入固定world腕轨迹loss。
- **glove支线仍保留。** [DexCap](https://github.com/j96w/DexCap)/[DexWild](https://github.com/dexwild/dexwild)贴近研究动机，但raw骨架→MANO/有效joint映射待验，SLAM腕与glove手指分别记录。先验收小样本，不因无原生MANO而从调查中删除。
- **STERA-10M不作首轮前置。** [HF](https://huggingface.co/datasets/fpvlabs/stera-10m)受控访问；[MobileEgo Anywhere](https://arxiv.org/html/2605.05945v1)明确WiLoR＋ARKit/LiDAR伪标签且无独立手GT。EgoFound3R使用它不意味着它变成MoCap。
- **Open-AoE/EgoSteer暂作伪标签背景。** [Open-AoE](https://huggingface.co/datasets/inclusionAI/OpenAoE-2000h)公开MANO/SLAM结果；[EgoSteer卡](https://huggingface.co/datasets/EgoSteer/EgoSteer-Egocentric/blob/main/README.md)明确部分手轨迹由HaWoR估计，另有源标注，state只保留腕/指尖。不能把老师自己的输出当独立高精GT，也不能把其74D布局直接当完整99D。

## 5. 不照搬的内容与验证边界

静态FreiHAND/RHD或随机相机/光照重复只增加外观/姿态条件，不构成新动态；HMP低variance过滤、HandX两手强动态过滤服务各自任务，TWM需保留静止抓握、单手活动、急停和干净输入。

论文16/48/60/81/128/150帧服务不同输入/输出与因果范围，不能平均成一个“业界标准窗口”。先用实际时间做0.5/1/2秒history对照，资源许可时再扩；当前恢复输入截止t，未来仅作辅助target。这些长度是待验证起点，不是文献证明的最优值。

每次新plan至少记录相关前作、实际数据角色、可借鉴机制、不能迁移的条件、替代旧建议的理由与可反驳实验；数值混合比例、corruption幅度和训练预算由远程pilot与验证决定，不凭论文总量预设。
