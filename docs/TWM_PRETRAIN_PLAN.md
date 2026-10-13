# TWM首轮数据与预训练方案

日期：2026-10-13，v1重规划。**用户已选每手99D MANO动态状态，并要求以后plan广泛对照最新论文。** 当前建议高质量核心为HOT3D可用训练recordings＋OakInk2 train；Re/GigaHands等为来源分层扩展，EgoDex为joint规模池，DexYCB承担独立恢复基线/真实前端适配，HOI4D先留出。数据角色与混合仍为建议，尚未下载完整数据、实现loader或训练。

本次不再只由截图八库决定训练组合。12项相关论文的数据职责、版本与来源见[论文证据表](TWM_DATA_PAPER_EVIDENCE.md)。99D替代原69D默认接口；选定表示不代表已验证训练效果。

## 1. 截图的数据用途先核对清楚

[StableHand v1 Appendix F/Table10](https://arxiv.org/html/2605.18553v1#A6)确实列HOT3D、ARCTIC、Re:InterHand、H2O、HOI4D、EgoDex、EgoVerse、Ego-Exo4D，约10M帧。但这是**quality network预训练**：

- 五个MANO库用参数解码的关节计算视觉预测误差。
- 三个joint-only库直接用joint MPJPE产生quality标签，不拟合MANO。
- 三个joint-only库仅用于质量网络；其生成运动恢复模型则分别在HOT3D/ARCTIC训练。

因此八库可以作为我们筛选数据的起点，不能把作者的quality训练组合直接复制成已验证的TWM运动预训练方案。“原生MANO”在这里指release包含参数，并不表示角度/shape是仪器直接测量。

截图中的0.24M/2.3M等是作者选用规模，不能当全库总量、有效train帧或连续运动时长。Re:InterHand的subjects/provenance和Ego-Exo4D的2D/3D量级还存在与官方资料不一致的表述，下文采用官方发布事实。

## 2. 八库的获取与适用性

| 数据与版本 | 获取门槛与已核验内容 | TWM角色建议 |
|---|---|---|
| **HOT3D-Clips train** | 官方HF直接发布；2,804个150帧/5秒clips，累计420,600帧条目、约3.89小时；这不是排除论文held-out与去重后的可训练净量；MANO、UmeTrack、world camera、shape齐备，各手仍有缺失字段 | **核心之一**；先排除基准留出原recordings，Aria/Quest都能做纯轨迹 |
| **ARCTIC full** | 官方username/password注册；world MANO与30Hz参考；不能只看2.1M多view images | 已有授权时可用，否则按免申请偏好暂不挡开工 |
| **Re:InterHand 3D train** | 官方S3脚本直下；MANO包约17GB（含meshes），original joints约4.7GB；官网7train/3test captures，train标注时刻109,763，名义30Hz | **完整参数拟合扩展**；不再与marker核心同等视作高精度，不用ego图像 |
| **H2O** | 需姓名/邮箱/机构登记，邮件凭据7天有效；未见人工审批要求；官方MANO和camera-to-world可用 | 完整状态可用，但当前暂作备选；只下pose也需登记 |
| **HOI4D** | 官网公开Hand Pose/Camera Parameters分包与镜像，无人工申请步骤；本轮未完成share包传输；最终标签为MANO拟合，原始采集15Hz | **首轮域外留出**；后续加入训练另报扩展版本 |
| **EgoDex train** | Apple直链ZIP，725h train、7h test、97h additional；30Hz，25点与各joint SE3、wrist/camera，confidence部分缺失；ZIP含视频 | **规模扩展首选**；joint几何监督，ARKit标签需质量分层 |
| **EgoVerse** | 公开registry/download pipeline；world21点/wrist可整项缺省，typical30Hz；需按实际episode筛选 | EgoDex之后接入；全库小时不能当完整手GT时长 |
| **Ego-Exo4D** | 签许可后审批，官网预计约2天；3D三角化21点，无原生MANO/独立wrist rotation；实际序列采样须验收 | 当前免申请原则下搁置；以后用于joint/quality监督 |

获取与字段一手来源：[HOT3D格式](https://github.com/facebookresearch/hot3d/blob/main/hot3d/clips/README.md)、[HF train目录](https://huggingface.co/datasets/bop-benchmark/hot3d/tree/main)、[ARCTIC下载](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md)、[Re:InterHand](https://mks0601.github.io/ReInterHand/)、[H2O](https://h2odataset.ethz.ch/)、[H2O schema](https://github.com/taeinkwon/h2odataset)、[HOI4D](https://hoi4d.github.io/)、[HOI4D schema](https://github.com/leolyliu/HOI4D-Instructions#human-hand-pose)、[EgoDex](https://github.com/apple-aiml-research/ml-egodex)、[EgoVerse](https://github.com/GaTech-RL2/EgoVerse/blob/main/CONTRIBUTING_DATA.md)、[Ego-Exo4D访问](https://ego-exo4d-data.org/)。

### 本轮纠正/补充的细节

- **HOT3D的full VRS与Clips是不同获取方式。** 旧记忆因full manifest把HOT3D降级，不能继续应用到公开Clips。Clips tar混合图像与labels，本次未找到官方全量pose-only轻包；训练只解析JSON，不需读RGB，但下载不能预先承诺只传JSON。
- Re:InterHand的`frame_list.txt`已过滤invalid，`frame_list_orig.txt`保留原索引；相邻行不一定相差1帧。随机相机/重光照版本不增加真实运动。官网train时刻109,763/30折算约1.016小时，是帧数推算，非已验收连续净时长。
- H2O官方代码每手读取flag1+translation3+pose48+shape10=62个值；README“59”是算术笔误。[player](https://github.com/taeinkwon/h2oplayer/blob/main/h2oplayer.py#L497)与[train列表](https://github.com/taeinkwon/h2odataset/blob/main/pose_lists/pose_train.txt)是接入依据，列表55,742行仍需按hand flag/gap筛选。
- HOI4D原论文15fps、20秒clip，20%帧人工标注后插值并全帧优化；最终label不是只剩3fps。公开`release.txt`2,971段是任务序列上限，不代表全部都有完整hand pose。[论文](https://hoi4d.github.io/HOI4D_cvpr2022.pdf)、[release](https://github.com/leolyliu/HOI4D-Instructions/blob/main/release.txt)
- EgoDex25点包含额外metacarpal，需解剖映射/rest-frame转换；finger SE3不能直接当MANO local rotations。其joint/camera/world与confidence说明见[官方schema](https://github.com/apple-aiml-research/ml-egodex)。
- EgoVerse腕/关键点可缺省、padding按total_frames裁切、1e9 sentinel不能当实测。采用2026-07-08方向修复后的版本；最新资源表标CC BY-SA4.0，继续保留原episode来源条款。
- Ego-Exo4D当前官方更新为4.3M自动3D手姿、21M自动2D，而非21M自动3D。manual和automatic、原始帧率与有效标注帧率分别记录。[changelog](https://docs.ego-exo4d-data.org/changelog/)、[pose schema](https://docs.ego-exo4d-data.org/annotations/ego_pose/)
- StableHand公开`stablehand-data`是244 HOT3D/155 ARCTIC评测clips与proposal/features，**不是八库训练cache**；不得训练后继续用它做原协议评测。[数据卡](https://huggingface.co/datasets/huajian-zeng/stablehand-data)

## 3. 重规划后的数据角色与顺序

**首轮高质量核心：HOT3D可用训练recordings＋OakInk2官方train。** HOT3D补ego目标域、缺测和独立marker参考；OakInk2补复杂双手操作、较长连续过程和OptiTrack参数。这比“先挑最容易读取的MANO包”更符合高质量运动知识改善视觉的目标。

[OakInk2官方annotations](https://huggingface.co/datasets/kelvin34501/OakInk-v2/tree/main/anno_preview)627段约36.7GB可独立下载；[TaMF](https://github.com/oakink/OakInk2-TaMF)确实用其运动参数做条件生成，已有[431train/39val/157test序列清单](https://github.com/oakink/OakInk2-TaMF/tree/dev/asset/split)。默认只取431 train，val/test隔离。该split尚未证实subject-disjoint，复杂/primitive切片与父recording的跨split关系需验收；若冲突，先合并父序列后重建split并单独报告协议，不把627段全部当train。

| 角色 | 推荐数据 | 使用方式与验收条件 |
|---|---|---|
| 高质量完整状态核心 | HOT3D可用train＋OakInk2 train | 完整99D＋几何监督；marker拟合也有误差，保留原参考与模型点 |
| 拟合参数多样性扩展 | Re:InterHand train，随后GigaHands train | 各自来源和质量分层；先单源加入消融，不能与marker核心合称纯MoCap |
| joint-only规模扩展 | EgoDex train；之后EgoVerse/HoloAssist完整手段 | 有效joint几何loss＋可靠腕标签；不可用角度/shape不强监督，核心数据持续回放 |
| 真实视觉错误适配 | HOT3D/OakInk2可用paired train；DexYCB s0 train | 只在train运行HaWoR并提取数值错误，校准corruption/适配TWM；RGB仍不输入TWM |
| 独立同数据恢复基线 | DexYCB s0 train/eval | 参考DIP的合成恢复及PAD/UniHand/HandFlow协议，单独训练对照，不和“DexYCB未见”混报 |
| 域外验证 | 首轮HOI4D全留出；HO3D作为TWM未见域 | 不参与训练/校准/调参；HaWoR已训HO3D，因此HO3D仅叫TWM未见，不叫整系统zero-shot |
| glove与新marker支线 | DexCap小样本；HandX自采；以后DexWild | raw人体骨架/腕标定通过再接joint监督；不使用机器人retarget action；短窗不能冒充长时world |
| local/relative后续扩展 | SHOW3D v3 | 手是三角化＋IK、world随背包rig移动，先只用local articulation或已验证relative状态 |

TWM不读图像，所以HOT3D Aria/Quest均可供motion；Re不限定ego渲染，不因随机相机/重光照重复运动；静态FreiHAND/RHD不能复制成伪连续motion。

**先建立全局split排除表。** 建议以HaWoR的110train/27held-out原recording为HOT主协议，核对StableHand/UniHand/JoHan衍生clips；Clips、full和HandX聚合都按原recording＋timestamp排除held-out。官方2,804段train不是可直接全部用后再评公开GT benchmark的清单。精确排除清单尚待取得并固定，之后才统计train净量。[关键证据](TWM_DATA_PAPER_EVIDENCE.md#3-hot3d最需要修正的split)

该110/27针对Aria原录制，不能当整个Aria＋Quest的完整清单。Quest可另用官方train原recordings并划内部val，排查source/subject/时间关系，作为明确披露的motion扩展条件；若评测要求subject-disjoint，则同subject的Quest也必须排除。各基线使用相同训练池或另报数据扩展消融。

HaWoR称27段为val；若本项目用它作最终held-out比较，就从其110 train中再划内部validation用于早停、history/corruption与配比选择，不能用27段调参后叫独立test。Oak官方39 val负责调参、157 test最终检验；主体不重叠与否另行记录。HOI4D只评GT-valid帧，完全out-of-sight缺GT时不宣称恢复了真值；受控完全缺测主要在HOT/Oak有效reference上制造并检验。

**规模扩展不预设50:50。** 先按source/subject/task及静止/运动/缺测分层；pilot统计有效时长、完整腕字段、拟合质量、速度谱、重复量后再设采样。比较core-only、core＋fit、core＋EgoDex与真实HaWoR适配，同训练预算报告新增源的收益/退化。不要由论文总帧数决定损失权重。

ARCTIC/GRAB是很好的文献质量对照，但当前注册偏好下不挡开工；H2O/Ego-Exo4D亦不前置。STERA、Open-AoE/EgoSteer等伪标签不进入首轮高质量核心；存在字段/访问/老师偏差等限定，详见[论文证据](TWM_DATA_PAPER_EVIDENCE.md#4-之前忽略的来源与当前角色)。

### 为什么替代之前的建议

| 之前的思路 | 本次处理 | 依据 |
|---|---|---|
| GRAB/ARCTIC先建prior | 保留质量依据，暂降获取优先级 | HMP有效先例＋用户免申请偏好 |
| HOT3D＋Re优先，OakInk2之后 | 小样本接口可保留；正式核心改HOT＋Oak | Oak独立marker与TaMF训练先例更贴近idea，Re格式完整不等于高精 |
| 由StableHand截图八库决定全部训练池 | 改按任务与GT来源筛选，补DexYCB/GigaHands/HandX等角色 | QN、motion prior、visual recovery目标不同 |
| HOI4D直接第二批训练 | 先保留域外，再单独做加入训练版本 | ACE held-out与EgoFound3R in-domain是两种协议 |
| native/joint各半、约2秒作为默认 | 改为pilot驱动配比和history长度对照 | 论文窗口/监督条件不同，不能视为最优配方 |

## 4. 已选输出：每手99D MANO运动状态

用户已采纳动态编码，物理输出保持wrist位置、wrist/root旋转和finger articulation：

\[
x_t=[p_t^A(3),\;\rho(R_t^A)(6),\;\rho(Q_{t,1})(6),\ldots,\rho(Q_{t,15})(6)]\in\mathbb R^{99}.
\]

- `p`定义为解码后实际wrist joint位置；MANO translation与wrist offset通过adapter正确处理。
- `R`为统一MANO root/wrist旋转，`Q`为按源配置解码并应用hand mean后的15个local finger rotations；不能直接把各源45D转成同一6D。全部6D解码为合法SO(3)，99是编码长度，不是99个物理DoF。
- shape `beta[10]`为序列/人物条件，另存，不让网络每帧任意改变手形。
- 固定anchor A只用于网络条件化；输出恢复world wrist与姿态后，可用MANO得到21点与778vertices。
- 当源β/骨架不完整时记录shape mask；joint-only数据需要形态标定或一致骨长方案，不能给所有人固定默认β后把形状差异当姿态误差。
- native预训练可使用release的β条件；真实视觉评测采用HaWoR或训练好的序列估计shape，不能暗用评测GT shape。reference-shape诊断若运行，单独标明协议。

**TWM主要输出当前恢复建议U_t，不替代后续Corrector的最终状态。** 辅助未来运动头只作训练任务；v0不要求TWM输出w/q，真实观测与最终质量头属于后续校正训练。

54D（root p3＋root6＋finger axis-angle45）是同语义的另一编码，StableHand采用它；99D将所有旋转统一为6D。69D关节坐标只保留为历史/可选对照，不再作为首轮必做选型比较。状态编码已定，adapter和效果仍待验收。

## 5. 输入：支持参数与joint两种观测，不给缺失角度造GT

统一接口建议：

\[
W_\theta(H_t,O_t,S_t^{(k)},M_t,w_t^{(k)},\Delta t,\beta)\rightarrow(U_t^{(k)},h_t^{W,(k)}).
\]

H含过去已完成状态、有效性/预测质量和时间；candidate始终是统一MANO状态；O是固定原始数值观测bundle，包含可用的wrist/pose和3D joints。M分别标记pose/joint/candidate/history可用性，w为数值质量提示。这里O是此前Z的明确字段化提案，不添加图像输入。

原生MANO样本可同时构造受损参数与解码关节；joint-only样本保留其原始点与mask，缺少的pose channels置无效。不能把填充值当有效MANO。v0原生池也随机drop全部参数观测，令模型只依赖joint observations，这样扩展时不会临时引入未训练的新输入。

candidate初始化有raw参数时从raw开始，缺失部分由历史/既有模型估计并标来源；joint-only样本不直接把不可见GT角度复制给candidate。首帧无有效腕或历史时不制造确定global GT。

网络首版建议共享左右手参数、保留handedness、每手独立历史。候选随迭代改变，O/H/anchor同帧固定，保持与后续闭环语义一致。

**v0模型与直接输出建议：** 先用小型temporal Transformer完成确定性条件恢复，输入token按腕/手指和时间组织，附mask、quality与dt；当前读出头直接预测99D的U，另有共享表示的短期forecast头。历史只到t−1，当前raw/candidate允许相互读取；当前恢复头不能读取未来target。先验收它与GRU/普通时序基线的收益，再检验结构编码的额外作用，不因StableHand使用flow就把flow预先定为必要。层数/width沿用框架起步候选，在远程资源验收后配置；这不代表完整HaWoR链已变成严格因果。

## 6. 主任务：当前状态条件恢复

从连续reference轨迹中取历史和当前帧，构造部分观测与候选，再监督当前完整状态：

\[
\widetilde O_t=\operatorname{Corrupt}(x_t^*,J_t^*),\qquad
U_t=W(H_t,\widetilde O_t,S_t^{(0)},M_t,\widetilde w_t,\Delta t,\beta).
\]

扰动包含：

| 扰动 | 要覆盖的实际问题 |
|---|---|
| 单joint/整指参数或点缺失 | 局部遮挡和部分检测 |
| wrist位置/旋转与finger旋转偏差 | 深度/姿态错误 |
| 连续缺测、偏差、drift | 多帧失败 |
| 历史估计误差 | 自身反馈的输入分布 |
| 错误高置信、质量提示失准 | 不把confidence视为真值 |
| 干净输入与静止/急停样例 | 避免过度修复和抹平真实动作 |

旋转扰动在SO(3)上生成，不把非法6D向量直接当真实旋转。完全整手遮挡的恢复存在多解，单个输出是估计而非保证唯一动作，不默认高质量。

训练顺序建议：

1. **合成受损恢复。** 先用原生完整轨迹训练，history也含可控误差；保证clean、missing、wrong、false-high-confidence均有验证。
2. **候选与自身历史适配。** 保留固定raw，使用相关的部分恢复candidate，逐步加入模型自己的输出；不只用干净GT历史。
3. **混合标签扩展。** 在保持native监督池的同时加入EgoDex/EgoVerse的joint监督，比较其收益；未来GT只作forecast target，不进入当前/历史输入。

真实HaWoR error分布随后由paired序列校准；TWM仍只读取数值状态。质量网络训练要有真实visual prediction与reference配对，不能只凭一条轨迹就当已有真实观测错误。

## 7. 损失与监督mask

\[
\mathcal L=\lambda_J\mathcal L_{joints}
+\lambda_p\mathcal L_{wrist\ position}
+\lambda_R\mathcal L_{wrist\ rotation}
+\lambda_Q\mathcal L_{finger\ rotation}
+\lambda_v\mathcal L_{velocity}
+\lambda_f\mathcal L_{short\ forecast}.
\]

| 数据提供什么 | 可用的监督 |
|---|---|
| 原生MANO＋位姿 | pose rotations测地误差＋wrist误差＋解码joint几何；保留fit来源 |
| 原生关节更可靠（如HOT3D UmeTrack） | joint几何优先使用原始reference，与MANO-fit点分别记录 |
| 仅3D joints | MANO解码后对有效点计算几何loss；finger参数直接loss mask为0 |
| 无独立腕朝向 | wrist旋转直接loss mask为0；掌frame派生量另标来源 |
| 局部点，没有可靠world | 只做local articulation，不补造world wrist GT |

参数损失为原生label监督，不能因为某字段填了零而默认存在角度GT。速度比较GT有限差分与真实dt，不惩罚真实快动作幅度。纯MANO输出已提供骨架/表面耦合，额外骨长loss只在实际需要时添加，不照抄69D独立坐标约束。

在joint-only样本上使用可微MANO层，因此无需先离线拟合全部θ/β才能加入训练；但twist/shape不唯一，保留原生参数池与形态条件，不宣称所有参数都有独立强GT。

## 8. 时间、坐标与采样契约

论文DIP/HandFlow用16帧、UniHand48、HandX60、HMP128，但任务/输入不同，不作为统一最优值。首轮建议按真实时间比较约0.5/1/2秒history；连续训练片段需额外覆盖当前与辅助future target。15/30Hz用真实dt及padding/mask，不上采样后当新增数据。短期forecast horizon、corruption长度/幅度和batch/步数在远程pilot与validation上固定，不能用域外HOI4D调参。

- 依据真实frame index/timestamps切段；缺帧/长gap/reset不跨窗，不能用排序后相邻行冒充连续帧。
- wrist与所有状态在同一固定anchor；camera frame先经正确方向的camera transform到固定world，避免把头动当手动。
- 正确处理PCA、hand mean、`flat_hand_mean`、quaternion order、左右MANO模型和单位；先用源loader解码，再统一旋转，不直接把各库45数字拼一起。
- Re:InterHand的MANO参数trans与PLY/camera translation单位不能按同一规则批量÷1000，遵循官方renderer。
- OakInk2提供mocap/RGB不同frame索引；不要把mocap_frame_id简单按RGB30Hz算时间。源解码`flat_hand_mean=True, center_idx=0`、FK后加tsl；源码支持米制wrist解释，仍需样本验收。
- 多view、重光照、不同格式和重叠窗口不当新运动量；source/subject/original sequence在切窗前划分。
- UmeTrack→MANO存在模型拟合误差，分清模型状态一致性与原始关键点精度，监督mapping应经过投影/骨长检查。

## 9. 首轮交付与验收

| 步骤 | 可检查的产物 |
|---|---|
| split与小样本接入 | 固定HOT原recording排除清单、Oak官方split及父序列映射；取两库少量train段，字段/帧率/单位/shape表 |
| 统一状态 | 99D encoder/decoder、MANO↔wrist root转换；源解码joint roundtrip |
| restoration baseline | noisy vs restored的joint/wrist真实误差、clean退化、gap/突变分组 |
| 条件与同数据基线 | history-only vs当前部分状态；DexYCB协议下与DIP/普通时序恢复公平比较 |
| 规模/质量扩展 | core-only、core＋Re/GigaHands、core＋EgoDex逐项收益和退化；来源分层 |
| 真实错误与域外 | synthetic-only vs HaWoR数值适配；HOT held-out、Oak test、HOI4D域外独立报告 |

基线包含last-state、可用上下文内的插值/外推、GRU/普通temporal模型，以及DIP纯运动受损恢复的公平对照。不能把用未来端点的插值与严格past-only模型混成同一输入条件。[前作与评测范围](WESTWORLD_AND_HAND_RECONSTRUCTION_REVIEW.md)

先证明真实几何恢复能力，再训练图像Corrector与质量头。所有数据下载、预处理、MANO解码验收和实验在`ssh hongfei`的`/data/hongfei/zhou/Hand_Reconstruction`运行；本机处理源码/论文/文档，GitHub不上传数据与权重。
