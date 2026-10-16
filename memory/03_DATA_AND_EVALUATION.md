# HandLoop 数据与评测记忆

**2026-10-16前两项修改的数据与监督验收：** 见[执行协议](../docs/VISUAL_DATA_GEOMETRY_V3_2026-10-16.md)和[来源职责审计](../docs/DATA_EXPANSION_SOURCE_AUDIT_2026-10-16.md)。远程Dex视觉train800录制／50861有效帧／10主体，原8录制577val原样保留，不训练s0 test。HOT Aria214-1真实RGB烟测2物理clip／600双手states／300图像通过；99parent各1clip完整准备被相机投影精度检查拦截，尚不称已入池。Meta缺失thumbbase slot1不监督，Dex21／Meta21按源mapping解码，参考fitted MANO不是独立sensor GT。HumanTouch固定3文件6379522bytes／1597帧的native结构、dt、valid／quaternion及SHA通过；native根与tracker约9cm差待核标定，ready=false，不参与motion，现完整99D池仍四库。新增raw腕与final自身β绝对geometry、mask-before-arithmetic及真实双卡全局sum/count通过；train-onlyRGB梯度无optimizer，完整模型训练效果未验证。根`evaluations/data_expansion_geometry_v3_20261016/`保留失败与通过记录，原图／逐帧标签不入Git。

**2026-10-16 AUC补测完成：** 同test1015保存预测，官方HaWoR→预选best300 final的PA-AUC **89.83933%→90.62120%（+0.78187个百分点）**；绝对AUC13.59279%→33.08518%，仅腕平移对齐71.10035%→77.20214%。last1000 PA90.73607%仅终点对照；val577 PA89.40736%→best91.32986%。沿作者0–50mm／100阈值、逐joint PCK≤与梯形积分归一化；官方EvalUtil每split69项独立核对AUC／曲线差均0，原五项几何均值与上轮CPU结果最大差8.882e-16mm。论文PA-AUC90.5%为不同完整集合，当前小子集不作排名。Pod CPU仅FK／指标，无视觉前向、训练、权重或best选择改变。[协议与结果](../docs/HAWOR_BASELINE_METRIC_AUDIT_2026-10-16.md)、[小型摘要](../reports/hawor_auc_audit_20261016.json)。

**2026-10-16 HaWoR论文指标差距核查完成：** 论文DexYCB Table1的4.76mm是PA-MPJPE，原报告70.532mm是绝对MPJPE，均为毫米，不能直接比较。Pod CPU重解码同一保存预测后，test1015原官方HaWoR PA **5.08159mm**、预选best300 final **4.68934mm**（同集PA下降7.719%）；baseline比论文数值高6.756%，非十倍。仍16录制／单视角prefix128子集，不能据4.689<4.76宣称胜过论文完整test。新增可选PA评测，原四项逐帧差<0.001mm，无训练／视觉前向或best重选。[核查与来源](../docs/HAWOR_BASELINE_METRIC_AUDIT_2026-10-16.md)、[小摘要](../reports/hawor_paper_metric_audit_20261016.json)。

**2026-10-16联合失败与TWM分侧深诊断已完成（新增冻结推理／无optimizer）：** 本条替代下方“终点motion／future尚无新评估”和“仅索引、未新推理”的现状判断；原条目保留为前一阶段历史。D37 best10000、D40预选best300、终点last1000三份权重，在同一原D37 val split上按四源×左右各128窗固定选择，共1024窗；五种原v3模式clean/noisy/reanchor/position_burst4/missing，模型间逐batch输入／目标hash配对相同。history30、min_future3、stride4、future最多60、position_feature_scale=0.05m、β conditioning关闭，99D动态state／β10另存均不变。四源等权的noisy腕左／右从pretrain **2.782／2.945mm**变为best **3.548／3.689mm**，说明两侧部分保持性下降；不能由此称所有模式或未来均退化。1s四源、2s仅三源；GT仍为准备好的fitted MANO参考，非独立MoCap测量。窗口可能重叠，五模式复用同1024窗，不计作5120个独立样本。

真实状态干预覆盖原fullval577／test1015状态，固定best raw、最后candidate和完成历史，仅换TWM：test左／右腕pretrain70.849／29.069→jointbest62.512／26.561mm，适配有益，不能因synthetic保持变差直接换回旧权重；candidate已含首轮TWM，非整模块因果消融。oracle过去改善、empty history不跨split稳定，GT干预仅诊断。几何／相机分解确认少量逆尺度尾部大错：subject06/113405 frame14，参考隐含s16.827、MANO深度1.134m，last.raw s3.5／5.454m，final腕误差4.201m；左>500mm为baseline0、best5、last6帧，均保留原指标。固定99D的shape替换不支持shape是米级定位大错主因，但未排除β的上游训练梯度作用。

梯度probe固定train两侧各2条录制的16帧、eval／无dropout、无optimizer，前后被测参数hash相同；camera final/raw腕梯度存在样本相关冲突，左右不是统一反号。TWM均值gradient norm及rotation head份额见主报告；NLL/51后仍饱和moduleclip，故仅该全局缩放几乎不改本probe的裁剪后方向。四条均subject01、未左右动作配对，mixed eval不等于真实mixed训练／Adam重建，不能据此唯一归因。原fixed8左40/210、短history与TWM quality未知仍是协议／接口限制；未改split、loss、采样或best。[主诊断、指标与控制边界](../docs/JOINT_FAILURE_DEEP_DIAGNOSIS_2026-10-16.md)、[TWM保持性协议](../docs/TWM_HAND_RETENTION_DIAGNOSTIC_2026-10-16.md)、[聚合数值与全部来源SHA](../reports/joint_failure_deep_diagnosis_20261016.json)。执行版本88c7c00／梯度5154594／相机11159dd／报表a72154d；训练版本仍87a2bcc／eb8d0b9，remote产物`evaluations/joint_failure_deep_diagnosis_20261016/`。

以下同日覆盖与视觉终点评测为此前阶段的历史记录，不覆盖上面的新增专项结果。

**2026-10-16左手覆盖与失败定位：** 远程以D37源码87a2bcc实际TrajectoryWindows重新索引同collection SHA f733aa15…，history30/min_future3/stride4：train左317226窗/1389478状态，右319384窗/1398657状态，占49.83/50.17%；val左30509窗/右30861。Oak173430/173430、HOT66004/66004、Ume74605/76785、Dex3187/3165（左/右），无unknown侧；fullcoverage16完整遍历已让全部左窗参与。视觉train左8录制575帧/右8录制553帧；val左2/143、右6/434；fixed8仅左40states/2clips、右170/6，并非按手别均衡选择。test左9/554、右7/461；左MPJPE原61.602→best.raw109.754→candidate60.209→final63.735mm，右81.263→raw29.715→final29.517；左最终腕72.013→62.515mm、腕对齐17.831→11.923mm仍好，不能说左手整体未学会。左9录制5改善4退化，最坏subject06/113405原32.580→final99.914mm；不事后删样本。源码有手别embedding/左侧MANO和正确镜像还原，未发现确定编码bug；shape与联合适配仍候选。只读索引和已有predictions，不做新推理、改权重或改训。[诊断与限制](../docs/LEFT_HAND_COVERAGE_AND_FAILURE_DIAGNOSIS_2026-10-16.md)、[小型报告](../reports/left_hand_diagnosis_20261016.json)。

**D40最终配对评测完成（2026-10-16 04:25:29UTC／纽约00:25）：** 联合训练已complete1000（03:49:16UTC），训练源码eb8d0b9；原fixed8预选best300不因test改选，last1000仅终点对照。远程评测执行快照e72f7ed完成fullval24clips/577states/8recordings与独立s0 test37clips/1015states/16recordings。test官方HaWoR→best final腕76.335→46.187mm（下降39.49%）、root9.849→8.552°、finger8.921→8.407°、绝对MPJPE21 70.532→48.194mm（31.67%）、腕对齐MPJPE14.480→11.414mm。**左右手不均衡：左手绝对MPJPE61.60197→63.73519mm退化，右手81.26346→29.51666mm改善，不能概称双手均已泛化。** 281/1015帧MPJPE比baseline差>1mm；best自身raw腕对齐10.734→11.414mm也退化，最后candidate已含首轮TWM，candidate→final末轮误改不代表整个TWM负贡献。test best/last的90%xyz椭球覆盖仅60.49%/36.65%；last绝对MPJPE点均值47.238mm略低，但相对baseline recording-bootstrap区间跨0。

test固定subject03–10各前2条s0 test录制，在任何testlabel解码前锁定，不补选；严格raw frame_id<128而非128valid帧，9左/7右录制、0拒绝，manifestSHA=`b5a87d00878f3132692fba81aba68fea38a8fb6dd6980faf82e9fc39e43963a4`。test录制未进入prepared train/val；视觉联合训练仅subject01/02，但TWM纯轨迹预训练见过test主体的其它train录制，非所有阶段未见主体。仍GT crop、固定camera836212060125作为world gauge、nominal30Hz、官方fitted MANO/joints；prediction使用自身β/官方leftfix，referenceβ仅GT/no-left-fix，FK逐帧审计通过。非PA、非独立MoCap或官方HaWoR全8view/完整ego benchmark。本轮仅新增视觉终点评测，**尚未重做终点全量四库motion保持与未来预测评估**；旧小保持集不可替代。[最终协议、CI和限制](../docs/VISUAL_JOINT_FINAL_EVALUATION_2026-10-16.md)、[逐组小型摘要与来源SHA](../reports/visual_joint_final_comparison_20261016.json)。

**D40联合效果历史快照（2026-10-16 01:32UTC）：** 同24val clips等间隔固定8个（索引0/3/7/10/13/16/20/23）、210有效states/3150finger rotations，clip32/验证batch1与source identity未随训练batch4→8变化。HaWoR全量微调使当前raw也改变，必须三层对照原raw→当前raw→联合final；250步腕88.250→65.478→48.598mm，root11.729→9.243→8.591°、finger9.747→7.589→7.530°。candidate是最后一轮TWM前、已含首轮恢复/跨帧历史的候选，非纯插件消融；37.1%state腕退化>1mm，90%三维椭球coverage40%/anchor平均轴std32.622mm。motion保持同四源各16窗、逐指标有有效targets的源等权，counts对初始完全相同；noisy当前腕2.959→3.514mm、1s100.370→94.874mm（四源47targets）、2s152.983→172.615mm（三源37targets，hold169.452）；不同于旧D37每源256窗。GT为Dex fitted MANO、GT crop/固定camera，指标是腕L2毫米和root/手指local测地角，不是MPJPE；最多32帧自身历史、非长录制闭环，future仅motion。只读现有验证及远程CPU汇总，不做新模型推理。[完整报告](../docs/VISUAL_JOINT_PROGRESS_2026-10-16.md)、[数据与定义](../reports/visual_joint_progress_20261016_0132.json)。

**D37效果历史快照／D38延后loss（2026-10-15纽约18:22）：** 用户明确loss调整留到下版本，本轮原六项不变，未启动新方案。训练快照8,010/10k，完整验证8,000；同run固定val普通腕4.652→2.881mm/root1.990→1.316°/finger1.058→0.637°，位置四帧6.520→4.539mm。5k→8k score0.00731648→0.00577622（下降21.05%），仍改善但放缓。clean0→0.601mm、12.60% >1mm退化；近准当前/错误历史0.267→1.296mm、30.96% >1mm退化，仍未解决。未来腕1s96.954对hold119.461mm、2s140.930对147.506mm；2s仅三源，HOT已略优、Ume仍退化；1s root及2s finger较5k略退。当前90%xyz覆盖普通78.81%/位置事件54.49%，未校准。最近1000更新2.799s/步，剩1990约93min，估计纽约19:55（吞吐持续条件）。只读远程CPU汇总绘图，当前/未来同一validation SHA、宏平均和固定输入检查通过；仍是拟合MANO/合成观测历史，非闭环或真实HaWoR。[8k报告与曲线](../docs/TWM_TRANSIENT_8K_PROGRESS_2026-10-15.md)、[对照摘要](../reports/twm_progress_comparison_20261015_2220.json)。

**2026-10-15损失候选的评测要求（未做新实验）：** 文献对照不更换四库/GT/split。建议同架构、原v3/固定扰动种子和匹配窗口曝光量检验辅助项；独立对照future起点detach，不能把共享骨干也称隔离。需要current几何/clean与reanchor退化、受控单/3–4帧真实时间恢复、急停/低运动保持、分源/时域forecast与hold/CV、匹配聚合的branch选择指标、coverage/区间宽度、采样调用数和闭环/真实HaWoR。单NLL或CFM未被验证更优；统计loss不是独立物理证据。[12篇对照与建议](../docs/TWM_LOSS_LITERATURE_REVIEW_2026-10-15.md)。

**D37未来预测专项（2026-10-15纽约16:09）：** 运行快照5,150/10k，完整验证5,000；只读已有日志，远程CPU汇总/绘图，未改训。普通噪声最高权重整段分支腕终点误差0.1/0.5/1/2s为10.116/51.376/102.369/145.775mm，对应恢复当前后hold15.717/68.938/119.369/147.556mm，下降35.6/25.5/14.2/1.2%。1s root27.288对33.061°、finger9.120对9.500°；2s三源无Dex，HOT/Ume腕退化、Oak改善，不能称稳定长时优势。干净输入2s腕141.923对147.422mm，仅改善3.7%；low_motion整段腕ADE6.228对hold5.114mm退化。未来逐轴90%边缘覆盖95.09%，1s/2s总每轴std94.620/127.241mm，宽区间不等于精确。混合熵1.315nat不证明模式质量；selected ADE75.625与oracle minADE47.452权重不同，差值非纯分支选择损失。所有future宏平均/各步时域counts一致已验；非真实HaWoR或闭环。[完整预测报告](../docs/TWM_FORECAST_PROGRESS_2026-10-15.md)、[摘要与曲线](../reports/twm_forecast_progress_20261015_2012.json)。

**D37中途验证（2026-10-15 19:48UTC）：** 运行快照4,700/10k、完整验证4,500；每源固定256窗/每模式1024，四源等权，拟合MANO参考、原recording-disjoint split。普通噪声腕4.652→3.331mm、root1.990→1.578°、finger1.058→0.821°；位置专门四帧事件6.520→5.244mm。clean腕0→0.884mm且30.66%窗口>1mm退化，近准当前/错误历史0.267→1.471mm且44.04% >1mm退化；缺失finger0.685→0.741°也退化。3.5k→4.5k score下降32.9%，但不是所有指标单调。1s腕103.236mm vs hold119.462；2s仅三源146.895 vs147.650mm，root反略差；90%当前xyz覆盖普通75.98%/位置burst46.39%。14模式candidate跨验证不变、宏平均重新校验通过。burst只是随机相位的单次当前恢复，非十步滚动/回灌；旧64clip/真实HaWoR/独立传感器验证未完成。本轮仅既有记录远程CPU汇总绘图，训练继续。[协议、各模式、分源及限制](../docs/TWM_TRANSIENT_10K_PROGRESS_2026-10-15.md)、[摘要](../reports/twm_transient_v3_progress_20261015_1944.json)。

更新日期：2026-10-14。Oak/HOT有效train/Ume real/Dex原包均完成，531.87GB；已有限字段抽查；D23现已完成Oak子集99D准备、window loader与TWM首轮训练，完整FK和其余库仍未验。[实际字段审计](../docs/TWM_FOUR_DATASETS_AUDIT_2026-10-14.md)。设计依据：[START_HERE §5、§7](../HandLoop_Project/START_HERE.md)、[实现清单 §2、§4、§6–8](../HandLoop_Project/03_References_And_Implementation.md)。数据来源见[专项调研](07_TWM_DATASET_SURVEY_2026-10-13.md)，方法引用见[写作核验](05_WRITING_AND_REFERENCES.md)。

## 两个数据池

[八库采集与标注报告](../docs/DATASET_CAPTURE_AND_ANNOTATION.md)按原测量、标定/拟合、人工/QC、发布参考分层。HumanTouch团队MANUS＋VIVE系统已核，公开episode配置待验；DexWild固定tracker处理代码未显式完成camera-world组合，发布版本需核；它仍是glove候选，不能凭world_abs名直接训练world腕GT。

**当前首轮目标池（v2.1建议）：** OakInk2＋HOT3D有效train＋独立UmeTrack train＋HumanTouch；DexWild可靠local字段可先并入，world链核清前不做腕GT；GRAB/ARCTIC有授权优先加入，DexCap小型补充。首个checkpoint不要求所有库齐备，至少一条完整腕＋五指链通过验收。未固定最终配比；D23先实现Oak单库pilot，其余仍待转换。[归纳表](../docs/TWM_PRETRAIN_PLAN.md#0-当前推荐的预训练组合)

**精度工程判断（建议）：** 完整world腕＋手指参考更信校准/质检后的外部多相机MoCap；局部手指分别验marker拟合与EMF glove标定，不能由设备类别严格排序。原生→MANO残差独立报告；HumanTouch/DexWild继续主要候选。没有八库统一GT实测排名。[证据与指标边界](../docs/DATASET_CAPTURE_AND_ANNOTATION.md#精度判断工程建议与实测排名分开)

**已定D15–D18：免申请、每手99D、论文机制参考、传感器姿态精度优先。** 用户本轮纠正v1过度沿视觉论文选库，v2恢复GRAB/ARCTIC/Oak/HOT的MoCap核心候选，新增UmeTrack，HumanTouch/DexWild升主要glove。实际先可公开来源，注册有授权再接；Re/EgoDex降后续视觉标签扩展。混合未定。[v2计划](../docs/TWM_PRETRAIN_PLAN.md)、[传感器精度核验](11_SENSOR_FIRST_PRETRAIN_2026-10-13.md)

论文用法已经分角色整理在[完整方法框架§9](../docs/METHOD_FRAMEWORK.md#9-数据选择采用用户最新的免申请原则)：HMP独立motion prior；HaWoR视觉估计数据与HOT3D infiller不同；StableHand八库主要服务quality network；PAD-Hand用DexYCB/HO3D训练评测、TACO跨域留出。DexYCB/HO3D有直接入口，适合参考paired恢复协议，但视觉拟合标签和第三视角不能替代头戴独立MoCap结论。

**HaWoR对比复核：** 视觉训练四库HOT573K/ARCTIC165K/DexYCB169K/HO3D66K，infiller单HOT；主定量HOT Aria27 validation/world＋Dex test/camera，EPIC定性。建议首轮主HOT补Dex，ARCTIC/HO3D另加则标为扩展。当前官方HOT evaluator为GT crop/visibility、右手、GT-valid交集、100帧含scale对齐，未调用infiller；部署detector/双手/不可见恢复另报。Dex作者fork默认s0_test，缺预测ID会跳过，须锁定共同覆盖。本轮只核资料，没有运行baseline。[完整协议与资源缺口](../docs/HAWOR_DATA_AND_COMPARISON_PROTOCOL.md)

**2026-10-14作者报告误差复核：** 原论文Table3 HOT3D PA/W/WA-MPJPE=4.79/33.20/11.27mm、RTE0.78%；Table4(b)不可见片段infiller=6.22/66.25/37.22mm、RTE7.41%，线性/球面插值W75.01mm、最后姿态W116.79mm。Table1 DexYCB PA整体/中重遮挡/重遮挡=4.76/5.03/5.07mm。PA逐帧对齐，W起始和WA整段100帧similarity（含scale），均非D28合成受损wrist指标；不能凭D28的5.435mm声称优于HaWoR。只是论文/官方代码核验，无新模型实验，未恢复旧训练。[数值及定义](../docs/HAWOR_DATA_AND_COMPARISON_PROTOCOL.md#2026-10-14复核的论文报告数值)。

**D30误差分布匹配：** 用户强调模拟HaWoR真实错误。D29仍复用旧sample_batch：当前每轴σ25mm、root12°/finger8°，10%乘3；历史4mm/1.5°且quality不依实际误差。位置高斯解析平均距离约47.9mm（含severe），不是HaWoR误差。建议train/calibration提取pre-infiller的wrist/SO(3)/local与world joint残差、时间相关性/连续缺失，先验adapter与shape约定；排HOT27/Dex test/TWM val。独立覆盖当前准确而历史错误，保留少量严重失败，quality与几何误差关联但部署不用oracle。已对照PAD-Hand真实WiLoR配对、DIP两类训练、StableHand质量增广和HaWoR连续mask；具体幅度/比例未定、无新训练修改或校准结果。[校准建议及证据](../docs/TWM_DIRECT_REFINEMENT_PROPOSAL_2026-10-14.md#8-2026-10-14corruption应贴近hawor实际误差)。

**D31混合训练澄清：** 用户不要求本轮先做真实HaWoR标定，要求暂停当前训练并考虑干净/有噪条件混合、严重少量。D29已暂停814步；建议20%全净、20%当前近准/历史中噪、35%双方独立轻中噪、20%局部缺失、5%局部强异常。幅度改为实际位移范数近0–1/轻1–6/中6–15/强20–50mm、SO(3)总角，避免每轴σ带来隐含长尾；默认quality unknown，所有监督GT不受损。配置仅草案，尚未接入trainer或证明适配HaWoR，具体比例/幅度待验。[设置与验收](../docs/TWM_DIRECT_REFINEMENT_PROPOSAL_2026-10-14.md#9-d31先暂停混合干净轨迹与适量受损条件)。

**D34训练执行更新（2026-10-15UTC）：** 保留D32四源参考/recording split、12模式固定validation与20/20/35/20/5 mixed_v2；旧run停3,110步，新run每卡512/global1024、从头10k。目标10.24M次样本抽取，非独立新样本数；batch/调度改变后不能只按同step比较。旧1k→3k weighted current score0.02225→0.01281、root3.871→2.573°、finger3.215→1.773°，但noisy腕输出7.287→7.210mm仍高于输入7.158mm；clean/reanchor退化及未来误差在摘要同报。指标为四库均值的等权宏平均、源拟合MANO wrist mm/局部SO(3)角，非独立joint/mesh或真实HaWoR；未宣称已收敛。[曲线与设置](../docs/TWM_MOE_LONG_HORIZON_RUN_2026-10-15.md#d34每卡512从头训练10000步)、[小型摘要](../reports/twm_batch512_10k_20261015.json)。

**D34腕诊断（2026-10-15，非新增全量评测）：** 冻结4,400步、四库各8窗32窗，原7d53fa4源码/hash一致且无参数更新；ordinary候选5.534→模型5.431mm，不替代4k1024窗结果。±10mm位置干预的残差平均反向响应仅0.005–0.034mm，FP32相近；head有非零current几何梯度，future项量级相当/部分负cosine。固定融合全结果保留，不在GT拟合融合比例。输入尺度/identity倾向/多任务牵制为假设，不把common-bias不可辨识当全部失败解释。[诊断记录](../docs/TWM_WRIST_DIAGNOSIS_2026-10-15.md)、[摘要](../reports/twm_wrist_diagnosis_20261015.json)。

**D34的4k中途验证（纽约2026-10-15）：** 本轮新run ordinary/noisy腕7.158→7.180mm、root3.999→2.373°、finger3.224→1.561°；clean腕0→0.688mm（>1mm退化14.26%），reanchor 0→0.571mm（4.39%）。固定每源256窗、四库等权，分库与样本/未来时域有效数保存在[摘要](../reports/twm_b512_progress_20261015_0521.json)；宏平均和p95不当总体pool统计。score同轮3k→4k降4.03%，未宣称收敛或真实HaWoR收益。1s预测优于hold、2s在三库有效窗上未优于hold；90%当前xyz椭球覆盖82.03%。[效果文档](../docs/TWM_B512_PROGRESS_2026-10-15.md)。

**D32实施：** 已按用户要求接入mixed v2，源GT/split未变；20/20/35/20/5、有界位移范数/总角、quality unknown，旧sample_batch保留。4096窗远程语义检查通过，9种新fixed modes加3种legacy共12种，best按clean/reanchor/noisy/missing/strong加权0.2/0.2/0.35/0.2/0.05的current restoration loss选择。reanchor固定当前clean，是诊断分层，不等于mixed内clean/near各半的精确期望风险。新正式run已实际更新，尚无新模型最终恢复收益；旧协议保留但batch/RNG/quality条件不同，不能将更轻噪声的更小误差直接当胜过D28。[运行与口径](../docs/TWM_MOE_LONG_HORIZON_RUN_2026-10-15.md#d32混合有界噪声训练已启动)。

**四个已下载来源的后续审计：** 字段条件允许用于pure-motion TWM，建议MoCap三库核心＋Dex train补充，不按四库等精度/等权监督。Oak raw_mano有16 quaternion/tsl/beta；HOT/Ume MANO为PCA15＋root axis-angle/transl，Dex为PCA45＋非flat mean的51D，需源模型展开后编码99D。Ume样本native与MANO可独立缺失；HOT有真实ref timestamps，Ume抽查包仅frame IDs；Dex TAR顺序乱序且多view共享motion。固定train后排序、mask、时间与腕原点/FK验收仍待做，不能把全包含val/test都用于训练。[代码与样本范围](../docs/TWM_FOUR_DATASETS_AUDIT_2026-10-14.md)

v1只要求HOT/Oak、先扩Re/EgoDex的建议被D18替代：第一批共同验收可用MoCap及glove，按质量和字段能力训练；99D不要求源原生MANO。DexYCB仍可纯轨迹恢复对照，固定第三视角不是排除条件；它的论文用频率不使其GT优先于独立sensor参考。

**Motion-only pool** 用于独立训练TWM，不要求配对图像。需要可靠native姿态、时间、骨架与校准；只有local finger也可参与99D几何恢复，缺腕/未测旋转单独mask，不冒充完整state GT。部分监督契约已有提案，待实现验收。

**2026-10-14物理验证讨论：** [数学与验证方案](../docs/TWM_PHYSICAL_GUARANTEES_AND_VALIDATION_2026-10-14.md)建议分开测SO(3)/内部FK/所定义角度范围、全局参考系等变误差、独立GT几何及速度/角速度误差、急转/遮挡/历史污染。真实dt与SO(3)差分必要，帧内轮次不当物理时间；小加速度不当物理正确。若未来报告动力学，需有来源的惯性/驱动/外力参考，预测自身自由force不能形成独立验证。所有为拟议协议，无已运行结果。

**Video–reference paired pool** 用于训练/评估图像取证、可靠性与 Corrector。需要同步视频、相机参数、参考状态，以及真实 HaWoR 输出。纯人工 corruption 的表现不能代替真实视觉错误上的表现。

## 数据资源与使用边界

| 资源 | 项目中的角色 | 已知限制与下一步 |
|---|---|---|
| GRAB + ARCTIC | 恢复高优先MoCap核心候选 | Vicon→MANO/SMPL-X；无RGB无妨，注册仍需已有授权；GRAB personalized template、ARCTIC fit error需保留 |
| OakInk2 | MoCap核心候选：marker＋复杂操作 | 627段annotations约36.7GB独立可下；TaMF431/39/157序列split；用train并验父序列/PT-CT重复、FK/时间映射；非已核实主体留出 |
| HOT3D-Clips / full VRS | MoCap核心候选：native marker参考 | HF直下；官方train累计42万帧不是净训练量，须排HaWoR/衍生benchmark held-out原recordings。full另需manifest；Aria/Quest均motion可用，官方test无手GT |
| Re:InterHand 3D train | 完整参数拟合扩展，降出高质量核心 | S3直下，7train/3test；scan/视觉拟合，原帧索引切gap，trans与PLY/camera单位分开；单源加入消融 |
| HOI4D / H2O | 首轮域外留出 / 登记后备 | HOI4D15Hz拟合，只评GT-valid；缺out-of-sight GT不声称真值恢复。H2O登记，未见人工审批，30Hz列表有gap |
| DexWild | 主要glove候选，优先验收 | 公开human 7,917episodes/2,924,588transitions；MANUS手指、tracker/camera腕，已有平滑与dt/world待验；不用LEAP action |
| HumanTouch | 主要glove候选，优先验收 | HF公开、CC BY-NC4.0已核；25点/方向＋world腕，pose可单取。十份meta约132.75h累计非QC净量，manifest/校准待验 |
| DexCap | 小规模glove轨迹及adapter对照 | 公开30分钟wiping+60分钟packaging；Rokoko指关节、T265腕部；不能冒充论文全集或无漂移world GT |
| UmeTrack独立集 | 新增自由手势/pinch marker候选 | 官方HF直下，36相机MoCap＋scan；native几何比converted MANO更接近reference；real/synthetic同motion去重，train内划val |
| OpenGraph MANUS | 小规模local glove/adapter | 25点/手＋40finger角，parquet独立；未见独立world wrist SE3，做部分监督 |
| AMASS | 必须筛选子集的MoCap来源 | 原论文明确无hand markers时手姿为平均值；GRAB/TCDHands/SAMP有证据，不能整库当真实手指动态；与独立GRAB去重 |
| GigaHands / TACO | 单独的多视角视觉估计扩展池 | 可提供轨迹但非独立finger marker GT；TACO的MoCap主要用于物体/头相机；GigaHands检查重心化版本 |
| EgoDex / HoloAssist / SHOW3D | 规模、泛化或质量分层实验 | 设备/视觉估计，不能称glove-only；SHOW3D的world是移动背包rig。详见专项调研 |
| HandX | 自采marker补充；聚合短窗后备 | 大部分来自HoloAssist/GigaHands等；先选自采或按mapping回源split，val复用test须改。两手强动态过滤不照搬，长时world回原库 |
| TCDHands / SAMP / FPHA / HUMOTO / BigHand2.2M | 手势、局部姿态或腕运动补充 | 规模、公开子集、完整腕朝向和连续性条件不同，不能统一当完整69D监督 |
| DexYCB / InterHand / Re:InterHand / HO-3D；AssemblyHands等 | 视觉拟合补充或跨域评测候选 | 前几项已核验部分release与限制；AssemblyHands尚未专项审核。采用前需同输入、同坐标、标签可比 |

上述均为来源事实与使用建议，尚未选定最终训练混合。完整事实、许可入口与未验收事项见[2026-10-13专项调研](07_TWM_DATASET_SURVEY_2026-10-13.md)。预训练只需要运动状态，不必限制头戴相机或配对RGB。

用户指出数采团队覆盖不足后，[产业补充调查](09_INDUSTRY_HAND_DATASETS_2026-10-13.md)新增SynData（10fps/ITW qpos映射）、OpenGraph（glove局部关节）、Daimon DataDex（手部分量待查）、OOJU（Quest26joints）、EgoVerse（完整手字段可缺省）、Lightwheel（申请开放视觉手标注）和UniHand（已有数据格式聚合）。规模扩展不能只沿首轮学术清单；先列全来源，再根据完整性、质量、许可筛选。所有数据验收/处理远程运行；D19当前优先ds已授权Pod。

GT 来源必须按 **joint articulation 与 wrist/global pose** 分开记录，不能一个 `gt_source=glove` 字段掩盖腕部来自 SLAM。设备输出也可能有误差；对 GT 来源和可用性的描述，不等于估计出了传感器误差分布。

MANO标签通常由视觉/depth、marker或其他手模型拟合得到，不是直接仪器角度读数。HOT3D官方说明MANO由UmeTrack求解、精度稍低，推荐UmeTrack关键点监督；原始joint/marker与模型解码点应分开保存。joint标签也可经可微MANO loss训练参数输出，不必为每条样本生成MANO角度。详见[标注来源与表示调查](10_MANO_REPRESENTATION_AND_ANNOTATION.md)。

**八库用途与规模（2026-10-13）：** StableHand Table10约10M用于quality-network，后三个joint-only库不拟合MANO；生成器另按HOT3D/ARCTIC训练。作者选用帧数不当全库或连续train量。EgoDex为公开规模扩展建议，EgoVerse先筛完整wrist/joints，Ego-Exo4D需审批暂不前置；EE4D当前官方4.3M自动3D、21M自动2D，不沿用论文21M3D口径。StableHand 399段缓存属于benchmark评测，不能作八库训练全集。[核验与来源](../docs/TWM_PRETRAIN_PLAN.md)

## 当前 schema 的状态

**每手99D已按D16选定**：p为实际wrist joint、R/Q为统一解码后的root/15个local旋转，β序列条件另存。native参数loss与joint-only几何loss分mask仍为实现契约；未知腕朝向/shape/角度不造GT。下列69D字段分析为历史转换参考，未成为现行输出或运行实现。[输入、损失与shape边界](../docs/TWM_PRETRAIN_PLAN.md)

**69D字段满足性（2026-10-13）：** HaWoR和数据不必原生输出69个数；只要能给一致参考系的wrist position/rotation、标准21点与时间信息，就可构造3+6+20×3。69是表示长度，非独立物理DoF。

| 来源 | 字段层面判断 | 尚需验证 |
|---|---|---|
| OakInk2 | MANO/SMPL-X位姿、shape、translation具备解码完整状态的信息 | FK、wrist点/root区别、单位/帧映射 |
| DexYCB / HO3D | MANO参数及参考位姿可构建完整状态 | camera/world约定、PCA/axis-angle解码与缺失mask；标签为视觉拟合 |
| DexCap raw | 采集代码21点+orientation，README双腕6DoF | release保存字段、palm/tracker→wrist标定、归一化尺度及SLAM质量 |
| DexWild raw | MANUS full、tracker与timesteps有接入基础 | 原始关节schema及world腕部变换未数值验收，不能直接宣称已完整69D |
| HumanTouch | 25点/方向及wrist7D契约支持几何监督 | 25→MANO语义/rest-frame、tracker偏置与实际parquet待验；CC BY-NC4.0已核 |
| SynData / OpenGraph | 分别为ITW qpos/11点、25点但无独立world wrist SE3 | 前者需FK/人体映射；后者只先做局部或部分监督，不补造全局GT |

结论是已有完整状态资源可支持方案，并非所有调研库均可直接提供同质量69D监督。只有21点而无独立腕旋转时，可定义几何掌frame，但须统一且标记为派生标签；纯TWM预训练不需相机，图像取证/paired阶段则需相机和视频。

[03 文件 §4](../HandLoop_Project/03_References_And_Implementation.md) 有 `HandFrame` dataclass 示意和建议张量，不是运行实现：

| 字段组 | 已有示意 | 必须补齐的约定 |
|---|---|---|
| 时间和身份 | `timestamp_s, sequence_id, subject_id, handedness` | 时间源、同步误差、FPS/dt、左右手 track identity、切段/reset |
| 状态 | `wrist_pos_world (3,), wrist_rot_world (3,3), joints_local (20,3)` | 关节顺序、腕坐标定义、左右手、scale、单位、bone mapping |
| 原始观测 | `raw_detected, joint_valid_mask` | 手检测、joint visibility、状态可用性和 GT label availability 分开 |
| 相机 | `camera_K, camera_T_world` | 变换方向、坐标轴、crop/full-image/feature transform、畸变处理 |
| 参考状态 | `gt_wrist_pos_world, gt_wrist_rot_world, gt_joints_local` | 缺字段/缺帧 mask、joint 与 wrist 分别的标签来源 |
| 建议网络张量 | history `[B,K,69]`、current/raw `[B,69]`、`dt [B,K]` | padding、当前 dt、mask 的 20/21/22 粒度、quality 的 `num_parts` |

存储长度单位建议米；报告腕部/关节误差使用毫米；角度误差使用度。时间戳比较和向量差分必须以实际 `dt` 为准，不能默认不同数据同 FPS。

## 分割与可比性

在训练前记录 train/validation/test 划分及主体、序列、场景标识；避免同一轨迹相邻片段跨划分造成泄漏。先定义 OOD 是跨被试、场景、数据集还是视觉 backbone，不将同域随机切分叫跨域泛化。

**v1新增split契约：** HOT官方train包含HaWoR27个held-out Aria原录制的公开GT，Clips/full/HandX聚合/StableHand缓存共享原ID+timestamp排除表；若27作最终比较，另从110 train划内部val调参。Oak沿431train/39val/157test清单但不宣称subject-disjoint；验父recording重复。DexYCB训练的同数据对照独立命名，HOI4D初版不训练/调参、只GT-valid评价；HO3D仅TWM未见，HaWoR本身见过。

**后续已落实：** 下载manifest及脚本已固定27个HaWoR ID，并从BOP官方train排349段/25源录制；另外2个ID整版缺席。2455个train TAR已完整下载，评测349段没有下载，full27记录待准备。官方export_gt会生成val.json，静态名单文件未在tree发布不等于无法生成split。该排除完成不代表其他benchmark/subject-disjoint验收已完成。

相机世界坐标和参考世界坐标需有可复现的固定对齐协议。不能逐帧用评测 GT 优化对齐后，再声称获得绝对世界精度。是否允许一次序列级 gauge 对齐、尺度对齐及其数据来源仍需定案并单独报告。

## 指标及其含义

| 指标 | 用途与报告条件 |
|---|---|
| World-space MPJPE | 世界坐标下所有指定 joints 的真实位置误差；注明 root 是否包含、单位、对齐协议 |
| Camera-space MPJPE | 相机坐标手部误差；帮助区分世界轨迹与局部几何问题 |
| Wrist-local joint error | articulation 恢复；不能替代 global wrist 精度 |
| Wrist translation / rotation | mm 与 SO(3) 测地角度；腕局部 frame convention 必须一致 |
| Velocity error | 与 GT 速度比较，反映运动误差；不是越慢越好 |
| 遮挡分组误差 | 分 visible/occluded、缺失长度与程度；说明分组依据来自标注还是估计 |
| 高 confidence 错误子集 | 检验原始错误高置信观测能否被纠正；阈值与误差标准须预先记录 |
| 再可见后 error decay | 随恢复可见后的时间/帧数报告重锚定过程；定义事件起点与窗口 |
| 准确关节退化率 | 检查修复是否破坏原先准确测量；阈值须固定 |
| Runtime 与延迟 | 相同硬件、batch、图像尺寸、迭代数、chunk/lookahead 条件 |

PA-MPJPE 可作补充，不能代替 world-space accuracy。指标具体聚合方式、阈值、报告置信区间的方式均尚未决定，不预填数字。

**2026-10-13论文/代码核验：** HaWoR `66c7d410…` 在100帧段的前两帧求含scale的similarity alignment后算W-MPJPE，WA用整段对齐；不是无对齐绝对世界误差。StableHand v1的首帧/全段对齐定义，又与Enhanced `0471a45e…` 的W无对齐/WA逐帧wrist减除不同。坐标anchor、shape来源及对齐前处理也应记录。建议兼容论文指标与明确world gauge/固定尺度的几何误差并列，不能跨版本拼表。固定SHA、函数及来源见[论文精读的指标部分](../docs/WESTWORLD_AND_HAND_RECONSTRUCTION_REVIEW.md#世界误差与复现协议)。

## 能检验贡献的对照

| 对照 | 要隔离的作用 |
|---|---|
| HaWoR full vs pre-infiller | 分清现成 temporal/infiller 与自有模块收益 |
| Raw + smoothing / Kalman / GRU / temporal Transformer | 改进是否超越一般时间模型 |
| Prediction-only TWM vs restoration TWM | 当前受损状态条件恢复是否必要 |
| 不输入当前可信局部状态 vs 输入 | 是否确实利用当前观测约束缺失部分 |
| Fixed confidence gate vs learned reassessment | 是否改善 false-high-confidence 和突发动作 |
| One-shot vs N=2/3 | 迭代收益及计算代价 |
| 有无 candidate-guided independent visual re-observation | 收益是否来自新图像证据，而非重复计算 |
| 有无最终状态反馈 | 递归状态维护的作用 |
| 有无自身状态 rollout training | 是否改善训练/部署历史分布差异 |
| HMP、DIP、StableHand、PAD-Hand 等 | 与已有 motion prior/refinement 的真实差异；DIP也做受损当前条件恢复，不只预测未来 |
| DIR/WiLoR/GeoHand等取证机制 | 与已有候选投影/查询及多轮细化区分，保持相同视觉前端和时间信息 |

跨数据集/被试、突发动作、长遮挡和再可见诊断应覆盖上述关键版本。相同 frozen TWM 接不同视觉 backbone、预训练数据规模扩展、机器人 retargeting 是后续验证方向，不是首轮必须全做的工程。

## 开工 sanity checks

1. 检查视频中的真实可见关节与 world→camera→image 投影；包括左右手和 crop 映射。
2. 检查 local/world/anchor roundtrip、rotation 正交性与 `det≈+1`、rotation6D 解码。
3. 保留插值前 detection mask；确认独立视觉证据不是 HaWoR 自己的投影点。
4. 检查时间同步和骨架映射；不同来源 GT 不可按未说明的逐帧优化对齐。
5. Clean state 输入不被无故破坏；corruption restoration 和 last-value/interpolation/GRU 作可比较验证。
6. 对 missing 输入检查预测质量是否反映未知，而非默认高置信。
7. 覆盖错误高置信、突然转向和重新出现；检查历史只在最终帧输出后更新一次。
8. 记录 HaWoR 时序、infiller、坐标转换和缓存配置，避免 baseline 混淆。

## 实验记录约定

每次实验记录日期、run ID、问题、代码/第三方 commit、配置、随机种子、数据 split/hash、GT 来源与字段、输入模态、坐标/对齐协议、训练预算、指标定义、结果路径和失败原因。真实日志与产物优先于文字印象。

完整视觉方法结果仍为**未运行**；2026-10-14新增Oak合成恢复pilot中途实测，见[单库运行记录](../docs/TWM_OAKINK2_PILOT_RUN_2026-10-14.md)。旧讨论的 30/12/5 mm 等只是示意，不进入结果表。旧版 `GapClosed=(E_camera−E_ours)/(E_camera−E_sensor)` 是可选概念指标；若 sensor 本身作标签而取 `E_sensor=0`，只表示相对该标签的误差下降，不意味着传感器物理误差为零。[历史出处 §11.4](../HandLoop_Project/01_Conversation_Research_Archive.md)

## 2026-10-14 Oak 单库 pilot 实测

D23在ds将官方64train/8val原录制转换99D（128/16单手轨迹、204502/18032单手帧），seed20261014、30Hz、H30、future3、窗口stride4；test标注不读，subject-disjoint未宣称。来源为MoCap拟合MANO，实际world腕以center_idx0＋tsl规则确定；无完整MANO FK/mesh验收。代码4cb1da8训练50,000步，09:05 UTC step7275仍在运行。固定1024val窗口/模式，step7000 noisy wrist49.090→8.657mm、root22.538→2.252°、finger15.421→2.259°；clean被改成4.551mm/0.932°/1.642°，不能只报收益。不是MPJPE或真实HaWoR结果；自身历史、真实前端错误、其它库与动力学未测。完整配置、原始产物位置、smoke provenance勘误及恢复限制见[运行记录](../docs/TWM_OAKINK2_PILOT_RUN_2026-10-14.md)。

D24后续：用户主动暂停于28,213步，latest/best保存且可读，最后固定验证step28,000 noisy腕5.980mm/finger1.601°、missing腕5.903mm/finger1.759°、clean退化3.626mm/0.998°；非best.pt重新评价。下一结构对照参考WestWorld架构与参数理由，输入输出按99D当前恢复任务适配，尚未启动新run。[暂停及方向](../docs/TWM_WESTWORLD_GUIDED_DESIGN_2026-10-14.md)。

D25复用原Oak64train/8val准备数据、fixed1024val窗口/模式与corruption协议，从头训练新6.33M结构化Mamba。相比旧pilot架构/容量/lr及Mamba指定optimizer分组均改变，不称严格单因素消融。12项GPU数值检查＋3步真实数据smoke通过，09:47UTC启动正式run，非MPJPE/真实HaWoR或物理实测。[完整配置与产物](../docs/TWM_STRUCTURED_MAMBA_RUN_2026-10-14.md)。

09:50UTC新run核step650存活，step500验证noisy腕49.090→49.204mm/root22.538→3.981°/finger15.421→5.413°；clean退化5.270mm/2.727°/3.393°，腕尚未改善。该早期结果不判最终架构优劣，best/latest已存。[准确口径](../docs/TWM_STRUCTURED_MAMBA_RUN_2026-10-14.md)。

## 2026-10-14 D25完整训练结果

新run正常完成50k（纽约08:57），代码378deda，latest50k/best49.5k可读且所有模型浮点值finite。最终fixed1024val窗口/模式：noisy腕49.090→4.594mm/root22.538→1.461°/finger15.421→1.254°；missing37.637→4.411mm/16.426→1.457°/10.910→1.249°；clean0→1.276mm/≈0→0.440°/≈0→0.500°。腕20k后趋稳，旋转/clean保持改善。best选择loss.0060013，49.5knoisy腕4.476mm、clean1.082mm；不与末步混用。8个val录制的相关滑动窗口、参考小噪声历史、合成受损与拟合MANO GT边界保持；不能据新50k比旧28k说明Mamba更优。[完整记录](../docs/TWM_STRUCTURED_MAMBA_RUN_2026-10-14.md)、[小摘要](../reports/twm_structured_mamba_20261014_final.json)。

## D35视觉框架接入数据与smoke（2026-10-15 UTC）

实现/接入使用既有DexYCB subject01、camera836212060125、right手，s0 train录制20200709_141754的63帧、val录制20200709_142123的64帧，先recording分割后切窗，未读test。RGB为原JPEG640×480，reference复用已审计的fitted MANO/腕J0；GT joint投影crop，固定camera gauge/metres，nominal frame_id/30而非实测timestamp。角色是paired视觉适配smoke，不改变四库pure-motion训练池/精度优先原则，也不是HaWoR原论文全量复现。

真实官方HaWoR post-IAM缓存、14项NPY/record的SHA/bytes、recording不泄漏、mmap和tail masks通过；source/cache及权重hash详见[报告](../reports/visual_feedback_framework_20261015.json)。默认clip32，val64全部完整参考帧进入单current NLL；腕为未对齐J0欧氏mm，root/finger为SO(3)测地角，raw指标另外按原availability计数，缺测不记零误差。20步正常反馈val腕65.048mm（旁路95.029）但root5.856°/finger8.493°（旁路4.936/6.779）退化，xyz90%coverage0。推理隐藏插件历史不是独立重训消融，不能宣称公平因果收益。未来4×60只验有限性与接口，未验新候选分布下预测精度；真正detector/移动相机/双手/长录制与独立几何评测仍待完成。


**2026-10-15短时漂移审计与D36：** 原5000步冻结模型，64条val（每源16）上单帧30mm→29.913mm、block4→29.972mm；因果五帧外推单帧3.327mm但clean3.119mm且block4为28.097mm。带oracle边界和未来的插值block4=2.182mm仅说明离线空间，不是公平在线结果。4096旧sampler窗在>6mm无历史3/4 enclosed run，但>10mm存在1419窗，不能泛称没见过短异常。确认缺口为显式可靠前缀—持续局部误差—恢复事件，非全局常数解释全部；[证据](../docs/TWM_TRANSIENT_DRIFT_AUDIT_2026-10-15.md)。用户随后授权v3修复＋两轮：GT/split/单位不变，完整遍历改变source权重，预期622updates/epoch；具体检查/启动待执行，[协议](../docs/TWM_TRANSIENT_EPOCH_RUN_2026-10-15.md)。


**D36启动验收（06:56UTC）：** full shuffled coverage每epoch636,610窗（Oak346,860/HOT132,008/Ume151,390/Dex6,352），无DDP padding、每卡512、尾353，622updates/epoch；改为自然数据比例，非等源25%。v3在每模式1024抽样上范数/角/未来隔离/隐藏值不泄露/unknown quality检查通过；mixed出现28个>6mm历史enclosed3/4run，reanchor79。初次checker把6D旋转也用position容差对比导致失败，改成分别几何单位后通过；未改变sampler来迎合检查。真实tiny片段两卡1→4更新跨epoch/partial batch/resume与独立STOP保存通过。正式验收到50更新/0.080426epoch/51,200窗口，各源/五case均正；checkpoint10 finite与CPU/双rank hash一致，W&B服务端到50。尚只有启动/初始验证，不宣称腕纠错已改善或真实HaWoR收益。GT/split/米单位/rotation codec仍同原四库拟合MANO；详见[协议](../docs/TWM_TRANSIENT_EPOCH_RUN_2026-10-15.md)、[摘要](../reports/twm_transient_epoch2_20261015.json)。


**D36完成及D37替代（2026-10-15）：** 两epoch已1244更新/1,273,220窗口使用，四源次数准确×2；best/latest当时同1244且finite，W&B58f012f8 finished。四库各256窗宏平均固定v3 val：ordinary腕4.652→4.217mm，burst4全组5.042→5.237mm、position-only6.520→6.329mm，clean0→1.457mm/59.28% >1mm退化，root/finger多数略退化，不能称短时纠错成功。源拟合MANO/合成条件，非真实HaWoR。用户随后明确从头10k，新Pod两卡512、原v3与full-coverage保留；64clip新冻结对照暂缓，未冒称已完成。历史best-only清理18保留/22删除（含oldlatest，best仍可用），记录见[完成审计](../reports/twm_transient_epoch2_completed_20261015.json)、[清理](../reports/twm_best_only_cleanup_20261015.json)。
