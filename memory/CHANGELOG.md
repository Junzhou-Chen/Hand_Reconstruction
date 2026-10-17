# HandLoop 项目记忆变更记录

**2026-10-17用户改30k并已实际重启：** 九库10k旧run按指令在816步合作保存停止，model／Adam finite、双rank和真实字节hash一致，optimizer／RNG／sampler／日志完整保留。采用fresh完整30k cosine日程重跑（非改strict resume），warmup500、每卡512／global1024、九库固定manifest／split／QC、99D＋future60及模型／loss／噪声均不变。纽约2026-10-17 01:37:31 EDT已实际到step20，checkpoint10、九源采样及W&B服务端记录验收通过；新codee759bb1，新run为runs/twm_nine_sources_moe_transient_v3_b512_30k_20261017，deadline保持10-20 01:10EDT。此决定替代此前10k目标，旧启动验收为历史；不代表30k已完成或已证实精度收益。[当前运行](../docs/TWM_NINE_SOURCE_30K_RESTART_2026-10-17.md)、[验收](../reports/twm_nine_source_30k_restart_20261017.json)。

**2026-10-16补算并核验完整Dex测试五指标：** 新增remote-only saved预测评测器、冻结官方函数独立校验器和TF32／分组精度探针；在Pod完成1280视角／78768状态W／WA／PA／Accel／RTE，保留proper PA与reset／T−2诊断。保存初次严格PA验收失败及TF32修复证据；13自测、37官方／真实案例和12旧几何对照通过。新增[结果／定义／分母文档](../docs/DEXYCB_WORLD_METRICS_2026-10-16.md)、[身份与数值摘要](../reports/dexycb_world_metrics_20261016.json)，链接原完整结果并更新本轮事实记忆。原始逐帧audit／fixture／预测及MANO资产仅远程／忽略目录，无新训练或神经推理。

**2026-10-16夜间用户最终确定九库联合TWM预训练：** 原OakInk2／HOT3D／独立UmeTrack／DexYCB全部保留，新增ARCTIC／InterHand30fps／TACO／GigaHands／HO-3D v2。此明确决定替代本轮七库／八库和暂不加HO-3D的中间范围；HumanTouch退出并已对匹配PID6739发送SIGTERM、保留已下载文件，DexWild继续暂缓。旧四库prepared manifest路径与原固定SHA必须一致，不因扩充删库、换库或重分旧split。 InterHand四JSON约394.87MB已下载，真实时序index与8手序列FK smoke通过；ARCTIC双录制／双手384状态逐帧FK smoke通过，03:50UTC两库完整CPU预处理PID7186／7187启动。TACO15.23GB和Giga11.30GB数值包已在Pod后台下载／验hash／解压，真实字段已抽查，尚未完成整库适配验收。HO-3D按物理录制去重，并从motion train排除与evaluation相同SB1组；参考来源精度、转换一致性和真实模型收益分开，不把任何脚本存在或下载中称为已训练。 新增三个数值下载／五库99D适配／严格保留原四的collection builder／带失败依赖和STOP的CPU准备队列以及九库配置。队列Pod隔离验收通过（失败后独立阶段继续、依赖blocked、STOP父子退出、已有manifest身份不升级）；无模型架构或效果结论变更。 [执行文档](../docs/TWM_NINE_SOURCE_PRETRAIN_2026-10-16.md)。

**2026-10-16升级真实手部渲染并筛选优势test案例：** 新增[持续窗口选样脚本](../scripts/select_hand_demo_cases.py)，扩展renderer平滑法线／柔光／抗锯齿、默认去骨架、直接高清裁切、明示整段指标及腕口显示端面；全部保存预测／ownβ／真实投影保持，端面不进指标。扫描1280视角，解码159候选9768帧，三段45帧的绝对／投影／腕对齐／PA均值改善；按test主动挑选及候选范围明确披露。Pod实际渲染v2，四视频完整解码、135／269帧、faststart与本机SHA及首中末帧视觉验收通过，cap前后投影0变化及原RGB视频SHA一致。新增[文档](../docs/HAND_SHOWCASE_2026-10-16.md)及[约20KB报告](../reports/hand_showcase_20261016.json)，更新入口／决策／数据／状态；媒体、资产和逐帧预测仍忽略，无新训练或神经推理。

**2026-10-16夜间下载HumanTouch、暂缓DexWild并广查直接人手motion：** 新增HumanTouch全数值下载器及限流退避，复用shared downloader可选callback且旧默认保持；远程4项续传／hash、429关闭恢复通过。新增Range拼接器及小fixture检查、DexWild源探测与单episode数值回读；批量失败及用户叫停均保留，不宣称全量成功。新增[获取记录](../docs/SENSOR_TRAJECTORY_DOWNLOADS_2026-10-16.md)、[官方直接MANO与BVH源核查](../docs/TWM_DIRECT_HUMAN_MOTION_DATA_2026-10-16.md)、小型JSON状态；纠正Dex公开10点而非full25、Inter/Re帧编号不等于30Hz单位等。更新用户决定、数据、状态与sensor记忆；无新训练、模型或数据本体提交。

**2026-10-16夜间重训前的数据扩充总结：** 新增[候选优先级与接入状态](../docs/TWM_PRETRAIN_EXPANSION_SUMMARY_2026-10-16.md)，链接既有池统计／官方发布，区分完整MANO适配、native部分监督和视觉估计扩展。补MySign、SynData glove子集精确口径、HOI4D当前直链；纠正ARCTIC subject模板属于SMPL-X。ARCTIC下载完成但未入训、HumanTouch样本ready=false，未启动新任务或变更训练方案；仅同步本轮调研文档与新增事实记忆。

**2026-10-16制作并实际验收视频重建demo：** 新增remote-only renderer、[使用与来源文档](../docs/HAND_OVERLAY_DEMO_2026-10-16.md)、[约15KB验收摘要](../reports/hand_overlay_demo_20261016.json)，.gitignore忽略demos。用户现有Dex测试片段与保存预测生成原RGB／baseline／best4200三列、mesh骨架及放大窗；5秒原速／约10秒慢放，四视频完整解码和本机SHA通过。v1显示窗口不够放大，v2仅共同裁切改进且几何／原视频SHA不变；左右手中帧人工查看通过。媒体留忽略目录，无新模型实验。

**2026-10-16广泛复核其他论文如何报告精度：** 新增[11篇官方全文／主表审计](../docs/HAND_ACCURACY_REPORTING_SURVEY_2026-10-16.md)，覆盖2024–2025代表作与2026 Hand3R／ScaleHP／InfiniHand；纠正助手“PA仅辅助”的笼统表述，推荐局部与绝对空间并列。记录W对齐、joint数和动态GT定义差异。更新记忆入口／数据／状态／写作／来源；只调研和文档，无远程实验、指标实现或研究方案新增决定。

**2026-10-16完整DexYCB测试最终汇总：** 新增[双指标结果表、分支／左右手诊断与协议边界](../docs/DEXYCB_FULL_TEST_RESULTS_2026-10-16.md)、[约90KB机器可读报告](../reports/dexycb_full_test_final_20261016.json)，完整测试协议链接最终结果；核对1280视角／78768状态、三方逐帧覆盖及W&B finished终值。best相对官方绝对MPJPE下降53.36%、PA下降12.02%，finger稍退化且不能独归因TWM。官方全文与fork evaluator复核PA／AUC；遮挡组未测。只读取数／更新文档和新增事实记忆，无新实验。

## 2026-10-16 复核HaWoR状态如何进入视觉插件及改进候选

- 在[视觉反馈框架记录](../docs/VISUAL_FEEDBACK_FRAMEWORK_2026-10-15.md)补现行两轮职责与源码事实，保留D35单轮冻结历史。明确F／Z／β分别使用，raw是当前微调主干的固定帧内观测，Z每轮仍独立传入TWM。
- 记录共享状态投影相加仅在该嵌入支路对角色对称的数学事实，区分整体几何与TWM分槽；逐组availability不是visibility／质量，同帧σ／κ与隐藏特征没有直接回传。WiLoR／DIR／GeoHand主文与官方refiner核查，候选修改仍标建议。
- 保留99D／现HaWoR／纯数值TWM／当前5k，不改loss或启动模型。拟先分阶段几何诊断和单因素验证，不能将raw→final误差差值归因单一插件。更新方法、进展与变更记录，其他聊天改动保留。

## 2026-10-16 再次核对TWM原始数据规模与扩容建议

- 远程实际读取四库manifest与D37原metadata，确认2788135 train手状态、636610重叠窗及分源完整60步future覆盖；输出小型[池清单](../reports/twm_data_pool_inventory_20261016.json)。不是把视觉RGB扩大当新运动，亦不把重叠窗当独立录制。
- ARCTIC下载已complete343/343，267官方train参数计数384780双手状态，QC／单位／模板／J0／split验收前不得称新增训练净量。HumanTouch／DexWild／DexCap／DexterHand官方公开版本复核并保存[候选摘要](../reports/twm_public_candidate_review_20261016.json)；GRAB／HMP等职责复核，新资源访问未验者不抬为优先。
- 在[现有来源报告](../docs/DATA_EXPANSION_SOURCE_AUDIT_2026-10-16.md)补最新状态与建议：先ARCTIC，glove需native标定和partial监督，输出保留99D；建议未执行，未新增训练／大包下载、改当前5k或既定模型／loss。更新传感器与状态专题，保留其他聊天文档改动。

## 2026-10-16 只读核验两库5k训练中期效果

- 远程14:26UTC／纽约10:26快照1773/5000仍training；读取原始HaWoR参考和W&B全部35次fixed8验证，保存[小型进度摘要](../reports/visual_dex_hot_geometry_5k_progress_20261016.json)。源码a642e6e／原冻结清单与既定配置保持。
- best按原geometry score为1500，腕46.412→9.424mm；最新1750 final MPJPE12.431mm。分清整体相对原模型的改善与插件增量：1750微调raw腕9.770→final10.102mm，手指16.722→14.417°；不能将整体腕收益写成TWM独立收益。
- 明确仅8clips／222完整状态、非全量val／独立test，近期波动与位置分布欠覆盖均保留。未停止／恢复训练、修改loss／数据、启动新模型评测或改变后台下载。

**2026-10-16最新用户决定：先用已验收Dex＋HOT跑5000步，其他数据继续后台准备。** 用户明确效果优先、不必完全复现HaWoR，可增加与测试隔离的来源；当前指定此前63652独立RGB／73524 train手状态池立即重训5k。本条替代等四库就绪才启动、tiny Dex-only及前轮等用户回来再训练的安排。固定6314ec2…混源清单／2773train窗，原fixed8验证、两主干全量低LR／每卡8段／两轮／history30／噪声与geometry目标均保持；5000总更新＝3750视觉＋1250 motion回放。从官方HaWoR＋D37重新初始化，全新Adam和5000步cosine；不续用旧100步优化器。ARCTIC／HO3D下载与HOT全clips预处理继续，新来源不在本run中自动入池；旧best与旧100步保存。已于08:25UTC／纽约04:25正式双卡开训，源码a642e6e，08:40实测30/5000。step4的1076 model／3135 Adam浮点张量finite且CPU／双rank hash一致，step8两rank14组梯度seen非零／代表参数改变均通过，每卡peak63.50GiB，W&B ff3108fd online。HO3D下载校验与HOT全clips预处理随后complete，ARCTIC仍下载；本run数据不变。见[两库5k协议](../docs/VISUAL_DEX_HOT_GEOMETRY_5K_2026-10-16.md)；启动通过不代表新精度结论。

**2026-10-16下载自查与训练范围澄清：** 用户表示等下载完成再来，本轮只添加并实测只读查询helper，不启动训练或改变后台worker。本机可执行`ssh ds 'python3 /sciclone/data10/jchen57/Hand_Reconstruction/download_runtime/check_hawor_data_progress.py'`，同时显示Dex／ARCTIC／HO3D下载、校验文件数／失败数／进程与HOT预处理；不读取私有凭据。complete表示下载及校验完成，checking未完成；HOT预处理单独计数。解释作者逐帧／视角ID清单不等于新增小样本要求，公开train可组织完整有效训练池，val/test排除；当前Dex单视角与HOT BOP覆盖仍有限，不能把完整下载包称为四库全量已接入。5000总更新不保证全池一遍，未据此改变预算。见[当前协议](../docs/HAWOR_ALIGNED_VISUAL_TRAINING_2026-10-16.md)。

**2026-10-16最新用户澄清：视觉联合数据对齐HaWoR四库，保留几何监督与5000步。** 本条替代刚准备且从未启动的“原1128帧Dex-only 5k”数据选择；按官方§8.1使用HOT3D／ARCTIC／DexYCB／HO3D来源，作者573K／165K／169K／66K是其选帧数，未公开完整训练清单／混合概率，不声称完全复现约1M帧。TWM motion仍原四库／D37；双主干低LR全量、每卡8段、3:1／两轮／history30／噪声与raw J0＋final自身β直接几何均保持。新增5k配置显式要求四个visual train sources，真实Dex＋HOT清单的缺库拒绝与旧配置不变已在Pod验证；ARCTIC／HO3D adapter尚未完成，5k未开训。HOT已按86f25ec在Pod展开全部已下载1167 Aria clips，原父录制split及27评测排除保持，补齐既有cv2依赖路径后07:44实测320clips；metadata的99父录制不等于论文110完整train视频。用户直接提供账号后已用SSH管道保存远程600权限私有文件、无凭据值入源码／日志／Git，ARCTIC官方小ZIP SHA验收通过，07:43启动343包后台下载，07:44 verified2／failed0。Dex全包已有，扩大单视角50861 train有效状态不是作者169K；HO3D官方v2获取流程处理中，版本对应66K仅工程推断，尚非已接训练。旧混源正常停100且CPU checkpoint审计通过，旧best保留。[当前协议与准备状态](../docs/HAWOR_ALIGNED_VISUAL_TRAINING_2026-10-16.md)。

**2026-10-16新用户决定：撤回扩展数据，原Dex几何监督5000步。** 用户要求取消当前扩大数据，只先检验第2项直接几何监督，并从1000提高到约4000／5000步；实现选5000上限。本条替代前轮扩大数据＋1000步安排，第3／4项条件／历史／噪声／回放不改。混源run已07:14:26UTC正常stopped100/1000，checkpoint100与双rank hash3fff5e8f…保存；不删数据或旧best。新视觉source恢复原67462dce…清单：16train录制／1128帧／46窗，val8录制／577帧／24窗；不用扩大Dex或HOT RGB。原四库motion与D37仍保持，从官方HaWoR＋D37＋新插件、全新Adam重新起训，不载入混源权重。保留NLL1e-4＋raw腕／final自身β绝对joint距离各/0.05m；同LR／batch8／3:1／两轮／history30。实现另将同一原val每50步覆盖全24窗，非新增训练标签；cosine按5000从头安排，非旧1k简单续跑。当前新配置与协议已准备，启动待验，尚无新精度结论。[本轮协议](../docs/VISUAL_GEOMETRY_ORIGINAL_DEX_5K_2026-10-16.md)。

**2026-10-16数据／几何v3启动验收（本聊天远程实测）：** 完成前两项修改：Dex＋HOT73,524 train手状态／63,652不同RGB／2773窗，val4037状态，原Dex577val字节保留；严格SHA／split／loader通过。相机精度假警报86f25ec修复，真实mixed全主干及DDP全局计数通过。06:50:51UTC新联合训练启动，16/1000快照、14组双rank梯度／更新与checkpoint4 CPU finite／hash通过，每卡peak63.50GiB，W&B98ff0737在线。只改数据和raw J0／final自身β几何；motion仍四库／D37，HumanTouch native样本未入训，原第3／4／5项未改。初始化变差已记录，不由启动推出收益。更新[协议](../docs/VISUAL_DATA_GEOMETRY_V3_2026-10-16.md)及小型启动／验收报告；并发本地整理的其他内容不纳入本聊天提交。

2026-10-16用户采纳前两项、暂缓第3／4／5项：扩查HumanTouch／DexWild／DexCap／GRAB／ARCTIC／DexterHand／HandX／TACO的官方开放、GT来源、motion与视觉职责，新增固定HumanTouch native样本获取与结构检查（ready=false，不造99D）。已扩大Dex视觉train为50861有效帧／800录制／10主体，val577字节保留；实现HOT真实RGB标定链和不可变多源合并，烟测通过但full99clip投影精度自检失败待修，失败产物保留。新增可配置raw J0与final自身β绝对joint监督、按源21点mapping和loss-only标签字段，双rank计数／mask／GT隔离及两源独立真实梯度通过，旧DDPfixture无输入失败由7e1535c修复。新增来源审计、执行协议、v3配置，更新入口／02／03／04／08；NLL1e-4＋两项/0.05m是train-only梯度支持的初值而非收益证明。1000步／LR／batch／历史／噪声／回放未改，当前尚未启动新run。旧best和其他聊天的PA／AUC审计全部保留。

**2026-10-16 AUC补测完成：** 同test1015保存预测，官方HaWoR→预选best300 final的PA-AUC **89.83933%→90.62120%（+0.78187个百分点）**；绝对AUC13.59279%→33.08518%，仅腕平移对齐71.10035%→77.20214%。last1000 PA90.73607%仅终点对照；val577 PA89.40736%→best91.32986%。沿作者0–50mm／100阈值、逐joint PCK≤与梯形积分归一化；官方EvalUtil每split69项独立核对AUC／曲线差均0，原五项几何均值与上轮CPU结果最大差8.882e-16mm。论文PA-AUC90.5%为不同完整集合，当前小子集不作排名。Pod CPU仅FK／指标，无视觉前向、训练、权重或best选择改变。[协议与结果](../docs/HAWOR_BASELINE_METRIC_AUDIT_2026-10-16.md)、[小型摘要](../reports/hawor_auc_audit_20261016.json)。

**2026-10-16 HaWoR论文指标差距核查完成：** 论文DexYCB Table1的4.76mm是PA-MPJPE，原报告70.532mm是绝对MPJPE，均为毫米，不能直接比较。Pod CPU重解码同一保存预测后，test1015原官方HaWoR PA **5.08159mm**、预选best300 final **4.68934mm**（同集PA下降7.719%）；baseline比论文数值高6.756%，非十倍。仍16录制／单视角prefix128子集，不能据4.689<4.76宣称胜过论文完整test。新增可选PA评测，原四项逐帧差<0.001mm，无训练／视觉前向或best重选。[核查与来源](../docs/HAWOR_BASELINE_METRIC_AUDIT_2026-10-16.md)、[小摘要](../reports/hawor_paper_metric_audit_20261016.json)。

2026-10-16下一轮训练分析：记录用户“以原生baseline→final的平均纠正为主，无逐帧保证要求”，复核已有20点fixed8曲线、46clip／1128帧的小视觉池与scheduler；补查可用group的预训练raw=candidate和真实多轮条件差异。核官方HaWoR compute_loss（其3D为root-relative，不自动约束绝对深度）、StableHand、PoseBERT、DIP、DF、WHAM与WestWorld训练协议，提出数据覆盖、raw绝对腕／finaljoint监督、条件／历史对齐、回放和3k/5k版本化预算的单因素建议。新增[下一轮训练分析](../docs/NEXT_TRAINING_ANALYSIS_2026-10-16.md)，更新入口／02／04；未实施新loss／采样／schedule／权重／checkpoint选择，未新训或做模型推理，不将候选写成用户决定。

2026-10-16联合失败深诊断完成：按用户“先充分分析、再讨论怎么改”，在授权Pod完成冻结三权重（D37best10000、D40预选best300、last1000）×1024固定val窗口×五种原v3模式配对评测，原视觉val577／test1015状态的固定raw/candidate/完成history换TWM及oracle/empty控制，逐轮几何／shape分解、官方camera逆尺度反算，以及train两侧各2条16帧录制的eval只读梯度。输入／目标hash和各probe范围前后模型／源checkpoint／被测参数hash通过；无optimizer、无训练或权重变动。执行88c7c00、梯度5154594、camera11159dd、聚合a72154d，与原训练87a2bcc／eb8d0b9分开。

关键结论保留两面证据：pretrain noisy左／右腕2.782／2.945→jointbest3.548／3.689mm，两侧部分保持退化；但同一真实savedbest上下文，jointbest62.512／26.561优于pretrain70.849／29.069mm，不能直接换回旧TWM。少数camera尺度低估经逆比例放大到米级，左>500mm baseline0／best5／last6帧，最坏last final腕4.201m，不删样本或换best。shape固定状态控制不支持米级主因，β上游作用未排除；局部camera final/raw冲突非统一左反号；NLL/51后仍饱和moduleclip，不能据1020系数比直接修复。短history／quality未知／fixed8左19%与单subject梯度的限制同报，没有唯一根因或新loss/freeze/camera/history方案决定。

新增[主诊断](../docs/JOINT_FAILURE_DEEP_DIAGNOSIS_2026-10-16.md)、[TWM保持性](../docs/TWM_HAND_RETENTION_DIAGNOSTIC_2026-10-16.md)、[聚合报告](../reports/joint_failure_deep_diagnosis_20261016.json)与图，更新03／04／08；远程`evaluations/joint_failure_deep_diagnosis_20261016/`保留完整窗口索引、逐窗逐帧CSV、numeric输出与日志，Git不含数据／权重／许可资产。下方“无新增推理／尚无终点motion评估”保留为上阶段历史，现由固定1024窗专项补充，非完整val或官方ego benchmark。

2026-10-16左手覆盖核查：按用户追问，在原Pod以实际D37 loader/manifest重新索引，train左317226/右319384窗口（49.83/50.17%，四库均有左），视觉train左575/右553帧，排除未覆盖左手；源码手别embedding/PCA/镜像/anchor未发现确定bug。已有test分阶段左MPJPE61.602→raw109.754→candidate60.209→final63.735mm，左腕/腕对齐最终仍好，指出联合定位适配与补偿而非先验左手全部失效。fixed8选择左40/右170states，test左9录制5好4差/右7均好；不调整test、权重、loss或训练，shape/镜像偏差不写成已证根因。索引/汇总均远程，无新网络推理；新增[诊断](../docs/LEFT_HAND_COVERAGE_AND_FAILURE_DIAGNOSIS_2026-10-16.md)及[小型来源报告](../reports/left_hand_diagnosis_20261016.json)，更新03/04及入口。

2026-10-16 04:25:29UTC（纽约00:25）：D40训练已03:49:16UTC complete1000，训练eb8d0b9、原fixed8预选best300/last1000终点对照。按用户要求在原Pod完成e72f7ed远程fullval24clips/577states/8rec与独立s0 test37clips/1015states/16rec配对RGB＋MANO FK评测；test官方HaWoR→best腕76.335→46.187mm（−39.49%）、MPJPE70.532→48.194mm（−31.67%）。新发现必须保留：左手MPJPE61.60197→63.73519mm退化、右手81.26346→29.51666mm改善，非双手普遍泛化；27.68%帧MPJPE退化>1mm，best自身raw腕对齐与末轮candidate→final亦退化，候选已含首轮TWM不等于整TWM消融。best/last test90%coverage仅60.49%/36.65%，last MPJPE均值略好但recording-bootstrap区间跨0。test在解码前固定subject03–10各2条/raw frame_id<128、9左7右/0拒绝、与preparedtrain/val不重叠，manifestSHA b5a87d00878f3132692fba81aba68fea38a8fb6dd6980faf82e9fc39e43963a4；TWM见过相同主体其它train运动，仍GTcrop固定camera拟合参考，非官方全流程benchmark。`evaluation_driver_status` complete04:25:29.195702UTC；训练/评测均complete，未改训/权重/Job，没有新增终点全量motion保持或未来能力评估。[最终报告](../docs/VISUAL_JOINT_FINAL_EVALUATION_2026-10-16.md)、[SHA、分组与小汇总](../reports/visual_joint_final_comparison_20261016.json)。

历史快照：2026-10-16 01:32UTC（纽约10-15 21:32）：只读D40联合训练记录/W&B，实际253/1000、最新fixedval/best250，两rankhash一致；8clips/210states/同identity，腕原raw88.250→当前raw65.478→final48.598mm，root8.591°/finger7.530°。保留训练波动、37.1%局部腕退化、90%coverage40%/std32.622mm和NLL回升。motion同四源16窗noisy当前腕2.959→3.514mm/指0.696→0.896°，1s100.370→94.874mm、2s三源152.983→172.615mm（差于hold169.452）；不能称运动能力全保持或最后候选为纯插件。远程CPU汇总、zhou生成曲线，未改训/新推理。[报告](../docs/VISUAL_JOINT_PROGRESS_2026-10-16.md)、[数值](../reports/visual_joint_progress_20261016_0132.json)。

2026-10-16 00:59UTC（纽约10-15）：D40按用户要求RGB每卡4→8，旧65步共同保存停止并归档，eb8d0b9显式允许单项batch变化完整resume，保持clip32/motion32/LR/3:1/总1000。实际76步、checkpoint69双rankhash一致；73步两rank14组finite/非零seen/代表参数改变全通过，每卡峰值allocated63.498GiB、单时刻nvidia-smi79446MiB。W&B同run3abfc864服务端running/75、runtimebatch8/resume_start65；optimizer/scheduler/RNG恢复，loader游标重建。实现新增配置守卫、执行metadata历史、运行batch scalar、续跑首8梯度检查/第4次early保存；无精度或吞吐改善断言。[摘要](../reports/visual_joint_batch8_resume_20261016.json)、[协议](../docs/VISUAL_JOINT_FINETUNE_2026-10-16.md)。

2026-10-16 00:43UTC（纽约10-15）：D39按用户双主干低LR全量决定，以eeac1fc正式双卡每卡4×32帧/motion32实际10/1000更新、37.742GiB峰值，14组梯度/参数改变两rank通过、latest4模型hash相同，W&B3abfc864服务端running/step8。在线RGB＋两轮MANO16候选取证，1128train/577val帧；D37已complete10k/best10000固定。CPU/AMP/真实RGB、smoke4→8恢复、1076model/3135Adam finite和trained16RGB当前/未来接口通过。保留初次AMP失败、同inode临时别名修复及初始化两轮几何差于raw的负面基线；尚无训练后fixedval提升、不改D38loss。[新协议](../docs/VISUAL_JOINT_FINETUNE_2026-10-16.md)、[启动摘要](../reports/visual_joint_launch_20261016.json)。

2026-10-15 22:32UTC：按用户要求复核另一聊天“梳理项目记忆文档”搭建的D35 HaWoR反馈首版。确认每帧一次插件/一次TWM、σ/κ只回插件、旧视觉best内嵌D32且未接D37；核远程best/缓存/RGB预测/HaWoR仍在，旧latest及独立参考已按best-only删除。补源码与产物存续说明，无训练/模型执行/精度重测或新设计决定。[框架](../docs/VISUAL_FEEDBACK_FRAMEWORK_2026-10-15.md)、[审计摘要](../reports/visual_feedback_integration_audit_20261015.json)。

2026-10-15 22:22UTC：D38用户明确loss放下版，本轮只查效果。D37训练8010/10k、完整验证8000：普通腕4.652→2.881mm，clean0→0.601mm/>1mm退化12.60%，reanchor0.267→1.296mm/30.96%；5k→8k score下降21.05%。1s/2s腕96.954/140.930mm，长期仍弱且旋转不全改善；两份汇总同validation SHA、聚合/固定输入核验通过，远程CPU绘图，无改训/新推理。[报告](../docs/TWM_TRANSIENT_8K_PROGRESS_2026-10-15.md)、[摘要](../reports/twm_progress_comparison_20261015_2220.json)。

2026-10-15 20:24UTC：按用户要求完成12篇损失/训练协议与关键官方实现对照，纠正“主流都单loss”“StableHand单FM”“WestWorld为MoE balance背书”等潜在误读。建议概率主线＋同架构逐项辅助/初值梯度消融，生成路线另需采样及可靠性接口重设；全部标未采纳/未实施，D37原配置保持，未添加新损失或运行实验。[完整调研](../docs/TWM_LOSS_LITERATURE_REVIEW_2026-10-15.md)。

2026-10-15 20:09UTC：用户要求专项查看未来预测。D37完整验证5000，腕0.1/0.5/1/2s相对hold改善35.6/25.5/14.2/1.2%；2s仅三源，HOT/Ume退化，干净输入亦仅改善3.7%，low_motion比hold差。新增只读forecast汇总/绘图脚本，核宏平均与时域counts，所有分析远程CPU；报告ADE/oracle加权差异、逐轴coverage与宽区间、多模态未证实和非闭环边界，训练继续。[专项报告](../docs/TWM_FORECAST_PROGRESS_2026-10-15.md)、[摘要](../reports/twm_forecast_progress_20261015_2012.json)。

2026-10-15 19:48UTC：D37只读效果检查，运行4,700/10k、完整验证4,500。普通腕4.652→3.331mm/位置四帧6.520→5.244mm，3.5k→4.5k score下降32.9%；clean0→0.884mm且30.66% >1mm退化、近准当前/错误历史0.267→1.471mm，仍有失败。14模式candidate跨验证不变及等源聚合通过，修复reviewer漏读forecast NLL/漏收模式/续跑initial时间边界，远程CPU汇总绘图；不改训练。[结果](../docs/TWM_TRANSIENT_10K_PROGRESS_2026-10-15.md)、[摘要](../reports/twm_transient_v3_progress_20261015_1944.json)。

2026-10-15 16:10UTC：D37新Pod正式87a2bcc从头10k已验收到40更新，checkpoint10 finite/CPU与双rank hash相等，W&B a9cb4888服务器到40；full-coverage循环/step budget的3→6更新跨三遍历、尾批次/resume通过。历史18best保留、22非best已删除2.51GB并SHA检查，source/data/metadata/logs留存。[运行](../reports/twm_transient_v3_10k_launch_20261015.json)、[清理](../reports/twm_best_only_cleanup_20261015.json)。

2026-10-15：D36两轮正常完成及固定验证失败如实归档；用户改D37为新Pod从头10k，并明确历史只留best。新step-budget/full-coverage双卡3→6更新/resume/三遍历通过，正式已发起待实际更新验收。历史18best finite/SHA通过，22非best删除约2.51GB，元数据/日志/推理结果保留。

2026-10-15：收到D36异步回复，完整遍历每个窗口一次/先跑2epochs从默认工程选择更新为用户明确确认；配置与已运行任务一致，无需重启。用户可登录ds，先前连接问题已恢复。

2026-10-15 06:56UTC：D36正式25fee71从头两卡每卡512按2个完整epoch启动，验收到50更新/0.080426epoch/51,200窗口；四源五case参与、首checkpoint有限/两rank同步，W&B服务器曲线通过。语义、全索引覆盖、跨epoch续跑、尾批次和STOP集成检查通过；保留初次checker单位错误及SSH/pull失败修正。原目录untracked视觉文件未动，独立data10源码worktree运行，D35不改。[验收](../reports/twm_transient_epoch2_20261015.json)。

2026-10-15：保存冻结短时漂移诊断，校正阈值覆盖的过强表述；D36用户要求修复并改两epoch。D34已停5519并finite/sync审计，接入v3事件增广、内部位置尺度及全覆盖epoch/cursor；远程检查与启动待执行。

## 2026-10-14 中文 Introduction 回顾与重写

- 按用户要求回读原始引言及首段讨论，结合现行方法，在对话中给出中文参考稿；保留原始档案，新稿尚待用户确认。
- 复核 DexCap/HaWoR/DIP/StableHand v1 原论文及 HMP 官方项目，修正“前作主要只预测未来”的笼统对比；收紧设备普及范围与物理能力措辞，聚焦条件恢复、视觉取证、可靠性重估和最终状态反馈。
- 更新写作专题与进展，无新用户设计决定、实现、训练或性能结果；研究贡献仍是待验证的设计表述。

## 2026-10-14 四库完成与实际字段审计

- 用户要求查看是否下载完、四库是否均适合TWM。核所有remote status complete，634/2461/1021/13文件全verified/failed0，531865069350B；Dex完成于04:15:58 UTC。ARCTIC继续D22暂缓。
- 新增远程-only样本审计源码056122f，06:08:55 UTC在Pod/zhou CPU读取3Oak×20帧、4HOT clips、4Ume real序列、6Dex labels与calibration；详细JSON仅留data10/audits，小型摘要见[四库记录](../docs/TWM_FOUR_DATASETS_AUDIT_2026-10-14.md)。未读图像/跑网络/训练。
- 核实际Meta PCA15、Dex PCA45+hand mean、Oak quaternion16；Ume native与MANO独立缺标签、序列非固定帧数，无绝对info timestamps；HOT真实dt有jitter；Dex TAR迭代乱序。更新99D解码/时间/mask/split注意事项，不宣称FK、真值精度或全库质量通过。
- 建议三MoCap来源核心＋Dex train补充/目标域适配，维持sensor优先与glove候选，未固定混合比；train/test严格分开。更新入口、状态、数据、表示专题及下载完成记忆，正常同步GitHub。

## 2026-10-14 ARCTIC暂缓与DexYCB优先

- 用户回复尚未注册ARCTIC，当前先下载DexYCB，登记D22，替代D21本轮推进ARCTIC的安排；保留清单但不继续索要凭据、注册或启动ARCTIC。
- Dex已启动进程不做重启或终止；本轮只读查询最新状态，未运行预处理/训练。进度以真实远程记录为准，不把旧约30GB快照当当前完成状态。
- 两次SSH关闭后只读重试成功，04:00:17 UTC快照Dex仍alive/running，已写85.51/127.61GB含partial，verified7/checking2/failed0。连接原因未定，未改配置或进程；未宣称下载完成。
- 更新入口、决定、进展与下载操作说明，维持数据/日志仅在远程的约定。

## 2026-10-13 追加DexYCB与ARCTIC获取

- D21用户明确下载两库；继续计算Pod执行，不在登录节点传输大数据。Dex13个官方Drive分包127611176595B后台启动，3workers；03:36 UTC快照约10.66GB含partial、verified3/failed0。
- ARCTIC官方343包（cropped＋4个标注/元数据）清单与SHA固定repo49f3eb4b；匿名返回登录HTML，现有服务器配置无账号，已问用户私有路径。状态waiting_credentials，没有真实数据下载或认证成功记录；不通过第三方包绕注册。
- 新增下载器、无secret配置模板、7项远程检查；扩展ds helper的--datasets/--credentials，不重跑已完成三库。Dex按Drive大小/gzip CRC/TAR格式＋自算SHA验收，不声称作者SHA匹配；ARCTIC按官方SHA。
- gdown6.4.2隔离安装于data10，calibration14464B实测成功；7项HTTP续传/HTML拒绝/gzip完整性检查在Pod通过。维护.part忽略；新增[状态与配置文档](../docs/COMPARISON_DATA_DOWNLOADS.md)，更新入口/决定/进展/运维。无解压、99D转换、模型推理或训练。

## 2026-10-13 下载完成与HaWoR对比数据复核

- 用户报告三库完成后，只读核远程status：Oak634/HOT2461/Ume1021文件全部verified、failed0，合计404.25GB；原下载进程正常退出。保存完成时间，不重复hash404GB、解压或训练。
- 核HaWoR论文§4/6–8与官方main固定SHA66c7d410：视觉训练四库973K取整帧、infiller单HOT；量化主HOT Aria27val/world与Dex test/camera，ARCTIC/HO3D不作原论文主测试，EPIC定性。
- 新建[HaWoR数据与对比协议](../docs/HAWOR_DATA_AND_COMPARISON_PROTOCOL.md)：HOT full27与BOP349clips/25recordings区别、官方GT crop/右手/valid/100帧含scale对齐、不调infiller；Dex作者fork默认s0_test、遮挡evaluator及缺预测跳过、推理/依赖缺口。
- 勘误：HaWoR export_gt生成val.json，无需另找未发布静态split；官方HO3D已公开evaluation几何GT。建议HOT主＋Dex补，未把建议升级为用户已选split，未改变pure-motion pool、启动新数据下载或模型实验。
- 更新入口、数据专题、状态、决定注记与运维；代码/文档正常同步GitHub，数据和日志留远程，原始档案不改。

## 2026-10-13 三库远程后台下载

- D20用户要求下载OakInk2/HOT3D train/UmeTrack，澄清ds登录入口CPU少，需进入现有计算Pod；脚本确认正常后启动并结束对话，由用户后续检查。
- 新增`download_twm_data.py`、`ds_downloads.py`与4项续传/校验检查，固定官方源revision，支持Range、重试、hash、锁、状态与后台进程；`.gitignore`补partial/坏文件隔离规则。
- 在ds home建立GitHub checkout，数据只存远程data10；12CPU Pod内4项HTTP检查全部通过。代码`853d2da`于22:57 UTC启动三库各4workers；22:58:42 UTC三进程存活，首批校验通过、failed=0，仍在下载。
- 范围Oak全627姿态＋元数据36.67GB，HOT排349个HaWoR评测source clips后2455TAR约257.07GB，Ume HF real train1020TAR约110.51GB，合计404.25GB。HOT BOP不包含27评测ID中的2个，记录缺席而不误判为映射失败；官方train IDs/device一致性已验。
- 新增[下载与检查](../docs/TWM_DATA_DOWNLOADS.md)，更新入口/进展/远程记录，原始档案不变。没有下载到本机、提交数据/日志、生成99D、训练或宣称全量下载完成。

## 2026-10-13 初始建库

- 按用户“仔细理解完整项目细节和 idea，构建项目记忆库”的要求，阅读全部分文档、核对总档案精确重复关系，并检查旧方法图。
- 新建根目录 `MEMORY.md`、`AGENTS.md`，以及方法、决策、数据评测、进展、写作引用、来源索引六份专题记忆和来源 manifest。
- 保留核心纠偏：当前状态恢复优先；高 confidence 可错；可信局部观测约束其他关节；纯轨迹 TWM 与真实图像证据分工；帧内校正和跨帧反馈。
- 区分用户明确决定、现行整合设计、建议超参、历史废弃方案及未实现事项；登记 TWM 输入契约、mask、坐标方向、初始化、质量目标等开放问题。
- 核验部分官方论文与源码；记录 HaWoR infiller 内含 camera-to-world 转换，以及 HOT3D/DexCap/EgoDex 的关键数据限制。
- 全部原始材料保持不变。本轮没有训练、实测性能、投稿结论或模型实现。
- 完成独立方法复核与外部来源复核；检查 39 个本地链接有效、6 个原文件哈希未变、来源 manifest 一致。

## 2026-10-13 TWM公开数据专项调研

- 按用户“仔细调研有哪些开源数据集可以pretrain TWM”的要求，核验官方论文、项目、代码、许可入口和部分公开release元数据；新增[专项报告](07_TWM_DATASET_SURVEY_2026-10-13.md)。
- 建议GRAB+ARCTIC建立纯轨迹基线，OakInk2扩任务，HOT3D按评测协议使用公开train GT，DexWild扩手套规模；HumanTouch已有文件，许可与完整发布量待核实。建议未升级为用户决策。
- 区分marker/glove、视觉拟合和设备追踪；分开finger/wrist来源、world坐标和有效性；纠正整库AMASS真实手指、多视角图片当时间帧、TACO手MoCap、HandX全量自采等误读。
- 审计HandX source mapping/test NPZ索引；核验OakInk2 annotation文件索引、HumanTouch X001/X002元数据；均不等于全库数值质量验收。
- 更新MEMORY入口、数据评测、进展、引用边界；未改原始档案、未实现loader、未训练，也未启动TB级数据下载。
- 交叉复核重点数据集表述，检查本地链接、关键时长换算与原始六个文件哈希；未发现原档改动或无效本地链接。

## 2026-10-13 GitHub与远程实验约定

- 用户明确GitHub仓库、main分支、忽略大文件，并指定本机写代码/论文、hongfei远程执行全部实验；登记为D13/D14。
- 初始化Git，新增`.gitignore`、README及开发/远程专题；忽略数据、权重、视频、缓存、原始日志和私有配置，保留论文源文件和小型结果。
- 成功核验GitHub初始无refs和SSH连接，远程根目录、git与nvidia-smi存在。最终同步结果见[开发与远程约定](08_DEVELOPMENT_AND_REMOTE.md)。
- 初始提交`20fe231`已推送main，hongfei已clone且与初始提交一致；创建忽略的data/checkpoints/outputs/logs。原档的Markdown换行空格保留，原始六文件哈希不变。

## 2026-10-13 产业数采覆盖补充

- 接受用户指出首轮偏学术、漏查产业团队的纠正；不再将首轮短名单视为完整公开资源全景。
- 新建[产业补充调查](09_INDUSTRY_HAND_DATASETS_2026-10-13.md)，覆盖SynData、OOJU、Daimon、OpenGraph、EgoVerse、Lightwheel、BeingBeyond，以及厂商商业样本/机器人数据的边界。
- 读取UniHand官方Human_Data.md及9个人手TAR索引（约68.43GB）；确认world wrist/MANO字段和来源聚合，不下载大文件。
- 更新入口、数据专题和后续优先级；保留来源质量分层，也列出可申请的大规模视觉数据。没有自动申请数据、联系公司或开展训练。

## 2026-10-13 完整方法框架与免申请数据原则

- 用户明确免申请公开数据优先、参考论文实际用法，登记D15；替代原先需注册数据默认开工顺序，保留调查事实及后备资源。
- 新建`docs/METHOD_FRAMEWORK.md`，连贯整理任务、单手69D、三主体/两闭环、统一TWM接口提案、残差几何、w/q与三阶段训练；明确当前无实现或实验结果。
- 补充预训练raw/candidate槽位、anchor残差、最终q取证、缺测/首帧与单手共享参数等建议契约，不升级为用户决定。
- 核验HMP/HaWoR/StableHand/PAD-Hand实际数据角色；同步更新入口、方法、决策、数据和下一步，原始研究档案保留。

## 2026-10-13 69D与HaWoR原生输出澄清

- 按用户提问核查HaWoR实际源码：原生为MANO root/15关节旋转、translation、shape；视频支持左右手，原infiller为218维双手输入。69D是本项目每手统一表示，不是HaWoR原生张量。
- 明确69D/66D是编码长度，非独立物理DoF；记录MANO translation与wrist joint区别和adapter转换公式。
- 核查DexCap采集代码21点与orientation，区分源码字段、release样本和尺度/坐标验收；补充公开数据的完整/部分69D监督能力表。
- 更新方法和记忆；无数据下载、模型推理或实验结果。

## 2026-10-13 主流表示与MANO标注核验

- 按用户请求核验代表性重建/运动方法，确认MANO参数路线常见，但内部特征与输出不同，HandX/METRO提供坐标路线；69D为本项目候选而非行业标准。
- 新建表示与标注专题，记录视觉/depth拟合、marker→SMPL-X/MoSh++、HOT3D UmeTrack→MANO及FreiHAND human-in-the-loop流程；区分拟合reference与直接测量。
- 补充MANO参数输出可通过joint/2D/depth损失训练，提出MANO输出+joint监督baseline与69D对照建议；最终编码未由用户定案。
- 核验PAD-Hand公开代码、StableHand54D、HO3D发布条款；保留HaWoR论文“独立手”和当前218拼接实现的不同证据。
- 更新方法、状态和入口，原始档案保留；没有模型运行、数据下载或实验指标。

## 2026-10-13 WestWorld与视觉手重建方法精读

- 按用户要求阅读WestWorld v2和代表性单帧、视频、运动先验/恢复方法；新增[精读报告](../docs/WESTWORLD_AND_HAND_RECONSTRUCTION_REVIEW.md)，含方法、原文定位、关键源码SHA和研究判断。
- 补入DIP纯运动受损恢复/序列内反馈，以及DIR/WiLoR/GeoHand候选取证前作，收窄首次贡献表述；新增JoHan等2026方法，区分预印本与正式发表。
- 确认WestWorld 256-bin配置、HaWoR非因果及missing-only补全、HMP局部prior与2D fallback、StableHand原论文/增强版差异、PAD-Hand公开demo范围。
- 核验HaWoR W/WA含尺度对齐、StableHand同名指标版本变化；更新引用、评测、进展与入口，不改变用户选定路线和表示，不改写原始档案。没有模型运行、权重/实验数据下载或性能复现。

## 2026-10-13 StableHand输出对照补充

- 按用户明确要求扩充StableHand对照，区分54D/手最终动态状态、flow velocity直接输出、独立beta与四路观测误差/quality。
- 核验固定公开实现中MANO translation不等于解码wrist xyz，登记rest-root偏移和实际字段顺序；保留论文v1与增强版残差表示的边界。
- 将StableHand初始观测质量与本项目最终状态质量区分；更新表示专题、论文报告和状态，无模型实现或性能复现，不将表示建议升级为用户决定。

## 2026-10-13 八库核验与预训练方案收敛

- 按用户截图核验HOT3D、ARCTIC、Re:InterHand、H2O、HOI4D、EgoDex、EgoVerse、Ego-Exo4D，新增[首轮预训练方案](../docs/TWM_PRETRAIN_PLAN.md)，重点数据选择、输入输出、corruption、混合监督与验收。
- 纠正HOT3D full VRS与公开Clips访问混用；Clips train可直下，当前建议HOT3D-Clips＋Re:InterHand先建基线，HOI4D/OakInk2与EgoDex后扩。帧条目累计不当去重净量，split先按原sequence/subject/capture。
- 明确StableHand八库用于质量网络、joint-only不拟合MANO；公开缓存为benchmark评测，不作训练全集。记录H2O62字段、HOI4D15Hz、EgoDex25点与EE4D官方2D/3D数量口径。
- 具体化99D MANO动态状态＋序列β、joint-only可微几何监督、当前恢复主任务＋短预测辅助；编码、混合、窗口和配比为未验证建议，未升级为用户决定。
- 更新入口、方法及专题记忆，保留其他会话的论文精读/StableHand补充和原始档案。无完整数据下载、loader、模型运行或结果。

## 2026-10-13 99D采纳与论文驱动的数据重规划

- 用户明确使用99D，登记D16，替代69D默认候选；D17登记以后plan广泛对照最新论文，写入AGENTS。
- 新增[12项论文数据依据](../docs/TWM_DATA_PAPER_EVIDENCE.md)，补DIP/TaMF/UniHand/HandX/HandFlow/ACE/EgoFound3R/JoHan等实际训练职责，不沿单篇QN八库筛选。
- 将[预训练计划](../docs/TWM_PRETRAIN_PLAN.md)更新v1：建议HOT可用train＋OakInk2核心，Re/Giga拟合扩展、EgoDex规模、DexYCB恢复基线/真实HaWoR数值适配、HOI4D先留出。数据混合仍未被用户定案，删除默认50:50最优配方暗示。
- 发现HOT官方train含论文另留出的公开GT benchmark原recordings；先全源排除、内划val再切窗。Oak沿431/39/157序列split但验PT/CT父录制和subject属性；HO3D仅称TWM未见，HOI4D只GT-valid评测。
- 方法与接口改为99D＋β，21点/mesh从MANO解码；51D切向Corrector残差仍为建议，骨长loss不默认必需，窗口/噪声由pilot验证。原档保留，未下载完整训练数据或运行模型。

## 2026-10-13 首轮选库建议进一步收敛

- 用户再次要求判断TWM预训练最适合哪些数据；明确第一个checkpoint只要求HOT/Oak核心，Re为优先姿态扩展、EgoDex为映射后规模扩展，不把所有候选当开工前置。
- 澄清DexYCB可以纯轨迹预训练；先单独DexYCB-only合成恢复对照与真实HaWoR错误适配，再判断是否混入核心。第三视角不是纯TWM的排除条件。
- 将“能支持第一轮验证”和“已覆盖完整运动分布”分开；来源增量需同预算验证，未新增用户采纳的混合决定，无数据下载或模型运行。

## 2026-10-13 传感器精度主导的v2纠偏

- 用户强调TWM无视觉、优先更准确手姿态、初始glove/MoCap方向有价值；登记D18，限定论文用库只是参考，替代v1 glove支线/Re-EgoDex优先扩展建议。
- 新建[传感器精度专题](11_SENSOR_FIRST_PRETRAIN_2026-10-13.md)，恢复GRAB/ARCTIC高优先MoCap、保留Oak/HOT，补公开UmeTrack自由手势/pinch；HumanTouch/DexWild升主要glove验收，DexCap/OpenGraph做小样本与部分监督。
- 通过官方ModelScope预加载dataset字段复核HumanTouch许可CC BY-NC4.0；固定HF SHA读取十份小型meta，累计28,674,858帧/60Hz约132.75h非QC净量，meta/pose索引episode计数不同须验收。旧“许可待核”已解决。
- 明确source-native姿态与99D模型状态分存、glove几何/部分方向监督、腕/手指质量分开；不因非MANO拒绝原始骨架，不伪造未测twist。GRAB personalized template、UmeTrack转MANO残差纳入验收。
- StableHand v1 QN数值输入但误差标签源于视觉prediction/reference，Enhanced有depth cues；借异构标签/故障任务，不将10M当clean运动池。更新计划、方法、入口及专题，无完整数据下载、模型运行或实验结果。

## 2026-10-13 完整99D输出契约确认

- 用户再次强调训练后的TWM应预测99D，补充D16：每手完整99D主状态输出不随源标签完整性改变，β仍另存。
- 在[预训练计划](../docs/TWM_PRETRAIN_PLAN.md#统一99d输出与标签完整性)明确完整MANO参考、glove几何与局部部分标签三种监督；离线IK拟合不等于新增测量，未知world腕/twist不可补造GT。
- 同步方法、入口、传感器专题和下一步；架构、adapter及loss实现仍未验收。仅文档更新，没有数据处理或训练。

## 2026-10-13 逐库完整适配边界澄清

- 用户追问所列数据是否全部完全适配；澄清这是适配事实核验，不是新的选库决定：完整99D输出或参与部分loss不等于每库单独满足完整腕＋手指任务。
- [预训练计划](../docs/TWM_PRETRAIN_PLAN.md#候选数据的完整适配判定)分开记录完整字段MoCap候选、高价值但需映射/腕链验收的glove、OpenGraph局部补充及其它未验扩查资源；不因非MANO降低glove科学优先级。
- 只读复核官方ARCTIC/Meta字段说明、HumanTouch episode契约与OpenGraph README。没有库通过本项目adapter或数值质量验收，没有数据预处理或模型运行。

## 2026-10-13 八库采集与标注来源精读

- 按用户要求新建[采集来源报告](../docs/DATASET_CAPTURE_AND_ANNOTATION.md)，逐一核硬件、原测量、扫描/标定、模型拟合、人工清理/QC与发布标签层；不改99D或glove优先原则。
- 补HumanTouch官方主页和DexTouch-WM论文MANUS/VIVE证据，保留HF与论文任务版本及逐episode配置未核；精确设备代际仍待验。
- 固定DexWild `5fa34af5115bb6f54320d0071c3b1d4864cd224a`审tracker/camera上游：当前路径未显式组合ZED world，world_abs名不证明世界真值；结论限该路径，不断言全部发布无world。DexCap固定采集代码核每手21节点/方向、四指LEAP与原人体分离，Rokoko同期资料EMF/IMU并存。
- 核OakInk2官方论文/补充的人工marker补齐与两阶段拟合；论文PDF仅为静态文献阅读，临时文本不提交。同步专题、入口与进展，无实际数据处理、模型实现或训练结果。

## 2026-10-13 精度比较的工程判断

- 用户问哪类更准确；补入[来源报告](../docs/DATASET_CAPTURE_AND_ANNOTATION.md#精度判断工程建议与实测排名分开)的分项判断：完整world腕＋手指参考更信校准外部MoCap，局部手指不一概判glove较差，HumanTouch/DexWild仍主要候选。
- 保留无同基准八库排名的边界；Meta native→MANO有直接精度方向证据，GRAB网络误差和DexCap终点漂移不冒充GT标签逐帧误差。状态为建议，未新增用户选库决定，无数据处理/评测。

## 2026-10-13 预训练数据执行归纳v2.1

- 按用户要求基于既有12项论文职责、八库采集与精度核验重新归纳：首轮目标建议OakInk2/HOT3D有效train/独立UmeTrack train/HumanTouch，glove从首轮共同验收。
- DexWild仍主要glove候选，可靠local字段可早接；已核world代码路径问题解决前不做world腕GT。GRAB/ARCTIC有授权优先加入，DexCap小补充，局部/视觉来源按各自角色后续验。
- 在[预训练计划§0](../docs/TWM_PRETRAIN_PLAN.md#0-当前推荐的预训练组合)区分目标池与首checkpoint依赖；不等待所有库，不预设比例，保持99D与部分监督、split/去重及远程实验约定。此为建议，未登记为用户采纳的混合。
- 更新入口、决策边界、数据/状态及传感器专题；无数据下载、转换实现或训练结果。

## 2026-10-13 ds现有GPU Pod操作学习

- 用户说明hongfei GPU忙，指定ssh ds及旧项目运维参考，随后明确授权使用昨天申请的jchen57-debug-mpd5p；登记D19，当前入口替代D14的hongfei，旧checkout保留备用。
- 新建[ds操作记录](../docs/DS_SERVER_OPERATIONS.md)，只读核包、远程运维/非敏感设置、Kubernetes context/用户任务、NFS及Job时限。附件与live配置Debug CPU/Eval GPU不同，模板不等于实时资源。
- 已进入用户Pod，12CPU/128Gi/2×约96GiB Blackwell；默认Torch2.6/cu126两卡tiny CUDA失败，zhou Torch2.12/cu130两卡8元素平方和140通过。记录失败与成功，非模型实验。
- 不提交原包、凭据、内部NFS地址或原始日志；未创建/删除Job、改旧项目、clone ds本项目、安装依赖或训练。更新当前计划中的远程位置，保留历史hongfei记录；home/data10项目路径仍为建议未创建。

## 2026-10-14 TWM直接校正与联合微调讨论

- 用户质疑独立Corrector性能瓶颈，要求评估统一TWM直接状态修正；新建[讨论方案](../docs/TWM_DIRECT_REFINEMENT_PROPOSAL_2026-10-14.md)，推荐先数值恢复和真实误差适配，再检验数值几何取证。
- 对照HMP、DIP、PAD-Hand、HaWoR、StableHand及近期JoHan/Prior-First条件化机制；区分架构可行、精确训练顺序证据、world腕/时间上下文与physics表述边界。
- 记录quality不足以传递新校正方向、TWM输入仍无RGB/features、内部残差不改变99D任务输出、Z/S/H与w/q语义及先验保持消融。
- 更新方法入口、未决架构讨论、状态与记忆；不将本轮提议登记为用户已决定删除C，不改原始档案，无模型实现或远程实验。

## 2026-10-14 TWM物理数学保证与独立验证

- 按用户要求参考WestWorld及其他world model，新增[数学与验证方案](../docs/TWM_PHYSICAL_GUARANTEES_AND_VALIDATION_2026-10-14.md)。核结构学习、Hamiltonian/Lagrangian、开放耗散模型、手部生物力学与硬投影论文的条件及范围。
- 给出SO(3)、内部FK、指定角度范围、anchor全局等变、固定目标下降与条件性反馈误差界；区分数学设计性质与独立GT恢复结果。
- 记录当前99D配置与动力学相空间、真实时间与恢复轮次的区别，能量输入/耗散和未知力的不可辨识；新增OneWorld评分/理想posterior边界，不将learned score当物理证明。
- 更新方法、决策讨论、评测、写作与入口；未采纳额外物理模块，不改变99D或启动远程实验，所有实现性质尚待数值验收。

## 2026-10-14 D23 OakInk2 单库预训练跑起

- 用户要求四库先选一库pretrain、睡前跑起来，数学暂缓；登记D23，保留99D/纯轨迹/当前恢复主任务。
- 新增Oak quaternion→99D准备、轨迹切窗、SO(3)几何、821,516参数TWM、训练/固定验证与Pod后台启动/状态脚本，正式训练代码4cb1da8。Oak官方64train/8val子集已远程准备，204502/18032单手帧；其余三库、完整FK尚未验。
- 远程8组数值检查和3步smoke通过；记录smoke CLI错报版本b04de13→实际8593fcb的远程勘误，正式launcher自动捕获SHA。clean quality捷径在正式启动前修复。
- 09:01UTC在既有Pod/GPU0后台启动50,000步，09:05UTC step7275 alive，best/latest持久保存；step7000 noisy腕49.090→8.657mm、finger15.421→2.259°，clean退化4.551mm/1.642°同报。不是MPJPE、真实HaWoR、闭环或物理结果。
- 新增[运行配置与快照](../docs/TWM_OAKINK2_PILOT_RUN_2026-10-14.md)，同步入口/README/方法/数据/表示/开发/状态记忆；原始日志、数据和权重留远程，不提交Git。

## 2026-10-14 D24 暂停首轮，重审WestWorld架构参考

- 用户要求暂停tiny pilot，并澄清不是完整复刻WestWorld，重点借鉴架构与参数理由，输入输出配合手部99D当前恢复任务。登记D24，替代继续D23 run的安排。
- 09:16UTC核PID/cmdline/output后SIGTERM，进程保存退出；torch.load只读验latest28,213/best27,500可读、SHA4cb1da8。末次step28,000验证与clean退化保留，权重和原始日志不上传。没有新Job、自动续训或新模型训练。
- 复查WestWorld v2 §4/§5.2/附录D及官方固定源码，新增[任务适配与配置依据](../docs/TWM_WESTWORLD_GUIDED_DESIGN_2026-10-14.md)；旧pilot保留作对照，新架构/超参是建议，不将其写成用户逐项采纳或验证性能。
- 更新入口/README/运行记录/方法/决策/状态/开发记忆，清除仍在训练和追50k的当前安排。

## 2026-10-14 D25 结构化Mamba重做并启动

- 用户明确采纳刚才结构参考方向并要求跑起后再结束；旧flat checkpoint保留，新模型从头训练、输出99D当前状态。登记D25，替代D24不另启训练的安排。
- 新增MANO树embedding与同帧attention/FFN→逐关节官方Mamba，6/256/4＋SSM64/4/2，6330892参数；preserve特殊初始化及no-weight-decay、padding/dt隔离。359b88e模型＋378deda环境复现脚本已同步，配置lr2e-4，其余数据/噪声沿旧pilot，不称单因素消融。
- 建完全隔离Torch2.10/cu130环境、官方wheel核SHA，pip check及GPU fused FP32/BF16通过；完整主干12项数值检查（因果/压缩误差0.0）与真实Oak3步smoke通过。包装器Pod缺git的记录问题已补真实SHA，SSH瞬断重试成功。
- 09:47UTC在既有Pod/GPU0后台启动PID1276、独立正式run目录；09:50UTC核step650持续运行、loss有限、500步best/latest约76MB已存，约4.48steps/s。早期旋转收益与腕未改善/clean退化同报；旧run不恢复。新增[模型/环境/检查/运行记录](../docs/TWM_STRUCTURED_MAMBA_RUN_2026-10-14.md)，数据/环境/权重/原始日志仅data10。

## 2026-10-14 D25五万步完成核验

- 用户要求查看运行情况，远程只读确认结构化Mamba在纽约08:57正常complete50000，并核latest50k/best49.5k可读、模型浮点值finite，100次验证/2000training日志无非有限/Traceback/OOM标记。没有追加训练或重启。
- 末步合成noisy腕49.090→4.594mm、finger15.421→1.254°；missing腕37.637→4.411mm；clean退化1.276mm/0.440°/0.500°同报。腕20k后趋稳；仅固定相关val窗口/参考历史恢复，不称真实HaWoR或架构优越。
- 新增[小型final JSON](../reports/twm_structured_mamba_20261014_final.json)，更新运行记录/入口/README/数据/状态/开发；best49.5k结果与末步区分，原completion_audit与权重只在data10。

## 2026-10-14 D26四库联合授权、合法MANO与准备验收

- 用户要求将已下载OakInk2/HOT3D/UmeTrack/DexYCB统一预训练，登记D26；替代单库作为当前执行范围，保留D25已完成50k和D24旧flat暂停记录，不覆盖原始档案。D18 glove/MoCap精度优先、99D＋β另存与D22 ARCTIC暂缓不变。
- 用户自行官网注册、登录并确认MANO非商业研究许可后授权下载；官方v1.2 ZIP本机/远程SHA一致。原始模型与license保留，派生NumPy/SciPy转换数值/dtype与无Chumpy加载检查通过。资产/数据不进Git，[有限adapter摘要](../reports/twm_four_source_adapters_20261014.json)仅记录可审计小型结果。
- Meta官方decoder18帧mesh/joints/full pose等价、99D FK最大差1.19e-7 m；Dex左右各6帧及bounded train/val检查通过（21点最大约1.50e-7 m）。这些是源实现、PCA/mean/腕中心/单位与编码一致性，不是独立native/MoCap精度；真实缺标签未补作完整GT。
- lazy loader、source-balanced sampler、分库固定验证、完整GT fail-closed等9项远程检查通过。13ba665增加resume config/manifest一致性、未知sampling拒绝和leaf hash核验；临时远程测试hash正确pass、修改leaf后fail。FD probe 4→2,564→4，Pod soft/hard限制均1,048,576；范围是当时双cache各256序列×5 maps，不替代新配置吞吐测试。
- 真实四库3步smoke源码2913f28，loss/梯度有限，样本Dex3/HOT6/Oak9/Ume6；receipt位于data10项目根`audits/four_source_smoke_20261014/receipt.json`，run为`runs/twm_four_sources_smoke_20261014`。仅为工程smoke；随后正式训练以2b40aa9单独启动，见下方启动验收条目。
- Oak改8 workers另建`prepared/oakink2_99d_full_v2`，旧v1部分保留；full_v2已complete431 train/39 val录制、940手序列，状态1,414,552/113,382。每序列32帧共30,080帧FK往返最大joint1.49e-7 m/mesh1.88e-7 m，同一MANO实现对照，非独立source验证。之后Dex full_v1与Meta HOT/Ume full_v3均complete，完整统计/collection与正式启动验收见下项；准备中间版本与部分产物保留作历史，不再当当前状态。
- 记录Meta NPY保存性能瓶颈与lossless NPZ局部约9.2×结果，明确非端到端吞吐。实际β审计zero＋7组源β/两侧两pose显示左手同腕joint差3.74–19.81mm、mesh差5.60–24.28mm，right/zeroβ与full-pose旋转参数差0；root工程选择本轮`condition_on_beta=false`，输入mask关闭，源β及真实validity仍存。未统一shape语义，未来开启需适配/消融，不升级为用户永久架构决定。
- 该准备阶段只更新MEMORY及决定、进展、变更、开发五份记忆，未由记忆任务重启转换或新增模型实验；后续正式启动与统计已据实际记录补入下项。

## 2026-10-14 当前Mamba物理与数学证明复核

- 用户再次要求仔细证明当前模型的物理/数学性质；在源码2b40aa9和官方Mamba v2.3.2/scan上复核，新增[现有物理文档§8–12](../docs/TWM_PHYSICAL_GUARANTEES_AND_VALIDATION_2026-10-14.md#8-2026-10-14当前mamba-twm的证明范围复核)。
- 明确已实现SO(3)更新、固定源β/decoder的内部16点FK骨长、外部anchor wrapper等变与旋转→内部joint误差界；逐样本等变需合法旋转及同步旋转噪声，有dropout需同随机量。
- 推导任意损失理想条件风险不增、局部MSE信息收益/网络误差分解、高斯隐藏状态posterior协方差减少，区分标准理论与当前实际网络结果。
- 核实际drive ΔBu，避免套完整ZOH凸组合门控；pre-LN＋冻结有限参数的SSM内部有界不代表完整反馈收缩。保存常数head bias漂移反例、K30小增益条件及不可观测误差下界。
- 更新方法、决定和进展记忆；未修改训练源码/config、启动模型实验、采纳额外物理模块或新增性能数字。

## 2026-10-14 D26全量准备完成与正式训练启动验收

- Oak full_v2、HOT/Ume full_v3及Dex full_v1全部complete；共8,418手序列，train/val状态2,788,135/271,100。有效窗口636,610/61,370；Dex两条train序列无足够连续窗口，loader实际7,608 train/808 val序列，与manifest总数分别记录。
- HOT于17:45:27 UTC最后完成2455 TAR/4910手序列，Ume1020 TAR/1728手序列也已完成；Meta全部TAR严格重解码、全部状态重算。旧NPZ只有等价验证后保留，不能将文件存在当复用验收；Ume native-invalid包含native-absent，各9,725不可相加。
- 一次性coordinator验四个complete manifest与collection后，于17:51:39 UTC以2b40aa9790c1d18651d6ac810d23a72d3c7a5922在既有Pod/GPU0/PID2599从头启动。正式输出为data10项目根`runs/twm_four_sources_structured_mamba_20261014`。四leaf SHA与collection全部匹配，完整路径/配置/来源见[正式启动摘要](../reports/twm_four_source_20261014_launch.json)及[运行记录](../docs/TWM_FOUR_SOURCE_RUN_2026-10-14.md)。
- 首条训练日志17:54:37 UTC step25，loss/梯度有限，四源Dex389/HOT421/Oak389/Ume401；首验证保存step1000。19:02:32 UTC验收快照step17,250；root后续19:05:47 UTC核step18,075、loss0.0269482/梯度1.10944、四源累计289582/289120/288354/289744。它是运行中快照，不是50k完成或正式最终效果。
- 19:02:30 UTC一次CPU只读加载latest17,000/best16,000，各163个浮点模型张量全finite，code/from-scratch/β关闭/manifest一致。没有运行网络、额外训练或重启；检查只经tool返回，未单独保存远程receipt，不与后续更晚步数混用。
- D26四库范围是用户授权；库间等概率与β conditioning关闭是本轮工程临时选择，源β/真实validity和99D不变。β输入实际GPU forward/backward验证通过，模型shape projection输入4×11全零、梯度有限。D25已完成50k仍单独保留，D18 glove/MoCap优先与D22 ARCTIC暂缓有效。
- 更新限定五份记忆的过时准备状态；保留历史attempt、源实现一致性非独立GT精度、synthetic恢复非真实HaWoR收益等边界，并保留同日数学文档复核内容。
- root随后保存[启动JSON](../reports/twm_four_source_20261014_launch.json)和完整运行记录：19:07:32 UTC step18,550、742条training事件finite、四库实际采样；step18,000合成noisy等库均值腕49.542→6.155mm、finger15.481→2.153°，同时记录clean退化2.925mm/1.388°。补充审计后续全量事实，不改模型、配置或在跑进程。


## 2026-10-14 D27双GPU、大batch迁移与运行

- 用户要求“两张卡一起跑、batch调高”，登记D27，仅替代D26单卡执行配置；四库、当前恢复、99D、结构化Mamba、β另存/conditioning暂关、源GT/split与研究方向保持。D26旧单卡已SIGTERM保存停止于26,346步，原run及19:07历史验证保留。
- GPU1实际batch128/256/512/768/1024全部finite；512/卡峰值allocated34.16GiB/reserved34.79GiB、240.7样本/s，768/1024分别237.3/229.2样本/s。工程选择512/卡、global1024，不当最优batch或正式DDP吞吐。验证仍64×16上限，Dex实际394窗；LR及50,000优化步不变，剩余23,654×1024=24,221,696样本抽取，是旧batch64剩余预算16倍。
- 实际DDP smoke26346→26446完成100更新；真实resume后仅rank1 SIGTERM，共同保存stopped26498。CPU检查163个model浮点张量及499个Adam递归浮点项（含超参，非499个张量）finite，两rank完整model state keys/dtype/shape/bytes SHA一致；验证了双卡训练、同步、保存及恢复，不等于独立几何效果。
- 正式源是旧单卡latest26346，SHA256 `09a10b8f800cac5e8336da3a14cf69cfce9228ff8674be84182a84102e1bf27f`，不是smoke26498。代码`10aa98327eeaa3ff0afb49eef1e6ff73879c7d22`于19:47:53 UTC启动data10项目根`runs/twm_four_sources_structured_mamba_ddp_20261014`，launcher2963、ranks2968/2969、GPU0/1。迁移保留模型/AdamW/绝对步数，重建rank RNG与采样，不宣称逐样本精确续跑。
- 19:52:33 UTC正式actualstep26450（新增104更新），四源累计实际106,496=104×1024样本，各rank53,248且四源均正；loss0.0208302/梯度0.15867，早期435.9样本/s，不当稳态比较。updated checkpoint26350已CPU读取，163个model/489个AdamW浮点张量finite、Adam step26350，CPU重算完整模型hash与两rank/replica audit一致。19:52:47 UTC两卡瞬时util100%、显存38,172/36,686MiB，PID2968/2969；不是稳态或峰值结论。保存checkpoint步数与后续running快照分开，尚未完成50k。[双卡运行记录](../docs/TWM_DUAL_GPU_RUN_2026-10-14.md)、[验收摘要](../reports/twm_dual_gpu_20261014.json)。
- 仅更新MEMORY、决策、进展、开发远程与CHANGELOG五份记忆，将active run统一为D27双卡；未修改数学文档、模型任务或数据协议。

## 2026-10-14 D28暂停旧dense TWM与冻结效果验证

- 用户要求暂停当前运行、保留checkpoint并先验证效果；通过STOP共同保存在32,783步停止，rank2968/2969退出，不自动resume。仅针对D27旧dense 6.33M结构化Mamba，未修改其他并行模型或用户Job。
- latest32,783/best28,000模型与AdamW finite，原文件及评测快照SHA一致；00:12:06 UTC复核STOP/暂停状态、4个只读评测进程结束及GPU无活跃计算，原产物保留。
- 新增冻结评测c8e9a8d、控制/绘图9b20493，全部Pod执行；固定3,466窗、全量61,370窗/模式，best原fixed选定指标复现差0、控制normal与fixed逐库一致，未读test或改split。
- latest全量noisy腕48.703→5.435mm、finger15.444→1.808°，hold8.961mm/2.839°；clean腕0→0.454mm，四库宏平均6.89%窗增加>1mm。best full noisy腕5.389mm略好，finger/clean较差，两者有取舍。
- 固定干预normal5.476mm、当前全缺5.451mm、当前GT＋带噪历史5.317mm；模型能改变输出但平均GT误差对当前可靠观测利用不足，主要历史先验。记录训练噪声/质量提示设计的原因假设，不当已验证修复、真实HaWoR收益或物理失败定理。
- 保存[完整报告](../docs/TWM_PAUSED_EVAL_2026-10-14.md)、[小型JSON](../reports/twm_paused_eval_20261014.json)与[实测图](../reports/figures/twm_paused_20261014.png)，更新当前记忆，未恢复训练。初次包装器缺评测父目录导致mkdir前置失败，修复后首次启动，无评测数据丢失。

## 2026-10-14 HaWoR原论文报告误差核验

- 按用户要求回查原论文Table1/3/4(b)、补充§8.3与官方对齐函数，补入HOT3D主表PA/W/WA=4.79/33.20/11.27mm、不可见infiller6.22/66.25/37.22mm、DexYCB整体PA4.76mm及遮挡分组。更新[对比协议](../docs/HAWOR_DATA_AND_COMPARISON_PROTOCOL.md)和数据评测记忆。
- 区分joint位置/腕位置/旋转及对齐口径，不能把D28合成腕误差5.435mm与HaWoR直接排名；无新训练/推理、没有恢复暂停run。

## 2026-10-15 UTC D29当前恢复＋MoE＋长轨迹分布与在线训练

- 用户明确要求当前受损状态恢复同时预测更长未来轨迹、加入MoE、由实现者选时间编码及输出不确定性，并修改/跑起模型、配置W&B；覆盖早期短预测和默认无MoE限制，旧D28 run保持暂停。
- 新13,530,032参数模型：6层d256结构化Mamba＋每层4个dense MLP experts，物理秒Fourier；当前99D＋4整段mode×60步约2秒，Gaussian平移/归一化SO(3) isotropic matrix Fisher，未来latent Mamba速度解码与逐步合法旋转积分。保留raw/candidate/history、四库GT/split、β另存与本轮conditioning关闭。
- 新loader保留原636610/61370窗口与最少3未来帧，tail/gap mask；Dex无2秒标签，完整60步/物理秒覆盖分开保存。新增分库/分时域hold与常速度、moving/low-motion、NLL/coverage与within/total uncertainty指标；不以长预测、多样性或合法旋转宣称真实动力学/视觉收益。
- 远程概率积分/导数、因果/padding、FP32/BF16全部321个parameter梯度、完整objective无效标签不变性通过；真实batch32/64/128/256/384三次AdamW更新全finite，选256/卡global512。双卡smoke3步＋resume至5步、CPU checkpoint/hash检查通过。
- 正式4798b62于00:19:13 UTC从头启动独立run `twm_four_sources_moe_long_horizon_20261015`，launcher4034/ranks4039/4040；00:22:37 UTC实际120步/61440样本，W&B server step110可读。已检查正式checkpoint10，322个model与963个AdamW浮点张量finite，CPU hash与两rank一致；不把验收步数与更晚运行快照混用。
- W&B0.30.0实际在线；默认团队yc-node及个人空间403，定位black-ant后新建独立handloop-twm成功。用户新key验证后仅存远程600私有文件、child环境传入，不改旧netrc、不进入Git/config/receipt。正式run [943f9c0c](https://wandb.ai/black-ant/handloop-twm/runs/943f9c0c)；每100步保存/500步val/10步log，16:25UTC共同保存早于Job到期，不自动重建Job。
- 保存[设计、概率推导及运行记录](../docs/TWM_MOE_LONG_HORIZON_RUN_2026-10-15.md)与[小型启动JSON](../reports/twm_moe_long_horizon_20261015.json)，更新方法/决策/进展/开发记忆与AGENTS。时域/模式/权重为工程起点，无新模型最终精度或真实视觉闭环结果；D28当前可靠观测利用不足仍须新实验复查。

## 2026-10-14纽约 D30 HaWoR错误分布与corruption校准目标

- 用户强调纠错应模拟现有HaWoR的错误；记录为分布匹配目标，具体噪声幅度/比例与训练执行调整未定。核实D29调用旧sample_batch，当前与历史噪声强弱悬殊、quality独立，不能假定新架构已解决D28历史主导。
- 对照PAD-Hand真实WiLoR配对、DIP通用与特定前端训练、StableHand质量增广、HaWoR连续mask，补[直接恢复讨论§8](../docs/TWM_DIRECT_REFINEMENT_PROPOSAL_2026-10-14.md#8-2026-10-14corruption应贴近hawor实际误差)及决策/数据/进展记忆。明确每轴σ≠平均欧氏误差≠PA/W-MPJPE、排除最终评测集拟合、当前准/历史错、部件与时间相关性和oracle质量边界。无新模型实验或run变更。

## 2026-10-15 UTC D31暂停当前训练与混合噪声草案

- 用户要求暂停正在跑的模型，架构在另一聊天修改；澄清可混合高质量纯轨迹与受损输入，轻中误差为主、严重少量，不必先完成HaWoR误差标定。
- 已通过共同STOP在00:36:36UTC保存停止814步，rank4039/4040退出；原latest814/best500与独立快照SHA一致，322个model/963个AdamW浮点张量finite，CPU完整模型hash与两rank一致。00:37:42UTC核验，随后GPU无计算进程；原run、旧D28权重、用户Job保留，不自动resume。[暂停摘要](../reports/twm_moe_pause_20261015.json)。
- 两个只读agent独立审查混合比例、单位、质量提示与验收。建议20%全净/20%当前准历史错/35%轻中/20%缺失/5%强异常；范数有界位移与SO(3)总角、相关方向控制、默认quality unknown，GT始终未受损。具体数值为待验证工程假设，不称HaWoR标定结果。
- 保存[§9设置与边界](../docs/TWM_DIRECT_REFINEMENT_PROPOSAL_2026-10-14.md#9-d31先暂停混合干净轨迹与适量受损条件)和[版本化配置草案](../config/twm_corruption_mixed_v2_proposal.json)，标明未接trainer；未改模型架构、旧sampler或启动新实验。更新当前执行/开发/数据/决策记忆，D29启动快照转为历史。

## 2026-10-15 UTC D32混合噪声实施并正式运行

- 用户要求按D31当前设置跑起，授权接入与新训练；保留原13.53M架构、四库GT/split/β策略，工程选择独立新run从头，旧D29停814/旧D28停32783均保留STOP。
- 0d83a54实现versioned bounded mixed sampler、active profile及显式配置、12条件验证、加权best与实际类别计数；没有改旧sample_batch。位移为实际范数、旋转为总角，quality unknown，GT/future不受损。原profile提案保持历史。
- 远程4096窗语义检查通过，双卡global512三更新/12条件初始验证与CPU checkpoint审计通过；模型322/AdamW963张量finite、两rank/CPU完整hash一致。未为未改变的模型架构重复大规模概率/因果测试。
- 00:54:45UTC正式启动新run twm_four_sources_moe_mixed_v2_20261015，launcher4496/ranks4501/4502，2卡每卡256。00:57:06UTC实际50步/25600样本、四库与五类均参与，loss/grad finite；正式审计checkpoint10 finite/两rank一致，不与运行步数混同。
- W&B cd487764服务端state=running、optimizer_step40与train/loss/grad实际可读；私有key不打印/入Git。计划50k、16:25UTC共同保存早于Job时限，不自动新建Job。[协议及边界](../docs/TWM_MOE_LONG_HORIZON_RUN_2026-10-15.md#d32混合有界噪声训练已启动)、[小型摘要](../reports/twm_mixed_v2_20261015_launch.json)。尚无训练后最终效果或真实视觉纠错结果。

## 2026-10-14（纽约）：TWM之后的视觉baseline接入与联合训练讨论

- 用户询问如何接入HaWoR及联合训练；核实际D32模型接口、现行/直接恢复两方案与官方HaWoR、DIP、HMP、StableHand、PAD-Hand、DIR和最新JoHan主来源。新增[直接恢复讨论§10](../docs/TWM_DIRECT_REFINEMENT_PROPOSAL_2026-10-14.md)的可实施协议，不把建议当用户已定或代码事实。
- 明确pre-infiller接入仍保留cam-to-world，先验99D几何/投影、固定前端零适配与公平evaluator，再做真实预测适配、视觉evidence/w/q/校正联合、自身历史rollout及可选HaWoR部分解冻；记录no_grad/CPU导出与缓存的梯度边界。
- 保留长future辅助当前的训练/跨帧两条路径、整段mode与future/GT不作当前实测；当前TWM已读Z，不能再用同Z似然当独立Bayes相乘。σ/κ、原观测w、最终q分别监督/校准，HaWoR非因果上下文与GT crop/对齐条件单列。
- 更新方法/进展/决定说明；视觉adapter和联合trainer仍未实现，未改模型源码、D32运行、数据split或启动新实验，独立Corrector是否合并仍待用户决定。

## 2026-10-14（纽约）：D33视觉侧反馈插件与精简训练目标

- 用户要求让预训练HaWoR视觉侧读取TWM校正信息，认真处理时序依赖，并明确反对多项手工带权loss。新增[讨论§11](../docs/TWM_DIRECT_REFINEMENT_PROPOSAL_2026-10-14.md#11-d33预训练hawor读出侧反馈插件与单一视觉训练目标)，§10多loss/额外模块清单标记为早一轮建议。
- 复核官方权重、ViT/IAM/decoder/PAM路径与inference梯度边界；推荐保留16帧双向缓存，在PAM后新视觉分支逐帧读取已完成历史，保留raw/candidate并交纯数值TWM最终读出；零输出初始化、下一帧反馈、固定crop和非因果边界写明。
- 推荐视觉阶段冻结两主干、仅训练插件，单一最终current state NLL；候选通道输入梯度不足才微调少量current读出。保留长future能力并独立复测，不以单个公式掩盖current/future隐式权衡或用σ/κ冒充质量q。
- 更新MEMORY与方法/决策/进展；D33用户方向和具体工程建议分开。没有改源码、运行配置、D32训练或启动视觉实验；无新增性能结果。

## 2026-10-15 UTC D34收敛复核与每卡512从头10k

- 用户要求检查当前约3k步的训练情况，将batch翻倍为每卡512/global1024，并独立从头训练10,000步；本次只替代D32执行配置，D33视觉侧反馈方向及其尚未实现的边界完整保留，没有启动视觉训练。
- 旧D32于02:02:24.032654UTC共同保存停止3,110步，latest与best均保留并另存快照；原件/快照SHA相等，322个model与963个AdamW浮点张量有限，CPU完整模型hash与两rank一致。旧run不自动resume，不覆盖checkpoint。
- 固定合成验证1,000→3,000步的实际20/20/35/20/5加权restoration score为0.022254976823→0.012809634894，noisy手指误差3.21507418→1.77306231°；noisy腕输出7.28698158→7.21049546mm，仍高于candidate输入7.15831929mm，不能据此称全面收敛。曲线使用当前checkpoint实测score，不使用best-so-far冒充当前值；step0零初始化copy-input单列，不当训练收益。指标为四库等权均值、合成受损与源拟合MANO参考，不是实际HaWoR纠错或独立sensor精度。
- 单卡真实512样本三次完整更新finite，峰值allocated63,481,519,104bytes（59.12GiB），仅为探针测量，不当正式DDP峰值或稳态吞吐；完整mask objective invariance检查通过。
- 新配置[每卡512/10k](../config/twm_four_sources_moe_mixed_v2_b512_10k.json)，代码7d53fa4a1b14f31e08f41ba545556e59adadd31f；模型/GT/split/mixed_v2不变，LR2e-4和warmup500保留，cosine终点改为10k。目标10,240,000次样本抽取，不与旧global512的相同步数作为公平样本预算比较。新run `twm_four_sources_moe_mixed_v2_b512_10k_20261015` 于02:09:50.156267UTC由launcher4672发起，从头初始化；rank4677/4678在GPU0/1训练，10k仍是目标。
- 02:12:13.931674UTC正式运行验收为step20、累计20,480样本，GPU0/1峰值allocated分别63.9163/63.9073GB；单独审计的是checkpoint10，其322个model/963个AdamW浮点张量有限、CPU完整模型hash与两rank一致，不能把step10权重审计冒充step20权重检查。[W&B12cc0781](https://wandb.ai/black-ant/handloop-twm/runs/12cc0781)服务端state=running，step20的loss与grad实际可读；正式启动验收已通过，未完成10k或最终效果验证。
- 运行与后续实测见[训练记录D34](../docs/TWM_MOE_LONG_HORIZON_RUN_2026-10-15.md#d34每卡512从头训练10000步)与[小型摘要](../reports/twm_batch512_10k_20261015.json)。同时修正决策记忆中“当前D27双卡运行”的过时表述，明确其已按D28停止32,783步。

## 2026-10-15 UTC D35视觉反馈完整首版实现与验收

- 用户明确要求搭建D33方案；实现官方HaWoR frozen bridge/可微MANO99D codec、2.458M视觉读出插件、纯数值TWM单步恢复/跨块history、单一current NLL trainer、严格cache/checkpoint/resume、RGB track推理与60步分布导出。原多轮Corrector/w/q不是首版代码事实，D34 pure-motion run保持原样。
- 官方源码10blob/权重严格load与前向parity、左右MANO几何、16项反馈合同、raw/GT NaN指标边界通过；cache为127帧Dex train/val独立录制、right单camera、GT crop/nominal30Hz，14tensor/record SHA通过。不将GT-crop固定camera验收写成任意RGB自动检测/SLAM成功。
- 代码4f5b4af/bacd221/1f842cf；TWM使用已停D32 step3110副本，A参数2,458,118、W13,530,032。32帧clip先3步保存再strictresume到20，395个model/216个AdamW浮点张量finite、72optimizer状态step20、W hash不变；W&B[e00b708e](https://wandb.ai/black-ant/handloop-twm/runs/e00b708e)服务端finished/step20实际可读。真实16RGB输出当前[1,16,99]与未来[1,16,4,60,99]、world covariance正定/全finite。
- val64帧：旁路→正常反馈NLL4558.707→1497.547、腕95.029→65.048mm，但root4.936→5.856°、finger6.779→8.493°退化，xyz90%coverage均0。记录为能训练的框架smoke，未证明整体精度/不确定性校准/future性能；空raw暖启动虽finite仍可能很不准。保留依赖失败记录，不挑正指标宣传。
- 新[框架文档](../docs/VISUAL_FEEDBACK_FRAMEWORK_2026-10-15.md)与[完整摘要](../reports/visual_feedback_framework_20261015.json)，更新README/MEMORY/方法/决策/数据/状态/远程约定；权重/缓存/原日志/预测均只留远程，小型结果入Git。

## 2026-10-15 TWM真实误差微调与先验保持讨论

- 用户要求思考局部/全量微调；源码复核当前readout仅2,570参数、主干共享、future以当前恢复为积分起点。记录约2.10M局部对照与完整13.53M范围的静态预算，未跑新数值实验。
- 核DIP/PAD-Hand真实预测输入、WHAM双阶段全文及保留AMASS/差分LR源码、LP-FT适用边界。建议插件暖身后低LR全量适配与motion恢复/长预测回放，明确采样比仍有取舍、冻结future头与current-only回放不足以保护长预测。
- 补[现有框架文档](../docs/VISUAL_FEEDBACK_FRAMEWORK_2026-10-15.md#2026-10-15twm真实误差适配与先验保持讨论建议未实施)及入口/状态/未决事项；没有新用户解冻决定、模型/训练器改动、权重或远程动作，D35仍仅127帧20步smoke。

## 2026-10-15 D34纯轨迹训练4k效果检查

- 用户要求查看当前训练效果；只读run status、metadata和既有固定validation，未暂停/改训或启动额外模型前向。7d53fa4原双卡每卡512 run快照4,180/10k；最新完整验证4k、每源256窗。D35视觉smoke与本次结果分开。
- ordinary root/finger降低40.7%/51.6%，但腕7.158→7.180mm未改善；clean引入0.688mm且14.26%窗口>1mm退化，reanchor为0.571mm/4.39%。同轮3k→4k weighted score降4.03%，不把best-so-far或负NLL当收敛。1s预测腕103.568<hold120.242mm，2s150.508>hold148.162mm；90%当前xyz覆盖82.03%，不称校准。
- 新增只读[摘要脚本](../scripts/review_probabilistic_twm_run.py)，核对宏平均、保存各时域有效数与原validation字节SHA；远程复用已有绘图脚本并审图。配置/source/data SHA、12模式/分库及曲线在[小型摘要](../reports/twm_b512_progress_20261015_0521.json)，[效果与限制](../docs/TWM_B512_PROGRESS_2026-10-15.md)。更新当前入口、状态及数据记忆；没有新设计决定或真实视觉/独立传感器效果结论。

## 2026-10-15 D34腕位置缺少校正的源码与数值诊断

- 用户要求分析原因；核输入米单位直接进入root token、loss已有0.05m缩放、残差加法/梯度连通、future从current均值积分。排除把3/99维简单均分或位置head断开当解释；未修改模型/训练执行，归一化和current-only仅建议。
- 冻结checkpoint4400、原7d53fa4源码hash/两rank CPU模型hash匹配，四库各8窗；±10mm输入偏移只触发约0.005–0.034mm平均反向位置残差，BF16/FP32相近。current几何head梯度0.411，weighted future geometry0.459；future NLL梯度0.232/cos−0.204，仅为小样本牵制迹象，不当因果证明。参数/buffer前后完全相等、无optimizer，GPU1额外峰值3.877GB；正式训练05:37UTC仍4450步。
- 保存[完整解释](../docs/TWM_WRIST_DIAGNOSIS_2026-10-15.md)、[诊断摘要](../reports/twm_wrist_diagnosis_20261015.json)及[脚本](../scripts/diagnose_twm_wrist.py)。区分input尺度/identity/multitask待验假设、common-bias条件性不可辨识、32窗探索和4k固定1024窗效果；记录所有固定融合比例，不选择有利样本冒充全量改善。更新入口、状态、数据与远程记忆，无新用户设计决定。

## 后续记录要求

每次记录日期、实质变化、依据和相关产物。实验失败和决定撤回同样记录；不要把计划完成、代码完成和实验验证完成混为一个状态。
