# 手部重建与运动恢复论文如何报告精度

核查日期：纽约2026-10-16。用户在完整DexYCB测试后询问PA的作用，并要求查看其他论文的实际报告方式。本页依据官方全文、实验表及补充材料；包含2024–2025代表作与2026公开预印本，非穷尽排名。没有新增实验、改指标或按结果重选checkpoint。

## 核验结论

单图局部手形重建的核心精度常用PA-MPJPE／PA-MPVPE，搭配mesh F-score或AUC；时序恢复同时使用PA、非PA／腕对齐和加速度误差；世界空间方法把局部手形、整体位置／轨迹和时序精度一起报告。**不能因PA-AUC百分点增益小就删除PA，也不能仅用PA验证完整99D或绝对空间恢复。**

此前助手建议“PA为辅助”是针对本项目完整状态定位的解释，不代表主流论文将PA视为次要。经本轮更广泛对照，建议HandLoop以局部几何与整体定位并列主结果，参数角误差作分解诊断；这是助手建议，尚非用户新决定。

## 官方论文实际指标

| 论文与已核位置 | 评测集合 | 实际报告的精度 | 对齐／解释要点 |
|---|---|---|---|
| [HaMeR，CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Pavlakos_Reconstructing_Hands_in_3D_with_Transformers_CVPR_2024_paper.pdf)，§5.1 Tables1–2 | FreiHAND、HO3Dv2 | PA-MPJPE、PA-MPVPE，F@5mm／15mm；HO3D另有AUC_J/V | 单图3D主表突出PA；未单列绝对腕位置或root旋转。FreiHAND本文为6.0／5.7mm、F0.785／0.990，不能与不同输入/split直接排位。 |
| [WiLoR，CVPR2025](https://arxiv.org/html/2409.12259v2)，§5.2 Tables3–4 | FreiHAND、HO3Dv2 | PA-MPJPE、PA-MPVPE、F@5mm／15mm、AUC_J/V | 与HaMeR同类局部重建报告；§5.3 Table6另报MPFJE／MPFVE、Jitter、RTE。该段RTE为腕跨帧位移描述，不能自动等同HaWoR相对GT轨迹误差。 |
| [Hamba，NeurIPS2024](https://arxiv.org/html/2407.09646v2)，§4 Tables1–3／附录A.2 | FreiHAND、HO3Dv2/v3 | PA-MPJPE、PA-MPVPE、F@5mm／15mm，HO3D AUC_J/V | 明确PA，区分有无TTA；5/15是毫米距离阈值，不是百分比。 |
| [HMP，WACV2024](https://arxiv.org/html/2312.16737v1)，§4.1 Tables3–4 | HO3Dv3、DexYCB s0 | PA-MPJPE、RA-MPJPE、RA-ACC | RA去掉根平移；不是绝对camera/world位置。角／时序先验收益用两类几何与运动误差展示。HO3D采用有GT训练部分，不是hidden-test榜；原文加速度单位写mm/s²。 |
| [DIP，ICCV2025](https://arxiv.org/html/2508.01835v1)，§4.1 Tables1–2 | DexYCB、零样本HO3Dv2 | Dex：MJE、P-MJE、ACCL、KIN、STA；HO3D：P-MJE、P-MVE、F@5/15 | MJE/P-MJE为PA前／后位置误差；ACCL明确是与GT的加速度差，mm/frame²。KIN／STA为论文定义约束违反量（°），不是直接GT finger旋转误差。非PA列的root预处理本轮未从评测源码核实，不能冒称绝对空间误差。 |
| [PoseBERT，TPAMI作者全文](https://arxiv.org/html/2208.10211v2)，§IV-B TableVII／IX | DexYCB s0 | 恢复：MPJPE、PA-MPJPE、Accel；未来5/10/15/30帧分别报MPJPE与PA-MPJPE | 手部21joint；Accel为预测与GT加速度差。非PA列仍需核root约定；未以root/finger旋转角当主表。 |
| [HaWoR，CVPR2025](https://arxiv.org/html/2501.02973v1)，Tables1／3、补§8.3 | DexYCB；HOT3D | Dex：PA-MPJPE、AUC；HOT：PA、W、WA-MPJPE、RTE、Accel | Dex表头MPJPE实际PA。HOT的W开头帧对齐、WA整段对齐（100帧块），不等于完全无对齐／仅腕对齐。RTE为轨迹相对误差%，Accel m/s²。 |
| [StableHand，2026-05作者v1](https://arxiv.org/html/2605.18553v1)，Tables1–2、AppendixG | HOT3D、ARCTIC | PA-MPJPE、W-MPJPE、WA-MPJPE、AccEr；ARCTIC另有MRRPE | PA逐帧；W首帧单一Sim(3)；WA整段单一Sim(3)。AccEr为30fps预测与GT加速度差，MRRPE为双腕相对位移向量误差。附录B及Tables4/5采用16-joint MANO约定，与本项目21joint不能直接拼数。Enhanced仓库版另属版本，未混入论文v1。 |
| [Hand3R，2026-02预印本v1](https://arxiv.org/html/2602.03200v1)，§IV-B/C TablesI–II | DexYCB、HOI4D | Dex：PA-MPJPE／AUC及遮挡分组；HOI4D：C-MPJPE、W-MPJPE、WA-MPJPE | C为绝对camera joint误差；W开头帧、WA整段对齐，分别报告30／100帧短长视频。局部手形与空间轨迹是独立实验维度。 |
| [ScaleHP，2026-06预印本v1](https://arxiv.org/html/2606.25619v1)，§4.2–4.3 Tables1–2 | FreiHAND、DexYCB、HO3Dv3 | CS-MPJPE、R-MPJPE、P-MPJPE | CS完全无对齐、R只腕平移、P为PA，正对应本项目三个alignment层次。论文明确绝对metric任务以CS为核心，同时保留对齐指标。Table1的FreiHAND-only checkpoint与Table3数据集内训练不同；136.3和30.0的Dex CS值不能混称同一设定。 |
| [InfiniHand，2026-09预印本v1](https://arxiv.org/html/2609.35743v1)，§4.1 Tables1–2、AppendixA | ARCTIC、HOT3D、EgoDex；HOI4D仅camera | camera：MP-p、PA-p、EPE-p、GO-p、CT-p及检测指标；world：PA／W／WA-MPJPE | MP腕相对，GO根朝向，CT腕位置，-p处罚漏检；本篇W完全无附加对齐，WA整场景双手共享Sim(3)，与HaWoR/StableHand的W定义不同。 |

补充：[Ego-Exo4D官方EgoHandPose协议](https://docs.ego-exo4d-data.org/benchmarks/ego_pose/ego_hand_pose/)同样并列camera-relative MPJPE和腕相对手的PA-MPJPE（mm），支持把空间定位与局部姿态分开报告；这是benchmark协议，不单列为新算法论文。

## 从实际主表能推出什么

1. **PA是核心几何指标之一。** HaMeR／WiLoR／Hamba的标准3D表以其为主，HMP／DIP／PoseBERT的恢复与预测也保留PA；不应因为完整状态定位更重要就将它弱化到可选。
2. **绝对空间任务需要另一组指标。** ScaleHP把CS／R／P分开，Hand3R分local／global表；InfiniHand同时记录腕位置、朝向和PA。研究目标不同决定主精度的组织方式。
3. **AUC不是所有论文必报。** HMP、DIP、PoseBERT的主要恢复表更强调位置和动态误差。保留PA-MPJPE有广泛可比性，不能只盯PA-AUC的百分点。
4. **运动误差要对GT。** 与GT的速度／加速度差用于判断动态重建；单纯降低预测自身加速度可能抹平真实动作。须注明fps、单位、缺帧／边界处理与对齐。
5. **参数角误差是分解诊断。** 本项目root_deg／finger_deg有助检验99D各部分，但不能替代关节／mesh几何主表；DIP的KIN／STA也不等于这两个GT角误差。

## 同名指标的不可混用处

- MPJPE可能是绝对、腕相对或PA，必须查正文与实现；HaWoR／Hand3R Dex表头简写MPJPE但正文指定PA。
- W-MPJPE在HaWoR／StableHand为开头相似对齐，在InfiniHand为无额外对齐；WA也可能按100帧块或完整双手场景，不可仅按名称横比。
- PA可能约束SO(3)或用自由O(3)，是否允许反射应记录；本项目完整Dex采用FreiHAND自由O(3)，并另存proper诊断，不假设所有论文相同。
- MANO16与带fingertips21joint、F@5mm与二维PCK@0.05、AUC[0,1]与百分制、加速度mm/frame²与m/s²不能混用。
- 漏检是否跳过、GT crop／自动detector、左右手、单图／16帧双向／流式、预测自身β／GT β、test split和source训练量均影响可比性。

## 对本项目的建议（未实施的新协议）

在当前DexYCB主表并列 **绝对camera MPJPE、腕平移对齐MPJPE、PA-MPJPE、PA-AUC**；mesh精度可加绝对或PA-MPVPE，但当前只完成绝对／腕对齐MPVPE，不能把PA-MPVPE当已有。

root位置／朝向与finger角度保留分解表；进一步使用速度／加速度对GT误差验证TWM的时序作用，需要独立实现及mask/fps验收，当前完整测试尚未报告。以后HOT3D世界轨迹表再按明确参考协议计算W／WA／RTE，不能给Dex固定相机结果换名。完整60步future分布的预测评测另定，不将PoseBERT的单点短时预测直接等同本模型分布输出。

上述是根据文献的报告组织建议；没有改训练／test集合、重选best或启动新评测。[当前完整测试事实](DEXYCB_FULL_TEST_RESULTS_2026-10-16.md)保持。
