# TWM物理约束的数学保证与验证方案

日期：2026-10-14。用户要求参考WestWorld及其他world model，研究当前99D恢复模型能否从数学上符合物理、如何验证以及怎样修改。建议采用可证明的几何解码与明确约束，再用独立参考检验当前恢复和运动一致性。真实受控动力学需额外的物理变量和辨识条件，不能由轨迹预训练本身保证。本方案供架构讨论，尚未实施或替代现行设计；与[TWM直接校正讨论](TWM_DIRECT_REFINEMENT_PROPOSAL_2026-10-14.md)兼容。

## 1. 需要证明的命题

| 命题 | 所需条件 | 当前适用程度 |
|---|---|---|
| 输出是合法旋转、满足指定关节约束 | 明确输出参数化或可行域映射 | 可通过解码结构获得条件性保证 |
| 内部骨架连接、参考系转换一致 | 固定shape、正确FK与坐标契约 | 可推导并做数值验收 |
| 帧内迭代降低一个明确目标 | 固定评价目标、合法路径和步长/接受规则 | 可选的算法保证；不是机械能守恒 |
| 输出更接近真实运动 | 独立高质量reference与留出协议 | 需实验；合法姿态不等于真实姿态 |
| 满足真实动力学、识别真实质量/力 | 速度/动量、惯性、驱动/外力及可辨识性 | 当前数据不足以普遍证明 |

当前TWM学习的是状态后验恢复，概念上为 `p(S_t | H_t,Z_t,O_t,M_t,w_t,beta)`。99D是配置编码，对应无解剖限制时的流形 `R^3 × SO(3)^16`，内在维度51；不是包含动量的99个独立物理坐标。物理动力学的状态还涉及速度/动量及驱动。

同帧 `S_t^(k) → S_t^(k+1)` 是估计迭代。k和flow/diffusion采样时间都不是物理时间t；速度、加速度和力学残差必须沿真实timestamps的最终轨迹计算。

## 2. 相关论文实际保证什么

| 主源 | 数学结构或检验 | 能借鉴的性质和限制 |
|---|---|---|
| [WestWorld v2，§3–4/E.1](https://arxiv.org/html/2603.14392v2) | LCRS结构embedding、action条件、状态分箱CE；物理单位预测误差 | 结构有助于统计动力学学习；没有输出力学约束或普遍守恒保证。Physical-space不是物理定律验证 |
| [OneWorld v1，§3/A/E](https://arxiv.org/html/2609.30946v1) | 学习多action分支的共享机制兼容性，加入loss和采样引导；理想posterior的evidence-ratio恒等式 | 一致解释与真实物理配置仍需区分；实际分数不是校准Bayes factor，无通用物理正确性定理 |
| [HNN，§2/3.2](https://arxiv.org/html/1906.01563v3) | `xdot=J grad H_theta`，J反对称 | 自治、无输入/耗散的连续系统保持learned H；不自动等于真实机械能，数值求解也需另审 |
| [LNN，§2/4](https://arxiv.org/html/2003.04630v2) | 学L并用Euler–Lagrange导出加速度 | 显式解需速度Hessian可逆；自治、无非保守力才有相应能量守恒；不允许直接把6D冗余量当独立角度 |
| [DeLaN，§3–4](https://arxiv.org/html/1907.04490v1) | 惯性矩阵`M=LL^T`并限制L正对角，按EL构造运动方程 | 可保证M正定及特定结构；训练使用力矩，不能由被动姿态直接反推唯一真实受控动力学 |
| [SymODEN，§3/4.3](https://arxiv.org/html/1909.12077v2) | Hamiltonian结构加已知control；位置/速度轨迹训练 | 原文展示mass/energy/input matrix缩放的不可辨识性；预测正确不等于物理参数唯一正确 |
| [Dissipative SymODEN，§2–3](https://arxiv.org/html/2002.08860v2) | `(J-D)grad H+B u`，D半正定 | 保证指定模型的输入功率与耗散收支；更适合开放系统，但需要有意义的输入、能量与耗散模型 |
| [BMC，§3.1及附录](https://arxiv.org/html/2003.09282v2) | 骨长、掌结构及屈伸/外展耦合范围的可微soft loss | 提供手部约束定义；soft loss只鼓励满足，不能升级为所有输入的严格保证 |
| [Πnet，§2.1](https://proceedings.iclr.cc/paper_files/paper/2026/file/66ba03db7e50b5db5593fe759527a74f-Paper-Conference.pdf) | NN输出投影到非空闭凸集合，目标函数可非凸 | 可行性由投影定义而来；不能把MANO的非凸姿态/mesh集合直接当成该凸集合 |
| [Physion++，§3–4](https://arxiv.org/html/2306.15668v2) | 控制外观、改变物理隐属性并给交互历史，测试物理预测 | 借鉴控制变量与独立测试的思想；benchmark表现不等于普遍定理，也不要求本项目增加物体模拟 |

[Physically Viable World Models §3.1](https://arxiv.org/html/2605.30542v1)强调按任务选择足够的变量、参数和约束，并承认可辨识性限制；这是设计原则，支持限定当前恢复需要的物理抽象，不是通用正确性证明。

## 3. 守恒结构的证明及其适用范围

HNN令相空间状态x包含canonical位置和动量，定义

\[
\dot x=J\nabla H_\theta(x),\qquad J^\top=-J.
\]

若H无显式时间依赖且没有输入/耗散，则

\[
\frac{dH_\theta}{dt}=\nabla H_\theta^\top J\nabla H_\theta=0.
\]

最后一个等式来自反对称矩阵的二次型恒为零。它对模型结构成立，因而比训练中添加一个energy loss更强。但它保持的是H_theta；例如H恒为0也满足该恒等式，却可能预测完全错误的运动。机械能的物理意义、轨迹准确性与离散积分误差都需要另外检验。

开放受控形式为

\[
\dot x=(J-D)\nabla H_\theta+B u,\qquad D\succeq0,
\]
\[
\frac{dH_\theta}{dt}
=-\nabla H_\theta^\top D\nabla H_\theta+u^\top y,
\qquad y=B^\top\nabla H_\theta.
\]

这证明指定模型的耗散与输入功率平衡，不要求运动时H恒定。人手有肌肉驱动、手臂与接触作用，目前不能用“全手能量必须守恒”筛除急停或发力动作。

广义动力学可写为 `M(q) vdot + C(q,v)v + g(q) = B tau + J_c(q)^T lambda + f_ext`；配置导数与速度可由 `qdot=N(q)v`联系，不能对99D普通差分直接套最小坐标动力学。[MIT多体动力学](https://underactuated.mit.edu/multibody.html)

如果允许一个不受限制的广义力f，则对任意足够光滑轨迹都可设置 `f=M vdot+C v+g` 令残差为零；这不能独立验证正确性。[PAD-Hand](https://arxiv.org/html/2603.26068v2)从reference轨迹计算pseudoforce，属于经过近似动力学算子变换的reference监督，不是直接测量力。它不等同于上述用预测自身自由力形成恒等式的无效验证；两者必须分开。

## 4. 当前99D可以采用的可证明解码

以下保留此前方案阶段的数学推导与候选实现条件。D25之后的实际SO(3)实现、Mamba内部稳定性与条件恢复理论，按本次源码复核另见§8–12；不能将所有候选约束都称为已实现。

### 4.1 合法旋转

已有合法旋转R时，令TWM内部输出切向增量delta：

\[
R^+=\exp([\delta]_\times)R.
\]

反对称矩阵Omega满足 `(exp Omega)^T exp Omega=I`，且 `det(exp Omega)=exp(tr Omega)=1`，故R+属于SO(3)。16个旋转分别按此更新，再编码为6D，对外仍输出完整99D。

直接6D解码也可保证合法性：对非零、不共线的a/b执行Gram–Schmidt，再以叉积生成第三轴。若a为零或正交分量为零，必须明确回退到上一合法旋转或单位矩阵；仅在分母加epsilon不能严格保证单位正交。浮点实现须按容差验证，而不是声称误差绝对为零。[旋转6D实现](https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#rotation_6d_to_matrix)

### 4.2 内部骨架一致性

固定beta，对内部FK骨架父子偏移b_i(beta)，有

\[
j_i-j_{\pi(i)}=G_{\pi(i)}b_i(\beta),\qquad G_{\pi(i)}\in SO(3),
\]
\[
\|j_i-j_{\pi(i)}\|=\|b_i(\beta)\|.
\]

证明来自正交旋转保持范数。它约束父子连接和固定内部骨长。MANO的16个内部FK关节中心与最终mesh指尖必须区分：21点中的mesh指尖及posed mesh另行回归点不都满足刚性杆假设，LBS和pose blend shapes可改变表面距离。[官方FK/LBS](https://github.com/vchoutas/smplx/blob/main/smplx/lbs.py)、[HaMeR关节扩展](https://github.com/geopavlakos/hamer/blob/main/hamer/models/mano_wrapper.py)

### 4.3 指定解剖范围

先定义解剖角坐标a及允许集合A，而不是对6D编码分量截断。对单轴区间可用

\[
a_i=l_i+(u_i-l_i)\sigma(z_i),
\]

因sigmoid在0和1之间，对任意有限网络输出都满足所选区间。也可投影到明确的A后生成合法旋转并重新编码99D；精确映射输出属于A，但复杂非凸投影的数值可行性须验收。

独立角度区间不包含全部关节耦合、自交和个体差异。BMC用屈伸与外展的联合范围说明这一点；本项目范围须用校准/参考与train数据定义，在held-out参考上检查覆盖，不以约束排除真实少见动作。硬保证只针对所定义集合，不能直接称“所有人体解剖均正确”。

若A非空闭凸且真实参数a*属于A，则精确欧氏投影满足

\[
\|P_A(z)-a^*\|^2
\le\|z-a^*\|^2-\|z-P_A(z)\|^2.
\]

证明：投影最优性给出 `<z-P_A(z),a*-P_A(z)>≤0`，展开平方即可。这是不增加该参数范数误差的条件保证，不等同于MANO解码MPJPE。非凸集合一般没有这个结论；`A={-1,1},0<z<1,a*=-1`即是反例，错误范围或近似IK也不能沿用该证明。

### 4.4 全局参考系一致性

采用同帧固定anchor `A=(p_A,R_A)`，将输入转成

\[
\bar p=R_A^\top(p-p_A),\qquad \bar R=R_A^\top R.
\]

全局刚体变换G=(R_G,t_G)同时作用于历史、raw、candidate与anchor。则

\[
(R_GR_A)^\top\big[(R_Gp+t_G)-(R_Gp_A+t_G)\big]=\bar p,
\]
\[
(R_GR_A)^\top R_GR=\bar R.
\]

所以规范化后的数值输入不变，TWM的anchor输出不变，转换回world后满足 `W(G·inputs;G·A)=G·W(inputs;A)`。这是参考系等变性，可不依赖神经网络具体权重。前提是全部world点、方向、旋转、相机extrinsics与重力等条件按各自规则变换并表达为anchor不变量，其余条件不变；输出head只返回anchor量，再转换回world。随机模型需使用相同随机量，或将命题改为输出分布等变。它保证换坐标系不改变同一个运动，不能保证原SLAM坐标没有漂移。

## 5. 同帧更新的可选下降保证

定义固定评价目标

\[
E(S)=L_{obs}(S;Z,O,w)+\lambda_p E_{prior}(S;H,\beta)+\lambda_c E_{constraints}(S).
\]

E是重建代价或learned prior score，不是机械能。TWM提出合法候选；只接受可行且 `E(candidate)≤E(current)` 的候选，否则保留当前状态，即可保证该E不增。这是确定性验收规则，不需独立learned Corrector，但可能停滞，也有额外计算成本。

若要证明存在严格下降步，需E在合法路径附近可微，方向d满足 `<grad E,d><0`，再沿 `S(alpha)=S⊕alpha d`做Armijo backtracking。一阶展开保证充分小的正alpha能通过。非下降网络方向可拒绝或使用合法负梯度备用方向；约束边界和不可微投影须单独处理。

欧氏闭凸集合C的简洁版本：若E梯度L-Lipschitz，x属于C，`y=P_C(x-eta grad E(x))`，则

\[
E(y)\le E(x)-\left(\frac1\eta-\frac L2\right)\|y-x\|^2.
\]

投影最优性与descent lemma相加即得；`0<eta<2/L`给下降系数。MANO非凸流形不能直接套用该凸命题。即便E有下界且能量值收敛，也不自动到达全局最优或真实姿态。

固定评价E是必要条件。若每轮替换w/O使E变成不同函数，不能跨轮宣称单调下降；若使用状态相关图像查询，要把它定义成同一个完整评价函数，或只报告固定子问题的下降。反例 `E(x)=(x-1)^2,真实x*=0`：从0到1降低E却增加真值误差。

## 6. 跨帧稳定性的条件性命题

将状态和反馈质量均计入系统。若能证明在声明域上

\[
e_t\le\rho e_{t-1}+b_t,\qquad0\le\rho<1,\quad b_t\le\bar b,
\]

则递推有

\[
e_t\le\rho^t e_0+\frac{\bar b(1-\rho^t)}{1-\rho}.
\]

b须包含观测噪声、模型误差及外部输入误差。对K帧历史，若 `e_t≤sum_i a_i e_(t-i)+b_bar`，a_i非负且和rho小于1、b_bar非负，并定义 `E0=max_(1-K≤t≤0)e_t`，则可得上界 `max(E0,b_bar/(1-rho))`。短历史本身不证明稳定，`F(h)=2h`仍会放大误差。有限测试估计的Lipschitz常数不能冒充全域证书，6D退化与旋转chart切点也影响域假设。首轮先做历史扰动放大测试，无需为此自动重构整个网络。

## 7. 独立验证和最小修改

| 验证 | 指标与测试 | 必须保留的条件 |
|---|---|---|
| 旋转合法 | `||R^T R-I||_F`、`|det R-1|`、退化回退计数 | 包含零/近共线编码边界；数值容差与dtype明示 |
| 骨架/解剖 | 内部FK长度残差、定义范围的角度违例率/幅度；held-out参考覆盖 | 内部16中心与最终21点分开；范围来源、个体标定和beta固定 |
| 参考系一致 | 对全局刚体变换前后计算joint/wrist等变误差 | 同步转换相机/anchor；不是独立真实world精度证明 |
| 当前恢复 | local/world MPJPE、wrist位置/朝向误差、干净输入退化 | 独立reference，原split/source，统一shape/坐标；不只看physics score |
| 运动一致 | 与reference比较线速度/角速度及加速度，按真实dt | 旋转用SO(3) Log差分；分组急停/急转，不能把小速度当成功 |
| 遮挡与反馈 | 连续missing、valid-but-wrong、假高质量、历史污染、再可见恢复 | 使用预测质量和自身历史；不偷用GT错误作为推理输入 |
| 迭代下降 | 固定E的逐轮曲线、接受/拒绝率、停滞与计算量 | 同时报告真值误差，不能由E下降替代几何结果 |
| 更强力学 | 有独立力/质量/接触参考时测试动力学/功率残差和不同驱动 | 未观测力不从同一预测自由构造；受控/模拟验证与自然手评测分开 |

所有区间、噪声幅度、阈值和权重在train/validation确定。训练physics loss的评分器之外，使用原生marker/glove参考及不同测试条件验证，借鉴Physion++/OneWorld的控制变量与独立验证思想；不把同图反复query称为物理action干预。

建议首轮保持统一99D恢复任务，落实SO(3)解码、固定beta及参考系规范；增加有来源的解剖约束并比较soft/hard版本，量化准确输入退化。对直接TWM方案，硬解码放在内部输出层，不另引入大型状态融合网络。固定目标验收/下降更新作为下一项可选对照；HNN/DeLaN不是默认替换整个TWM。

若以后要声称真实动力学，可保留99D恢复主输出，另研究带明确输入、正定惯量和耗散结构的辅助动力学分支；先在物理量已知的受控系统验证，再讨论自然手的不可辨识部分。当前姿态库不能因此自动升级为力/质量真值。

完成相应实现和数值验收后，首轮拟主张的是：**在声明的解码和约束假设下保证几何可行性与参考系一致性，并以独立reference验证当前运动恢复。** 增加约束是否改善真实误差、是否值得引入下降规则，仍由消融决定。

## 8. 2026-10-14：当前Mamba TWM的证明范围复核

用户再次要求认真推导物理约束与数学性质，数学讨论恢复；没有要求本轮修改训练架构或启动新实验。本节按本地源码快照2b40aa9复核geometry.py、twm.py、structured_mamba.py与train_twm.py，并对照官方Mamba v2.3.2 source。以下是声明假设下的数学推导与静态代码事实，没有新增远程数值实验、模型性能或全域数值证书。

| 性质 | 可证明的精确范围 | 当前实现状态 |
|---|---|---|
| SO(3)旋转合法 | 合法基础旋转经Exp左乘，精确实数成立 | 已实现；浮点按容差及无溢出域验收 |
| 固定内部骨长 | 同handedness、同decoder、固定β的16内部FK关节 | 解码器性质；trainer没有骨长/joint loss |
| 全局SE(3)等变 | 同步变换全部输入与anchor，规范化后推理再逆变换 | trainer有to_anchor，完整部署wrapper仍未验 |
| 条件信息降低最优风险 | 同一真实分布、损失与允许输出集合下的理想决策类 | 任务理论；不是实际有限训练网络的性能证书 |
| Mamba内部有界 | 冻结有限参数、pre-LN、正epsilon、精确计算 | 可从当前结构推导；未量化实际常数 |
| 完整反馈误差收缩 | 全部历史路径满足小增益条件 | 未认证，当前未自身历史rollout |
| 解剖限位/固定目标下降 | 另加明确hard映射/接受规则 | 未采纳或实现 |
| 真实受控动力学 | 惯性、速度、驱动与外力的明确来源及辨识条件 | 当前轨迹监督不能普遍保证 |

当前几何实现的退化6D回退有更明确的保证：单位第一轴为b1时，选择最不平行的坐标轴e_k，至少有|b1_k|≤1/sqrt(3)，故其正交投影长度≥sqrt(2/3)>0，可构造合法第二轴。state_apply_delta的16组Exp左乘因此保持SO(3)。结论假设中间计算不发生溢出/非有限；有限float输入本身不排除norm等中间量溢出。参考[geometry.py](../src/handloop/geometry.py)。

§4.4的SE(3)等变是规范化wrapper性质，普通TWM单体直接输入world量不具备该保证。所有可用旋转合法、anchor有效且随全局变换同步变换、同mask/quality/dt/shape与eval模式下，anchor输入严格不变；缺失占位在anchor之后统一处理。无历史时固定world单位anchor不直接满足此条件。有dropout时需相同随机量或只主张分布等变。world历史噪声也需同时旋转position/root切向噪声，不能把同seed同world向量当逐样本等变；各向同性高斯可给分布等变。退化6D在world先用固定坐标轴回退亦非等变，故物理输入要求合法旋转；anchor后统一处理mask占位不改变证明。

关闭四库β conditioning只改变网络输入，不改变外部固定β FK的代数性质；但同β数字跨source不等于同一shape。解码须保留源decoder与β约定。内部骨长保证不涵盖全部21点mesh指尖或所有LBS表面距离。

## 9. 条件恢复为何有理论依据

### 9.1 任意损失的最优风险不增

令F表示history/mask等已有信息，G表示再加入当前数值观测后的信息，F⊆G；允许输出集合C非空且与真实联合分布固定，损失非负可测，并存在有限风险的允许估计器。定义

\[
R_F^*=\inf_{f\ {\rm F\text{-}measurable},\,f\in C}\mathbb E\ell(S,f),
\qquad
R_G^*=\inf_{g\ {\rm G\text{-}measurable},\,g\in C}\mathbb E\ell(S,g).
\]

任何仅使用F的估计器也可在G下忽略新增信息而运行，故决策类包含关系直接给出

\[
R_G^*\le R_F^*.
\]

此结论不要求高斯或凸集合，也可使用现有SmoothL1＋chordal损失；但比较的是不受限理想决策类的最优期望风险，不是指定容量Mamba的两次实训结果。它不保证每个样本/每个关节误差下降，也不是本项目首次提出的统计定理。

### 9.2 局部MSE的信息收益与网络误差

在避开旋转切点的局部坐标X中，参考坐标固定或由F可测历史确定，不依赖未知GT。假设E||X||²有限，定义R(f)=E||X−f||²，记μ_F=E[X|F]、μ_G=E[X|G]；冻结参数、eval模式下f_theta为G可测且平方可积。本小节比较欧氏坐标MSE，不施加额外非凸姿态输出约束，也不自动等于MPJPE。条件均值的正交性给出

\[
R(\mu_F)-R(\mu_G)
=\mathbb E\|\mu_G-\mu_F\|^2,
\]
\[
R(f_\theta)-R(\mu_F)
=\mathbb E\|f_\theta-\mu_G\|^2
-\mathbb E\|\mu_G-\mu_F\|^2.
\]

证明：展开X−μ_F=(X−μ_G)+(μ_G−μ_F)的平方；由于E[X−μ_G|G]=0，交叉项期望为零。第二式同理。

因此实际网络兑现收益的条件是逼近/泛化误差小于新增信息收益。当前trainer并非全局MSE，不应将其直接称为MMSE或已恢复真实posterior；合成→真实HaWoR分布变化也不能由该定理消除。标准参考：[MIT MMSE与正交性笔记](https://ocw.mit.edu/courses/6-011-introduction-to-communication-control-and-signal-processing-spring-2010/a6bddaee5966f6e73450e6fe79ab0566_MIT6_011S10_notes.pdf)。

### 9.3 可见部分如何约束隐藏部分

观测集合O固定，或把mask选择并入H后明确假设局部状态(X_O,X_U)|H仍服从联合高斯，协方差块为Σ_OO、Σ_OU、Σ_UO、Σ_UU。观测Y=X_O+ε，ε与(X_O,X_U)在给定H后条件独立且服从N(0,R)，R正定。令V=Σ_OO+R，高斯条件分布给出

\[
\mu_U^+=\mu_U+\Sigma_{UO}V^{-1}(Y-\mu_O),
\]
\[
\Sigma_{UU}^+
=\Sigma_{UU}-\Sigma_{UO}V^{-1}\Sigma_{OU}
\preceq\Sigma_{UU}.
\]

对任意方向v，减少量为||V^{-1/2}Σ_OU v||²≥0；严格减少要求Σ_OU v≠0。这解释跨关节/时间条件依赖为何有用，而不是“可见保留、隐藏独立预测”。非零cross-covariance不保证全部隐藏方向改善；误设观测噪声R也不提供真实误差保证。真实visibility相关缺失不能未经检查沿用联合高斯假设。现模型没有显式covariance/Kalman gain，人工quality不能充当校准R。参考：[Stanford高斯条件分布](https://web.stanford.edu/class/stats306b/doc/stats306b-spring14-lecture9_scribed.pdf)。

candidate由history/raw/mask确定性构造，单独改变candidate不会增加原始贝叶斯信息；迭代可改善算法近似，但只有读取原状态未保留的真实视觉证据才可能增加恢复条件。重复查询同一图像不等于独立新样本。

## 10. 当前旋转损失与内部关节几何误差

相对旋转角θ∈[0,π]满足tr(RᵀR*)=1+2cosθ，因此现有chordal项

\[
c=\tfrac12\|R-R^*\|_F^2
=4\sin^2(\theta/2),\qquad
\frac4{\pi^2}\theta^2\le c\le\theta^2.
\]

小角度时c=θ²+O(θ⁴)。这提供旋转损失与平方角误差的联系，但平均chordal下降不能保证每个角误差或MPJPE下降。

同β、同内部FK下，设路径骨向量b_k、长度ℓ_k、累计父旋转G_π(k)，有j_i=p+Σ_{k∈path(i)}G_π(k)b_k。由三角不等式与||R−R*||₂=2sin(Θ/2)，

\[
\|\hat j_i-j_i^*\|
\le e_p+\sum_{k\in{\rm path}(i)}
2\ell_k\sin\!\frac{\Theta_{\pi(k)}}2
\le e_p+\sum_{k\in{\rm path}(i)}
\ell_k\sum_{a\in{\rm ancestors}(\pi(k))}\theta_a.
\]

式中ancestors(π(k))定义为从root到π(k)且包含两端的路径节点集合，即包含父节点自身的局部旋转。该界把腕和局部旋转目标连接到内部几何；β不同须另加骨向量差项，mesh指尖/LBS另需分析，不能替代实际MPJPE。

## 11. 当前Mamba能证明的稳定性

### 11.1 先使用实际scan递推

官方Mamba v2.3.2中a_dn=exp(A_log,dn)>0、Δ_dt=softplus(l_dt)>0，实际递推为

\[
h_{dn,t}=e^{-a_{dn}\Delta_{dt}}h_{dn,t-1}
+\Delta_{dt}B_{n,t}u_{dt}.
\]

来源：[官方Mamba类](https://github.com/state-spaces/mamba/blob/v2.3.2/mamba_ssm/modules/mamba_simple.py)、[官方selective_scan_ref](https://github.com/state-spaces/mamba/blob/v2.3.2/mamba_ssm/ops/selective_scan_interface.py)。驱动项是ΔBu；不能将[原论文](https://arxiv.org/html/2312.00752v2)完整ZOH推导的凸组合门控公式直接视为此实现。

在卷积driver及Δ/B完全相同、仅SSM初始隐状态不同的两次递推中，差r满足

\[
r_{dn,t}
=\exp\!\left(-a_{dn}\sum_{\tau=1}^t\Delta_{d\tau}\right)r_{dn,0}.
\]

若统一a_min、Δ_min均正，则ρ=exp(−a_min Δ_min)<1，||r_t||≤ρ^t||r_0||。仅Δ_t>0但累计有限，不能证明初始影响最终消失。

### 11.2 pre-LayerNorm给出的条件性内部有界证明

当前每个Mamba前有LayerNorm，维度D=256、epsilon>0。冻结有限参数、精确实数运算下，

\[
\left\|\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\right\|_2^2
=\frac{D\sigma^2}{\sigma^2+\epsilon}\le D,
\qquad
\|{\rm LN}(x)\|_2\le\sqrt D\|\gamma\|_\infty+\|b_{\rm LN}\|_2.
\]

有限线性/固定长度卷积/连续激活将这个有界输入映到有界域；故softplus输入上下有界，Δ_min>0，Δ_max有限，B/u有界。令驱动范数上界为G，

\[
\|h_t\|\le\rho\|h_{t-1}\|+G
\le\rho^t\|h_0\|+\frac{G(1-\rho^t)}{1-\rho}.
\]

这是内部bounded-state与初始状态遗忘；常数依赖冻结权重，可能极松，未提供实际数值证书。浮点underflow/overflow须另外验收。这里的隐状态范数或Lyapunov量不是手的机械能。

### 11.3 内部稳定不是完整输出/反馈收缩

输入不同时，Φ(u)h+g(u)的差包含[Φ(u)−Φ(u')]h'与g(u)−g(u')，不会像同驱动初态差那样相消。attention、FFN、残差、head、candidate base及anchor依赖都需要计入完整映射增益。模型没有全局Jacobian收缩证书；物理dt也只是token条件，未与Mamba内部Δ强制相等。

现有架构允许简单反例：所有root/anchor朝向固定为I，missing候选取上一最终状态且candidate_availability为true，translation head令W=0、bias=c/0.1，则world位置p_t=p_(t−1)+c=p_0+tc；即使内部SSM有界，每次修正也有界，世界位置仍可漂移。每次forward重置SSM的30帧窗口也不改变这个反例。

令U_f=sqrt(D)||γ_final||_∞+||b_LN,final||₂，final LayerNorm可为单次修正给出依赖权重的有限界：

\[
\|\Delta p\|\le0.1(\|W_p\|_2U_f+\|b_p\|_2),\quad
\|\Delta\theta_j\|\le0.5(\|W_R\|_2U_f+\|b_R\|_2).
\]

0.1/0.5是缩放而非预设物理硬上限。

### 11.4 30帧反馈的可选小增益命题

令e_t≥0为完整状态误差、b_bar≥0，若完整world输出和quality反馈在声明域满足e_t≤Σ_{i=1}^{30}a_i e_(t−i)+b_t，a_i≥0、ρ=Σa_i<1、b_t≤b_bar，且rollout始终位于声明域（或域前向不变），则令d=b_bar/(1−ρ)，v_t=(e_t−d)_+。有v_t≤Σa_i v_(t−i)，对t≥1逐30帧归纳得到

\[
e_t\le d+\rho^{\lceil t/30\rceil}
\max_{-29\le s\le0}(e_s-d)_+.
\]

a_i必须覆盖历史填补candidate、anchor变化、逆坐标变换和未来q误差。当前没有认证这些a_i/b_bar，也未自身历史rollout；有限扰动测试仅是经验增益，不是全域定理。

## 12. 不可观测性与下一项验证边界

若两种合法状态S1/S2在同一已有history/当前观测条件下不可区分，令满足三角不等式的距离d(S1,S2)=D，确定性恢复器只能给相同输出。三角不等式说明至少一种误差≥D/2；给定同一输入后两状态条件概率各为1/2时，平均平方距离误差≥D²/4。骨长、关节范围或未知输入动力学不自动排除这种多解；全局gauge也不会因参考系等变而变得可识别。

主动手的肌肉驱动、接触力和惯性未独立测量；任意轨迹配一个自由力即可使动力学残差为零，不能自证真实物理正确。关节约束映射只保证定义的集合可行；非凸MANO/人体集合投影也不能套§4.3的凸投影误差不增定理。

本次建议的理论主线是“合法几何与内部运动学结构＋参考系一致＋条件恢复的信息依据”，Mamba内部稳定作限定的结构性质。实际恢复、准确观测保护、跨帧反馈和独立GT几何仍靠对应实验。未决定加入HNN、hard限位、谱范数约束、下降验收或改变当前四库训练。
