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

以下命题是数学推导与候选实现条件，不是本项目已实现定理。

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
