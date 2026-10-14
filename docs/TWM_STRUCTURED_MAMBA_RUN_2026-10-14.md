# 结构化 Mamba TWM：重做与远程运行记录

2026-10-14。用户D25明确要求按WestWorld指导的架构适配方案实现并跑起，确认运行后再结束。旧flat Transformer pilot保留在28,213步，不覆盖或恢复旧run。新模型从头初始化。

## 本轮实际模型

- 6个时空分离block、hidden256、4个同帧attention heads、FFN512、dropout0.1。
- 每帧16个MANO解剖token，加入joint identity和parent/finger/depth结构embedding。源内部顺序为wrist/index/middle/pinky/ring/thumb，每指3个旋转，不用thumb-first的21点绘图顺序。
- 每block：pre-LN同帧关节self-attention＋残差 → pre-LN dense FFN＋残差 → 每关节沿时间pre-LN官方`mamba_ssm.Mamba`＋残差。Mamba state64、conv4、expand2，无机器人action和MoE。
- 对外完整99D当前恢复状态；内部51D切向更新再SO(3)组合。β10、handedness、dt作为条件，0.1s预测辅助；raw/candidate/history及17组availability/quality继续分开。
- 不存在的padding在进入整个主干前压缩，padding的dt被置零，输出回填零；真实漏观测保留有效时间槽并用availability表示。每次forward重新处理输入历史，无流式SSM cache或自身输出rollout。
- Mamba专门初始化保留，AdamW按其`_no_weight_decay`标志处理A/D参数。模型参数保留FP32，BF16 autocast用于网络，几何与loss用FP32。

这是参考架构机制后的手部适配，不是WestWorld完整复现；6/256/4和SSM64/4/2是有来源的起始配置，没有手部最优性或物理保证。[设计依据](TWM_WESTWORLD_GUIDED_DESIGN_2026-10-14.md)、[新主干](../src/handloop/models/structured_mamba.py)、[TWM接口](../src/handloop/models/twm.py)

## 数据与训练协议

沿旧pilot已验的OakInk2准备数据，不重新下载或转换：官方64train/8val原录制，128/16单手轨迹，204,502/18,032单手帧，50,128/4,380窗口，test标注未读。来源MoCap拟合MANO，source revision `21705616140d726607027e70d58b7837f442ffd8`；p为源腕中心规则的world wrist（米），quat wxyz→columns6D，β另存。名义30Hz、H30、future offset3、window stride4，source gap不跨窗。[详细GT/坐标/split](TWM_OAKINK2_PILOT_RUN_2026-10-14.md)

配置：[twm_oakink2_structured_mamba_pilot.json](../config/twm_oakink2_structured_mamba_pilot.json)。seed20261014、batch64、计划50,000步、lr2e-4、weight decay0.01（Mamba指定参数除外）、warmup500＋cosine、梯度裁剪1，每500步验证并保存，每25步记录状态。与旧pilot相比除了架构/容量，还有lr3e-4→2e-4和Mamba特定optimizer分组变化，不能当严格的单因素性能消融。

受损分布和监督沿旧pilot：15%clean、40%missing、45%noisy，当前位置25mm/root12°/finger8°噪声轴分量，部分严重3×及整指/整手missing。历史来自参考加小噪声，非模型递归历史；quality是含假高与unknown的随机提示，非校准误差。loss为腕SmoothL1＋root/finger chordal rotation，forecast权重0.1。数据与噪声来源无需RGB。

固定val每模式1,024窗口，记录candidate baseline与restored腕mm、root/finger SO(3)角度；按窗口平均，不是主体留出或recording等权统计。best.pt按noisy/missing loss选取，不约束clean退化。没有MANO资产/FK，不能称MPJPE/mesh；无真实HaWoR误差、最终q、视觉取证、动力学或跨库验证。

## 独立运行环境

zhou Torch2.12/cu130可以运行旧模型，但没有Mamba，现有nvcc12.6与Blackwell/Torch组合不适合直接编译；官方匹配扩展wheel针对Torch2.10。由zhou的Python3.11建立完全隔离venv（system_site_packages=False），在data10安装固定Torch2.10.0+cu130与匹配cp311/cu13/CXX11ABI TRUE扩展，不修改zhou。

运行Python：`/sciclone/data10/jchen57/Hand_Reconstruction/runtime/twm_mamba_torch210_v1/bin/python`。官方Mamba v2.3.2.post1 wheel SHA256 `7bacd9decf359e122fdd6131c3591ba5d81fce86602732fa5e30426274b07983`；causal-conv1d v1.7.0 SHA256 `46aee6009555c922bd06348d4845694f6e12ecbcd5d532ed526b6039e806fa16`，下载后核hash。

可复现源码：[环境安装](../scripts/setup_mamba_runtime.py)、[fused CUDA检查](../scripts/check_mamba_runtime.py)、[完整主干数值检查](../scripts/check_structured_twm_numerics.py)。runtime_receipt、pip reports/freeze、wheel和install.log仅远程；本机只AST/JSON/链接/diff静态检查。

## 路径与检查

| 产物 | 远程路径 |
|---|---|
| 代码 | `/sciclone/home/jchen57/project/Hand_Reconstruction` |
| 准备数据 | `/sciclone/data10/jchen57/Hand_Reconstruction/prepared/oakink2_99d_pilot_v1/manifest.json` |
| 正式run | `/sciclone/data10/jchen57/Hand_Reconstruction/runs/twm_oakink2_structured_mamba_20261014` |
| runtime | `/sciclone/data10/jchen57/Hand_Reconstruction/runtime/twm_mamba_torch210_v1` |

同一用户既有Pod `jchen57-debug-mpd5p`、GPU0；不新建/删除Job。现有Job约纽约2026-10-15 12:37到期；权重与日志在持久data10，后台进程不依赖SSH会话。

```sh
ssh ds 'python3 /sciclone/home/jchen57/project/Hand_Reconstruction/scripts/ds_train_twm.py status'
```

helper默认指向新config/Python/run，旧pilot status需显式`--output`旧目录。新run的latest.pt与best.pt每500步原子保存，满50,000步自然结束；不自动重复训练。代码版本由launcher读取真实Git HEAD，运行metadata记录主干类、框架版本、Python与配置。checkpoint可继续训练但未保存DataLoader游标，不能宣称逐batch精确重放。

实际依赖、完整数值检查、真实数据smoke和后台启动核验追加在下方；只依据实测填写，不把源码完成视为已跑起。

## 远程检查与正式启动事实

- 运行时：Torch2.10.0+cu130、Triton3.6.0、Mamba2.3.2.post1、causal-conv1d1.7.0、NumPy2.4.6，pip check通过。GPU0 tiny检查确认调用fused mamba_inner_fn，FP32/BF16前向反向均有限，FP32 fast/slow最大差0.0；不是训练效果评价。
- 完整主干参数6,330,892。6×256的12项GPU数值检查通过：实际Mamba与关节→时间轴hook、骨架索引、非零输出heads的梯度、时间因果、padding压缩/NaN/dt隔离、padding分支backward、empty history、future offset隔离、原始6D列合法旋转、BF16 backward。因果与padding压缩最大差均0.0，BF16旋转正交/determinant最大误差2.38e-7；检查batch2/H30峰值allocated约197.78MiB，不是正式batch64峰值。JSON/log在audits/structured_twm_numerics_20261014.*。
- 数值检查首个记录包装器尝试在Pod调用git失败，但checker本身已通过、原stdout已保留；后续读取结果并用已核的登录节点checkout版本补齐JSON。另一次新建SSH发生UNKNOWN port65535断开，重试成功；无模型数值失败或训练恢复操作。
- 真实Oak数据3步smoke于09:47:01UTC完成，returncode0/phase=complete，loss与梯度guard无异常；latest.pt约76.18MB。独立目录runs/twm_oakink2_structured_mamba_smoke_20261014。初始验证输出由零head保持candidate，不能称三步恢复效果；best loss Infinity表示未到正式选择检查点的验证步。
- 正式代码版本 **378deda47b123d64546404bf9780c19bcc48bb31**，09:47:50UTC（纽约05:47）由helper启动PID1276/GPU0，独立正式run目录如上。模型从头训练，未加载旧pilot或WestWorld权重。启动工具返回process_alive=true；持续运行和checkpoint核验另追加。

## 后台持续运行与保存核验

09:50:28UTC（纽约05:50）退出启动SSH后重新连接：PID1276存活，phase=training，step650，loss0.280873，梯度有限，约4.48steps/s，正式batch64峰值allocated约4.733GB。第500步latest.pt=76,179,339B，best.pt=76,177,435B已保存；best恢复loss=0.279725。按这一阶段吞吐，50,000步约3小时量级，实际以状态为准。

第500步固定验证（每模式1024窗口）是早期结果，不能判断最终模型优劣：

| 模式 | 腕candidate→restored(mm) | Root candidate→restored(°) | Finger candidate→restored(°) |
|---|---:|---:|---:|
| noisy | 49.090→49.204 | 22.538→3.981 | 15.421→5.413 |
| missing | 37.637→38.172 | 16.426→3.637 | 10.910→5.017 |
| clean | 0→5.270 | ≈0→2.727 | ≈0→3.393 |

旋转已有恢复，但腕暂未改善、clean退化需保留。不得用这组少量早期步数宣称超过旧模型/真实HaWoR或符合物理；训练继续，旧结果保持独立。启动、fused依赖、完整数值、真实数据smoke、checkpoint周期保存均已有实测，本轮用户要求的“重做并跑起来”已完成。
