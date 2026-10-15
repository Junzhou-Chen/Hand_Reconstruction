# D35：HaWoR视觉反馈框架实现与验收

日期：2026-10-15 UTC（纽约10-14）。用户在D33讨论后明确要求“尝试搭建出来这个完整的框架”，授权本轮实现和必要的远程验收。具体网络尺寸、20步smoke预算和优化器设置是工程起点，不是最优配置。正式效果验证与完整自动视频前端分开记录。

## 实现范围

模型链为：真实RGB裁剪 → 官方冻结HaWoR ViT/IAM/head/PAM → 缓存真实图像patch和不可覆盖的raw状态 → 新视觉反馈插件读取已完成历史 → 51D几何candidate → 纯数值TWM最终当前分布 → 下一帧历史。第一版每帧一次candidate/一次TWM，不新增独立大Corrector或w/q头。

- HaWoR固定官方版本`66c7d4108d58a716deccd192cb7645170cdc7bd7`，loader验证10个关键Git blob。原16帧双向路径不接反馈；新分支位于其读出之后。官方motion infiller不调用。
- TWM输入仍只有history/raw/candidate、mask、物理时间、左右手和可选shape；图像、相机features不进入TWM。`return_forecast=False`只跳过future decoder，默认True维持既有训练兼容，state-dict键不变。
- 新插件：d256、2层，每层history cross-attention、image cross-attention、16关节self-attention和FFN；patch=192×1280。当前raw和历史数值状态形成query，当前相机/crop中的腕投影形成位置描述；首版并非16个MANO joint投影的局部采样器。最后dp/root/finger残差头零初始化，其余层正常初始化。
- 99D为真实腕J0 p3＋16个absolute/local旋转的column6D；β另存。51D残差通过SO(3)复合，不逐元素相加6D。左右手共享参数、显式handedness。
- 每帧使用上一完成腕为固定anchor；无历史时用可用raw暖启动。历史默认30帧、帧间detach，显式reset/身份变化/超过0.25秒gap清对应track；padding不提交。未知质量不冒充1，σ/κ只作分布描述。
- 当前mean/std/κ来自同一次TWM读出。NLL在anchor内计算，另外输出world mean与完整平移covariance；不能把anchor对角std直接当world对角std。未来保留4模式×60步和对应world covariance，mode由预测概率选择，不用GT选mode。

这里world covariance仅把当前条件Gaussian从anchor旋转到world，未传播anchor/SLAM误差或模型认知不确定性；不是完整世界状态后验的校准保证。

完整RGB入口`HandLoopFeedbackPipeline.run_track`要求原始检测mask、bbox、intrinsics、camera_to_world和时间戳。它保留world转换，不自行实现YOLO/DROID-SLAM/Metric3D，也不把缺少的相机轨迹默认为identity。部署可接原HaWoR相机与跟踪结果；本轮Dex固定camera gauge仅用于接入验收，不能当移动头戴相机SLAM成功。整个系统保留16帧lookahead。

## 文件与接口

| 文件 | 职责 |
|---|---|
| `src/handloop/visual/hawor.py` | 官方checkpoint严格加载、16帧裁剪/时间路径、真实IAM patches、post-PAM raw、原始mask |
| `src/handloop/visual/codec.py` | MANO mean/左右手/腕J0、可微state↔geometry、camera/world变换 |
| `src/handloop/models/visual_feedback.py` | 插件、顺序step/clip、跨块历史、TWM接入与单一NLL |
| `src/handloop/visual/pipeline.py` | 真实RGB与反馈模型组合，pixel-center一致的有效crop坐标 |
| `src/handloop/data/visual_clips.py` | recording分割、NPY mmap、完整性校验、严格tail masks |
| `scripts/prepare_visual_feedback_cache.py` | 选定RGB source提取及官方冻结HaWoR缓存 |
| `scripts/train_visual_feedback.py` | 自身历史训练、唯一current NLL、验证、checkpoint/resume与W&B |
| `scripts/run_visual_feedback.py` | 给定检测/相机的RGB track推理、分布及历史导出 |
| `scripts/check_hawor_bridge.py` / `scripts/check_visual_feedback.py` | 官方等价、几何与梯度/时序合同验收 |
| `scripts/setup_visual_feedback_runtime.py` | 官方固定资产与隔离依赖准备，不修改既有Torch |

缓存输入为`patch_features[B,T,192,1280]`与`raw_world[B,T,99]`，另有timestamps、frame_valid、17组availability、image_valid、handedness、world_to_camera、intrinsics和crop_transform。`target_world/target_valid`只进入loss，forward参数使用allowlist隔离GT。

历史由模型自身最终输出构成，支持`initial_history/final_history`跨16帧块继续。训练随机采样的clip起点重置历史，故smoke不能替代长录制递归验证。`adapter_history=False`仅隐藏插件历史，TWM仍读相同自身历史；`use_adapter=False`给出原HaWoR＋TWM基线。

## 训练目标与边界

\[
\mathcal L=-\frac1{|\mathcal V|}\sum_{t\in\mathcal V}
\log\left[\mathcal N(p_t^*;\mu_t,\operatorname{diag}\sigma_t^2)
\prod_{j=0}^{15}\operatorname{MF}_{\rm iso}(R_{t,j}^*;\bar R_{t,j},\kappa_{t,j})\right].
\]

只有完整17组reference可用且frame_valid为真的帧进入这个目标；先选择有效GT再做几何运算，masked NaN不能污染梯度。没有并列2D/mesh/joint/velocity/cycle/MoE/quality loss。AdamW weight decay、学习率及grad clip属于显式优化器设置；单NLL仍有条件独立与不确定性建模假设，不宣称自动校准。

默认原HaWoR与TWM全冻，只训练插件；`system.train()`仍保持TWM eval，避免冻结dropout改变。梯度保留`NLL → 冻结TWM → candidate → 插件`。可选仅解冻TWM四个current mean/uncertainty头，motion core/future头仍固定；本轮默认不开。

长forecast不在视觉阶段重新加辅助loss；冻结参数不能保证新candidate分布下forecast精度不变。后续应分别评测可见lookahead覆盖内与覆盖外的未来时域。几何误差、准确输入退化和coverage作为指标，NLL下降不能单独证明更准。

## 真实接入数据协议

只使用已下载DexYCB的subject01、固定相机`836212060125`，官方s0 train/val各一条录制，切窗前分割，不读test：

| split | recording | source frame IDs | 帧数 |
|---|---|---|---:|
| train | `20200709_141754` | 9–71 | 63 |
| val | `20200709_142123` | 3–66 | 64 |

共127张原始RGB JPEG，640×480，均右手；GT复用四库准备时已审计的完整MANO/reference及解剖腕，源SHA和camera calibration记录在source manifest。bbox由参考joint投影构造，**是GT-crop协议，无自动detector**；baseline与feedback使用相同输入。时间为`frame_id/30`的标称30Hz，不是设备实测capture timestamp。world gauge设为这一固定camera坐标，单位米，不能据此报告egocentric world泛化或独立传感器精度。

HaWoR按16帧非重叠块推理，不跨source gap；末块复制最后帧只用于官方前端padding，不增加监督或历史。feature以FP16缓存，raw/geometry为FP32，timestamp为FP64。crop_transform包含真实整数裁剪和resize pixel center，映射到去除左右各32像素后的192×256区域normalized[-1,1]。

缓存manifest只有全部录制写入后才标complete，逐NPY记录SHA256和bytes。loader首次/重开核SHA，驻留期间检查文件stat变化；同recording跨split、尾padding监督、修改mmap源均被禁止。缓存和权重只保留远程。

## 远程位置与运行方式

本轮复用用户Pod，不停止/恢复/改变现有TWM训练。独立代码checkout：`/sciclone/home/jchen57/project/HandLoop_Feedback_20261015`。数据根为`/sciclone/data10/jchen57/Hand_Reconstruction`。

```bash
export HANDLOOP_DATA=/sciclone/data10/jchen57/Hand_Reconstruction
export HANDLOOP_CODE=/sciclone/home/jchen57/project/HandLoop_Feedback_20261015
export HANDLOOP_PYTHON=$HANDLOOP_DATA/runtime/twm_mamba_torch210_v1/bin/python
export HANDLOOP_HAWOR=$HANDLOOP_DATA/model_assets/hawor_v1
export PYTHONPATH=$HANDLOOP_CODE/src:$HANDLOOP_DATA/runtime/hawor_extra/site-packages:$HANDLOOP_HAWOR/source
cd "$HANDLOOP_CODE"
```

原runtime保持Torch2.10/cu130；extras安装到独立staging，全部导入成功后再切换，不在包复制期间运行读者。`setup_visual_feedback_runtime.py --data-root "$HANDLOOP_DATA" --assets --install-extras`可重建；MANO沿既有合法资产`model_assets/mano_v1_2/prepared_v1/derived`，不处理账号或另下许可资产。

```bash
"$HANDLOOP_PYTHON" scripts/prepare_visual_feedback_cache.py --stage cache \
  --source-manifest "$HANDLOOP_DATA/prepared/visual_feedback_dex_sources_v1/source_manifest.json" \
  --output "$HANDLOOP_DATA/prepared/visual_feedback_dex_cache_v1" \
  --hawor-root "$HANDLOOP_HAWOR/source" \
  --hawor-checkpoint "$HANDLOOP_HAWOR/weights/hawor.ckpt" \
  --hawor-config "$HANDLOOP_HAWOR/weights/model_config.yaml" \
  --mano-models "$HANDLOOP_DATA/model_assets/mano_v1_2/prepared_v1/derived" --device cuda:1
```

训练配置`config/visual_feedback_v1.json`默认20步、batch1、clip32（跨两个HaWoR原块，能填满30帧历史），属于框架smoke。`--twm-checkpoint`必须指向冻结参考副本，不能在训练过程中跟随另一个run的latest变化。W&B通过正常私有环境配置，不在CLI/JSON/Git中传key。

```bash
"$HANDLOOP_PYTHON" scripts/train_visual_feedback.py \
  --manifest "$HANDLOOP_DATA/prepared/visual_feedback_dex_cache_v1/manifest.json" \
  --twm-checkpoint /path/to/frozen_twm_reference.pt \
  --config config/visual_feedback_v1.json --output /path/to/new/visual_run \
  --code-version COMMIT --device cuda:1
```

`--stop-after-updates`只限制本次执行，不改变可续跑config；`--resume latest.pt`严格核config/cache SHA/TWM reference SHA，恢复模型、optimizer、RNG，但不承诺精确重放DataLoader cursor。保存完整反馈模型（含冻结TWM），RGB推理无需旧的预训练文件仍在原路径。

`run_visual_feedback.py --track-json ... --checkpoint ...`读取显式RGB/camera/detection轨迹，输出数值预测、anchor std/world covariance与可跨块载入的history。输入JSON必须记录`translation_units="metres"`、coordinate_frame、timestamp_source；frames字段见脚本docstring。

## 验收记录

完整摘要：[framework JSON](../reports/visual_feedback_framework_20261015.json)。实现版本`4f5b4af`，边界修复`bacd221`，实际训练/组合推理使用`1f842cf`；续跑前9个关键源码/config SHA全部核对。

| 项目 | 实际结果 |
|---|---|
| 官方checkpoint | 严格load缺失/多余键均0；10个源码blob通过。IAM/PAM/β/camera与官方逐值相同；rotation matrix差≤1.19e-7，right腕J0差1.49e-8m |
| MANO与几何 | 左右shape、腕J0、FK/SE(3)往返、冻结MANO输入梯度通过，GPU几何差≤2.38e-7 |
| 核心参数 | TWM13,530,032；新增插件2,458,118；二者合计15,988,150，不含冻结HaWoR |
| 核心16项合同 | 全通过；current-only/full当前输出最大差0；zero-init候选差≤1.19e-7，最终与插件旁路差0；candidate梯度非零、冻结权重hash不变 |
| 数据/缺测 | 127张JPEG、全部NPY hash、split/mmap/tail通过；缺测NaN raw计数为0、masked NaN GT不会污染loss/指标；仅代表数值安全 |
| 真实训练 | 32帧跨块clip、batch1，先3步保存再严格resume到20步，02:45:39UTC完成；TWM完整hash不变 |
| checkpoint | 第20步395个model浮点张量、216个AdamW浮点张量全finite，72个optimizer参数状态均step20；另存step3原件及审计 |
| RGB组合入口 | 16张真实val RGB → 当前[1,16,99]、未来[1,16,4,60,99]；浮点输出finite、world covariance正定、mode概率和误差1.19e-7 |
| 在线监控 | [W&B e00b708e](https://wandb.ai/black-ant/handloop-twm/runs/e00b708e)服务端state=finished、optimizer_step20、最终NLL1497.547119实际可读 |

冻结TWM参考是D32已停止的step3110完整副本，训练代码`0d83a54`，checkpoint SHA=`bc89a078a1328e26fa8329212a58d671960a9a092a7d68e8426ba5566c8b16ed`；不是D34新run最新权重或最终模型。视觉run为`runs/visual_feedback_framework_smoke_v1_20261015`，latest20 SHA=`ae2671bda3aa7f651520b20e4367c59f6a87db2cba16e94f8db8bda4791e89e3`。实际完整预测在`evaluations/visual_feedback_framework_20261015/rgb_predictions16.pt`，weights/cache/predictions不上传GitHub。

64个val帧、固定camera未对齐参考状态的smoke结果如下；NLL为完整state均值，不除51，与pure-motion训练日志per-DoF项不能直接比较：

| 第20步验证方式 | 当前NLL | 腕mm | root° | finger° | xyz90% coverage |
|---|---:|---:|---:|---:|---:|
| 原HaWoR＋TWM，旁路插件 | 4558.707 | 95.029 | 4.936 | 6.779 | 0 |
| 训练后插件，验证时隐藏其历史 | 1962.288 | 67.436 | 5.665 | 8.544 | 0 |
| 正常反馈插件 | 1497.547 | 65.048 | 5.856 | 8.493 | 0 |

raw HaWoR本身腕95.071mm、root4.879°、finger6.808°。腕改善同时root/finger退化，不能宣称整体纠错已成功；推理时隐藏历史仅为干预，不是独立重训的公平消融。正常反馈的平均xyz std仅4.058mm而腕误差65.048mm、90% coverage为0，表明当前分布明显过于自信，不能用σ/κ作已校准可靠性。只有一个train和一个val录制，不能以此作泛化结论。缺测warmup合同测试亦出现约1.14m腕误差而数值finite，明确保留失败边界。

详细记录：[官方比对](../reports/hawor_bridge_parity_attempt2_20261015.json)、[反馈合同](../reports/visual_feedback_backend_contract_20261015.json)、[指标边界](../reports/visual_feedback_metrics_boundary_20261015.json)、[训练状态](../reports/visual_feedback_smoke_20261015.json)、[保存/续跑](../reports/visual_feedback_checkpoint_20261015.json)、[RGB入口](../reports/visual_feedback_rgb_inference_20261015.json)。

早期依赖失败保留：原runtime无PIL/skimage；首次向在读target复制SciPy时遭遇`__pycache__`并发FileExists，改为staging；旧HF hub pin与基础Transformers5.19冲突，改为与基础版本一致的hub2.2/safetensors0.8。均发生视觉验收环境，不改现TWM runtime和run。

官方central左右translation与桥接完全一致；非中心主点left的原center恢复公式由`2cx`改为真实图像宽度，合成case产生32.170mm差且符合解析式，这是显式几何勘误，不当重建精度收益。真实RGBsmoke仅右手，不能据左右codec/parity检查扩大为双手真实评测。

## 后续正式研究仍需完成

从train扩大paired训练池并保持独立validation；补真实detector/camera输入、双手和移动相机长录制；在同数据/预算下比较插件不给历史、正常反馈、错误反馈与原HaWoR/HaWoR＋TWM；补独立joint/mesh几何与长forecast评测。小样本NLL拟合、实现等价、平滑或正常运行均不构成完整方法性能结论。

设计依据：[D33讨论§11](TWM_DIRECT_REFINEMENT_PROPOSAL_2026-10-14.md#11-d33预训练hawor读出侧反馈插件与单一视觉训练目标)、[HaWoR官方源码](https://github.com/ThunderVVV/HaWoR/tree/66c7d4108d58a716deccd192cb7645170cdc7bd7)、[公平评测协议](HAWOR_DATA_AND_COMPARISON_PROTOCOL.md)。

## 2026-10-15：TWM真实误差适配与先验保持讨论（建议，未实施）

用户进一步询问：合成受损轨迹与真实HaWoR误差分布不同，后续是否应局部或全量微调TWM。用户认为全量也可能合理，尚未指定最终解冻范围。本节是源码审计与论文对照后的建议，不改变以上D35冻结smoke事实；本轮未修改模型、训练器、权重或远程运行。

### 为什么值得适配

统一接口仍是99D，改变的是误差在不同方向/关节/时间上的分布，而非状态维数。真实前端可能产生深度方向偏差、跨关节相关错误、遮挡连续缺失和16帧窗口内相关误差；闭环历史还是模型自身已完成输出，与独立合成扰动也不同。这些是需由train数据核验的机制假设，不是本轮已完成的全库残差统计。应先排除单位、参考系、MANO/左右手等实现错误，不能用训练补偿格式错误。

TWM是条件恢复网络，运动规律与观测可信性的处理混合在共享权重中，并没有可独立冻结的纯motion-prior子网。只改输入层/读出头限制较大；改变Mamba/MoE也会改变运动表征。参数少、低学习率、冻结专家但改router，都不是防遗忘保证。

### 现有代码与参数范围

以下预算来自实际mixed_v2配置及源码维度核算，不是新远程梯度实验：

| TWM解冻范围 | 参数数 | 含义 |
|---|---:|---|
| 4个当前均值/不确定性头 | 2,570 | 当前已支持的可选current_readout模式，尚未实跑；不能代表主干适配 |
| current_projection＋最后完整结构block/MoE＋final_norm＋4头 | 2,102,286 | 可作局部对照；会改变共享运动特征，尚无此训练scope |
| 完整当前计算路径 | 12,652,578 | 当前NLL可连接到的路径，不保证每个参数在给定数据中均有非零梯度 |
| 完整TWM | 13,530,032 | 另含877,454个future专属参数，需要未来训练目标 |

新增视觉插件始终另有2,458,118参数。当前trainer只支持adapter/current_readout，并检查冻结梯度/hash；TWM被强制eval，HaWoR采用离线缓存且raw detach。正式解冻需显式实现参数分组、训练模式和梯度审计，不能仅改requires_grad就宣称完成。

### 推荐主路线与对照

正式研究倾向于允许**插件暖身后，小学习率全量TWM适配，并继续训练原motion任务**；局部解冻是过渡或同预算对照，不设永久只调2,570个头的上限。先固定HaWoR，避免同时改变数值输入源和运动网络；这不要求TWM接图像。新插件与预训练TWM使用不同学习率，具体倍数/步数由训练集和独立validation确定，不能从论文直接当最优值。13.53M规模没有因为参数量而必须优先引入LoRA的依据。

正式数据应扩大至多录制/主体/动作/相机条件，并固定隔离的validation。现有63帧train、64帧val各约2.1秒，同主体、右手、固定camera、GT crop，只够smoke；完整30帧历史＋当前＋60步未来需要91个连续有效状态，现有两条短录制均不满足。重叠切窗不能创造独立动作。当前使用D32 step3110中途参考，也不能代表后续预训练最终能力。

### 保持能力依靠数据和验收

建议保留两类更新：真实HaWoR/反馈候选批使用最终current state NLL；纯轨迹回放批继续当前恢复及coherent长轨迹概率训练，GT仅作监督，TWM输入始终纯数值。无需默认再加平滑、teacher蒸馏、参数距离或多项物理加权loss。回放来自合法train分区，原val/test不得混入。两类批的抽样比例仍是任务权衡，必须明示并验证；不能把采样当作没有权重。若采用current＋60future联合NLL，horizon和有效state归一化也决定梯度份额，不能因只有一条公式就称无取舍。

**仅回放current NLL无法主动保住长预测。** 当前D35用return_forecast=false，future专属路径没有梯度；即便打开forecast而loss仍仅current NLL，结论相同。冻结future头也不足：future依赖变化的共享表征，且直接以恢复后当前state为积分起点。当前纠偏导致未来整体移动可能是正确变化，不能把所有输出差异当遗忘。

保存原预训练参考，训练工作副本；用独立集同时检查真实当前腕/joint/mesh几何、准确输入被改坏的程度、原motion恢复、未来绝对几何及相对当前的运动增量、NLL/coverage和闭环长录制漂移。能力容差在训练前按基线波动声明，以此早停/拒收/回滚。该机制仅给出有限保持集上的实证约束，不是所有运动上的数学保证。只扩大sigma也可改善严重失配时的NLL，不能取代几何改善；D35旋转退化及过度自信已说明两者需同时验收。SO(3)合法性等结构性质仍由算子保持，不等于学习到的动力学统计不会遗忘。

### 官方依据与边界

- [DIP全文Appendix A](https://arxiv.org/html/2508.01835v1)：模型针对性版本直接使用HaMeR预测与参考轨迹配对，支持学习真实视觉错误分布；未清楚交代从通用checkpoint继续微调及冻结协议，不能当作全量微调/防遗忘证明。
- [PAD-Hand全文§4.1、Appendix E](https://arxiv.org/html/2603.26068v2)：使用真实WiLoR初始预测训练refiner，支持估计器误差适配；不证明本项目的partial/full选择或sigma/kappa校准。
- [WHAM全文§3.3](https://arxiv.org/html/2312.07531v2)：AMASS合成投影/noise/mask预训练后，真实视频阶段保持图像encoder冻结，训练新integrator并微调motion encoder及motion/trajectory decoders；论文LR为新模块1e-4、预训练模块1e-5。[官方stage2数据混合源码](https://github.com/yohanshin/WHAM/blob/main/lib/data/datasets/mixed_dataset.py)仍包含AMASS，[配置](https://github.com/yohanshin/WHAM/blob/main/configs/yamls/stage2.yaml)和[optimizer分组](https://github.com/yohanshin/WHAM/blob/main/lib/utils/utils.py)支持较低LR与保留motion数据的机制。body任务与我们的99D手部恢复不同；论文30epochs与当前源码40epochs不可混称同一协议，论文比例/学习率不直接照搬。
- [LP-FT全文](https://arxiv.org/pdf/2202.10054)：在其分类实验和受限线性理论中支持先拟合新head再全量适配；只能作为插件暖身的参考，不能外推为Mamba恢复的保证，低LR本身也非普适防遗忘定理。
