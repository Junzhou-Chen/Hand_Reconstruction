# D34：HaWoR视觉反馈框架实现与验收

日期：2026-10-15 UTC（纽约10-14）。用户在D33讨论后明确要求“尝试搭建出来这个完整的框架”，授权本轮实现和必要的远程验收。具体网络尺寸、20步smoke预算和优化器设置是工程起点，不是最优配置。正式效果验证与完整自动视频前端分开记录。

## 实现范围

模型链为：真实RGB裁剪 → 官方冻结HaWoR ViT/IAM/head/PAM → 缓存真实图像patch和不可覆盖的raw状态 → 新视觉反馈插件读取已完成历史 → 51D几何candidate → 纯数值TWM最终当前分布 → 下一帧历史。第一版每帧一次candidate/一次TWM，不新增独立大Corrector或w/q头。

- HaWoR固定官方版本`66c7d4108d58a716deccd192cb7645170cdc7bd7`，loader验证10个关键Git blob。原16帧双向路径不接反馈；新分支位于其读出之后。官方motion infiller不调用。
- TWM输入仍只有history/raw/candidate、mask、物理时间、左右手和可选shape；图像、相机features不进入TWM。`return_forecast=False`只跳过future decoder，默认True维持既有训练兼容，state-dict键不变。
- 新插件：d256、2层，每层history cross-attention、image cross-attention、16关节self-attention和FFN；patch=192×1280。当前raw和历史数值状态形成query，当前相机/crop中的腕投影形成位置描述；首版并非16个MANO joint投影的局部采样器。最后dp/root/finger残差头零初始化，其余层正常初始化。
- 99D为真实腕J0 p3＋16个absolute/local旋转的column6D；β另存。51D残差通过SO(3)复合，不逐元素相加6D。左右手共享参数、显式handedness。
- 每帧使用上一完成腕为固定anchor；无历史时用可用raw暖启动。历史默认30帧、帧间detach，显式reset/身份变化/超过0.25秒gap清对应track；padding不提交。未知质量不冒充1，σ/κ只作分布描述。
- 当前mean/std/κ来自同一次TWM读出。NLL在anchor内计算，另外输出world mean与完整平移covariance；不能把anchor对角std直接当world对角std。未来保留4模式×60步和对应world covariance，mode由预测概率选择，不用GT选mode。

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

CPU codec、source RGB和loader合同检查已完成；官方checkpoint parity、真实cache、反馈梯度、短程训练与RGB组合推理的数值结果将在完成后补入，未完成项不视为通过。

早期依赖失败保留：原runtime无PIL/skimage；首次向在读target复制SciPy时遭遇`__pycache__`并发FileExists，改为staging；旧HF hub pin与基础Transformers5.19冲突，改为与基础版本一致的hub2.2/safetensors0.8。均发生视觉验收环境，不改现TWM runtime和run。

## 后续正式研究仍需完成

从train扩大paired训练池并保持独立validation；补真实detector/camera输入、双手和移动相机长录制；在同数据/预算下比较插件不给历史、正常反馈、错误反馈与原HaWoR/HaWoR＋TWM；补独立joint/mesh几何与长forecast评测。小样本NLL拟合、实现等价、平滑或正常运行均不构成完整方法性能结论。

设计依据：[D33讨论§11](TWM_DIRECT_REFINEMENT_PROPOSAL_2026-10-14.md#11-d33预训练hawor读出侧反馈插件与单一视觉训练目标)、[HaWoR官方源码](https://github.com/ThunderVVV/HaWoR/tree/66c7d4108d58a716deccd192cb7645170cdc7bd7)、[公平评测协议](HAWOR_DATA_AND_COMPARISON_PROTOCOL.md)。
