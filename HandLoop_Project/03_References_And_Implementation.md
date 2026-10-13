# References, Code Interfaces, and First Experiments

> 用于新的 Work 项目开始实际 code/reproduction。名称、license、权重下载方式、数据使用协议、GT 标签定义以及 2026-10 之后的更新应进入对应仓库再次核对；**这里只列讨论中已出现的资源及能明确识别的接口，不将研究中的推测当成已验证实现**。

## 1. 相关工作（方法优先）

| 论文 / 系统 | 初步用途 | 官方/论文链接 | 注意事项 |
|---|---|---|---|
| **HaWoR (CVPR 2025)** | **已决定的 egocentric visual backbone**；camera-space hand、world-space alignment、temporal processing | [GitHub](https://github.com/ThunderVVV/HaWoR)、[arXiv](https://arxiv.org/abs/2501.02973) | 记录代码版本、权重、相机参数和 temporal lookahead |
| **WestWorld (2026)** | 借鉴 structured trajectory modeling，**不用机器人 checkpoint 或原始动作条件训练任务** | [GitHub](https://github.com/511205787/WestWorld) | robot actions + multi-system structure + discretized channels，不是直接 69D wrist/hand model |
| **HMP (2024)** | 早期独立 MoCap motion prior + visual hand recovery | [arXiv](https://arxiv.org/abs/2312.16737) | 重要 novelty 基线 |
| **StableHand (2026)** | 视觉质量感知的运动重建及 flow/trajectory prior | [arXiv](https://arxiv.org/abs/2605.18553) | 与本工作很相近；研究差异要通过强实验 |
| **PAD-Hand** | 物理/不确定性感知的手部运动恢复 | 查 CVPR 2026 官方文献 | 与 confidence+state recovery 相关，引用时核实 |
| **EgoInfinity (2026)** | metric geometry + RGB hand-object reconstruction；历史替代视觉方案 | [GitHub](https://github.com/Rice-RobotPI-Lab/EgoInfinity) | 非最终决定的主 backbone |
| **Do as I Do (2026)** | 下游 reconstruction→robot retargeting / validation | 按论文标题搜索官方项目 | 不必完整复刻 |
| **WiLoR (2025)** | single-frame RGB hand baseline；HaWoR 原有相关组件 | CVPR OpenAccess 查核 | 不取代已决定的 HaWoR |
| **MINT / OmniHands / ACE-Ego-Hand / ViDiHand / EgoFound3R** | 后续 backbone / occlusion 强基线候选 | 以 2026 官方仓库为准 | code/train checkpoints availability 随时变化 |
| **ManipTrans / DemoDiffusion** | residual trajectory refinement 的邻近思想 | 按论文名检索 | 更偏 retargeting，与 hand visual estimation 不能混同 |

请不要在论文中未经验证就使用归档中曾出现的精度数字、SOTA 排名或“首次提出”断言。

## 2. 数据资源（训练和评测用，数据集论文不是主要 Method）

| 数据来源 | 讨论中的用法 | 链接 | 核查事项 |
|---|---|---|---|
| **HOT3D** | 第一视角 RGB + professional MoCap/reference；paired training/eval | [GitHub](https://github.com/facebookresearch/hot3d)、[项目站](https://facebookresearch.github.io/hot3d/) | 哪些 split 有公开 3D GT；camera stream 的单/多视角；时间对齐；hand shape/topology |
| **DexCap** | instrumented glove + RGB-D + wrist pose，motion-only 和 paired | [GitHub](https://github.com/j96w/DexCap)、[论文](https://arxiv.org/abs/2403.07788) | Rokoko joint / T265 wrist SLAM 的不同误差来源、标定变换、数据访问协议 |
| **EgoDex** | 大规模运动预训练候选 | [GitHub](https://github.com/apple-aiml-research/ml-egodex) | Vision Pro 设备视觉 tracking **不等于独立 glove GT**，应质量分层 |
| **AMASS** | 额外高质量人体 MoCap motion pool | [项目主页](https://amass.is.tue.mpg.de/) | 哪些子集有 hand articulation、license、骨架映射 |
| **ARCTIC** | occlusion / HOI 验证候选 | 原论文官方 | 标签空间、相机模态、与 HaWoR 是否同条件 |

## 3. 官方 HaWoR 代码的真实接口（优先阅读）

1. `https://github.com/ThunderVVV/HaWoR/blob/main/scripts/scripts_test_video/hawor_video.py`
   - `hawor_motion_estimation(...)` 读取 tracklets，bbox 缺帧可做插值，再运行模型 inference。
   - 输出中看到 `init_root_orient`, `init_hand_pose`, `init_trans`, `init_betas` 等，并保存 `cam_space/.../*.json`。
   - **原始 `valid = ...['det']` 会在 bbox interpolation 后改变**；需要在插值前保存 raw `det` 才能区分 observed/inferred。
   - `hawor_infiller(...)` 可独立调用，第一版 **建议绕开而作为 baseline**。

2. `https://github.com/ThunderVVV/HaWoR/blob/main/lib/models/hawor.py`
   - `HAWOR.forward_step` 通过 `self.backbone`、`self.st_module`、`self.mano_head`、`self.motion_module` 输出 pose/shape/trans。
   - 代码里 2D joints 可以通过 `self.project(j3d, ...)` 生成，即 `pred_keypoints_2d` 有可能只是模型自己的重投影，不可与同源投影比较并宣称得到了图像检验。
   - 图像特征中间张量可以通过明确的 hook/返回值接口进行缓存，先从最小跑通版本开始，不必重新训练 HaWoR。

3. `https://github.com/511205787/WestWorld/blob/main/models/WestWorld/westworld.py`
   - 使用 observation/action channels、structural embeddings、bin discretization next-token loss 等。
   - 模型代码复杂度与人体手 current-state restoration 任务并不完全匹配，**只借鉴相应结构思想**。

## 4. 数据接口 schema（我们自己定义，不是某数据集的原始格式）

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class HandFrame:
    timestamp_s: float
    handedness: str                 # "left" or "right"

    # Standardized world-space wrist pose.
    wrist_pos_world: np.ndarray     # (3,), meters
    wrist_rot_world: np.ndarray     # (3,3), proper rotation matrix

    # Local 3D hand joints excluding wrist/root.
    joints_local: np.ndarray        # (20,3), meters

    # Original visual measurement availability.
    raw_detected: bool              # before bbox interpolation
    joint_valid_mask: np.ndarray    # (20,), observed or missing, may be estimated
    camera_K: Optional[np.ndarray]  # (3,3)
    camera_T_world: Optional[np.ndarray]  # (4,4)

    # Optional training-only reference with separate provenance.
    gt_wrist_pos_world: Optional[np.ndarray]
    gt_wrist_rot_world: Optional[np.ndarray]
    gt_joints_local: Optional[np.ndarray]

    # Metadata for coordinate/time/GT provenance.
    sequence_id: str
    subject_id: str
    dataset_name: str
    gt_source: str                  # optical mocap / glove / device tracking / none
```

这是约定 schema 而不是已运行的 loader；不同数据集可能缺部分字段。`joint_valid_mask` 表示标注的可用性，不自动等同“模型准确度”；可靠性由训练的误差头预测。

最终训练张量可以例如：

- `state`: `[B,K,69]`（保存 6D wrist rotation + 20×3 wrist-local joints）；
- `current_state`: `[B,69]`；
- `raw_observation`: `[B,69]`；
- `obs_mask`: `[B,21]` 或 `[B,22]`，需要明确定义 root/各关节；
- `dt`: `[B,K]`；
- `history_quality`: `[B,K,num_parts]`；
- `visual_feature`: 特征图 `[B,C,H,W]` 或相应视频特征；
- `gt_state`: `[B,69]`（只在训练/评测时读）。

注意：采用 root+20 关节时，是否把 palm 作为单独 token 以及有无重叠关节，需要通过具体 skeleton mapping 确认，不要直接把“21 joints”与“21 non-root joints”混淆。

## 5. PyTorch 目录建议

```text
handloop/
  README.md
  configs/
    data.yaml
    twm.yaml
    corrector.yaml
  third_party/
    HaWoR/                  # Git submodule or pinned commit
  datasets/
    gt_adapters/
      hot3d.py
      dexcap.py
      egodex.py
    schemas.py
    canonical_hand.py
    timestamps.py
  models/
    twm.py
    corruption.py
    image_evidence.py
    corrector.py
    state_geometry.py
  scripts/
    extract_hawor.py
    verify_coordinates.py
    train_twm.py
    evaluate_twm.py
    train_corrector.py
    rollout_eval.py
  metrics/
    mpjpe.py
    wrist_pose.py
    visibility.py
  tests/
    test_geometry.py
    test_hand_flip.py
    test_pose_roundtrip.py
    test_timestamp_alignment.py
```

## 6. 第一轮必须跑的 Sanity Checks

- 单手 3D points 经 world→camera 投影，与视频图像真实可见位置接近；左右手 flip 对齐。
- wrist-relative state→world→local roundtrip numerical error 足够小；rotation matrices 正交、det 接近 +1。
- glove/mocap GT 与 HaWoR 无 GT 泄漏对齐；严格按照同步采集时间戳，不做每帧 GT 优化对齐。
- `clean GT -> TWM -> GT` 输出稳定；缺失/扰动数据修复的效果必须明显胜过简单最后一帧复制（在实际有区别的数据上）。
- 对所有输入都被标记 missing 时预测 uncertainty 应上升，不能产生虚假的 100% confidence。
- 视觉原始正确的点不能在修复后大量退化；覆盖“真 high confidence 错误”和突发转向测试。
- 短时回放测试：若关闭帧内 re-observation，N=2 与 N=1 不能轻率宣称区别来自新证据。
- 监督 vs 评测 GT 来源不同、训练前处理与推理前处理一致。
- 检查 legacy HaWoR infiller/temporal component 的 overlap；定量比较 `HaWoR original/full` 版本。

## 7. 最重要的初始评测表（实验前空模板）

| Model | W-MPJPE↓ | Wrist translation↓ | Visible joints↓ | Occluded joints↓ | Recovery after reappearance↓ |
|---|---:|---:|---:|---:|---:|
| HaWoR full | TBD | TBD | TBD | TBD | TBD |
| HaWoR raw (before infilling) | TBD | TBD | TBD | TBD | TBD |
| Raw + Kalman | TBD | TBD | TBD | TBD | TBD |
| Raw + Temporal Transformer | TBD | TBD | TBD | TBD | TBD |
| Raw + prediction-only TWM | TBD | TBD | TBD | TBD | TBD |
| Raw + restoration TWM | TBD | TBD | TBD | TBD | TBD |
| + evidence-guided iterative corrector | TBD | TBD | TBD | TBD | TBD |
| + confidence re-assessment / feedback (full) | TBD | TBD | TBD | TBD | TBD |

**所有数值待实验，绝不能从历史讨论里随意引用示例数字。**

## 8. 第一周交付目标（建议，而非已开始工作）

- 下载/pin HaWoR 代码与 checkpoint，核对 license、dependencies、最低 GPU；
- 核验一小份 HOT3D reference+camera 与 HaWoR 输出能否同 timestamp 对齐；
- 实现 skeleton state adapter + plotting sanity check；
- 构造 pure trajectory corruption synthetic batch 并训练一个简单 restoration baseline；
- 用真实 HaWoR prediction errors 试验恢复模型的有效性，和 Kalman/temporal baseline 对比。

只有得到真实结果，才开始给 CVPR Intro 和 abstract 写任何 numerical claim。
