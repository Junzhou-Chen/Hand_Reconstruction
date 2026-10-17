# 原视频手部重建叠加Demo

**后续显示更新：** 用户随后要求更好的手部材质和优势 test 案例，已另做[平滑渲染与精选案例](HAND_SHOWCASE_2026-10-16.md)。本页 v2 仍保留为按最长片段选样的原始版本，两轮选样口径不同。

日期：纽约2026-10-16。用户希望直观看到预测手与视频真人手的重合，已实际制作并完成视觉及视频解码验收。此页记录真实模型输出可视化，不是AI生成动作或GT重建动画。

## 交付视频

本机忽略目录：`outputs/demos/dexycb_overlay_20261016/v2/`；远程：`/sciclone/data10/jchen57/Hand_Reconstruction/demos/dexycb_overlay_20261016/render_v2/`。

| 文件 | 内容 | 已验实际帧数／时长 |
|---|---|---|
| comparison.mp4 | 原RGB／官方HaWoR／HandLoop best4200三列，上排原图、下排共同手部放大 | 148帧，4.933秒，30fps，1920×922 |
| comparison_slow_0p5x.mp4 | 上述对比的明确命名0.5倍慢放 | 295帧，9.833秒，30fps，1920×922；重复帧，无新姿态插值 |
| original.mp4 | 两个原RGB录制片段顺序串接 | 148帧，4.933秒，640×480 |
| handloop_overlay.mp4 | best4200自身mesh与骨架直接叠加原RGB | 148帧，4.933秒，640×480 |

四个视频均为H264/yuv420p、faststart，无音轨；在Pod用FFmpeg完整解码，非仅检查容器／文件存在。本机下载后重新核对SHA。截图`comparison_segment1_middle.png`／`comparison_segment2_middle.png`和contact_sheet.jpg已人工视觉复核；视频已请求在Codex当前聊天文件面板打开。

## 输入与模型来源

在完整DexYCB s0 held-out测试源中，按每侧**最长无缺帧片段、录制ID字典序破同分**选取一段，选样未读取模型误差：

1. 右手：`20200820-subject-03/20200820_135810/836212060125`。
2. 左手：`20200928-subject-07/20200928_153732/836212060125`。

每段原frame ID0–73，共74连续帧，按Dex名义30Hz还原；帧时间来自frame_id/30，不是假称实测timestamp。画面明确Segment1/2与手别，不把两段串接冒称同一连续录制。

预测直接复用已完成的[完整三模型测试](DEXYCB_FULL_TEST_RESULTS_2026-10-16.md)：官方baseline.raw与验证预选best4200.final，各自`predictions.npz`中的99D与自身β。图像、参考、预测、frame ID与源manifest SHA严格核对。没有重新训练或运行HaWoR/TWM神经网络；仅在Pod GPU上batch256进行MANO几何解码、CPU双面深度缓冲渲染和视频编码。

## 投影、显示及解释边界

- 从99D腕J0／16个旋转解码MANO778顶点及Dex21骨架；左右手使用对应模型与正确tip索引，prediction left-shape fix开启。
- 用原始camera_to_world逆变换与真实K投影到640×480原图，**没有PA、GT腕移位、旋转／尺度对齐或GT β**。
- 两个方法使用相同青色、alpha0.5、光照和原图。下排是三列共同的逐帧输入bbox放大，顶排同一细框标区域；只改变观看窗口，不改变模型几何。
- 本版是有标定、GT crop的受控测试视频；不能称为任意无标定视频／自动检测／头戴SLAM的部署demo。保留原16帧双向块／32帧反馈reset测试协议。
- 自遮挡使用透视深度缓冲；真实物体没有额外深度遮挡，显示为透视叠加，画面已标注。原RGB未抹除，遮挡处骨架仍可能可见。

投影诊断仅用于这两个片段，与3D benchmark分开；21个joint均在画内、778顶点全为正深度。对源joint_3d以真实K投影，平均逐关节点像素距离为：

| 片段 | 官方baseline | best4200 |
|---|---:|---:|
| 右手74帧 | 7.084px | 5.479px |
| 左手74帧 | 6.735px | 4.228px |

这不是segmentation IoU／整库新结果，也不能用两段视觉观感证明完整3D／动态精度。

## 显示修订与复现

v1采用整段input bbox并集作为固定zoom，实际检查发现右手运动范围大时反而缩小显示。v1与源码snapshot留在远程；v2默认逐帧共同input bbox，且保留`--zoom-mode segment`重现旧显示。已验证v1/v2逐帧投影误差变化0，`original.mp4`及`handloop_overlay.mp4` SHA逐字相同，只有三列比较的观看窗口改变。

实现：[render_hand_overlay_demo.py](../scripts/render_hand_overlay_demo.py)。运行只允许remote Pod。固定环境：Torch2.10/cu130 runtime、原完整测试src和hawor_extra；FFmpeg通过独立`runtime/demo_video_tools_v1`的imageio-ffmpeg提供，不改训练环境。包获取收据与安装log留在demo父目录，FFmpeg版本和renderer／模型／参考／产物SHA见[小型报告](../reports/hand_overlay_demo_20261016.json)。

```bash
TASK_ROOT=/sciclone/data10/jchen57/Hand_Reconstruction
export PYTHONPATH="$TASK_ROOT/code/dexycb_full_test_20261016/src:$TASK_ROOT/runtime/hawor_extra/site-packages:$TASK_ROOT/runtime/demo_video_tools_v1"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"$TASK_ROOT/runtime/twm_mamba_torch210_v1/bin/python"   "$TASK_ROOT/demo_code/overlay_20261016/scripts/render_hand_overlay_demo.py"   --root "$TASK_ROOT"   --source-root "$TASK_ROOT/prepared/visual_dexycb_s0_full_test_20261016"   --evaluation-root "$TASK_ROOT/evaluations/dexycb_s0_full_three_models_20261016"   --selection "$TASK_ROOT/demos/dexycb_overlay_20261016/selection.json"   --mano-models "$TASK_ROOT/model_assets/mano_v1_2/prepared_v1/derived"   --output "$TASK_ROOT/demos/dexycb_overlay_20261016/render_fresh"   --device cuda:0 --decode-batch-size 256 --cpu-threads 2 --zoom-mode per-frame
```

selection.json为固定上述两段的源manifest相对路径。需要使用新输出目录，脚本拒绝覆盖已有完整视频。原视频、帧预览、MANO资产和逐帧预测不入Git；代码、`.gitignore`、本页及小型报告可同步。
