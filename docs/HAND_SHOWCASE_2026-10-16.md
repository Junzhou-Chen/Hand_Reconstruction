# 手部渲染升级与精选 test 案例

日期：纽约 2026-10-16。用户要求改善既有重建手的渲染，并寻找本方法明显优于 baseline 的 test 案例。旧版 [v2 demo](HAND_OVERLAY_DEMO_2026-10-16.md) 保留；本轮明确按测试误差挑选优势案例，不再沿用旧版“不看误差、各手取最长片段”的展示选样规则。

## 产物与渲染

本机：`outputs/demos/dexycb_showcase_20261016/render_v2/`；远程：`/sciclone/data10/jchen57/Hand_Reconstruction/demos/dexycb_showcase_20261016/render_v2/`。初次平滑材质试版 `render_v1/` 作为历史保留。

- `comparison.mp4`：三段各 45 帧，共 135 帧／4.5 秒，原 RGB／官方 HaWoR／HandLoop 三列，1920×948，原速 30 fps。
- `comparison_slow_0p5x.mp4`：明确标注的 0.5× 慢放，269 帧／8.967 秒，重复已有帧，不插值姿态。
- `original.mp4`、`handloop_overlay.mp4`：同样三个片段的原 RGB 与单独 HandLoop 叠加。
- `comparison_segment{1,2,3}_middle.png` 与 `case{1,2,3}_{first,middle,last}.png`：逐段预览／验收帧。

新默认材质使用面积加权顶点法线、透视正确的法线插值、三点柔光和适度高光、2× supersampling；默认移除黑色骨架与白色关节点。不移动、细分或平滑预测顶点。两模型使用相同蓝色材质、alpha=0.76、灯光、采样和观看尺度。

最后检查首末帧发现开放 MANO 腕口的空洞明显，v2 增加仅用于显示的腕口端面：按原 faces 唯一边界环、只连接原有边界顶点生成三角面，使用独立端面法线；实测两手原 1538 面均追加 14 面，不增加或移动 778 个预测顶点、不改变原手部表面法线、也不进入任何指标。显示 mesh 因而比原预测 mesh 多一个端面，不能将其描述为原拓扑完全未改；`--open-wrist` 保留原开放腕口，legacy 始终开放。

完整场景等比置于 300 像素高的上下文区域；下方 440 像素高的手部细节是用共同裁切的显示内参 `A_display @ K` 直接重新光栅化，不把原图中已经画小的 mesh 再放大。抗锯齿只下采样 mesh 前景与 coverage，编码前帧缓冲的 mesh 外背景保留输入 RGB 字节；H264 本身是有损编码。此处高清只针对预测 mesh，原始 RGB 没有新增纹理。

## 如何选样

固定已有完整 DexYCB s0 test 的官方 `baseline.raw` 和验证集预选 `best4200.final`；未重新选择 checkpoint、训练模型或运行神经网络。先扫描全部 **1280 视角／78768 有效状态**的既有 3D 指标，再按左右手和四种误差排序组成候选池，仅解码其中 **159 视角／9768 状态**；有 **54 个视角**满足持续窗口条件。这里的 2D 排名仅覆盖该候选池，不能称为全 test 的最优 2D 窗口。

窗口优先 45 连续帧，候选连续段不足时最低 30 帧；本次最终三段均为 45 帧。要求绝对 MPJPE 至少下降 10 mm 且 20%，投影关节误差至少下降 2 px 且 25%，两项各自至少 80% 的帧改善，HandLoop 投影均值≤10 px。排序优先持续 2D 改善，并兼顾腕对齐后的局部误差；最后保留左右手及不同被试／物理录制，第三段优先不同抓取目标。

| 案例 | 原帧 ID（含端点） | 抓取目标 ID | 绝对 MPJPE mm，HaWoR→HandLoop | 投影误差 px | 腕对齐 MPJPE mm | PA-MPJPE mm |
|---|---|---|---|---|---|---|
| 右手／subject-04 | 9–53 | 17 | 60.441→27.231 | 10.277→4.541 | 25.273→11.002 | 7.908→6.719 |
| 左手／subject-06 | 29–73 | 17 | 83.625→17.725 | 9.019→3.357 | 14.295→7.748 | 4.906→3.578 |
| 左手／subject-07 | 23–67 | 11 | 142.091→32.117 | 11.727→5.466 | 31.651→13.506 | 7.022→5.537 |

三段投影误差均 **45/45 帧改善**；绝对 MPJPE 改善帧比例分别 88.89%、95.56%、93.33%。这些是整段均值，画面标签也明确为 segment mean。覆盖三个被试、三个物理录制、两相机、两种抓取目标；前两段抓取目标相同，桌面物体集合不同。

源键：

1. `20200903-subject-04/20200903_113236/932122062010`
2. `20200918-subject-06/20200918_120717/836212060125`
3. `20200928-subject-07/20200928_153953/836212060125`

**这是按 test 真值主动挑选的定性优势案例，不代表随机样本或整库平均。** 完整测试结果仍以 [三模型完整报告](DEXYCB_FULL_TEST_RESULTS_2026-10-16.md) 为准；本次不更新 benchmark，不把收益单独归因于 TWM。

## 几何与协议边界

使用各模型保存的 99D 动态状态和自身预测 β、对应左右手 MANO、真实全图 K 和原 `camera_to_world`。没有 PA、GT 腕移位、旋转／尺度校正或 GT shape 替换；PA 指标只用于诊断，不用于显示。GT 来源是 DexYCB supplied 21 joints／fitted MANO，不是独立 MoCap。

沿用已完成测试的 GT crop、固定相机参考系、16 帧双向 HaWoR 块和 32 帧反馈 reset；不是新增任意视频检测／头戴 SLAM 部署。每段有明显 case 标题，不把三个录制拼成同一个连续场景。30 Hz 来自原 frame ID 名义帧率。

mesh 使用自遮挡 z-buffer；没有真实物体深度遮挡，属于 see-through overlay，画面已有说明。可选 `--show-skeleton` 是 X-ray 诊断，默认不绘制。

## 验证与复现

所有几何解码、光栅化和视频编码均在 `ssh ds` 的 `jchen57-fineturn-tl4zr` Pod；复用已验 Torch2.10/cu130 与原完整测试代码／MANO 资产，限制 CPU 线程为 2。没有在本机运行模型或处理数据。

远程合成三角形合同检查确认：编码前 mesh 外背景字节一致、输入顶点不变、legacy overlay 与旧函数逐像素相同、直接 detail 投影与显示 affine 投影最大差为 0 px。最终四视频均完整解码，帧数／尺寸／30 fps／faststart 通过；选择阶段与渲染阶段的投影均值误差小于 0.001 px。v2 对 v1 逐帧投影误差变化为 0 px，原 RGB 身份及 `original.mp4` SHA 也完全一致。最终视频解码、投影数值交叉核验及本机复制检查见 [小型报告](../reports/hand_showcase_20261016.json)；详细验收 JSON、完整选择与候选排名留在上述 demo 目录。

实现：[选样脚本](../scripts/select_hand_demo_cases.py)、[渲染脚本](../scripts/render_hand_overlay_demo.py)。远程代码 snapshot 为 `demo_code/showcase_20261016/`，依赖原 `code/dexycb_full_test_20261016/{src,scripts}`、`runtime/hawor_extra/site-packages` 和 `runtime/demo_video_tools_v1`。原始 RGB、视频、逐帧预测、MANO 资产及完整日志不入 Git。

默认新版显示可显式传入 `--render-style polished --alpha .76 --supersample 2 --full-context-height 300 --zoom-height 440`。旧版重现需 `--render-style legacy --alpha .5 --zoom-height 320 --zoom-mode per-frame`；旧选样文件另行指定，不能混同两轮不同片段。两脚本完整 CLI 与保存的 SHA／选择规则足以在新输出目录复现本轮，不覆盖历史产物。
