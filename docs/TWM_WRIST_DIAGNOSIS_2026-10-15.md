# 腕位置为什么尚未有效校正：源码与冻结诊断

2026-10-15纽约01:37。用户要求分析D34腕位置未改善的原因。本轮没有改训练模型/目标/噪声、没有优化步骤；下述原因排序是待验证解释，不是新的用户设计决定。正式10k训练继续，05:37:47UTC只读确认其仍在step4,450训练。

## 目前能确认什么

4k完整固定验证显示ordinary/noisy腕7.158→7.180mm，而root/finger分别改善40.7%/51.6%；该结论见[中途效果](TWM_B512_PROGRESS_2026-10-15.md)。本次单独加载原run的latest **step4,400**，在四库固定val各8窗、共32窗做扰动与梯度检查。它不是4k的1024窗评测，也不代表新全量效果。

使用`git archive 7d53fa4`的scripts/src/config独立源码快照，逐项匹配run_metadata的source SHA；checkpoint CPU完整模型hash与两rank保存hash一致。模型/全部buffer在诊断前后逐项相等，未建optimizer。GPU1额外峰值allocated3.877GB，正式训练未暂停。主产物为[小型诊断JSON](../reports/twm_wrist_diagnosis_20261015.json)，[诊断源码](../scripts/diagnose_twm_wrist.py)。

| 检查 | 实际结果与解释 |
|---|---|
| 当前平移更新 | BF16平均更新范数0.562mm，各坐标跨样本std为0.206/0.267/0.301mm；并非完全零输出，但相对数毫米误差很小 |
| 当前位置外加偏移 | 同时对raw/candidate平移各轴加±10mm，保持历史/旋转不变；残差对偏移的**平均有符号斜率**为−0.001745/−0.000501/−0.003408，即平均只反向响应0.01745/0.00501/0.03408mm。理想完全抵消的斜率是−1，完全保留输入是0；不把各样本有符号均值当成绝对响应范数 |
| FP32对照 | 平均更新0.564mm，±10mm残差斜率−0.001736/−0.000465/−0.003397，仍接近0。直接换FP32推理没有恢复输入依赖；不能据此排除训练阶段精度的影响 |
| 1mm输入在投影层的可见性 | BF16 root投影仅约23.7%–28.6%通道发生数值变化，FP32为100%；FP32通道可见仍未形成有效纠正，说明不只是推理舍入 |
| 位置读出是否断梯度 | current restoration对translation_head.weight梯度范数0.41095，非零finite；head权重范数0.01118。输出残差/监督路径未断开 |

这32窗的腕误差candidate5.534mm、model5.431mm，不能拿它替代4k完整验证；因此也不能说模型在所有样本上绝对无法校正。固定25%历史hold＋75%当前为5.216mm，固定50%当前＋50%过去5帧线性外推为5.367mm，提示输入中有部分可用互补信息；报告保留所有固定比例/外推结果，没有在GT上优化融合权重。这是小样本探索，不证明该基线普遍优于模型。

## 原因假设及证据强度

**1. 位置输入的数值尺度是优先嫌疑。** `_state_tokens`把米单位p直接放进root token，随后与rotation6D、mask、joint/hand/time/structure embeddings共同编码；没有按典型位移尺度缩放p。当前7mm错误只对应0.007，rotation6D及embedding通常是更大的数量级。loss中除以0.05m、输出head乘0.1m，均不等于输入做了归一化。实测位置扰动响应弱、head梯度非零，与“已学到对位置近乎不敏感的小更新”相符，但尚未通过重新训练对照证明尺度是主因。

线性投影原则上可以自行学习缩放，所以缺少显式归一化不构成数学上的不可学习证明。源码：[输入token](../src/handloop/models/twm.py:130)、[当前读出](../src/handloop/models/probabilistic_twm.py:347)、[几何loss](../scripts/train_probabilistic_twm.py:60)。AMP对部分运算选择低精度，参照[PyTorch2.10官方说明](https://docs.pytorch.org/docs/2.10/amp.html)；本项目的特定通道变化率来自上述实测，不能由AMP文档直接推定。

**2. mixed_v2使“先保留当前”成为容易的策略。** 20%全净＋20%当前clean/near、历史受损，明确要求不要覆盖准确当前。5%strong中只有一半覆盖wrist/root，约2.5%窗口是强腕异常；availability和unknown quality不提供oracle噪声等级。其余ordinary位置误差较轻，历史有时比当前更差。没有足够的位置残差识别能力时，零初始化残差很容易维持近似identity策略。这不意味着应删除干净样本，而是需要同时学会识别什么时候该改；目前这一点尚未做到。[active profile](../config/twm_corruption_mixed_v2.json)

**3. 长预测直接牵动当前位置头，存在多任务牵制。** 未来位置从`current_state.p + cumsum(v*dt)`积分，future loss可直接改变当前均值头，不仅训练未来decoder。在同一32窗eval-mode诊断里，使用真实配置权重后的梯度为：

| loss项 | 配置权重 | translation head梯度范数 | 与current restoration梯度cosine |
|---|---|---|---|
| current restoration | 1 | 0.41095 | 1 |
| current NLL/DoF | 0.05 | 0.30744 | +0.4335 |
| forecast NLL/DoF | 0.3 | 0.23207 | −0.2040 |
| best-trajectory geometry | 0.2 | 0.45876 | +0.0443 |
| velocity | 0.05 | 0 | 无直接translation-head路径 |

未来梯度量级与当前项相当，forecast NLL在这批样本有负cosine，未来geometry近乎正交；这支持存在牵制的可能，**不是长期梯度冲突/因果主因证明**。需要不同batch/训练期及单因素训练对照。不能由总loss下降推出当前位置均值变好；NLL还会随尺度/concentration改变。[目标与积分](../scripts/train_probabilistic_twm.py:88)

**4. 纯轨迹恢复有可辨识边界，不能解释掉所有失败。** 若观测为`z_t=p_t+b`且b为整段共同常偏移，任意常量c给出的`p'_t=p_t+c, b'=b-c`产生同一观测和时间差分。在无绝对位置约束的一般情形，轨迹本身不能唯一恢复该偏移。当前history-derived anchor进一步移除了输入的共同平移；真实SLAM/camera漂移可能需要视觉、相机定位或外部锚点。

这个条件性边界不是“当前所有腕错误都无解”：mixed_v2还有独立噪声和干净历史，相关方向OU也不等于整段固定偏移。上面的简单融合说明至少这批样本存在部分可用信息。手指的局部姿态统计依赖和整体world腕位置的约束不同，不能期望同一结构先验自动解决两者。

## 没有证据支持的简化解释

- 不是没有位置loss：位置Smooth-L1独立求均值后与root/finger项相加，并非把3个平移数直接放进99维均分后忽略。
- 不是位置head没有接到输出或被固定为零：`base.p+delta.p`及非零梯度、非零更新均已检查。
- FP32推理没有明显改善，不能把全部原因归为BF16；同样，单次FP32推理不能证明BF16训练无影响。
- 不能保证更多步数/更大batch会自行解决；也没有证据需要增加模型容量。

## 优先建议，尚未执行

首先独立验证位置输入尺度：仅在feature编码处将相对p除以由train集确定的特征尺度（例如0.05m只是候选），外部99D、GT和输出仍以米为单位。保留raw/candidate/history分开和干净样本，检查毫米级输入敏感性、腕误差及准确输入退化。

然后用同预算current-only对照判断future是否牵制；也可只切断future经当前均值的直达梯度作为更窄对照，但共享encoder仍接收future梯度，不能声称完全隔离。上述修改要单因素验证，不能同时改尺度/噪声/loss后归因。当前10k run保留作对照，本轮没有实施这些训练改动。

远程目录：`/sciclone/data10/jchen57/Hand_Reconstruction/evaluations/twm_wrist_diagnosis_20261015/`，含`training_code_7d53fa4/`、`probe_v1.json`及`diagnose_source_v1.py`；训练checkpoint仍在原run，不新增或上传权重。诊断模型SHA=`81d9ea33e7c99d2616041e2b798d3b7d5a8528894587821455eb6f0b6dc0364c`，数据manifest SHA见JSON。
