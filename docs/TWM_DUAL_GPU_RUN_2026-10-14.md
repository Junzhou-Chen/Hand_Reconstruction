# 四库TWM双GPU与大batch迁移

2026-10-14，用户要求使用现有两张GPU一起训练并提高batch。此修改仅改变执行和batch配置：纯轨迹当前恢复、完整99D、β另存/conditioning暂关、六层结构化Mamba、源GT与split不变。现有单卡训练在26,346步保存退出，旧run保留；新阶段明确迁移模型、AdamW状态与绝对优化步数，重新建立rank采样/RNG，不冒称逐样本精确续跑。

正式双卡run于19:47:53 UTC（纽约15:47）启动，源码`10aa98327eeaa3ff0afb49eef1e6ff73879c7d22`；19:52:33 UTC已实际训练到26,450步（本阶段104 updates），两卡都参与梯度训练，尚未完成原50,000步。配置/资源/完整性摘要见[双卡验收JSON](../reports/twm_dual_gpu_20261014.json)。来源数据和旧单卡阶段见[四库协议](TWM_FOUR_SOURCE_RUN_2026-10-14.md)。

## 显存与吞吐选择

单卡基线19:27:10 UTC：batch64、约4.228step/s，训练样本吞吐约270.6/s；GPU0显存5,338MiB/97,887MiB、利用率93%，GPU1只有2MiB/0%。两卡同NUMA，经NODE PCIe桥互联，没有NVLink；双向P2P读写可用。Pod仍12CPU/128GiB/2GPU，未改或续期用户Job。host memory使用约127.54GiB中file cache约123.03GiB、匿名RSS约3.24GiB，不能当作127GiB训练张量；OOM kill为0，failcnt82是历史计数。

GPU1实际探针源码a5b4723，用四库各一个真实训练窗口重复到候选batch，载入单卡step25,000的模型权重，fresh AdamW；每档warmup2＋5次测量，BF16网络/FP32几何，loss/梯度/model/optimizer均finite。计时包括corruption、forward/backward/AdamW，排除loader和DDP；这不是质量、稳态正式吞吐或最大安全batch结论。

| 每卡batch | 峰值allocated GiB | 峰值reserved GiB | 样本/s（单卡探针） |
|---:|---:|---:|---:|
| 128 | 8.63 | 8.82 | 314.0 |
| 256 | 17.13 | 17.47 | 297.4 |
| 512 | 34.16 | 34.79 | 240.7 |
| 768 | 51.19 | 52.14 | 237.3 |
| 1024 | 68.22 | 69.50 | 229.2 |

五档均运行通过，没有OOM。选用**每卡512、全局batch1024**，显存显著提高且保留余量，避免更大档的吞吐下降；这是本次工程折中，不是最优batch结论。不能按两卡显存相加来估计单样本容量，每卡各有一份完整模型。

配置为[twm_four_sources_structured_mamba_ddp.json](../config/twm_four_sources_structured_mamba_ddp.json)，其中`batch_size`明确表示global batch，`world_size=2`；`eval_batch_size=64`、`eval_batches=16`保留原验证窗口集合与corruption批次分割。仍用原50,000优化步及LR2e-4/warmup500/cosine绝对步调度，不线性放大LR。旧阶段26,346×64=1,686,144样本；剩余23,654×1024=24,221,696样本，较原剩余单卡预算增加16倍。各阶段按步数或epoch直接比较并不等价；应使用固定分库验证与累计样本数。

## DDP执行契约

[train_twm_ddp.py](../scripts/train_twm_ddp.py)使用torchrun两进程，各绑定一张卡，实际NCCL all-reduce探针通过才继续。各rank独立seed的有放回等库/库内窗口采样，允许偶然重复，不换成按全库窗口数采样。等local batch和local mean loss经DDP梯度平均对应global mean，不额外除world；backward后clip，再AdamW。

rank0单独用未包装模型做固定分库验证，其他rank等广播；best决策先广播，所有rank共同收集RNG，仅rank0原子保存plain model/AdamW及每rank的corruption/CPU/CUDA/Python/NumPy/sampler RNG。每次保存时对完整model state的keys/dtype/shape/bytes做SHA256，要求两份模型逐字节一致。参数总和只是日志监控，不代替这个检查。sampler块内未消费indices/loader cursor仍未完整记录，不宣称exact replay。

只有显式`--migrate-from`允许原config的batch/device/thread执行字段变化，模型/任务/manifest SHA须相同；旧sampler.num_samples不适合新batch，所以迁移重建rank流。DDP同world resume须config完全一致，并恢复rank状态。handler只设stop flag，每步固定collective边界归约共同停止并保存；[ds_train_twm_ddp.py](../scripts/ds_train_twm_ddp.py)的stop写STOP文件，resume显式归档旧STOP，直接入口遇到旧STOP则拒绝零步停机。首次迁移/恢复后的训练日志即保存updated checkpoint，之后每500步保存、每1000步验证。

依据：[PyTorch2.10 DDP](https://docs.pytorch.org/docs/2.10/generated/torch.nn.parallel.DistributedDataParallel.html)、[分布式通信](https://docs.pytorch.org/docs/2.10/distributed.html)。没有新增模型模块、物理损失或视觉输入。

## 路径与历史边界

共同远程根目录：`/sciclone/data10/jchen57/Hand_Reconstruction`；专属runtime仍`runtime/twm_mamba_torch210_v1/bin/python`，zhou未改。

| 产物 | 相对路径 |
|---|---|
| 四库collection | `prepared/twm_four_sources_99d_v1/manifest.json` |
| 保留的单卡源run | `runs/twm_four_sources_structured_mamba_20261014` |
| GPU1 batch探针 | `audits/twm_batch_gpu1_20261014_v1/{receipt.json,process.json,benchmark.log}` |
| 双卡smoke | `runs/twm_ddp_smoke_20261014_v1` |
| smoke配置/独立验收 | `audits/twm_ddp_smoke_20261014_v1` |
| 新正式双卡run | `runs/twm_four_sources_structured_mamba_ddp_20261014` |

一次启动前断言把`/proc/2599`存在误认为旧训练仍活跃，导致smoke配置未生成、launcher报missing config。实际旧状态已经stopped，PID2599为cmdline空的zombie且无GPU进程；核清后新smoke启动。没有再次给旧进程发信号、删除Job或覆盖已保存权重。

本轮验证仅覆盖实际双卡训练、同步、保存和恢复；合成恢复的几何指标仍是source fitted MANO腕/旋转，不是独立joint/mesh GT、真实HaWoR效果或物理正确性保证。

## 实际双卡验收

smoke使用global128/per-card64，源码0ad45f9，从26,346到26,446共100次真实四库更新；最终模型两rank的完整state SHA一致。首阶段CPU检查163个模型浮点tensor与AdamW递归499个浮点值/tensor finite，plain model keys、2份rank状态；这不是大batch质量试验。随后10aa983从26,446真实resume，仅rank1在19:45:34 UTC收到SIGTERM；两rank共同保存退出于26,498（52 updates），CPU hash等于两rank及replica audit、Adam step正确，两卡没有遗留活跃进程。smoke权重不接入正式训练，正式仍从原单卡26,346迁移。

正式launcher PID2963，GPU0/rank0 PID2968、GPU1/rank1 PID2969。parent checkpoint SHA为`09a10b8f800cac5e8336da3a14cf69cfce9228ff8674be84182a84102e1bf27f`，新metadata明确`single-GPU batch/DDP migration`，模型、optimizer和绝对step保留。19:52:33 UTC step26,450；总训练样本106,496=104×1024，各rank53,248，四库每rank计数均为正且归约一致。该阶段约0.426step/s、435.9sample/s，loss0.020830/grad0.158670有限；相对旧阶段约270.6sample/s是早期实测吞吐差异，含缓存/loader影响，不是质量收益或稳态扩展比。19:52:47 UTC两卡瞬时util均100%，nvidia-smi显存38,172/36,686MiB；瞬时资源快照不是全程平均。

首updated latest checkpoint为step26,350（4次更新），约76.21MB；CPU读取166个model keys/163个模型浮点tensor及489个AdamW tensor全finite，Adam step26,350，151个decay＋12个no-decay参数，plain keys及rank_states=[0,1]正确。两rank、CPU重算与replica audit的模型SHA均为`90e1dda2f5905c473ab6f3134c7758fd6532d25ab408139fc4dfda3fff7d22b9`。该checkpoint检查时刻与后续26,450进度分开；保存后文件继续按配置推进，不能把此旧hash应用到后来的latest。

独立formal receipt为data10项目根`audits/twm_ddp_formal_20261014/launch_receipt.json`，SHA`7f8b72fb5972c7a9d0440b34b9fe9003f9c4159f63b844d321cfa7a2c70d3d59`。首次CPU检查本身通过，但包装器误调用Pod不存在的git导致receipt未写；移除该依赖，用执行metadata版本，并只读重载一次后补存receipt。没有重跑网络或改训练进程；源码版本在ds登录节点核查。大batch下validation保持初始/定期验证语义，不将initial结果贴到后来step当作重新评测。
