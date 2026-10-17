# 九库 TWM：暂停10k并重跑30k

用户于2026-10-17要求“先暂停，改成三万步之后重新跑”。**本记录将此前10,000步预算替换为30,000步；数据与模型协议不变。**

截至纽约时间 **2026-10-17 01:37:31 EDT**，新run已实际更新到 **20/30,000**，checkpoint10及W&B在线服务端训练记录验收通过。[当前W&B](https://wandb.ai/black-ant/handloop-twm/runs/70f78bc5)。

## 旧实验完整保存

原`runs/twm_nine_sources_moe_transient_v3_b512_10k_20261017`通过run/STOP合作停止在 **816步**，并完成最后一次save。CPU加载验证model和Adam全部有限、两rank hash与真实模型字节一致；optimizer、rank RNG、sampler进度、best/latest、日志与配置都保留。旧torchrun已退出，GPU释放后才启动新run。旧W&B状态finished表示正常停止，**不代表10k已完成**：[旧run](https://wandb.ai/black-ant/handloop-twm/runs/fa17b1ef)。

## 本轮改动与执行选择

- 新配置：[30k config](../config/twm_nine_sources_moe_transient_v3_b512_30k.json)。仅steps 10000→30000、W&B name及fresh-30k标签变化；其余source config字段逐项一致。
- 采用全新run **from scratch**、全新Adam／W&B id，按完整30k cosine日程训练。用户要求是暂停并改30k重跑；fresh是为保持完整日程作出的执行选择。原trainer strict resume要求resolved config完全一致，未绕过检查将10k checkpoint套入改过的配置。
- warmup500、LR2e-4、weight decay0.01、每卡512／global1024、双卡、current restoration＋future60、99D、Mamba／MoE／uncertainty／噪声／损失／seed不变；beta10另存，conditioning=false。
- 九库集合SHA仍为`a1c09d67b3528c6f108a0d36825382a6054e8ae5865367a1431dba9085c2fe90`。原四库及五新增来源的文件／split／QC和窗口原样复用；未下载或重分数据。train2,037,891窗／val310,192窗。GT来源与名义时基边界见[九库准备](TWM_NINE_SOURCE_PRETRAIN_2026-10-16.md)。
- save每100步、首个10步另存，val每500步，best按原验证规则，不使用test。

## 实际启动与验收

Pod `jchen57-fineturn-tl4zr`；CUDA runtime为`/sciclone/data10/jchen57/Hand_Reconstruction/runtime/twm_mamba_torch210_v1/bin/python`。新代码独立worktree `code/twm_nine_sources_30k_20261017`，commit `e759bb1ced39052e16293a11c3236503f7e4cc8e`，保留旧170c347快照。新143个源／配置SHA已由登录端clean Git导出，并在无Git Pod逐字节验证；aggregate `9030761676bc9e7c7be4f25293e3d60f2f42b5f557d2c8ef05999f1d3002c4b7`。

新训练目录：`/sciclone/data10/jchen57/Hand_Reconstruction/runs/twm_nine_sources_moe_transient_v3_b512_30k_20261017`；torchrun PID 8106，后台独立于SSH运行。`status.json`／`metrics.jsonl`／`train.log`及weights保留远程。正常合作截止仍是 **纽约2026-10-20 01:10 EDT（UTC05:10）**，对应当前Pod安全时限，不因预算增加而越过allocation。写新run的STOP可合作保存停止。

已实际验证新metadata steps及target_steps均为30000、initialization=from scratch；九库实际采样>0；checkpoint同fd哈希／CPU加载、model与Adam finite、两rank一致；W&B API收到同row optimizer_step≥10和finite train/loss。[完整小型验收与来源](../reports/twm_nine_source_30k_restart_20261017.json)。启动验收不代表新预算已经完成或已提升下游精度。

代码、记忆与小型结果同步GitHub。旧／新数据、权重、缓存和原始日志均留在忽略的远程目录。
