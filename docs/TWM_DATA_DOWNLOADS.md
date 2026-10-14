# TWM数据下载与检查

日期：2026-10-13。用户要求下载OakInk2、HOT3D train、UmeTrack，并明确进入其计算Pod下载；脚本检查通过、后台任务跑起来后结束对话，由用户后续检查。本轮只做数据获取，没有解码轨迹、生成99D或训练。

**完成更新：** 用户报告完成后，已只读核验远程三个`download_status.json`：phase均为complete，全部4116个文件verified、failed=0，字节数与清单一致，进程正常退出。Oak完成于2026-10-13 23:35:43 UTC（纽约19:35），Ume于10-14 00:20:58 UTC（纽约10-13 20:20），HOT于10-14 01:22:17 UTC（纽约10-13 21:22）。这是下载器完成时的hash验收记录，本轮没有重新读取404GB做全量hash，也未解压/转换。

随后用户D21追加DexYCB/ARCTIC；新两库需`--datasets dexycb arctic`查询，账号与实际状态见[对比数据下载](COMPARISON_DATA_DOWNLOADS.md)。本页默认三库均已完成，不再重复启动。

## 运行位置与范围

登录入口`ssh ds`仅调用kubectl；下载和校验在`jchen57-debug-mpd5p`（namespace `jchen57`，12CPU/128Gi）执行。当时三个独立后台进程各4个下载worker，不使用GPU；当前三库已正常完成，无需重新启动。若以后另有下载，退出SSH不停止进程，Pod终止会停止进程。

- 代码：`/sciclone/home/jchen57/project/Hand_Reconstruction`。
- 数据根：`/sciclone/data10/jchen57/Hand_Reconstruction/datasets`，下设`oakink2/`、`hot3d/`、`umetrack/`。
- Python：`/sciclone/home/jchen57/.conda/envs/zhou/bin/python`，已有requests，无新增安装。
- 本次下载源码版本：`853d2da`；从GitHub clone到ds并fast-forward同步。首次启动：2026-10-13 22:57:17 UTC，即纽约18:57左右。

| 数据 | 实际下载范围 | 文件数（含元数据） | 字节 / 十进制GB |
|---|---|---:|---:|
| OakInk2 | 全部627份姿态标注＋program元数据＋TaMF四份split；不下载图像/objects | 634 | 36,673,901,351 / 36.67GB |
| HOT3D | 官方train Aria/Quest3，先按原录制排除HaWoR评测来源；2455个mixed TAR＋小型元数据 | 2461 | 257,066,644,565 / 257.07GB |
| UmeTrack | HF 2024格式的1020个real train TAR＋README；暂不取synthetic或test | 1021 | 110,513,346,839 / 110.51GB |

合计404,253,892,755 B，约404.25GB。HOT3D和UmeTrack的官方TAR同时包含图像和标注；下载完整包以保留官方hash校验，TWM后续只消费轨迹。尚未做解压，因此这些不是转换后存储量。

启动后22:58:42 UTC快照：三进程均存活，Oak已写1.43GB/verified30，HOT1.70GB/verified19，Ume1.36GB/verified10，三库failed均0。已写量含partial，后续以实时status为准；没有把该快照当全量完成。

## 后续检查与续跑

在登录入口运行以下命令，脚本会进入Pod查询：

```bash
ssh ds
cd /sciclone/home/jchen57/project/Hand_Reconstruction
python3 scripts/ds_downloads.py status
```

每库目录保存：

- `download_manifest.json`：固定源revision、文件大小/官方hash、URL、选取范围和排除项。
- `download_status.json`：约每20秒更新，包含phase、已写字节、verified/failed数和错误摘要。
- `download_process.json`：PID、Pod、启动时间、worker数。
- `download.log`：后台原始进度/错误日志。
- `*.partial`：未完成文件。仅尺寸和hash校验通过后改为正式文件名；校验坏文件另存`.corrupt-*`。

`process_alive=true`结合持续更新的status说明进程仍在运行；`phase=complete`且`failed=0`表示清单文件全部通过校验。`present_bytes`包含未完成文件，不等同于已校验字节，重启时随扫描逐步恢复。若Pod结束导致状态时间停滞，以进程状态为准。

失败/中断后保留数据目录，重新启动会跳过已校验文件并续传partial：

```bash
python3 scripts/ds_downloads.py start
```

如原Pod已经到期，先取得新的授权Pod名称，再显式指定：

```bash
python3 scripts/ds_downloads.py start --pod NEW_POD_NAME
python3 scripts/ds_downloads.py status --pod NEW_POD_NAME
```

新Pod仍须挂载相同home/data10并能使用该Python。现有Job约2026-10-15 16:37 UTC（纽约12:37）到期；不会因进程后台运行延长Job，持久文件可供续跑。脚本不创建或删除Job。

## 数据版本、split与验收边界

- [OakInk2公开数据](https://huggingface.co/datasets/kelvin34501/OakInk-v2)固定`21705616140d726607027e70d58b7837f442ffd8`。[TaMF split](https://github.com/oakink/OakInk2-TaMF/tree/4cb9c461b92c032715374bd252c7179c70a14bde/asset/split)固定`4cb9c461b92c032715374bd252c7179c70a14bde`，train431/val39/test157；脚本验证记录互斥且覆盖全部627。下载全标注不表示val/test可用于训练，也不证明subject disjoint。
- [HOT3D BOP clips](https://huggingface.co/datasets/bop-benchmark/hot3d)固定`30fe9674782f32e1e5edba98476b6ff4300132c5`。脚本验证train目录clip IDs与`clip_splits.json`一致、device与`clip_definitions.json`一致，再按[HaWoR官方评测命令](https://github.com/ThunderVVV/HaWoR/blob/66c7d4108d58a716deccd192cb7645170cdc7bd7/README.md)的27个源录制ID排除。BOP版只出现其中25个，`P0011_72efb935`、`P0012_119de519`整版缺席；排除名单仍保留27个。实际移除349个官方train clips、37.21GB；原train2804→下载2455。其他目标benchmark交集仍待核，不称全面无泄漏。
- [UmeTrack HF公开训练包](https://huggingface.co/datasets/facebook/hand_tracking_challenge_umetrack)固定`19f6b2cad9eb17be4ebedfe4503203980a25e016`。real与synthetic的采集motion存在重复，首轮跳过1021个synthetic包；两组文件不严格一一对应，synthetic独有覆盖待查。原[UmeTrack_data](https://github.com/facebookresearch/UmeTrack_data)另有约3.27GB的旧版training纯JSON，但split/schema不同，本轮没有静默替换HF训练包。

固定revision避免续传期间源文件变化。大文件用HF API `lfs.sha256`校验，小HF文件校验Git blob SHA1，固定GitHub元数据保存SHA256；不把临时Xet CDN ETag当文件hash。原始数据、下载清单和日志都留在远程data10，不提交GitHub；源码和本文正常同步。

Pod内4项HTTP检查均通过：partial Range续传、服务器忽略Range时完整覆盖、坏partial隔离重下、错误hash禁止生成正式文件。实际源文件的访问、split与checksum还通过首批下载检查，完整下载结果应以后续status为准。校验只证明传输完整，native→99D、单位/FK/腕原点与标签质量尚未验收。
