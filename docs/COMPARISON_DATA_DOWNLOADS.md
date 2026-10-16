# DexYCB与ARCTIC下载

**2026-10-16最新范围：** 用户要求视觉联合训练使用与HaWoR相同的HOT3D／ARCTIC／DexYCB／HO3D四库，替代下方10月14日“ARCTIC暂缓”的当前执行安排。ARCTIC仍实测缺账号，不能自动登录或宣称已取得；已请求服务器私有配置路径，其余数据继续准备。见[四库对齐及实际缺口](HAWOR_ALIGNED_VISUAL_TRAINING_2026-10-16.md)。下文下载时点与旧Pod是历史记录。

记录日期：2026-10-13（纽约）。用户在HaWoR数据复核后明确要求下载DexYCB与ARCTIC。沿D20操作方式，在`jchen57-debug-mpd5p`（12CPU/128Gi）运行，数据存远程data10；没有解压、跑模型或训练。

**2026-10-14安排更新：** 用户回复尚未注册ARCTIC，当前先下载DexYCB。ARCTIC暂缓，现有清单和脚本保留，不继续要求账号或自动注册/启动。下方ARCTIC配置流程仅供用户以后明确恢复时使用。

**随后完成核验：** Dex于2026-10-14 04:15:58 UTC（纽约00:15）完成，13包全verified、failed0、127611176595B与manifest一致，进程正常退出。下面running数字是历史快照，不是当前状态，不重复启动。后续四库小样本标注审计见[适用性记录](TWM_FOUR_DATASETS_AUDIT_2026-10-14.md)，没有全量预处理/99D转换。

最新进度快照2026-10-14 04:00:17 UTC（纽约00:00）：DexPID329仍alive/running，已写85,506,118,574 / 127,611,176,595 B（85.51/127.61GB，含partial），verified7/checking2/downloading1/pending3/failed0。本轮两次SSH查询被跳板连接关闭，之后只读重试成功；没有修改SSH配置或下载进程，原因未确定。

## 当前状态与范围

代码`20c64c2`于2026-10-14 03:33:43 UTC（纽约10-13 23:33）启动获取；检查脚本版本`8f5d9f1`随后同步并在Pod执行。初始状态快照03:34:03 UTC：

| 数据 | 范围 | 大小 | 状态 |
|---|---|---:|---|
| DexYCB | 10个完整subject包＋calibration＋models＋bop，共13包 | 127,611,176,595 B（127.61GB / 118.85GiB） | PID329，3workers，已写约1.62GB含partial，verified1/failed0，后台运行 |
| ARCTIC | 339个官方cropped images包＋raw_seqs/meta/splits_json/splits，共343包 | 官网约134GB，精确压缩bytes待认证后获取 | waiting_credentials，进程已退出，没有数据包下载 |

数据目录：

- `/sciclone/data10/jchen57/Hand_Reconstruction/datasets/dexycb`
- `/sciclone/data10/jchen57/Hand_Reconstruction/datasets/arctic`

退出SSH或聊天不停止Dex后台进程；Pod到期会停止，文件可续传。当前Job预计纽约10月15日12:37到期，不新建或删除Job。下载阶段不使用GPU。

## 查询与续跑

在ds登录入口执行，helper会进入计算Pod查询：

```bash
ssh ds
cd /sciclone/home/jchen57/project/Hand_Reconstruction
python3 scripts/ds_downloads.py status --datasets dexycb arctic
```

每库有`download_manifest.json`、`download_status.json`、`download_process.json`、`download.log`与`download_receipts.json`。status约20秒更新，`checking`表示正在完整压缩包校验，不是传输停滞；`present_bytes`含未完成文件。`phase=complete`且failed0才是清单完成。新两库均须显式`--datasets`，避免默认重扫先前三库。

Dex中断后保留partial并续跑：

```bash
python3 scripts/ds_downloads.py start --datasets dexycb --workers 3
```

## ARCTIC账号配置

官方链接匿名请求会返回HTTP200登录HTML，不能当成ZIP。服务器已检查本项目配置、相关环境变量与旧VLM配置的键存在性，没有ARCTIC账号；未输出任何其他凭据。已向用户请求已有账号或服务器私有配置路径。当前下载器只记录等待，不反复请求大包。

该问题已得到D22回复：用户尚未注册、先Dex。当前没有需要用户立即完成的ARCTIC账号步骤。

在服务器用模板建立私有文件，并填入已有ARCTIC账号：

```bash
cp config/arctic_credentials.example.json config/arctic_credentials.local.json
chmod 600 config/arctic_credentials.local.json
```

用服务器编辑器填写`username`、`password`，不要发到聊天。文件受`.gitignore`的`*.local.json`排除。随后启动：

```bash
python3 scripts/ds_downloads.py start --datasets arctic --workers 3
```

已有另一私有JSON文件时加`--credentials /absolute/private/path.json`，同样只含上述两字段；也支持Pod中的ARCTIC_USERNAME/PASSWORD环境变量。没有账号则需先在[ARCTIC官网](https://arctic.is.tue.mpg.de/)注册；这些dataset包仅需要ARCTIC账号，MANO/SMPL-X账户针对各自模型资产，不属于本轮数据下载前置。[官方数据说明](https://github.com/zc-alexfan/arctic/blob/49f3eb4b7f01663172576357bb3d95695d7a7689/docs/data/README.md)

账号配置后先校验`splits_json.zip`小包再排队图像；401/403或登录HTML立即终止认证尝试。认证后的官方实际Range行为尚未实测；续传若返回200会覆盖重下，206必须起点正确，完整包必须匹配官方SHA256。

## 来源与完整性

**DexYCB**使用[官网公开分包](https://dex-ycb.github.io/)的Google Drive IDs；精确bytes来自对应公开Drive viewer metadata，官网“119G”为近似GiB口径。只选分包，不同时下载119G整包。工具gdown6.4.2隔离安装于`/sciclone/data10/jchen57/Hand_Reconstruction/download_runtime`，没有更改zhou环境，使用无cookies公开下载及正常大文件确认。quota/权限失败报告受阻，不换第三方镜像绕限制。

官网未公布作者MD5/SHA。每包按源metadata大小、完整gzip EOF CRC、TAR首header验收，自算SHA256写receipt；不能称匹配作者SHA。已有文件按receipt SHA复核，partial由gdown续传。模型包是物体几何，不是MANO许可模型。[gdown固定源码](https://github.com/wkentaro/gdown/blob/v6.4.2/gdown/download.py)

**ARCTIC**清单与SHA表固定官方repo`49f3eb4b7f01663172576357bb3d95695d7a7689`，下载POST使用用户账号并验证TLS，拒绝HTML，stream写入、Range续传与SHA检查。不获取额外SMPL-X/MANO资产、baseline weights、features或背景图。[下载实现](https://github.com/zc-alexfan/arctic/blob/49f3eb4b7f01663172576357bb3d95695d7a7689/scripts_data/download_data.py)、[官方checksum](https://github.com/zc-alexfan/arctic/blob/49f3eb4b7f01663172576357bb3d95695d7a7689/bash/assets/checksum.json)

本轮ARCTIC选择cropped版：ego view保留全视野并缩至原0.3倍，其他8view按物体GT框裁剪；评测须披露，不能称原2K视频完全同协议。full原图约649GB，脚本支持`download_comparison_data.py plan arctic --images full`另建scope，但未选取/启动。数据划分仍需保持，公开validation适于额外比较，test GT隐藏。[crop源码](https://github.com/zc-alexfan/arctic/blob/49f3eb4b7f01663172576357bb3d95695d7a7689/scripts_data/crop_images.py)

Pod内7个synthetic HTTP/压缩包检查通过：206续传、200覆盖、HTML拒绝、完整partial直接验hash完成、正常gzip TAR、截断和坏CRC拒绝。Dex calibration真实下载14464B且完成验收；ARCTIC凭据缺失，所以不能把fixture检查写成真实登录/数据下载已成功。原包未解压，没有99D或实验结果。
