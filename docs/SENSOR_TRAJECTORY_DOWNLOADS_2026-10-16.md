# HumanTouch轨迹下载与DexWild暂缓记录

日期：纽约2026-10-16夜间（UTC10-17）。用户先授权下载DexWild trajectory并询问HumanTouch，随后明确“DexWild这个先放一放”，将调查焦点改为可直接下载、适配99D的人手运动数据。本页记录已执行的获取和失败；新选库结果见[直接人手运动数据](TWM_DIRECT_HUMAN_MOTION_DATA_2026-10-16.md)。

## 运行位置与当前状态

全部下载、校验及HDF5小样本提取均在`ds`的Pod `jchen57-fineturn-tl4zr`（namespace `jchen57`）执行；登录节点只转发kubectl。持久根目录`/sciclone/data10/jchen57/Hand_Reconstruction`，数据本体不回本机、不上传GitHub。

- **HumanTouch继续后台下载**：`datasets/humantouch_trajectories_v1`，固定HF revision`81822a10cac5def2638db546eaeced581d71a8b5`。初始8workers遇到HTTP429后，已保存旧状态，停旧进程、重新校验已有文件并改为4workers／最多4 resolver请求每秒；读取剩余quota与reset动态降速，429关闭响应并全局退避，不消耗单文件普通错误重试次数。续传PID6739，03:24:29UTC启动；不是重新下载已通过hash的文件。
- **DexWild已按用户要求暂缓，不自动续跑**：`datasets/dexwild_human_trajectories_v1`，status=`stopped_by_user`、无活跃下载进程。批量任务在用户叫停前已经因HTTPRangeError退出，五task均0个完整episode；同一阶段HumanTouch出现429，但未记录各Dex任务的HTTP码，**不能将其根因直接断言为429**。已有单episode smoke独立保存在audits目录，不等于全量下载成功。

最新进度、启动源码SHA、原失败与新状态分别保存在[小型快照](../reports/sensor_trajectory_download_status_20261016.json)。重启后`present_bytes`／verified从重新扫描开始恢复，旧文件不会因计数暂降而丢失；下载中和已校验字节亦不同。此前基于35MB/s的1–2h估算未包含后续限流，不能继续作为完成承诺。

03:28UTC快照：HumanTouch存活、running，present17,417,580,448B／110,417,888,793B，5,418个文件verified、4个进行中、failed0，限流续传后的429计数0；原16个429失败已重新纳入本轮扫描／续传。不是全量完成，也未开始99D转换。

## HumanTouch下载范围

[官方HF](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch)公开且非gated。按固定实际文件树选取：README、`X001–X010/data/chunk-*/episode_*.parquet`、`X*/meta/**/*.json|jsonl`，不选独立RGB／depth视频或上传缓存。

| 内容 | 文件数 | 字节 |
|---|---:|---:|
| 实际公开的episode parquet | 17,576 | 108,985,108,170 |
| JSON／JSONL元数据及标定说明 | 17,669 | 1,432,776,863 |
| README | 1 | 3,760 |
| 合计 | **35,246** | **110,417,888,793（110.42GB）** |

Parquet包含每手25点的位置／方向／validity、head／wrist位姿、时间戳等，**也内嵌触觉数值字段**；这不是纯99D文件，但无需下载视频。保留源文件利于后续标定审计，未进行MANO拟合或入训。

源缺档：X008有744条episode metadata、仅711个parquet，33条未公开parquet已记录为source gap；不是本下载器失败或需要无限重试的文件。X005 info的722与实际tree723也不能混作可用量。已有parquet均有对应episode metadata；元数据累计小时数不等于QC后净连续时长。

原下载器已具备HF LFS SHA256／Git blob SHA1完整校验、尺寸检查、`.partial` Range续传、坏partial隔离、验后rename及20s状态。新增HTTP限速只通过可选callback注入，旧调用默认不变。远程四项恢复／错误hash测试及429退避关闭恢复检查通过。

## DexWild实际公开内容与已完成检查

[固定官方发布](https://huggingface.co/datasets/boardd/dexwild-dataset/tree/df5cd775f985e79036ff23ad579abbe7752b8df1)的人体五task为87个分卷，共1,817,035,653,120B。每task拼接为未压缩TAR，第一个member是单个`human_<task>_data.hdf5`；不能照README的采集目录图假定存在可直接单取的`*_manus_full.pkl`。

已实现有界HTTP Range拼接fileobj，严格验206／Content-Range／总size／响应长度，不会因200忽略Range就读全TB。源part LFS hash保留，但**没有读取完整part，故不能声称整卷SHA256校验通过**。提取的episode另存local SHA256与逐数组回读收据。只选MANUS、tracker、zed pose／ts、timesteps和metadata，排除camera／PNG／JPG及LEAP机器人动作。

真实pour首episode验收：

- 621个源episode中固定首个`ep_0000`；MANUS为`[552,71]float64`，71=`timestamp＋10×(xyz＋xyzw)`，**不是完整25点**。官方[71维解码](https://github.com/dexwild/dexwild/blob/main/dexwild_ros2/src/retargeting/retargeting/leap_v2_ik.py#L306)与[short导出](https://github.com/dexwild/dexwild/blob/main/dexwild_ros2/src/glove/glove/read_and_send_zmq.py#L173)支持十点解释；具体版本、骨点对应仍需几何验收。
- 有多版8D腕轨迹、原zed tracker、zed pose／时间戳及timesteps；只原样保存，未把`world`命名认作已验证world GT。腕来自ZED fiducial cube与SLAM链，不是MANUS手指之外另一个已证明独立精度的腕传感器。
- 输出HDF5 **558,096B**，数值逻辑量534,344B；含源目录读取、块缓存边界等的网络量43,515,904B／171请求，约66s。这个首样本开销不能外推整库下载ETA，也不是已下载所有raw MANUS。
- 小样本保存在`audits/sensor_trajectory_20261017/dexwild_smoke/pour/ep_0000.h5`及receipt；Range跨卷／窗口／seek／EOF、错误HTTP拒绝及6,672B真实HDF5 fixture检查通过。批量五task随后失败，未完成全量功能验收。

独立h5py下载runtime为`runtimes/sensor_trajectory_download`（h5py3.16.0、继承zhou基础依赖），没有向训练环境安装h5py。源码部署于持久`tools/sensor_trajectory_download_20261017`，与GitHub同步的源码SHA在快照中记录。

## 查询与后续续传

查看当前HumanTouch状态：

```bash
ssh ds
kubectl exec -it jchen57-fineturn-tl4zr -n jchen57 -- /bin/bash
cat /sciclone/data10/jchen57/Hand_Reconstruction/datasets/humantouch_trajectories_v1/download_status.json
cat /sciclone/data10/jchen57/Hand_Reconstruction/datasets/humantouch_trajectories_v1/http_rate_limit_status.json
```

进程若已退出，确认没有同目录活跃下载器后，HumanTouch可用同一命令恢复：

```bash
/sciclone/home/jchen57/.conda/envs/zhou/bin/python -u \
  /sciclone/data10/jchen57/Hand_Reconstruction/tools/sensor_trajectory_download_20261017/download_humantouch_trajectories.py \
  download --output /sciclone/data10/jchen57/Hand_Reconstruction/datasets/humantouch_trajectories_v1 --workers 4
```

不要因为HumanTouch继续而重启DexWild；后者已明确暂缓。两库下载均`training_ready=false`；现full17组motion trainer没有native部分监督接口，不能将十点／25点直接当完整99D参数GT。没有启动新预训练或修改现有模型／评测。
