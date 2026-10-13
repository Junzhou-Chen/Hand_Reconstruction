# ds / SciClone Kubernetes服务器操作记录

核验日期：2026-10-13。用户说明hongfei GPU已占用，指定改用`ssh ds`并参考旧项目运维；随后明确授权使用其昨天申请的`jchen57-debug-mpd5p`。本轮完成文档/配置/挂载核验和极小CUDA检查，没有创建/删除Job、修改VLM_KD或安装依赖，未运行模型、数据处理或训练。

来源：用户提供的`kubectl_server_request_bundle_20260930.tar.gz`（只读tar内容）；ds上`/sciclone/home/jchen57/project/VLM_KD/{how_to_use.md,MIGRATION.md,config/project.env,conda.sh,script/*.sh}`中的运维部分；实时kubectl/NVIDIA/Python检查。未读取旧项目method。原包、完整配置、凭据、内部NFS地址与其他用户任务信息不提交Git。

## 1. 登录与当前可复用资源

`ssh ds`实际登录`cm.geo.sciclone.wm.edu`，这是Kubernetes登录入口。该主机有kubectl/envsubst/git和资源查询工具，没有nvidia-smi；GPU计算在Pod内运行。当前context为`default-context`，namespace为`jchen57`，UID/GID与旧模板一致。

```bash
ssh ds
kubectl get pods -n jchen57 -o wide
kubectl exec -it jchen57-debug-mpd5p -n jchen57 -- /bin/bash
```

用户已明确授权复用此Pod，当前不需要另申请资源；不要照旧文档删除这个现有Job，也不要直接SSH到GPU节点。Pod名称可能重建后变化，后续运行先核状态/名称。

| 实时核验项 | 2026-10-13结果 |
|---|---|
| Job / Pod | `jchen57-debug` / `jchen57-debug-mpd5p`，Running |
| 节点 | `dsci-chen-01.sciclone.wm.edu` |
| 已分配资源 | 12 CPU、128Gi RAM、2 GPU |
| GPU | 两张RTX PRO6000 Blackwell Max-Q，每张97887MiB，约96GiB |
| 检查时占用 | 两张均约2MiB、0% utilization，仅为当时快照 |
| 镜像 | `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel` |
| 启动时间 | 2026-10-12 16:37:28 UTC |
| activeDeadline / TTL | 259200秒（3天）；结束后TTL3600秒 |
| 按配置推算到期 | 2026-10-15 16:37:28 UTC，纽约12:37:28左右 |

节点资源表当时6/6 GPU已分配，其中2张属于此Pod；“无新增空闲配额”不妨碍使用用户已申请的两张。`getnodestats/getgpuusage`实际读取定期更新的日志，不是实时GPU利用率；应核刷新时间，Pod内nvidia-smi用于实际占用判断。resourcequota查询返回Forbidden，未获配额信息，不绕过权限；此不影响访问已授权Pod。

## 2. Python环境与CUDA验证

新`kubectl exec`进程不会继承Job初始化shell的Conda激活。实际默认Python为`/opt/conda/bin/python`，而不是旧项目的zhou环境。

| 环境 | 实测版本 | 两张卡的小型CUDA检查 |
|---|---|---|
| `/opt/conda`默认 | Python3.11.11；Torch2.6.0+cu126，CUDA12.6；arch列表不含sm_120 | 两卡均报`no kernel image is available for execution on the device` |
| `/sciclone/home/jchen57/.conda/envs/zhou` | Torch2.12.0+cu130，CUDA13.0；arch含sm_120 | 两卡均通过，结果140.0 |

检查在Pod内分别调用两套环境的Python：每卡仅建立8个float32元素，计算`(arange(8)**2).sum()`并同步；预期140。时间2026-10-13 22:30:43 UTC，本项目代码版本当时`1bf812e`，使用内联检查代码，无模型/dataset/GT/split。这证明该CUDA操作可用，不代表MANO、HaWoR或TWM全部依赖已验。

后续进入Pod时应明确激活环境，或直接指定已验证的Python：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /sciclone/home/jchen57/.conda/envs/zhou
which python
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

也可调用`/sciclone/home/jchen57/.conda/envs/zhou/bin/python`。本轮没有改造现有环境；TWM依赖与多卡通信尚未验。旧项目`conda.sh`会加载VLM_KD数据/cache键并创建目录，学习其机制即可，TWM不应直接沿用旧项目全部配置。

## 3. 持久存储与项目同步

登录节点与Pod共享相同NFS路径：用户home、data10、scr10、scr20、scr30。它们已确认在Pod内可见且可写；scr目录的长期清理/配额政策未核，不能因NFS就承诺永久保存。

- 代码建议放`/sciclone/home/jchen57/project/Hand_Reconstruction`。
- 数据、权重、checkpoint、输出与日志建议放`/sciclone/data10/jchen57/Hand_Reconstruction`下分目录。
- 上述两个本项目路径是建议，本轮未创建；已确认home项目目录不存在。
- `/dev/shm`为64Gi memory-backed emptyDir，适合临时缓存/共享内存，Pod删除后消失；使用会计入内存，不放唯一checkpoint。
- `df`显示home所在文件系统当时98%已用，data10约73%；这是共享文件系统状态，不是用户个人可用配额。

Pod默认工作目录`/workspace`不在上述项目持久路径。代码同步可在登录节点完成：那里有git；Pod默认PATH未找到git。共享home使Pod直接看到登录节点更新后的checkout。继续本机编辑→GitHub→远程持久checkout→Pod执行，数据/模型不上传GitHub。当前尚未clone本项目到ds。

## 4. 将来需要新Job时的参考流程

当前用户Pod继续复用。仅在另行需要新任务时，借鉴以下流程，而不是执行附件中的旧删除命令：

1. 用`getnodestats/getgpuusage`查看申请状态；准备本项目独立Job名、label、namespace、路径、挂载和环境。
2. 先用模板渲染器`--render`检查，确认项目变量已展开；再提交渲染结果。
3. 查看jobs/pods、describe调度原因、logs启动问题；Running后exec进入，明确环境，再运行具体任务。
4. GPU申请即使容器sleep也持续保留资源；退出exec shell不释放Job。只清理本项目新建且不再使用的任务，不操作本轮用户预申请Job。

旧模板使用nodeSelector＋对应NoSchedule toleration；requests/limits均显式设置CPU/RAM/GPU。渲染器只替换VLM_KD变量，保留容器运行时变量。`how_to_use.md`残留直接`kubectl apply -f debug.yaml`的旧写法；当前README/MIGRATION要求先渲染，Kubernetes不会展开shell模板。

## 5. 附件、远程配置与实际Job不是同一快照

| 来源 | Debug | Eval |
|---|---|---|
| 2026-09-30附件配置 | 6CPU / 128Gi / 2GPU | 12CPU / 400Gi / 3GPU |
| 2026-10-13 VLM_KD project.env | 12CPU / 128Gi / 2GPU | 12CPU / 400Gi / 4GPU |
| 当前已运行Job | 12CPU / 128Gi / 2GPU | 未创建/未运行本轮Eval Job |

模板默认不代表TWM需求或剩余资源。两份旧YAML都主要初始化环境后sleep，不会自动训练或评测；包内也没有完整conda初始化文件，不能单独复制包就保证新项目可运行。新任务需适配自己的目录/loader必需键/环境/运行命令与持久日志，并考虑3天上限和checkpoint续跑。
