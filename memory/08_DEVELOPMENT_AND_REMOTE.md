# 开发、GitHub同步和远程实验约定

记录日期：2026-10-13。来源：用户本轮明确指令；本文件属于已采纳工作约定，不是助手建议。

## GitHub与版本管理

- 远程：`https://github.com/Junzhou-Chen/Hand_Reconstruction.git`。
- remote名称：`origin`；主分支：`main`。
- 将代码、论文源文件、项目记忆、配置模板、数据获取/处理脚本和小型实验摘要及时commit并push；用户已经授权本项目的正常同步，无需每次重复请求。
- 不上传数据集、大型下载包、MANO/SMPL-X等受许可模型资产、权重/checkpoint、生成视频、缓存和完整运行日志；规则统一维护在根目录`.gitignore`。
- 不提交凭据、私钥、token和机器私有配置。Git不支持按文件大小自动忽略，提交前仍须检查暂存清单；`.gitignore`不会自动取消已跟踪文件。
- 小型手工整理的指标/表格放在`reports/`并跟踪，原始结果放在忽略目录；论文图和源文件不做全局图像/PDF忽略。
- 正常push，不force-push覆盖远程历史；有远程新提交时先核查并保留已有工作。

## 机器分工

| 用途 | 位置 |
|---|---|
| 编辑代码、论文、文档、Git同步、静态检查 | 本机 `/Users/junzhou/Project/Visual_Hand_Reconstruction` |
| 初步实验服务器 | `ssh hongfei` |
| 用户指定远程根目录 | `/data/hongfei/zhou` |
| 本项目远程checkout | `/data/hongfei/zhou/Hand_Reconstruction` |
| 数据、权重、缓存、完整日志 | 远程存储，不进入Git |

训练、模型推理、数据预处理、性能评测及依赖GPU的sanity checks全部远程运行。本机不承担实验，即便当前可运行一个小模型也不改变此约定。远程项目目录名沿用GitHub仓库名，是本次实施选择。

建议将本项目的数据/checkpoint/outputs/logs分别放在远程checkout的`data/`、`checkpoints/`、`outputs/`、`logs/`，这些目录已经忽略。后续若共享大数据，可使用`/data/hongfei/zhou`下独立数据目录并通过配置引用。

## 运行与复现

每次实验记录本地已推送commit、远程实际commit、配置、数据版本/split、命令、设备、随机种子与产物路径。服务器长期任务采用可恢复会话或调度器；具体环境、GPU资源和调度方式在首次实验前检查，不预先占GPU或启动训练。

常规同步流程：本机整理并提交→push `origin/main`→远程确认工作区状态→fast-forward更新→远程运行。若远程修改了代码，先保存并提交这些改动，不用reset或rsync覆盖。

## 本轮连接核验

- GitHub `git ls-remote`成功，初始化前远程无refs。
- `ssh hongfei`连接成功，`/data/hongfei/zhou`存在；可找到`git`和`nvidia-smi`。
- GitHub push与远程checkout的最终结果由本轮执行后补记；连接成功不等于已跑实验。
