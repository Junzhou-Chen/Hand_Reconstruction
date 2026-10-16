# 视觉联合训练对齐 HaWoR 数据来源

**后续用户决定：** 以实际效果和测试隔离为主，不要求完整复现作者；先用已验收DexYCB＋HOT3D的63652独立RGB／73524手状态跑5000步，其余下载／预处理并行继续。**不再要求凑齐四库才开训**；当前执行见[两库5k协议](VISUAL_DEX_HOT_GEOMETRY_5K_2026-10-16.md)。下文保留前轮四库准备过程。

日期：2026-10-16。用户最新澄清：“也没有必要完全小规模的那个DXYCB训练集……baseline它训练的时候用哪些数据集，我们跟baseline一样就好了。”

本条**替代**刚准备的“恢复原1128帧 DexYCB-only＋5000步”数据选择；该配置没有启动。保留用户已选的直接几何监督、双主干低学习率全量微调及约4000–5000步预算，本轮仍选5000。原混源 run 已在100步保存停止，旧best不覆盖。

## 官方依据与可复现边界

新核对 [HaWoR全文§8.1](https://arxiv.org/html/2501.02973v1)：其视觉 hand motion estimation network 使用：

| 数据来源 | 作者采用的训练帧数 | 本轮职责 |
|---|---:|---|
| HOT3D | 573K | 头戴RGB、动态相机、world参考 |
| ARCTIC | 165K | 双手操控物体的RGB和姿态参考 |
| DexYCB | 169K | RGB、固定相机与遮挡参考 |
| HO3D | 66K | 手物遮挡RGB和姿态参考 |

取整相加973K，作者称约1M。不是完整库大小，也不是公开的采样概率。不能按这个比例宣称复现作者 sampler。

[官方README](https://github.com/ThunderVVV/HaWoR#training)本轮仍写训练代码稍后发布。论文没有给出逐帧 manifest、DexYCB训练setup、ARCTIC视角选择、HO3D版本与完整筛选规则。当前能对齐四库来源、采用可审计的公开划分并记录差异；**不能承诺恢复作者完全相同的约1M帧**。

HaWoR原视觉训练250K iterations、冻结WiLoR ViT、总batch32，这些是作者协议，**本轮不覆盖用户已选的全量低LR联合适配与5000总更新**。仍从官方HaWoR checkpoint＋D37 best重新初始化，而非重训整个WiLoR基础模型。

论文§8.2的motion infiller只用HOT3D，与上述四库视觉网络不同。本轮TWM纯运动预训练／回放仍为已验收OakInk2、HOT3D、独立UmeTrack、DexYCB，D37不重训。不会把ARCTIC／HO3D视觉拟合参考自动加入pure-motion GT池。

## 当前资源核查与已执行部分

计算Pod：`jchen57-fineturn-tl4zr`；持久根目录：`/sciclone/data10/jchen57/Hand_Reconstruction`。本机只改源码和文档，数据下载、预处理及数值检查均远程。

- **DexYCB：** 官方13包已下载完整，127611176595 bytes。扩大版视觉准备已完成，train800个单视角录制／50861有效状态／1980窗，保留原val577；这是单相机与valid帧处理范围，**不是作者169K**。下一步应按公开s0／预留val、8视角的物理录制划分核验覆盖，再定最终manifest；不能用其他视角把同一录制移入不同split。
- **HOT3D：** 官方train包下载完整，但为BOP版150帧clips。原RGB pilot只选99个clips；已在07:29:36UTC后台启动已下载Aria全部1167个clips的准备，源代码固定`86f25ec842043bce2bcbdab17dbe539bacc3eef3`。沿原pure-motion划分，train86父录制／1019clips／305700待筛选手状态，internal-val13父录制／148clips／44400待筛选状态；这些是预处理前metadata计数，不是最终有效样本。两个手状态不等于两张独立RGB。固定27个HaWoR评测录制全排除；99个已下载可用父录制也不等于作者110个完整train视频。
- **ARCTIC：** 官方manifest已有，实际状态`waiting_credentials`。核查项目默认私有文件、data10配置与Pod环境，未配置ARCTIC账号。用户已回复“已有账号，我来配置私有文件”；建议路径`private/arctic_credentials.json`、权限600，只含username/password，不在聊天发送密码。MANO账号不能替代[ARCTIC单独账号](https://arctic.is.tue.mpg.de/)；待私有配置到位先验官方小包再排队图像。这是运行凭据缺失，不是用户批准缺失。
- **HO3D：** [作者README](https://github.com/shreyashampali/ho3d)给出v2、v3匿名公开OneDrive链接。浏览器已核v2目录中的`HO3D_v2.zip`约34.5GB及README：55个train sequences、13个evaluation sequences、显式`train.txt`。v2作为与论文66K接近的候选先准备；**这是工程推断，作者未确认版本**。新增远程匿名官方UI下载脚本，隔离安装Playwright1.56.0，不读取浏览器cookies、密码或用第三方镜像。获取包不代表来源adapter／坐标／99D契约已经验收。

HOT全量预处理输出：`prepared/visual_hot3d_aria_all_downloaded_v4_20261016`；过程与log位于`evaluations/hawor_training_source_alignment_20261016/`。使用原完整下载SHA收据＋size/stat，未重算每个大TAR SHA；重新编码RGB与每个参考文件仍由预处理写身份／几何检查。原99clip与扩大Dex产物全部保留。

首次HOT启动遗漏既有`hawor_extra/site-packages`路径，在导入cv2时退出、未创建输出；07:34:42UTC已按原验收环境补齐PYTHONPATH重启，首次失败与重试log／receipt都保留。HO3D首次浏览器启动缺系统库；浏览器包与缺失库采用独立下载目录，未修改训练环境或全局系统，不把启动失败写成已下载数据。

HO3D包同时含evaluation，后续只从train.txt选取训练，并按physical sequence／共享多机位录制分组留内部val。公开evaluation几何标签不进入训练。v2/v3不得混版本后仍称同一源。源OpenGL→目标相机、MANO full-pose/mean／translation、关节顺序与beta必须用官方可视化源码、源投影与FK交叉验证；没有测量的rotation不填成有效GT。ARCTIC亦需核源个性化模板、左右手与camera/world变换，不能用beta0伪造同一GT。两库adapter目前未完成，不能仅新增source字符串便开训。

## 本轮配置与启动条件

准备配置：`config/visual_joint_geometry_hawor_sources_5k.json`。5000**总optimizer更新**＝原3:1下3750视觉＋1250motion replay，双GPU、每卡8×32 RGB帧／motion32，HaWoR1e-6／TWM1e-5／插件1e-4、warmup20、余弦到10%、两轮／history30／原噪声均不变。保留NLL1e-4＋raw J0绝对距离/0.05m＋final自身预测beta的有效绝对joint距离/0.05m；初值不是已证最优。

新配置显式要求四个visual train sources，训练入口在CUDA／模型构建前核对来源集合；只给Dex或Dex＋HOT时应直接报告缺失，不把它们悄悄称为四库训练。该检查只核来源身份，不替代各库完整几何／split验收。

`eval_batches=24`目前只是验证计算预算，**不再意味着覆盖新四库完整val**；最终各源内部validation清单及覆盖数待数据准备完成后固定并披露，不根据test选择。原Dex577及旧test1015保留作历史受控比较；正式四库新训练不据其test筛数据、超参或best。

**当前状态：数据对齐和准备中，5000步训练未启动。** ARCTIC／HO3D获取与两库adapter仍需完成；完整四库RGB／参考／左右手／物理录制边界验收后再正式启动并检查双rank梯度、checkpoint和在线日志。新数据准备进度不等于模型收益。

## 07:44UTC获取实测更新

用户随后直接提供账号截图，已按明确授权将凭据经加密SSH管道写入远程`private/arctic_credentials.json`，实测权限600；不在源码、日志、报告或Git保存账号／密码值。本机OCR明文只经进程管道传输，没有另存本地凭据文件。前两次子进程SSH入口关闭，未写入私有文件；随后普通SSH管道成功。

ARCTIC下载器07:43:43UTC启动，先取得并匹配官方`splits_json.zip` SHA，再并行下载。07:44快照`running`、verified2／downloading3／pending338／failed0，已写535675617B；动态known bytes不是完整库精确总量。获取代码仍固定旧已审计下载器，3workers，343个cropped／raw／meta／splits包；数据下载不意味着任意train标签已通过本项目adapter。

HOT补齐运行环境后的重试已实际320/1167 clips、635个手轨迹record；最终有效帧数仍待全部完成及manifest落盘。原失败日志保留。

来源检查已在Pod上对**真实已完成Dex＋HOT manifest**实测：显式要求这两源通过、历史配置无要求通过、要求HaWoR四源明确拒绝缺失ARCTIC／HO3D；无模型前向或optimizer更新。新增代码本机仅AST／JSON／diff静态检查。

07:45:59UTC快照：ARCTIC仍running、verified3／failed0、已写1025251120B；HO3D行选择修复后已进入真实`downloading`、已写1913417728B，尚未完成ZIP CRC／SHA验收。HOT已340/1167 clips。小型快照见[`reports/hawor_visual_training_source_alignment_20261016.json`](../reports/hawor_visual_training_source_alignment_20261016.json)；完整下载收据／失败log留远程。不把已写字节称为完整可用训练数据。

## 用户自行查询与“完整训练”的含义

用户随后表示等下载完成再来，本轮仅提供查询与说明，不启动训练或调整worker。只读helper同时查看三个下载状态、校验文件数／失败数、worker是否仍活着与HOT RGB预处理进度；不读取私有账号配置。

在本机执行：

```bash
ssh ds 'python3 /sciclone/data10/jchen57/Hand_Reconstruction/download_runtime/check_hawor_data_progress.py'
```

ARCTIC和HO3D各自`phase=complete`才是下载与本轮校验完成；`checking`仍在校验，`running/downloading`仍下载，`failed`需检查。HOT RGB预处理是另一阶段，单独看complete／总clips。退出查询不会停止后台进程。

“作者选帧清单”指具体source／sequence／view／frame ID，不是要求本项目额外抽小样本。HaWoR公开训练帧总数不能唯一确定这个列表；可以按公开训练划分组织完整的有效训练池，无须凑成作者973K。该解释不等于当前已覆盖四库全量：Dex RGB仍单视角、HOT为已下载BOP clips，缺口与有效性剔除仍须逐源披露／补齐；val/test不能混入。5000是optimizer预算，与数据池是否完整是两个概念，不保证每个样本都遍历一次。
