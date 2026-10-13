# 产业数采团队与大规模手部数据补充调查

核验日期：2026-10-13。用户指出首轮清单覆盖数采团队不足，本轮据此扩查公司、厂商、数据平台和聚合发布。结论来自发布方网页、公开文件索引和schema；没有下载训练数据或运行实验。

**随后用户决定D15：优先无需申请、直接可下载，参考论文实际用法。** 本文保留申请开放/商业资源的调查事实，但Lightwheel等不再是当前开工候选；最新方法与数据角色见[完整方法框架](../docs/METHOD_FRAMEWORK.md)。

## 对首轮结论的修正

首轮偏重学术MoCap和已知glove数据，不能代表公开人手数据全景。SynData、Lightwheel、OOJU、OpenGraph、EgoVerse、Daimon以及BeingBeyond都需要纳入调查。

“采集很多”与“公开完整手部状态”之间存在几种情况：商业授权库；需申请的公开库；少量公开样例；只发布机器人retarget结果；只有RGB；多种旧数据的统一格式聚合。不能以此解释掉检索遗漏，也不能把不同状态都算成可直接预训练的小时数。

本次进一步按 **设备参考轨迹、视觉/设备估计、聚合数据、待访问验收** 分层。分层依据是标注来源与可用性，不是已经测得质量高低。

## 新增的设备数据与产业人手数据

### PsiBot / SynData：应优先核验的大规模外骨骼来源

[官方HF数据卡](https://huggingface.co/datasets/PsiBotAI/SynData)、[glove-origin目录](https://huggingface.co/datasets/PsiBotAI/SynData/tree/main/glove-origin)、[manifest](https://huggingface.co/datasets/PsiBotAI/SynData/blob/main/glove-origin/manifest.json)、[发布历史](https://huggingface.co/datasets/PsiBotAI/SynData/commits/main)。2026-05已有上传，HF总文件约29.3TB，glove-origin约6.55TB。

| 分区 | 官方数据卡clips | 能否当完整新增手轨迹 |
|---|---:|---|
| glove-origin | 95,383 | 最先验收；793 volumes，外骨骼来源，但导出状态已转换模型 |
| glove-with-tactile | 36,780 | 有手部与触觉；需核查与origin的重叠 |
| glove-rep-bg | 3,526 | 换背景版本；不能先认定为新增运动 |
| ego | 313,674 | 主要RGB/depth/IMU，不能全算带hand labels |

[官方数据卡](https://huggingface.co/datasets/PsiBotAI/SynData)及[renderer说明](https://huggingface.co/datasets/PsiBotAI/SynData/blob/main/tools/syndata_render/README.md)给出双腕 `(T,7)`、xyzw，双手各 `qpos(T,30)`、keypoints `(T,11,3)`，统一10fps。30关节是**retargeted ITW hand model**；11点是五指尖、五proximal-link点和wrist。应使用所附URDF/FK检查人体映射，不能直接称标准21-joint GT，也不能未经核实就认定它只是机器人action。

数据卡标注CC BY4.0，另有身份识别/监控等用途限制。`.zarr.tar`任务包可下，metadata可轻量单独读，尚未发现官方pose-only全集。10fps对快速动作恢复有限制；腕部world定义、单位和有效时长待样本验收。**这是上一轮实质性遗漏，优先级应提高到与DexWild/HumanTouch并列的接入调查，而不是立即将全部clip混进69D训练。**

### 戴盟 / Daimon-Infinity：重点追踪DataDex分区

[官方代码与说明](https://github.com/dmrobot-admin/Daimon-Infinity)、[中文文档](https://github.com/dmrobot-admin/Daimon-Infinity/blob/main/README_CN.md)、[ModelScope全集](https://modelscope.cn/datasets/daimonrobotics/Daimon-Infinity)、[Lite](https://modelscope.cn/datasets/daimonrobotics/Daimon-Infinity-Lite)。官方2026-04-15宣布首批1,000小时，多数是DataClaw夹爪，少数为DataDex/TacExoGaze人体外骨骼手套。

DataDex采集手指编码器、腕tracker及同步视觉。通用schema有 `observation.state[114]`，`finger0–35`对应每手18个finger encoder，tracker共享参考系。但这不证明每个发布文件都给完整有效的双手状态；还需encoder校准、FK与人体关节映射。

许可CC BY-NC-SA4.0。ModelScope在本轮未返回可读文件清单，因此 **DataDex实际时长、fps、schema实例和完整性尚未核验**。首批1,000小时、分析的1,209小时和目标10,000小时不能混用，尤其不能写成千小时已公开人体手指GT。公开入口值得优先追查。

### OOJU：确认保留原始人手，区分机器人流

[官方数据总览](https://ooju.world/datasets.html)明确保留完整手骨架；不是遥操完成后仅导出机器人动作。

| 发布 | 实际公开量 | 人手字段与来源 | 许可 |
|---|---|---|---|
| [WILD-Mani Kitchen](https://huggingface.co/datasets/OOJU/wild-mani-kitchen) | 102 episodes、79,244帧、30Hz，推算44.02分钟 | Quest3，双腕7D、每手26×3 positions；parquet独立于视频 | Apache2.0 |
| [Galaxea Pick Cube](https://huggingface.co/datasets/OOJU/ooju-galaxea-pick-cube) | 177 episodes、26.7分钟、38,027 XR帧 | 每手26个7D joint poses，raw human JSON与robot JSON分开 | Apache2.0 |
| [Unitree G1 Pick Place Cup](https://huggingface.co/datasets/OOJU/unitree-g1-pick-place-cup) | 47个raw recordings | 应取raw human；1,057 segments/120fps默认export属于机器人模拟管线 | CC BY-NC4.0 |

Galaxea数据采集于2026-08-13～16，人手XR约23.08–25.47fps，机器人50Hz，不能共用名义fps；米/xyzw约定和原始human frame仍须样本检查。这是OOJU的人机配对数据，不是Galaxea官方GOD机器人库。Kitchen的扩展到1,000+ episodes是目标/分批访问声明，当前公开102条。

OOJU适合验证26-joint adapter与设备估计扩展；其Quest手部是视觉追踪，不能改称独立marker/glove GT。

### OpenGraph Labs：可直接下载的MANUS小样本

[官方HF](https://huggingface.co/datasets/OpenGraphLabs-Research/manus-egocentric-sample)已发布20 episodes、96,544帧、30Hz，推算53.64分钟，总约16.4GB；Apache2.0，parquet与视频/深度分开。MANUS Metagloves原生约70Hz，发布数据对齐为30Hz。

每帧 `hand_joints[150]` 为双手各25个3D点，另有 `finger_angles[40]` 和IMU。**未见独立world wrist SE3字段**，关节点坐标/单位还须检查；可作真实手套articulation补充，不能直接假定完整69D监督。实际info.json为LeRobot v2.1，README写v3，接入以实际schema为准。

[fracapuano/manus-mano-poses](https://huggingface.co/datasets/fracapuano/manus-mano-poses)另有746KB parquet、2,278帧/60Hz=37.95秒、CC BY4.0；右手21点、腕相对坐标、米。可用来研究格式映射，不能当大规模运动集，也不是完整MANO参数拟合。

## 大规模视觉/设备轨迹与申请开放资源

### EgoVerse：应按episode字段筛选，不能只看总小时

[官网](https://egoverse.ai/)、[公开explorer](https://partners.mecka.ai/egoverse)、[下载代码](https://github.com/GaTech-RL2/EgoVerse)、[schema](https://github.com/GaTech-RL2/EgoVerse/blob/main/CONTRIBUTING_DATA.md)。官网快照4,003小时/约80K episodes，当前explorer显示4,253小时17分钟/450,703 episodes；两处当前统计不一致，差异原因未核实。已核验实际条目、SQL registry和S3/Zarr下载流程，未遍历所有云对象。

标准支持 `obs_head_pose(T,7)`、左右 `obs_wrist_pose(T,7)` 和 `obs_keypoints(T,63)`，XYZ+wxyz、21点world骨架。**wrist和keypoints可缺省**，最低要求可能仅有 `obs_ee_pose`；后者的掌部末端frame不是解剖wrist。因此4,253小时不能当全部完整手轨迹。

来源包括Aria MPS、MediaPipe+depth、OAK-D、Ultraleap等，主要为视觉/设备估计；world为序列SLAM参考系，不证明无漂移或跨episode统一。检查实际采样率、插值、`total_frames`、padding和sentinel。仓库2026-07-08修过Aria wrist/EE orientation，需固定版本。

逐episode保留来源许可，已见Apache2.0样例；同日后续核验最新`CONTRIBUTING_DATA.md`资源表标CC BY-SA4.0，**不能用代码MIT推定数据许可或覆盖原episode条款**。优先筛human embodiment、完整wrist+keypoints、有效连续长度和许可；当前建议在EgoDex后扩展，来源与clean sensor池分开。[八库预训练核验](../docs/TWM_PRETRAIN_PLAN.md)

### Lightwheel EgoSuite：大规模访问入口真实存在，数量需分层

[官方2026-08-26公告](https://huggingface.co/blog/LightwheelAI/egosuite-open100k)宣称首批10,000小时上线，100,000小时是计划总量。[EgoStandard](https://huggingface.co/datasets/LightwheelAI/EgoStandard)、[EgoPro](https://huggingface.co/datasets/LightwheelAI/EgoPro)采用HF Buckets，Git入口只有约24KB不能说明没数据；其规划80K+10K、8K+2K也不能视为已核验发布量。

需要申请，官方预计审核2–3工作日，许可名为`commercial-training-no-resale-v1.0`，官方说明允许科研及商业训练，具体使用按其条款。本次没有代用户申请或读取获批Bucket manifest，不能确认当前完整可下载净时长。

[EgoDemo](https://huggingface.co/datasets/LightwheelAI/EgoDemo)有50小时样本，实际HF repo约1.48TB，同样有访问条件。含4种带标注子集和2种纯视频子集，**50小时并非全有手轨迹**。数据卡给world hand positions `(21,3)`、rotations `(21,4)`、wxyz/float32；官方介绍视频hand-pose annotation管线，本次未找到glove/MoCap采集依据，暂归视觉估计候选。LeRobot v3与MCAP是同episode不同格式，应按 `source_episode_uuid` 去重。

标注FPS、质量、world稳定性和完整性待样本验收。其规模使它值得进入主调研表，但若用于训练，须写成视觉估计扩展，不能继续把整池称为glove/MoCap-only。

## BeingBeyond / UniHand：真正公开的统一格式，但大部分是已有来源

[官方数据卡](https://huggingface.co/datasets/BeingBeyond/UniHand_Preview)、[文件目录](https://huggingface.co/datasets/BeingBeyond/UniHand_Preview/tree/main)、[人手格式文档](https://huggingface.co/datasets/BeingBeyond/UniHand_Preview/blob/main/human_data/Human_Data.md)。本次直接读取官方API人手目录及7,328字节的Human_Data.md。

整个Preview约601GB，还包含robot和spatial understanding；实际 `human_data` 有9个TAR，合计 **68,432,271,360 bytes，约68.43GB**，对应EgoDex、ARCTIC、DexYCB、FPHA、H2O、HOI4D、OakInk2、TACO、Taste-Rob。这是文件索引求和，未下载TAR，不表示68GB都是纯pose。

每个文件级parquet包含 `camera_c2w`、左右手 `trans_w`、`rot_axis_angle_w`、MANO `theta/beta`、validity horizon。文档声明世界系wrist运动，对接本项目比重新实现九套格式方便；仍需确认root定义与源数据转换准确性。Motion与video在包内不同目录，不必在loader里读视频，但现有下载是TAR打包。

instruction/description episode行只是某个file的 `[start_timestep,end_timestep)` 切片，可能重叠；不能把instruction数当独立轨迹。增强/镜像也不能当新原始采集。跨源预训练必须保留source、原sequence和split，并核查各源许可；Preview卡本次未看到足够明确的统一数据许可。

[Being-H0官网](https://research.beingbeyond.com/being-h0)展示多源手运动学习；[Being-H0.8官网](https://research.beingbeyond.com/being-h08)的UniHand3.0采用500,000+小时raw video，含公开、合作方与自采。**这不是500,000小时统一手姿标签的公开下载承诺。** 对TWM而言Preview的主要价值是转换工程和扩展来源，不应与原数据重复算作新数据。

## 暂不能据此认定为公开完整人体轨迹的产业项目

| 团队/项目 | 本轮核验到的事实 | 处理 |
|---|---|---|
| [DexScale](https://dexscale.github.io/) | robot actions；Download仍为Coming soon | 保留线索，未算可下载人手库 |
| [穹彻CoMiner/Noe-0](https://www.noematrix.ai/news/Noematrix-WAIC-2025) | 官方展示数采套件和工具链 | 未找到对应完整人体手指+腕的公开release |
| [千寻Moz数据](https://docs.spirit-ai.com/zh/open-resources/pi.html) | 联系售后取得TOS密钥，用于Moz微调 | 获取受限且人手字段未核实 |
| [Galaxea Open-World](https://opengalaxea.github.io/G0/) | 机器人行为数据 | 不能推定保留原始人体骨架 |
| [松灵/星海图×智源平台](https://www.modelscope.cn/brand/view/baai-roboxstudio) | 列287小时机器人本体数据 | 真人遥操不等于人体状态release |
| [逐际FluxVLA](https://github.com/FluxVLA/FluxVLA) | 机器人训练/部署平台 | 本轮未核到完整新增人手release |
| [EgoMimic](https://github.com/SimarKareer/EgoMimic) | [HF](https://huggingface.co/datasets/gatech/EgoMimic/tree/main)有3个人类HDF5约128.92GB；[处理代码](https://github.com/SimarKareer/EgoMimic/blob/main/egomimic/scripts/aria_process/aria_to_robomimic.py)输出3D/6D掌心位置和actions_xyz | 官方处理代码未输出手指关节或腕旋转，已发布HDF5内部未逐项检查；不能据此列入完整69D监督池 |
| [MANUS官方](https://www.manus-meta.com/products/quantum-metagloves) / [Pro Haptic](https://www.manus-meta.com/products/metagloves-pro-haptic) | 约9段未编辑FBX样例，后者时长合计约166秒 | 真样例但未核清ML训练许可，非千小时公开库 |
| [Rokoko motion dataset](https://www.rokoko.com/mocap/motion-dataset) | 声称160万+assets、10,000+小时，70%含body+finger | 商务再许可/联系获取样本，不是已公开下载全集 |
| [NVIDIA EgoScale](https://research.nvidia.com/labs/gear/egoscale/) | 20,854小时human视频、wrist运动与retargeted dexterous actions | 未找到公开数据下载许可；训练规模不等于发布规模 |
| [DYNA-2](https://www.dyna.co/dyna-2) / [基础设施](https://www.dyna.co/research/dyna-2-infrastructure) | 公开介绍大规模egocentric采集 | 未确认完整原生手骨架下载 |
| [Sunday](https://www.sunday.ai/technology) | Skill Capture Glove和分布式采集网络 | 未确认公开数据文件与训练许可 |
| [Physical Intelligence](https://www.physicalintelligence.company/download/human_to_robot.pdf) | 人类视频/视觉手点实验，四任务14小时 | 未找到数据下载；openpi开源不等于数据开放 |
| [1X world model raw data](https://huggingface.co/datasets/1x-technologies/world_model_raw_data) | 有公开数据，CC BY-NC-SA4.0；EVE机器人25D状态 | 手部主要closure标量，不是人体五指 |
| [RC Dexterity](https://dexterity.roboticscenter.ai/data) | 触觉glove和定制手pose，样例当时标coming October2026 | 未核实可下载文件；也不是另一家Dexterity公司 |

对未找到的资源只能写“本轮未确认”，不能断言不存在。数采行业持续发布，后续应沿官方数据组织、HF/ModelScope目录和schema查新，而非只搜论文里的hand dataset。

## 更新后的接入优先级

1. 保留GRAB/ARCTIC/OakInk2/HOT3D作为已有MoCap参考与小规模baseline，不把它们当完整市场清单。
2. **优先验收SynData、HumanTouch、DexWild**，核查原始人体信息保留、可映射关节、有效时长与腕部质量；产业规模不能代替这一检查。
3. **EgoVerse与Lightwheel进入规模扩展候选**，OOJU用于完整设备手骨架adapter；对视觉/设备数据采用独立source和quality标签，不因不是glove而从调查清单隐藏。
4. 聚合库优先评估格式复用价值与去重；申请访问的库先完成许可/manifest核验，再统计实际公开净时长。

所有下载、样本数值检查、FK/模型推理和训练都在hongfei远程项目进行；本机仅处理文档与获取说明。此处是研究建议，未自动选定最终混合、下载全集或向公司发送申请。
