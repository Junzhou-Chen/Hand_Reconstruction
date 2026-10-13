# 传感器轨迹优先的TWM预训练

日期：2026-10-13。用户纠正：TWM不需要视觉，选库应重视更准确的手姿态，初始glove/MoCap方向有价值；论文用库作参考，不主导训练池。**已定的是精度优先原则与99D输出；具体混合仍为建议。**

建议恢复GRAB/ARCTIC/OakInk2/HOT3D的高质量MoCap核心候选，将HumanTouch/DexWild提升为主要glove验收对象；新增公开UmeTrack补自由手势和pinch。按免申请偏好，实际先验收Oak/HOT/UmeTrack与HumanTouch/DexWild，GRAB/ARCTIC有既有授权再接入。没有下载完整数据、做数值标定或训练。

## 1. 选库依据与优先级

| 数据 | 手指与腕部来源 | 纯轨迹适合程度与接入条件 |
|---|---|---|
| [GRAB](https://github.com/otaheri/GRAB) | Vicon marker→模型拟合；完整连续手运动，120Hz | 高优先MoCap核心候选；无RGB无妨。需注册；个性化v_template需保留/拟合，不能默认β=0无损复现 |
| [ARCTIC](https://github.com/zc-alexfan/arctic/blob/master/docs/data/data_doc.md) | Vicon→world MANO，双手灵巧操作；有fitting_err/raw marker | 高优先MoCap核心；纯参数即可，需注册。误差字段可筛拟合，不把视觉重建指标当标签误差 |
| [OakInk2](https://github.com/oakink/OakInk2) | OptiTrack→SMPL-X/MANO；部分遮挡marker人工修复 | 直接公开的完整状态核心；取annotations，不依赖RGB；验父序列/时间/个体shape |
| [HOT3D](https://github.com/facebookresearch/hot3d/blob/main/hot3d/clips/README.md) | optical marker＋scan→UmeTrack→MANO | 直接公开Clips；native几何参考优先，不因有ego图像提高纯TWM权重；split先排benchmark原录制 |
| [UmeTrack独立集](https://huggingface.co/datasets/facebook/hand_tracking_challenge_umetrack) | 36-camera MoCap＋scan；分离手势、双手交互与pinch | **新增直接公开marker候选**；world wrist/native pose/converted MANO；补非抓取动作，真实/合成共用motion需去重 |
| [HumanTouch](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch) | MANUS EMF、25点/方向；world腕位姿契约，具体腕硬件需核 | **主要glove候选，优先验收**；parquet/meta独立于视频，CC BY-NC4.0已核实；源骨架与MANO转换仍需验收 |
| [DexWild human](https://huggingface.co/datasets/boardd/dexwild-dataset) | MANUS full；ArUco腕tracker＋ZED/相机变换 | **主要glove候选**；公开MIT；7,917episodes/2,924,588transitions为发布量，actual dt/world/已有平滑须验；只取raw human |
| [DexCap raw](https://huggingface.co/datasets/chenwangj/DexCap-Data) | Rokoko手指、T265 SLAM腕 | 公开1.5h，适合快速glove adapter；别用LEAP action，别把腕SLAM当光学参考 |
| [OpenGraph MANUS](https://huggingface.co/datasets/OpenGraphLabs-Research/manus-egocentric-sample) | 每手25点、双手40个finger-angle通道 | 公开约54min、parquet可单取；未见独立world wrist SE3，做local部分监督/映射验收 |
| TCDHands / HandX自采 | finger markers / OptiTrack skin markers | 手势与独立marker补充；前者小且需AMASS注册，后者须source筛自采与审canonical/world；SAMP精细手指覆盖较弱 |

本轮完整适配判定：GRAB/ARCTIC/Oak/HOT/UmeTrack具备完整任务字段条件；HumanTouch/DexWild/DexCap保持高价值glove候选，需映射及腕链/质量验收；OpenGraph当前只确认局部监督，独立world腕待核。其它扩查条目仍按source/字段审查。**所有候选均未完成本项目adapter验收，不得将“能参与某项loss”写成“每个库完全适配”。** 完整任务训练池须有经过验收的腕＋五指连续参考，部分来源不能单独提供全部任务监督。[适配判定](../docs/TWM_PRETRAIN_PLAN.md#候选数据的完整适配判定)

这里按监督来源、可审计性和覆盖筛选，**没有同一真值基准支持这些库的毫米级精度排序**。marker/glove输出也经过骨架解算和标定；传感器品牌、关节数量、名义fps不等于全部99D均有独立真值。

### UmeTrack与HOT3D需要保留native参考

[UmeTrack论文§4](https://arxiv.org/html/2211.00099v1#S4)描述36相机MoCap、scan、30Hz/15秒序列，真实与Unity合成逐段复用运动。当前HF发布与原论文统计版本不同，净有效train量待按源ID与时间验收，不将render重复当新轨迹。[官方toolkit](https://github.com/facebookresearch/hand_tracking_toolkit/blob/main/README.md#important-notes)明确converted MANO整体稍逊于UmeTrack，推荐native keypoints监督。decoder读取world wrist transform；MANO registration失败帧保留mask。

3mm是marker直径，不是手关节GT误差。UmeTrack20轴与MANO15个SO(3)不同，不能由转换得到全部rotation独立真值；姿态loss与native几何loss分别标来源。

## 2. HumanTouch访问与规模勘误

固定HF快照`81822a10cac5def2638db546eaeced581d71a8b5`，更新时间2026-09-06。本轮公开API显示private=false/gated=false；许可由HF README指定的[ModelScope官方summary](https://modelscope.cn/datasets/chuqiaoLyu/Xspark-HumanTouch/summary)核验。

summary页面`window.__detail_data__`转义JSON顶层字段为`Id=211198, Namespace=chuqiaoLyu, Name=Xspark-HumanTouch, License=CC-BY-NC-4.0`。已用只读请求解析并复核，**此前“具体许可未知”已解决**；许可不等于几何精度验收。

以下来自十份公开`X*/meta/info.json`，每份声明fps=60：

| 分区 | Frames | Episodes |
|---|---:|---:|
| X001 | 3,832,214 | 2,217 |
| X002 | 2,733,581 | 3,836 |
| X003 | 2,389,801 | 2,054 |
| X004 | 2,677,193 | 1,853 |
| X005 | 2,333,104 | 722 |
| X006 | 3,420,213 | 1,362 |
| X007 | 2,345,109 | 1,313 |
| X008 | 2,763,774 | 744 |
| X009 | 3,454,211 | 1,897 |
| X010 | 2,725,658 | 1,610 |
| 合计 | 28,674,858 | 17,608 |

来源示例：[X001](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch/blob/81822a10cac5def2638db546eaeced581d71a8b5/X001/meta/info.json)、[X010](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch/blob/81822a10cac5def2638db546eaeced581d71a8b5/X010/meta/info.json)。按声明60Hz累计约132.75h，**未去重、未数值QC，不是训练净量**；官方“约100h”与metadata分口径保存。

公开pose parquet索引计17,576文件，与metadata episodes不同：X005 meta722/文件723、X008 meta744/文件711。manifest需按实际episode/timestamp验收，不由此断言轨迹坏或未发布。统一60Hz含插值/SLERP，不代表各硬件原生频率。

[官方README](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch/blob/main/README.md)给出自接触辅助标定骨长/关节偏移与episode人工审核；尚无全库独立毫米误差验证。[episode契约](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch/blob/main/X001/meta/humantouch/episodes/episode_000001.json)提供米、xyzw、world wrist与左右25点/方向、validity/source/session。只取parquet/meta训练运动，不读RGB、触觉或机器人action；触觉辅助标定是采集质量资料，不是新增TWM输入。

## 3. 99D输出不要求各库原生MANO

保留两层数据：**source-native测量/骨架**与**canonical99D模型状态**。用户再次明确TWM主输出必须是每手完整99D，即wrist p3＋root6＋15×finger6，shape条件另存。所有这些候选可以按有效字段参与同一99D输出模型训练，但不能承诺处理后全部拥有完整99D真值；离线拟合仅产生带来源/残差的派生估计，缺world腕或未测twist仍是监督缺失。joint-only可以直接用MANO解码几何监督，不要求先拟合全库。

1. 有经核验MANO参数：按源hand mean/rest frame解码，支持rotation监督；同时保留native几何reference，记录拟合来源。
2. glove有原始joints：预测99D→可微MANO解码→有效joint几何loss；不能因拟合了MANO便将全部角度作为传感器直接GT。
3. 有bone directions/orientations：通过sensor/rest/parent-frame标定后，仅对确实对应的旋转分量监督；未测twist或耦合角度mask掉。
4. 仅local finger标签也可参与，腕p/R loss按可用性分别mask。SLAM腕漂移不应污染已可靠的local手指监督。
5. tracker→解剖wrist偏移、骨长/β与左右手映射先验收；少量拟合可以估计转换误差，不要求全库先离线生成MANO伪标签。

[GRAB官方decoder](https://github.com/otaheri/GRAB/blob/master/grab/save_grab_vertices.py)使用个性化v_template。β10-only拟合可能存在形态表达残差，保存source template/原几何，不能把所有人β=0后把形状差异算成姿态误差。[FSGlove](https://arxiv.org/html/2509.21242v1)的shape-aware标定提供校准思路，硬件/代码开放不等于已发布大规模训练轨迹。

验收记录单位/时间、源关节定义、校准、native→99D joint残差、有效性、fit error、腕跳变与split。没有独立真值时，骨长稳定/低抖动只能筛异常，不能证明毫米精度。实际preprocess/FK与精度测试全部在hongfei执行。

## 4. StableHand参考范围

[v1 §3.2/F](https://arxiv.org/html/2605.18553v1)中QN不读图像feature，读数值MANO轨迹/flag/camera；其训练质量标签仍来自视觉预测与reference误差。八库约10M服务QN，joint-only三库不拟合MANO；生成器含视觉条件，按HOT/ARCTIC训练。[Enhanced说明](https://github.com/huajian-zeng/stablehand/blob/0471a45e325ffa167cc6a8484d712a21befd3e47/docs/enhanced_version.md)和[source](https://github.com/huajian-zeng/stablehand/blob/0471a45e325ffa167cc6a8484d712a21befd3e47/model/quality_network.py)另加入depth cues，不能将所有版本QN笼统称纯motion。

可借鉴的是异构标注分层、腕/手指质量分组与故障扰动；**99D joint-only几何恢复是本项目方案，非StableHand已验证的motion预训练协议**。高质量reference选择独立于它的八库名单。[HMP](https://arxiv.org/html/2312.16737v1)从GRAB/TCDHands/SAMP学无图像prior，更接近最初思路；其当前代码GRAB/ARCTIC train与原论文分开保存。

## 5. 当前建议执行顺序

- **第一批共同验收**：OakInk2/HOT3D/UmeTrack的native/99D桥接，与HumanTouch/DexWild原人体骨架；DexCap/OpenGraph小样本辅助glove adapter。glove不再被默认降为远期支线。
- **开始训练**：已通过验收的MoCap与glove段构成质量分层pool；缺字段做部分监督，不为等所有loader而暂停第一条链。GRAB/ARCTIC有授权后直接进入同一高优先候选。
- **检验来源价值**：matched-budget比较marker-only、marker＋glove、glove-only适用监督范围；优先看真实joint/wrist误差与干净输入退化，不以平滑论优劣。
- **后续才扩视觉标签**：Re/EgoDex/GigaHands等作为单独扩展实验，不能再因格式方便/论文频繁使用而优先于sensor参考；真实HaWoR错误适配属于后续训练阶段。

建议不声称任一厂家数据最准确，不预设最优混合比；用高质量标注与可审计标定建立先验，再验证它是否纠正真实视觉输出。
