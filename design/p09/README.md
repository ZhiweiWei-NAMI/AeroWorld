# P09 当前真值与采集输入

P09 生产世界真值和采集输入，P01 消费真值训练。基础冻结世界、v14 执行版本和 L6 R2 是不同来源，不能把一个版本的轨迹和另一个版本的语义图混用。所有大产物位于 `/mnt/data1/weizhiwei/AERO_WORLD_runtime/p09/`，不进入 Git；原始 `aw_data/` 和 `arms/` 保持不变。

| 内容 | 当前产物或入口 | 含义 |
|---|---|---|
| L6 真实收发与执行 | `l6_receipt_r2/summary.json` | native 收发、实际动作及物理后果 |
| L6 当前语义图 | `l6_objective_current/current_binding.json` | R2 完整目标轨迹与当前采集输入绑定 |
| 210／ARM 核心来源 | `core_source_current/source_index.json` | 真实来源、计算模型、适用性及缺失原因 |
| 基础世界能量图 | `core_source_current/energy_semantic_current/source_index.json` | 可计算 SOC 进入既有 typed 状态和低电量谓词；与执行版本分开 |
| UE 输入及来源 | `ue_input_current/source_status_manifest.json` | 完整输入、天气更新、背景 yaw 修复、实际重采范围 |
| 重采清单 | `ue_input_current/recapture_episodes.csv`、`recapture_arms.csv` | episode／arm、变化帧、入口及原因 |

最终 UE 输入包含156个完整episode、54个episode天气更新、690个ARM天气更新和211个ARM完整窗口。统一重采清单为153个episode、211个ARM，分别有8152和1791个变化采集时刻；完整重采分别为27693和14314个时刻。三个仅非采集时刻航向改变的episode不列入重采。所有窗口以包内capture_plan为准，服务器尚未执行UE采集。

L6 保留原 9 m/s 异常动作，真实接收后在 tick275 执行；4 m/s 限制在310执行，GCS/UAV状态分别在311/315应用，UAV在390恢复，470执行降落，588着陆、589停稳。九个 native 包均实际收到。实际未收到包数为0；不据此虚构无线丢包负例。

L6 语义构建实际用时467.226秒，写出33个产物，图拒绝数0；有9条事件、9条后果、52条转换、312条连续性记录。闭合为 `PASS + EXPLICIT_GAPS`：GCS阶段在245通过，其余三项保持N/A。当前路线偏差为真实false；native接收成功不等于认证拒绝，保护生效不等于安全控制不可用。当前本体尚无消费这些回执的 typed command/control producer。完整回执与执行状态保留，不据此虚构本体正例。

L2-1_v2__seed00 保留已接受的负例：tick265失联阶段通过，恢复闭合失败。当前执行中恢复patch被显式跳过，UAV继续移动并切到私有备用链路；原发布版的恢复记录属于另一个执行版本。补回原始dust字段的接线本身不改变轨迹或天气数值。

基础来源计算得到51301条可计算的全局UAV能量记录；62842条观测缺少此前能量、天气或充电历史，保持明确缺口。能量来自已声明参数模型、实际运动和对应天气，不是电池遥测。计算任务索引中的6380条候选任务与低负载参考分别记录，不能回填成冻结世界里没有记录的历史任务。ARM只对声明窗口、实体和谓词范围作结论；窗口外采用不可变来源引用。

P01初态入口读取真实 `world_truth_graph_base.initial_assertions` 和现有delta。当前导入器已实际消费60集的113条新增tick0 SOC状态。模型组合只读取一次native初态，索引构建时核对base/delta来源路径；同一实体保留各谓词角色声明的真实类型。真实TRAIN样本cutoff5、目标10/15的load_index和semantic_text已通过：679条初态各一次，191个图实体，原有unknown保留。不伪造初始delta，不新增监督字段。固定数据划分、采样、seed和loss未改，现有训练缓存没有被替换。使用新真值时须显式选择来源索引里的根路径。

当前210个episode没有可用的非零dust场景。原始明确记录的0保留其生产来源；缺失字段不再补成0。缺少原始天气依据的旧dust默认值已移除；渲染影响单独按当前renderer实际计算。天气元数据不是传感器测量，现有P01 dust archive也不是监督头输入。

L6相对原发布密集轨迹的首个位置变化为271，capture网格为275；相对R1为466，GCS位置未改。场景声明的初始yaw在capture35恢复为35度；全局UAV停止水平运动时保留其真实源yaw。这些视觉输入变化进入统一重采清单。仅标签、事件接线或不改变renderer输出的天气元数据修复无需重采。UE的RGB、depth、segmentation采集由用户执行，本次服务器运行没有产生新的传感器帧。

复现实现见 [L6](../../Dataset/semantic_simulation/p09_l6_5_receipt_v1/README.md)、[L2](../../Dataset/semantic_simulation/p09_dimension_repair_v1/README.md)。`Dataset/semantic_simulation/p09_core_sources.py` 生产来源索引；`Dataset/semantic_truth/p09_energy_graph_update.py` 把可计算能量接回现有图；`Dataset/tools/p09_release.py` 组装采集输入。入口使用项目airfogsim Python，传入明确的原始项目、已执行结果和新输出目录。已完成产物不作为新的仿真；路径引用与重新执行负责追溯。

原始多模态文件可保留在用户本地，服务器保留episode、tick/frame、modality、采集版本、原始文件路径、位姿/相机参数、编码器checkpoint路径与embedding对应关系。embedding通常不能无损还原LiDAR点云。可实现的验证是读取本地原始点云，以保存的位姿变换到同一坐标系，与模型实际输出的点云或占据预测比较；若模型仅输出embedding，则先保留原始点云或下采样预览，不能声称已有重建能力。共同main已提供粗粒度LiDAR range/return解码路径；这不是Sonata embedding的精确逆变换。本次未训练或验证该解码器，也不重建旧编码器契约。

基础能量图的大文件物理存放于原项目 `aw_data/p09_current/energy_semantic_current/frozen_baseline/`，运行目录内原路径以符号链接保持可用。只迁移本次生成的数据，原始冻结输入不变；该目录与其他大型产物均不进入Git。
