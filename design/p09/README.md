# P09 当前数据与复现入口

新生成的数据归入已有项目目录，大数据不进入 Git。UE 传感器采集由用户管理，本次整理不生成重采清单。

| 内容 | 项目内入口 |
|---|---|
| L6 原生收发、执行回执和目标轨迹 | `Dataset/episodes/L6-5_v1__seed00/` |
| 已采用执行的原生证据 | `Dataset/episodes/<episode>/native_execution/` |
| 世界语义状态和本体图 | `aw_data/objective_semantic_truth/<episode>/` |
| UAV 能量、生命周期、计算任务来源 | `aw_data/domain_state_supplement/source_index.json` |
| UE 轨迹与场景输入 | `aw_data/render_ready_episodes_capture_filtered/<episode>/` |

`source_index.json` 逐 episode 记录执行来源及 `objective_semantic_truth.matches_current_execution`。已有原生执行证据但没有同次执行完整语义图的 episode 标为 false；P01 的完整语义导入拒绝这类 episode，防止旧 baseline 图混入训练；真实回执和原生状态仍保留在对应 native_execution 目录。L6 的 `actual_execution_trajectories.jsonl` 保存原生目标轨迹，`trajectories.jsonl` 保存组装后的完整世界轨迹。

能量值来自声明参数模型、真实运动与对应天气；不能称为电池遥测。缺少此前能量、天气或充电历史时保留来源缺口。候选计算任务与历史任务分开。ARM 仍按原有窗口、实体和谓词范围解释。

L6 的命令接收、权限保护、恢复和着陆以实际回执和状态为依据；这些事实不自动等同于本体中的认证拒绝、控制不可用或路线偏差。L2 的当前执行保留恢复闭合失败，不借用旧执行的恢复记录。

可复用代码：

- [L6 收发与语义投影](../../Dataset/semantic_simulation/p09_l6_5_receipt_v1/README.md)。
- [L2 修复](../../Dataset/semantic_simulation/p09_dimension_repair_v1/README.md)。
- `Dataset/semantic_simulation/p09_core_sources.py`：生成核心来源表。
- `Dataset/semantic_truth/p09_energy_graph_update.py`：将可计算能量写入选定 baseline 的现有状态和图。
- `Dataset/tools/p09_release.py`：对显式选择的正式 episode 修正有真实来源依据的全局 UAV yaw。

使用项目 airfogsim Python；入口参数见各模块 `--help`。固定数据划分、采样、seed 和 loss 不变，已有训练缓存未被替换。
