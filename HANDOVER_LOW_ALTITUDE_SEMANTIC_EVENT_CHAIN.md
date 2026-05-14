# Low Altitude Semantic Event Chain Handoff

## 通用规则

1. 工作区根目录是 `E:\DynamicCityCreatorSamples`。
2. Python 固定使用 `E:\conda\envs\aeroagentsim\python.exe`。
3. UE 采集默认复用已经进入 PIE 的会话和 AirSim RPC `127.0.0.1:41451`，普通 episode 或 chunk 采集不能关闭 UE/PIE。
4. 只有需要重新编译 C++ 或用户明确要求时，才允许关闭 UE/PIE。
5. 输出目录必须确定性，不创建 timestamp/version 目录。失败 episode 用相同 output root 重跑。
6. 全量语义 truth 的权威源是 `E:\DynamicCityCreatorSamples\Dataset\render_ready_episodes`，但该目录禁止直接作为 UE 采集输入。
7. UE 正式采集只使用过滤后的输入根目录：`E:\DynamicCityCreatorSamples\Dataset\render_ready_episodes_capture_filtered`。
8. `trajectories.jsonl` 只用于 F 盘 Matplotlib/离线轨迹检查，不是 UE 控制输入。UE host 读取 `truth_frames.jsonl`。
9. 过滤后的 truth frame 坐标空间是 `map_enu`。UE host dry-run 已确认 `raw_pos == transformed_pos`，车辆只保留既有 road-snap 的 z/yaw 修正。
10. 非采集 UAV 是普通 UE scene asset，由 `truth_frame_scene_sync` 根据 truth frame 离散设置位姿。正式链路不再接受 `pose_sync` 或 `airsim_move` 作为 UAV scene-control backend。

## 验收标准

1. `Dataset\render_ready_episodes_capture_filtered` 下必须有 210 个 episode 目录，和 `Dataset\render_ready_episodes` 一一对应。
2. 每个过滤包必须包含有效的 `truth_frames.jsonl`、`render_host_config.json`、`scenario_package.json`、`episode_manifest.json`、`scenario_plan.json`、`global_entity_roster.json`、`trajectories.jsonl`。
3. 每个 `render_host_config.json` 和 `scenario_package.json` 的 episode/root/truth 路径必须指向 `Dataset/render_ready_episodes_capture_filtered/<EPISODE_ID>`，不能指向 `Dataset/render_ready_episodes/<EPISODE_ID>`。
4. 每个 `render_host_config.json` 必须包含 `truth_frame_coordinate_space: map_enu` 和有效 `event_script_path`。
5. 每个 episode 的 capture window 内必须出现 pedestrian、vehicle、uav 三类动态主体；允许行人或车辆穿过 ROI 后消失，不要求每一帧都同时存在 P/V/U。
6. 每个 episode 首帧必须保留 `facility` 和 `airspace_corridor`，以保证 NFZ/capture boundary、corridor、pad、charger、event facility 等语义基础设施不会被过滤掉。
7. 采集前至少对代表 episode 执行 dry-run 坐标预览，确认 `truth_frame_coordinate_space=map_enu` 且实体 `raw_pos` 与 `transformed_pos` 一致。
8. UE 正式采集每次只跑一个视角和一个模态，避免显存和内存累积。

## 当前进展

1. 已实现过滤脚本：`E:\DynamicCityCreatorSamples\Dataset\tools\filter_render_ready_truth_for_capture.py`。
2. 已生成过滤根目录：`E:\DynamicCityCreatorSamples\Dataset\render_ready_episodes_capture_filtered`。
3. 已完成 210/210 episode 过滤输出。
4. 已修复过滤阶段的 P/V/U 类别保护：当某一帧源 truth 存在某类 P/V/U，但 capture 可见性过滤会删空该类时，补回距离 capture/inspect 最近的一个语义候选，避免轻量化过滤破坏 episode 语义主体。
5. 全量过滤结果检查通过：
   - episode 目录：210/210
   - 有效 truth：210/210
   - config 指向 filtered root：210/210
   - scenario package 指向 filtered root：210/210
   - `truth_frame_coordinate_space=map_enu`：210/210
   - `event_script_path` 存在：210/210
   - episode 级 P/V/U 出现：210/210
   - 首帧保留 facility：210/210
   - 首帧保留 airspace corridor：210/210
   - trajectories 存在：210/210
6. 已抽测 UE host dry-run：
   - `L1-1_v1__seed00`
   - `L4-4_v1__seed00`
   - `X5_comm_failure_to_pad_contention__seed00`
7. dry-run 结论：filtered truth 按 `map_enu` 读取，实体 `raw_pos == transformed_pos`；车辆 road-snap 的 z/yaw 修正属于既有 runtime 行为。

## 下一步指令

### 只在 full truth 发生变化后重建 filtered 包

```powershell
cd E:\DynamicCityCreatorSamples
& E:\conda\envs\aeroagentsim\python.exe .\Dataset\tools\filter_render_ready_truth_for_capture.py `
  --input-root .\Dataset\render_ready_episodes `
  --output-root .\Dataset\render_ready_episodes_capture_filtered `
  --overwrite `
  --workers 4
```

### 采集前坐标 dry-run

```powershell
cd E:\DynamicCityCreatorSamples
& E:\conda\envs\aeroagentsim\python.exe .\Plugins\SumoImporter\Scripts\episode_render_host.py `
  --config .\Dataset\render_ready_episodes_capture_filtered\L1-1_v1__seed00\render_host_config.json `
  --dry_run_coords `
  --coord_preview_limit 10
```

### UE ground 视角单模态采集模板

```powershell
cd E:\DynamicCityCreatorSamples
& E:\conda\envs\aeroagentsim\python.exe .\Plugins\SumoImporter\Scripts\episode_render_host.py `
  --config .\Dataset\render_ready_episodes_capture_filtered\<EPISODE_ID>\render_host_config.json `
  --host 127.0.0.1 `
  --port 41451 `
  --camera-role ground `
  --camera-id <GROUND_CAMERA_ID> `
  --modality <rgb|depth|seg> `
  --uav-scene-control-backend truth_frame_scene_sync `
  --segmentation-backend ue_custom_stencil
```

### UE UAV 视角单模态采集模板

```powershell
cd E:\DynamicCityCreatorSamples
& E:\conda\envs\aeroagentsim\python.exe .\Plugins\SumoImporter\Scripts\episode_render_host.py `
  --config .\Dataset\render_ready_episodes_capture_filtered\<EPISODE_ID>\render_host_config.json `
  --host 127.0.0.1 `
  --port 41451 `
  --camera-role uav `
  --camera-id <UAV_CAMERA_ID_OR_ENTITY_ID> `
  --modality <rgb|depth|seg> `
  --uav-capture-backend airsim_native `
  --uav-scene-control-backend truth_frame_scene_sync `
  --segmentation-backend ue_custom_stencil `
  --airsim-capture-vehicle CaptureUAV_0 `
  --airsim-capture-entity <UAV_ENTITY_ID> `
  --capture-view-id <VIEW_ID>
```

### 路径红线

```text
UE 正式采集输入，只能使用：
E:\DynamicCityCreatorSamples\Dataset\render_ready_episodes_capture_filtered\<EPISODE_ID>\render_host_config.json

不要把下面这个全量目录作为 UE host config/root：
E:\DynamicCityCreatorSamples\Dataset\render_ready_episodes
```
