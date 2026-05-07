# AeroWorld Capture Runtime Audit

本文档记录全量数据采集前的运行约束、目录契约、长期风险和审计口径。当前采集路线以完成数据为第一目标：复用已进入 PIE 的 UE 会话，UAV 图像只走 AirSim native 单采集机，ground 固定世界相机只采 RGB。

## 采集技术路线

- Ground fixed-world camera：只采 `rgb`，通过 editor hook `AAeroFixedWorldCaptureCamera` 输出。
- UAV camera：只通过 AirSim native `CaptureUAV_0` 输出 `rgb/depth/seg`。
- `CaptureUAV_0` 是采集平台，会替换当前指定的 source UAV 视角；其他 UAV 仍按 runtime multirotor 渲染为背景事件实体。
- 每次运行只允许一个 camera 的一种 modality。多视角、多通道用多轮 deterministic rerun 写入同一个 episode output root。
- UAV editor-hook capture fallback 已禁用；`seg` 采集前 Python 会用 UE PIE actor 注册和 AirSim `simSetSegmentationObjectID` 设置语义类 ID。

## 目录命名规则

统一目录结构：

```text
<episode_output_root>/<batch_id>/<capture_view_id>/<modality>/<frame_stem>.<ext>
```

其中：

- `episode_output_root`：用户传入的 `--output_dir`，不得包含 timestamp/version 子目录。
- `batch_id`：site contract 中的稳定批次 ID，例如 `site.intersection_a`。
- `capture_view_id`：稳定视角 ID。全量采集时必须显式传入，建议格式为 `<scenario>__<source_uav_entity>__<view_name>`。
- `modality`：固定为 `rgb`、`depth`、`seg` 之一。
- `frame_stem`：由 truth frame 生成，包含 tick 和 frame id。

Ground RGB 示例：

```text
Saved/AirSim/full_capture/L4-5_v1/site.intersection_a/ground_overview/rgb/tick_000900__frame_L4-5_v1__seed00_tick_900.png
Saved/AirSim/full_capture/L4-5_v1/site.intersection_a/ground_overview/rgb/tick_000900__frame_L4-5_v1__seed00_tick_900.json
```

UAV 多通道示例，同一 tick、同一 `capture_view_id` 的 sibling 目录必须可对齐：

```text
Saved/AirSim/full_capture/L4-5_v1/site.intersection_a/L4-5_v1__uav_ped_nearmiss__bottom/rgb/tick_000900__frame_L4-5_v1__seed00_tick_900.png
Saved/AirSim/full_capture/L4-5_v1/site.intersection_a/L4-5_v1__uav_ped_nearmiss__bottom/depth/tick_000900__frame_L4-5_v1__seed00_tick_900.npy
Saved/AirSim/full_capture/L4-5_v1/site.intersection_a/L4-5_v1__uav_ped_nearmiss__bottom/seg/tick_000900__frame_L4-5_v1__seed00_tick_900.png
Saved/AirSim/full_capture/L4-5_v1/site.intersection_a/L4-5_v1__uav_ped_nearmiss__bottom/seg/tick_000900__frame_L4-5_v1__seed00_tick_900__airsim_raw.png
```

每个输出 root 会写入：

```text
<episode_output_root>/capture_storage_manifest.json
```

每帧 sidecar 必须包含：

- `storage_layout_version: capture_storage_v1`
- `storage_rule`
- `capture_alignment_key = episode_id:batch_id:tick:capture_view_id`
- `capture_view_id`
- `channel_id`
- `modality_output_dir`
- `relative_primary_output_path`
- `deterministic_overwrite_scope: modality_output_dir`
- `single_camera_single_modality_capture: true`

## Modality 语义

- `rgb`：PNG，UAV 来自 AirSim `ImageType.Scene`；ground 来自 editor hook RGB。
- `depth`：`.npy`，float32，单位米，只用于 UAV；可选 debug preview PNG 不作为训练主数据。
- `seg`：AirSim semantic class ID-color segmentation。主输出是语义类颜色图：
  - `<frame_stem>__airsim_raw.png` 保留 AirSim 原始 segmentation 图用于审计。
  - `<frame_stem>.png` 与 AirSim class ID-color 输出一致，不做二值化。
  - sidecar 写 `segmentation_kind: airsim_semantic_class_id_color` 和 `semantic_segmentation_claim: true`。
  - 静态类包括 `city_base_background`、6 个 building subtype；动态类包括 UAV、vehicle、pedestrian、roadwork、traffic control、facility、hazard trigger、service/misc props。

论文表述不得声称当前 `seg` 能区分 road/terrain/water/material；`BP_CityBaseGenerator0` 统一作为 merged city-base background 类。建筑 subtype 可在训练或论文统计时合并为 `building`。

## 多 UAV 视角调度

70 个事件的多 UAV 视角不是一次运行同时打开多个 AirSim 采集机，而是：

1. 每个 scenario 先从 truth frame/event chain 中选择需要覆盖事件全过程的 source UAV 列表。
2. 对每个 source UAV 分配稳定 `capture_view_id`。
3. 每个 `capture_view_id` 分别运行 `rgb`、`depth`、`seg` 三轮。
4. 如果一个 scenario 有多个关键 UAV，就增加多个 `capture_view_id` sibling 目录；不要增加多个 AirSim capture vehicle。
5. 俯视 ground camera 只采 RGB，用于整体事故地点审计；UAV 视角并集负责覆盖事件发生前后流程。

命令模板：

```powershell
python Plugins\SumoImporter\Scripts\episode_render_host.py `
  --config Dataset\render_ready_episodes\<episode>\render_host_config.json `
  --output_dir Saved\AirSim\full_capture\<scenario_id> `
  --camera-role uav `
  --uav-capture-backend airsim_native `
  --airsim-capture-vehicle CaptureUAV_0 `
  --airsim-capture-entity <source_uav_entity_id> `
  --capture-view-id <scenario>__<source_uav_entity_id>__bottom `
  --modality <rgb|depth|seg> `
  --tick_stride <capture_stride> `
  --simulation_tick_stride 1
```

## 长期运行潜在不足

1. 坐标和高程仍是第一风险。traffic bundle、truth frame、UE world cm、AirSim NED 的转换必须保持单一方向，不能再出现 double offset。地面高程有起伏，行人/车辆需要依赖 resolver 或运行时地面投影，不能硬写统一 z。
2. 事件物体可见性必须通过 smoke 验证。不可见或仅 metadata 注册的实体不能当作训练图像中的可见对象。
3. UAV camera pose 仍需要每个 scenario 验证。若 AirSim 回读 pose 显示 `[0,0,0]` 或画面向天空，说明当前 PIE/车辆/camera 状态不可信，不能进入长跑。
4. `seg` 现在是 AirSim 语义类 ID-color。若某类没有可注册 component，sidecar 必须记录 zero-match，不允许从 RGB 猜类别或伪造像素。
5. 深度和 segmentation 都走 `simGetImages`，长跑存在 RPC、GPU readback、PNG/Numpy 编码和内存压力。必须小 chunk 运行，失败后保留 PIE 现场。
6. 每轮只采一个 camera 的一个 modality。多视角或多通道并行采集会增加显存和 RPC 压力，不作为默认路线。
7. 动态 actor 生命周期要保持可审计。普通 episode/chunk 结束只清理脚本实体，不关闭 UE/PIE；只有 C++ 编译或用户明确要求时关闭。
8. `event_fired_after` 是 scripted causal delay，不等同于物理动作完成。论文表述应区分 scripted causal chain 和 fully physics-completed action chain。
9. validation pass 不能替代图像 smoke。坐标、碰撞、朝向、可见性和相机覆盖必须用代表场景图像抽查。
10. 重跑同一 modality 会删除并覆盖该 modality 目录；不同 modality 是 sibling 目录，不互相删除。全量脚本必须按 root/batch/view/modality 粒度断点重跑。

## 全量采集前检查

必须先完成：

- UE 已手动进入 PIE，AirSim RPC 可连，`CaptureUAV_0` 不下坠。
- `E:\HuaweiMoveData\Users\weizhiwei\Documents\AirSim\settings.json` 中保留一个 `CaptureUAV_0`。
- `python Plugins\SumoImporter\Scripts\episode_render_host.py ... --segmentation-registry-audit-only`
- `python Dataset\tools\validate_scene_grounding.py --dataset-root Dataset`
- `python Dataset\tools\validate_event_reachability.py --dataset-root Dataset`
- 对至少一个行人场景、一个多 UAV 场景、一个天气场景做 `rgb/depth/seg` smoke。
- 检查同一 tick 的三通道 sidecar `capture_alignment_key`、source UAV pose、camera pose 是否一致。
- 检查 ground RGB 是否覆盖事故整体地点，UAV view union 是否覆盖事件前后流程。

## 审计结论

当前代码已把采集路线收敛为两条：ground RGB editor hook 和 UAV AirSim native RGB-D-semantic-class segmentation。目录结构和 sidecar 已显式写入，可被后续 Claude 或人工审计。剩余风险主要不是文件组织，而是 AirSim/UE actor 注册、坐标/相机/可见性/内存稳定性，必须通过 registry audit、小 chunk smoke 和分段全量采集暴露并处理。
