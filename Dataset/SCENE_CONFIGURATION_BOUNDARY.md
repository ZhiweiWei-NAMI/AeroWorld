# 场景配置边界文档 & GPT 生成提示词

## 使用方式

将本文档完整交给 GPT，并附带以下指令：

> 根据这个配置边界文档，为 `Dataset/scenarios/` 下 64 个场景各自生成：
> 1. 更新后的 `spec.py`（用 SpecCompiler 的 ScenarioSpec 格式，不是模板化的占位符）
> 2. 编译产物 `event_script.json`
>
> 每个场景必须满足下面定义的语义落点、实体解析、触发条件和验证标准。

---

## 零、基础设施清单（写死在 prompt 里，GPT 不能编造）

### 可用资产 (`logical_asset_id`)

```
UAV:
  uav.inspect.quad.v1           — 巡检四旋翼
  uav.airsim.flying_pawn.v1     — 通用飞行器
  uav.airsim.cv_pawn.v1         — 视觉飞行器

地面车辆:
  vehicle.ground.boxcar.v1          — 民用车
  vehicle.emergency.suv.v1          — 应急 SUV
  vehicle.emergency.police_suv.v1   — 警车
  vehicle.emergency.ambulance.v1    — 救护车
  vehicle.service.box.v1            — 服务货车

行人:
  pedestrian.cityops.basic.v1   — 标准行人

路政/施工道具:
  prop.roadwork.barrier.v1            — 路障
  prop.roadwork.construction_fence.v1 — 施工围挡
  prop.roadwork.traffic_cone.v1       — 锥桶

交通设施:
  prop.traffic_control.police_sign.v1   — 警察指示牌
  prop.traffic_control.signal_light.v1  — 交通信号灯
  prop.incident.police_tape.v1          — 警戒带

其他道具:
  prop.service.delivery_bag.v1  — 配送袋
  prop.service.backpack.v1      — 背包
  prop.misc.phone.v1            — 手机
  prop.misc.umbrella.v1         — 雨伞

设施:
  facility.landing_pad.visible.v1   — 可见起降场
  facility.radio.base_tower.v1      — 无线通信塔
  facility.charger.cityops.v1       — 充电桩
  facility.barrier.basic            — 可编程路障

触发器/区域:
  trigger.no_fly.box.v1               — 禁飞区盒子
  trigger.hazard.construction.box.v1  — 施工危险区盒子
  trigger.hazard.generic.box.v1       — 通用危险区盒子

语义标记（无渲染）:
  semantic.landing_pad, semantic.spawn_zone, semantic.asset_anchor
```

### 可用动作类型

| 动作 | type | 关键参数 |
|------|------|----------|
| 生成实体 | `spawn_entity` | `entity_id, asset_id, position_enu_m, [rotation_deg], [visual_state]` |
| 移动实体 | `move_entity` | `entity_id, waypoints_enu_m, velocity_mps` |
| 移除实体 | `remove_entity` | `entity_id` |
| 设置视觉状态 | `set_visual_state` | `entity_id, visual_state: {mode, lights_on?}` |
| 播动画 | `play_animation` | `ped_id, animation_path, start_section, play_rate, loop_count` |
| 设天气 | `set_weather` | `profile, [overrides]` |
| 截图 | `capture_screenshot` | `camera_id` |
| 生成人群 | `spawn_crowd` | `group_id, count, spawn_origin_enu_m, [spawn_box_extent_cm], seed` |
| 清除人群 | `clear_crowd` | `group_id` |

### 可用触发类型

| 触发 | type | 关键参数 |
|------|------|----------|
| 绝对 tick | `tick` | `tick` |
| 天气状态 | `weather_state` | `parameter, operator, value, [sustain_ticks]` |
| 实体距离 | `entity_proximity` | `entity_a, entity_b, distance_m, [operator], [min_true_ticks]` |
| 事件完成 | `event_fired` | `event_id` |
| 组合条件 | `composite` | `operator: "AND"/"OR", children: [trigger_id, ...]` |

### 天气参数 (weather_state)

`parameter` 可选: `rain`, `fog`, `wind_speed`, `visibility_m`
`operator` 可选: `gte`, `lte`, `eq`, `gt`, `lt`
`value`: 浮点数，范围如下：

| 参数 | 范围 | 单位 |
|------|------|------|
| `rain` | 0.0–1.0 | 归一化强度 |
| `fog` | 0.0–1.0 | 归一化密度 |
| `wind_speed` | 0.0–30.0 | m/s |
| `visibility_m` | 0.0–20000.0 | 米 |

### 坐标系

- 地图: `donghu_road_topo` (武汉东湖)
- 本地坐标: ENU (East-North-Up)，原点在 `[lat=30.5609, lon=114.3627, alt=24.0]`
- 运行时转换: `CoordinateTransform(enabled=True, axis_mapping="XY_To_XY", yaw_deg=245, translation_enu_m=[...], scale_enu=[1,1,1])`
- 道路数据源: `traffic_bundle/lane_center_samples.csv`（lane 中心线采样点），`lane_meta.csv`（lane 元数据），`lane_connections.csv`（lane 拓扑连接）

### 可用的 scene_setup 放置模式

| 模式 | 字段 | 适用类别 |
|------|------|----------|
| `world_pose` | `position_enu_m, rotation_deg` | 通用 |
| `lane_anchor` | `edge_id, lane_index, longitudinal_s, lateral_offset_m` | 车辆、路障 |
| `sidewalk_anchor` | `lane_edge_id, longitudinal_s, offset_from_curb_m` | 行人、锥桶 |
| `crosswalk_anchor` | `crosswalk_id, side` | 行人过街 |
| `facade_anchor` | `building_id, outward_normal_enu, stand_off_m` | 建筑撞击 |
| `pad_anchor` | `pad_instance_id, approach_side` | landing pad |
| `anchor_ref` | `anchor_id, offset_enu_m, yaw_deg` | 依附于已有实体 |
| `box_volume` | `center_enu_m, extent_m` | 禁飞区、危险区 |
| `polygon_prism` | `polygon_enu_m, base_z_m, height_m` | 不规则区域 |

---

## 一、配置边界：按类别定义每类场景"必须包含什么"

GPT 在生成每个场景时必须确保以下内容全部出现在 spec 或 scene_setup 中。

### 类别 A：空域事件 (L1 — geofence, altitude, intrusion, congestion)

**场景列表**: L1-1_v1, L1-1_v2, L1-2_v1, L1-3_v1, L1-3_v2, L1-4_v1, L1-4_v2

**场景 setup 边界**:
- [ ] 至少 1 架 UAV 在起点（`world_pose`，z≥25m）
- [ ] 至少 1 个空域约束对象（`trigger.no_fly.box.v1` 或 `trigger.hazard.generic.box.v1`），用 `box_volume` 或 `polygon_prism` 放置
- [ ] 如果场景涉及边界侵入（L1-1），禁飞区必须指定边界坐标，不是空占位符
- [ ] UAV 起点必须在约束区域外的合理位置

**事件链边界（至少 3 步）**:
1. UAV 接近边界 → `move_entity` 从起点到接近边界位置（不要只用模板的 [58,24,32]）
2. 触发：`entity_proximity`（UAV ↔ 约束区域中心距离 < 阈值）**或** `tick` 作为替代
3. 冲突动作：UAV 减速/调头/悬停 → `move_entity` 或 `set_visual_state`（mode:"hover"）
4. 恢复：UAV 返回安全位置

**不应出现的错误**:
- 不要把 `$param.primary_id` 写成 `action_id` 前缀导致不展开
- 不要给 L1 场景套 rain gate
- 不要用统一 [58,24,32] waypoint

---

### 类别 B：基础设施故障 (L2 — comm station, GNSS, charger, landing pad, traffic signal)

**场景列表**: L2-1_v1, L2-1_v2, L2-2_v1, L2-2_v2, L2-3_v1, L2-3_v2, L2-4_v1, L2-4_v2, L2-5_v1

**场景 setup 边界**:
- [ ] 必须有基础设施实体实例：
  - L2-1（通信站）→ `facility.radio.base_tower.v1`，用 `world_pose` 放置
  - L2-2（GNSS）→ 同上 tower + 至少 1 架 UAV 在 urban canyon 场景（贴近建筑飞行）
  - L2-3（充电）→ `facility.charger.cityops.v1`，用 `world_pose` 放置
  - L2-4（起降场）→ `facility.landing_pad.visible.v1`，用 `pad_anchor` 或 `world_pose` 放置，**至少 2 架 UAV**
  - L2-5（信号灯）→ `prop.traffic_control.signal_light.v1`，用 `world_pose` 放在路口，**至少 2 辆车 + 1 辆警车**

**事件链边界**:
- L2-1: 通信站降级 → UAV 行为变化（不是只设 visual_state）→ 备用链路恢复
- L2-2: GNSS 信号异常 → UAV 偏离路线（实际 move）→ 视觉重定位 → 纠正
- L2-3: 充电桩不可用 → UAV 检查 → 重路由到备用桩
- L2-4: 两个 UAV 争用同一 pad → 优先级裁决 → 第二个 UAV 改道
- L2-5: 信号灯 all-red 故障 → 交通混乱 → 警察到场 → 手动指挥。**关键：primary 是 traffic_signal，secondary 是 police_unit。不要对信号灯做 move_entity。**

**不应出现的错误**:
- L2-5 不要对 traffic_signal 调用 `move_entity`
- L2-4 不要只有一个 UAV
- L2-2 不要让 UAV 在原地不动就宣称"偏离"
- 所有 L2 不要套 rain gate
- 不要把 `action_id` 写成 `move_$param.primary_id`

---

### 类别 C：动态约束 (L3 — roadwork, temporary NFZ, hazmat isolation)

**场景列表**: L3-1_v1, L3-2_v1, L3-2_v2, L3-3_v1, L3-3_v2

**场景 setup 边界**:
- [ ] L3-1（道路施工）必须包含在 scene_setup 中声明的：
  - `prop.roadwork.barrier.v1` × 3（一字排列，用 `lane_anchor` 沿 lane 一侧放置，lateral_offset_m=1.5）
  - `prop.roadwork.construction_fence.v1` × 2（路障两端封闭）
  - `prop.roadwork.traffic_cone.v1` × 5（沿施工区边界排列）
  - 1 辆车（`vehicle.ground.boxcar.v1`）停在被封闭 lane 上
  - 1 架 UAV 巡检
- [ ] L3-2（临时禁飞区）→ `trigger.no_fly.box.v1` 动态生成（不是预置静态 no_fly）
- [ ] L3-3（危化品隔离）→ `trigger.hazard.generic.box.v1` + 至少 2 个行人 + 1 辆救护车 + UAV

**事件链边界**:
- L3-1: barrier 出现（spawn）→ lane 封闭生效 → 车辆绕行 → UAV 巡检报告
- L3-2: 禁飞区宣布（spawn）→ UAV 接近触发 proximity → 改道
- L3-3: 危化品泄漏 → 隔离区声明 → 人员疏散 → 救护车到达 → UAV 从外部监测

**不应出现的错误**:
- L3-1 不要只设 barrier 的 visual_state 而不 spawn 实例
- L3-1、L3-3 不需要 rain gate
- 所有 L3 的 barrier/traffic_cone 位置必须用 `lane_anchor` 模式（lane 边缘投影）

---

### 类别 D：智能体交互 (L4 — UAV×UAV, UAV×建筑, 迫降, 落车, UAV×行人, 行人×车, 人群, 车×车, 救护车, AV 故障)

**场景列表**: L4-1_v1, L4-1_v2, L4-2_v1, L4-2_v2, L4-3_v1, L4-3_v2, L4-3_v3, L4-4_v1, L4-4_v2, L4-5_v1, L4-5_v2, L4-5_v3, L4-6_v1, L4-6_v2, L4-7_v1, L4-7_v2, L4-8_v1, L4-8_v2, L4-9_v1, L4-9_v2, L4-10_v1, L4-10_v2, L4-11_v1, L4-11_v2

**按子类别设置边界**:

#### D1. UAV×UAV 冲突 (L4-1)
- [ ] 2 架 UAV，起点在不同位置（水平距离 ≥ 15m），z 不同（如 25m 和 30m）
- [ ] 各自向对方方向移动（收敛轨迹）
- [ ] 触发: `entity_proximity`（两 UAV 距离 < 8m）或 `tick`
- [ ] 冲突: 一架悬停，另一架改道
- [ ] 恢复: 两架分离后恢复正常巡逻

#### D2. UAV×建筑撞击 (L4-2)
- [ ] 1 架 UAV + 1 栋建筑的 facade 参考点（用 `facade_anchor`）
- [ ] UAV 移动轨迹从远到近，最终 trajectory 的最近点距建筑 facade ≤ 3m
- [ ] 触发: `entity_proximity`（UAV ↔ facade 参考点距离）
- [ ] 冲突: 紧急避让/减速
- [ ] 恢复: 绕飞通过或返航
- [ ] 不能用 [58,24,32] 这种不存在建筑的坐标

#### D3. UAV 迫降 (L4-3)
- [ ] 1 架 UAV（故障状态） + 地面人群（`spawn_crowd` 或至少 2 个行人）
- [ ] UAV 从正常高度下降到低高度（z: 30→8→3）
- [ ] 触发: UAV 下降到 threshold 高度 + 群体 proximity
- [ ] 冲突: 人群闪避 + 应急响应
- [ ] 恢复: UAV 安全着陆

#### D4. UAV 落车顶 (L4-4)
- [ ] 1 架 UAV + 1 辆车（都在运动）
- [ ] UAV 轨迹必须和车辆轨迹有交点（不是平行线）
- [ ] 触发: `entity_proximity`（UAV ↔ 车辆距离 < 3m，z 差 < 2m）
- [ ] 冲突: 车辆急刹，UAV 残骸落地
- [ ] 不要用 vehicle 的 move 轨迹完全平行于 UAV

#### D5. UAV×行人接近 (L4-5)
- [ ] 1 架 UAV + 1 个行人
- [ ] UAV 从高处下降到行人头顶高度（z: 30→15→5）
- [ ] 触发: `entity_proximity`（UAV ↔ 行人，距离 < 5m）
- [ ] 冲突: UAV 悬停或拉升
- [ ] 不要用 `set_visual_state` 替代 UAV 的实际移动

#### D6. 行人×车辆 (L4-6, L4-7)
- [ ] L4-6: 行人 + 车辆，行人从 sidewalk 进入 roadway（`crosswalk_anchor` → `lane_anchor`）
- [ ] L4-7: 行人跌倒 + UAV 检测 + 响应
- [ ] 触发: `entity_proximity`（行人 ↔ 车 < 4m）
- [ ] 冲突: 车辆刹车
- [ ] 行人必须用 pedestrian.cityops.basic.v1
- [ ] L4-7 必须用 `play_animation` 播放跌倒动画

#### D7. 人群疏散 (L4-8)
- [ ] 用 `spawn_crowd` 生成 ≥ 8 人群
- [ ] 疏散触发 → 人群移动到安全区
- [ ] UAV 监控（`capture_screenshot`）
- [ ] 结束后 `clear_crowd`

#### D8. 车辆碰撞 (L4-9)
- [ ] 2 辆车，不同 lane，相向而行到同一路口
- [ ] 触发: `entity_proximity`（两车距离 < 6m）
- [ ] 冲突: 两车同时刹车

#### D9. 救护车优先 (L4-10)
- [ ] 1 辆救护车（`vehicle.emergency.ambulance.v1`）+ 至少 1 辆民用车
- [ ] 救护车 lights_on=true
- [ ] 民用车让行
- [ ] 不应只设 visual_state 不设 movement

#### D10. AV 传感器故障 (L4-11)
- [ ] 1 辆自动驾驶车 + 1 架 UAV
- [ ] AV 故障 → 安全停车 → 阻塞 lane → UAV 报告
- [ ] 不要只设 visual_state

**统一的 L4 不应出现的错误**:
- 所有 L4 不需要 rain gate（除非是 L5 天气场景）
- entity 必须在 scene_setup 中声明，不能在 event_script 里凭空引用
- `action_id` 不能包含 `$param.` 引用

---

### 类别 E：环境效应 (L5 — rain, fog, wind, light shift, temperature)

**场景列表**: L5-1_v1, L5-1_v2, L5-1_v3, L5-2_v1, L5-2_v2, L5-3_v1, L5-3_v2, L5-4_v1, L5-5_v1

**场景 setup 边界**:
- [ ] 至少 1 架 UAV + 选配 1 辆车/1 个行人
- [ ] 天气触发用 `weather_state`，不是 tick

**子类别**:

| 场景 | weather_state.parameter | 阈值 | 触发后动作 |
|------|------------------------|------|-----------|
| L5-1 (rain) | `rain` | gte 0.5 | UAV 降速，车辆湿滑减速，visibility 下降 |
| L5-2 (fog) | `fog` | gte 0.5 | visibility 骤降，视觉导航失效 |
| L5-3 (wind) | `wind_speed` | gte 12.0 | payload 摆动，UAV 姿态异常 |
| L5-4 (light) | `tick`（模拟黄昏时间） | — | camera 曝光不足，切换红外 |
| L5-5 (temp) | `tick`（模拟高温过程） | — | battery derating，UAV 航程缩短 |

**事件链边界**:
1. 天气条件满足（weather_state trigger）
2. 主体行为变化（降速/切换模式/改变姿态）
3. 连续劣化（更大的 weather intensity）
4. 恢复或终止

**不应出现的错误**:
- 这是唯一需要 weather_state trigger 的类别。其他 L1/L2/L3/L4/L6 如果不需要天气，不要画蛇添足加 rain gate
- L5-4 和 L5-5 不能用 `weather_state`（没有 light/temp 参数），必须用 `tick`

---

### 类别 F：数字层安全 (L6 — C2 loss, C2 degradation, GNSS spoofing, jamming, GCS intrusion)

**场景列表**: L6-1_v1, L6-1_v2, L6-2_v1, L6-2_v2, L6-3_v1, L6-3_v2, L6-4_v1, L6-4_v2, L6-5_v1, L6-5_v2

**场景 setup 边界**:
- [ ] 至少 1 架 UAV + tower/地面站对象
- [ ] L6-1 (C2 丢失): 1 架 UAV + 1 个 tower (facility.radio.base_tower.v1)，UAV RTH 动作
- [ ] L6-3 (GNSS spoofing): 1 架 UAV，偏离预设路线的轨迹（实际 offset ≥ 10m）
- [ ] L6-4 (jamming): 2 架 UAV，同时受影响
- [ ] L6-5 (GCS 入侵): 1 架 UAV + 地面站标记

**事件链边界**:
1. 数字层异常触发（tick 模拟 C2/Spoofing 发生时刻）
2. UAV 行为偏离（实际 move_entity 到错误位置）
3. 安全机制介入（RTH / 锁定 / 备用链路）
4. 恢复

**不应出现的错误**:
- 所有 L6 不需要 rain gate
- L6-3 spoofing 必须有实际的轨迹偏离（不是原地不动改 visual_state）
- L6-1 RTH 必须有从当前位置移动到 home 位置的 waypoint

---

### 类别 G：跨层事件链 (X — 多 ODD 层联动)

**场景列表**: X1, X2, X3, X4, X5, X6

**场景 setup 边界**:
- [ ] 至少涉及 2 个 ODD 层的实体和机制
- [ ] 每个 X 场景的事件链 ≥ 5 步

| 场景 | 涉及的层 | 实体 | 链 |
|------|---------|------|-----|
| X1 | rain(L5) → C2_loss(L6) → forced_landing(L4) | UAV + tower + crowd | rain 触发 → C2 劣化 → 迫降 → 人群反应 |
| X2 | GNSS_spoof(L6) → geofence_violation(L1) | UAV + NFZ | spoofing 开始 → 轨迹偏移 → 侵入 NFZ → 纠正 |
| X3 | ped_fall(L4) → emergency_response | ped + UAV + ambulance | 跌倒 → UAV 检测 → 救护车到达 |
| X4 | fog(L5) → uav_conflict(L4) | 2 UAV | 起雾 → visibility 下降 → 两 UAV 接近 → 紧急避让 |
| X5 | comm_failure(L2) → pad_contention(L2) | 2 UAV + 2 pad | station 故障 → 仅一个 pad 可用 → 优先级裁决 |
| X6 | crowd_evacuation(L4) → airspace_lockdown(L3) | crowd + UAV + NFZ | 疏散 → 空域锁定 → NFZ 声明 |

**不应出现的错误**:
- 跨层事件链的每一步必须在不同层有实际变化（不只是 log 不同）
- X1 的 rain 用 weather_state 触发，后续 C2 loss 用 composite 衔接（rain AND C2 条件满足）

---

## 二、scene_setup.json 模板

每个场景的 `scene_setup.json` 必须包含以下结构。这是生成 event_script 的前提——没有 scene_setup 的场景不算完整。

```jsonc
{
  "$schema": "scene_setup_v1",
  "scenario_id": "L4-2_v1",
  "description": "UAV navigation fault drives into building facade",

  "map_ref": {
    "map_id": "donghu_road_topo",
    "geo_reference": { "lat": 30.5609, "lon": 114.3627, "alt": 24.0 },
    "coordinate_frame": "ENU"
  },

  "entities": [
    {
      "entity_id": "uav_facade_fault_01",
      "logical_asset_id": "uav.inspect.quad.v1",
      "category": "uav",
      "placement_mode": "world_pose",
      "placement": {
        "position_enu_m": [55.0, 18.0, 30.0],
        "rotation_deg": { "yaw_deg": 225.0 }
      },
      "route_waypoints_enu_m": [
        [50.0, 20.0, 30.0],
        [46.0, 23.0, 22.0],
        [43.0, 25.0, 18.0]
      ],
      "initial_state": { "mode": "patrol", "velocity_mps": 8.0 },
      "activation_tick": 0
    },
    {
      "entity_id": "tower_facade_east",
      "logical_asset_id": "semantic.asset_anchor",
      "category": "facade_anchor",
      "placement_mode": "facade_anchor",
      "placement": {
        "building_id": "building_east_block_A",
        "outward_normal_enu": [1.0, -0.5, 0.0],
        "stand_off_m": 1.0,
        "position_enu_m": [43.0, 25.0, 15.0]
      },
      "initial_state": {},
      "activation_tick": 0
    }
  ],

  "spawn_sequencing": [
    { "entity_id": "tower_facade_east", "tick": 0 },
    { "entity_id": "uav_facade_fault_01", "tick": 0 }
  ],

  "weather_profile": {
    "initial": "clear",
    "transitions": []
  },

  "cameras": [
    {
      "camera_id": "demo_high_overview",
      "placement_mode": "world_pose",
      "placement": {
        "position_enu_m": [50.0, 20.0, 60.0],
        "rotation_deg": { "pitch_deg": -70.0, "yaw_deg": 0.0 }
      },
      "fov_deg": 90.0
    }
  ],

  "validation_rules": [
    {
      "rule": "proximity_check",
      "entity_a": "uav_facade_fault_01",
      "entity_b": "tower_facade_east",
      "at_tick": 450,
      "max_distance_m": 5.0,
      "description": "UAV must pass within 5m of facade at conflict tick"
    },
    {
      "rule": "waypoint_count_min",
      "entity_id": "uav_facade_fault_01",
      "min_count": 3,
      "description": "UAV must have at least 3 waypoints defining the approach trajectory"
    },
    {
      "rule": "entity_resolvable",
      "entity_id": "uav_facade_fault_01",
      "description": "Entity must exist in scene_setup before event_script references it"
    },
    {
      "rule": "asset_in_catalog",
      "entity_id": "uav_facade_fault_01",
      "logical_asset_id": "uav.inspect.quad.v1",
      "description": "Asset ID must match an entry in asset_catalog.json"
    }
  ]
}
```

### 语义放置模式速查

| 主体类别 | placement_mode | 关键字段 |
|----------|---------------|----------|
| UAV | `world_pose` | `position_enu_m: [x,y,z]`, z ≥ 20m |
| 车辆 on lane | `lane_anchor` | `edge_id, lane_index, longitudinal_s, lateral_offset_m` |
| 车辆 off lane | `world_pose` | `position_enu_m: [x,y,z]`, z=0 |
| 行人 on sidewalk | `sidewalk_anchor` | `lane_edge_id, longitudinal_s, offset_from_curb_m` |
| 行人 on crosswalk | `crosswalk_anchor` | `crosswalk_id, side` |
| UAV vs 建筑 | `facade_anchor` | `building_id, outward_normal_enu, stand_off_m` |
| 路障/锥桶/围挡 | `lane_anchor` | `edge_id, lane_index, longitudinal_s, lateral_offset_m` (offset = 路肩距离) |
| 禁飞区 | `box_volume` | `center_enu_m, extent_m` |
| 危险区 | `box_volume` 或 `polygon_prism` | `center + extent` 或 `polygon + base_z + height` |
| 信号灯 | `world_pose` | 路口坐标，z ≈ 4m |
| 人群 | `world_pose` | `spawn_origin_enu_m, spawn_box_extent_cm` |

---

## 三、event_script.json 模板

基于 SpecCompiler 的输出格式，以下是每个 event_script.json 必须满足的结构约束：

```jsonc
{
  "$schema": "event_script_v1",
  "scenario_id": "L4-2_v1",
  "description": "Navigation fault drives UAV into building facade",

  "parameters": {
    // 仅放"运行时可能变化"的可调参数
    // 实体 ID、坐标等硬数据应放在 scene_setup.json
    "approach_tick": 300,
    "conflict_tick": 450,
    "resolution_tick": 550
  },

  "triggers": [
    {
      "trigger_id": "trig_approach_phase",
      "type": "tick",
      "tick": 300
    },
    {
      "trigger_id": "trig_proximity_alert",
      "type": "entity_proximity",
      "entity_a": "uav_facade_fault_01",
      "entity_b": "tower_facade_east",
      "distance_m": 8.0,
      "operator": "lte",
      "min_true_ticks": 3
    },
    {
      "trigger_id": "trig_evasion_done",
      "type": "event_fired",
      "event_id": "evasion_maneuver"
    }
  ],

  "events": [
    {
      "event_id": "approach_phase",
      "trigger_ref": "trig_approach_phase",
      "priority": 1,
      "max_fire_count": 1,
      "actions": [
        {
          "action_id": "move_approach",
          "type": "move_entity",
          "entity_id": "uav_facade_fault_01",
          "waypoints_enu_m": [
            [50.0, 20.0, 30.0],
            [46.0, 23.0, 22.0]
          ],
          "velocity_mps": 8.0
        }
      ],
      "on_fire": { "emit_events": ["proximity_alert"] },
      "log_event": {
        "topic": "evt_L4-2_v1_approach",
        "category": "uav_mission",
        "title": "UAV approaching building facade",
        "severity": "info",
        "overlay": "uav_mission",
        "target_ids": ["uav_facade_fault_01", "tower_facade_east"]
      }
    },
    {
      "event_id": "proximity_alert",
      "trigger_ref": "trig_proximity_alert",
      "priority": 2,
      "max_fire_count": 1,
      "actions": [
        {
          "action_id": "move_evasion",
          "type": "move_entity",
          "entity_id": "uav_facade_fault_01",
          "waypoints_enu_m": [
            [46.0, 23.0, 22.0],
            [40.0, 26.0, 35.0]
          ],
          "velocity_mps": 12.0
        },
        {
          "action_id": "capture_conflict",
          "type": "capture_screenshot",
          "camera_id": "demo_high_overview"
        }
      ],
      "log_event": {
        "topic": "evt_L4-2_v1_conflict",
        "category": "uav_mission",
        "title": "Evasion maneuver near facade",
        "severity": "warning",
        "overlay": "uav_mission",
        "target_ids": ["uav_facade_fault_01", "tower_facade_east"]
      }
    }
  ]
}
```

### 强制约束

1. **`action_id` 必须是具体字符串**，不能包含 `$param.` 引用。
2. **`entity_id` 必须能在 `scene_setup.json` 的 entities 列表中找到**。
3. **`waypoints_enu_m`** 每个场景的坐标必须根据场景类别和语义放置模式定制，不能复用 [58,24,32] 模板值。
4. **weather_state trigger** 仅在 L5（环境）和 X1（跨层含天气）场景中使用。其他 L1/L2/L3/L4/L6 场景不要加 rain/fog/wind gate。
5. **proximity trigger** 的 distance_m 必须根据主体实际尺寸和场景空间尺度取值：
   - UAV↔UAV: 8–15m
   - UAV↔building: 5–10m
   - UAV↔vehicle: 5–10m
   - UAV↔pedestrian: 3–8m
   - vehicle↔vehicle: 4–8m
   - vehicle↔pedestrian: 3–6m
6. **composite trigger** 的 children 引用的 trigger_id 必须在同一文件中已定义。

---

## 四、GPT 提示词

以下是直接可以交给 GPT 的提示词。在发送之前确保把本文档前面所有内容作为上下文一起发送。

---

```
你是一个 UE5 城市级低空场景配置专家。你的任务是重新生成 Dataset/scenarios/ 下 64 个 P0 场景的完整配置。

## 背景

当前 repo (AeroWorld DynamicCityCreator) 有一套 UE5 资产放置框架、事件脚本解释器和场景编译管线，但 handcrafted 场景层被一个单一 Python 模板批量生成导致以下问题：
1. 所有场景共用同一套 waypoints [58,24,32] → [54,22,30] → [50,20,24] → [56,26,32]，不区分 UAV、车辆、行人
2. 所有场景都套了 rain 门控，即使非天气场景也是如此
3. entity ID 只是名字占位符，没有解析到 asset catalog 中的真实资产
4. 缺乏场景专属的语义放置和空间验证
5. action_id 包含 $param 引用，展开不正确

## 你的任务

为以下 64 个场景每个生成两个文件：

### 文件 1: scene_setup.json
声明场景中所有实体的实例化信息。模板见上文"二、scene_setup.json 模板"。
关键要求：
- 每个 entity 必须使用正确的 logical_asset_id（只能从"零、基础设施清单"中选择）
- 每个 entity 必须使用正确的语义放置模式（lane_anchor / sidewalk_anchor / facade_anchor / pad_anchor / world_pose / box_volume / polygon_prism）
- 坐标不能千篇一律，不同场景必须有不同的合理位置
- 包含至少 2 条 validation_rules
- 包含至少 1 个 camera 定义

### 文件 2: event_script.json
场景的时间逻辑和事件链。模板见上文"三、event_script.json 模板"。
关键要求：
- 使用 trigger 类型必须匹配场景类别（L5 用 weather_state，不需要天气的场景不要加 rain gate）
- proximity trigger 的 distance_m 必须合理（不能 0.2m 或 999m）
- 每个 action 的 action_id 是具体字符串，不是 $param.xxx
- entity_id 必须在 scene_setup.json 中有声明
- waypoints 必须根据实体类型设置合理的 z 值（UAV z≥20m，车辆 z≈0，行人 z≈0）

## 场景清单

以下是 64 个场景，按类别分组。每个场景的 spec.py 已存在于 `Dataset/scenarios/<path>/` 下。

### A — 空域 L1 (7 个)
L1-1_v1  L1-1_v2    geofence violation, RTH
L1-2_v1             altitude deviation from assigned corridor
L1-3_v1  L1-3_v2    noncooperative intruder in operational airspace
L1-4_v1  L1-4_v2    corridor congestion, multi-UAV

### B — 基础设施 L2 (9 个)
L2-1_v1  L2-1_v2    comm station failure, service degradation
L2-2_v1  L2-2_v2    GNSS urban canyon multipath drift
L2-3_v1  L2-3_v2    charger unavailable, reroute to backup
L2-4_v1  L2-4_v2    landing pad emergency contention
L2-5_v1             traffic signal all-red, police intervention

### C — 动态约束 L3 (5 个)
L3-1_v1             road construction, lane closure, detour
L3-2_v1  L3-2_v2    temporary no-fly zone mid-operation
L3-3_v1  L3-3_v2    hazmat leak, isolation zone, evacuation

### D — 智能体 L4 (24 个)
L4-1_v1  L4-1_v2    UAV-UAV converging conflict
L4-2_v1  L4-2_v2    UAV building/structure strike
L4-3_v1  L4-3_v2    UAV forced landing near crowd
L4-4_v1  L4-4_v2    UAV falls onto ground vehicle
L4-5_v1  L4-5_v2 L4-5_v3  UAV low-altitude pedestrian near-miss
L4-6_v1  L4-6_v2    pedestrian jaywalk vehicle conflict
L4-7_v1  L4-7_v2    pedestrian fall, UAV detection, response
L4-8_v1  L4-8_v2    crowd evacuation, UAV monitoring
L4-9_v1  L4-9_v2    vehicle-vehicle intersection conflict
L4-10_v1 L4-10_v2   ambulance priority passage
L4-11_v1 L4-11_v2   AV sensor failure, safe stop, lane blockage

### E — 环境 L5 (9 个)
L5-1_v1 L5-1_v2 L5-1_v3  rain cascade multi-agent
L5-2_v1 L5-2_v2          sudden fog visual failure
L5-3_v1 L5-3_v2          crosswind payload swing
L5-4_v1                  dusk light shift, camera underexposure
L5-5_v1                  high temperature battery derating

### F — 数字层 L6 (10 个)
L6-1_v1 L6-1_v2  C2 link loss, return-to-home
L6-2_v1 L6-2_v2  intermittent C2 degradation, delay/slowdown
L6-3_v1 L6-3_v2  GNSS spoofing, route hijack
L6-4_v1 L6-4_v2  wideband jamming, multi-UAV
L6-5_v1 L6-5_v2  GCS intrusion, abnormal UAV behavior

### G — 跨层 X (6 个)
X1   rain → C2 loss → forced landing
X2   GNSS spoof → geofence violation
X3   pedestrian fall → emergency response chain
X4   fog → multi-UAV conflict
X5   comm failure → pad contention
X6   crowd evacuation → airspace lockdown

## 输出格式

对每个场景，生成：

```
=== L4-2_v1 ===
目录: Dataset/scenarios/L4_agents/collision/L4-2_v1/

--- scene_setup.json ---
{ ... }

--- event_script.json ---
{ ... }
```

## 自检清单（生成每个场景后必须逐项确认）

- [ ] 所有 entity_id 能在 scene_setup.json 的 entities 中找到
- [ ] 所有 logical_asset_id 来自基础清单，没有编造
- [ ] 没有 rain gate 出现在 L1/L2/L3/L4/L6 场景
- [ ] L5 场景必须有 weather_state trigger
- [ ] 没有 action_id 包含 $param. 引用
- [ ] waypoints 的 z 坐标匹配实体类别（UAV z≥20，地面 z≈0，信号灯 z≈4）
- [ ] 没有对 traffic_signal 或 base_tower 调用 move_entity
- [ ] proximity distance 在合理范围内
- [ ] 每个场景有至少 2 条 validation_rules
- [ ] L2-5 的 primary 是 traffic_signal，secondary 是 police，有至少 2 辆车
- [ ] L3-1 有至少 3 个 barrier + 2 个 fence + 5 个 cone（spawn 动作）
- [ ] L4-3 有 crowd 或至少 2 个明确的行人实体
- [ ] L4-10 ambulance 有 lights_on: true
- [ ] X 类场景的事件链 ≥ 5 步且跨至少 2 个 ODD 层

请开始逐个生成所有 64 个场景。
```

---

## 五、你自己后续的人工审核重点

当 GPT 输出后，你需要重点检查以下内容（这些是 GPT 最容易出错的地方）：

1. **坐标是否合理**：打开 `traffic_bundle/lane_center_samples.csv` 和 `lane_meta.csv`，抽样验证 GPT 给的 lane_anchor 坐标是否在真实 lane 上
2. **weather_state 误用**：搜索所有非 L5/X1 场景的 `weather_state`，确认没有残留 rain gate
3. **entity 交叉引用**：对每个 event_script.json 提取所有 `entity_id`，在对应的 `scene_setup.json` 中确认存在
4. **asset_id 合法性**：对每个 `logical_asset_id` 在 `asset_catalog.json` 中 grep 确认存在
5. **action_id 残留**：搜索 `$param.` 关键词，确认全部清除
6. **路障/锥桶/围挡的 placement 语义**：确认 L3-1 等场景使用了 `lane_anchor` 而非裸 `world_pose`
7. **人群疏散**：确认 L4-8 使用了 `spawn_crowd` 而非逐个 spawn 行人
8. **空域约束对象**：确认 L1/L3 场景的 NFZ/hazard_zone 使用了 `box_volume` 或 `polygon_prism` 放置模式
