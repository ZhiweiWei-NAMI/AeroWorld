const TICK_MIN = 0;
const TICK_MAX = 900;
const TICK_STEP = 5;
const CONTEXT_WINDOW = 25;
const CONTEXT_LIMIT = 60;

const PREDICATE_LABELS = Object.freeze({
  "airspace.aircraft_airborne": "飞行器已离地",
  "airspace.aircraft_at_home_pad": "飞行器位于起降点",
  "airspace.aircraft_in_landing_zone": "飞行器进入着陆区",
  "airspace.aircraft_inside_protected_airspace": "飞行器位于保护空域",
  "airspace.aircraft_inside_restricted_airspace": "飞行器进入禁限空域",
  "airspace.aircraft_is_non_cooperative": "飞行器未配合管制",
  "airspace.aircraft_moving_in_corridor": "飞行器正在航路内飞行",
  "airspace.aircraft_occupies_airspace_corridor": "飞行器占用航路",
  "airspace.aircraft_outside_assigned_altitude_band": "飞行高度超出指定范围",
  "airspace.aircraft_stationary_in_corridor": "飞行器在航路内悬停",
  "airspace.aircraft_touchdown": "飞行器已着陆",
  "airspace.airspace_corridor_over_capacity": "航路容量已超限",
  "communication.channel_bandwidth_over_allocated": "通信带宽分配超限",
  "communication.communication_session_interrupted": "通信会话中断",
  "communication.data_flow_latency_above_threshold": "数据传输延迟过高",
  "communication.handover_in_progress": "通信链路正在切换",
  "communication.link_degraded": "通信链路质量下降",
  "communication.link_unavailable": "通信链路不可用",
  "communication.message_transmission_failed": "消息发送失败",
  "communication.packet_loss_above_threshold": "通信丢包率过高",
  "communication.retransmission_in_progress": "消息正在重传",
  "compute.compute_node_unavailable": "计算节点不可用",
  "compute.compute_resource_over_allocated": "计算资源分配超限",
  "compute.compute_task_deadline_missed": "计算任务超时",
  "compute.compute_task_execution_failed": "计算任务执行失败",
  "compute.compute_task_migrating": "计算任务正在迁移",
  "compute.compute_task_offloaded": "计算任务已卸载",
  "compute.compute_task_queued": "计算任务正在排队",
  "emergency.aircraft_in_return_to_home": "飞行器正在返航",
  "emergency.aircraft_in_safe_hold": "飞行器处于安全悬停",
  "emergency.emergency_isolation_active": "紧急隔离已启用",
  "emergency.emergency_vehicle_priority_active": "应急车辆优先通行",
  "emergency.evacuation_in_progress": "人员正在疏散",
  "emergency.mission_aborted": "任务已中止",
  "emergency.temporary_no_fly_zone_active": "临时禁飞区已启用",
  "environment.fog_active": "现场有雾",
  "environment.low_illumination_active": "现场光照不足",
  "environment.rain_active": "现场正在降雨",
  "environment.temperature_outside_operating_range": "温度超出工作范围",
  "environment.visibility_below_threshold": "能见度不足",
  "environment.wind_speed_above_threshold": "风速过高",
  "facility.aircraft_battery_low": "飞行器电量不足",
  "facility.charging_queue_over_capacity": "充电队列已超容量",
  "facility.charging_station_unavailable": "充电站不可用",
  "facility.landing_pad_contention": "多个飞行器争用起降点",
  "facility.landing_pad_occupied": "起降点已被占用",
  "ground.pedestrian_in_crosswalk": "行人位于人行横道",
  "ground.road_segment_closed": "道路已封闭",
  "ground.traffic_signal_malfunction_active": "交通信号灯故障",
  "ground.vehicle_moving_in_lane": "车辆正在车道内行驶",
  "ground.vehicle_queue_exceeds_length": "车辆排队过长",
  "ground.vehicle_stopped_in_lane": "车辆停在车道内",
  "positioning.aircraft_deviates_from_planned_route": "飞行器偏离计划航线",
  "positioning.gnss_quality_degraded": "卫星定位质量下降",
  "positioning.gnss_spoofing_active": "飞行器受到定位欺骗信号影响",
  "positioning.navigation_error_exceeds_tolerance": "导航误差超限",
  "regulation.pedestrian_has_crosswalk_priority": "行人享有人行横道优先权",
  "regulation.traffic_signal_applies_to_vehicle": "交通信号适用于该车辆",
  "regulation.vehicle_crosses_stop_line_on_red": "车辆红灯越过停止线",
  "regulation.vehicle_exceeds_posted_speed": "车辆超过限速",
  "regulation.vehicle_violates_following_distance": "车辆跟车距离不足",
  "regulation.vehicle_violates_right_of_way": "车辆未按路权让行",
  "safety.aircraft_near_building_structure": "飞行器距离建筑过近",
  "safety.aircraft_near_ground_vehicle": "飞行器距离车辆过近",
  "safety.aircraft_near_pedestrian": "飞行器距离行人过近",
  "safety.aircraft_pair_below_separation": "两架飞行器距离过近",
  "safety.crowd_evacuation_active": "人群疏散已启动",
  "safety.forced_landing_in_populated_area": "飞行器在人群区域迫降",
  "safety.vehicle_pedestrian_conflict": "车辆与行人距离过近",
  "security.communication_jamming_impact_active": "通信受到干扰",
  "security.control_command_integrity_violated": "控制指令完整性异常",
  "security.ground_control_station_compromised": "地面控制站已失陷",
  "security.security_control_unavailable": "安全控制不可用",
  "security.unauthorized_command_received": "收到未授权指令",
  "utm.airspace_authorization_active": "空域授权有效",
  "utm.airspace_authorization_revoked": "空域授权已撤销",
  "utm.deconfliction_active": "空中冲突协调已启动",
  "utm.flight_plan_approved": "飞行计划已批准",
  "utm.operational_intent_conflict": "运行意图发生冲突",
});

const EVENT_FAMILY_LABELS = Object.freeze({
  uav_takeoff: "飞行器起飞",
  uav_landing: "飞行器着陆",
  restricted_region_intrusion: "飞行器进入禁限空域",
  altitude_excursion: "飞行高度偏离",
  uav_separation_conflict: "飞行器间隔冲突",
  uav_pedestrian_near_miss: "飞行器与行人险情",
  uav_vehicle_conflict: "飞行器与车辆冲突",
  vehicle_pedestrian_conflict: "车辆与行人冲突",
  corridor_congestion: "航路拥堵",
  structure_near_miss: "飞行器接近建筑",
  c2_degradation_loss: "通信链路异常",
  gnss_signal_degradation: "卫星定位质量下降",
  gnss_spoofing_effect: "定位欺骗影响",
  charger_unavailable: "充电站不可用",
  pad_contention: "起降点争用",
  signal_malfunction: "交通信号灯故障",
  road_closure: "道路封闭",
  crowd_evacuation: "人群疏散",
  rain_fog_environment: "天气环境变化",
  high_temp_battery_derating: "高温导致电池降额",
  jamming_gcs_intrusion: "通信干扰或控制站入侵",
  charging_queue_over_capacity: "充电队列超容量",
});

const INTENT_LABELS = Object.freeze({
  abort: "中止任务",
  airspace_boundary_conflict: "空域边界冲突",
  all_eight_explicit_cohort_members_satisfy_the_physical_safe_zone_dwell_threshold: "8 名人员均已安全撤离",
  altitude_deviation: "飞行高度偏离",
  ambulance: "救护车到场",
  arbitration: "执行协调裁决",
  avoid_or_rth: "避让或返航",
  both_target_pedestrians_satisfy_the_physical_safe_zone_dwell_threshold: "两名行人均已安全撤离",
  charger_decision: "选择充电设施",
  charging_service: "执行充电服务",
  clear: "风险解除",
  clearance: "获得通行许可",
  collision_or_near_miss: "碰撞或险情",
  context: "环境条件变化",
  corridor_congestion: "航路拥堵",
  crosswind_threshold_reached: "侧风达到阈值",
  crowd_evacuation: "人群疏散",
  degraded_route: "航路条件下降",
  detour: "执行绕行",
  digital_anomaly: "数字系统异常",
  drift: "航迹漂移",
  facade_proximity: "接近建筑外墙",
  fast_intruder_crossing: "高速闯入者穿越",
  forced_landing: "执行迫降",
  hazard_hold: "风险持续",
  inspect: "执行巡检",
  intruder_conflict: "闯入冲突",
  ir_switch: "切换红外模式",
  isolation: "执行隔离处置",
  jaywalk: "行人违规横穿",
  landing_or_terminal_resolution: "着陆或完成最终处置",
  low_alt_inspect: "执行低空巡检",
  offset: "航迹偏移",
  pad_contention: "起降点冲突",
  patrol_resumption: "恢复巡航",
  payload_swing: "载荷摆动",
  police_arrives_for_manual_traffic_control: "警员到场人工疏导",
  recovery_or_restore: "恢复正常状态",
  retreat: "撤离风险区域",
  safe_hold: "安全悬停",
  safe_stop: "安全停车",
  slowdown: "执行减速",
  temporary_no_fly_zone_declared_mid_operation: "临时禁飞区启用",
  the_affected_uav_exits_the_just_activated_no_fly_region_then_routes_around: "飞行器驶离禁飞区并绕行",
  traffic_or_facility_incident: "交通或设施故障",
  uav_checks_the_unavailable_charger: "飞行器检查不可用充电桩",
  uav_confirms_the_fallen_pedestrian_after_governed_3d_proximity_dwell: "飞行器确认倒地行人",
  uav_convergence: "两架飞行器逐渐接近",
  uav_descends_from_30m_to_low_height: "飞行器下降至低空",
  uav_detect: "飞行器发现异常",
  uav_fault_appears_near_a_ground_crowd: "飞行器在人群附近发生故障",
  uav_follows_abnormal_command_path: "飞行器执行异常指令航迹",
  uav_moves_above_the_fallen_pedestrian_before_detection_is_confirmed: "飞行器移动至倒地行人上方",
  uav_takes_off_from_visible_home_pad_and_enters_mission_airspace: "飞行器起飞并进入任务空域",
  vehicle_brake: "车辆制动",
  weather_degradation: "天气条件恶化",
  yield: "执行让行",
});

const TRUTH_LABELS = Object.freeze({
  true: "成立",
  false: "未成立",
  unknown: "数据不足",
  out_of_scope: "不适用",
});

const DIRECTION_LABELS = Object.freeze({ rising: "出现", falling: "解除" });

const STATE_VALUE_LABELS = Object.freeze({
  available: "可用",
  unavailable: "不可用",
  visible: "可见",
  hidden: "不可见",
  active: "启用",
  inactive: "未启用",
  submit_to_ue: "已提交渲染",
  authoritative_input: "权威输入",
  landing_pad: "起降点",
  facility: "设施",
  uav: "飞行器",
  vehicle: "车辆",
  pedestrian: "行人",
  airspace_corridor: "航路",
  takeoff: "起飞中",
  airborne: "飞行中",
  hover: "悬停",
  landed: "已着陆",
});

const PREDICATE_DOMAIN_LABELS = Object.freeze({
  airspace: "空域状态",
  communication: "通信状态",
  compute: "计算状态",
  emergency: "应急状态",
  environment: "环境状态",
  facility: "设施状态",
  ground: "地面交通状态",
  positioning: "定位状态",
  regulation: "交通规则",
  safety: "安全状态",
  security: "安防状态",
  utm: "空中交通管理",
});

const PREDICATE_MEASUREMENT_LABELS = Object.freeze({
  "airspace.aircraft_outside_assigned_altitude_band": "飞行高度",
  "communication.data_flow_latency_above_threshold": "传输延迟",
  "communication.packet_loss_above_threshold": "丢包率",
  "environment.temperature_outside_operating_range": "环境温度",
  "environment.visibility_below_threshold": "能见度",
  "environment.wind_speed_above_threshold": "风速",
  "facility.aircraft_battery_low": "剩余电量",
  "facility.charging_queue_over_capacity": "充电请求数",
  "ground.vehicle_queue_exceeds_length": "排队长度",
  "positioning.navigation_error_exceeds_tolerance": "导航误差",
  "regulation.vehicle_exceeds_posted_speed": "车辆速度",
  "regulation.vehicle_violates_following_distance": "跟车距离",
  "safety.aircraft_near_building_structure": "飞行器与建筑距离",
  "safety.aircraft_near_ground_vehicle": "飞行器与车辆距离",
  "safety.aircraft_near_pedestrian": "飞行器与行人距离",
  "safety.aircraft_pair_below_separation": "两机距离",
  "safety.vehicle_pedestrian_conflict": "车辆与行人距离",
});

const FILE_LABELS = Object.freeze({
  "L0 state": "原始状态数据",
  evidence: "判定数值证据",
  "predicate transitions": "状态变化记录",
  "graph delta": "语义关系变化",
  "semantic events": "识别事件记录",
  "physical trace": "场景事件日志",
  "event realization": "事件执行明细",
});

const ACCESS_LABELS = Object.freeze({
  "tick offset": "按时刻定位",
  "truth-id offset": "按证据定位",
  cached: "已载入",
  "small-log cache": "日志已载入",
});

const CLOSURE_LABELS = Object.freeze({ PASS: "已验收" });

const MAP_ENTITY_GROUPS = Object.freeze({
  uav: { label: "无人机", color: "#087f8c" },
  vehicle: { label: "车辆", color: "#245b84" },
  pedestrian: { label: "行人", color: "#c36816" },
  facility: { label: "固定设施", color: "#2f7d4f" },
  traffic_light: { label: "交通灯", color: "#8d3d74" },
  context: { label: "场景标记", color: "#6c5d48" },
});

const MAP_STATIC_LAYERS = Object.freeze([
  { label: "道路", className: "road" },
  { label: "建筑", className: "building" },
  { label: "飞行航路", className: "corridor" },
  { label: "场景边界", className: "boundary" },
  { label: "禁飞区", className: "no-fly" },
]);

function predicateLabel(predicateId) {
  return PREDICATE_LABELS[predicateId] || String(predicateId).split(".").at(-1).replaceAll("_", " ");
}

function eventFamilyLabel(eventFamilyId) {
  return EVENT_FAMILY_LABELS[eventFamilyId] || String(eventFamilyId ?? "关联事件").replaceAll("_", " ");
}

function eventNodeLabel(node) {
  const id = String(node?.id || node?.event_id || "");
  const suffix = id.endsWith("_a") ? "A" : id.endsWith("_b") ? "B" : "";
  if (id.includes("converging_approach")) return "两机开始接近";
  if (id.includes("conflict_resolution")) return "两机执行避让";
  if (id.includes("resume_patrol")) return "两机开始分离并恢复巡航";
  if (id.includes("takeoff")) return suffix ? `${suffix} 机起飞` : "飞行器起飞";
  if (id.includes("landing")) return suffix ? `${suffix} 机任务结束，返航着陆` : "任务结束，飞行器返航着陆";
  return INTENT_LABELS[node?.intent] || EVENT_FAMILY_LABELS[node?.event_family_id] || "场景事件";
}

function friendlyEntityName(entityId) {
  const id = String(entityId || "");
  if (id.includes("uav") && id.endsWith("_a")) return "飞行器 A";
  if (id.includes("uav") && id.endsWith("_b")) return "飞行器 B";
  if (id.startsWith("pad_home_")) return "主起降点";
  if (id.startsWith("pad_")) return "起降点";
  if (id.includes("charger")) return "充电设施";
  if (id.includes("corridor")) return "飞行航路";
  return id;
}

function subjectKindLabel(kind) {
  return ({
    uav: "飞行器",
    vehicle: "车辆",
    pedestrian: "行人",
    facility: "设施",
    corridor: "航路",
    airspace_corridor: "航路",
    region: "区域",
    building: "建筑",
    signal: "交通信号灯",
  })[kind] || "场景对象";
}

function directionLabel(direction) {
  return DIRECTION_LABELS[direction] || "变化";
}

function truthLabel(value) {
  return TRUTH_LABELS[value] || String(value ?? "数据不足");
}

function tickLabel(value) {
  return `第 ${value} 刻`;
}

function compactTickLabel(value) {
  return `${value}刻`;
}

function predicateDomainLabel(predicateId) {
  return PREDICATE_DOMAIN_LABELS[String(predicateId).split(".", 1)[0]] || "场景状态";
}

function entityLabel(bindings) {
  const ids = [...new Set(Object.values(bindings || {}).filter((value) => typeof value === "string" && value))];
  return ids.length ? ids.join(" · ") : "无关联对象";
}

function fieldLabel(field) {
  const normalized = String(field).split(".").at(-1);
  const labels = {
    "position_enu_m[0]": "东向坐标 X",
    "position_enu_m[1]": "北向坐标 Y",
    "position_enu_m[2]": "高度坐标 Z",
    "velocity_enu_mps[0]": "东向速度 Vx",
    "velocity_enu_mps[1]": "北向速度 Vy",
    "velocity_enu_mps[2]": "垂直速度 Vz",
    "assigned_landing_zone_pose_enu_m[0]": "着陆区东向坐标 X",
    "assigned_landing_zone_pose_enu_m[1]": "着陆区北向坐标 Y",
    "assigned_landing_zone_pose_enu_m[2]": "着陆区高度 Z",
    "home_pad_pose_enu_m[0]": "起降点东向坐标 X",
    "home_pad_pose_enu_m[1]": "起降点北向坐标 Y",
    "home_pad_pose_enu_m[2]": "起降点高度 Z",
    z_agl_m: "离地高度",
    speed_mps: "速度",
    assigned_altitude_m: "指定飞行高度",
    altitude_deviation_m: "高度偏差",
    aircraft_inside_corridor: "位于指定航路",
    cooperative_flag: "配合管制",
    rth_mode_active: "返航模式",
    restricted_region_active: "禁限空域启用",
    boundary_margin_m: "边界余量",
    minimum_restricted_boundary_distance_m: "距禁限空域边界",
    xy_distance_to_home_pad_m: "距起降点水平距离",
    xy_distance_to_assigned_landing_zone_m: "距指定着陆区水平距离",
    home_pad_id: "起降点",
    home_pad_ontology_class_id: "起降点类型",
    assigned_landing_zone_id: "指定着陆区",
    assigned_landing_zone_ontology_class_id: "着陆区类型",
    containing_corridor_ids: "所在航路",
    entity_role: "任务角色",
    local_ground_reference_z_m: "地面参考高度",
    takeoff_ground_z_m: "起飞点地面高度",
    nearest_aircraft_distance_m: "最近飞行器距离 (m)",
    nearest_aircraft_id: "最近飞行器",
    nearest_building_distance_m: "最近建筑距离 (m)",
    nearest_building_id: "最近建筑",
    nearest_ground_vehicle_distance_m: "最近地面车辆距离 (m)",
    nearest_ground_vehicle_id: "最近地面车辆",
    nearest_pedestrian_distance_m: "最近行人距离 (m)",
    nearest_pedestrian_id: "最近行人",
    nearest_population_distance_m: "最近人群距离 (m)",
    nearest_population_id: "最近人群",
    nearest_vehicle_distance_m: "最近车辆距离 (m)",
    nearest_vehicle_id: "最近车辆",
    state: "当前状态",
    activity_type: "活动状态",
    availability: "可用状态",
    service_capacity: "服务容量",
    capacity: "容量",
    entity_category: "主体类别",
    entity_kind: "主体类型",
    entity_type: "数据类型",
    label_class: "标注类别",
    visibility_state: "可见状态",
    submission_state: "渲染提交状态",
    global_roster: "全局固定名录",
    offstage: "位于场外",
    runtime_visible: "当前可见",
    activation_tick: "启用时刻",
    yaw_deg: "偏航角",
    pitch_deg: "俯仰角",
    roll_deg: "横滚角",
  };
  return labels[field] || labels[normalized] || normalized.replaceAll("_", " ");
}

function formatValue(value) {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(2).replace(/\.00$/, "");
  }
  if (typeof value === "boolean") return value ? "是" : "否";
  if (value === null || value === undefined) return "无数据";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

function recordMeasurements(record) {
  if (Array.isArray(record.measurements)) return record.measurements;
  return Object.entries(record.numeric_values || {}).map(([path, value]) => ({ path, value, unit: "" }));
}

function measurementLabel(record, path) {
  return PREDICATE_MEASUREMENT_LABELS[record.metric_id] || fieldLabel(path);
}

function measurementValue(measurement) {
  const unit = measurement.unit && measurement.unit !== "scalar" ? ` ${measurement.unit}` : "";
  return `${formatValue(measurement.value)}${unit}`;
}

function stateMeasurementValue(measurement) {
  const value = measurement.value;
  const rendered = typeof value === "string" && STATE_VALUE_LABELS[value]
    ? STATE_VALUE_LABELS[value]
    : typeof value === "number" && !Number.isInteger(value)
      ? value.toFixed(6).replace(/0+$/, "").replace(/\.$/, "")
      : formatValue(value);
  const unit = measurement.unit && measurement.unit !== "scalar" ? ` ${measurement.unit}` : "";
  return `${rendered}${unit}`;
}

function entityIdsFromContext(context) {
  return [...new Set([
    ...(context?.selection?.entities || []),
    ...(context?.physical_logs || []).flatMap((record) => record.target_ids || []),
  ].filter(Boolean))];
}

function semanticMilestoneLabel(record) {
  const label = predicateLabel(record.predicate_id);
  return record.direction === "falling" ? `${label}已解除` : `${label}出现`;
}

function chainSemanticMilestones(chain, frame) {
  const authoredTicks = new Set(chain.nodes.map((node) => Number(node.dispatch_tick)));
  const relatedEventIds = new Set(
    (frame.semantic_milestones || [])
      .filter((record) => record.phase_kind === "onset" && authoredTicks.has(Number(record.event_trigger_tick)))
      .map((record) => record.event_id),
  );
  return (frame.semantic_milestones || []).filter((record) => relatedEventIds.has(record.event_id));
}

function processFocusForMilestone(frame, milestone) {
  if (!milestone) return null;
  const phases = (frame.semantic_milestones || []).filter((record) => record.event_id === milestone.event_id);
  const ticks = phases.map((record) => Number(record.tick));
  return {
    eventId: milestone.event_id,
    label: eventFamilyLabel(milestone.event_family_id),
    startTick: Math.min(...ticks),
    endTick: Math.max(...ticks),
    entityIds: [...new Set(phases.flatMap((record) => Object.values(record.bindings || {})).filter(Boolean))],
  };
}

function primaryEvidenceSummary(context) {
  const ticks = context.transition
    ? new Set([context.transition.from_tick, context.transition.to_tick])
    : new Set([context.tick]);
  const points = context.evidence_records
    .filter((record) => ticks.has(record.tick))
    .flatMap((record) => recordMeasurements(record).map((measurement) => ({ record, measurement })))
    .filter(({ measurement }) => typeof measurement.value === "number")
    .sort((left, right) => left.record.tick - right.record.tick);
  if (!points.length) return `${context.counts.evidence_records} 条判定证据`;
  const first = points[0];
  const last = points.at(-1);
  const label = measurementLabel(first.record, first.measurement.path);
  if (first.record.tick !== last.record.tick) {
    return `${label}：${measurementValue(first.measurement)} → ${measurementValue(last.measurement)}`;
  }
  return `${label}：${measurementValue(last.measurement)}`;
}

function physicalActionSummary(record) {
  const actions = record?.raw?.action_realizations || [];
  return actions.flatMap((action) => {
    const entity = friendlyEntityName(action.entity_id);
    if (action.action_type === "set_visual_state" && action.expected_state === "hover") {
      return [`${entity}进入悬停`];
    }
    if (action.action_type !== "move_entity") return [];
    const altitude = action.terminal_enu_m?.[2];
    return [altitude === undefined
      ? `${entity}改变航迹`
      : `${entity}改变航迹，目标高度 ${formatValue(altitude)} m`];
  }).join("；");
}

function element(tag, className = "", text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function clampTick(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return TICK_MIN;
  return Math.min(TICK_MAX, Math.max(TICK_MIN, Math.round(numeric / TICK_STEP) * TICK_STEP));
}

function formatBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes)) return "-";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
}

function trajectoryPlotPath(episodeId, tick) {
  return `/api/episodes/${encodeURIComponent(episodeId)}/trajectory-plot?tick=${encodeURIComponent(tick)}`;
}

function truthPill(value) {
  const truth = String(value ?? "unknown");
  const pill = element("span", `sw-truth sw-truth--${truth.replaceAll("_", "-")}`, truthLabel(truth));
  pill.title = truthLabel(truth);
  return pill;
}

function rawDetails(record, label = "查看技术详情") {
  const details = element("details", "sw-raw-details");
  const summary = element("summary", "", label);
  const body = element("pre");
  body.textContent = JSON.stringify(record, null, 2);
  details.append(summary, body);
  return details;
}

function emptyState(message) {
  return element("div", "sw-empty", message);
}

function mapEntityGroup(category) {
  if (["uav", "vehicle", "pedestrian", "facility", "traffic_light"].includes(category)) {
    return category;
  }
  return "context";
}

function sceneEntityDescriptor(record) {
  const values = record?.values;
  const pose = values?.truth_pose;
  const position = pose?.position_enu_m;
  if (!values || !pose || !Array.isArray(position) || position.length < 3) {
    throw new TypeError(`主体 ${record?.subject_id || "unknown"} 缺少 ENU 真值位姿`);
  }
  const numericPosition = position.slice(0, 3).map(Number);
  if (!numericPosition.every(Number.isFinite)) {
    throw new TypeError(`主体 ${record.subject_id} 的 ENU 真值位姿无效`);
  }
  if (typeof values.entity_category !== "string" || !values.entity_category) {
    throw new TypeError(`主体 ${record.subject_id} 缺少实体类别`);
  }
  if (typeof values.entity_kind !== "string" || !values.entity_kind) {
    throw new TypeError(`主体 ${record.subject_id} 缺少实体类型`);
  }
  const yaw = Number(pose.rotation_deg?.yaw_deg);
  if (!Number.isFinite(yaw)) {
    throw new TypeError(`主体 ${record.subject_id} 缺少有效偏航角`);
  }
  const category = values.entity_category;
  return {
    id: record.subject_id,
    category,
    group: mapEntityGroup(category),
    kind: values.entity_kind,
    state: String(values.state ?? ""),
    position: numericPosition,
    yaw,
    dimensions: values.dimensions_m || values.sumo_vehicle?.dimensions_m || {},
  };
}

function sceneProjection(sceneMap, width, height) {
  const [minX, minY, maxX, maxY] = sceneMap.roi.bbox_enu_m.map(Number);
  const padding = 16;
  const horizontalSpan = maxY - minY;
  const verticalSpan = maxX - minX;
  const scale = Math.min(
    (width - padding * 2) / horizontalSpan,
    (height - padding * 2) / verticalSpan,
  );
  const plotWidth = horizontalSpan * scale;
  const plotHeight = verticalSpan * scale;
  const left = (width - plotWidth) * 0.5;
  const top = (height - plotHeight) * 0.5;
  return {
    left,
    top,
    width: plotWidth,
    height: plotHeight,
    scale,
    point(position) {
      return [
        left + (Number(position[1]) - minY) * scale,
        top + (maxX - Number(position[0])) * scale,
      ];
    },
    contains(position) {
      return position[0] >= minX && position[0] <= maxX
        && position[1] >= minY && position[1] <= maxY;
    },
  };
}

function traceWorldPath(context, projection, points, close = false) {
  points.forEach((point, index) => {
    const [x, y] = projection.point(point);
    if (index) context.lineTo(x, y);
    else context.moveTo(x, y);
  });
  if (close) context.closePath();
}

function drawSceneEntity(context, projection, entity, focused) {
  const [x, y] = projection.point(entity.position);
  const metadata = MAP_ENTITY_GROUPS[entity.group];
  const angle = ((entity.yaw - 90) * Math.PI) / 180;
  context.save();
  context.translate(x, y);
  context.rotate(angle);
  context.fillStyle = metadata.color;
  context.strokeStyle = "#ffffff";
  context.lineWidth = 1.25;

  if (entity.group === "vehicle") {
    const length = Math.max(8, Math.min(15, Number(entity.dimensions.length || 5) * projection.scale));
    const width = Math.max(4, Math.min(7, Number(entity.dimensions.width || 1.8) * projection.scale));
    context.fillRect(-length * 0.5, -width * 0.5, length, width);
    context.strokeRect(-length * 0.5, -width * 0.5, length, width);
    context.beginPath();
    context.moveTo(length * 0.5, 0);
    context.lineTo(length * 0.2, -width * 0.5);
    context.lineTo(length * 0.2, width * 0.5);
    context.closePath();
    context.fillStyle = "#ffffff";
    context.fill();
  } else if (entity.group === "uav") {
    context.beginPath();
    context.moveTo(0, -7);
    context.lineTo(5, 0);
    context.lineTo(0, 7);
    context.lineTo(-5, 0);
    context.closePath();
    context.fill();
    context.stroke();
    context.beginPath();
    context.moveTo(-8, 0);
    context.lineTo(8, 0);
    context.moveTo(0, -8);
    context.lineTo(0, 8);
    context.strokeStyle = metadata.color;
    context.lineWidth = 2;
    context.stroke();
  } else if (entity.group === "pedestrian") {
    context.beginPath();
    context.arc(0, 0, 4, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  } else if (entity.group === "traffic_light") {
    context.fillRect(-3, -7, 6, 14);
    context.strokeRect(-3, -7, 6, 14);
  } else if (entity.group === "facility") {
    const noFly = entity.kind.endsWith(".no_fly_zone");
    context.beginPath();
    if (entity.kind.endsWith(".landing_pad")) context.arc(0, 0, 7, 0, Math.PI * 2);
    else if (entity.kind.endsWith(".base_station")) {
      context.moveTo(0, -7);
      context.lineTo(7, 6);
      context.lineTo(-7, 6);
      context.closePath();
    } else {
      context.rect(-6, -6, 12, 12);
    }
    context.fillStyle = noFly ? "#b53b43" : metadata.color;
    context.fill();
    context.stroke();
    context.fillStyle = "#ffffff";
    context.font = "700 8px sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    const symbol = entity.kind.endsWith(".landing_pad")
      ? "H"
      : entity.kind.endsWith(".charger")
        ? "+"
        : noFly
          ? "×"
          : "•";
    context.fillText(symbol, 0, 0.5);
  } else {
    context.beginPath();
    context.moveTo(0, -5);
    context.lineTo(5, 0);
    context.lineTo(0, 5);
    context.lineTo(-5, 0);
    context.closePath();
    context.fill();
    context.stroke();
  }
  context.restore();

  if (focused) {
    context.save();
    context.beginPath();
    context.arc(x, y, 11, 0, Math.PI * 2);
    context.strokeStyle = "#d4303f";
    context.lineWidth = 2;
    context.stroke();
    context.restore();
  }
  if (entity.group === "uav" && focused) {
    context.save();
    context.fillStyle = "#075a67";
    context.font = "700 9px sans-serif";
    context.textBaseline = "middle";
    context.fillText(`${formatValue(entity.position[2])} m`, x + 10, y - 9);
    context.restore();
  }
  return { entityId: entity.id, x, y, radius: focused ? 13 : 10, entity };
}

function drawUeTopViewAxes(context, projection) {
  const originX = projection.left + 22;
  const originY = projection.top + projection.height - 22;
  const length = 18;
  context.save();
  context.strokeStyle = "#53645f";
  context.fillStyle = "#53645f";
  context.lineWidth = 1.5;
  context.font = "700 9px sans-serif";
  context.textBaseline = "middle";

  context.beginPath();
  context.moveTo(originX, originY);
  context.lineTo(originX, originY - length);
  context.lineTo(originX - 3, originY - length + 5);
  context.moveTo(originX, originY - length);
  context.lineTo(originX + 3, originY - length + 5);
  context.moveTo(originX, originY);
  context.lineTo(originX + length, originY);
  context.lineTo(originX + length - 5, originY - 3);
  context.moveTo(originX + length, originY);
  context.lineTo(originX + length - 5, originY + 3);
  context.stroke();

  context.textAlign = "center";
  context.fillText("+X", originX, originY - length - 7);
  context.textAlign = "left";
  context.fillText("+Y", originX + length + 5, originY);
  context.restore();
}

function drawSceneMapCanvas(canvas, sceneMap, frame, focusedEntityIds) {
  const bounds = canvas.getBoundingClientRect();
  if (bounds.width < 1 || bounds.height < 1) return [];
  const pixelRatio = window.devicePixelRatio || 1;
  const pixelWidth = Math.round(bounds.width * pixelRatio);
  const pixelHeight = Math.round(bounds.height * pixelRatio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const context = canvas.getContext("2d");
  if (!context) throw new Error("浏览器无法创建场景画布");
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, bounds.width, bounds.height);
  context.fillStyle = "#f7f9f8";
  context.fillRect(0, 0, bounds.width, bounds.height);

  const projection = sceneProjection(sceneMap, bounds.width, bounds.height);
  context.save();
  context.beginPath();
  context.rect(projection.left, projection.top, projection.width, projection.height);
  context.clip();

  const [minX, minY, maxX, maxY] = sceneMap.roi.bbox_enu_m.map(Number);
  context.strokeStyle = "#e5e9e7";
  context.lineWidth = 1;
  for (let x = Math.ceil(minX / 20) * 20; x <= maxX; x += 20) {
    const [, screenY] = projection.point([x, minY]);
    context.beginPath();
    context.moveTo(projection.left, screenY);
    context.lineTo(projection.left + projection.width, screenY);
    context.stroke();
  }
  for (let y = Math.ceil(minY / 20) * 20; y <= maxY; y += 20) {
    const [screenX] = projection.point([minX, y]);
    context.beginPath();
    context.moveTo(screenX, projection.top);
    context.lineTo(screenX, projection.top + projection.height);
    context.stroke();
  }

  sceneMap.buildings.forEach((building) => {
    context.beginPath();
    building.rings_enu_m.forEach((ring) => traceWorldPath(context, projection, ring, true));
    context.fillStyle = "#d8ddda";
    context.strokeStyle = "#a7b1ad";
    context.lineWidth = 0.8;
    context.fill("evenodd");
    context.stroke();
  });

  sceneMap.roads.forEach((road) => {
    context.beginPath();
    traceWorldPath(context, projection, road.points_enu_m);
    context.strokeStyle = "#b7c0c3";
    context.lineCap = "round";
    context.lineJoin = "round";
    context.lineWidth = Math.max(2.5, Math.min(12, Number(road.width_m) * projection.scale + 1.5));
    context.stroke();
    context.strokeStyle = "#f8faf9";
    context.lineWidth = Math.max(1.5, Math.min(10, Number(road.width_m) * projection.scale));
    context.stroke();
  });

  sceneMap.corridors.forEach((corridor) => {
    context.beginPath();
    if (corridor.points_enu_m.length === 1) {
      const [x, y] = projection.point(corridor.points_enu_m[0]);
      context.arc(x, y, 7, 0, Math.PI * 2);
    } else {
      traceWorldPath(context, projection, corridor.points_enu_m);
    }
    context.setLineDash([7, 5]);
    context.strokeStyle = "#0e8992";
    context.lineWidth = 2;
    context.stroke();
    context.setLineDash([]);
  });

  const boundary = sceneMap.capture_boundary;
  context.beginPath();
  traceWorldPath(context, projection, boundary.polygon_enu_m, true);
  const noFlyBoundary = boundary.classification === "no_fly_zone";
  if (noFlyBoundary) {
    context.fillStyle = "rgba(181, 59, 67, 0.12)";
    context.fill();
  }
  context.setLineDash(noFlyBoundary ? [] : [8, 5]);
  context.strokeStyle = noFlyBoundary ? "#b53b43" : "#826b42";
  context.lineWidth = noFlyBoundary ? 2.25 : 1.75;
  context.stroke();
  context.setLineDash([]);

  sceneMap.constraints
    .filter((constraint) => constraint.entity_id !== boundary.boundary_id)
    .forEach((constraint) => {
      const [x, y] = projection.point(constraint.position_enu_m);
      context.beginPath();
      context.arc(x, y, 9, 0, Math.PI * 2);
      context.strokeStyle = "#b53b43";
      context.lineWidth = 2;
      context.stroke();
      context.beginPath();
      context.moveTo(x - 6, y - 6);
      context.lineTo(x + 6, y + 6);
      context.moveTo(x + 6, y - 6);
      context.lineTo(x - 6, y + 6);
      context.stroke();
    });

  const focused = new Set(focusedEntityIds);
  const entities = frame.entity_states
    .map(sceneEntityDescriptor)
    .filter((entity) => entity.category !== "airspace_corridor" && projection.contains(entity.position))
    .sort((left, right) => (
      Number(focused.has(left.id)) - Number(focused.has(right.id))
      || left.group.localeCompare(right.group)
      || left.id.localeCompare(right.id)
    ));
  const hitTargets = entities.map((entity) => (
    drawSceneEntity(context, projection, entity, focused.has(entity.id))
  ));
  context.restore();

  context.strokeStyle = "#9eaaa6";
  context.lineWidth = 1;
  context.strokeRect(projection.left, projection.top, projection.width, projection.height);
  drawUeTopViewAxes(context, projection);
  return hitTargets;
}

function setButtonPressed(buttons, activeValue) {
  buttons.forEach((button, value) => {
    const active = value === activeValue;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

function renderChainFlow(container, payload, milestones, selectedId, selectedTransitionId, currentTick, handlers) {
  container.replaceChildren();
  const nodes = new Map(payload.nodes.map((node) => [node.id, node]));
  const incoming = new Map(payload.edges.map((edge) => [edge.target_id, edge]));
  const ordered = [
    ...payload.nodes.map((node) => ({ kind: "event", tick: Number(node.dispatch_tick), order: 1, node })),
    ...milestones.map((milestone) => ({
      kind: "condition",
      tick: Number(milestone.tick),
      order: milestone.phase_kind === "onset" ? 0 : 2,
      milestone,
    })),
  ].sort((left, right) => (
    left.tick - right.tick || left.order - right.order
  ));
  ordered.forEach((row, index) => {
    if (row.kind === "condition") {
      const milestone = row.milestone;
      const button = element(
        "button",
        `sw-chain-step sw-chain-step--condition sw-chain-step--${milestone.direction}`,
      );
      button.type = "button";
      button.classList.toggle("is-selected", milestone.transition_id === selectedTransitionId);
      button.classList.toggle("is-current", row.tick === currentTick);
      const rail = element("span", "sw-chain-step__rail");
      rail.append(element("i", "sw-chain-step__dot"));
      if (index < ordered.length - 1) rail.append(element("i", "sw-chain-step__line"));
      const content = element("span", "sw-chain-step__content");
      content.append(
        element("strong", "", semanticMilestoneLabel(milestone)),
        element(
          "small",
          "",
          milestone.phase_kind === "terminal" ? "风险状态结束" : "风险状态开始",
        ),
        element(
          "em",
          "",
          [...new Set(Object.values(milestone.bindings || {}))].map(friendlyEntityName).join(" · "),
        ),
      );
      button.append(
        rail,
        element("span", "sw-chain-step__time", compactTickLabel(row.tick)),
        content,
      );
      button.title = `${semanticMilestoneLabel(milestone)}，${tickLabel(row.tick)}`;
      button.addEventListener("click", () => handlers.transition(milestone));
      container.append(button);
      return;
    }

    const node = row.node;
    const provenance = node.scheduling_provenance?.kind;
    const edge = incoming.get(node.id);
    const sourceNames = edge?.source_ids.map((id) => eventNodeLabel(nodes.get(id))) || [];
    const relation = provenance === "script_scheduled"
      ? "独立发生"
      : node.is_root
        ? "起始事件"
        : node.intent === "landing_or_terminal_resolution"
          ? `任务收尾，发生在“${sourceNames.join("、")}”之后`
          : `承接“${sourceNames.join("、") || "前序事件"}”`;
    const button = element("button", "sw-chain-step");
    button.type = "button";
    button.classList.toggle("is-selected", node.id === selectedId);
    button.classList.toggle("is-current", row.tick === currentTick);
    const rail = element("span", "sw-chain-step__rail");
    rail.append(element("i", "sw-chain-step__dot"));
    if (index < ordered.length - 1) rail.append(element("i", "sw-chain-step__line"));
    const time = element("span", "sw-chain-step__time", compactTickLabel(row.tick));
    const content = element("span", "sw-chain-step__content");
    content.append(element("strong", "", eventNodeLabel(node)), element("small", "", relation));
    if (edge) {
      content.append(element(
        "em",
        "",
        `相隔 ${edge.delta_ticks_min} 刻 · ${edge.validation_status.startsWith("runtime_pass") ? "顺序已验证" : "顺序需检查"}`,
      ));
    }
    button.append(rail, time, content);
    button.title = `${eventNodeLabel(node)}，${tickLabel(node.dispatch_tick)}，${relation}`;
    button.addEventListener("click", () => handlers.authored(node));
    container.append(button);
  });
}

function renderChainSelection(container, node) {
  container.replaceChildren();
  if (!node) {
    container.append(emptyState("该时刻没有场景事件"));
    return;
  }
  const provenance = node.scheduling_provenance?.kind;
  const provenanceLabel = provenance === "script_scheduled"
    ? "独立发生"
    : provenance === "none_declared"
      ? "起始事件"
      : "由前序事件触发";
  const heading = element("div", "sw-chain-selection__head");
  heading.append(
    element("strong", "", eventNodeLabel(node)),
    element("span", `sw-provenance sw-provenance--${provenance || "causal"}`, provenanceLabel),
  );
  const facts = element("dl", "sw-compact-facts");
  [
    ["发生时刻", tickLabel(node.dispatch_tick)],
    ["执行结果", node.runtime_status === "fired" ? "已执行" : "未执行"],
  ].forEach(([term, value]) => {
    facts.append(element("dt", "", term), element("dd", "", value));
  });
  container.append(heading, facts, rawDetails(node));
}

function transitionSearchText(record) {
  return `${predicateLabel(record.predicate_id)} ${predicateDomainLabel(record.predicate_id)} ${directionLabel(record.direction)} ${record.predicate_id} ${record.tuple_id} ${record.tick}`.toLowerCase();
}

function assertionSearchText(record) {
  return `${predicateLabel(record.predicate_id)} ${predicateDomainLabel(record.predicate_id)} ${truthLabel(record.value)} ${record.predicate_id} ${record.tuple_id} ${entityLabel(record.bindings)}`.toLowerCase();
}

function renderTransitionRows(container, frame, search, selected, onSelect) {
  const query = search.trim().toLowerCase();
  const rows = frame.transition_markers.filter((record) => !query || transitionSearchText(record).includes(query));
  rows.sort((left, right) => {
    const leftSelected = left.transition_id === selected?.transitionId;
    const rightSelected = right.transition_id === selected?.transitionId;
    if (leftSelected !== rightSelected) return leftSelected ? -1 : 1;
    return left.tick - right.tick || left.transition_id.localeCompare(right.transition_id);
  });
  container.replaceChildren();
  if (!rows.length) {
    container.append(emptyState("没有匹配的状态变化"));
    return rows.length;
  }
  const fragment = document.createDocumentFragment();
  rows.forEach((record) => {
    const button = element("button", "sw-predicate-row sw-predicate-row--transition");
    button.type = "button";
    if (record.tick === frame.tick) button.classList.add("is-current");
    if (selected?.transitionId === record.transition_id) button.classList.add("is-selected");
    const time = element("span", "sw-predicate-row__time", compactTickLabel(record.tick));
    const body = element("span", "sw-predicate-row__body");
    body.append(
      element("strong", "", predicateLabel(record.predicate_id)),
      element("small", "", predicateDomainLabel(record.predicate_id)),
    );
    const direction = element("span", `sw-direction sw-direction--${record.direction}`, directionLabel(record.direction));
    button.append(time, body, direction);
    button.title = `${predicateLabel(record.predicate_id)}，${tickLabel(record.tick)}${directionLabel(record.direction)}`;
    button.addEventListener("click", () => onSelect(record));
    fragment.append(button);
  });
  container.append(fragment);
  return rows.length;
}

function renderAssertionRows(container, frame, search, truthFilter, selected, onSelect) {
  const query = search.trim().toLowerCase();
  const rows = frame.predicates.filter((record) => {
    if (truthFilter !== "all" && record.value !== truthFilter) return false;
    return !query || assertionSearchText(record).includes(query);
  });
  const truthOrder = { true: 0, false: 1, unknown: 2, out_of_scope: 3 };
  rows.sort((left, right) => {
    const leftSelected = left.predicate_id === selected?.predicateId && left.tuple_id === selected?.tupleId;
    const rightSelected = right.predicate_id === selected?.predicateId && right.tuple_id === selected?.tupleId;
    if (leftSelected !== rightSelected) return leftSelected ? -1 : 1;
    return truthOrder[left.value] - truthOrder[right.value]
      || predicateLabel(left.predicate_id).localeCompare(predicateLabel(right.predicate_id), "zh-CN")
      || left.tuple_id.localeCompare(right.tuple_id);
  });
  container.replaceChildren();
  if (!rows.length) {
    container.append(emptyState("没有匹配的当前结果"));
    return rows.length;
  }
  const fragment = document.createDocumentFragment();
  rows.forEach((record) => {
    const button = element("button", "sw-predicate-row sw-predicate-row--assertion");
    button.type = "button";
    if (selected?.predicateId === record.predicate_id && selected?.tupleId === record.tuple_id) {
      button.classList.add("is-selected");
    }
    const body = element("span", "sw-predicate-row__body");
    body.append(
      element("strong", "", predicateLabel(record.predicate_id)),
      element("small", "", entityLabel(record.bindings)),
    );
    const truth = truthPill(record.value);
    const update = element("span", "sw-predicate-row__update", compactTickLabel(record.last_update_tick));
    button.append(body, update, truth);
    button.title = `${predicateLabel(record.predicate_id)}，${entityLabel(record.bindings)}，${truthLabel(record.value)}`;
    button.addEventListener("click", () => onSelect(record));
    fragment.append(button);
  });
  container.append(fragment);
  return rows.length;
}

function renderTimelineMarkers(container, frame, chain, tick, processFocus, handlers) {
  container.replaceChildren();
  const lanes = [
    {
      label: "场景事件",
      className: "authored",
      rows: chain.nodes.map((node) => ({ id: node.id, tick: node.dispatch_tick, title: eventNodeLabel(node), raw: node })),
      select: handlers.authored,
    },
    {
      label: "状态变化",
      className: "predicate",
      rows: frame.transition_markers.map((row) => ({ id: row.transition_id, tick: row.tick, title: predicateLabel(row.predicate_id), raw: row })),
      select: handlers.transition,
    },
    {
      label: "识别事件",
      className: "event",
      rows: (frame.all_event_markers || frame.event_markers).map((row) => ({ id: row.event_id, tick: row.tick, title: eventFamilyLabel(row.event_family_id), raw: row })),
      select: handlers.event,
    },
    {
      label: "真实状态 · 5刻",
      className: "state",
      rows: (frame.state_track || []).map((row) => ({
        id: `state:${row.tick}`,
        tick: row.tick,
        title: `${row.record_count} 条原始状态`,
        raw: row,
      })),
      select: handlers.state,
    },
  ];
  lanes.forEach((lane) => {
    const row = element("div", `sw-marker-lane sw-marker-lane--${lane.className}`);
    row.append(element("span", "sw-marker-lane__label", lane.label));
    const track = element("div", "sw-marker-lane__track");
    if (lane.className === "state" && processFocus) {
      const process = element("i", "sw-state-process-range");
      process.style.left = `${(processFocus.startTick / TICK_MAX) * 100}%`;
      process.style.width = `${((processFocus.endTick - processFocus.startTick) / TICK_MAX) * 100}%`;
      process.title = `${processFocus.label} · ${processFocus.startTick}–${processFocus.endTick}刻`;
      track.append(process);
    }
    const playhead = element("i", "sw-marker-playhead");
    playhead.style.left = `${(tick / TICK_MAX) * 100}%`;
    track.append(playhead);
    lane.rows.forEach((marker) => {
      const button = element(
        "button",
        lane.className === "state" ? "sw-state-point" : "sw-timeline-marker",
      );
      button.type = "button";
      button.style.left = `${(Number(marker.tick) / TICK_MAX) * 100}%`;
      button.classList.toggle("is-current", Number(marker.tick) === tick);
      button.title = `${marker.title} · ${tickLabel(marker.tick)}`;
      button.setAttribute("aria-label", `${marker.title}，${tickLabel(marker.tick)}`);
      button.addEventListener("click", () => lane.select(marker.raw, button));
      track.append(button);
    });
    row.append(track);
    container.append(row);
  });
}

function appendMeasurements(container, record) {
  const list = element("dl", "sw-value-grid");
  recordMeasurements(record).forEach((measurement) => {
    list.append(
      element("dt", "", measurementLabel(record, measurement.path)),
      element("dd", "", measurementValue(measurement)),
    );
  });
  container.append(list);
}

function renderEntityButton(entityId, onSelect, className = "") {
  const friendly = friendlyEntityName(entityId);
  const button = element("button", `sw-entity-button ${className}`.trim());
  button.type = "button";
  button.append(element("strong", "", friendly));
  if (friendly !== entityId) button.append(element("code", "", entityId));
  button.title = `${friendly} · 查看当前时刻真实状态`;
  button.addEventListener("click", (event) => onSelect(entityId, event.currentTarget));
  return button;
}

function stateRecordsForEntity(frame, entityId) {
  return [...frame.entity_states, ...frame.states]
    .filter((record) => record.subject_id === entityId);
}

function stateUnitForField(name) {
  const leaf = String(name).split(".").at(-1).replace(/\[\d+\]$/, "");
  const suffixes = [
    ["_mps", "m/s"],
    ["_ms", "ms"],
    ["_hz", "Hz"],
    ["_wh", "Wh"],
    ["_w", "W"],
    ["_bytes", "byte"],
    ["_deg", "deg"],
    ["_m", "m"],
    ["_c", "°C"],
    ["_ratio", "ratio"],
    ["_ticks", "tick"],
  ];
  return suffixes.find(([suffix]) => leaf.endsWith(suffix))?.[1] || "scalar";
}

function flattenEntityState(value, prefix = "") {
  if (Array.isArray(value)) {
    return value.flatMap((item, index) => flattenEntityState(item, `${prefix}[${index}]`));
  }
  if (value && typeof value === "object") {
    return Object.entries(value).flatMap(([key, item]) => (
      flattenEntityState(item, prefix ? `${prefix}.${key}` : key)
    ));
  }
  return [{ name: prefix, value, unit: stateUnitForField(prefix) }];
}

function stateRecordMeasurements(record) {
  if (Array.isArray(record.measurements)) return record.measurements;
  if (record.values && typeof record.values === "object") {
    return flattenEntityState(record.values);
  }
  throw new TypeError(`主体状态记录 ${record.id} 缺少属性数据`);
}

function stateMeasurements(records) {
  const unique = new Map();
  records.flatMap(stateRecordMeasurements).forEach((measurement) => {
    if (!unique.has(measurement.name)) unique.set(measurement.name, measurement);
  });
  return unique;
}

function stateMeasurement(measurements, name) {
  return [...measurements.values()].find((measurement) => (
    measurement.name === name || measurement.name.endsWith(`.${name}`)
  ));
}

function subjectStateRecords(frame) {
  const truthIds = new Set(frame.entity_states.map((record) => record.subject_id));
  const directSemanticStates = frame.states.filter((record) => truthIds.has(record.subject_id));
  return [...frame.entity_states, ...directSemanticStates];
}

function appendAttributeGroup(container, title, entries) {
  if (!entries.length) return;
  const group = element("section", "sw-entity-group");
  group.append(element("h3", "", title));
  const list = element("dl", "sw-attribute-grid");
  entries.forEach((measurement) => {
    const row = element("div", "sw-attribute-row");
    row.dataset.search = `${fieldLabel(measurement.name)} ${measurement.name} ${formatValue(measurement.value)}`.toLowerCase();
    row.append(
      element("dt", "", fieldLabel(measurement.name)),
      element("dd", "", stateMeasurementValue(measurement)),
    );
    row.title = measurement.name;
    list.append(row);
  });
  group.append(list);
  container.append(group);
}

function buildEntityPopover(frame, entityId, onClose) {
  const records = stateRecordsForEntity(frame, entityId);
  const popover = element("aside", "sw-entity-popover");
  popover.setAttribute("role", "dialog");
  popover.setAttribute("aria-label", `${friendlyEntityName(entityId)}真实状态`);

  const header = element("header", "sw-entity-popover__header");
  const identity = element("div");
  const friendly = friendlyEntityName(entityId);
  identity.append(
    element("span", "sw-section-label", `${tickLabel(frame.tick)} · 同刻主体真值`),
    element("h2", "", friendly),
  );
  if (friendly !== entityId) identity.append(element("code", "", entityId));
  const close = element("button", "sw-entity-popover__close", "×");
  close.type = "button";
  close.title = "关闭状态窗口";
  close.setAttribute("aria-label", "关闭状态窗口");
  close.addEventListener("click", onClose);
  header.append(identity, close);
  popover.append(header);

  if (!records.length) {
    popover.append(emptyState(`${tickLabel(frame.tick)}的真值帧中没有该主体`));
    return popover;
  }

  const measurements = stateMeasurements(records);
  const x = stateMeasurement(measurements, "position_enu_m[0]");
  const y = stateMeasurement(measurements, "position_enu_m[1]");
  const z = stateMeasurement(measurements, "position_enu_m[2]");
  const overview = element("section", "sw-entity-overview");
  if (x || y || z) {
    const coordinate = element("div", "sw-entity-coordinate");
    coordinate.append(element("span", "", "ENU 位置坐标"));
    coordinate.append(element(
      "strong",
      "",
      `X ${x ? stateMeasurementValue(x) : "-"} · Y ${y ? stateMeasurementValue(y) : "-"} · Z ${z ? stateMeasurementValue(z) : "-"}`,
    ));
    overview.append(coordinate);
  }
  const highlights = element("div", "sw-entity-highlights");
  [
    ["speed_mps", "当前速度"],
    ["z_agl_m", "离地高度"],
    ["nearest_aircraft_distance_m", "最近飞行器"],
  ].forEach(([name, label]) => {
    const measurement = stateMeasurement(measurements, name);
    if (!measurement) return;
    const item = element("div");
    item.append(element("span", "", label), element("strong", "", stateMeasurementValue(measurement)));
    highlights.append(item);
  });
  if (highlights.childElementCount) overview.append(highlights);
  popover.append(overview);

  const tools = element("div", "sw-entity-popover__tools");
  const search = element("input");
  search.type = "search";
  search.placeholder = "搜索属性名称或字段";
  search.setAttribute("aria-label", "搜索主体状态属性");
  tools.append(search);
  popover.append(tools);

  const body = element("div", "sw-entity-popover__body");
  const entries = [...measurements.values()];
  const coordinateNames = new Set([x?.name, y?.name, z?.name].filter(Boolean));
  const groups = [
    ["位置与高度", (name) => /altitude|z_agl|distance_to_(home_pad|assigned_landing)|ground_reference|takeoff_ground/.test(name)],
    ["运动状态", (name) => /speed|velocity|yaw|heading/.test(name)],
    ["飞行与管制", (name) => /corridor|cooperative|rth_mode|restricted_region|entity_role|landing_zone|home_pad/.test(name)],
    ["周边距离", (name) => /(^|\.)nearest_/.test(name)],
  ];
  const used = new Set(coordinateNames);
  groups.forEach(([title, matches]) => {
    const rows = entries.filter((measurement) => !used.has(measurement.name) && matches(measurement.name));
    rows.forEach((measurement) => used.add(measurement.name));
    appendAttributeGroup(body, title, rows);
  });
  appendAttributeGroup(body, "其他属性", entries.filter((measurement) => !used.has(measurement.name)));

  const sourceDetails = rawDetails(
    records.map((record) => ({
      observation_id: record.id,
      observation_family: record.api_id,
      quality: record.quality,
      state_scope: record.state_scope,
      state_authority: record.state_authority,
      record_ref: record.record_ref,
      source_refs: record.source_refs,
    })),
    `查看 ${records.length} 条来源记录`,
  );
  sourceDetails.classList.add("sw-entity-source");
  body.append(sourceDetails);
  popover.append(body);

  search.addEventListener("input", () => {
    const query = search.value.trim().toLowerCase();
    body.querySelectorAll(".sw-attribute-row").forEach((row) => {
      row.hidden = Boolean(query) && !row.dataset.search.includes(query);
    });
    body.querySelectorAll(".sw-entity-group").forEach((group) => {
      group.hidden = ![...group.querySelectorAll(".sw-attribute-row")].some((row) => !row.hidden);
    });
  });
  return popover;
}

function renderEvidenceRecords(container, context, onEntity) {
  container.replaceChildren();
  if (!context.evidence_records.length) {
    container.append(emptyState("没有匹配的证据记录"));
    return;
  }
  context.evidence_records.forEach((record) => {
    const row = element("article", "sw-context-record sw-context-record--evidence");
    const header = element("header", "sw-context-record__head");
    const identity = element("div");
    identity.append(element("strong", "", predicateLabel(record.metric_id)));
    if (record.subject_id) identity.append(renderEntityButton(record.subject_id, onEntity, "sw-entity-button--inline"));
    else identity.append(element("small", "", "场景整体"));
    header.append(identity, element("span", "sw-tick-tag", tickLabel(record.tick)));
    row.append(header);
    appendMeasurements(row, record);
    row.append(rawDetails(record));
    container.append(row);
  });
}

function renderStateRecords(container, frame, query, focusedEntities, onEntity) {
  container.replaceChildren();
  const normalizedQuery = query.trim().toLowerCase();
  const groups = new Map();
  subjectStateRecords(frame).forEach((record) => {
    const subject = groups.get(record.subject_id) || {
      id: record.subject_id,
      kind: record.subject_kind,
      records: [],
    };
    subject.records.push(record);
    groups.set(record.subject_id, subject);
  });
  const focus = new Set(focusedEntities);
  const subjects = [...groups.values()].filter((subject) => {
    if (!normalizedQuery) return true;
    const text = `${friendlyEntityName(subject.id)} ${subject.id} ${subject.kind} ${subject.records
      .flatMap((record) => stateRecordMeasurements(record).map((measurement) => `${measurement.name} ${formatValue(measurement.value)}`))
      .join(" ")}`.toLowerCase();
    return text.includes(normalizedQuery);
  }).sort((left, right) => {
    const focusDelta = Number(focus.has(right.id)) - Number(focus.has(left.id));
    if (focusDelta) return focusDelta;
    const relationDelta = Number(left.id.includes("|")) - Number(right.id.includes("|"));
    return relationDelta || left.id.localeCompare(right.id);
  });

  if (!subjects.length) {
    container.append(emptyState("当前时刻没有匹配的主体状态"));
    return;
  }
  const summary = element("div", "sw-state-snapshot-summary");
  summary.append(
    element("strong", "", `${tickLabel(frame.tick)} · ${subjects.length} 个主体`),
    element("span", "", `${frame.entity_states.length} 个同刻真值主体，点击查看完整属性`),
  );
  container.append(summary);
  subjects.forEach((subject) => {
    const measurements = stateMeasurements(subject.records);
    const row = element("button", "sw-state-subject-row");
    row.type = "button";
    if (focus.has(subject.id)) row.classList.add("is-focused");
    const identity = element("span", "sw-state-subject-row__identity");
    identity.append(
      element("strong", "", friendlyEntityName(subject.id)),
      element("code", "", subject.id),
    );
    const facts = element("span", "sw-state-subject-row__facts");
    const position = [0, 1, 2].map((index) => stateMeasurement(measurements, `position_enu_m[${index}]`));
    if (position.some(Boolean)) {
      facts.append(element(
        "span",
        "",
        `ENU ${position.map((measurement) => measurement ? stateMeasurementValue(measurement) : "-").join(" / ")}`,
      ));
    }
    const speed = stateMeasurement(measurements, "speed_mps");
    facts.append(element(
      "small",
      "",
      `${subjectKindLabel(subject.kind)} · ${subject.records.length} 组属性${speed ? ` · 速度 ${stateMeasurementValue(speed)}` : ""}`,
    ));
    const action = element("span", "sw-state-subject-row__action", "查看");
    row.append(identity, facts, action);
    row.addEventListener("click", (event) => onEntity(subject.id, event.currentTarget));
    container.append(row);
  });
}

function renderPhysicalLogs(container, context, onEntity) {
  container.replaceChildren();
  if (!context.physical_logs.length) {
    container.append(emptyState("没有匹配的事件执行日志"));
    return;
  }
  context.physical_logs.forEach((record) => {
    const row = element("article", `sw-context-record sw-context-record--${record.kind}`);
    const header = element("header", "sw-context-record__head");
    const identity = element("div");
    identity.append(
      element("strong", "", eventNodeLabel(record)),
      element("small", "", record.ticks.length ? record.ticks.map(tickLabel).join("、") : "未记录发生时刻"),
    );
    const logKind = record.kind === "event_trace" ? "场景日志" : "执行明细";
    header.append(identity, element("span", `sw-log-kind sw-log-kind--${record.kind}`, logKind));
    const facts = element("div", "sw-log-facts");
    facts.append(
      element("span", "", `${record.action_count} 个动作`),
      element("span", "", `${record.target_ids.length} 个相关对象`),
    );
    row.append(header, facts);
    const actionSummary = physicalActionSummary(record);
    if (actionSummary) row.append(element("div", "sw-action-summary", actionSummary));
    if (record.target_ids.length) {
      const subjects = element("div", "sw-context-record__subject");
      record.target_ids.forEach((id) => subjects.append(renderEntityButton(id, onEntity, "sw-entity-button--inline")));
      row.append(subjects);
    }
    row.append(rawDetails(record.raw));
    container.append(row);
  });
}

function renderFiles(container, context) {
  container.replaceChildren();
  context.files.forEach((record) => {
    const row = element("article", "sw-file-row");
    const identity = element("div");
    identity.append(element("strong", "", FILE_LABELS[record.label]), element("code", "", record.file));
    const metadata = element("div", "sw-file-row__meta");
    metadata.append(
      element("span", "", formatBytes(record.size_bytes)),
      element("span", "", ACCESS_LABELS[record.access]),
      element("span", "", record.record_count === null ? "按需定位" : `${record.record_count} 条记录`),
    );
    row.append(identity, metadata);
    container.append(row);
  });
}

function renderContextSummary(container, context, handlers) {
  container.replaceChildren();
  const selection = element("div", "sw-context-selection");
  const identity = element("div", "sw-context-selection__identity");
  const predicateId = context.selection.predicate_id;
  const realization = context.physical_logs.find((record) => (
    record.kind === "event_realization" && record.nearest_tick === context.tick
  )) || context.physical_logs.find((record) => record.kind === "event_realization");
  const actionSummary = physicalActionSummary(realization);
  const conclusion = context.transition
    ? `${tickLabel(context.tick)}，${predicateLabel(predicateId)}：${directionLabel(context.transition.direction)}`
    : context.assertion
      ? `${tickLabel(context.tick)}，${predicateLabel(predicateId)}：${truthLabel(context.assertion.value)}`
      : realization
        ? `${tickLabel(context.tick)}，${eventNodeLabel(realization)}`
        : `${tickLabel(context.tick)}的真实状态`;
  identity.append(
    element("span", "sw-section-label", "判定结论"),
    element("strong", "", conclusion),
    element(
      "small",
      "sw-context-selection__reason",
      predicateId ? primaryEvidenceSummary(context) : actionSummary || `${context.counts.state_records} 条当前状态`,
    ),
  );
  selection.append(identity);
  if (context.transition) {
    selection.append(element("span", `sw-direction sw-direction--${context.transition.direction}`, directionLabel(context.transition.direction)));
  } else if (context.assertion) {
    selection.append(truthPill(context.assertion.value));
  }
  container.append(selection);

  const flow = element("div", "sw-derivation-flow");
  const stateStep = element("button", "sw-derivation-step sw-derivation-step--state");
  stateStep.type = "button";
  stateStep.append(element("span", "", "依据数据"), element("strong", "", `${context.counts.state_records} 条相关状态`));
  stateStep.addEventListener("click", () => handlers.tab("state"));
  const predicateStep = element("div", "sw-derivation-step sw-derivation-step--predicate");
  predicateStep.append(
    element("span", "", "判定结果"),
    element(
      "strong",
      "",
      context.transition
        ? `${truthLabel(context.transition.from_value)} → ${truthLabel(context.transition.to_value)}`
        : context.assertion
          ? truthLabel(context.assertion.value)
          : "未选择状态",
    ),
  );
  const event = context.linked_events[0];
  const eventStep = element("button", "sw-derivation-step sw-derivation-step--event");
  eventStep.type = "button";
  eventStep.append(
    element("span", "", "关联事件"),
    element("strong", "", event ? eventFamilyLabel(event.event_family_id) : "无关联事件"),
  );
  eventStep.addEventListener("click", () => handlers.tab("logs"));
  flow.append(stateStep, element("i", "", "→"), predicateStep, element("i", "", "→"), eventStep);
  container.append(flow);

  if (actionSummary) {
    const action = element("div", "sw-action-summary sw-action-summary--prominent");
    action.append(element("span", "sw-section-label", "实际执行"), element("strong", "", actionSummary));
    container.append(action);
  }

  const entityIds = entityIdsFromContext(context);
  if (entityIds.length) {
    const entities = element("div", "sw-entity-strip");
    entities.append(element("span", "sw-section-label", "相关对象"));
    entityIds.forEach((id) => entities.append(renderEntityButton(id, handlers.entity)));
    container.append(entities);
  }

  if (context.predicate_changes.length > 1) {
    const history = element("div", "sw-history-strip");
    history.append(element("span", "sw-section-label", "同一状态的变化"));
    context.predicate_changes.forEach((record) => {
      const button = element("button", "sw-history-change", `${compactTickLabel(record.to_tick)} ${directionLabel(record.direction)}`);
      button.type = "button";
      button.title = `${tickLabel(record.to_tick)}${directionLabel(record.direction)}`;
      button.addEventListener("click", () => handlers.transition({
        transition_id: record.transition_id,
        predicate_id: record.predicate_id,
        tuple_id: record.tuple_id,
        tick: record.to_tick,
        direction: record.direction,
      }));
      history.append(button);
    });
    container.append(history);
  }

  if (predicateId) {
    container.append(rawDetails(
      {
        selection: context.selection,
        transition: context.transition,
        assertion: context.assertion,
        semantic_operations: context.semantic_operations,
      },
      "查看判定技术详情",
    ));
  }
}

export async function renderSemanticWorkbench(ctx) {
  const { root, api, store, ui } = ctx;
  if (!(root instanceof Element)) throw new TypeError("renderSemanticWorkbench requires ctx.root");
  if (
    typeof api?.timeline !== "function"
    || typeof api?.sceneMap !== "function"
    || typeof api?.semanticContext !== "function"
  ) {
    throw new TypeError("semantic workbench API contract is unavailable");
  }

  const listing = await api.episodes();
  if (!Array.isArray(listing?.items) || !listing.items.length) throw new Error("Episode 索引为空");

  let frame = null;
  let chain = null;
  let sceneMap = null;
  let mapHitTargets = [];
  let context = null;
  let loadedEpisode = null;
  let selection = store.selection;
  let selectedAuthoredId = null;
  let processFocus = null;
  let focusedEntityIds = [];
  let entityPopover = null;
  let entityPopoverAnchor = null;
  let predicateMode = "transitions";
  let predicateQuery = "";
  let truthFilter = "all";
  let contextTab = "evidence";
  let contextQuery = "";
  let loadSerial = 0;
  let contextSerial = 0;
  let tickTimer = null;
  let contextTimer = null;
  let trajectoryLoadedKey = null;
  let disposed = false;

  const view = element("section", "semantic-workbench");
  const header = element("header", "sw-header");
  const headerIdentity = element("div", "sw-header__identity");
  headerIdentity.append(
    element("span", "sw-eyebrow", "场景数据检查"),
    element("h1", "", "场景语义检查"),
  );
  const headerStatus = element("div", "sw-header__status");
  const runtimeBadge = element("span", "sw-status-badge", "事件顺序 -");
  const frameBadge = element("span", "sw-status-badge", "当前时刻 -");
  const loadStatus = element("span", "sw-load-status", "等待数据");
  loadStatus.setAttribute("role", "status");
  headerStatus.append(runtimeBadge, frameBadge, loadStatus);
  header.append(headerIdentity, headerStatus);

  const toolbar = element("section", "sw-toolbar");
  const episodeField = element("label", "sw-field sw-field--episode");
  episodeField.append(element("span", "", "场景数据"));
  const episodeSelect = element("select");
  listing.items.forEach((record) => {
    const option = element("option", "", `${record.episode_id} · ${CLOSURE_LABELS[record.closure_status]}`);
    option.value = record.episode_id;
    episodeSelect.append(option);
  });
  episodeField.append(episodeSelect);

  const tickControls = element("div", "sw-tick-controls");
  const previousTick = element("button", "sw-icon-button", "‹");
  previousTick.type = "button";
  previousTick.title = "上一个时刻";
  previousTick.setAttribute("aria-label", "上一个时刻");
  const tickInput = element("input", "sw-tick-input");
  tickInput.type = "number";
  tickInput.min = String(TICK_MIN);
  tickInput.max = String(TICK_MAX);
  tickInput.step = String(TICK_STEP);
  tickInput.setAttribute("aria-label", "当前时刻");
  const nextTick = element("button", "sw-icon-button", "›");
  nextTick.type = "button";
  nextTick.title = "下一个时刻";
  nextTick.setAttribute("aria-label", "下一个时刻");
  tickControls.append(previousTick, tickInput, nextTick);

  const frameStats = element("div", "sw-frame-stats");
  toolbar.append(episodeField, tickControls, frameStats);

  const scenePanel = element("section", "sw-scene-panel");
  const sceneHeader = element("header", "sw-scene-header");
  const sceneIdentity = element("div");
  sceneIdentity.append(
    element("span", "sw-section-label", "真实场景"),
    element("h2", "", "场景俯视图"),
  );
  const sceneTick = element("span", "sw-scene-tick", "-");
  sceneHeader.append(sceneIdentity, sceneTick);
  const sceneLayout = element("div", "sw-scene-layout");
  const sceneCanvasWrap = element("div", "sw-scene-canvas-wrap");
  const sceneCanvas = element("canvas", "sw-scene-canvas");
  sceneCanvas.setAttribute("role", "img");
  sceneCanvas.setAttribute("aria-label", "当前时刻场景二维俯视图");
  const sceneTooltip = element("div", "sw-scene-tooltip");
  sceneTooltip.hidden = true;
  const sceneLoading = element("div", "sw-scene-loading", "正在读取场景底图…");
  sceneCanvasWrap.append(sceneCanvas, sceneTooltip, sceneLoading);
  const sceneAside = element("aside", "sw-scene-aside");
  sceneAside.append(element("h3", "", "当前主体"));
  const sceneEntityCounts = element("div", "sw-scene-entity-counts");
  sceneAside.append(sceneEntityCounts, element("h3", "", "场景图层"));
  const sceneLegend = element("div", "sw-scene-legend");
  MAP_STATIC_LAYERS.forEach((layer) => {
    const item = element("div", "sw-scene-legend__item");
    item.append(
      element("i", `sw-scene-swatch sw-scene-swatch--${layer.className}`),
      element("span", "", layer.label),
    );
    sceneLegend.append(item);
  });
  const sceneFacts = element("dl", "sw-scene-facts");
  sceneAside.append(sceneLegend, sceneFacts);
  sceneLayout.append(sceneCanvasWrap, sceneAside);
  scenePanel.append(sceneHeader, sceneLayout);

  const trajectoryPanel = element("section", "sw-scene-panel sw-trajectory-panel");
  const trajectoryHeader = element("header", "sw-scene-header");
  const trajectoryIdentity = element("div");
  trajectoryIdentity.append(
    element("span", "sw-section-label", "真实轨迹"),
    element("h2", "", "车辆 · 行人 · 无人机 Matplotlib"),
  );
  const trajectoryTick = element("span", "sw-scene-tick", "-");
  trajectoryHeader.append(trajectoryIdentity, trajectoryTick);
  const trajectoryBody = element("div");
  trajectoryBody.style.minHeight = "220px";
  const trajectoryImage = element("img", "sw-trajectory-image");
  trajectoryImage.alt = "当前时刻车辆、行人和无人机的 Matplotlib 轨迹真值图";
  trajectoryImage.decoding = "async";
  trajectoryImage.hidden = true;
  trajectoryImage.style.display = "block";
  trajectoryImage.style.width = "100%";
  trajectoryImage.style.height = "auto";
  const trajectoryLoading = element("div", "sw-scene-loading", "等待轨迹图…");
  trajectoryBody.append(trajectoryImage, trajectoryLoading);
  const trajectoryCaption = element(
    "div",
    "sw-trajectory-caption",
    "图像来自 render_ready_episodes_capture_filtered 的真实轨迹数据；仅绘制当前 tick 之前的轨迹。",
  );
  trajectoryPanel.append(trajectoryHeader, trajectoryBody, trajectoryCaption);

  const timeline = element("section", "sw-timeline");
  const range = element("input", "sw-timeline-range");
  range.type = "range";
  range.min = String(TICK_MIN);
  range.max = String(TICK_MAX);
  range.step = String(TICK_STEP);
  range.setAttribute("aria-label", "场景时间轴");
  const markerLanes = element("div", "sw-marker-lanes");
  const ruler = element("div", "sw-ruler");
  for (let tick = 0; tick <= 900; tick += 100) {
    const mark = element("span", "", tick);
    mark.style.left = `${(tick / TICK_MAX) * 100}%`;
    ruler.append(mark);
  }
  timeline.append(range, markerLanes, ruler);

  const grid = element("div", "sw-grid");
  const chainPane = element("section", "sw-pane sw-pane--chain");
  const chainHeader = element("header", "sw-pane__header");
  const chainTitle = element("div");
  chainTitle.append(element("span", "sw-section-label", "1 · 发生了什么"), element("h2", "", "场景进程"));
  const chainCount = element("span", "sw-pane__count", "-");
  chainHeader.append(chainTitle, chainCount);
  const chainFlow = element("div", "sw-chain-flow");
  const chainSelection = element("section", "sw-chain-selection");
  chainPane.append(chainHeader, chainFlow, chainSelection);

  const predicatePane = element("section", "sw-pane sw-pane--predicates");
  const predicateHeader = element("header", "sw-pane__header sw-pane__header--stacked");
  const predicateHeadingRow = element("div", "sw-pane-heading-row");
  const predicateTitle = element("div");
  predicateTitle.append(element("span", "sw-section-label", "2 · 状态怎么变"), element("h2", "", "状态变化"));
  const predicateCount = element("span", "sw-pane__count", "-");
  predicateHeadingRow.append(predicateTitle, predicateCount);
  const modeControl = element("div", "sw-segmented");
  const modeButtons = new Map();
  [["transitions", "变化记录"], ["assertions", "当前结果"]].forEach(([value, label]) => {
    const button = element("button", "", label);
    button.type = "button";
    button.dataset.mode = value;
    modeButtons.set(value, button);
    modeControl.append(button);
  });
  predicateHeader.append(predicateHeadingRow, modeControl);
  const predicateTools = element("div", "sw-list-tools");
  const predicateSearch = element("input");
  predicateSearch.type = "search";
  predicateSearch.placeholder = "搜索状态名称或对象";
  predicateSearch.setAttribute("aria-label", "搜索状态名称或对象");
  const truthSelect = element("select", "sw-truth-filter");
  truthSelect.setAttribute("aria-label", "判定结果筛选");
  [["all", "全部结果"], ["true", "成立"], ["false", "未成立"], ["unknown", "数据不足"], ["out_of_scope", "不适用"]]
    .forEach(([value, label]) => {
      const option = element("option", "", label);
      option.value = value;
      truthSelect.append(option);
    });
  predicateTools.append(predicateSearch, truthSelect);
  const predicateList = element("div", "sw-predicate-list");
  predicatePane.append(predicateHeader, predicateTools, predicateList);

  const contextPane = element("section", "sw-pane sw-pane--context");
  const contextHeader = element("header", "sw-pane__header");
  const contextTitle = element("div");
  contextTitle.append(element("span", "sw-section-label", "3 · 为什么这样判定"), element("h2", "", "判定依据"));
  const contextCount = element("span", "sw-pane__count", "-");
  contextHeader.append(contextTitle, contextCount);
  const contextSearchBar = element("form", "sw-context-search");
  const contextSearch = element("input");
  contextSearch.type = "search";
  contextSearch.placeholder = "搜索对象、状态字段或事件";
  contextSearch.setAttribute("aria-label", "搜索对象、状态字段或事件");
  const clearSearch = element("button", "sw-icon-button", "×");
  clearSearch.type = "button";
  clearSearch.title = "清除关联搜索";
  clearSearch.setAttribute("aria-label", "清除关联搜索");
  const runSearch = element("button", "sw-command-button", "搜索");
  runSearch.type = "submit";
  contextSearchBar.append(contextSearch, clearSearch, runSearch);
  const contextSummary = element("div", "sw-context-summary");
  const tabs = element("div", "sw-context-tabs");
  tabs.setAttribute("role", "tablist");
  const tabButtons = new Map();
  [["evidence", "数值证据"], ["state", "主体状态"], ["logs", "执行日志"], ["files", "数据文件"]].forEach(([value, label]) => {
    const button = element("button", "", label);
    button.type = "button";
    button.dataset.tab = value;
    button.setAttribute("role", "tab");
    tabButtons.set(value, button);
    tabs.append(button);
  });
  const contextBody = element("div", "sw-context-body");
  contextPane.append(contextHeader, contextSearchBar, contextSummary, tabs, contextBody);

  grid.append(chainPane, predicatePane, contextPane);
  view.append(header, toolbar, scenePanel, trajectoryPanel, timeline, grid);
  root.replaceChildren(view);
  const sceneResizeObserver = new ResizeObserver(() => {
    if (!disposed) renderSceneMap();
  });
  sceneResizeObserver.observe(sceneCanvasWrap);

  function closeEntityPopover() {
    if (!entityPopover) return;
    entityPopover.remove();
    entityPopover = null;
    entityPopoverAnchor = null;
    document.removeEventListener("pointerdown", handleOutsidePopover, true);
    window.removeEventListener("resize", positionEntityPopover);
  }

  function positionEntityPopover() {
    if (!entityPopover || !entityPopoverAnchor?.isConnected) return;
    const anchor = entityPopoverAnchor.getBoundingClientRect();
    const width = Math.min(540, window.innerWidth - 24);
    entityPopover.style.width = `${width}px`;
    const height = entityPopover.offsetHeight;
    const left = Math.min(window.innerWidth - width - 12, Math.max(12, anchor.left));
    const below = anchor.bottom + 8;
    const top = below + height <= window.innerHeight - 12
      ? below
      : Math.max(12, anchor.top - height - 8);
    entityPopover.style.left = `${left}px`;
    entityPopover.style.top = `${top}px`;
  }

  function handleOutsidePopover(event) {
    if (entityPopover?.contains(event.target) || entityPopoverAnchor?.contains(event.target)) return;
    closeEntityPopover();
  }

  function showEntityPopover(entityId, anchor) {
    closeEntityPopover();
    entityPopoverAnchor = anchor;
    entityPopover = buildEntityPopover(frame, entityId, closeEntityPopover);
    document.body.append(entityPopover);
    positionEntityPopover();
    document.addEventListener("pointerdown", handleOutsidePopover, true);
    window.addEventListener("resize", positionEntityPopover);
  }

  function handleWorkbenchKeydown(event) {
    if (event.key === "Escape") closeEntityPopover();
  }
  document.addEventListener("keydown", handleWorkbenchKeydown);

  function setBusy(message) {
    loadStatus.textContent = message;
    loadStatus.classList.toggle("is-busy", message !== "就绪");
  }

  function setTrajectoryStatus(tick, message) {
    trajectoryTick.textContent = `${tickLabel(tick)} · ${message}`;
  }

  function clearTrajectoryImage() {
    trajectoryLoadedKey = null;
    trajectoryImage.hidden = true;
    trajectoryImage.removeAttribute("src");
    trajectoryLoading.hidden = false;
    trajectoryLoading.textContent = "等待轨迹图…";
    trajectoryTick.textContent = "-";
  }

  function loadTrajectoryPlot(episodeId, tick) {
    if (disposed) return;
    const key = `${episodeId}:${tick}`;
    if (trajectoryLoadedKey === key) return;
    trajectoryLoadedKey = key;
    setTrajectoryStatus(tick, "读取中");
    trajectoryLoading.hidden = false;
    trajectoryLoading.textContent = "正在读取轨迹图…";
    trajectoryImage.onload = () => {
      if (disposed || trajectoryLoadedKey !== key) return;
      trajectoryImage.hidden = false;
      trajectoryLoading.hidden = true;
      setTrajectoryStatus(tick, "已载入");
    };
    trajectoryImage.onerror = () => {
      if (disposed || trajectoryLoadedKey !== key) return;
      trajectoryLoadedKey = null;
      trajectoryImage.hidden = true;
      trajectoryLoading.hidden = false;
      trajectoryLoading.textContent = "轨迹图读取失败";
      setTrajectoryStatus(tick, "读取失败");
    };
    trajectoryImage.src = trajectoryPlotPath(episodeId, tick);
  }

  function updateTickControls(tick) {
    tickInput.value = String(tick);
    range.value = String(tick);
    frameBadge.textContent = tickLabel(tick);
    previousTick.disabled = tick <= TICK_MIN;
    nextTick.disabled = tick >= TICK_MAX;
  }

  function renderFrameStats() {
    frameStats.replaceChildren();
    if (!frame) return;
    [
      ["当前判定", frame.counts.predicates],
      ["本刻变化", frame.counts.transitions],
      ["本刻事件", frame.counts.events],
      ["真实主体", frame.counts.entity_states],
    ].forEach(([label, value]) => {
      const item = element("div");
      item.append(element("span", "", label), element("strong", "", value));
      frameStats.append(item);
    });
  }

  function renderSceneSummary() {
    sceneEntityCounts.replaceChildren();
    sceneFacts.replaceChildren();
    if (!sceneMap || !frame) return;
    const descriptors = frame.entity_states
      .map(sceneEntityDescriptor)
      .filter((entity) => entity.category !== "airspace_corridor");
    const counts = new Map();
    descriptors.forEach((entity) => {
      counts.set(entity.group, (counts.get(entity.group) || 0) + 1);
    });
    Object.entries(MAP_ENTITY_GROUPS).forEach(([group, metadata]) => {
      const item = element("div", "sw-scene-entity-count");
      const swatch = element("i", "sw-scene-entity-dot");
      swatch.style.backgroundColor = metadata.color;
      item.append(
        swatch,
        element("span", "", metadata.label),
        element("strong", "", counts.get(group) || 0),
      );
      sceneEntityCounts.append(item);
    });
    [
      ["地图", sceneMap.map_id],
      ["固定范围", `${formatValue(sceneMap.roi.width_m)} × ${formatValue(sceneMap.roi.height_m)} m`],
      ["道路", `${sceneMap.counts.road_segments} 段`],
      ["建筑", `${sceneMap.counts.buildings} 栋`],
    ].forEach(([term, value]) => {
      sceneFacts.append(element("dt", "", term), element("dd", "", value));
    });
  }

  function renderSceneMap() {
    if (!sceneMap || !frame) return;
    sceneLoading.hidden = true;
    sceneTick.textContent = `${tickLabel(frame.tick)} · UE 顶视`;
    sceneCanvas.setAttribute(
      "aria-label",
      `${frame.episode_id} ${tickLabel(frame.tick)} UE 坐标顶视图`,
    );
    mapHitTargets = drawSceneMapCanvas(
      sceneCanvas,
      sceneMap,
      frame,
      focusedEntityIds,
    );
    renderSceneSummary();
  }

  function scenePointer(event) {
    const bounds = sceneCanvas.getBoundingClientRect();
    return {
      x: event.clientX - bounds.left,
      y: event.clientY - bounds.top,
    };
  }

  function sceneTargetAt(point) {
    return [...mapHitTargets].reverse().find((target) => (
      Math.hypot(point.x - target.x, point.y - target.y) <= target.radius
    ));
  }

  function hideSceneTooltip() {
    sceneTooltip.hidden = true;
    sceneCanvas.style.cursor = "default";
  }

  function showSceneTooltip(target, point) {
    const metadata = MAP_ENTITY_GROUPS[target.entity.group];
    sceneTooltip.replaceChildren(
      element("strong", "", friendlyEntityName(target.entity.id)),
      element(
        "span",
        "",
        `${metadata.label} · ${target.entity.state || "无状态"} · Z ${formatValue(target.entity.position[2])} m`,
      ),
      element(
        "code",
        "",
        `X ${formatValue(target.entity.position[0])} · Y ${formatValue(target.entity.position[1])}`,
      ),
    );
    const width = 260;
    const left = Math.min(sceneCanvas.clientWidth - width - 8, Math.max(8, point.x + 14));
    const top = Math.min(sceneCanvas.clientHeight - 74, Math.max(8, point.y + 14));
    sceneTooltip.style.left = `${left}px`;
    sceneTooltip.style.top = `${top}px`;
    sceneTooltip.hidden = false;
    sceneCanvas.style.cursor = "pointer";
  }

  function renderPredicatePane() {
    if (!frame) return;
    setButtonPressed(modeButtons, predicateMode);
    truthSelect.hidden = predicateMode !== "assertions";
    const count = predicateMode === "transitions"
      ? renderTransitionRows(predicateList, frame, predicateQuery, selection, selectTransition)
      : renderAssertionRows(predicateList, frame, predicateQuery, truthFilter, selection, selectAssertion);
    predicateCount.textContent = `${count} / ${predicateMode === "transitions" ? frame.transition_markers.length : frame.predicates.length}`;
  }

  function renderTimeline() {
    if (!frame || !chain) return;
    renderTimelineMarkers(markerLanes, frame, chain, frame.tick, processFocus, {
      authored: selectAuthored,
      transition: selectTransition,
      event: selectObjectiveEvent,
      state: selectStateTick,
    });
  }

  function updateContextTabCounts(counts) {
    const labels = {
      evidence: "数值证据",
      state: "主体状态",
      logs: "执行日志",
      files: "数据文件",
    };
    tabButtons.forEach((button, value) => {
      button.textContent = `${labels[value]} ${counts[value]}`;
    });
  }

  function clearJudgementContext(frameReady = false) {
    context = null;
    ++contextSerial;
    contextSummary.replaceChildren();
    contextCount.textContent = "未选择判定";
    const stateCount = frameReady && frame
      ? new Set(subjectStateRecords(frame).map((record) => record.subject_id)).size
      : 0;
    updateContextTabCounts({ evidence: 0, state: stateCount, logs: 0, files: 0 });
    setButtonPressed(tabButtons, contextTab);
    if (frameReady && frame && contextTab === "state") {
      renderStateRecords(
        contextBody,
        frame,
        contextQuery,
        focusedEntityIds,
        showEntityPopover,
      );
      return;
    }
    contextBody.replaceChildren(
      frameReady
        ? emptyState("选择一条状态变化或当前结果")
        : element("div", "sw-inline-loading", "正在读取当前时刻…"),
    );
  }

  function setContextTab(nextTab) {
    contextTab = nextTab;
    setButtonPressed(tabButtons, contextTab);
    if (!context) {
      if (contextTab === "state" && frame) {
        renderStateRecords(contextBody, frame, contextQuery, focusedEntityIds, showEntityPopover);
        return;
      }
      contextBody.replaceChildren(emptyState("选择一条状态变化或当前结果"));
      return;
    }
    if (contextTab === "evidence") renderEvidenceRecords(contextBody, context, showEntityPopover);
    if (contextTab === "state") renderStateRecords(contextBody, frame, contextQuery, focusedEntityIds, showEntityPopover);
    if (contextTab === "logs") renderPhysicalLogs(contextBody, context, showEntityPopover);
    if (contextTab === "files") renderFiles(contextBody, context);
  }

  function renderContext() {
    if (!context) return;
    renderContextSummary(contextSummary, context, {
      tab: setContextTab,
      transition: selectTransition,
      entity: showEntityPopover,
    });
    const counts = {
      evidence: context.counts.evidence_records,
      state: new Set(subjectStateRecords(frame).map((record) => record.subject_id)).size,
      logs: context.counts.physical_logs,
      files: context.files.length,
    };
    updateContextTabCounts(counts);
    contextCount.textContent = `${context.counts.linked_events} 个关联事件 · ${entityIdsFromContext(context).length} 个对象`;
    setContextTab(contextTab);
  }

  async function loadContext() {
    if (!frame || disposed) return;
    if (["state", "tick"].includes(selection?.kind)) {
      clearJudgementContext(true);
      setBusy("就绪");
      return;
    }
    const serial = ++contextSerial;
    contextBody.replaceChildren(element("div", "sw-inline-loading", "正在定位关联记录…"));
    const params = { tick: frame.tick, window: CONTEXT_WINDOW, limit: CONTEXT_LIMIT };
    if (selection?.kind === "transition" && selection.transitionId) params.transitionId = selection.transitionId;
    if (selection?.kind === "predicate") {
      params.predicateId = selection.predicateId;
      params.tupleId = selection.tupleId;
    }
    if ((selection?.kind === "event" || selection?.kind === "authored") && selection.eventId) {
      params.eventId = selection.eventId;
    }
    if (contextQuery.trim()) params.q = contextQuery.trim();
    try {
      const payload = await api.semanticContext(store.episodeId, params);
      if (disposed || serial !== contextSerial) return;
      if (payload.schema_name !== "semantic_context_dashboard_view") {
        throw new Error("语义上下文接口契约不匹配");
      }
      context = payload;
      if (selection?.kind !== "state") {
        const contextualEntities = entityIdsFromContext(context);
        if (contextualEntities.length) {
          focusedEntityIds = [...new Set([
            ...(processFocus?.entityIds || []),
            ...contextualEntities,
          ])];
        }
      }
      renderContext();
      renderTimeline();
      renderSceneMap();
      setBusy("就绪");
    } catch (error) {
      if (disposed || serial !== contextSerial) return;
      context = null;
      contextSummary.replaceChildren();
      contextBody.innerHTML = ui.errorPanel(error);
      contextCount.textContent = "读取失败";
      setBusy("关联读取失败");
      ui.toast(error.message, "error");
    }
  }

  function chooseDefaultSelection() {
    if (selection?.kind === "transition") {
      const marker = frame.transition_markers.find((row) => row.transition_id === selection.transitionId);
      if (marker && marker.tick === frame.tick) return;
    }
    if (selection?.kind === "predicate" && selection.tick === frame.tick) return;
    if (["state", "tick"].includes(selection?.kind) && selection.tick === frame.tick) return;
    const exact = frame.transitions[0];
    if (exact) {
      selection = {
        kind: "transition",
        transitionId: exact.transition_id,
        predicateId: exact.predicate_id,
        tupleId: exact.tuple_id,
        tick: exact.to_tick,
      };
      store.patch({ selection });
      return;
    }
    const assertion = frame.predicates.find((row) => row.value === "true") || frame.predicates[0];
    selection = assertion
      ? {
          kind: "predicate",
          predicateId: assertion.predicate_id,
          tupleId: assertion.tuple_id,
          tick: frame.tick,
        }
      : { kind: "tick", tick: frame.tick };
    store.patch({ selection });
  }

  function renderChain() {
    const milestones = chainSemanticMilestones(chain, frame);
    chainCount.textContent = `${chain.counts.nodes} 个事件 · ${milestones.length} 个状态变化`;
    runtimeBadge.textContent = chain.runtime_validation_status === "PASS" ? "事件顺序正常" : "事件顺序需检查";
    runtimeBadge.classList.toggle("is-pass", chain.runtime_validation_status === "PASS");
    renderChainFlow(
      chainFlow,
      chain,
      milestones,
      selectedAuthoredId,
      selection?.transitionId,
      frame.tick,
      { authored: selectAuthored, transition: selectTransition },
    );
    const selectedNode = chain.nodes.find((node) => node.id === selectedAuthoredId)
      || chain.nodes.find((node) => node.dispatch_tick === frame.tick)
      || null;
    renderChainSelection(
      chainSelection,
      selectedNode,
    );
    renderSceneMap();
  }

  async function loadFrame() {
    const episodeId = store.episodeId;
    const tick = clampTick(store.tick);
    const serial = ++loadSerial;
    const episodeChanged = loadedEpisode !== episodeId;
    closeEntityPopover();
    setBusy("读取语义帧");
    updateTickControls(tick);
    episodeSelect.value = episodeId;
    if (episodeChanged) {
      sceneLoading.hidden = false;
      sceneLoading.textContent = "正在读取场景底图…";
      sceneTooltip.hidden = true;
      mapHitTargets = [];
    }
    try {
      const requests = [api.timeline(episodeId, tick)];
      if (episodeChanged) requests.push(api.eventChain(episodeId), api.sceneMap(episodeId));
      const payloads = await Promise.all(requests);
      if (disposed || serial !== loadSerial) return;
      frame = payloads[0];
      if (episodeChanged) {
        chain = payloads[1];
        sceneMap = payloads[2];
        if (sceneMap?.schema_name !== "scene_map_dashboard_view") {
          throw new Error("场景底图接口契约不匹配");
        }
        if (sceneMap.episode_id !== episodeId) {
          throw new Error("场景底图返回了错误的 Episode");
        }
        loadedEpisode = episodeId;
        selection = null;
        selectedAuthoredId = null;
        processFocus = null;
        focusedEntityIds = [];
        context = null;
        contextQuery = "";
        contextSearch.value = "";
        contextTab = "evidence";
        ++contextSerial;
        store.patch({ selection: null });
      }
      if (!frame || frame.episode_id !== episodeId || frame.tick !== tick) {
        throw new Error("语义帧返回了错误的 Episode 或 tick");
      }
      if (!chain || chain.episode_id !== episodeId) throw new Error("场景事件链不可用");
      if (!sceneMap || sceneMap.episode_id !== episodeId) throw new Error("场景底图不可用");
      chooseDefaultSelection();
      renderFrameStats();
      renderSceneMap();
      loadTrajectoryPlot(episodeId, tick);
      renderChain();
      renderPredicatePane();
      renderTimeline();
      if (["state", "tick"].includes(selection?.kind)) {
        clearJudgementContext(true);
        setBusy("就绪");
        return;
      }
      await loadContext();
    } catch (error) {
      if (disposed || serial !== loadSerial) return;
      setBusy("数据读取失败");
      sceneLoading.hidden = false;
      sceneLoading.textContent = "场景读取失败";
      grid.replaceChildren();
      const errorNode = element("div", "sw-workbench-error");
      errorNode.innerHTML = ui.errorPanel(error);
      grid.append(errorNode);
      ui.toast(error.message, "error");
    }
  }

  function focusSemanticProcess(milestone) {
    processFocus = processFocusForMilestone(frame, milestone);
    focusedEntityIds = processFocus?.entityIds || [];
  }

  function changeTick(nextTick, nextSelection = null) {
    const tick = clampTick(nextTick);
    hideSceneTooltip();
    selection = nextSelection || { kind: "state", tick };
    if (["state", "tick"].includes(selection.kind)) {
      selectedAuthoredId = null;
      processFocus = null;
      focusedEntityIds = [];
      contextQuery = "";
      contextSearch.value = "";
      clearJudgementContext(false);
    }
    store.patch({ tick, selection });
    updateTickControls(tick);
    loadFrame();
  }

  function selectStateTick(record) {
    contextQuery = "";
    contextSearch.value = "";
    contextTab = "state";
    changeTick(record.tick, { kind: "state", tick: Number(record.tick) });
  }

  function selectTransition(record) {
    contextQuery = "";
    contextSearch.value = "";
    selection = {
      kind: "transition",
      transitionId: record.transition_id,
      predicateId: record.predicate_id,
      tupleId: record.tuple_id,
      tick: Number(record.tick ?? record.to_tick),
    };
    const milestone = (frame.semantic_milestones || []).find((row) => row.transition_id === record.transition_id);
    focusSemanticProcess(milestone);
    const authoredMatches = milestone
      ? chain.nodes.filter((node) => Number(node.dispatch_tick) === Number(milestone.event_trigger_tick))
      : [];
    selectedAuthoredId = authoredMatches.length === 1 ? authoredMatches[0].id : null;
    const tick = clampTick(selection.tick);
    store.patch({ tick, selection });
    if (!frame || frame.tick !== tick) loadFrame();
    else {
      renderChain();
      renderPredicatePane();
      loadContext();
    }
  }

  function selectAssertion(record) {
    contextQuery = "";
    contextSearch.value = "";
    selection = {
      kind: "predicate",
      predicateId: record.predicate_id,
      tupleId: record.tuple_id,
      tick: frame.tick,
    };
    selectedAuthoredId = null;
    processFocus = null;
    focusedEntityIds = [...new Set(Object.values(record.bindings || {}).filter(Boolean))];
    store.patch({ selection });
    renderChain();
    renderPredicatePane();
    loadContext();
  }

  function selectAuthored(record) {
    selectedAuthoredId = record.id;
    const milestones = chainSemanticMilestones(chain, frame).filter((row) => (
      Number(row.event_trigger_tick) === Number(record.dispatch_tick)
    ));
    const objectiveEventIds = new Set(milestones.map((row) => row.event_id));
    const onset = objectiveEventIds.size === 1
      ? milestones.find((row) => row.phase_kind === "onset") || milestones[0]
      : null;
    focusSemanticProcess(onset);
    if (onset) {
      contextQuery = "";
      contextSearch.value = "";
      selection = {
        kind: "transition",
        transitionId: onset.transition_id,
        predicateId: onset.predicate_id,
        tupleId: onset.tuple_id,
        tick: Number(onset.tick),
      };
    } else {
      contextQuery = record.id;
      contextSearch.value = record.id;
      selection = { kind: "authored", eventId: record.id, tick: record.dispatch_tick };
    }
    const tick = clampTick(record.dispatch_tick);
    store.patch({ tick, selection });
    if (!frame || frame.tick !== tick) loadFrame();
    else {
      renderChain();
      renderPredicatePane();
      loadContext();
    }
  }

  function selectObjectiveEvent(record) {
    contextQuery = "";
    contextSearch.value = "";
    const milestones = (frame.semantic_milestones || []).filter((row) => row.event_id === record.event_id);
    const onset = milestones.find((row) => row.phase_kind === "onset") || milestones[0];
    focusSemanticProcess(onset);
    selection = onset
      ? {
          kind: "transition",
          transitionId: onset.transition_id,
          predicateId: onset.predicate_id,
          tupleId: onset.tuple_id,
          tick: Number(onset.tick),
        }
      : { kind: "event", eventId: record.event_id, tick: record.tick };
    const authoredMatches = chain.nodes.filter((node) => Number(node.dispatch_tick) === Number(record.tick));
    selectedAuthoredId = authoredMatches.length === 1 ? authoredMatches[0].id : null;
    const tick = clampTick(selection.tick);
    store.patch({ tick, selection });
    if (!frame || frame.tick !== tick) loadFrame();
    else loadContext();
  }

  episodeSelect.addEventListener("change", () => {
    ++contextSerial;
    clearTrajectoryImage();
    closeEntityPopover();
    frame = null;
    chain = null;
    selection = null;
    context = null;
    sceneMap = null;
    mapHitTargets = [];
    selectedAuthoredId = null;
    processFocus = null;
    focusedEntityIds = [];
    contextQuery = "";
    contextSearch.value = "";
    contextTab = "evidence";
    sceneLoading.hidden = false;
    sceneLoading.textContent = "正在读取场景底图…";
    sceneTick.textContent = "-";
    sceneEntityCounts.replaceChildren();
    sceneFacts.replaceChildren();
    hideSceneTooltip();
    frameStats.replaceChildren();
    contextCount.textContent = "未选择判定";
    updateContextTabCounts({ evidence: 0, state: 0, logs: 0, files: 0 });
    chainFlow.replaceChildren(element("div", "sw-inline-loading", "正在读取新场景…"));
    predicateList.replaceChildren(element("div", "sw-inline-loading", "正在读取新场景…"));
    contextSummary.replaceChildren();
    contextBody.replaceChildren(element("div", "sw-inline-loading", "正在读取新场景…"));
    store.patch({ episodeId: episodeSelect.value, tick: TICK_MIN, selection: null });
    loadFrame();
  });
  previousTick.addEventListener("click", () => changeTick(store.tick - TICK_STEP));
  nextTick.addEventListener("click", () => changeTick(store.tick + TICK_STEP));
  tickInput.addEventListener("change", () => changeTick(tickInput.value));
  range.addEventListener("input", () => {
    const tick = clampTick(range.value);
    selection = { kind: "state", tick };
    clearJudgementContext(false);
    updateTickControls(tick);
    clearTimeout(tickTimer);
    tickTimer = setTimeout(() => changeTick(tick), 140);
  });
  sceneCanvas.addEventListener("pointermove", (event) => {
    const point = scenePointer(event);
    const target = sceneTargetAt(point);
    if (!target) {
      hideSceneTooltip();
      return;
    }
    showSceneTooltip(target, point);
  });
  sceneCanvas.addEventListener("pointerleave", hideSceneTooltip);
  sceneCanvas.addEventListener("click", (event) => {
    const target = sceneTargetAt(scenePointer(event));
    if (target) showEntityPopover(target.entityId, sceneCanvas);
  });
  modeControl.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-mode]");
    if (!button) return;
    predicateMode = button.dataset.mode;
    renderPredicatePane();
  });
  predicateSearch.addEventListener("input", () => {
    predicateQuery = predicateSearch.value;
    renderPredicatePane();
  });
  truthSelect.addEventListener("change", () => {
    truthFilter = truthSelect.value;
    renderPredicatePane();
  });
  tabs.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-tab]");
    if (button) setContextTab(button.dataset.tab);
  });
  contextSearchBar.addEventListener("submit", (event) => {
    event.preventDefault();
    contextQuery = contextSearch.value;
    if (!context) {
      setContextTab("state");
      return;
    }
    loadContext();
  });
  contextSearch.addEventListener("input", () => {
    if (!context) {
      contextQuery = contextSearch.value;
      if (contextTab === "state") {
        renderStateRecords(contextBody, frame, contextQuery, focusedEntityIds, showEntityPopover);
      }
      return;
    }
    clearTimeout(contextTimer);
    contextTimer = setTimeout(() => {
      contextQuery = contextSearch.value;
      loadContext();
    }, 360);
  });
  clearSearch.addEventListener("click", () => {
    contextSearch.value = "";
    contextQuery = "";
    if (!context) {
      setContextTab(contextTab);
      return;
    }
    loadContext();
  });

  setButtonPressed(modeButtons, predicateMode);
  setButtonPressed(tabButtons, contextTab);
  episodeSelect.value = store.episodeId;
  await loadFrame();

  return () => {
    disposed = true;
    clearTimeout(tickTimer);
    clearTimeout(contextTimer);
    closeEntityPopover();
    sceneResizeObserver.disconnect();
    document.removeEventListener("keydown", handleWorkbenchKeydown);
  };
}
