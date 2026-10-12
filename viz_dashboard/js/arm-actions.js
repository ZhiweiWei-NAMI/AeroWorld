// ARM 动作格式化 / 身份对齐 / 分支对照（纯数据模块）。
// 不 import DOM / arm-common；表格与浮层 DOM 在 arm-action-ui.js。
// 分支基线 = 组中唯一 factual_observation；缺 factual 或有多个 factual 时
// 显式标为不可比，不 fallback、不拿干预当基线。

export const ACTION_TYPE_LABEL = Object.freeze({
  move_entity: "移动实体",
  set_visual_state: "设置视觉状态",
  set_runtime_state: "设置运行时状态",
  capture_screenshot: "截图请求",
  spawn_entity: "生成实体",
  remove_entity: "移除实体",
  set_pedestrian_activity: "设置行人活动",
  set_weather: "设置天气",
});

const STATUS_LABEL = Object.freeze({
  ok: "执行记录 ok",
  omitted_by_explicit_intervention: "干预省略",
  omitted_by_configuration: "配置省略",
});

export const NO_RECORD_TEXT = "窗口内无此动作记录";
export const NO_RECORD_FILTERED_TEXT = "当前时间筛选内无记录";
export const NO_PARAMS_COMPARABLE_TEXT = "命令参数未提供，参数不可比";
export const NO_FACTUAL_BASELINE_TEXT = "本组没有事实观察分支，无法进行基线对比";
export const MULTI_FACTUAL_BASELINE_TEXT = count =>
  `本组有 ${count} 个事实观察分支，未指定唯一基线，不做对比`;
export const DIFF_TEXT = "与事实观察基线不同";

export function statusLabel(status, actionType) {
  if (status === undefined || status === null || status === "") return "无执行状态";
  if (actionType === "capture_screenshot" && status === "ok") return "截图请求已登记";
  return STATUS_LABEL[status] || String(status);
}

export function isOmittedStatus(status) {
  return status === "omitted_by_explicit_intervention"
    || status === "omitted_by_configuration";
}

export function actionTypeLabel(type) {
  return ACTION_TYPE_LABEL[type] || String(type);
}

// 稳定键排序的 JSON 序列化，用于参数比较。
export function stableKeyString(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value) ?? "null";
  if (Array.isArray(value)) return `[${value.map(stableKeyString).join(",")}]`;
  const keys = Object.keys(value).sort();
  return `{${keys.map(k => `${JSON.stringify(k)}:${stableKeyString(value[k])}`).join(",")}}`;
}

// 必需字段校验：非空适当类型；tick 必须有限整数。null/undefined 明确报错。
function requireNonEmptyString(value, what) {
  if (typeof value !== "string" || value === "") {
    throw new Error(`${what} 必须是非空字符串，实际为 ${JSON.stringify(value) ?? String(value)}`);
  }
  return value;
}

function requireFiniteIntegerTick(value, what) {
  if (typeof value !== "number" || !Number.isInteger(value) || !Number.isFinite(value)) {
    throw new Error(`${what} 必须是有限整数 tick，实际为 ${JSON.stringify(value) ?? String(value)}`);
  }
  return value;
}

// 对象扁平化（路径 -> 值），用于 state_patch 展示。
function flatten(obj, prefix = "", out = {}) {
  for (const [key, value] of Object.entries(obj)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (value !== null && typeof value === "object" && !Array.isArray(value)) {
      flatten(value, path, out);
    } else {
      out[path] = value;
    }
  }
  return out;
}

function fmtNum(v) {
  return typeof v === "number" ? String(Math.round(v * 1000) / 1000) : String(v);
}

function fmtPos(p) {
  if (!Array.isArray(p) || p.length < 2) return stableKeyString(p);
  return `(${p.map(fmtNum).join(", ")})`;
}

// 单条普通动作的参数摘要（中文，逐行）。effect 若存在则纳入实测差异。
export function formatActionParams(action, result, effect) {
  if (!action || typeof action !== "object") throw new Error("动作行缺少 action 对象");
  const lines = [];
  const push = (k, v) => { if (v !== undefined && v !== null && v !== "") lines.push(`${k}：${v}`); };
  switch (action.type) {
    case "move_entity": {
      const wp = action.waypoints_enu_m;
      if (Array.isArray(wp) && wp.length) {
        push("命令/目标路径", `${fmtPos(wp[0])} → ${fmtPos(wp[wp.length - 1])} · ${wp.length} 个航点`);
      } else {
        lines.push("命令/目标路径：本记录未提供航点");
      }
      if (action.velocity_mps !== undefined) push("命令速度", `${fmtNum(action.velocity_mps)} m/s`);
      // 动作结果记录，不等同已飞行距离
      if (result?.path_length_m !== undefined) push("路径长度记录", `${fmtNum(result.path_length_m)} m`);
      break;
    }
    case "set_visual_state": {
      const vs = action.visual_state;
      if (vs !== undefined && vs !== null) push("命令 visual_state", stableKeyString(vs));
      else lines.push("本记录未提供动作参数");
      break;
    }
    case "set_runtime_state": {
      const patch = action.state_patch;
      if (patch !== null && typeof patch === "object") {
        const flat = flatten(patch);
        const parts = Object.entries(flat).map(([k, v]) => `${k}=${typeof v === "object" ? stableKeyString(v) : String(v)}`);
        push("命令/目标 state_patch", parts.join("；"));
      } else {
        lines.push("本记录未提供动作参数");
      }
      const sched = result?.scheduled_tick;
      const eff = result?.effective_tick;
      if (sched !== undefined || eff !== undefined) {
        push("调度/生效 tick", `scheduled_tick ${sched ?? "?"} · effective_tick ${eff ?? "?"}`);
      }
      break;
    }
    case "capture_screenshot": {
      push("相机配置", action.camera_id ?? "(未声明)");
      if (result?.capture_id !== undefined) push("请求编号", result.capture_id);
      break;
    }
    case "spawn_entity": {
      if (action.asset_id !== undefined) push("asset", action.asset_id);
      if (action.position_enu_m !== undefined) push("命令/目标位置", fmtPos(action.position_enu_m));
      if (action.rotation_deg !== undefined) push("rotation", stableKeyString(action.rotation_deg));
      if (!lines.length) lines.push("本记录未提供动作参数");
      break;
    }
    case "remove_entity": {
      push("对象", action.entity_id ?? "(未声明)");
      if (result?.effective_tick !== undefined) push("effective_tick", result.effective_tick);
      break;
    }
    case "set_pedestrian_activity": {
      if (action.activity_type !== undefined) push("命令 activity", action.activity_type);
      else lines.push("本记录未提供动作参数");
      break;
    }
    case "set_weather": {
      if (action.profile !== undefined) push("profile", action.profile);
      if (action.overrides !== undefined) push("overrides", stableKeyString(action.overrides));
      if (effect && typeof effect === "object") {
        const before = effect.before ?? effect.before_state;
        const after = effect.after ?? effect.after_state;
        if (before && after) {
          const keys = [...new Set([...Object.keys(before), ...Object.keys(after)])];
          const diffs = keys.filter(k => stableKeyString(before[k]) !== stableKeyString(after[k]))
            .map(k => `${k}: ${typeof before[k] === "object" ? stableKeyString(before[k]) : before[k]} → ${typeof after[k] === "object" ? stableKeyString(after[k]) : after[k]}`);
          if (diffs.length) push("effect 实测", diffs.join("；"));
          else push("effect 实测", "before 与 after 无差异");
        } else {
          push("effect", stableKeyString(effect));
        }
      }
      if (!lines.length) lines.push("本记录未提供动作参数");
      break;
    }
    default:
      // 未知非空 type：保留原 type 与参数，不吞掉信息、不当成功。
      lines.push(`未知动作类型 ${action.type}，参数：${stableKeyString(action)}`);
  }
  return lines;
}

// 历史事件实现行（origin === "historical_event_realization"）展开成动作记录。
// 缺必需字段抛明确错误；evidence 保留整条原始行。
// 历史记录没有命令参数，参数一律显示中性“不可比”，不猜真值。
export function historicalRealizationRecords(row) {
  const raw = row?.evidence?.raw;
  if (!raw || raw.origin !== "historical_event_realization") {
    throw new Error("historicalRealizationRecords 收到非 historical_event_realization 行");
  }
  const src = raw.source_record;
  if (!src || typeof src !== "object") throw new Error("historical_event_realization 行缺少 source_record");
  const list = src.action_realizations;
  if (!Array.isArray(list) || !list.length) {
    throw new Error(`historical_event_realization 行 ${src.event_id ?? "?"} 缺少 action_realizations`);
  }
  return list.map((rz, index) => {
    requireNonEmptyString(rz.action_id, `action_realizations[${index}].action_id（event ${src.event_id ?? "?"}）`);
    requireNonEmptyString(rz.action_type, `action_realizations[${index}].action_type（event ${src.event_id ?? "?"}）`);
    requireFiniteIntegerTick(rz.dispatch_tick, `action_realizations[${index}].dispatch_tick（event ${src.event_id ?? "?"}）`);
    requireNonEmptyString(rz.status, `action_realizations[${index}].status（event ${src.event_id ?? "?"}）`);
    const scope = rz.entity_id ? rz.entity_id : "global";
    const metadataOnly = rz.metadata_only === true;
    const statusText = metadataOnly
      ? `${statusLabel(rz.status, rz.action_type)}（历史执行记录 · metadata_only，真实值见 truth snapshots）`
      : `${statusLabel(rz.status, rz.action_type)}（历史执行记录）`;
    return {
      kind: "action",
      historical: true,
      metadataOnly,
      realizationIndex: index,
      tick: rz.dispatch_tick,
      actionId: rz.action_id,
      type: rz.action_type,
      scope,
      status: rz.status,
      title: rz.action_id,
      detail: `${actionTypeLabel(rz.action_type)} · ${statusLabel(rz.status, rz.action_type)}`,
      evidence: row.evidence,
      raw: row,
      actionInfo: {
        key: `${rz.action_type}|${rz.action_id}|${scope}`,
        typeLabel: actionTypeLabel(rz.action_type),
        subject: rz.entity_id ? rz.entity_id : "global（事件级，无单一实体）",
        statusText,
        omitted: isOmittedStatus(rz.status),
        paramsLines: [NO_PARAMS_COMPARABLE_TEXT],
      },
    };
  });
}

// 普通动作行 → 单条记录。缺必需字段抛明确错误。
export function actionRecordFromRow(row) {
  const raw = row?.evidence?.raw;
  if (!raw || typeof raw !== "object") throw new Error("动作行缺少 evidence.raw");
  const action = raw.action;
  if (!action || typeof action !== "object") throw new Error("动作行缺少 action 对象");
  requireNonEmptyString(action.type, "动作行 action.type");
  requireNonEmptyString(action.action_id, "动作行 action.action_id");
  const status = raw.result?.status;
  requireNonEmptyString(status, `动作 ${action.action_id} 的 result.status`);
  requireFiniteIntegerTick(raw.tick, `动作 ${action.action_id} 的 tick`);
  const scope = action.entity_id ? action.entity_id
    : (action.camera_id ? action.camera_id : "global");
  const omitted = isOmittedStatus(status);
  const paramsLines = omitted
    ? ["（省略执行，无可核验参数结果）", ...formatActionParams(action, raw.result, raw.effect)]
    : formatActionParams(action, raw.result, raw.effect);
  return {
    kind: "action",
    historical: false,
    tick: raw.tick,
    actionId: action.action_id,
    type: action.type,
    scope,
    status,
    title: action.action_id,
    detail: `${actionTypeLabel(action.type)} · ${statusLabel(status, action.type)}`,
    evidence: row.evidence,
    raw: row,
    actionInfo: {
      key: `${action.type}|${action.action_id}|${scope}`,
      typeLabel: actionTypeLabel(action.type),
      subject: action.entity_id ?? (action.camera_id ? `相机配置 ${action.camera_id}` : "global（无单一实体）"),
      statusText: statusLabel(status, action.type),
      omitted,
      paramsLines,
    },
  };
}

// 一行 raw_actions → 0 或多条动作记录（当前分支的两种 origin 分别判断，不互为兜底）。
export function actionRecordsFromRow(row) {
  const raw = row?.evidence?.raw;
  if (!raw || typeof raw !== "object") throw new Error("动作行缺少 evidence.raw");
  if (raw.origin === "historical_event_realization") {
    return historicalRealizationRecords(row);
  }
  return [actionRecordFromRow(row)];
}

// 一个臂的全部动作记录（展开后）。
export function armActionRecords(arm) {
  const out = [];
  for (const row of arm.raw_actions) out.push(...actionRecordsFromRow(row));
  return out;
}

// 命令参数的可比较字符串。历史记录没有命令参数 → null（参数不可判）。
// 普通记录取完整 action 对象（稳定键排序）；identity 字段同身份本就相等。
export function actionCommandParams(record) {
  if (record.historical) return null;
  const action = record.raw?.evidence?.raw?.action;
  if (!action || typeof action !== "object") return null;
  return stableKeyString(action);
}

// 比较两方共有的实际 tick 与 status；两方都有完整命令参数时再比较参数。
// 历史记录缺参数不判参数差异；tick/status 不同仍标差异。
// created_schedule / origin / path 等固有 provenance 一律不参与比较。
export function actionsDiffer(a, b) {
  if (a.tick !== b.tick) return true;
  if (a.status !== b.status) return true;
  const pa = actionCommandParams(a);
  const pb = actionCommandParams(b);
  if (pa !== null && pb !== null && pa !== pb) return true;
  return false;
}

// 单记录可比形状（浮层/调试用）：tick、status、命令参数（可能为 null）。
export function actionComparable(record) {
  return stableKeyString({
    tick: record.tick,
    status: record.status,
    params: actionCommandParams(record),
  });
}

// 同身份基线摘要（浮层用）。没有对应身份返回 null。
export function factualIdentitySummary(arm, key) {
  const hits = armActionRecords(arm).filter(r => r.actionInfo.key === key);
  if (!hits.length) return null;
  return hits.map(r =>
    `t=${r.tick} · ${r.actionInfo.typeLabel} · ${r.actionInfo.statusText}`
    + (r.actionInfo.paramsLines.length ? ` · ${r.actionInfo.paramsLines.join("；")}` : ""));
}

// ── 分支对照表（纯数据部分）───────────────────────────────────────
// 返回 { baseline, baselineNote, selected, rows: [{key, label, sample,
//   cells: Map(armKey -> {records, marks, missing, filteredOut})}] }
// 基线 = 组中唯一 factual_observation，其记录取全量（不受 selectedArms 影响）。
// 对比规则：同身份下先按实际 tick 对齐，参数两方都有才比较；
// 记录条数或可比多重集合不同也标差异；基线自身不标。
export function buildActionComparison(detail, state) {
  const selected = detail.arms.filter(a => state.selectedArms.has(a.arm_key)
    && state.kinds.has("action"));
  const inRange = r => {
    if (!state.range) return true;
    return r.tick === null || (r.tick >= state.range[0] && r.tick <= state.range[1]);
  };
  const factualArms = detail.arms.filter(a => a.mode === "factual_observation");
  const baseline = factualArms.length === 1 ? factualArms[0] : null;
  const baselineNote = factualArms.length === 0
    ? NO_FACTUAL_BASELINE_TEXT
    : (factualArms.length > 1 ? MULTI_FACTUAL_BASELINE_TEXT(factualArms.length) : null);

  // 每臂全量记录与窗口内记录分开：基线必须来自全量过滤，不受选中列影响。
  const allByArm = new Map();
  for (const arm of detail.arms) allByArm.set(arm.arm_key, armActionRecords(arm));
  const byIdentity = (records, key) => records.filter(r => r.actionInfo.key === key);
  const baseRecords = baseline ? allByArm.get(baseline.arm_key).filter(inRange) : [];

  const keys = new Set();
  for (const arm of selected) {
    for (const r of allByArm.get(arm.arm_key)) {
      if (inRange(r)) keys.add(r.actionInfo.key);
    }
  }
  for (const r of baseRecords) keys.add(r.actionInfo.key);

  const baseByKey = new Map();
  for (const r of baseRecords) {
    if (!baseByKey.has(r.actionInfo.key)) baseByKey.set(r.actionInfo.key, []);
    baseByKey.get(r.actionInfo.key).push(r);
  }

  // 多重集合比较：与 actionsDiffer 同一可比语义（tick/status 必比；命令参数
  // 两方都有才比），逐条贪心配对，配不满即为差异。重复计数通过 used 集合保留，
  // 不把历史 params=null 与正常 params 误判成差异。
  const multisetDiffers = (base, recs) => {
    if (base.length !== recs.length) return true;
    const used = new Set();
    for (const r of recs) {
      const hit = base.find((b, i) => !used.has(i) && !actionsDiffer(r, b));
      if (hit === undefined) return true;
      used.add(base.indexOf(hit));
    }
    return false;
  };

  const rows = [...keys].sort().map(key => {
    const base = baseByKey.get(key) || [];
    const baseByTick = new Map();
    for (const b of base) {
      if (!baseByTick.has(b.tick)) baseByTick.set(b.tick, []);
      baseByTick.get(b.tick).push(b);
    }
    const sample = base[0]
      || selected.flatMap(arm => byIdentity(allByArm.get(arm.arm_key), key))[0] || null;
    const cells = new Map();
    let differsAny = false;
    for (const arm of selected) {
      const full = byIdentity(allByArm.get(arm.arm_key), key);
      const recs = full.filter(inRange);
      const missing = recs.length === 0;
      const filteredOut = missing && full.length > 0;
      let differsAnyCell = false;
      const marks = recs.map(r => {
        if (!baseline || arm === baseline) return false;
        const sameTick = baseByTick.get(r.tick) || [];
        const diff = !sameTick.some(b => !actionsDiffer(r, b));
        if (diff) differsAnyCell = true;
        return diff;
      });
      if (baseline && arm !== baseline) {
        if (multisetDiffers(base, recs)) differsAnyCell = true;
      }      if (differsAnyCell) differsAny = true;
      cells.set(arm.arm_key, { records: recs, marks, missing, filteredOut, differs: differsAnyCell });
    }
    return {
      key,
      label: sample
        ? `${sample.actionInfo.typeLabel} · ${sample.actionId} · ${sample.scope}`
        : key,
      sample,
      baselineCount: base.length,
      cells,
      differsAny,
    };
  });
  return { baseline, baselineNote, selected, rows };
}
