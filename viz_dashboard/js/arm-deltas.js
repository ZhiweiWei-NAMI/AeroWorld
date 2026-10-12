// ARM 差异审查纯数据模块：每个干预分支相对本组唯一 factual_observation 的
// 三类差异 —— 脚本动作、外部干预条件（配置声明）、实测事件结果。
// 只做数据与比较，不做 DOM；DOM 在 arm-delta-ui.js。
// 所有比较基于实际产物行；无法建立身份/不可比的字段显式标出，不做兜底。

import {
  armActionRecords, actionsDiffer, actionCommandParams,
  NO_FACTUAL_BASELINE_TEXT, MULTI_FACTUAL_BASELINE_TEXT,
} from "./arm-actions.js";

export { NO_FACTUAL_BASELINE_TEXT, MULTI_FACTUAL_BASELINE_TEXT };

// 本组唯一 factual_observation 基线；缺失或多个都显式不可比，不 fallback。
export function factualBaseline(detail) {
  const factualArms = detail.arms.filter(a => a.mode === "factual_observation");
  if (factualArms.length === 1) return { baseline: factualArms[0], baselineNote: null };
  if (factualArms.length === 0) {
    return { baseline: null, baselineNote: NO_FACTUAL_BASELINE_TEXT };
  }
  return { baseline: null, baselineNote: MULTI_FACTUAL_BASELINE_TEXT(factualArms.length) };
}

// ── 值包装：不存在（字段缺失）与 null 必须区分 ────────────────────
export const ABSENT = Symbol("absent");
export function present(value) { return { absent: false, value }; }
export function absent() { return { absent: true, value: null }; }
export function wrap(value) { return value === ABSENT ? absent() : present(value); }

function stableValue(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableValue).join(",")}]`;
  const keys = Object.keys(value).sort();
  return `{${keys.map(k => `${JSON.stringify(k)}:${stableValue(value[k])}`).join(",")}}`;
}

function sameValue(a, b) {
  if (a.absent !== b.absent) return false;
  if (a.absent) return true;
  return stableValue(a.value) === stableValue(b.value);
}

// 扁平化嵌套对象路径（数组作为整体值，不展开下标）。
function flattenValue(value, prefix, out) {
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    for (const [k, v] of Object.entries(value)) {
      flattenValue(v, prefix ? `${prefix}.${k}` : k, out);
    }
  } else {
    out[prefix] = value;
  }
  return out;
}

// ── 1. 外部干预条件（配置声明）──────────────────────────────────
// 只比较 configuration.content 中真实声明干预的四个根；
// arm_id / seed / profile / entity_ids / id / path / provenance 类字段不参与。
export const CONFIG_DELTA_ROOTS = Object.freeze([
  "sumo_intervention", "state_overrides", "retime", "mechanism",
]);

// 返回 [{field, baseline, current}]；baseline/current 为 {absent, value} 包装。
// 只有真实不同的字段出现；不可比（一边是对象一边是标量等）仍按值不同列出。
export function configDeltas(baselineArm, itvArm) {
  const baseCfg = baselineArm?.configuration?.content;
  const itvCfg = itvArm?.configuration?.content;
  if (!baseCfg || typeof baseCfg !== "object") {
    throw new Error(`臂 ${baselineArm?.arm_key ?? "?"} 缺少 configuration.content，无法对比外部干预条件`);
  }
  if (!itvCfg || typeof itvCfg !== "object") {
    throw new Error(`臂 ${itvArm?.arm_key ?? "?"} 缺少 configuration.content，无法对比外部干预条件`);
  }
  const out = [];
  for (const root of CONFIG_DELTA_ROOTS) {
    const b = Object.prototype.hasOwnProperty.call(baseCfg, root) ? wrap(baseCfg[root]) : absent();
    const c = Object.prototype.hasOwnProperty.call(itvCfg, root) ? wrap(itvCfg[root]) : absent();
    if (sameValue(b, c)) continue;
    const bObj = !b.absent && b.value !== null && typeof b.value === "object";
    const cObj = !c.absent && c.value !== null && typeof c.value === "object";
    if (bObj || cObj) {
      // 至少一侧是对象：逐字段列出（基线 null/缺失 → 逐字段 absent），
      // 避免“null → 整块对象”式 dump；真实 L3 差异是 release_tick 单字段。
      const fb = bObj ? flattenValue(b.value, root, {})
        : (b.absent ? {} : { [root]: b.value });
      const fc = cObj ? flattenValue(c.value, root, {})
        : (c.absent ? {} : { [root]: c.value });
      const fields = [...new Set([...Object.keys(fb), ...Object.keys(fc)])].sort();
      for (const field of fields) {
        const bv = Object.prototype.hasOwnProperty.call(fb, field) ? present(fb[field]) : absent();
        const cv = Object.prototype.hasOwnProperty.call(fc, field) ? present(fc[field]) : absent();
        if (sameValue(bv, cv)) continue;
        out.push({ field, baseline: bv, current: cv, baselineEvidence: baselineArm.configuration, currentEvidence: itvArm.configuration });
      }
    } else {
      out.push({ field: root, baseline: b, current: c, baselineEvidence: baselineArm.configuration, currentEvidence: itvArm.configuration });
    }
  }
  return out;
}

// ── 2. 脚本动作差异（相对事实观察基线）───────────────────────────
// state 由调用方构造（默认全选中、全窗口）。返回 {baseline, baselineNote,
// deltas: [{key, label, kind: "changed"|"incomparable"|"missing"|"added",
//           left, right, fieldDiffs}]}。
// left/right 为动作记录或 null；missing 时 left 为基线记录，added 时 right
// 为分支记录。每条记录最多配对一次，配对不靠重复分支条件，只靠一次性消去。

// 一对已配对记录的逐字段差异：tick、status，以及命令 action 对象的逐叶字段
// （数组作为整体值）。只存真实变化；一侧字段缺失用 absent 包装，不改成 null。
// 历史记录没有命令参数（raw.action 缺失）时不猜字段，只比 tick/status。
function pairedFieldDiffs(left, right) {
  const diffs = [];
  const lv = wrap(left.tick);
  const rv = wrap(right.tick);
  if (!sameValue(lv, rv)) diffs.push({ field: "tick", baseline: lv, current: rv, baselineEvidence: left.evidence, currentEvidence: right.evidence });
  const ls = wrap(left.status);
  const rs = wrap(right.status);
  if (!sameValue(ls, rs)) diffs.push({ field: "status", baseline: ls, current: rs, baselineEvidence: left.evidence, currentEvidence: right.evidence });
  const lraw = left.evidence?.raw;
  const rraw = right.evidence?.raw;
  const lAction = lraw && typeof lraw === "object" ? lraw.action : undefined;
  const rAction = rraw && typeof rraw === "object" ? rraw.action : undefined;
  const lObj = lAction !== null && typeof lAction === "object" && !Array.isArray(lAction);
  const rObj = rAction !== null && typeof rAction === "object" && !Array.isArray(rAction);
  // 任一侧没有实际 action 对象（含历史记录）：不猜参数字段。
  if (!lObj || !rObj) return diffs;
  const fl = flattenValue(lAction, "action", {});
  const fr = flattenValue(rAction, "action", {});
  const fields = [...new Set([...Object.keys(fl), ...Object.keys(fr)])].sort();
  for (const field of fields) {
    const lHas = Object.prototype.hasOwnProperty.call(fl, field);
    const rHas = Object.prototype.hasOwnProperty.call(fr, field);
    const lfv = lHas ? present(fl[field]) : absent();
    const rfv = rHas ? present(fr[field]) : absent();
    if (sameValue(lfv, rfv)) continue;
    diffs.push({ field, baseline: lfv, current: rfv, baselineEvidence: left.evidence, currentEvidence: right.evidence });
  }
  return diffs;
}

const byTickThenId = (a, b) =>
  (a.tick ?? -1) - (b.tick ?? -1) || String(a.actionId).localeCompare(String(b.actionId));

const actionLabel = r => `${r.actionInfo.typeLabel} · ${r.actionId} · ${r.scope}`;

export function actionDeltas(detail, state) {
  const { baseline: baselineArm, baselineNote } = factualBaseline(detail);
  if (!baselineArm) {
    return { baseline: null, baselineNote, deltas: [] };
  }
  const baseRecords = armActionRecords(baselineArm);
  const inRange = state?.range
    ? r => r.tick === null || (r.tick >= state.range[0] && r.tick <= state.range[1])
    : () => true;
  const deltas = [];
  const itvArms = detail.arms.filter(a => a.mode === "intervention");
  for (const itv of itvArms) {
    const itvRecords = armActionRecords(itv);
    // 按 actionInfo.key 分组；双方身份并集。
    const keys = [...new Set([
      ...baseRecords.map(r => r.actionInfo.key),
      ...itvRecords.map(r => r.actionInfo.key),
    ])].sort();
    for (const key of keys) {
      const baseW = baseRecords.filter(r => r.actionInfo.key === key && inRange(r));
      const curW = itvRecords.filter(r => r.actionInfo.key === key && inRange(r));
      if (baseW.length === 0 && curW.length === 0) continue;
      // 第一步：一对一消去完全相同的记录（tick/status/参数全同，
      // 双方命令参数都非 null；历史参数 null 不在此消去）。
      const used = new Array(baseW.length).fill(false);
      const remCur = [];
      for (const c of curW) {
        const hit = baseW.findIndex((b, i) =>
          !used[i] && actionCommandParams(c) !== null && actionCommandParams(b) !== null
          && !actionsDiffer(c, b));
        if (hit >= 0) used[hit] = true;
        else remCur.push(c);
      }
      const remBase = baseW.filter((_, i) => !used[i]);
      // 第二步：余下双方按 tick 排序逐对配对（每条最多配对一次）。
      remBase.sort(byTickThenId);
      remCur.sort(byTickThenId);
      const n = Math.max(remBase.length, remCur.length);
      for (let i = 0; i < n; i++) {
        const l = remBase[i];
        const r = remCur[i];
        if (l && r) {
          // 同 tick/status 但一方参数 null（如历史记录）：不可比，不冒充相同。
          const paramsNull = actionCommandParams(l) === null || actionCommandParams(r) === null;
          const kind = !actionsDiffer(l, r) && paramsNull ? "incomparable" : "changed";
          deltas.push({
            arm: itv, key, label: actionLabel(r), kind,
            left: l, right: r, fieldDiffs: pairedFieldDiffs(l, r),
          });
        } else if (l) {
          // 基线多余：分支窗口内无对应记录。
          deltas.push({ arm: itv, key, label: actionLabel(l), kind: "missing", left: l, right: null, fieldDiffs: [] });
        } else {
          deltas.push({ arm: itv, key, label: actionLabel(r), kind: "added", left: null, right: r, fieldDiffs: [] });
        }
      }
    }
  }
  return { baseline: baselineArm, baselineNote: null, deltas };
}

// ── 3. 实测事件结果差异（跨臂身份：family|type|bindings）─────────
// 不按散列 event_id 建身份；outcomes 通过同臂 occurrence.event_id 取 bindings。
// 比较已归一 occurrence 实际 tick、event_phases 的 phase_kind/tick、
// outcome 的 status/lifecycle_status/terminal_tick/window_censor_tick。
// generatedIDs / supporting_truth_ids 等固有 provenance 一律忽略。
// 多重集合保存重复事件，不覆盖。

// 跨臂事件身份：identityOf(arm, raw) 返回 {id, label}。
// arm.family 为 L1..L6 或 X：
// - X（业务谓词事件）：必需 predicate_id / kind / ontology_event / bindings，
//   key 为这四个字段的 stableValue 对象。
// - L4（生成事件）：必需 event_family_id / bindings；event_type_id 允许缺失，
//   用显式标识占位，不猜值。
// - 其他家族：必需 family（event_family_id）/ type（event_type_id）/ bindings，
//   缺失直接 throw。
function identityOf(arm, raw) {
  const family = arm.family;
  if (family === "X") {
    for (const f of ["predicate_id", "kind", "ontology_event"]) {
      if (raw[f] === undefined || raw[f] === null || raw[f] === "") {
        throw new Error(`X 臂事件行缺少 ${f}，无法建立跨臂身份`);
      }
    }
    if (raw.bindings === undefined || raw.bindings === null) {
      throw new Error("X 臂事件行缺少 bindings，无法建立跨臂身份");
    }
    const key = { predicate_id: raw.predicate_id, kind: raw.kind, ontology_event: raw.ontology_event, bindings: raw.bindings };
    const id = `X|${stableValue(key)}`;
    const label = `业务谓词 ${raw.predicate_id} · kind=${raw.kind} · ontology_event=${raw.ontology_event} · ${bindingsText(raw.bindings)}`;
    return { id, label };
  }
  if (family === "L4") {
    if (typeof raw.event_family_id !== "string" || !raw.event_family_id) {
      throw new Error("L4 事件行缺少 event_family_id，无法建立跨臂身份");
    }
    if (raw.bindings === undefined || raw.bindings === null) {
      throw new Error("L4 事件行缺少 bindings，无法建立跨臂身份");
    }
    const type = Object.prototype.hasOwnProperty.call(raw, "event_type_id")
      && raw.event_type_id !== undefined && raw.event_type_id !== null
      ? raw.event_type_id : "(event_type_id 缺失)";
    const id = `L4|${raw.event_family_id}|${type}|${stableValue(raw.bindings)}`;
    const label = `L4 · ${raw.event_family_id} · ${type} · ${bindingsText(raw.bindings)}`;
    return { id, label };
  }
  if (typeof raw.event_family_id !== "string" || !raw.event_family_id
    || typeof raw.event_type_id !== "string" || !raw.event_type_id
    || raw.bindings === undefined || raw.bindings === null) {
    throw new Error(`家族 ${String(family)} 事件行缺少 event_family_id / event_type_id / bindings，无法建立跨臂身份`);
  }
  const id = `${family}|${raw.event_family_id}|${raw.event_type_id}|${stableValue(raw.bindings)}`;
  const label = `${family} · ${raw.event_family_id} · ${raw.event_type_id} · ${bindingsText(raw.bindings)}`;
  return { id, label };
}

function bindingsText(bindings) {
  return bindings && typeof bindings === "object"
    ? Object.entries(bindings).map(([k, v]) => `${k}=${v}`).join(",") : String(bindings);
}

const asIntTick = (tick, evidence) => {
  if (!Number.isInteger(tick)) {
    throw new Error(`occurrence tick 不是归一整数：${JSON.stringify(tick)}`);
  }
  return tick;
};

// 每臂事件身份索引。无法建立身份的行显式记入 incommensurable。
export function eventIndex(arm) {
  const byId = new Map();
  const incommensurable = [];
  const idOfOcc = new Map(); // occurrence 行 → 已建立的身份
  for (const occ of arm.objective.occurrences) {
    const raw = occ.evidence.raw;
    let identity;
    try {
      identity = identityOf(arm, raw);
    } catch (err) {
      incommensurable.push({ arm, evidence: occ.evidence, reason: err.message });
      continue;
    }
    if (!byId.has(identity.id)) {
      byId.set(identity.id, {
        label: identity.label,
        occurrences: [],
        outcomes: [],
      });
    }
    idOfOcc.set(occ, identity);
    byId.get(identity.id).occurrences.push({
      eventId: raw.event_id ?? null,
      tick: asIntTick(occ.tick, occ.evidence),
      phases: (Array.isArray(raw.event_phases) ? raw.event_phases : [])
        .map(p => ({ phase_kind: p.phase_kind ?? null, tick: p.tick ?? null })),
      evidence: occ.evidence,
    });
  }
  // outcome 只通过同臂 occurrence.event_id 找到对应 occurrence，
  // 再用该 occurrence 的身份归入同一 byId 条目；找不到保持 incommensurable，
  // 不从 outcome 行猜 bindings。
  for (const out of arm.objective.outcomes) {
    const raw = out.evidence.raw;
    const eventId = raw.event_id;
    const occ = typeof eventId === "string"
      ? arm.objective.occurrences.find(o => o.evidence.raw.event_id === eventId)
      : undefined;
    const identity = occ ? idOfOcc.get(occ) : undefined;
    if (!identity) {
      incommensurable.push({
        arm, evidence: out.evidence,
        reason: `outcome ${eventId ?? "?"} 在本臂找不到同 event_id 且身份已建立的 occurrence，身份不可比`,
      });
      continue;
    }
    if (!byId.has(identity.id)) byId.set(identity.id, { label: identity.label, occurrences: [], outcomes: [] });
    byId.get(identity.id).outcomes.push({
      eventId: eventId ?? null,
      status: Object.prototype.hasOwnProperty.call(raw, "status") ? raw.status : ABSENT,
      lifecycle_status: Object.prototype.hasOwnProperty.call(raw, "lifecycle_status") ? raw.lifecycle_status : ABSENT,
      terminal_tick: Object.prototype.hasOwnProperty.call(raw, "terminal_tick") ? raw.terminal_tick : ABSENT,
      window_censor_tick: Object.prototype.hasOwnProperty.call(raw, "window_censor_tick") ? raw.window_censor_tick : ABSENT,
      evidence: out.evidence,
    });
  }
  return { byId, incommensurable };
}

// whole 记录：{occ, outcomes}。occ 通过 eventId 与 meta.outcomes 中
// eventId 相等的记录连接（保持重复计数，不按 eventId 散列去重）。
function buildWholes(meta) {
  const used = new Array(meta.outcomes.length).fill(false);
  const wholes = [];
  for (const occ of meta.occurrences) {
    const outs = [];
    for (let i = 0; i < meta.outcomes.length; i++) {
      if (!used[i] && meta.outcomes[i].eventId === occ.eventId) {
        outs.push(meta.outcomes[i]);
        used[i] = true;
      }
    }
    wholes.push({ occ, outcomes: outs });
  }
  return wholes;
}

// 签名：occ.tick + 排序后的 phase_kind/tick 数组 + 排序后的 outcomes 四字段
// （status/lifecycle_status/terminal_tick/window_censor_tick，先 wrap 再序列化）。
function wholeSignature(whole) {
  const phases = [...whole.occ.phases].sort((a, b) =>
    String(a.phase_kind).localeCompare(String(b.phase_kind)) || (a.tick - b.tick));
  const outs = [...whole.outcomes].map(o => ({
    status: wrap(o.status),
    lifecycle_status: wrap(o.lifecycle_status),
    terminal_tick: wrap(o.terminal_tick),
    window_censor_tick: wrap(o.window_censor_tick),
  })).sort((a, b) =>
    stableValue(a.status).localeCompare(stableValue(b.status))
    || stableValue(a.lifecycle_status).localeCompare(stableValue(b.lifecycle_status))
    || stableValue(a.terminal_tick).localeCompare(stableValue(b.terminal_tick))
    || stableValue(a.window_censor_tick).localeCompare(stableValue(b.window_censor_tick)));
  return stableValue({
    tick: whole.occ.tick,
    phases: phases.map(p => ({ phase_kind: p.phase_kind, tick: p.tick })),
    outcomes: outs.map(o => ({
      status: stableValue(o.status),
      lifecycle_status: stableValue(o.lifecycle_status),
      terminal_tick: stableValue(o.terminal_tick),
      window_censor_tick: stableValue(o.window_censor_tick),
    })),
  });
}

// 内部 outcome 配对：先消去签名相同的，余下按可比签名排序逐对配对。
function outcomeFieldDiffs(baseOuts, curOuts) {
  const diffs = [];
  const fields = ["status", "lifecycle_status", "terminal_tick", "window_censor_tick"];
  const sig = o => stableValue({
    status: wrap(o.status),
    lifecycle_status: wrap(o.lifecycle_status),
    terminal_tick: wrap(o.terminal_tick),
    window_censor_tick: wrap(o.window_censor_tick),
  });
  const used = new Array(curOuts.length).fill(false);
  const remBase = [];
  for (const b of baseOuts) {
    const hit = curOuts.findIndex((c, i) => !used[i] && sig(b) === sig(c));
    if (hit >= 0) used[hit] = true;
    else remBase.push(b);
  }
  const remCur = curOuts.filter((_, i) => !used[i]);
  remBase.sort((a, b) => sig(a).localeCompare(sig(b)));
  remCur.sort((a, b) => sig(a).localeCompare(sig(b)));
  const n = Math.max(remBase.length, remCur.length);
  for (let i = 0; i < n; i++) {
    const b = remBase[i];
    const c = remCur[i];
    for (const f of fields) {
      const bv = b ? b[f] : ABSENT;
      const cv = c ? c[f] : ABSENT;
      if (stableValue(bv) === stableValue(cv)) continue;
      diffs.push({
        field: f,
        baseline: wrap(bv), current: wrap(cv),
        baselineEvidence: b ? b.evidence : null,
        currentEvidence: c ? c.evidence : null,
      });
    }
  }
  return diffs;
}

// 返回 { baseline, baselineNote, deltas, incommensurable }。
// delta: { kind: "added"|"removed"|"changed", identity, label,
//          baseline: {occurrences, outcomes} | null,
//          current: {occurrences, outcomes} | null,
//          fieldDiffs }
export function resultDeltas(detail) {
  const { baseline: baselineArm, baselineNote } = factualBaseline(detail);
  if (!baselineArm) {
    return { baseline: null, baselineNote, deltas: [], incommensurable: [] };
  }
  const baseIdx = eventIndex(baselineArm);
  const deltas = [];
  const incommensurable = [...baseIdx.incommensurable];
  for (const itv of detail.arms.filter(a => a.mode === "intervention")) {
    const itvIdx = eventIndex(itv);
    incommensurable.push(...itvIdx.incommensurable);
    const ids = [...new Set([...baseIdx.byId.keys(), ...itvIdx.byId.keys()])].sort();
    for (const id of ids) {
      const base = baseIdx.byId.get(id);
      const cur = itvIdx.byId.get(id);
      const label = (base ?? cur).label;
      const baseWholes = base ? buildWholes(base) : [];
      const curWholes = cur ? buildWholes(cur) : [];
      // 一对一消去签名相同的 whole（保留重复计数）。
      const used = new Array(curWholes.length).fill(false);
      const remBase = [];
      for (const b of baseWholes) {
        const sigB = wholeSignature(b);
        const hit = curWholes.findIndex((c, i) => !used[i] && wholeSignature(c) === sigB);
        if (hit >= 0) used[hit] = true;
        else remBase.push(b);
      }
      const remCur = curWholes.filter((_, i) => !used[i]);
      if (remBase.length === 0 && remCur.length === 0) continue;
      const summary = wholes => ({
        occurrences: wholes.map(w => w.occ),
        outcomes: wholes.flatMap(w => w.outcomes),
      });
      // 余下双方按 occ.tick 排序逐对 changed；多余 removed/added。
      remBase.sort((a, b) => a.occ.tick - b.occ.tick);
      remCur.sort((a, b) => a.occ.tick - b.occ.tick);
      const n = Math.max(remBase.length, remCur.length);
      for (let i = 0; i < n; i++) {
        const b = remBase[i];
        const c = remCur[i];
        if (b && c) {
          const fieldDiffs = [];
          // tick
          if (b.occ.tick !== c.occ.tick) {
            fieldDiffs.push({
              field: "tick",
              baseline: present(b.occ.tick), current: present(c.occ.tick),
              baselineEvidence: b.occ.evidence, currentEvidence: c.occ.evidence,
            });
          }
          // 规范排序 phases 逐项比较
          const sortPhases = ps => [...ps].sort((a, b2) =>
            String(a.phase_kind).localeCompare(String(b2.phase_kind)) || (a.tick - b2.tick));
          const bp = sortPhases(b.occ.phases);
          const cp = sortPhases(c.occ.phases);
          const pn = Math.max(bp.length, cp.length);
          for (let j = 0; j < pn; j++) {
            const p1 = bp[j];
            const p2 = cp[j];
            const v1 = p1 ? stableValue({ phase_kind: p1.phase_kind, tick: p1.tick }) : undefined;
            const v2 = p2 ? stableValue({ phase_kind: p2.phase_kind, tick: p2.tick }) : undefined;
            if (v1 === v2) continue;
            fieldDiffs.push({
              field: `event_phases[${j}]`,
              baseline: p1 ? present({ phase_kind: p1.phase_kind, tick: p1.tick }) : absent(),
              current: p2 ? present({ phase_kind: p2.phase_kind, tick: p2.tick }) : absent(),
              baselineEvidence: b.occ.evidence, currentEvidence: c.occ.evidence,
            });
          }
          // outcomes：先消相同，再按可比签名排序配对，四字段逐项
          fieldDiffs.push(...outcomeFieldDiffs(b.outcomes, c.outcomes));
          deltas.push({
            arm: itv, identity: id, label, kind: "changed",
            baseline: summary([b]), current: summary([c]),
            fieldDiffs,
          });
        } else if (b) {
          deltas.push({
            arm: itv, identity: id, label, kind: "removed",
            baseline: summary([b]), current: null,
            fieldDiffs: [],
          });
        } else {
          deltas.push({
            arm: itv, identity: id, label, kind: "added",
            baseline: null, current: summary([c]),
            fieldDiffs: [],
          });
        }
      }
    }
  }
  return { baseline: baselineArm, baselineNote: null, deltas, incommensurable };
}

// 汇总：每个干预分支一组 delta 数据（纯数据，供 UI 与测试共用）。
export function buildArmDelta(detail) {
  const state = null; // 全窗口：动作 delta 不受分支筛选/时间筛选影响
  const actions = actionDeltas(detail, state);
  const results = resultDeltas(detail);
  const groups = [];
  const { baseline } = factualBaseline(detail);
  for (const itv of detail.arms.filter(a => a.mode === "intervention")) {
    const actionList = actions.deltas.filter(d => d.arm.arm_key === itv.arm_key);
    const resultList = results.deltas.filter(d => d.arm.arm_key === itv.arm_key);
    const configs = baseline ? configDeltas(baseline, itv) : [];
    groups.push({
      arm: itv,
      baseline,
      baselineNote: actions.baselineNote,
      actionDeltas: actionList,
      configDeltas: configs,
      resultDeltas: resultList,
      incommensurable: results.incommensurable.filter(i =>
        baseline
          ? (i.arm.arm_key === itv.arm_key || i.arm.arm_key === baseline.arm_key)
          : i.arm.arm_key === itv.arm_key),
      counts: {
        actions: actionList.length,
        config: configs.length,
        results: resultList.length,
      },
    });
  }
  return { groups, incommensurable: results.incommensurable };
}

