// ARM 审查共享显示辅助（#/arms 与 #/interventions 共用）。
// 只做呈现与证据定位，不做任何数据推断或兜底。

import { actionRecordFromRow, actionRecordsFromRow } from "./arm-actions.js";

export const MODE_LABEL = Object.freeze({
  factual_observation: "事实观察",
  intervention: "干预",
});

export const RECORD_KIND = Object.freeze({
  action: { label: "动作", color: "#f2b84b" },
  transition: { label: "谓词变化", color: "#35e4f2" },
  occurrence: { label: "事件发生", color: "#62e6a7" },
  outcome: { label: "终态/右删失", color: "#aa92ff" },
});

export function el(tag, cls = "", text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = String(text);
  return node;
}

export function panel(title, subtitle) {
  const node = el("section", "panel");
  const head = el("div", "panel-head");
  head.append(el("h2", "panel-header", title));
  if (subtitle) head.append(el("span", "panel-note", subtitle));
  node.append(head);
  return node;
}

export function evidenceRef(evidence) {
  if (!evidence || !evidence.path) return "无证据";
  const loc = evidence.line === null || evidence.line === undefined
    ? evidence.path
    : `${evidence.path}:${evidence.line}`;
  return el("code", "arm-loc mono", loc);
}

// 折叠的原始 JSON / 文本块
export function rawDetails(label, data) {
  const node = el("details", "arm-source");
  node.append(el("summary", "", label));
  node.append(el("pre", "arm-json", typeof data === "string" ? data : JSON.stringify(data, null, 2)));
  return node;
}

// 关系类型 → 中文呈现。
// 1) X 族 type === 'script_dependency' 显示“脚本调度依赖”；
// 2) 原始行有 causal_claim 时按 true/false 区分；
// 3) 字段缺失显示“未声明因果”。
export function relationLabel(type, raw) {
  if (type === "script_dependency") return "脚本调度依赖";
  if (raw && Object.prototype.hasOwnProperty.call(raw, "causal_claim")) {
    return raw.causal_claim === true ? "声明因果关系" : "证据关联";
  }
  return "未声明因果";
}

// 关系端点定位。返回 {kind, evidence?, tick?, occurrence?}。
// 1) materialized：端点在本臂窗口内物化（chain 节点或 objective 行命中）。
// 2) outside_window：端点不能在本窗物化，但能在本臂 occurrence 的
//    evidence.raw.event_phases 里找到 transition_id 相等的 phase，且该
//    phase.tick 确实在本臂 window.start_tick 之前（窗前引用）。
// 3) unresolved：其余一律未解析，不猜。
export function resolveEndpoint(endpoint, arm) {
  if (!endpoint || typeof endpoint !== "string") return { kind: "invalid" };
  for (const row of arm.chain.nodes) {
    const raw = row.evidence.raw;
    if (raw.node_id === endpoint) {
      return { kind: "materialized", evidence: row.evidence };
    }
  }
  for (const row of arm.objective.transitions) {
    const raw = row.evidence.raw;
    if (raw.transition_id === endpoint) return { kind: "materialized", evidence: row.evidence };
  }
  for (const row of arm.objective.occurrences) {
    const raw = row.evidence.raw;
    if (raw.event_id === endpoint) return { kind: "materialized", evidence: row.evidence };
  }
  for (const row of arm.objective.outcomes) {
    const raw = row.evidence.raw;
    if (raw.outcome_id === endpoint) return { kind: "materialized", evidence: row.evidence };
  }
  const start = Number.isFinite(arm.window?.start_tick) ? arm.window.start_tick : null;
  for (const occ of arm.objective.occurrences) {
    const raw = occ.evidence.raw;
    const phases = Array.isArray(raw.event_phases) ? raw.event_phases : [];
    for (const phase of phases) {
      if (phase && phase.transition_id === endpoint) {
        const tick = typeof phase.tick === "number" ? phase.tick : null;
        if (start === null || tick === null || tick >= start) return { kind: "unresolved" };
        return { kind: "outside_window", tick, occurrence: occ.evidence };
      }
    }
  }
  return { kind: "unresolved" };
}

export function isOutsideWindow(endpoint, arm) {
  const r = resolveEndpoint(endpoint, arm);
  return r.kind === "outside_window";
}

// 完整轨道键：谓词 id + 原始 bindings 序列化。绑定不同即为不同轨道。
export function predicateTrackKey(predicateKey) {
  if (!predicateKey) return "(无谓词)";
  return predicateKey;
}

// 显示用短标签：只保留谓词 id；完整键（含 bindings）放 title，绝不截断真实键。
export function shortPredicateLabel(predicateKey) {
  if (!predicateKey) return "(无谓词)";
  const idx = predicateKey.indexOf("|");
  return idx === -1 ? predicateKey : predicateKey.slice(0, idx);
}

export function shortBinding(predicateKey) {
  if (!predicateKey) return "";
  const idx = predicateKey.indexOf("|");
  if (idx === -1) return "";
  try {
    return JSON.stringify(JSON.parse(predicateKey.slice(idx + 1)));
  } catch {
    return predicateKey.slice(idx + 1);
  }
}

// 一个臂的统一记录流：动作/谓词变化/发生/终态，全部带真实 tick 与证据。
// 顶层行直接用 row.evidence；X 过渡的字段形状是 before/after（X 无 from_value/to_value）。
// 动作行通过 actionRecordsFromRow 展开（含 historical_event_realization 的
// action_realizations），每条带 actionInfo；不再产生“未命名动作”。
export function armRecords(arm) {
  const records = [];
  for (const row of arm.raw_actions) {
    const raw = row.evidence.raw;
    if (raw.origin === "historical_event_realization") {
      for (const rec of actionRecordsFromRow(row)) records.push(rec);
      continue;
    }
    records.push(actionRecordFromRow(row));
  }
  for (const row of arm.objective.transitions) {
    const raw = row.evidence.raw;
    const before = raw.before ?? raw.from_value;
    const after = raw.after ?? raw.to_value;
    records.push({
      kind: "transition",
      tick: row.tick,
      title: row.predicate_key || raw.predicate_id || "(无谓词)",
      detail: `${before ?? "?"} → ${after ?? "?"}`,
      status: after ?? null,
      evidence: row.evidence,
      raw: row,
    });
  }
  for (const row of arm.objective.occurrences) {
    const raw = row.evidence.raw;
    records.push({
      kind: "occurrence",
      tick: row.tick,
      title: raw.event_id || "(无事件 id)",
      detail: raw.event_type_id || raw.event_family_id || "",
      status: null,
      evidence: row.evidence,
      raw: row,
    });
  }
  for (const row of arm.objective.outcomes) {
    const raw = row.evidence.raw;
    const censored = raw.status === "right_censored";
    records.push({
      kind: "outcome",
      tick: raw.terminal_tick ?? null,
      title: raw.event_id || "(无事件 id)",
      detail: censored ? "右删失：窗口结束仍未终止"
        : raw.status === "pending" ? "终态待定" : `终态 ${raw.status}`,
      status: raw.status,
      censored,
      evidence: row.evidence,
      raw: row,
    });
  }
  return records.sort((a, b) => (a.tick ?? -Infinity) - (b.tick ?? -Infinity));
}

// 解析 hash 深链接参数，例如 #/arms?profile=..&seed=..&group=..
export function hashParams() {
  const query = location.hash.split("?")[1] || "";
  return Object.fromEntries(new URLSearchParams(query));
}

export function setHashParams(base, params) {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") query.set(key, value);
  }
  const suffix = query.toString();
  location.hash = suffix ? `#/${base}?${suffix}` : `#/${base}`;
}

// 写入 hash 但不触发 hashchange（用 history.replaceState）。
export function replaceHashParams(base, params) {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") query.set(key, value);
  }
  const suffix = query.toString();
  const hash = suffix ? `#/${base}?${suffix}` : `#/${base}`;
  history.replaceState(null, "", `${location.pathname}${location.search}${hash}`);
}

// ── 两页共用的组选择辅助 ────────────────────────────────────────────
// 模型：家族 → profile → seed → group，每级都是具体值（无“全部”占位）。
// 家族必须从所选 profile 所属 family 还原。
export function buildSelectionModel(groups) {
  const profiles = new Map(); // profile -> {family, seeds:Set, groups:[{seed,group}]}
  for (const g of groups) {
    if (!profiles.has(g.profile)) {
      profiles.set(g.profile, { family: g.family, seeds: new Set(), groups: new Map() });
    }
    const entry = profiles.get(g.profile);
    entry.seeds.add(g.seed);
    if (!entry.groups.has(g.seed)) entry.groups.set(g.seed, new Set());
    entry.groups.get(g.seed).add(g.group);
  }
  return profiles;
}

export function exactGroup(groups, { profile, seed, group }) {
  if (!profile || !seed || !group) return null;
  return groups.find(g => g.profile === profile && g.seed === seed && g.group === group) || null;
}

// 填充一个 select；current 必须在 values 中（调用方保证），values 非空。
export function fillSelect(select, values, current) {
  select.replaceChildren(...values.map(v => {
    const o = el("option", "", v);
    o.value = v;
    return o;
  }));
  select.value = values.includes(current) ? current : values[0];
}

// 通用三级选择器构造。opts:
// - groups: catalog.groups
// - initial: {profile, seed, group} 已验证的当前选择
// - onChange(next): 下一级变化回调（next = {profile, seed, group}），由调用方
//   负责写 hash 与加载；本函数只维护选择器联动并还原家族显示。
// 返回 { row, familySelect, profileSelect, seedSelect, groupSelect, current() }。
export function buildGroupSelector(groups, initial, onChange) {
  const profiles = buildSelectionModel(groups);
  const familySelect = el("select"); familySelect.setAttribute("aria-label", "家族");
  const profileSelect = el("select"); profileSelect.setAttribute("aria-label", "profile");
  const seedSelect = el("select"); seedSelect.setAttribute("aria-label", "seed");
  const groupSelect = el("select"); groupSelect.setAttribute("aria-label", "组");

  const families = [...new Set(groups.map(g => g.family))].sort();
  const profilesOfFamily = fam => [...profiles.entries()]
    .filter(([, v]) => v.family === fam).map(([p]) => p).sort();
  const seedsOf = prof => [...profiles.get(prof).seeds].sort();
  const groupsOf = (prof, seed) => [...profiles.get(prof).groups.get(seed)].sort();

  let current = { ...initial };

  const syncFamily = () => {
    fillSelect(familySelect, families, profiles.get(current.profile).family);
  };
  const syncAll = () => {
    syncFamily();
    fillSelect(profileSelect, profilesOfFamily(familySelect.value), current.profile);
    fillSelect(seedSelect, seedsOf(current.profile), current.seed);
    fillSelect(groupSelect, groupsOf(current.profile, current.seed), current.group);
  };

  familySelect.addEventListener("change", () => {
    const profile = profilesOfFamily(familySelect.value)[0];
    if (!profile || profile === current.profile) { syncAll(); return; }
    current = { profile, seed: seedsOf(profile)[0], group: groupsOf(profile, seedsOf(profile)[0])[0] };
    syncAll();
    onChange({ ...current });
  });
  profileSelect.addEventListener("change", () => {
    const profile = profileSelect.value;
    if (profile === current.profile) return;
    current = { profile, seed: seedsOf(profile)[0], group: groupsOf(profile, seedsOf(profile)[0])[0] };
    syncAll();
    onChange({ ...current });
  });
  seedSelect.addEventListener("change", () => {
    const seed = seedSelect.value;
    if (seed === current.seed) return;
    current = { ...current, seed, group: groupsOf(current.profile, seed)[0] };
    syncAll();
    onChange({ ...current });
  });
  groupSelect.addEventListener("change", () => {
    if (groupSelect.value === current.group) return;
    current = { ...current, group: groupSelect.value };
    onChange({ ...current });
  });

  syncAll();
  const row = el("div", "arm-selector-row");
  [[familySelect, "家族"], [profileSelect, "profile"], [seedSelect, "seed"], [groupSelect, "组"]]
    .forEach(([select, label]) => {
      const wrap = el("label", "arm-selector__field");
      wrap.append(el("span", "arm-selector__label", label), select);
      row.append(wrap);
    });
  return {
    row,
    current: () => ({ ...current }),
    // 深链接验证后由调用方强制重置（例如默认加载第一组时同步选择器显示）
    setCurrent(next) {
      current = { ...next };
      syncAll();
    },
  };
}
