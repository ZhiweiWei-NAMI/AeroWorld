// ARM 差异审查 DOM：每个 mode=intervention 独立卡片，只呈现与事实观察基线的差异。
// 卡片主体是紧凑 3 列对照表（字段 / 基线 / 当前），只列真实变化的字段；
// 双方 path:line + 原始 JSON 放默认关闭的 details。纯数据在 arm-deltas.js。

import {
  MODE_LABEL, el, panel, evidenceRef, rawDetails,
} from "./arm-common.js";
import { statusLabel } from "./arm-actions.js";
import {
  buildArmDelta, NO_FACTUAL_BASELINE_TEXT, ABSENT, wrap,
} from "./arm-deltas.js";

const NULL_TEXT = "null";
const NO_CHANGE_ACTIONS = "脚本动作未改变";
const NO_CHANGE_RESULTS = "所比事件时刻与终态未改变";
const UNCOMPARABLE_TEXT = "参数不可比（历史执行记录未提供命令参数）";
const MISSING_SIDE_TEXT = "窗口内无记录";
const FIELD_ABSENT_TEXT = "字段未提供";
// 只有真实 status 字符串是这两个省略值时才显示省略文案；
// wrapped.absent 一律是“字段未提供”，与干预省略无关。
const OMITTED_STATUSES = new Set([
  "omitted_by_explicit_intervention",
  "omitted_by_configuration",
]);

function modeLabel(arm) { return MODE_LABEL[arm.mode] || arm.mode || "未知模式"; }

// 值单元格文本。absent 一律“字段未提供”；仅真实 status 字符串为干预/配置省略
// 时经 statusLabel 显示省略文案，其余 status 保留原值；null 保持 null。
function cellText(wrapped, field) {
  if (wrapped.absent) return FIELD_ABSENT_TEXT;
  const v = wrapped.value;
  if (v === null) return NULL_TEXT;
  if (field === "status" && OMITTED_STATUSES.has(v)) return statusLabel(v);
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

// ── 通用紧凑 3 列表：字段 / 基线（事实观察） / 当前（干预）──────────
// rows: [{field, baseline:{absent,value}|null, current:同左, baselineEvidence,
//         currentEvidence}]。只列调用方给出的真实差异行。
function compactTable(rows) {
  const table = el("table", "arm-delta-table");
  const thead = el("thead");
  const hrow = el("tr");
  hrow.append(
    el("th", "", "字段"),
    el("th", "", "基线（事实观察）"),
    el("th", "", "当前（干预）"),
  );
  thead.append(hrow);
  table.append(thead);
  const tbody = el("tbody");
  for (const r of rows) {
    const tr = el("tr");
    tr.append(el("td", "arm-delta-table__field mono", r.field));
    tr.append(el("td", "mono", cellText(r.baseline, r.field)));
    tr.append(el("td", "mono", cellText(r.current, r.field)));
    tbody.append(tr);
  }
  table.append(tbody);
  return table;
}

// 一侧证据折叠：summary 只放基线/当前证据短标签，path:line 与原始 JSON
// 都在 details 内部（长路径默认隐藏）。
function evidenceDetails(label, evidence, rawLabel) {
  if (!evidence) return null;
  const details = el("details", "arm-delta-evidence");
  details.append(el("summary", "", label));
  details.append(evidenceRef(evidence));
  details.append(rawDetails(rawLabel, evidence.raw ?? evidence.content ?? evidence.text));
  return details;
}

// ── 动作差异区 ──────────────────────────────────────────────────
// d: {arm, key, label, kind, left, right, fieldDiffs}；baseline 为事实观察臂，
// missing 项的 onPick 传 left + delta.baseline（不拿基线记录查干预臂关系）。
function actionDeltaItem(d, baseline, onPick) {
  const item = el("div", "arm-delta-item");
  const head = el("div", "arm-delta-item__head");
  const kindText = {
    changed: "动作改变",
    incomparable: "参数不可比",
    missing: "分支缺失",
    added: "分支新增",
  }[d.kind] || d.kind;
  head.append(el("span", "arm-delta-item__kind", kindText));
  head.append(el("span", "arm-delta-item__label", d.label));
  item.append(head);

  if (d.kind === "incomparable") {
    // 中性提示，不冒充相同，也不计为已确认的动作变化。
    item.append(el("p", "arm-delta-incomparable", UNCOMPARABLE_TEXT));
  } else if (d.kind === "changed") {
    if (d.fieldDiffs.length) {
      item.append(compactTable(d.fieldDiffs));
    } else {
      item.append(el("p", "arm-delta-incomparable",
        "已配对成对但未建立可比参数差异（双方命令参数不可得），差异范围未知。"));
    }
  } else {
    // missing / added：完整记录变化，可列实际 tick/status/已有 paramsLines；
    // 缺的一侧明确窗口内无记录。
    const left = d.kind === "added" ? null : d.left;
    const right = d.kind === "missing" ? null : d.right;
    const row = el("table", "arm-delta-table arm-delta-table--record");
    const tbody = el("tbody");
    const addRow = (field, lText, rText) => {
      const tr = el("tr");
      tr.append(el("td", "arm-delta-table__field mono", field));
      tr.append(el("td", "mono", lText));
      tr.append(el("td", "mono", rText));
      tbody.append(tr);
    };
    const side = rec => {
      if (!rec) return MISSING_SIDE_TEXT;
      const bits = [];
      if (rec.actionInfo) {
        bits.push(`t=${rec.tick ?? "null"}`, rec.actionInfo.statusText);
        if (rec.actionInfo.paramsLines.length) {
          bits.push(rec.actionInfo.paramsLines.join("；"));
        }
      } else {
        bits.push(`t=${rec.tick ?? "null"}`, `status ${rec.status ?? "null"}`);
      }
      return bits.join(" · ");
    };
    addRow(d.kind === "added" ? "记录（分支新增）" : "记录（分支缺失）",
      side(left), side(right));
    row.append(tbody);
    item.append(row);
  }

  // 双方 path:line + rawJSON，默认关闭。
  const left = d.kind === "added" ? null : d.left;
  const right = d.kind === "missing" ? null : d.right;
  const evBox = el("div", "arm-delta-evidence-row");
  for (const [sideName, rec] of [["基线", left], ["当前", right]]) {
    if (!rec?.evidence) continue;
    const det = evidenceDetails(`${sideName}证据`, rec.evidence, `${sideName}记录原始 JSON`);
    if (det) evBox.append(det);
  }
  if (evBox.childNodes.length) item.append(evBox);

  // 查看完整记录：right 传 d.arm；missing 仅 left 时传本卡的 delta.baseline。
  const pickRecords = [];
  if (right) pickRecords.push([right, d.arm]);
  if (left && !right) pickRecords.push([left, baseline]);
  for (const [record, arm] of pickRecords) {
    if (!record || !onPick) continue;
    const btn = el("button", "arm-delta-open", "查看完整记录");
    btn.type = "button";
    btn.addEventListener("click", () => onPick(record, arm));
    item.append(btn);
  }
  return item;
}

// ── 外部干预条件区（配置声明，非执行实测）────────────────────────
// configDeltas 整合为一张 3 列表；双方配置证据只各折叠一次。
function configDeltaSection(configDeltas) {
  const item = el("div", "arm-delta-item");
  item.append(compactTable(configDeltas));
  item.append(el("p", "panel-note", "以上为 configuration 声明值，不是执行实测。"));
  const evBox = el("div", "arm-delta-evidence-row");
  const seen = new Set();
  for (const d of configDeltas) {
    for (const [sideName, ev] of [["基线", d.baselineEvidence], ["当前", d.currentEvidence]]) {
      if (!ev || seen.has(ev)) continue;
      seen.add(ev);
      const det = evidenceDetails(`${sideName}配置证据`, ev, `${sideName}配置原始 JSON`);
      if (det) evBox.append(det);
    }
  }
  if (evBox.childNodes.length) item.append(evBox);
  return item;
}

// ── 实测事件结果区 ──────────────────────────────────────────────
// d: {arm, identity, label, kind, baseline:{occurrences,outcomes}|null,
//     current:同左|null, fieldDiffs}
function resultDeltaItem(d) {
  const item = el("div", "arm-delta-item");
  const head = el("div", "arm-delta-item__head");
  const kindText = {
    added: "事件新增",
    removed: "事件消失",
    changed: "结果变化",
  }[d.kind] || d.kind;
  head.append(el("span", "arm-delta-item__kind", kindText));
  head.append(el("span", "arm-delta-item__label", d.label));
  item.append(head);

  if (d.kind === "changed" && d.fieldDiffs.length) {
    item.append(compactTable(d.fieldDiffs));
  } else {
    // added / removed / 无 fieldDiffs：列两侧实际 occ/outcome 摘要，缺侧明确无记录。
    const summaryRows = occs => occs.map(occ => {
      const phases = (occ.phases || []).map(p =>
        `${p.phase_kind ?? "null"}@${p.tick ?? "null"}`).join("、");
      return `t=${occ.tick ?? "null"}${phases ? ` · ${phases}` : ""}`;
    }).join("；");
    const outcomeRows = outs => outs.map(oc => {
      // status/lifecycle/terminal/censor 全部经 cellText(wrap(...))，字段缺失
      // （ABSENT Symbol）显示“字段未提供”，null 保持 null，绝不直接插值。
      const norm = v => wrap(v === undefined ? ABSENT : v);
      const status = cellText(norm(oc.status), "status");
      const lifecycle = cellText(norm(oc.lifecycle_status), "lifecycle_status");
      const tt = norm(oc.terminal_tick);
      const ttText = tt.absent ? FIELD_ABSENT_TEXT
        : tt.value === null
          ? (oc.status === "right_censored" ? "null（右删失）"
            : oc.status === "pending" ? "null（终态待定）" : "null")
          : cellText(tt, "terminal_tick");
      const cw = norm(oc.window_censor_tick);
      const censor = cw.absent ? ` · censor ${FIELD_ABSENT_TEXT}`
        : cw.value === null ? "" : ` · censor ${cellText(cw, "window_censor_tick")}`;
      return `status ${status} · ${lifecycle} · terminal_tick ${ttText}${censor}`;
    }).join("；");
    const row = el("table", "arm-delta-table arm-delta-table--record");
    const tbody = el("tbody");
    const addRow = (field, lText, rText) => {
      const tr = el("tr");
      tr.append(el("td", "arm-delta-table__field mono", field));
      tr.append(el("td", "mono", lText));
      tr.append(el("td", "mono", rText));
      tbody.append(tr);
    };
    const b = d.baseline;
    const c = d.current;
    addRow("发生", b ? (summaryRows(b.occurrences) || MISSING_SIDE_TEXT) : MISSING_SIDE_TEXT,
      c ? (summaryRows(c.occurrences) || MISSING_SIDE_TEXT) : MISSING_SIDE_TEXT);
    addRow("终态", b ? (outcomeRows(b.outcomes) || MISSING_SIDE_TEXT) : MISSING_SIDE_TEXT,
      c ? (outcomeRows(c.outcomes) || MISSING_SIDE_TEXT) : MISSING_SIDE_TEXT);
    row.append(tbody);
    item.append(row);
  }

  // 双方全部 occurrence/outcome 证据逐条折叠，按 path:line 去重（同一文件行
  // 可能同时出现在 occurrences 与 outcomes），不丢对应证据。
  const evBox = el("div", "arm-delta-evidence-row");
  for (const [sideName, side] of [["基线", d.baseline], ["当前", d.current]]) {
    if (!side) continue;
    const seen = new Set();
    const addEvidence = (label, evidence, rawLabel) => {
      if (!evidence) return;
      const key = `${evidence.path ?? ""}:${evidence.line ?? ""}`;
      if (seen.has(key)) return;
      seen.add(key);
      const det = evidenceDetails(label, evidence, rawLabel);
      if (det) evBox.append(det);
    };
    for (const occ of side.occurrences) {
      addEvidence(`${sideName}发生证据`, occ.evidence, `${sideName}occurrence 原始 JSON`);
    }
    for (const oc of side.outcomes) {
      addEvidence(`${sideName}终态证据`, oc.evidence, `${sideName}outcome 原始 JSON`);
    }
  }
  if (evBox.childNodes.length) item.append(evBox);
  return item;
}

// ── 单个干预分支的默认差异卡片 ───────────────────────────────────
export function buildArmDeltaCard(delta, opts = {}) {
  const { arm, baseline, baselineNote, actionDeltas, configDeltas, resultDeltas, incommensurable } = delta;
  const card = panel(`各 ARM 差异审查 · ${modeLabel(arm)} · ${arm.arm_id}`,
    baseline ? `相对事实观察基线 ${baseline.arm_id}` : null);
  card.classList.add("arm-delta-card");
  card.dataset.armId = arm.arm_id;
  if (baselineNote) {
    card.append(el("p", "callout callout-danger", baselineNote));
    return card;
  }
  // incomparable 动作不是已确认变化，单独计数。
  const confirmedActions = actionDeltas.filter(d => d.kind !== "incomparable").length;
  const incomparableActions = actionDeltas.length - confirmedActions;
  const summaryBits = [
    `动作变化 ${confirmedActions}`,
    incomparableActions ? `（另有参数不可比 ${incomparableActions}）` : "",
    `外部条件变化 ${configDeltas.length}`,
    `结果变化 ${resultDeltas.length}`,
  ].filter(Boolean);
  card.append(el("p", "arm-delta-card__summary mono", summaryBits.join(" · ")));

  // 1) 脚本动作差异
  const secActions = el("section", "arm-delta-section");
  secActions.append(el("h3", "arm-audit__label", "脚本动作差异（左基线 · 右当前）"));
  if (!actionDeltas.length) {
    secActions.append(el("p", "panel-body", NO_CHANGE_ACTIONS));
  } else {
    const changed = actionDeltas.filter(d => d.kind === "changed");
    const full = actionDeltas.filter(d => d.kind === "missing" || d.kind === "added");
    const uncomp = actionDeltas.filter(d => d.kind === "incomparable");
    for (const d of changed) secActions.append(actionDeltaItem(d, baseline, opts.onPick));
    for (const d of full) secActions.append(actionDeltaItem(d, baseline, opts.onPick));
    if (uncomp.length) {
      const note = el("p", "arm-delta-incomparable",
        `另有 ${uncomp.length} 条动作参数不可比，未计为已确认变化：`);
      secActions.append(note);
      for (const d of uncomp) secActions.append(actionDeltaItem(d, baseline, opts.onPick));
    }
  }
  card.append(secActions);

  // 2) 外部干预条件（配置声明）— 一张 3 列表
  const secConfig = el("section", "arm-delta-section");
  secConfig.append(el("h3", "arm-audit__label", "外部干预条件（配置声明）"));
  if (!configDeltas.length) {
    secConfig.append(el("p", "panel-body", "配置声明的干预条件与基线无字段差异。"));
  } else {
    secConfig.append(configDeltaSection(configDeltas));
  }
  card.append(secConfig);

  // 3) 实测事件结果
  const secResults = el("section", "arm-delta-section");
  secResults.append(el("h3", "arm-audit__label", "实测事件结果变化（左基线 · 右当前）"));
  if (!resultDeltas.length) {
    if (incommensurable.length) {
      const callout = el("div", "callout callout-danger");
      callout.append(el("p", "", `本分支有 ${incommensurable.length} 条事件/终态记录无法建立跨臂身份，无法完成比较：`));
      incommensurable.forEach(i => {
        const row = el("p", "panel-note");
        row.append(el("code", "mono", i.reason));
        if (i.evidence) row.append(evidenceRef(i.evidence));
        callout.append(row);
      });
      secResults.append(callout);
    } else {
      secResults.append(el("p", "panel-body", NO_CHANGE_RESULTS));
    }
  } else {
    resultDeltas.forEach(d => secResults.append(resultDeltaItem(d)));
  }
  if (resultDeltas.length && incommensurable.length) {
    const callout = el("div", "callout callout-danger");
    callout.append(el("p", "", `本分支另有 ${incommensurable.length} 条事件/终态记录无法建立跨臂身份，未参与比较：`));
    incommensurable.forEach(i => {
      const row = el("p", "panel-note");
      row.append(el("code", "mono", i.reason));
      if (i.evidence) row.append(evidenceRef(i.evidence));
      callout.append(row);
    });
    secResults.append(callout);
  }
  card.append(secResults);
  return card;
}

// 全部干预分支的差异卡片列表。无干预组写明没有干预分支。
export function buildArmDeltaSection(detail, opts = {}) {
  const box = panel("各 ARM 差异审查", "每个干预分支相对本组唯一事实观察分支 · 只显示差异 · 全窗口");
  const deltas = buildArmDelta(detail);
  if (!deltas.groups.length) {
    box.append(el("p", "panel-body", "本组没有干预分支，无差异可审查。"));
    return box;
  }
  if (!deltas.groups[0].baseline) {
    box.append(el("p", "callout callout-danger", deltas.groups[0].baselineNote
      || NO_FACTUAL_BASELINE_TEXT));
    return box;
  }
  for (const g of deltas.groups) {
    box.append(buildArmDeltaCard(g, opts));
  }
  return box;
}
