// ARM 动作对照表与节点浮层 DOM（依赖 arm-actions 数据与 arm-common 显示辅助）。
// arm-actions.js 保持纯数据；本模块是唯一做表格/浮层 DOM 的地方。

import { MODE_LABEL, el, evidenceRef, rawDetails } from "./arm-common.js";
import {
  NO_RECORD_TEXT, NO_RECORD_FILTERED_TEXT, DIFF_TEXT, factualIdentitySummary,
  buildActionComparison, armActionRecords,
} from "./arm-actions.js";

// ── 分支对照表（DOM）─────────────────────────────────────────────
// opts: { onPick(record, arm), diffOnly, onToggleDiffOnly }
export function buildActionTablePanel(detail, state, opts = {}) {
  const box = el("details", "arm-action-table-panel");
  box.open = true;
  const { baseline, baselineNote, selected, rows } = buildActionComparison(detail, state);
  const summary = el("summary", "", "");
  summary.append(
    el("span", "arm-action-table__title", "动作分支对照"),
    el("span", "panel-note", `分支 ${selected.length} 个 · 动作身份 ${rows.length} 组`
      + (baseline ? ` · 事实观察基线 ${baseline.arm_id}` : "")),
  );
  box.append(summary);

  const tools = el("div", "arm-action-table__tools");
  if (baselineNote) tools.append(el("p", "panel-note", baselineNote));
  const diffLabel = el("label", "arm-branch");
  const diffCheck = el("input");
  diffCheck.type = "checkbox";
  diffCheck.checked = !!state.diffOnly;
  diffCheck.addEventListener("change", () => {
    state.diffOnly = diffCheck.checked;
    if (opts.onToggleDiffOnly) opts.onToggleDiffOnly();
  });
  diffLabel.append(diffCheck, el("span", "", "仅看差异"));
  tools.append(diffLabel);
  box.append(tools);

  if (!selected.length) {
    box.append(el("p", "panel-body",
      state.kinds.has("action") ? "没有选中分支，无法生成动作对照。" : "记录类别未勾选“动作”，无动作对照。"));
    return box;
  }

  const visibleRows = state.diffOnly ? rows.filter(r => r.differsAny) : rows;
  if (!visibleRows.length) {
    box.append(el("p", "panel-body",
      state.diffOnly ? "当前筛选下没有与事实观察基线不同的动作身份。" : "当前筛选条件下没有动作记录。"));
    return box;
  }

  const scroll = el("div", "arm-action-scroll");
  const table = el("table", "arm-action-table");
  // 有界列宽：首列 + 每臂固定宽，配合 table-layout:fixed 折行而不是盖邻列。
  table.style.tableLayout = "fixed";
  table.style.minWidth = `${220 + selected.length * 260}px`;
  const colgroup = el("colgroup");
  colgroup.append(el("col"));
  selected.forEach(() => {
    const col = el("col");
    col.style.width = "260px";
    colgroup.append(col);
  });
  table.append(colgroup);

  const thead = el("thead");
  const headRow = el("tr");
  headRow.append(el("th", "", "动作身份"));
  selected.forEach(arm => {
    headRow.append(el("th", "", `${MODE_LABEL[arm.mode] || arm.mode} · ${arm.arm_id}`));
  });
  thead.append(headRow);
  const tbody = el("tbody");
  visibleRows.forEach(row => {
    const tr = el("tr");
    const th = el("th", "arm-action-rowhead");
    th.append(el("div", "arm-action-rowhead__type", row.sample?.actionInfo?.typeLabel ?? ""),
      el("div", "arm-action-rowhead__id mono", row.sample?.actionId ?? row.key),
      el("div", "arm-action-rowhead__subject", row.sample?.scope ?? ""));
    tr.append(th);
    selected.forEach(arm => {
      const cell = row.cells.get(arm.arm_key);
      const td = el("td", "arm-action-cell");
      if (!cell || cell.missing) {
        td.append(el("span", "arm-action-none",
          cell?.filteredOut ? NO_RECORD_FILTERED_TEXT : NO_RECORD_TEXT));
        tr.append(td);
        return;
      }
      if (cell.records.length && baseline && arm !== baseline && cell.differs) {
        td.append(el("span", "arm-action-diff-tag", DIFF_TEXT));
      }
      cell.records.forEach((record, i) => {
        const item = el("div", "arm-action-item");
        if (cell.marks[i]) item.classList.add("arm-action-item--diff");
        const btn = el("button", "arm-action-item__main");
        btn.type = "button";
        const line1 = el("div", "arm-action-item__line1");
        line1.append(
          el("span", "arm-action-item__type", record.actionInfo.typeLabel),
          el("span", "mono", ` t=${record.tick}`),
        );
        if (record.actionInfo.omitted) line1.append(el("span", "arm-action-omit-tag", "省略"));
        if (record.historical) line1.append(el("span", "arm-action-hist-tag", "历史执行记录"));
        const line2 = el("div", "arm-action-item__line2", record.actionInfo.statusText);
        const line3 = el("div", "arm-action-item__params", record.actionInfo.paramsLines.join("；"));
        btn.append(line1, line2, line3);
        btn.addEventListener("click", () => { if (opts.onPick) opts.onPick(record, arm); });
        btn.addEventListener("keydown", e => {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); if (opts.onPick) opts.onPick(record, arm); }
        });
        item.append(btn);
        item.append(rawDetails("原始证据", record.evidence?.raw));
        td.append(item);
      });
      tr.append(td);
    });
    tbody.append(tr);
  });
  table.append(thead, tbody);
  scroll.append(table);
  box.append(scroll);
  return box;
}

// ── 动作节点浮层（hover / 键盘 focus）────────────────────────────
// resolve(node) → { record, arm, factualArm } | null。
// 返回 { bindNode, hide, dispose }；调用方在切页/重渲染/筛选时调 hide()。
export function attachActionTooltips(resolve) {
  const tip = el("div", "arm-action-tooltip");
  tip.setAttribute("role", "tooltip");
  tip.style.position = "fixed";
  tip.style.pointerEvents = "none";
  tip.style.display = "none";
  let visible = false;

  const hide = () => {
    visible = false;
    tip.style.display = "none";
    tip.replaceChildren();
  };

  const clamp = (x, y) => {
    const pad = 8;
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    return {
      x: Math.max(pad, Math.min(x + 14, vw - w - pad)),
      y: Math.max(pad, Math.min(y + 16, vh - h - pad)),
    };
  };

  const showAt = (node, clientX, clientY) => {
    const ctx = resolve(node);
    if (!ctx || ctx.record?.kind !== "action") { hide(); return; }
    const { record, arm, factualArm } = ctx;
    tip.replaceChildren();
    const head = el("div", "arm-action-tooltip__head");
    head.append(
      el("span", "arm-action-item__type", record.actionInfo.typeLabel),
      el("span", "mono", ` t=${record.tick}`),
    );
    tip.append(head);
    const addLine = (k, v) => {
      const row = el("div", "arm-action-tooltip__row");
      row.append(el("span", "arm-action-tooltip__k", k), el("span", "arm-action-tooltip__v", v));
      tip.append(row);
    };
    addLine("分支", `${MODE_LABEL[arm.mode] || arm.mode} · ${arm.arm_id}`);
    addLine("实际 tick", String(record.tick));
    addLine("对象", record.actionInfo.subject);
    addLine(record.type === "capture_screenshot" ? "请求状态" : "执行状态", record.actionInfo.statusText);
    // 控制摘要篇幅：参数行最多 4 条，超出折叠计数；末行提示点击完整证据。
    const MAX_PARAM_LINES = 4;
    const params = record.actionInfo.paramsLines;
    params.slice(0, MAX_PARAM_LINES).forEach(line => addLine("参数", line));
    if (params.length > MAX_PARAM_LINES) {
      addLine("参数", `…其余 ${params.length - MAX_PARAM_LINES} 行省略，点击节点查看完整证据`);
    }
    if (factualArm) {
      const summary = factualIdentitySummary(factualArm, record.actionInfo.key);
      if (summary) {
        addLine("事实观察同身份", summary.join(" ‖ "));
      } else {
        addLine("事实观察同身份", "事实观察窗口内无对应身份记录");
      }
    }
    tip.append(evidenceRef(record.evidence));
    tip.append(el("div", "arm-action-tooltip__hint", "点击节点查看完整证据"));
    document.body.append(tip);
    tip.style.display = "block";
    const pos = clamp(clientX, clientY);
    tip.style.left = `${pos.x}px`;
    tip.style.top = `${pos.y}px`;
    visible = true;
  };

  const onKey = e => { if (e.key === "Escape" && visible) hide(); };
  const onEnter = e => showAt(e.currentTarget, e.clientX ?? 0, e.clientY ?? 0);
  const onMove = e => {
    if (!visible) { showAt(e.currentTarget, e.clientX, e.clientY); return; }
    const pos = clamp(e.clientX, e.clientY);
    tip.style.left = `${pos.x}px`;
    tip.style.top = `${pos.y}px`;
  };
  const onFocus = e => {
    const rect = e.currentTarget.getBoundingClientRect();
    showAt(e.currentTarget, rect.left + rect.width / 2, rect.bottom);
  };
  const onBlur = () => hide();

  const bind = node => {
    node.addEventListener("pointerenter", onEnter);
    node.addEventListener("pointermove", onMove);
    node.addEventListener("pointerleave", hide);
    node.addEventListener("focus", onFocus);
    node.addEventListener("blur", onBlur);
  };

  document.addEventListener("keydown", onKey, true);
  const dispose = () => {
    document.removeEventListener("keydown", onKey, true);
    hide();
    if (tip.parentNode) tip.parentNode.removeChild(tip);
  };
  return { bindNode: bind, hide, dispose };
}
