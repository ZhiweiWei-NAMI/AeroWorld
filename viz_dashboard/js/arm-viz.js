// ARM 可视化分区（#/arms 组详情顶部）：
//   1) 紧凑总览表：窗口区间 | factual 事件链+说明 | 各干预臂事件链+说明
//   2) factual / ARM 两个 card：谓词名词为边、实体为节点的图，按 tick 联动
//   3) 切换节点轴：所有事件链发生切换的 tick 紧凑排布
//   4) 谓词/实体悬浮窗：本 tick 四值、绑定、数值证据与 truth 文件来源
// 全部数据来自 /api/arms/viz（事件链与说明来自 objective 产物与 design 实验
// 定义），不做推断；缺失字段如实显示为空。

import { el, MODE_LABEL } from "./arm-common.js";
import { PREDICATE_LABELS, EVENT_FAMILY_LABELS } from "./arm-viz-labels.js";

const CHAIN_KIND_META = Object.freeze({
  action: { label: "动作", color: "#f2b84b" },
  predicate_transition: { label: "谓词切换", color: "#35e4f2" },
  event_occurrence: { label: "事件发生", color: "#62e6a7" },
  event_outcome: { label: "终态", color: "#aa92ff" },
});

const VALUE_COLOR = Object.freeze({
  true: "#62e6a7",
  false: "#5c7180",
  unknown: "#f2b84b",
  out_of_scope: "#47616d",
});

function predicateLabel(predicateId) {
  if (!predicateId) return "（无谓词）";
  return PREDICATE_LABELS[predicateId]
    || String(predicateId).split(".").at(-1).replaceAll("_", " ");
}

function chainLabel(item) {
  if (item.kind === "event_occurrence" || item.kind === "event_outcome") {
    return EVENT_FAMILY_LABELS[item.label] || item.label;
  }
  if (item.kind === "predicate_transition") {
    const base = PREDICATE_LABELS[item.label]
      || String(item.label).split(".").at(-1).replaceAll("_", " ");
    return `${base} ${item.detail}`;
  }
  return item.label;
}

function chainTickText(item) {
  if (item.censored) return "窗末";
  if (item.tick === null || item.tick === undefined) return "终态待定";
  return `t=${item.tick}`;
}

// ── 悬浮窗（固定定位自绘 DOM，Esc 关闭，键盘 focus 同样触发） ───────────

function createTooltip() {
  const tip = el("div", "armviz-tooltip");
  tip.setAttribute("role", "tooltip");
  tip.style.display = "none";
  document.body.append(tip);
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
    return {
      x: Math.max(pad, Math.min(x + 14, window.innerWidth - w - pad)),
      y: Math.max(pad, Math.min(y + 16, window.innerHeight - h - pad)),
    };
  };

  const show = (build, clientX, clientY) => {
    tip.replaceChildren(...build());
    if (!tip.childNodes.length) return;
    tip.style.display = "block";
    const pos = clamp(clientX, clientY);
    tip.style.left = `${pos.x}px`;
    tip.style.top = `${pos.y}px`;
    visible = true;
  };

  const bindNode = (node, build) => {
    const enter = e => {
      const rect = node.getBoundingClientRect();
      show(build, e?.clientX ?? rect.left, e?.clientY ?? rect.bottom);
    };
    const move = e => {
      if (!visible) { enter(e); return; }
      const pos = clamp(e.clientX, e.clientY);
      tip.style.left = `${pos.x}px`;
      tip.style.top = `${pos.y}px`;
    };
    const focus = () => {
      const rect = node.getBoundingClientRect();
      show(build, rect.left, rect.bottom);
    };
    node.addEventListener("pointerenter", enter);
    node.addEventListener("pointermove", move);
    node.addEventListener("pointerleave", hide);
    node.addEventListener("focus", focus);
    node.addEventListener("blur", hide);
  };

  const onKey = e => { if (e.key === "Escape" && visible) hide(); };
  document.addEventListener("keydown", onKey, true);

  return {
    bindNode,
    hide,
    dispose() {
      document.removeEventListener("keydown", onKey, true);
      hide();
      tip.remove();
    },
  };
}

const fmtNumber = value => (typeof value === "number"
  ? new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 3 }).format(value)
  : String(value));

function bindingsText(bindings) {
  return Object.entries(bindings || {})
    .map(([role, value]) => `${role}=${value}`)
    .join(" · ") || "（无绑定）";
}

function tooltipTitleRow(text) {
  return el("div", "armviz-tooltip__head", text);
}

function tooltipKv(k, v, mono = true) {
  const row = el("div", "armviz-tooltip__row");
  row.append(el("span", "armviz-tooltip__k", k));
  const value = el("span", mono ? "mono" : "",
    typeof v === "string" ? v : JSON.stringify(v));
  value.style.textAlign = "right";
  row.append(value);
  return row;
}

// ── 紧凑总览表 ─────────────────────────────────────────────────────────

function windowText(arm) {
  const w = arm.window || {};
  if (!Number.isFinite(w.start_tick) || !Number.isFinite(w.end_tick)) return "—";
  const fork = Number.isFinite(w.fork_tick) ? ` · fork ${w.fork_tick}` : "";
  return `[${w.start_tick}, ${w.end_tick}]${fork}`;
}

function chainBlocks(arm, { onPickTick, compact }) {
  const flow = el("div", "armviz-chain-flow");
  if (!arm.chain.length) {
    flow.append(el("span", "armviz-chain-none", "窗口内无事件链记录"));
  }
  arm.chain.forEach((item, index) => {
    if (index) flow.append(el("span", "armviz-chain-arrow", "→"));
    const meta = CHAIN_KIND_META[item.kind] || { label: item.kind, color: "#8ba2ad" };
    const chip = el("button", "armviz-chain-chip");
    chip.type = "button";
    const dot = el("i", "armviz-chain-dot");
    dot.style.background = meta.color;
    const tick = el("span", "armviz-chain-tick mono", chainTickText(item));
    tick.style.color = item.censored ? "#aa92ff" : "#8ba2ad";
    chip.append(dot, tick, el("span", "", chainLabel(item)));
    chip.title = `${meta.label} · ${item.label}${item.detail ? ` · ${item.detail}` : ""}`;
    chip.addEventListener("click", () => onPickTick(item.tick));
    flow.append(chip);
  });

  const desc = el("div", "armviz-desc");
  const parts = [];
  const design = arm.description?.design || {};
  for (const key of ["mechanism_detail", "design_statement", "design_rationale", "research_question"]) {
    if (typeof design[key] === "string") parts.push(design[key]);
  }
  const config = arm.description?.configuration || {};
  for (const key of ["rationale", "expected"]) {
    if (typeof config[key] === "string") parts.push(config[key]);
  }
  const short = parts.join(" ") || "";
  const line = el("p", "armviz-desc__text", short || "（design 与配置中无文字说明字段）");
  if (compact) line.classList.add("is-clamped");
  desc.append(line);
  if (short || design.source || design.window_basis) {
    const more = el("details", "armviz-desc__more");
    more.append(el("summary", "", "说明与来源"));
    if (short) more.append(el("p", "armviz-desc__full", short));
    if (design.window_basis) {
      more.append(tooltipKv("窗口依据", design.window_basis, false));
    }
    if (config.falsified_if) more.append(tooltipKv("falsified_if", config.falsified_if, false));
    if (design.design_falsified_if) more.append(tooltipKv("falsified_if (design)", design.design_falsified_if, false));
    if (design.source) more.append(el("code", "armviz-desc__src mono", design.source));
    if (config.source) more.append(el("code", "armviz-desc__src mono", config.source));
    desc.append(more);
  }
  return [flow, desc];
}

function chainCell(arm, opts) {
  const cell = el("td", "armviz-chain-cell");
  const [flow, desc] = chainBlocks(arm, opts);
  cell.append(flow, desc);
  return cell;
}

function overviewTable(viz, { onPickTick }) {
  const factual = viz.arms.find(a => a.mode === "factual_observation");
  const interventions = viz.arms.filter(a => a.mode !== "factual_observation");
  const table = el("table", "armviz-overview");
  table.style.tableLayout = "fixed";
  const colgroup = el("colgroup");
  [["armviz-col-window", 150], ["armviz-col-chain", null]].forEach(([cls, width]) => {
    const col = el("col", cls);
    if (width) col.style.width = `${width}px`;
    colgroup.append(col);
  });
  table.append(colgroup);

  const thead = el("thead");
  const headRow = el("tr");
  const thWindow = el("th", "", "窗口区间");
  const thFactual = el("th", "", "factual · 事件链与说明");
  const thArms = el("th", "", interventions.length
    ? `各 ARM · 事件链与说明（${interventions.length} 个干预分支）`
    : "各 ARM · 事件链与说明");
  headRow.append(thWindow, thFactual, thArms);
  thead.append(headRow);
  table.append(thead);

  const tbody = el("tbody");
  const row = el("tr");
  const windowCell = el("td", "armviz-window-cell");
  windowCell.append(el("code", "mono", factual ? windowText(factual) : "—"));
  if (factual?.description?.design?.window_basis) {
    const basis = factual.description.design.window_basis;
    windowCell.append(el("p", "armviz-window-basis",
      typeof basis === "string" ? basis : "窗口依据见 factual 说明与来源"));
  }
  const factualCell = factual
    ? chainCell(factual, { onPickTick, compact: true })
    : el("td", "armviz-chain-cell", "本组没有事实观察分支");
  const armsCell = el("td", "armviz-arms-cell");
  if (!interventions.length) {
    armsCell.append(el("p", "armviz-chain-none", "本组没有干预分支"));
  }
  interventions.forEach(arm => {
    const wrap = el("div", "armviz-arm-entry");
    const head = el("div", "armviz-arm-entry__head");
    head.append(el("span", "armviz-arm-entry__name", `${arm.arm_id}`),
      el("span", "armviz-arm-entry__mode", MODE_LABEL[arm.mode] || arm.mode || ""));
    wrap.append(head);
    const [flow, desc] = chainBlocks(arm, { onPickTick, compact: true });
    wrap.append(flow, desc);
    armsCell.append(wrap);
  });
  row.append(windowCell, factualCell, armsCell);
  tbody.append(row);
  table.append(tbody);
  return table;
}

// ── 谓词图 card：实体节点一行 + 谓词列，边上按 tick 呈现四值 ────────────

function shortPredicate(predicateId) {
  const label = predicateLabel(predicateId);
  return label.length > 14 ? `${label.slice(0, 13)}…` : label;
}

function buildGraphModel(graph) {
  // 动态边按 谓词+绑定 建立索引；静态边按谓词聚合（窗口内全貌）。
  const dynamicByKey = new Map();
  (graph.dynamic_edges || []).forEach(edge => {
    const key = `${edge.predicate_id}|${JSON.stringify(edge.bindings)}`;
    dynamicByKey.set(key, edge);
  });
  // 本 tick 涉及的实体（有动态边且绑定包含它，或为动作对象）。
  const activeEntityIds = new Set();
  dynamicByKey.forEach(edge => {
    Object.values(edge.bindings || {}).forEach(v => {
      if (typeof v === "string") activeEntityIds.add(v);
    });
  });
  (graph.action_entities || []).forEach(id => activeEntityIds.add(id));
  // 静态边涉及实体：窗口内发生过 transition 的绑定。
  const staticEntityIds = new Set();
  (graph.static_edges || []).forEach(edge => {
    edge.transitions.forEach(t => {
      Object.values(t.bindings || {}).forEach(v => {
        if (typeof v === "string") staticEntityIds.add(v);
      });
    });
  });
  const declared = (graph.nodes || []).filter(n => n.source === "manifest.scope");
  const derived = (graph.nodes || []).filter(n => n.source !== "manifest.scope"
    && (activeEntityIds.has(n.id) || staticEntityIds.has(n.id)));
  return { dynamicByKey, activeEntityIds, staticEntityIds, declared, derived };
}

function graphCard(graph, { tooltip, onPickTick, defaultOpen }) {
  const model = buildGraphModel(graph);
  const box = el("details", "armviz-graph");
  box.open = !!defaultOpen;
  const summary = el("summary", "", "");
  summary.append(
    el("span", "armviz-graph__title", `${MODE_LABEL[graph.mode] || graph.mode || ""} · ${graph.arm_id}`),
    el("span", "panel-note mono",
      ` 实体 ${model.declared.length + model.derived.length} · 窗内切换谓词 ${graph.static_edges.length} · 本 tick 边 ${model.dynamicByKey.size}`),
  );
  box.append(summary);

  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.classList.add("armviz-svg");
  const W = 920;
  const rowH = 34;

  // 顶部实体行：每个实体独占一行（active 在上、灰化在下），
  // 实体很多时不再横向挤成一团；hover 数据不变。
  const nodeById = new Map((graph.nodes || []).map(n => [n.id, n]));
  const orderedEntities = [
    ...model.declared.filter(n => model.activeEntityIds.has(n.id)).map(n => n.id),
    ...model.declared.filter(n => !model.activeEntityIds.has(n.id)).map(n => n.id),
    ...model.derived.map(n => n.id),
  ];
  const ENTITY_ROW_H = 30;
  const entityRowTop = 20;
  const entityY = new Map();
  orderedEntities.forEach((id, index) => {
    entityY.set(id, entityRowTop + index * ENTITY_ROW_H);
  });
  const predicateTopY = entityRowTop + orderedEntities.length * ENTITY_ROW_H + 42;
  const predicateCount = Math.max(graph.static_edges.length, 1);
  const H = predicateTopY + predicateCount * rowH + 18;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");

  const svgEl = (tag, attrs = {}) => {
    const node = document.createElementNS(svgNS, tag);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, String(v)));
    return node;
  };

  orderedEntities.forEach(id => {
    const node = nodeById.get(id) || { id, label: id };
    const y = entityY.get(id);
    const isActive = model.activeEntityIds.has(id);
    const g = svgEl("g", { class: "armviz-entity", tabindex: 0, role: "button" });
    const rect = svgEl("rect", {
      x: 30, y: y - 12, width: 176, height: 24, rx: 5,
      class: isActive ? "armviz-entity__box is-active" : "armviz-entity__box",
    });
    g.append(rect);
    const text = svgEl("text", {
      x: 40, y: y + 4,
      class: isActive ? "armviz-entity__label is-active" : "armviz-entity__label",
    });
    text.textContent = node.label.length > 22 ? `${node.label.slice(0, 21)}…` : node.label;
    g.append(text);
    tooltip.bindNode(g, () => {
      const rows = [tooltipTitleRow(node.label),
        tooltipKv("实体 id", id),
        tooltipKv("来源", node.source === "manifest.scope" ? "manifest.scope 声明范围" : "谓词绑定派生")];
      const involving = [];
      model.dynamicByKey.forEach(edge => {
        if (Object.values(edge.bindings || {}).includes(id)) {
          involving.push(`${predicateLabel(edge.predicate_id)}（${edge.value}）`);
        }
      });
      rows.push(tooltipKv("本 tick 涉及谓词", involving.length ? involving.join("；") : "（无动态边命中）", false));
      return rows;
    });
    svg.append(g);
  });

  // 谓词列：静态边每谓词一行；本 tick 命中的绑定边上色。
  const staticEdges = [...graph.static_edges].sort((a, b) =>
    predicateLabel(a.predicate_id).localeCompare(predicateLabel(b.predicate_id), "zh-CN"));
  staticEdges.forEach((edge, index) => {
    const y = predicateTopY + index * rowH + 12;
    const predicateId = edge.predicate_id || "（无谓词）";
    const isActiveHere = [...model.dynamicByKey.keys()]
      .some(key => key.startsWith(`${predicateId}|`));
    // 本 tick 命中的该谓词各绑定
    const hits = [...model.dynamicByKey.entries()]
      .filter(([key]) => key.startsWith(`${predicateId}|`))
      .map(([, value]) => value);
    const nodeX = 300;
    // 边：从每个绑定实体连到谓词节点
    const connectedEntities = new Set();
    hits.forEach(edgeHit => {
      Object.values(edgeHit.bindings || {}).forEach(v => {
        if (typeof v === "string" && entityY.has(v)) connectedEntities.add(v);
      });
    });
    if (!hits.length) {
      edge.transitions.slice(0, 3).forEach(t => {
        Object.values(t.bindings || {}).forEach(v => {
          if (typeof v === "string" && entityY.has(v)) connectedEntities.add(v);
        });
      });
    }
    connectedEntities.forEach(entityId => {
      const y1 = entityY.get(entityId);
      const midY = (y1 + y) / 2;
      svg.append(svgEl("path", {
        d: `M 206 ${y1} C 250 ${y1}, 250 ${midY}, ${nodeX - 8} ${midY} L ${nodeX - 8} ${y}`,
        class: hits.length ? "armviz-edge is-hit" : "armviz-edge",
      }));
    });
    // 谓词节点
    const g = svgEl("g", {
      class: "armviz-predicate", tabindex: 0, role: "button",
    });
    const label = svgEl("text", {
      x: nodeX - 182, y: y + 4, class: isActiveHere
        ? "armviz-predicate__label is-hit" : "armviz-predicate__label",
    });
    label.textContent = shortPredicate(predicateId);
    g.append(label);
    const valueBadges = svgEl("g", {});
    hits.forEach((edgeHit, hitIndex) => {
      const color = VALUE_COLOR[edgeHit.value] || "#5c7180";
      const bx = nodeX + 140;
      const by = y - 5 + hitIndex * 16;
      const badge = svgEl("g", { class: "armviz-value-badge" });
      badge.append(svgEl("rect", {
        x: bx, y: by, width: 200, height: 12, rx: 3,
        fill: "none", stroke: color, "stroke-width": 1,
      }));
      const bindings = bindingsText(edgeHit.bindings);
      const badgeText = svgEl("text", {
        x: bx + 100, y: by + 9.5, "text-anchor": "middle",
        fill: color, "font-size": 9,
      });
      badgeText.textContent = `${edgeHit.value} · ${bindings}`.slice(0, 42);
      badge.append(badgeText);
      valueBadges.append(badge);
    });
    svg.append(valueBadges);
    tooltip.bindNode(g, () => {
      const rows = [tooltipTitleRow(predicateLabel(predicateId)),
        tooltipKv("谓词 id", predicateId)];
      if (hits.length) {
        hits.forEach(edgeHit => {
          rows.push(tooltipKv("本 tick", `${edgeHit.value} · tick ${edgeHit.tick}`));
          (edgeHit.numbers || []).forEach(numberRow => {
            rows.push(tooltipKv("数值证据", `${numberRow.path} = ${fmtNumber(numberRow.value)}`));
          });
          rows.push(tooltipKv("绑定", bindingsText(edgeHit.bindings)));
        });
      } else {
        rows.push(tooltipKv("本 tick", "该谓词在本 tick 无物化真值行"));
      }
      rows.push(tooltipKv("窗口内切换",
        edge.transitions.map(t => `${t.from_tick}→${t.to_tick} ${t.from_value}→${t.to_value}`).join("；") || "（无）",
        false));
      rows.push(tooltipKv("证据", hits.length
        ? hits[0].evidence.path
        : "objective/predicate_transitions.jsonl", false));
      return rows;
    });
    svg.append(g);
  });

  svg.style.width = "100%";
  svg.setAttribute("preserveAspectRatio", "xMidYMin meet");
  const holder = el("div", "armviz-graph__holder");
  holder.append(svg);
  box.append(holder);

  if (!graph.truth_tick_available) {
    box.append(el("p", "panel-note", "该 tick 无本臂 predicate_truth 样本（不在窗口覆盖范围），仅显示窗口静态结构。"));
  }
  if (!model.dynamicByKey.size) {
    box.append(el("p", "panel-note", "该 tick 没有声明范围内实体的谓词真值命中。"));
  }
  return box;
}

// ── 切换节点轴 ─────────────────────────────────────────────────────────

function switchAxis(viz, { tooltip, onPickTick, window: win }) {
  const box = el("section", "panel armviz-switch");
  const head = el("div", "panel-head");
  head.append(el("h2", "panel-header", "事件链切换节点"),
    el("span", "panel-note", `全部 factual + ARM 切换 tick · ${viz.switch_nodes.length} 个节点 · 点击跳转`));
  box.append(head);
  const strip = el("div", "armviz-switch__strip");
  const [lo, hi] = win;
  const span = Math.max(1, hi - lo);
  viz.switch_nodes.forEach(node => {
    const left = ((node.tick - lo) / span) * 100;
    if (left < 0 || left > 100) return;
    const factualCount = node.events.filter(e => e.arm.endsWith("/arm_factual")
      || e.arm.endsWith("/factual")).length;
    const dot = el("button", "armviz-switch__dot"
      + (factualCount ? " has-factual" : ""));
    dot.type = "button";
    dot.style.left = `${left}%`;
    dot.textContent = String(node.tick);
    tooltip.bindNode(dot, () => [
      tooltipTitleRow(`t=${node.tick} · ${node.events.length} 条切换`),
      ...node.events.slice(0, 8).map(e => tooltipKv(
        e.arm.split("/").at(-1),
        `${CHAIN_KIND_META[e.kind]?.label || e.kind} · ${e.label}`, false)),
      ...(node.events.length > 8
        ? [tooltipKv("…", `其余 ${node.events.length - 8} 条见总览表`, false)] : []),
    ]);
    dot.addEventListener("click", () => onPickTick(node.tick));
    strip.append(dot);
  });
  box.append(strip);
  const scale = el("div", "armviz-switch__scale mono");
  scale.append(el("span", "", String(lo)), el("span", "", `fork ${win[2] ?? "—"}`),
    el("span", "", String(hi)));
  box.append(scale);
  return box;
}

// ── 分区入口 ───────────────────────────────────────────────────────────

export function renderArmVizSection({ root, viz, tick, tooltip, onTickChange, onPickTick }) {
  root.replaceChildren();
  const section = el("div", "armviz");

  // 1) 紧凑总览表
  const overviewPanel = el("section", "panel armviz-overview-panel");
  const overviewHead = el("div", "panel-head");
  overviewHead.append(el("h2", "panel-header", "总览 · 窗口与事件链"),
    el("span", "panel-note", "窗口区间 · factual 与各 ARM 事件链（点击节点跳转该 tick）"));
  overviewPanel.append(overviewHead);
  overviewPanel.append(overviewTable(viz, { onPickTick }));
  section.append(overviewPanel);

  // 2) factual / ARM 两个 card
  const cards = el("div", "armviz-cards");
  const factual = viz.arms.filter(a => a.mode === "factual_observation");
  const interventions = viz.arms.filter(a => a.mode !== "factual_observation");
  const graphFor = arm => viz.graphs?.find(g => g.arm_key === arm.arm_key);
  const buildCard = (title, arms, defaultOpen) => {
    const card = el("section", "panel armviz-card");
    const head = el("div", "panel-head");
    head.append(el("h2", "panel-header", title),
      el("span", "panel-note", `谓词为边 · 实体为节点 · tick ${tick}（输入框联动）`));
    card.append(head);
    if (!arms.length) {
      card.append(el("p", "panel-body", "本组没有该类分支。"));
      return card;
    }
    arms.forEach((arm, index) => {
      const graph = graphFor(arm);
      if (!graph) {
        card.append(el("p", "callout callout-danger", `分支 ${arm.arm_id} 缺少谓词图数据`));
        return;
      }
      card.append(graphCard(graph, {
        tooltip, onPickTick, defaultOpen: defaultOpen && index === 0,
      }));
    });
    return card;
  };
  cards.append(buildCard("factual · 谓词图", factual, true));
  cards.append(buildCard("ARM · 谓词图", interventions, true));
  section.append(cards);

  // 3) 切换节点轴
  const fork = viz.arms[0]?.window?.fork_tick;
  section.append(switchAxis(viz, {
    tooltip, onPickTick, window: [viz.arms[0]?.window?.start_tick ?? 0,
      viz.arms[0]?.window?.end_tick ?? 900, fork],
  }));

  // tick 控件（本分区自己的输入，与顶栏/其他视图互不干扰）
  const controls = el("div", "armviz-controls");
  const label = el("label", "armviz-controls__field");
  label.append(el("span", "armviz-controls__label", "谓词图 tick"));
  const input = el("input");
  input.type = "number";
  input.step = "5";
  input.min = "0";
  input.max = "900";
  input.value = String(tick);
  input.setAttribute("aria-label", "谓词图 tick");
  input.addEventListener("change", () => {
    const value = Number.parseInt(input.value, 10);
    if (!Number.isInteger(value) || value < 0 || value > 900 || value % 5 !== 0) {
      input.value = String(tick);
      return;
    }
    onTickChange(value);
  });
  label.append(input);
  controls.append(label);
  const step = (delta) => {
    const next = Math.max(0, Math.min(900, tick + delta));
    if (next !== tick) onTickChange(next);
  };
  const minus = el("button", "button-secondary", "−5");
  const plus = el("button", "button-secondary", "+5");
  minus.type = plus.type = "button";
  minus.addEventListener("click", () => step(-5));
  plus.addEventListener("click", () => step(5));
  controls.append(minus, plus);
  section.append(controls);

  root.append(section);
}
