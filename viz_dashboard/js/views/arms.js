// 逻辑时间链视图（#/arms）
//
// 每组展示所有分支（事实观察/干预），同一时间刻度对照四类真实记录
// （动作、谓词变化、事件发生、终态/右删失）。每臂一组轨道：动作轨、每个
// 完整 predicate_key（含 bindings）一条谓词轨、事件发生轨、终态轨。所有臂共享
// 一条时间刻度，fork 线只画在本臂轨道范围内。点击节点查证证据与关系；
// 所有呈现来自 arm_review 的实际文件行，不做推断。

import {
  MODE_LABEL, RECORD_KIND, el, panel, evidenceRef, rawDetails, relationLabel,
  resolveEndpoint, armRecords, predicateTrackKey, shortPredicateLabel, shortBinding,
  hashParams, replaceHashParams, exactGroup, buildGroupSelector,
  buildSelectionModel,
} from "../arm-common.js";
import { armActionRecords } from "../arm-actions.js";
import { attachActionTooltips, buildActionTablePanel } from "../arm-action-ui.js";
import { buildArmDeltaSection } from "../arm-delta-ui.js";
import { renderArmVizSection } from "../arm-viz.js";

const SVG_NS = "http://www.w3.org/2000/svg";

// 可视化分区悬浮窗：谓词/实体/切换节点 hover 呈现证据与数值。
function createVizTooltip() {
  const tip = document.createElement("div");
  tip.className = "armviz-tooltip";
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

function svgNode(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function modeLabel(mode) { return MODE_LABEL[mode] || mode || "未知模式"; }

// ── 组头：分叉/窗口/mode/基线说明 ──
function groupHeader(detail) {
  const box = panel(`组 ${detail.group}`, `家族 ${detail.family} · profile ${detail.profile} · seed ${detail.seed}`);
  const facts = el("div", "arm-facts");
  const windows = new Map();
  detail.arms.forEach(arm => {
    const w = arm.window || {};
    const key = `${w.start_tick}–${w.end_tick}`;
    if (!windows.has(key)) windows.set(key, { start: w.start_tick, end: w.end_tick, arms: [] });
    windows.get(key).arms.push(arm);
  });
  const forkTicks = [...new Set(detail.arms.map(a => a.window?.fork_tick).filter(v => typeof v === "number"))];
  const rows = [
    ["分支 mode", detail.arms.map(a => modeLabel(a.mode)).join(" · ")],
    ["分叉时刻", forkTicks.length ? forkTicks.join(" / ") : "未声明"],
    ["窗口范围", [...windows.keys()].join(" / ")],
    ["臂数", String(detail.arms.length)],
  ];
  rows.forEach(([k, v]) => {
    const row = el("div");
    row.append(el("span", "arm-facts__k", k), el("span", "arm-facts__v mono", v));
    facts.append(row);
  });
  box.append(facts);

  const hasFactual = detail.arms.some(a => a.mode === "factual_observation");
  const notes = [];
  if (hasFactual) notes.push("事实观察分支是本组对比基线，单列展示，不与其他分支合并。");
  else notes.push("本组没有事实观察分支，无基线可对比。");
  notes.forEach(text => box.append(el("p", "panel-note", text)));

  // 组级错误分类：窗口外引用不是后端错误；其余才作为记录错误呈现。
  if (detail.errors?.length) {
    const outside = detail.errors.filter(e => e.kind === "relation_endpoint_outside_window");
    const realErrors = detail.errors.filter(e => e.kind !== "relation_endpoint_outside_window");
    if (outside.length) {
      const callout = el("details", "arm-outside-summary");
      callout.append(el("summary", "", `本组有 ${outside.length} 条窗口外引用关系端点（窗前已发生，见各臂关系列表）`));
      outside.forEach(e => {
        const item = el("p", "panel-note");
        item.append(el("code", "mono", `${e.arm} · ${e.message}`));
        callout.append(item);
      });
      box.append(callout);
    }
    if (realErrors.length) {
      box.append(el("p", "callout callout-danger",
        `本组存在 ${realErrors.length} 条后端记录错误（详见下方错误列表），不会当作未发生处理。`));
    }
  }
  return box;
}

// ── 记录详情：中文类别、真实 tick、对象/谓词/事件、状态、证据，折叠原始 JSON ──
function recordDetail(record, arm) {
  const box = el("div", "arm-node-detail");
  const head = el("div", "arm-node-detail__head");
  head.append(el("span", `arm-record-chip arm-record-chip--${record.kind}`, RECORD_KIND[record.kind].label));
  const tickText = record.tick ?? (record.censored ? "null（右删失）" : "null（终态待定）");
  head.append(el("code", "mono", `actual_tick ${tickText}`));
  if (record.censored) head.append(el("span", "arm-censored-tag", "窗口结束仍未终止"));
  else if (record.displayTick !== undefined && record.displayTick !== null
    && record.displayTick !== record.tick) {
    head.append(el("span", "panel-note", `边界位置 display_tick ${record.displayTick}`));
  }
  box.append(head);

  const facts = el("dl", "arm-facts arm-facts--detail");
  let factCount = 0;
  const addRow = (k, v) => {
    if (v === undefined || v === null || v === "") return;
    facts.append(el("dt", "", k), el("dd", "", String(v)));
    factCount++;
  };
  const raw = record.evidence.raw;
  if (record.kind === "action") {
    if (record.actionInfo) {
      addRow("动作类型", record.actionInfo.typeLabel + (record.historical ? "（历史执行记录）" : ""));
      addRow("动作 id", record.actionId);
      addRow("对象", record.actionInfo.subject);
      addRow("执行状态", record.actionInfo.statusText);
      record.actionInfo.paramsLines.forEach(line => addRow("关键参数", line));
    } else {
      addRow("动作类型", raw.action?.type);
      addRow("动作 id", raw.action?.action_id);
      addRow("对象", raw.action?.entity_id);
      addRow("执行状态", raw.result?.status);
    }
    addRow("来源", raw.origin);
  } else if (record.kind === "transition") {
    addRow("谓词", raw.predicate_id);
    addRow("绑定", shortBinding(record.raw.predicate_key));
    // X 族实际字段是 before/after；普通族是 from_value/to_value。原始值保留。
    const before = raw.before ?? raw.from_value;
    const after = raw.after ?? raw.to_value;
    addRow("变化", `${before ?? "?"} → ${after ?? "?"}`);
    addRow("before 原始值", raw.before);
    addRow("after 原始值", raw.after);
    addRow("对象", raw.bindings ? Object.values(raw.bindings).join(", ") : null);
  } else if (record.kind === "occurrence") {
    addRow("事件 id", raw.event_id);
    addRow("事件类型", raw.event_type_id);
    addRow("事件族", raw.event_family_id);
    addRow("触发谓词", raw.trigger_predicate_id);
    addRow("绑定", raw.bindings ? JSON.stringify(raw.bindings) : null);
    addRow("ontology_event", raw.ontology_event);
  } else {
    addRow("事件 id", raw.event_id);
    addRow("终态 status", raw.status);
    addRow("terminal_tick", raw.terminal_tick === null || raw.terminal_tick === undefined
      ? raw.status === "right_censored" ? "null（右删失）" : "null（终态待定）"
      : raw.terminal_tick);
    addRow("window_censor_tick", raw.window_censor_tick);
    addRow("ontology_event", raw.ontology_event);
    addRow("ontology_certification", arm.window_status?.content?.ontology_certification);
  }
  if (factCount === 0) facts.append(el("div", "", "（该记录没有额外结构化字段）"));
  box.append(facts);
  // 详情保留完整谓词键，绝不截断
  if (record.kind === "transition" && record.raw?.predicate_key) {
    box.append(el("p", "arm-predicate-full mono", record.raw.predicate_key));
  }
  box.append(evidenceRef(record.evidence));
  box.append(rawDetails("原始 JSON", record.evidence?.raw));
  return box;
}

// ── 每臂轨道布局：动作轨 + 每个完整 predicate_key 一条谓词轨 + 发生轨 + 终态轨 ──
const TRACK_GAP = 8;
const TRACK_BASE = 30;      // 轨道内最小可用高度（同刻一条记录）
const STACK_STEP = 13;      // 同刻每多一条记录的纵向增量
const LABEL_W = 220;        // 左侧轨道标签宽（SVG 内）
const ARM_HEADER_H = 26;    // 每臂标题行高，标题与第一条轨道不重叠

// 堆叠分桶必须与绘制一致：按 recordDisplayTick（含窗前/右删失边界映射）。
function trackHeight(list, arm) {
  const maxStack = Math.max(1, ...[...list.reduce((m, r) => {
    const key = String(recordDisplayTick(r, arm));
    m.set(key, (m.get(key) || 0) + 1);
    return m;
  }, new Map()).values()]);
  return TRACK_BASE + (maxStack - 1) * STACK_STEP + TRACK_GAP;
}

// 计算一个臂的轨道结构（无渲染副作用，便于复用高度计算）。
// 窗前/右删失 marker 并入统一记录流，落在对应类别轨道上，可点击、受筛选控制。
function armTracks(arm, state) {
  const visible = r => {
    if (!state) return true;
    if (!state.kinds.has(r.kind)) return false;
    const dt = recordDisplayTick(r, arm);
    if (dt === null || dt === undefined) return false;
    return dt >= state.range[0] && dt <= state.range[1];
  };
  const records = armRecords(arm).filter(visible);
  const markers = armMarkers(arm).filter(visible);
  // 动作按身份（type|action_id|scope）分行，每行常驻可读标签。
  const actionIdentities = [...new Set(
    records.filter(r => r.kind === "action" && r.actionInfo).map(r => r.actionInfo.key)
  )];
  const trackDefs = actionIdentities.map(key => {
    const sample = records.find(r => r.kind === "action" && r.actionInfo?.key === key);
    return { id: `action:${key}`, label: key, kinds: ["action"], action_key: key, sample };
  });
  // 兜底（不应出现）：必需 actionInfo 缺失是数据错误，明确报错，不用未命名轨掩盖。
  const orphanActions = records.filter(r => r.kind === "action" && !r.actionInfo);
  if (orphanActions.length) {
    throw new Error(`臂 ${arm.arm_key} 有 ${orphanActions.length} 条动作记录缺少 actionInfo，不能生成轨道`);
  }
  const predKeys = [...new Set(
    records.filter(r => r.kind === "transition" && r.raw?.predicate_key)
      .map(r => predicateTrackKey(r.raw.predicate_key))
  )];
  predKeys.forEach(key => trackDefs.push({ id: `pred:${key}`, label: key, kinds: ["transition"], predicate_key: key }));
  trackDefs.push({ id: "occurrence", label: "事件发生", kinds: ["occurrence"] });
  trackDefs.push({ id: "outcome", label: "终态", kinds: ["outcome"] });
  const tracks = trackDefs.map(def => ({
    ...def,
    records: [...records, ...markers].filter(r => {
      if (!def.kinds.includes(r.kind)) return false;
      if (def.action_key) return r.actionInfo?.key === def.action_key;
      if (def.predicate_key) return predicateTrackKey(r.raw?.predicate_key) === def.predicate_key;
      return true;
    }),
  })).filter(t => t.records.length > 0 || ["occurrence", "outcome"].includes(t.id));
  return { records: [...records, ...markers], tracks };
}

// marker 落点：displayTick（窗前/右删失映射到边界，普通记录用真实 tick）
function recordDisplayTick(record, arm) {
  const start = arm.window?.start_tick;
  const end = arm.window?.end_tick;
  if (record.marker === "pre_window") return start;
  if (record.marker === "censored") return end;
  return record.tick;
}

// ── 主时间轴：每臂一组轨道，同刻堆叠，fork 只画在本臂 ──
function timeline(detail, state, onPick, tooltipApi) {
  const arms = detail.arms.filter(a => state.selectedArms.has(a.arm_key));
  const box = panel("逻辑时间链", "同一时间刻度 · 每臂按谓词+绑定分轨 · 点击节点查证");
  if (!arms.length) { box.append(el("p", "panel-body", "没有选中分支。")); return box; }

  const starts = arms.map(a => a.window?.start_tick).filter(Number.isFinite);
  const ends = arms.map(a => a.window?.end_tick).filter(Number.isFinite);
  if (!starts.length || !ends.length) {
    box.append(el("p", "panel-body", "分支窗口不完整，无法绘制时间轴。"));
    return box;
  }
  const t0 = Math.min(...starts, state.range[0]);
  const t1 = Math.max(...ends, state.range[1]);

  // 布局：标题 36px + 每臂轨道组 + 组间距
  const ARM_GAP = 26;
  const HEAD_H = 36;
  const armLayouts = arms.map(arm => {
    const { records, tracks } = armTracks(arm, state);
    const height = ARM_HEADER_H + tracks.reduce((sum, t) => sum + trackHeight(t.records, arm), 0);
    return { arm, records, tracks, height };
  });
  const H = HEAD_H + armLayouts.reduce((sum, l) => sum + l.height + ARM_GAP, 0);
  const width = state.width;
  const px = tick => ((tick - t0) / Math.max(1, t1 - t0)) * (width - LABEL_W - 24) + LABEL_W + 12;

  const svg = svgNode("svg", {
    viewBox: `0 0 ${width} ${H}`, class: "arm-chain", role: "img",
    "aria-label": "各分支逻辑时间链",
  });
  svg.style.width = `${width}px`;
  svg.style.minWidth = `${width}px`;
  // height 保持自动，避免 720 viewBox 在更大的容器里被 SVG 拉伸放大。

  // 共享时间刻度
  svg.append(svgNode("text", { x: 8, y: 14, fill: "#8ba2ad", "font-size": 11 }, `tick ${t0}`));
  svg.append(svgNode("text", { x: width - 8, y: 14, fill: "#8ba2ad", "font-size": 11, "text-anchor": "end" }, `tick ${t1}`));
  for (let t = Math.ceil(t0 / 50) * 50; t <= t1; t += 50) {
    svg.append(svgNode("line", { x1: px(t), x2: px(t), y1: 20, y2: H - 6, stroke: "rgba(30,52,64,.55)", "stroke-width": 1 }));
    svg.append(svgNode("text", { x: px(t), y: 32, fill: "#5c7180", "font-size": 9, "text-anchor": "middle" }, String(t)));
  }

  const kindColor = kind => RECORD_KIND[kind].color;
  let y = HEAD_H;

  armLayouts.forEach(({ arm, records, tracks, height }) => {
    const start = arm.window?.start_tick;
    const end = arm.window?.end_tick;
    const groupTop = y;
    const groupBottom = y + height;

    // 本臂窗口外区域灰化（只覆盖本臂范围）
    if (Number.isFinite(start)) {
      svg.append(svgNode("rect", {
        x: 0, y: groupTop, width: px(start), height, fill: "rgba(114,131,151,.07)",
      }));
    }
    if (Number.isFinite(end)) {
      svg.append(svgNode("rect", {
        x: px(end), y: groupTop, width: width - px(end), height, fill: "rgba(114,131,151,.07)",
      }));
    }

    // 臂标题（在 header 带内，与轨道不重叠）
    svg.append(svgNode("text", {
      x: 8, y: groupTop + ARM_HEADER_H - 10, fill: "#dce8ec", "font-size": 11, "font-weight": "bold",
    }, `${modeLabel(arm.mode)} · ${arm.arm_id}`));

    // 本臂 fork 线：只画在本臂轨道范围内，不贯穿其他臂；文字避开标题带
    const fork = arm.window?.fork_tick;
    if (Number.isFinite(fork)) {
      svg.append(svgNode("line", {
        x1: px(fork), x2: px(fork), y1: groupTop, y2: groupBottom,
        stroke: "#aa92ff", "stroke-width": 1.4, "stroke-dasharray": "5 4",
      }));
      svg.append(svgNode("text", {
        x: px(fork) + 4, y: groupTop + ARM_HEADER_H + 14, fill: "#aa92ff", "font-size": 10,
      }, `fork ${fork}`));
    }

    // 轨道
    let trackY = groupTop + ARM_HEADER_H;
    tracks.forEach(track => {
      const tHeight = trackHeight(track.records, arm);
      const midY = trackY + tHeight / 2 - TRACK_GAP / 2;
      // 轨道标签（长谓词键只显示谓词 id；完整键放 title）
      const labelText = track.predicate_key ? shortPredicateLabel(track.predicate_key) : track.label;
      const label = svgNode("text", { x: 8, y: midY + 4, fill: "#5c7180", "font-size": 9 });
      label.textContent = labelText.length > 20 ? `${labelText.slice(0, 19)}…` : labelText;
      if (track.predicate_key) {
        label.append(svgNode("title", {}, track.predicate_key));
      } else if (track.id.startsWith("pred:")) {
        label.append(svgNode("title", {}, track.id.slice(5)));
      } else if (track.id.startsWith("action:")) {
        const sample = track.sample;
        const full = sample
          ? `${sample.actionInfo.typeLabel} · ${sample.actionId} · ${sample.scope}`
          : track.id.slice(7);
        label.textContent = full.length > 26 ? `${full.slice(0, 25)}…` : full;
        label.append(svgNode("title", {}, full));
      }
      svg.append(label);
      // 动作身份轨：基线下方常驻一行该身份的动作简述（类型 + tick + 状态），
      // shared tick 位置不变，不给其他轨道加大行距。
      if (track.action_key) {
        const lines = track.records
          .filter(r => r.actionInfo)
          .sort((a, b) => (a.tick ?? -1) - (b.tick ?? -1))
          .map(r => `t=${r.tick} ${r.actionInfo.omitted ? "省略" : r.actionInfo.typeLabel}`);
        svg.append(svgNode("text", {
          x: 8, y: midY + 16, fill: "#8ba2ad", "font-size": 8.5,
        }, (lines.join(" · ") || "（无记录）").slice(0, 64)));
      }
      // 轨道基线（只在本臂窗口内画主线）
      if (Number.isFinite(start) && Number.isFinite(end)) {
        svg.append(svgNode("line", {
          x1: px(start), x2: px(end), y1: midY, y2: midY,
          stroke: "#315364", "stroke-width": 2,
        }));
        [start, end].forEach(t => svg.append(svgNode("line", {
          x1: px(t), x2: px(t), y1: midY - 6, y2: midY + 6,
          stroke: "#315364", "stroke-width": 2,
        })));
      }
      // 同刻堆叠
      const byTick = new Map();
      track.records.forEach(r => {
        const dt = recordDisplayTick(r, arm);
        if (dt === null || dt === undefined) return;
        if (!byTick.has(dt)) byTick.set(dt, []);
        byTick.get(dt).push(r);
      });
      byTick.forEach((list, dt) => {
        list.forEach((record, j) => {
          const cy = midY - (list.length - 1) * (STACK_STEP / 2) + j * STACK_STEP;
          const isMarker = record.marker === "pre_window" || record.marker === "censored";
          const isOmitted = record.kind === "action" && !!record.actionInfo?.omitted;
          let node;
          const shapeAttrs = {};
          if (record.kind === "transition" && record.raw?.predicate_key) {
            shapeAttrs["data-predicate-key"] = record.raw.predicate_key;
          }
          if (record.kind === "outcome" && record.censored) {
            shapeAttrs["data-censored"] = "true";
          }
          if (record.kind === "occurrence") {
            shapeAttrs["data-ontology-event"] = String(record.evidence?.raw?.ontology_event);
          }
          // 非动作节点保留可定位/可点击属性：kinds、arm、实际 tick（null 必须是
          // "null" 字符串）、display tick、键盘可达性与 pointer-events（保证
          // 空心 censored/pre_window 图形内部也能命中）。
          // 动作的这组属性只放在 group 上，不重复给 circle。
          const common = {
            ...shapeAttrs,
            ...(record.kind !== "action" ? {
              "data-kind": record.kind,
              "data-arm": arm.arm_key,
              "data-actual-tick": String(record.tick ?? "null"),
              "data-display-tick": String(dt),
              tabindex: 0, role: "button",
              "aria-label": `${RECORD_KIND[record.kind].label} actual_tick ${record.tick ?? "null"} ${record.title}`,
              "pointer-events": "all",
            } : {}),
          };
          if (isMarker) {
            // 窗前三角 / 右删失空心方块，可点击，受筛选控制
            if (record.marker === "pre_window") {
              node = svgNode("path", {
                ...common,
                d: `M ${px(dt) - 5} ${cy - 5} l 10 0 l -5 9 z`,
                fill: "none", stroke: kindColor(record.kind), "stroke-width": 1.6,
                "data-marker": "pre_window",
              });
            } else {
              node = svgNode("rect", {
                ...common,
                x: px(dt) - 5, y: cy - 5, width: 10, height: 10,
                fill: "none", stroke: kindColor(record.kind), "stroke-width": 1.6,
                "data-marker": "censored",
              });
            }
          } else if (isOmitted) {
            // omitted_by_explicit_intervention：空心标记，title 注明省略
            node = svgNode("circle", {
              ...common,
              cx: px(dt), cy, r: 4.2,
              fill: "none", stroke: kindColor(record.kind), "stroke-width": 1.6,
            });
          } else {
            node = svgNode("circle", {
              ...common,
              cx: px(dt), cy, r: 4.2,
              fill: kindColor(record.kind),
              stroke: "rgba(5,10,15,.7)", "stroke-width": 1,
            });
          }
          node.append(svgNode("title", {},
            `${RECORD_KIND[record.kind].label} · actual_tick ${record.tick}`
            + (record.kind === "action" && record.actionInfo
              ? ` · ${record.actionInfo.typeLabel} · ${record.actionInfo.statusText}`
              : isOmitted ? " · 省略" : "")
            + ` · ${record.title}`));
          const show = () => onPick(record, arm);
          // 非动作形状保留自身 click/keydown；动作形状的交互统一在 group 上绑。
          if (record.kind !== "action") {
            node.addEventListener("click", show);
            node.addEventListener("keydown", e => {
              if (e.key === "Enter" || e.key === " ") { e.preventDefault(); show(); }
            });
          }
          // 动作节点：可见形状（圆点/空心省略）放在 group 内，外面套一个透明
          // 大 hit 圆扩大命中区。data-* 与 role/tabindex 只在 group 保留一份，
          // click/keydown/浮层事件只绑 group，避免同一动作被查两次或焦点重复。
          if (record.kind === "action") {
            const card = svgNode("g", {
              class: "arm-action-node",
              "data-kind": record.kind,
              "data-action": record.actionId,
              "data-arm": arm.arm_key,
              "data-action-key": record.actionInfo?.key ?? "",
              "data-actual-tick": String(record.tick ?? ""),
              "data-omitted": isOmitted ? "true" : undefined,
              "data-display-tick": String(dt),
              tabindex: 0, role: "button",
              "aria-label": `动作 ${record.actionInfo?.typeLabel ?? record.title} actual_tick ${record.tick}`,
            });
            card.append(node);
            const hit = svgNode("circle", {
              cx: px(dt), cy, r: 9, fill: "transparent", "pointer-events": "all", class: "arm-action-hit",
            });
            card.append(hit);
            card.addEventListener("click", show);
            card.addEventListener("keydown", e => {
              if (e.key === "Enter" || e.key === " ") { e.preventDefault(); show(); }
            });
            if (tooltipApi) tooltipApi.bindNode(card);
            svg.append(card);
          } else {
            svg.append(node);
          }
        });
      });
      trackY += tHeight;
    });

    y = groupBottom + ARM_GAP;
  });

  const scroll = el("div", "arm-chain-scroll");
  scroll.append(svg);
  box.append(scroll);

  const legend = el("div", "arm-legend");
  Object.entries(RECORD_KIND).forEach(([key, meta]) => {
    const item = el("span", "arm-legend__item");
    const dot = el("i"); dot.style.background = meta.color;
    item.append(dot, el("span", "", `${meta.label}`));
    legend.append(item);
  });
  const pre = el("span", "arm-legend__item");
  const tri = el("i"); tri.style.background = "transparent"; tri.style.border = "1.6px solid #62e6a7"; tri.style.transform = "rotate(45deg)"; tri.style.width = "8px"; tri.style.height = "8px";
  pre.append(tri, el("span", "", "窗前已发生（左边界，真实 tick 见详情）"));
  const cens = el("span", "arm-legend__item");
  const sq = el("i"); sq.style.background = "transparent"; sq.style.border = "1.6px solid #aa92ff"; sq.style.width = "8px"; sq.style.height = "8px";
  cens.append(sq, el("span", "", "右删失（右边界，terminal_tick 仍 null）"));
  legend.append(pre, cens);
  box.append(legend);
  if (!armLayouts.some(l => l.records.some(r => state.kinds.has(r.kind)))) {
    box.append(el("p", "panel-note", "当前筛选条件下没有可见记录。"));
  }
  return box;
}

// 窗前已发生 / 右删失：并入统一记录流，可点击、受筛选控制
function armMarkers(arm) {
  const markers = [];
  const start = arm.window?.start_tick;
  const end = arm.window?.end_tick;
  // 窗前：phase.tick 在窗口 start 之前的引用，真实发生 tick 来自 phase
  for (const occ of arm.objective.occurrences) {
    const raw = occ.evidence.raw;
    const phases = Array.isArray(raw.event_phases) ? raw.event_phases : [];
    for (const phase of phases) {
      if (typeof phase?.tick === "number" && Number.isFinite(start) && phase.tick < start) {
        markers.push({
          kind: "occurrence",
          marker: "pre_window",
          tick: phase.tick,
          title: `${raw.event_id || "(无事件 id)"}（窗前 t=${phase.tick}）`,
          detail: "窗前已发生，窗口内未见 onset",
          status: null,
          evidence: occ.evidence,
          raw: occ,
        });
        break;
      }
    }
  }
  // Only a declared right-censored outcome belongs at the window boundary.
  for (const row of arm.objective.outcomes) {
    if (row.evidence?.raw?.status !== "right_censored") continue;
    markers.push({
      kind: "outcome",
      marker: "censored",
      tick: null,
      displayTick: end,
      title: `${row.evidence?.raw?.event_id || "(无事件 id)"}（右删失）`,
      detail: "右删失：窗口结束仍未终止，terminal_tick=null",
      status: row.status,
      censored: true,
      evidence: row.evidence,
      raw: row,
    });  }
  return markers;
}

// ── 选中节点的关系列表（按实际记录的 id 集合建边，不造新边）──
function recordIds(record) {
  const raw = record.evidence.raw;
  const ids = new Set();
  if (record.kind === "action") {
    // 动作只匹配已存在的 chain 节点 action:<action_id>，不造边
    const actionId = raw.action?.action_id;
    if (typeof actionId === "string") ids.add(`action:${actionId}`);
  } else {
    ["node_id", "transition_id", "event_id", "outcome_id"].forEach(field => {
      if (typeof raw[field] === "string") ids.add(raw[field]);
    });
  }
  return ids;
}

function relationsFor(record, arm) {
  const ids = recordIds(record);
  if (!ids.size) return [];
  return arm.chain.relations.filter(rel => ids.has(rel.source) || ids.has(rel.target));
}

// 端点按钮：可点击查看实际端点证据（path:line、raw），不只 title
function endpointNode(endpoint, arm) {
  const r = resolveEndpoint(endpoint, arm);
  const node = el("button", "arm-endpoint mono");
  node.type = "button";
  if (r.kind === "materialized") {
    node.classList.add("arm-endpoint--ok");
    node.textContent = endpoint;
  } else if (r.kind === "outside_window") {
    node.classList.add("arm-endpoint--outside");
    node.textContent = `${endpoint}（窗口外引用 t=${r.tick}）`;
  } else {
    node.classList.add("arm-endpoint--bad");
    node.textContent = `${endpoint}（无法解析）`;
  }
  const openEndpoint = () => {
    let existing = node.nextElementSibling;
    if (existing && existing.classList.contains("arm-endpoint-detail")) {
      existing.remove();
      return;
    }
    const detail = el("div", "arm-endpoint-detail");
    if (r.kind === "materialized") {
      detail.append(evidenceRef(r.evidence));
      detail.append(rawDetails("端点原始 JSON", r.evidence?.raw));
    } else if (r.kind === "outside_window") {
      detail.append(el("p", "panel-note", `事件相位引用时刻 t=${r.tick}（本臂窗口 start 之前）`));
      if (r.occurrence) {
        detail.append(evidenceRef(r.occurrence));
        detail.append(rawDetails("所在 occurrence 原始 JSON", r.occurrence.raw));
      }
    } else {
      detail.append(el("p", "panel-note", "本臂窗口内没有该端点的物化记录，也没有窗外相位引用。"));
    }
    node.after(detail);
  };
  node.addEventListener("click", openEndpoint);
  node.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openEndpoint(); } });
  return node;
}

function relationItem(rel, arm) {
  const item = el("li", "arm-relations__item");
  const head = el("div", "arm-relations__head");
  head.append(endpointNode(rel.source, arm));
  head.append(el("span", "arm-relations__arrow", "→"));
  head.append(endpointNode(rel.target, arm));
  item.append(head);
  item.append(el("p", "arm-relations__type", `${relationLabel(rel.type, rel.raw)} · ${rel.type}`));
  if (Object.prototype.hasOwnProperty.call(rel.raw || {}, "causal_claim")) {
    item.append(el("p", "panel-note", `原始 causal_claim: ${rel.raw.causal_claim}`));
  } else {
    item.append(el("p", "panel-note", "原始行未声明 causal_claim。"));
  }
  item.append(evidenceRef(rel.evidence));
  item.append(rawDetails("原始 relation JSON", rel.raw));
  return item;
}

function relationPanel(record, arm) {
  const relations = relationsFor(record, arm);
  const box = panel("选中节点的关系", "只列出该节点实际存储的关系，不从时间相邻推断因果");
  if (!relations.length) {
    box.append(el("p", "panel-body", "该节点没有存储的关系行。"));
    return box;
  }
  const list = el("ul", "arm-relations");
  relations.forEach(rel => list.append(relationItem(rel, arm)));
  box.append(list);
  return box;
}

// ── 每臂“存储关系”完整列表（可展开），含 X 的 script_dependency ──
function armRelationsPanel(arm) {
  const details = el("details", "arm-arm-relations");
  details.append(el("summary", "",
    `存储关系（${arm.chain.relations.length} 条）· ${arm.arm_id}`));
  const list = el("ul", "arm-relations");
  const relations = arm.chain.relations;
  if (!relations.length) {
    list.append(el("li", "panel-note", "该臂没有存储的关系行。"));
  }
  relations.forEach(rel => list.append(relationItem(rel, arm)));
  details.append(list);
  return details;
}

// ── 审计栏：预期与实测分开，保留 certification 与 limitations，默认折叠 ──
function auditPanel(arm) {
  const details = el("details", "arm-audit");
  details.append(el("summary", "", `审计与声明 · ${arm.arm_id} · ${modeLabel(arm.mode)}`));
  const box = el("div", "arm-audit__body");
  const expected = arm.window_status?.content?.expected
    ?? arm.configuration?.content?.expected;
  if (expected !== undefined) {
    box.append(el("h3", "arm-audit__label", "声明（configuration / window_status 的 expected）"));
    box.append(rawDetails("展开声明", expected));
  } else {
    box.append(el("p", "panel-note", "该臂没有声明 expected 字段。"));
  }
  const ws = arm.window_status?.content || {};
  box.append(el("h3", "arm-audit__label", "实测（window_status / objective）"));
  const facts = el("dl", "arm-facts");
  const addRow = (k, v) => {
    if (v === undefined || v === null) return;
    const row = el("div");
    row.append(el("span", "arm-facts__k", k), el("span", "arm-facts__v mono",
      typeof v === "object" ? JSON.stringify(v) : String(v)));
    facts.append(row);
  };
  addRow("window_status.status", ws.status);
  addRow("absence_is_only_within_window", ws.absence_is_only_within_window);
  addRow("left_boundary_policy", ws.left_boundary_policy);
  addRow("unknown_runtime_policy", ws.unknown_runtime_policy);
  addRow("ontology_certification", ws.ontology_certification);
  addRow("business_event_count", ws.business_event_count);
  addRow("pending_event_ids", Array.isArray(ws.pending_event_ids)
    ? ws.pending_event_ids.length : ws.pending_event_ids);
  addRow("right_censored_event_ids", Array.isArray(ws.right_censored_event_ids)
    ? ws.right_censored_event_ids.length : ws.right_censored_event_ids);
  box.append(facts);
  box.append(rawDetails("window_status 原文", ws));
  // L4 业务声明策略原文直接可见
  if (ws.declaration_policy !== undefined) {
    box.append(el("p", "panel-note", `declaration_policy 原文：${JSON.stringify(ws.declaration_policy)}`));
  }

  if (arm.audit && Object.keys(arm.audit).length) {
    box.append(el("h3", "arm-audit__label", "原始审计字段（certification、limitations）"));
    const audit = arm.audit;
    if (audit.certification !== undefined) {
      box.append(rawDetails("certification", audit.certification));
    }
    if (Array.isArray(audit.limitations) && audit.limitations.length) {
      const list = el("ul", "arm-limitations");
      audit.limitations.forEach(text => list.append(el("li", "", text)));
      box.append(el("h3", "arm-audit__label", "limitations"));
      box.append(list);
    }
    box.append(rawDetails("manifest 中其余审计字段", audit));
  }

  // L4/X：ontology_event=false 或相应原始认证字段必须可见
  const nonOntology = arm.objective.occurrences.filter(o => o.evidence?.raw?.ontology_event === false);
  if (nonOntology.length) {
    box.append(el("p", "callout callout-danger",
      `该臂有 ${nonOntology.length} 条 ontology_event=false 的事件发生（业务事件未进入本体），逐条见下方列表。`));
    nonOntology.forEach(o => box.append(rawDetails(`ontology_event=false · ${o.evidence.raw.event_id}`, o)));
  }
  if (arm.audit?.certification && typeof arm.audit.certification === "object") {
    const flat = JSON.stringify(arm.audit.certification);
    if (/false|"fail"|no_go|NO-GO/i.test(flat)) {
      box.append(el("p", "callout callout-danger", "certification 中存在 false / fail / NO-GO 字段，原始内容见审计栏。"));
    }
  }
  details.append(box);
  return details;
}

function armPicker(detail, state, redraw) {
  const box = panel("分支选择", "勾选需要在时间轴上对照的分支");
  const list = el("div", "arm-branches");
  detail.arms.forEach(arm => {
    const label = el("label", "arm-branch");
    const check = el("input");
    check.type = "checkbox";
    check.checked = state.selectedArms.has(arm.arm_key);
    check.addEventListener("change", () => {
      if (check.checked) state.selectedArms.add(arm.arm_key);
      else state.selectedArms.delete(arm.arm_key);
      redraw();
    });
    label.append(check, el("span", "", `${modeLabel(arm.mode)} · ${arm.arm_id}`));
    list.append(label);
  });
  box.append(list);

  const kinds = el("div", "arm-kinds");
  Object.entries(RECORD_KIND).forEach(([key, meta]) => {
    const label = el("label", "arm-branch");
    const check = el("input");
    check.type = "checkbox";
    check.checked = state.kinds.has(key);
    check.addEventListener("change", () => {
      if (check.checked) state.kinds.add(key); else state.kinds.delete(key);
      redraw();
    });
    const dot = el("i"); dot.style.background = meta.color; dot.className = "arm-kind-dot";
    label.append(check, dot, el("span", "", meta.label));
    kinds.append(label);
  });
  box.append(el("h3", "arm-audit__label", "记录类别"));
  box.append(kinds);

  const range = el("div", "arm-range");
  const lo = el("input"); lo.type = "number"; lo.value = state.range[0]; lo.setAttribute("aria-label", "起始 tick");
  const hi = el("input"); hi.type = "number"; hi.value = state.range[1]; hi.setAttribute("aria-label", "结束 tick");
  const hint = el("p", "arm-range__hint panel-note");
  range.append(el("span", "arm-facts__k", "时间范围"), lo, hi, hint);
  [lo, hi].forEach(input => input.addEventListener("change", () => {
    const a = Number.parseInt(lo.value, 10);
    const b = Number.parseInt(hi.value, 10);
    hint.textContent = "";
    if (!Number.isInteger(a) || !Number.isInteger(b)) {
      hint.textContent = "时间范围必须是整数 tick，未应用。";
      return;
    }
    if (a > b) {
      hint.textContent = "起始 tick 不能大于结束 tick，未应用。";
      return;
    }
    state.range = [a, b];
    redraw();
  }));
  box.append(el("h3", "arm-audit__label", "时间范围"));
  box.append(range);
  return box;
}

// ── Profile 概览：本场景的干预方案（每 profile+seed 先展示这一页） ──
// 每行“干预名称 | 实际改动 | 实测结果（factual→ARM）”，预期默认折叠，
// 点击行打开该组现有差异/时间链视图（写入完整 hash）。
function measuredText(measured, arm) {
  const parts = measured.map(m => {
    if (arm.is_business_observation) return `${m.label}：${m.text}`;
    return `${m.family || m.label}：${m.text}`;
  });
  return parts.join("；") || "（该臂未声明目标族）";
}

function expectedDetails(arm) {
  const details = el("details", "arm-plan-expected");
  details.append(el("summary", "", "配置预期（声明，非实测）"));
  if (arm.expected && Object.keys(arm.expected).length) {
    const list = el("ul", "arm-plan-expected__list");
    Object.entries(arm.expected).forEach(([k, v]) => {
      const li = el("li", "panel-note");
      li.append(el("span", "arm-facts__k", k), "：", (typeof v === "object" ? JSON.stringify(v) : String(v)));
      list.append(li);
    });
    details.append(list);
  } else {
    details.append(el("p", "panel-note", "该臂配置没有声明 expected。"));
  }
  details.append(rawDetails("配置原文证据", arm.evidence.configuration));
  return details;
}

function planRow(arm, onOpen) {
  const row = el("div", "arm-plan-row");
  const head = el("div", "arm-plan-row__head");
  head.setAttribute("role", "button");
  head.tabIndex = 0;
  const name = el("div", "arm-plan-row__name", arm.name);
  name.append(el("span", "arm-plan-row__id mono", arm.arm_id));
  const mech = el("div", "arm-plan-row__mech", arm.mechanism_summary);
  const measured = el("div", "arm-plan-row__measured");
  measured.append(el("span", "arm-plan-row__label", "实测"));
  if (arm.factual_measured) {
    measured.append(el("div", "arm-plan-row__value", `factual：${measuredText(arm.factual_measured, arm)}`));
  }
  measured.append(el("div", "arm-plan-row__value", `ARM：${measuredText(arm.measured, arm)}`));
  if (arm.window && arm.factual_window) {
    const b = arm.factual_window, a = arm.window;
    measured.append(el("div", "panel-note", `观察窗口：factual [${b.start_tick}, ${b.end_tick}] → ARM [${a.start_tick}, ${a.end_tick}]`));
  }
  if (arm.comparability_note) {
    measured.append(el("span", "panel-note", arm.comparability_note));
  }
  if (arm.window_status_status) {
    measured.append(el("span", "arm-plan-row__ws mono", `window_status: ${arm.window_status_status}`));
  }
  head.append(name, mech, measured, expectedDetails(arm));
  head.append(rawDetails("原始证据路径", arm.evidence));
  const open = () => onOpen(arm.group_key);
  head.addEventListener("click", e => {
    if (!e.target.closest(".arm-plan-expected, .arm-source, a, button")) open();
  });
  head.addEventListener("keydown", e => {
    if (e.target === head && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); open(); }
  });
  row.append(head);
  return row;
}

function profilePlanPanel(overview, onOpen) {
  const box = panel(
    `本场景的干预方案 · ${overview.profile} / ${overview.seed}`,
    `${overview.counts.groups} 组 · ${overview.counts.arms} 臂 · `
    + `${overview.counts.interventions} 个干预分支`,
  );
  box.append(el("p", "panel-note",
    "逐项对照 factual 与干预的目标事件。点击方案查看动作差异和完整时间链，配置预期可展开核对。"));
  if (!overview.counts.interventions) {
    box.append(el("p", "callout callout-danger",
      "本场景仅保留 factual，干预待重新设计。"));
  }
  overview.groups.forEach(group => {
    const groupBox = el("details", "arm-plan-group");
    groupBox.open = Boolean(group.interventions.length);
    groupBox.append(el("summary", "",
      `${group.group}（${group.family} · 干预 ${group.interventions.length} 个）`));
    if (group.note) groupBox.append(el("p", "panel-note", group.note));
    const factual = group.factual;
    if (factual && !group.interventions.length) {
      const factualRow = el("div", "arm-plan-factual");
      factualRow.append(el("span", "arm-plan-row__label", "基线"));
      factualRow.append(el("span", "arm-plan-row__value",
        `${factual.name} · ${measuredText(factual.measured, factual)}`));
      groupBox.append(factualRow);
      const open = el("button", "button-secondary", "查看事实窗口");
      open.addEventListener("click", () => onOpen(group.group_key));
      groupBox.append(open);
    }
    const rows = group.interventions.length
      ? group.interventions : group.arms.filter(a => a.mode !== "factual_observation");
    rows.forEach(arm => groupBox.append(planRow(arm, onOpen)));
    box.append(groupBox);
  });
  return box;
}

function profilePicker(catalogPayload, current, onChange) {
  const profiles = buildSelectionModel(catalogPayload.groups);
  const profileSelect = el("select"); profileSelect.setAttribute("aria-label", "profile");
  const seedSelect = el("select"); seedSelect.setAttribute("aria-label", "seed");
  const names = [...profiles.keys()].sort();
  profileSelect.replaceChildren(...names.map(name => {
    const o = el("option", "", name); o.value = name; return o;
  }));
  const syncSeeds = () => {
    const seeds = [...profiles.get(profileSelect.value).seeds].sort();
    seedSelect.replaceChildren(...seeds.map(name => {
      const o = el("option", "", name); o.value = name; return o;
    }));
    seedSelect.value = seeds.includes(current.seed) ? current.seed : seeds[0];
  };
  profileSelect.value = names.includes(current.profile) ? current.profile : names[0];
  syncSeeds();
  profileSelect.addEventListener("change", () => {
    syncSeeds();
    onChange({ profile: profileSelect.value, seed: seedSelect.value });
  });
  seedSelect.addEventListener("change", () => {
    onChange({ profile: profileSelect.value, seed: seedSelect.value });
  });
  const row = el("div", "arm-selector-row");
  [["profile", profileSelect], ["seed", seedSelect]].forEach(([label, select]) => {
    const wrap = el("label", "arm-selector__field");
    wrap.append(el("span", "arm-selector__label", label), select);
    row.append(wrap);
  });
  return row;
}

export async function renderArms(ctx) {
  const { root, api, store, navigate, ui } = ctx;
  let disposed = false;
  let sequence = 0;
  let resizeObserver = null;
  let tooltipApi = null;
  let vizTooltip = null;
  root.replaceChildren(el("p", "loading-indicator", "正在读取 ARM 目录…"));

  try {
    const params = hashParams();
    const payload = await api.armsCatalog();
    if (disposed) return undefined;
    root.replaceChildren();

    const view = el("div", "view arms-view");
    view.append(el("h1", "view-header", "逻辑时间链"));
    view.append(el("p", "view-subtitle mono",
      `实算 ${payload.counts.arms} 臂 / ${payload.counts.groups} 组 · `
      + Object.entries(payload.counts.by_mode).map(([m, n]) => `${MODE_LABEL[m] || m} ${n}`).join(" · ")));
    view.append(el("p", "panel-note",
      "所有节点来自实际产物文件行（evidence path:line），不做推断。右删失 marker 表示窗口结束仍未终止，真实 terminal_tick 仍为 null。"));

    // 无效显式参数显式报错，不偷偷换组
    const invalid = Object.keys(params).filter(k => !["profile", "seed", "group"].includes(k));
    if (invalid.length) throw new Error(`链接包含未知参数: ${invalid.join(", ")}`);
    const groups = payload.groups;
    if (!payload.delivery_cross_check.consistent) {
      const check = payload.delivery_cross_check;
      const warning = el("div", "callout callout-danger");
      warning.append(el("p", "", `部分数据暂不可用：${check.unavailable_groups.length} 个组缺失或正在更新。当前显示 ${payload.counts.groups} 个可用组。`));
      warning.append(rawDetails("查看不可用组及原因", check.mismatches));
      view.append(warning);
    }
    if (!groups.length) {
      view.append(el("p", "callout callout-danger", "当前没有完整可用的 ARM 组，请在生成完成后刷新。"));
      root.append(view);
      return;
    }
    const explicit = {};
    for (const key of ["profile", "seed", "group"]) {
      if (params[key]) explicit[key] = params[key];
    }
    const explicitKeys = Object.keys(explicit);
    let initial;
    if (explicitKeys.length === 3) {
      const hit = exactGroup(groups, explicit);
      if (!hit) throw new Error(
        `未知组合: profile=${explicit.profile} seed=${explicit.seed} group=${explicit.group}`);
      initial = explicit;
    } else if (explicitKeys.length === 0) {
      // 无参数进入 profile 概览模式：默认选第一个有干预分支的 profile（L2 类），
      // 没有 intervention 时退回第一个 profile，仍允许选 factual-only 场景。
      const withIntervention = groups.find(g => g.family === "L2" && g.arms.some(a => a.mode === "intervention"))
        || groups.find(g => g.arms.some(a => a.mode === "intervention")) || groups[0];
      initial = { profile: withIntervention.profile, seed: withIntervention.seed };
      replaceHashParams("arms", initial);
    } else if (explicitKeys.length === 2 && explicit.profile && explicit.seed) {
      // profile+seed 深链接：profile 概览模式
      initial = { profile: explicit.profile, seed: explicit.seed };
    } else {
      throw new Error(
        `深链接参数不完整: 需要 profile+seed（概览）或 profile+seed+group（组对照），`
        + `当前只有 ${explicitKeys.join(", ")}`);
    }

    const body = el("div", "arm-review-body");
    const errorBox = el("div");
    const vizBox = el("div", "arm-viz-box");
    view.append(vizBox, body, errorBox);
    root.append(view);

    // 可视化分区状态：tick 在组内持久，换组重置为 fork；悬浮窗独立实例。
    const vizState = { tick: null };
    vizTooltip = createVizTooltip();

    if (!params.group) {
      const pickerPanel = panel("场景选择", "选择 profile 和 seed 查看全部干预方案");
      const loadProfile = async selection => {
        const token = ++sequence;
        replaceHashParams("arms", selection);
        body.replaceChildren(el("p", "loading-indicator", "正在读取干预方案…"));
        try {
          const overview = await api.armProfile(selection.profile, selection.seed);
          if (disposed || token !== sequence) return;
          const now = hashParams();
          if (now.profile !== selection.profile || now.seed !== selection.seed || now.group) return;
          store.patch({ episodeId: `${selection.profile}__${selection.seed}` });
          body.replaceChildren(profilePlanPanel(overview, groupKey => {
            const [profile, seed, group] = groupKey.split("/");
            location.hash = `#/arms?${new URLSearchParams({profile, seed, group})}`;
          }));
        } catch (error) {
          if (!disposed && token === sequence) body.replaceChildren(el("p", "callout callout-danger", error.message));
        }
      };
      pickerPanel.append(profilePicker(payload, initial, loadProfile));
      view.insertBefore(pickerPanel, body);
      await loadProfile(initial);
      return () => {
        disposed = true;
        sequence++;
        if (vizTooltip) { vizTooltip.dispose(); vizTooltip = null; }
      };
    }
    const back = el("a", "button-secondary", "← 本场景全部干预方案");
    back.href = `#/arms?${new URLSearchParams({profile: initial.profile, seed: initial.seed})}`;
    view.insertBefore(back, body);

    const state = {
      selectedArms: new Set(),
      kinds: new Set(Object.keys(RECORD_KIND)),
      range: [0, 900],
      width: 720,
    };
    let detailBox = null;
    let currentDetail = null;

    // 动作节点浮层：hover / 键盘 focus，均显示事实观察同身份摘要；route 销毁时清理。
    tooltipApi = attachActionTooltips(node => {
      const armKey = node.getAttribute("data-arm");
      const actionKey = node.getAttribute("data-action-key");
      const tickStr = node.getAttribute("data-actual-tick");
      const arm = currentDetail?.arms.find(a => a.arm_key === armKey);
      if (!arm) return null;
      const tick = tickStr === "" || tickStr === null ? NaN : Number(tickStr);
      const record = armActionRecords(arm).find(r =>
        r.kind === "action" && r.actionInfo?.key === actionKey && r.tick === tick);
      if (!record) return null;
      const factualArm = currentDetail.arms.find(a => a.mode === "factual_observation") || null;
      return { record, arm, factualArm };
    });

    const showPicked = (record, arm) => {
      const d = el("div", "arm-picked");
      d.append(el("h3", "arm-audit__label", "选中节点"));
      d.append(recordDetail(record, arm));
      d.append(relationPanel(record, arm));
      let pickBox = detailBox.querySelector(".arm-picked-wrap");
      if (!pickBox) {
        pickBox = el("div", "arm-picked-wrap");
        detailBox.append(pickBox);
      }
      pickBox.replaceChildren(d);
      // 差异卡的“查看完整记录”落在默认收起的折叠区内：
      // 展开完整记录区并滚动到选中节点，保证点击后明细真正可见。
      const fullDetails = body.querySelector(".arm-full-records");
      if (fullDetails) fullDetails.open = true;
      pickBox.scrollIntoView({ block: "start" });
    };

    // 动作对照表的容器提升到 render 作用域：一次 redraw 同时重建表和 SVG。
    let actionTableWrap = null;
    let actionTableData = null;

    const redraw = () => {
      if (!currentDetail || !detailBox) return;
      // 筛选/重渲染会替换节点，先收起旧浮层避免悬空。
      if (tooltipApi) tooltipApi.hide();
      state.width = Math.max(720, body.clientWidth);
      const scrollPos = detailBox.querySelector(".arm-chain-scroll")?.scrollLeft || 0;
      const pickBox = detailBox.querySelector(".arm-picked-wrap");
      detailBox.replaceChildren();
      detailBox.append(timeline(currentDetail, state, showPicked, tooltipApi));
      if (pickBox) detailBox.append(pickBox);
      const sc = detailBox.querySelector(".arm-chain-scroll");
      if (sc) sc.scrollLeft = scrollPos;
      // 对照表与时间链同一次 redraw 重建（保留表自身滚动/展开状态）。
      if (actionTableWrap && actionTableData) {
        const prev = actionTableWrap.firstElementChild;
        const prevOpen = prev instanceof HTMLDetailsElement ? prev.open : true;
        const scrollTop = actionTableWrap.querySelector(".arm-action-scroll")?.scrollTop || 0;
        const scrollLeft = actionTableWrap.querySelector(".arm-action-scroll")?.scrollLeft || 0;
        const panelNode = buildActionTablePanel(actionTableData, state, {
          onPick: showPicked,
          onToggleDiffOnly: redraw,
        });
        panelNode.open = prevOpen;
        actionTableWrap.replaceChildren(panelNode);
        const sc2 = actionTableWrap.querySelector(".arm-action-scroll");
        if (sc2) { sc2.scrollTop = scrollTop; sc2.scrollLeft = scrollLeft; }
      }
    };

    // app.js 的 renderRoute 在脱离 DOM 的 mount 里 await renderer 后才挂载，
    // 初次 redraw 时 body.clientWidth 为 0，图会永久停在 720 且被 SVG 放大。
    // 等 body 真正进入文档后再观察尺寸，仅在宽度有效且发生变化时重绘。
    const setupResizeObserver = () => {
      if (disposed || resizeObserver) return;
      resizeObserver = new ResizeObserver(() => {
        if (disposed) return;
        const w = body.clientWidth;
        if (w <= 0) return;
        const next = Math.max(720, w);
        if (next === state.width) return;
        redraw();
      });
      resizeObserver.observe(body);
    };
    // renderArms 返回后才被挂载，此时 body 仍未连入文档；逐帧等待连接后建立观察。
    const waitConnected = () => {
      if (disposed || resizeObserver) return;
      if (body.isConnected) setupResizeObserver();
      else requestAnimationFrame(waitConnected);
    };
    if (body.isConnected) setupResizeObserver();
    else requestAnimationFrame(waitConnected);

    // 换组标识：与上次请求的 profile/seed/group 不同时重置可视化 tick。
    let vizGroupKey = null;

    const loadViz = async (profile, seed, group, tick, { resetTick }) => {
      const token = ++sequence;
      if (resetTick) vizState.tick = null;
      vizBox.replaceChildren(el("p", "loading-indicator", "正在组装可视化分区…"));
      try {
        const viz = await api.armViz(profile, seed, group, tick);
        if (disposed || token !== sequence) return;
        vizState.tick = tick;
        vizBox.replaceChildren();
        renderArmVizSection({
          root: vizBox,
          viz,
          tick,
          tooltip: vizTooltip,
          onTickChange: nextTick => loadViz(profile, seed, group, nextTick, { resetTick: false }),
          onPickTick: nextTick => {
            if (Number.isInteger(nextTick)) loadViz(profile, seed, group, nextTick, { resetTick: false });
          },
        });
      } catch (error) {
        if (!disposed && token === sequence) {
          vizBox.replaceChildren(el("p", "callout callout-danger", `可视化分区加载失败：${error.message}`));
        }
      }
    };

    const renderGroup = data => {
      currentDetail = data;
      state.selectedArms = new Set(data.arms.map(a => a.arm_key));
      state.range = [
        Math.min(...data.arms.map(a => a.window?.start_tick).filter(Number.isFinite)),
        Math.max(...data.arms.map(a => a.window?.end_tick).filter(Number.isFinite)),
      ];
      // 顶栏场景始终是 `${profile}__${seed}`，不从 manifest.episode_id 取
      // 顶栏 tick 用实际 window.fork_tick（后端已验证的 window），无 fork 用 start_tick
      const forkTicks = data.arms.map(a => a.window?.fork_tick).filter(Number.isFinite);
      const patchTick = forkTicks.length ? forkTicks[0] : data.arms[0]?.window?.start_tick;
      if (!Number.isInteger(patchTick)) throw new Error("分支窗口缺少 fork_tick 与 start_tick，无法设置顶栏 tick");
      store.patch({ episodeId: `${data.profile}__${data.seed}`, tick: patchTick });
      // 可视化分区：换组重置为 fork tick，组内切 tick 走增量请求
      const groupKey = `${data.profile}/${data.seed}/${data.group}`;
      const resetTick = vizGroupKey !== groupKey;
      vizGroupKey = groupKey;
      loadViz(data.profile, data.seed, data.group, resetTick ? patchTick : (vizState.tick ?? patchTick), { resetTick });
      body.replaceChildren();
      body.append(groupHeader(data));
      // 默认差异审查：每个干预分支一张卡片，只显示与事实观察基线的差异。
      // 分支选择器放在完整记录折叠区内：筛选只影响完整记录，不作用于差异卡。
      const fullDetails = el("details", "arm-full-records");
      const fullSummary = el("summary", "",
        `完整记录与时间链（分支 ${data.arms.length} 个 · 默认收起，展开查看全量证据）`);
      fullDetails.append(fullSummary);
      const fullBody = el("div", "arm-full-records__body");
      fullBody.append(armPicker(data, state, redraw));
      body.append(buildArmDeltaSection(data, { onPick: showPicked }));
      // 动作对照表：分支选择之后、时间链之前，默认展开。
      actionTableData = data;
      actionTableWrap = el("div", "arm-action-table-wrap");
      fullBody.append(actionTableWrap);
      detailBox = el("div");
      fullBody.append(detailBox);
      redraw();
      // 每臂可展开的存储关系列表（含 X 的 script_dependency）
      const relationsBox = el("div", "arm-arm-relations-list");
      data.arms.forEach(arm => relationsBox.append(armRelationsPanel(arm)));
      fullBody.append(relationsBox);
      // 每臂审计栏，默认折叠
      const auditBox = el("div", "arm-audit-list");
      data.arms.forEach(arm => auditBox.append(auditPanel(arm)));
      fullBody.append(auditBox);
      const links = el("div", "arm-links");
      const cfgLink = el("a", "panel-body", "查看同组干预配置 →");
      cfgLink.href = `#/interventions?profile=${encodeURIComponent(data.profile)}`
        + `&seed=${encodeURIComponent(data.seed)}&group=${encodeURIComponent(data.group)}`;
      links.append(cfgLink);
      fullBody.append(links);
      if (data.errors?.length) {
        const outside = data.errors.filter(e => e.kind === "relation_endpoint_outside_window");
        const realErrors = data.errors.filter(e => e.kind !== "relation_endpoint_outside_window");
        if (realErrors.length) {
          const errBox = panel("后端记录错误", `${realErrors.length} 条 · 窗口外引用 ${outside.length} 条单独列出`);
          realErrors.forEach(e => {
            const item = el("p", "panel-note");
            item.append(el("code", "mono", `${e.arm} · ${e.kind}`), ` ${e.message}`);
            errBox.append(item);
          });
          fullBody.append(errBox);
        }
      }
      fullDetails.append(fullBody);
      body.append(fullDetails);
    };

    const selector = buildGroupSelector(groups, initial, next => {
      // 只写 hash，由 app 的 hashchange 统一重渲染，避免 loadGroup+syncHash 重入
      replaceHashParams("arms", next);
      loadGroupFromHash();
    });
    const selectorPanel = panel("组选择", "家族 / profile / seed / group 联动 · hash 深链接");
    selectorPanel.append(selector.row);
    // 组选择置顶：标题、副标题之后，返回链接 / 可视化分区 / 组内容之前。
    const firstContent = view.querySelector(".arm-viz-box, .arm-review-body, .button-secondary");
    if (firstContent) view.insertBefore(selectorPanel, firstContent);
    else view.append(selectorPanel);

    const loadGroupFromHash = async () => {
      const params2 = hashParams();
      const next = {};
      for (const key of ["profile", "seed", "group"]) next[key] = params2[key] || "";
      const token = ++sequence;
      if (!next.profile || !next.seed || !next.group) {
        selector.setCurrent(initial);
        errorBox.replaceChildren(el("p", "callout callout-danger",
          "深链接参数不完整：需要 profile、seed、group 三个参数。"));
        return;
      }
      const hit = exactGroup(groups, next);
      if (!hit) {
        errorBox.replaceChildren(el("p", "callout callout-danger",
          `未知组合: profile=${next.profile} seed=${next.seed} group=${next.group}`));
        return;
      }
      errorBox.replaceChildren();
      body.replaceChildren(el("p", "loading-indicator", "正在读取该组全部臂的产物…"));
      try {
        const data = await api.armGroup(next.profile, next.seed, next.group);
        if (disposed || token !== sequence) return;
        // 异步返回后检查当前 hash 是否仍是这次请求的组合
        const now = hashParams();
        if (now.profile !== next.profile || now.seed !== next.seed || now.group !== next.group) return;
        selector.setCurrent(next);
        renderGroup(data);
      } catch (error) {
        if (!disposed && token === sequence) {
          errorBox.replaceChildren(el("p", "callout callout-danger", error.message));
        }
      }
    };

    await loadGroupFromHash();
  } catch (error) {
    if (tooltipApi) { tooltipApi.dispose(); tooltipApi = null; }
    if (vizTooltip) { vizTooltip.dispose(); vizTooltip = null; }
    root.replaceChildren(ui.errorPanelDOM(error));
  }
  return () => {
    disposed = true;
    sequence++;
    if (tooltipApi) { tooltipApi.dispose(); tooltipApi = null; }
    if (vizTooltip) { vizTooltip.dispose(); vizTooltip = null; }
    if (resizeObserver) { resizeObserver.disconnect(); resizeObserver = null; }
  };
}
