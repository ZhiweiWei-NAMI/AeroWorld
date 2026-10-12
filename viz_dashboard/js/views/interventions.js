// 干预配置视图（#/interventions）
//
// 与 #/arms 共用同一组深链接参数与同一个选择辅助。展示该组当前配置的各臂列表、
// 实际 mode、窗口、配置来源，以及完整配置文本的折叠详情。只呈现配置里已存在的
// execution/expected/declared mechanism 字段，不推断意图类别。

import {
  MODE_LABEL, el, panel, rawDetails, hashParams, replaceHashParams,
  exactGroup, buildGroupSelector,
} from "../arm-common.js";

function errorPanelDOM(error) {
  const box = el("section", "error-panel");
  box.setAttribute("role", "alert");
  box.append(el("div", "error-panel__title", "数据链路异常"));
  box.append(el("p", "", error?.message || String(error)));
  box.append(el("p", "muted", "系统不会把缺失或读取失败显示为“未发生”。请检查服务终端中的原始错误。"));
  return box;
}

function modeLabel(mode) { return MODE_LABEL[mode] || mode || "未知模式"; }

function factsList(pairs) {
  const node = el("dl", "arm-facts");
  pairs.forEach(([k, v]) => {
    if (v === undefined || v === null) return;
    const row = el("div");
    row.append(el("span", "arm-facts__k", k), el("span", "arm-facts__v mono",
      typeof v === "object" ? JSON.stringify(v) : String(v)));
    node.append(row);
  });
  return node;
}

export async function renderInterventions(ctx) {
  const { root, api, store } = ctx;
  let disposed = false;
  let sequence = 0;
  root.replaceChildren(el("p", "loading-indicator", "正在读取 ARM 目录…"));

  try {
    const params = hashParams();
    const payload = await api.armsCatalog();
    if (disposed) return undefined;
    root.replaceChildren();

    const view = el("div", "view arms-view");
    view.append(el("h1", "view-header", "干预配置"));
    view.append(el("p", "view-subtitle mono",
      `实算 ${payload.counts.arms} 臂 / ${payload.counts.groups} 组 · 每臂配置见 interventions/ 下的对应文件`));

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
      initial = (({ profile, seed, group }) => ({ profile, seed, group }))(groups[0]);
      replaceHashParams("interventions", initial);
    } else {
      throw new Error(
        `深链接参数不完整: 需要 profile、seed、group 三个参数，当前只有 ${explicitKeys.join(", ")}`);
    }

    const body = el("div", "arm-review-body");
    const errorBox = el("div");
    view.append(body, errorBox);

    const renderGroup = data => {
      const forkTicks = data.arms.map(a => a.window?.fork_tick).filter(Number.isFinite);
      const patchTick = forkTicks.length
        ? forkTicks[0]
        : data.arms[0]?.window?.start_tick;
      if (!Number.isInteger(patchTick)) throw new Error("分支窗口缺少 fork_tick 与 start_tick，无法设置顶栏 tick");
      store.patch({ episodeId: `${data.profile}__${data.seed}`, tick: patchTick });
      body.replaceChildren();
      const head = panel(`组 ${data.group}`, `家族 ${data.family} · profile ${data.profile} · seed ${data.seed}`);
      head.append(factsList([
        ["臂数", data.arms.length],
        ["mode", data.arms.map(a => modeLabel(a.mode)).join(" · ")],
        ["窗口", data.arms.map(a => `${a.window?.start_tick}–${a.window?.end_tick}`).join(" / ")],
        ["分叉时刻", [...new Set(data.arms.map(a => a.window?.fork_tick).filter(Number.isFinite))].join(" / ") || "未声明"],
      ]));
      const chainLink = el("a", "panel-body", "查看同组逻辑时间链 →");
      chainLink.href = `#/arms?profile=${encodeURIComponent(data.profile)}`
        + `&seed=${encodeURIComponent(data.seed)}&group=${encodeURIComponent(data.group)}`;
      head.append(chainLink);
      body.append(head);

      data.arms.forEach(arm => {
        const box = panel(`${modeLabel(arm.mode)} · ${arm.arm_id}`, `家族 ${data.family}`);
        const facts = factsList([
          ["arm_key", arm.arm_key],
          ["窗口", arm.window ? `${arm.window.start_tick}–${arm.window.end_tick}` : null],
          ["fork", arm.window?.fork_tick],
          ["phase", arm.window?.phase],
          ["配置来源", arm.configuration?.path],
          ["臂目录", arm.path],
        ]);
        box.append(facts);

        const content = arm.configuration?.content || {};
        if (content.execution !== undefined) {
          box.append(el("h3", "arm-audit__label", "execution（配置声明）"));
          box.append(rawDetails("展开 execution", content.execution));
        }
        if (content.expected !== undefined) {
          box.append(el("h3", "arm-audit__label", "expected（配置声明）"));
          box.append(rawDetails("展开 expected", content.expected));
        }
        const mechanism = content.declared_mechanism ?? content.mechanism
          ?? content.execution?.mechanism;
        if (mechanism !== undefined) {
          box.append(el("h3", "arm-audit__label", "declared mechanism（配置声明）"));
          box.append(rawDetails("展开 mechanism", mechanism));
        }
        if (content.intent !== undefined) {
          box.append(factsList([["intent", content.intent]]));
        }
        if (arm.configuration?.text) {
          box.append(rawDetails("完整配置文本", arm.configuration.text));
        }
        box.append(rawDetails("manifest 原文", arm.manifest?.content));
        body.append(box);
      });

      if (data.errors?.length) {
        const errBox = panel("后端记录错误", `${data.errors.length} 条`);
        data.errors.forEach(e => {
          const item = el("p", "panel-note");
          item.append(el("code", "mono", `${e.arm} · ${e.kind}`), ` ${e.message}`);
          errBox.append(item);
        });
        body.append(errBox);
      }
    };

    const selector = buildGroupSelector(groups, initial, next => {
      replaceHashParams("interventions", next);
      loadGroupFromHash();
    });
    const selectorPanel = panel("组选择", "与逻辑时间链共用同一组参数");
    selectorPanel.append(selector.row);
    view.insertBefore(selectorPanel, body);
    root.append(view);

    const loadGroupFromHash = async () => {
      const params2 = hashParams();
      const next = {};
      for (const key of ["profile", "seed", "group"]) next[key] = params2[key] || "";
      const token = ++sequence;
      if (!next.profile || !next.seed || !next.group) {
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
      body.replaceChildren(el("p", "loading-indicator", "正在读取该组全部臂的配置与实际 mode…"));
      try {
        const data = await api.armGroup(next.profile, next.seed, next.group);
        if (disposed || token !== sequence) return;
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
    root.replaceChildren(errorPanelDOM(error));
  }
  return () => { disposed = true; sequence++; };
}
