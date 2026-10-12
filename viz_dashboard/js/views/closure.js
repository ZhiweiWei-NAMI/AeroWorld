const STATUS_EXPLANATIONS = {
  PASS: "必要阶段均有契约要求的客观证据。",
  FAIL: "至少一个必要阶段被证据否定或未满足契约。",
  UNKNOWN: "必要观测缺失；unknown 不会被折叠为 false。",
  NOT_APPLICABLE: "该叙事阶段明确不在当前可执行语义范围内，不计作证据。",
};

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function present(value) {
  return value !== undefined && value !== null && value !== "";
}

function statusOf(value) {
  const normalized = String(value || "UNKNOWN").toUpperCase();
  return Object.hasOwn(STATUS_EXPLANATIONS, normalized) ? normalized : "UNKNOWN";
}

function renderLoading(root, episodeId) {
  root.innerHTML = `
    <section class="view closure-view" aria-busy="true">
      <header class="view-header">
        <div>
          <p class="eyebrow">EPI ACCEPTANCE CONTRACT</p>
          <h2>EPI 闭合</h2>
          <p class="view-subtitle mono">${episodeId}</p>
        </div>
        <span class="loading-indicator">核对阶段证据…</span>
      </header>
    </section>`;
}

function renderFailure(root, ui, error) {
  root.innerHTML = `
    <section class="view closure-view">
      <header class="view-header">
        <div><p class="eyebrow">EPI ACCEPTANCE CONTRACT</p><h2>EPI 闭合载入失败</h2></div>
      </header>
      <div class="callout callout-danger" role="alert">
        <strong>closure API 未返回可用数据</strong>
        <p>${ui.escape(error instanceof Error ? error.message : String(error))}</p>
      </div>
    </section>`;
}

function renderTags(ui, values, emptyText, className = "chip") {
  const items = asArray(values);
  if (!items.length) return `<span class="muted">${ui.escape(emptyText)}</span>`;
  return `<span class="chip-list">${items
    .map((item) => `<span class="${className}">${ui.escape(String(item))}</span>`)
    .join("")}</span>`;
}

function renderRecordIds(ui, values, emptyText) {
  const ids = asArray(values);
  if (!ids.length) return `<span class="muted">${ui.escape(emptyText)}</span>`;
  return `<ul class="identifier-list">${ids
    .map((id) => `<li><code>${ui.escape(String(id))}</code></li>`)
    .join("")}</ul>`;
}

function formatContractValue(value) {
  if (!present(value)) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function missingFieldIdentity(entry) {
  if (typeof entry === "string") return entry;
  if (!entry || typeof entry !== "object") return String(entry);
  return entry.field || entry.path || entry.state_field || JSON.stringify(entry);
}

function renderMissingFields(ui, values, emptyText) {
  const fields = asArray(values);
  if (!fields.length) {
    return `<div class="evidence-complete"><span aria-hidden="true">✓</span><p>${ui.escape(emptyText)}</p></div>`;
  }
  return `<ul class="missing-field-list">${fields
    .map((entry) => {
      const field = missingFieldIdentity(entry);
      const stage = typeof entry === "object" && entry ? entry.stage_key : undefined;
      const reason = typeof entry === "object" && entry ? entry.reason : undefined;
      const source = typeof entry === "object" && entry ? entry.source : undefined;
      return `<li>
        <code>${ui.escape(field)}</code>
        ${present(stage) ? `<span>阶段 ${ui.escape(String(stage))}</span>` : ""}
        ${present(source) ? `<span>来源 ${ui.escape(String(source))}</span>` : ""}
        ${present(reason) ? `<p>${ui.escape(String(reason))}</p>` : ""}
      </li>`;
    })
    .join("")}</ul>`;
}

function renderClause(ui, result, index) {
  const status = statusOf(result.status);
  const clause = result.clause;
  const matchIds = asArray(result.matching_record_ids);
  return `
    <article class="clause-result clause-${status.toLowerCase()}">
      <header>
        <span>CLAUSE ${String(index + 1).padStart(2, "0")}</span>
        <strong class="status-badge status-${status.toLowerCase()}">${status}</strong>
      </header>
      <pre>${ui.escape(formatContractValue(clause) || "契约子句未返回")}</pre>
      <dl class="fact-list compact">
        <div><dt>判定原因</dt><dd>${ui.escape(result.reason || "未声明")}</dd></div>
        <div><dt>候选匹配</dt><dd>${ui.escape(
          present(result.candidate_match_count) ? String(result.candidate_match_count) : "未声明",
        )}</dd></div>
      </dl>
      <details class="raw-contract"><summary>匹配记录 ${matchIds.length}</summary>${renderRecordIds(ui, matchIds, "无匹配记录")}</details>
    </article>`;
}

function renderStage(ui, stage, arrayIndex, closureMissingFields) {
  const status = statusOf(stage.stage_goal_status);
  const stageIndex = present(stage.stage_index) ? stage.stage_index : arrayIndex;
  const stageKey = stage.stage_key || `stage-${stageIndex}`;
  const required = asArray(stage.required_evidence_types);
  const observed = asArray(stage.observed_evidence_types);
  const missingEvidence = asArray(stage.missing_evidence_types);
  const combinedMissingFields = [
    ...asArray(stage.missing_fields),
    ...asArray(closureMissingFields).filter(
      (entry) => entry && typeof entry === "object" && entry.stage_key === stage.stage_key,
    ),
  ];
  const missingFields = [
    ...new Map(combinedMissingFields.map((entry) => [JSON.stringify(entry), entry])).values(),
  ];
  const clauses = asArray(stage.clause_results);
  const ticks = asArray(stage.supporting_ticks);
  const supportingIds = asArray(stage.supporting_record_ids);
  const selectedEvents = asArray(stage.selected_event_ids);
  const physicalTick = present(stage.supporting_tick)
    ? String(stage.supporting_tick)
    : ticks.length
      ? ticks.join(" · ")
      : "无物理 tick";

  return `
    <article class="closure-stage stage-${status.toLowerCase()}" data-stage-status="${status}">
      <div class="stage-rail" aria-hidden="true"><span>${String(Number(stageIndex) + 1).padStart(2, "0")}</span><i></i></div>
      <div class="stage-content">
        <header class="stage-header">
          <div>
            <p class="stage-key mono">${ui.escape(String(stageKey))}</p>
            <h3>${ui.escape(stage.stage_goal || "未声明阶段目标")}</h3>
            <p>${ui.escape(stage.event_family_id || stage.stage_kind || "未声明事件族")}</p>
          </div>
          <div class="stage-verdict">
            <strong class="status-badge status-${status.toLowerCase()}">${status}</strong>
            <span>${ui.escape(stage.implementation_status || "未声明实现状态")}</span>
          </div>
        </header>

        <div class="stage-evidence-strip">
          <div><span>证据 tick</span><strong>${ui.escape(physicalTick)}</strong></div>
          <div><span>required</span><strong>${required.length}</strong></div>
          <div><span>observed</span><strong>${observed.length}</strong></div>
          <div class="${missingEvidence.length || missingFields.length ? "has-gap" : "no-gap"}">
            <span>missing</span><strong>${missingEvidence.length + missingFields.length}</strong>
          </div>
        </div>

        <dl class="fact-list stage-summary">
          <div><dt>判定原因</dt><dd>${ui.escape(stage.status_reason || "未声明")}</dd></div>
          <div><dt>阶段类型</dt><dd>${ui.escape(stage.stage_kind || "未声明")}</dd></div>
          <div><dt>时间解释</dt><dd>${ui.escape(stage.temporal_interpretation || "未声明")}</dd></div>
        </dl>

        <details class="stage-details" ${status !== "PASS" ? "open" : ""}>
          <summary>查看阶段证据与缺口</summary>
          <div class="stage-details-grid">
            <section>
              <h4>必要证据类型</h4>
              ${renderTags(ui, required, "未声明必要证据")}
            </section>
            <section>
              <h4>实际观测证据</h4>
              ${renderTags(ui, observed, "没有观测证据")}
            </section>
            <section class="stage-gaps">
              <h4>缺失证据类型</h4>
              ${renderTags(ui, missingEvidence, "无缺失证据类型", "chip chip-danger")}
              <h4>缺失状态字段</h4>
              ${renderMissingFields(ui, missingFields, "本阶段无缺失状态字段")}
            </section>
            <section>
              <h4>支撑记录</h4>
              ${renderRecordIds(ui, supportingIds, "没有支撑记录 ID")}
              <h4>选中事件</h4>
              ${renderRecordIds(ui, selectedEvents, "本阶段未选择事件")}
            </section>
          </div>

          ${
            asArray(stage.binding_sets).length
              ? `<section class="binding-evidence"><h4>对象绑定</h4><pre>${ui.escape(
                  JSON.stringify(stage.binding_sets, null, 2),
                )}</pre></section>`
              : ""
          }
          ${
            present(stage.closure_statement) || present(stage.not_applicable_reason)
              ? `<div class="callout callout-info"><strong>闭合声明</strong><p>${ui.escape(
                  stage.closure_statement || stage.not_applicable_reason,
                )}</p></div>`
              : ""
          }
          ${
            clauses.length
              ? `<section class="clause-results"><h4>契约子句判定</h4>${clauses
                  .map((result, index) => renderClause(ui, result, index))
                  .join("")}</section>`
              : `<p class="muted">本阶段未返回 clause_results</p>`
          }
        </details>
      </div>
    </article>`;
}

function statusCounts(closure, stages) {
  const declared = closure.stage_status_counts;
  if (declared && typeof declared === "object" && !Array.isArray(declared)) return declared;
  return stages.reduce((counts, stage) => {
    const status = statusOf(stage.stage_goal_status);
    counts[status] = (counts[status] || 0) + 1;
    return counts;
  }, {});
}

export async function renderClosure(ctx) {
  const { root, api, store, ui } = ctx;
  const episodeId = store.episodeId;
  if (!episodeId) {
    root.innerHTML = ui.empty("请先选择 Episode，再核验 EPI 闭合状态。");
    return;
  }

  renderLoading(root, ui.escape(String(episodeId)));
  try {
    const closure = await api.closure(episodeId);
    const stages = asArray(closure.stages);
    const status = statusOf(closure.status);
    const counts = statusCounts(closure, stages);
    const missingFields = asArray(closure.missing_fields);
    const motivation = formatContractValue(closure.motivation);

    root.innerHTML = `
      <section class="view closure-view">
        <header class="view-header closure-header">
          <div>
            <p class="eyebrow">EPI ACCEPTANCE CONTRACT · OBJECTIVE EVIDENCE</p>
            <h2>EPI 闭合</h2>
            <p class="view-subtitle"><code>${ui.escape(String(closure.epi_id || "未声明 EPI"))}</code> · <code>${ui.escape(
              String(episodeId),
            )}</code></p>
          </div>
          <div class="closure-verdict status-${status.toLowerCase()}">
            <span>FINAL VERDICT</span>
            <strong>${status}</strong>
            <p>${STATUS_EXPLANATIONS[status]}</p>
          </div>
        </header>

        <div class="closure-overview-grid">
          <section class="panel motivation-panel">
            <p class="eyebrow">WHY · EPI 动机</p>
            <h3>验收目标</h3>
            ${
              motivation
                ? `<p>${ui.escape(motivation)}</p>`
                : `<p class="declaration-gap">闭合产物未提供 motivation 字段；不根据 EPI 名称臆测动机。</p>`
            }
          </section>
          <section class="panel stage-count-panel">
            <p class="eyebrow">STAGE MATRIX</p>
            <h3>${stages.length} 个必要阶段</h3>
            <div class="status-counts">
              <span class="status-pass"><b>${counts.PASS || 0}</b>PASS</span>
              <span class="status-fail"><b>${counts.FAIL || 0}</b>FAIL</span>
              <span class="status-unknown"><b>${counts.UNKNOWN || 0}</b>UNKNOWN</span>
              <span class="status-not_applicable"><b>${counts.NOT_APPLICABLE || 0}</b>N/A</span>
            </div>
          </section>
          <section class="panel missing-fields-panel ${missingFields.length ? "has-gap" : "no-gap"}">
            <p class="eyebrow">MISSING STATE</p>
            <h3>具体缺失字段</h3>
            ${renderMissingFields(ui, missingFields, "闭合 API 报告无缺失状态字段")}
          </section>
        </div>

        <section class="closure-sequence">
          <header class="section-header">
            <div><p class="eyebrow">REQUIRED → OBSERVED → VERDICT</p><h3>阶段证据链</h3></div>
            <span>严格按 closure.stages 顺序呈现</span>
          </header>
          <div class="closure-stages">
            ${
              stages.length
                ? stages.map((stage, index) => renderStage(ui, stage, index, missingFields)).join("")
                : ui.empty("闭合记录没有 stages")
            }
          </div>
        </section>

        <footer class="closure-provenance panel">
          <dl class="fact-list compact">
            <div><dt>schema</dt><dd><code>${ui.escape(closure.schema_name || "未声明")}</code></dd></div>
            <div><dt>schema version</dt><dd><code>${ui.escape(closure.schema_version || "未声明")}</code></dd></div>
            <div><dt>stage contract</dt><dd><code>${ui.escape(closure.stage_contract_profile_id || "未声明")}</code></dd></div>
            <div><dt>contract digest</dt><dd><code>${ui.escape(closure.stage_contract_digest || "未声明")}</code></dd></div>
          </dl>
        </footer>
      </section>`;
  } catch (error) {
    renderFailure(root, ui, error);
  }
}
