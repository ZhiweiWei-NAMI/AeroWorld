const OPERATION_MODE_PREFIX = "operation.mode.";

function renderLoading(root, title) {
  root.innerHTML = `
    <section class="view catalog-view" aria-busy="true">
      <header class="view-header">
        <div>
          <p class="eyebrow">ONTOLOGY REGISTRY</p>
          <h2>${title}</h2>
        </div>
        <span class="loading-indicator">读取权威 registry…</span>
      </header>
    </section>`;
}

function renderFailure(root, ui, error) {
  root.innerHTML = `
    <section class="view catalog-view">
      <header class="view-header">
        <div>
          <p class="eyebrow">ONTOLOGY REGISTRY</p>
          <h2>目录载入失败</h2>
        </div>
      </header>
      <div class="callout callout-danger" role="alert">
        <strong>registry API 未返回可用数据</strong>
        <p>${ui.escape(error instanceof Error ? error.message : String(error))}</p>
      </div>
    </section>`;
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function unique(values) {
  return [...new Set(values.filter((value) => value !== undefined && value !== null && value !== ""))];
}

function compactJson(value) {
  return JSON.stringify(value, null, 2);
}

function conditionText(condition) {
  if (condition === undefined || condition === null || condition === "") return "";
  if (typeof condition === "string") {
    const trimmed = condition.trim();
    if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
      try {
        return conditionText(JSON.parse(trimmed));
      } catch {
        return condition;
      }
    }
    return condition;
  }
  if (typeof condition !== "object") return String(condition);
  if (Array.isArray(condition)) return condition.map(conditionText).filter(Boolean).join(" AND ");
  if (Array.isArray(condition.any_of)) {
    return condition.any_of.map((item) => `(${conditionText(item)})`).join(" OR ");
  }

  const parts = [];
  if (condition.predicate_id) parts.push(condition.predicate_id);
  if (condition.direction) parts.push(`direction=${condition.direction}`);
  if (condition.value !== undefined) parts.push(`value=${String(condition.value)}`);
  if (condition.at) parts.push(`at=${condition.at}`);
  if (condition.event_tick) parts.push(`event_tick=${condition.event_tick}`);
  if (condition.hold_samples !== undefined) parts.push(`hold_samples=${condition.hold_samples}`);
  if (condition.hold_parameter) parts.push(`hold=${condition.hold_parameter}`);
  return parts.length ? parts.join(" · ") : compactJson(condition);
}

function fieldLabel(field) {
  if (typeof field === "string") return field;
  if (!field || typeof field !== "object") return String(field);
  const name = field.field || "未命名字段";
  const source = field.source ? ` ← ${field.source}` : "";
  const unit = field.unit ? ` [${field.unit}]` : "";
  return `${name}${unit}${source}`;
}

function predicateId(predicate) {
  return String(predicate.api_id);
}

function canonicalPredicateRows(predicates) {
  const parameterized = predicates.find(
    (predicate) => predicateId(predicate) === `${OPERATION_MODE_PREFIX}{mode}`,
  );
  const concreteModes = predicates.filter((predicate) => {
    const id = predicateId(predicate);
    return id.startsWith(OPERATION_MODE_PREFIX) && id !== `${OPERATION_MODE_PREFIX}{mode}`;
  });
  if (concreteModes.length) {
    throw new Error("catalog API 必须只返回 operation.mode.{mode} 参数化模板");
  }
  return { rows: predicates.slice(), hasParameterizedMode: Boolean(parameterized) };
}

function renderChipList(ui, values, emptyText) {
  const items = asArray(values);
  if (!items.length) return `<span class="muted">${ui.escape(emptyText)}</span>`;
  return `<span class="chip-list">${items
    .map((item) => `<span class="chip">${ui.escape(String(item))}</span>`)
    .join("")}</span>`;
}

function renderPredicateRow(ui, predicate, truthValues) {
  const id = predicateId(predicate);
  const sources = asArray(predicate.sources);
  const unknownFields = asArray(predicate.unknown_policy?.unknown_if_missing);
  const thresholds = asArray(predicate.threshold_parameters);
  const scopes = asArray(predicate.scope_types);
  const roles = asArray(predicate.argument_roles);
  const modes = asArray(predicate.mode_values);
  const search = [
    id,
    predicate.binding_kind,
    predicate.ontology_module,
    predicate.expression,
    ...sources.map(fieldLabel),
    ...scopes,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  return `
    <details class="catalog-record predicate-record"
      data-catalog-record
      data-module="${ui.escape(predicate.ontology_module || "未分类")}"
      data-scope="${ui.escape(scopes.join(" "))}"
      data-search="${ui.escape(search)}">
      <summary>
        <span class="record-id mono">${ui.escape(id)}</span>
        <span class="record-kind">${ui.escape(predicate.binding_kind || "未声明 tuple")}</span>
        <span class="record-expression mono">${ui.escape(predicate.expression || "权威 registry 未声明表达式")}</span>
        <span class="record-disclosure" aria-hidden="true">⌄</span>
      </summary>
      <div class="catalog-record-body">
        <section>
          <h4>语义与 tuple</h4>
          <dl class="fact-list">
            <div><dt>本体模块</dt><dd>${ui.escape(predicate.ontology_module || "未声明")}</dd></div>
            <div><dt>作用域</dt><dd>${renderChipList(ui, scopes, "未声明")}</dd></div>
            <div><dt>参数角色</dt><dd>${renderChipList(
              ui,
              roles.map((role) => {
                if (typeof role === "string") return role;
                if (role.role) return role.role;
                if (role.key) return role.class ? `${role.key}: ${role.class}` : role.key;
                return compactJson(role);
              }),
              "无参数角色",
            )}</dd></div>
            ${modes.length ? `<div><dt>mode 参数域</dt><dd>${renderChipList(ui, modes, "无")}</dd></div>` : ""}
          </dl>
        </section>
        <section>
          <h4>输入字段</h4>
          ${
            sources.length
              ? `<ul class="source-field-list">${sources
                  .map((field) => `<li><code>${ui.escape(fieldLabel(field))}</code></li>`)
                  .join("")}</ul>`
              : `<p class="muted">权威 registry 未声明输入字段</p>`
          }
        </section>
        <section>
          <h4>四值规则</h4>
          <p class="truth-domain">${truthValues.map((value) => ui.truthBadge(value)).join(" ")}</p>
          <dl class="fact-list">
            <div><dt>求值规则</dt><dd><code>${ui.escape(conditionText(predicate.rule) || "未声明")}</code></dd></div>
            <div><dt>unknown 条件</dt><dd>${renderChipList(ui, unknownFields, "未声明")}</dd></div>
            <div><dt>缺记录标记</dt><dd><code>${ui.escape(predicate.unknown_policy?.missing_record_field || "未声明")}</code></dd></div>
            <div><dt>out_of_scope</dt><dd>绑定对象不属于声明作用域时，不折叠为 false</dd></div>
          </dl>
        </section>
        <section>
          <h4>阈值参数</h4>
          ${renderChipList(ui, thresholds, "无阈值参数")}
          ${
            asArray(predicate.mode_expressions).length > 1
              ? `<details class="raw-contract"><summary>各 mode 原始表达式</summary><pre>${ui.escape(
                  compactJson(predicate.mode_expressions),
                )}</pre></details>`
              : ""
          }
        </section>
      </div>
    </details>`;
}

function bindCatalogFilters(root) {
  const search = root.querySelector("[data-catalog-search]");
  const primary = root.querySelector("[data-catalog-primary]");
  const secondary = root.querySelector("[data-catalog-secondary]");
  const count = root.querySelector("[data-visible-count]");
  const records = [...root.querySelectorAll("[data-catalog-record]")];

  const update = () => {
    const term = search.value.trim().toLowerCase();
    let visible = 0;
    for (const record of records) {
      const matchesTerm = !term || record.dataset.search.includes(term);
      const primaryValue = record.dataset.module ?? record.dataset.family ?? "";
      const scopeValues = (record.dataset.scope ?? "").split(" ");
      const matchesPrimary = !primary.value || primaryValue === primary.value;
      const matchesSecondary = !secondary.value || scopeValues.includes(secondary.value) || record.dataset.mapping === secondary.value;
      const show = matchesTerm && matchesPrimary && matchesSecondary;
      record.hidden = !show;
      if (show) visible += 1;
    }
    count.textContent = String(visible);
  };

  search.addEventListener("input", update);
  primary.addEventListener("change", update);
  secondary.addEventListener("change", update);
}

export async function renderPredicateCatalog(ctx) {
  const { root, api, ui } = ctx;
  renderLoading(root, "谓词目录");
  try {
    const catalog = await api.catalogs();
    const predicates = asArray(catalog.predicates);
    const truthValues = asArray(catalog.truth_values);
    const canonical = canonicalPredicateRows(predicates);
    const rows = canonical.rows;
    const modules = unique(rows.map((predicate) => predicate.ontology_module || "未分类")).sort();
    const scopes = unique(rows.flatMap((predicate) => asArray(predicate.scope_types))).sort();
    const registryCount = catalog.counts?.registry_predicates ?? predicates.length;
    const modeNote = canonical.hasParameterizedMode
      ? "API 已返回 operation.mode.{mode} 参数化模板，未重复展开具体模式"
      : "权威 registry 未声明 operation.mode.* 模板，界面未生成虚构条目";

    root.innerHTML = `
      <section class="view catalog-view predicate-catalog-view">
        <header class="view-header">
          <div>
            <p class="eyebrow">ONTOLOGY REGISTRY · FOUR-VALUED LOGIC</p>
            <h2>谓词目录</h2>
            <p class="view-subtitle">直接投影 Dataset 权威 registry；缺失观测保持 unknown，不与 false 混同。</p>
          </div>
          <div class="view-metrics" aria-label="谓词统计">
            <div><strong>${ui.escape(String(registryCount))}</strong><span>registry 模板</span></div>
            <div><strong data-visible-count>${rows.length}</strong><span>当前显示</span></div>
          </div>
        </header>

        <div class="callout callout-info mode-collapse-note">
          <strong>参数化模板检查</strong>
          <span>${ui.escape(modeNote)}</span>
        </div>

        <div class="catalog-toolbar" role="search">
          <label class="search-field">
            <span>检索 ID / 表达式 / 输入字段</span>
            <input data-catalog-search type="search" placeholder="例如 aircraft_airborne / z_agl_m" autocomplete="off">
          </label>
          <label>
            <span>本体模块</span>
            <select data-catalog-primary>
              <option value="">全部模块</option>
              ${modules.map((module) => `<option value="${ui.escape(module)}">${ui.escape(module)}</option>`).join("")}
            </select>
          </label>
          <label>
            <span>作用域</span>
            <select data-catalog-secondary>
              <option value="">全部作用域</option>
              ${scopes.map((scope) => `<option value="${ui.escape(scope)}">${ui.escape(scope)}</option>`).join("")}
            </select>
          </label>
        </div>

        <div class="catalog-column-head" aria-hidden="true">
          <span>谓词 API ID</span><span>tuple 类型</span><span>计算表达式</span><span></span>
        </div>
        <div class="catalog-records">
          ${
            rows.length
              ? rows.map((predicate) => renderPredicateRow(ui, predicate, truthValues)).join("")
              : ui.empty("registry 中没有谓词模板")
          }
        </div>
      </section>`;
    bindCatalogFilters(root);
  } catch (error) {
    renderFailure(root, ui, error);
  }
}

function eventTrigger(rule) {
  const base = {
    predicate_id: rule.trigger_predicate_id,
    direction: rule.trigger_direction,
    event_tick: rule.event_tick,
  };
  const supports = asArray(rule.support_conditions);
  return {
    base: conditionText(base),
    supports: supports.map(conditionText),
  };
}

function declaredBlock(ui, value, undeclaredText) {
  const text = conditionText(value);
  if (!text) return `<span class="declaration-gap">${ui.escape(undeclaredText)}</span>`;
  return `<code>${ui.escape(text)}</code>`;
}

function renderEventRule(ui, rule) {
  const trigger = eventTrigger(rule);
  const response = rule.required_response;
  const termination = rule.terminal_condition ?? rule.terminal;
  const failure = rule.failure_condition;
  const family = String(rule.event_family_id);
  const search = [
    rule.rule_id,
    family,
    rule.event_type,
    rule.mapping_class,
    trigger.base,
    conditionText(response),
    conditionText(termination),
    conditionText(failure),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  return `
    <details class="catalog-record event-rule-record"
      data-catalog-record
      data-family="${ui.escape(family)}"
      data-mapping="${ui.escape(rule.mapping_class || "未声明")}"
      data-search="${ui.escape(search)}">
      <summary>
        <span class="record-id mono">${ui.escape(rule.event_type || "未声明 event type")}</span>
        <span class="record-kind">${ui.escape(family)}</span>
        <span class="record-expression mono">${ui.escape(trigger.base || "权威 registry 未声明触发条件")}</span>
        <span class="record-disclosure" aria-hidden="true">⌄</span>
      </summary>
      <div class="event-contract-grid">
        <section class="contract-block trigger-contract">
          <p class="contract-index">01 · TRIGGER</p>
          <h4>触发条件</h4>
          ${declaredBlock(ui, trigger.base, "权威 registry 未声明触发条件")}
          ${
            trigger.supports.length
              ? `<div class="support-conditions"><span>附加守卫</span>${trigger.supports
                  .map((support) => `<code>${ui.escape(support)}</code>`)
                  .join("")}</div>`
              : `<p class="muted">无附加守卫</p>`
          }
        </section>
        <section class="contract-block response-contract">
          <p class="contract-index">02 · RESPONSE</p>
          <h4>响应条件</h4>
          ${declaredBlock(ui, response, "权威 registry 未声明响应条件")}
        </section>
        <section class="contract-block terminal-contract">
          <p class="contract-index">03 · TERMINATE</p>
          <h4>终止条件</h4>
          ${declaredBlock(ui, termination, "权威 registry 未声明终止条件")}
        </section>
        <section class="contract-block failure-contract">
          <p class="contract-index">04 · FAILURE</p>
          <h4>失败条件</h4>
          ${declaredBlock(ui, failure, "权威 registry 未声明失败条件")}
        </section>
      </div>
      <footer class="catalog-record-footer">
        <span><b>规则</b> <code>${ui.escape(rule.rule_id || "未声明")}</code></span>
        <span><b>映射</b> ${ui.escape(rule.mapping_class || "未声明")}</span>
        <span><b>参与角色</b> ${ui.escape(
          asArray(rule.participant_roles)
            .map((role) => (typeof role === "string" ? role : role.role))
            .filter(Boolean)
            .join(" · ") || "未声明",
        )}</span>
      </footer>
    </details>`;
}

export async function renderEventCatalog(ctx) {
  const { root, api, ui } = ctx;
  renderLoading(root, "事件规则目录");
  try {
    const catalog = await api.catalogs();
    const rules = asArray(catalog.event_rules);
    const families = unique(rules.map((rule) => rule.event_family_id)).sort();
    const mappingClasses = unique(rules.map((rule) => rule.mapping_class)).sort();
    const uniqueEventTypes = unique(rules.map((rule) => rule.event_type));
    const declaredFailures = rules.filter((rule) => conditionText(rule.failure_condition)).length;

    root.innerHTML = `
      <section class="view catalog-view event-catalog-view">
        <header class="view-header">
          <div>
            <p class="eyebrow">OBJECTIVE EVENT CONTRACTS</p>
            <h2>事件规则目录</h2>
            <p class="view-subtitle">事件数量取自当前 registry；触发、响应、终止与失败声明分栏呈现。</p>
          </div>
          <div class="view-metrics" aria-label="事件规则统计">
            <div><strong>${rules.length}</strong><span>occurrence 规则</span></div>
            <div><strong>${uniqueEventTypes.length}</strong><span>唯一 type</span></div>
            <div><strong data-visible-count>${rules.length}</strong><span>当前显示</span></div>
          </div>
        </header>

        <div class="callout ${declaredFailures === rules.length ? "callout-info" : "callout-warning"}">
          <strong>声明完整性</strong>
          <span>失败条件已声明 ${declaredFailures}/${rules.length}；空缺代表 registry 未声明，不代表失败未发生。</span>
        </div>

        <div class="catalog-toolbar" role="search">
          <label class="search-field">
            <span>检索 type / family / predicate</span>
            <input data-catalog-search type="search" placeholder="例如 c2_link_loss / communication" autocomplete="off">
          </label>
          <label>
            <span>事件族</span>
            <select data-catalog-primary>
              <option value="">全部事件族</option>
              ${families.map((family) => `<option value="${ui.escape(family)}">${ui.escape(family)}</option>`).join("")}
            </select>
          </label>
          <label>
            <span>映射类别</span>
            <select data-catalog-secondary>
              <option value="">全部映射</option>
              ${mappingClasses.map((mapping) => `<option value="${ui.escape(mapping)}">${ui.escape(mapping)}</option>`).join("")}
            </select>
          </label>
        </div>

        <div class="catalog-column-head" aria-hidden="true">
          <span>事件 type ID</span><span>事件族</span><span>触发条件</span><span></span>
        </div>
        <div class="catalog-records">
          ${rules.length ? rules.map((rule) => renderEventRule(ui, rule)).join("") : ui.empty("registry 中没有事件规则")}
        </div>
      </section>`;
    bindCatalogFilters(root);
  } catch (error) {
    renderFailure(root, ui, error);
  }
}
