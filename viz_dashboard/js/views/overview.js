const NUMBER_FORMAT = new Intl.NumberFormat('zh-CN');
const PERCENT_FORMAT = new Intl.NumberFormat('zh-CN', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const TRUTH_ORDER = [
  { key: 'true', label: 'TRUE', detail: '证据充分且命题成立' },
  { key: 'false', label: 'FALSE', detail: '证据充分且命题不成立' },
  { key: 'out_of_scope', label: 'OUT OF SCOPE', detail: '对象不属于该谓词适用域' },
  { key: 'unknown', label: 'UNKNOWN', detail: '适用但权威输入不足' },
];

function hasNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function formatCount(value) {
  return hasNumber(value) ? NUMBER_FORMAT.format(value) : '—';
}

function formatDate(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? String(value) : date.toLocaleString('zh-CN', { hour12: false });
}

function statusTone(status) {
  if (status === 'PASS' || status === 'ready' || status === 'complete') return 'ok';
  if (status === 'FAIL' || status === 'failed' || status === 'error') return 'danger';
  if (status === 'running' || status === 'building') return 'active';
  return 'muted';
}

function renderLoading(root) {
  root.innerHTML = `
    <section class="view view-overview" aria-busy="true">
      <div class="view-loading">
        <span class="loading-pulse" aria-hidden="true"></span>
        <span>正在读取语料级精简汇总…</span>
      </div>
    </section>`;
}

function renderFailure(root, escape, error) {
  root.innerHTML = `
    <section class="view view-overview">
      <header class="view-header">
        <div><p class="eyebrow">CORPUS CONTROL</p><h1>语料总览</h1></div>
      </header>
      <div class="panel error-panel" role="alert">
        <strong>汇总接口读取失败</strong>
        <p>${escape(error instanceof Error ? error.message : String(error))}</p>
      </div>
    </section>`;
}

function metricCard(label, value, suffix, detail, tone = '') {
  return `
    <article class="metric-card ${tone ? `metric-card--${tone}` : ''}">
      <span class="metric-label">${label}</span>
      <div class="metric-value">${value}${suffix ? `<small>${suffix}</small>` : ''}</div>
      <span class="metric-detail">${detail}</span>
    </article>`;
}

function renderTruthDistribution(distribution) {
  const total = TRUTH_ORDER.reduce(
    (sum, item) => sum + (hasNumber(distribution[item.key]) ? distribution[item.key] : 0),
    0,
  );

  const segments = TRUTH_ORDER.map((item) => {
    const value = distribution[item.key];
    const width = hasNumber(value) && total > 0 ? (value / total) * 100 : 0;
    return `<span class="truth-bar__segment truth-${item.key}" style="width:${width}%" title="${item.label} ${PERCENT_FORMAT.format(width)}%"></span>`;
  }).join('');

  const rows = TRUTH_ORDER.map((item) => {
    const value = distribution[item.key];
    const ratio = hasNumber(value) && total > 0 ? `${PERCENT_FORMAT.format((value / total) * 100)}%` : '—';
    return `
      <li class="truth-stat truth-stat--${item.key}">
        <span class="truth-dot" aria-hidden="true"></span>
        <span class="truth-stat__name">${item.label}<small>${item.detail}</small></span>
        <strong>${formatCount(value)}</strong>
        <span>${ratio}</span>
      </li>`;
  }).join('');

  return `
    <div class="truth-distribution">
      <div class="truth-bar" aria-label="四值语义分布">${segments}</div>
      <ul class="truth-stat-list">${rows}</ul>
      <footer>矩阵单元总数 <strong>${formatCount(total)}</strong></footer>
    </div>`;
}

function renderClosureCounts(closureCounts, episodeCount) {
  const statuses = ['PASS', 'FAIL', 'UNKNOWN'];
  return statuses.map((status) => {
    const count = closureCounts[status];
    const ratio = hasNumber(count) && hasNumber(episodeCount) && episodeCount > 0
      ? `${PERCENT_FORMAT.format((count / episodeCount) * 100)}%`
      : '—';
    return `
      <div class="status-row">
        <span class="status-badge status-badge--${statusTone(status)}">${status}</span>
        <strong>${formatCount(count)}</strong>
        <span>${ratio}</span>
      </div>`;
  }).join('');
}

function renderEventFamilies(eventFamilies, escape) {
  if (!eventFamilies.length) {
    return '<li class="empty-inline">汇总中未返回事件族。</li>';
  }

  const maximum = Math.max(...eventFamilies.map((family) => hasNumber(family.count) ? family.count : 0), 1);
  return eventFamilies.map((family) => {
    const width = hasNumber(family.count) ? (family.count / maximum) * 100 : 0;
    return `
      <li class="family-row">
        <span title="${escape(family.id)}">${escape(family.id)}</span>
        <div class="family-bar"><i style="width:${width}%"></i></div>
        <strong>${formatCount(family.count)}</strong>
      </li>`;
  }).join('');
}

export async function renderOverview(ctx) {
  const { root, api, ui } = ctx;
  const escape = (value) => ui.escape(String(value ?? ''));
  renderLoading(root);

  let summary;
  try {
    summary = await api.summary();
  } catch (error) {
    renderFailure(root, escape, error);
    return;
  }

  const { corpus, truth_distribution: truthDistribution, generation, anomaly_count: anomalyCount, event_families: eventFamilies } = summary;
  const generationProgress = hasNumber(generation.completed_episodes) && hasNumber(generation.total_episodes) && generation.total_episodes > 0
    ? (generation.completed_episodes / generation.total_episodes) * 100
    : 0;
  const tickContract = hasNumber(corpus.tick_start) && hasNumber(corpus.tick_end) && hasNumber(corpus.tick_step)
    ? `tick ${corpus.tick_start}–${corpus.tick_end} / 步长 ${corpus.tick_step}`
    : 'tick 契约未返回';

  root.innerHTML = `
    <section class="view view-overview">
      <header class="view-header overview-header">
        <div>
          <p class="eyebrow">CORPUS CONTROL / OBJECTIVE SEMANTIC TRUTH</p>
          <h1>低空时序语义真值系统</h1>
          <p>语料级质量与四值语义分布；所有数字来自服务端对 210 个单集小型汇总文件的按需聚合。</p>
        </div>
        <div class="contract-chip"><span>FORMAL CONTRACT</span><strong>${escape(tickContract)}</strong></div>
      </header>

      <div class="metric-grid overview-metrics">
        ${metricCard('EPI', formatCount(corpus.epi_count), '个', '独立语义动机与阶段契约', 'cyan')}
        ${metricCard('EPISODE', formatCount(corpus.episode_count), '集', 'EPI × seed00/01/02', 'blue')}
        ${metricCard('谓词模板', formatCount(corpus.predicate_template_count), '个', '参数化 operation.mode 只计一个模板', 'violet')}
        ${metricCard('事件类型', formatCount(corpus.event_type_count), '个', `${formatCount(eventFamilies.length)} 个事件族`, 'amber')}
        ${metricCard('异常记录', formatCount(anomalyCount), '条', '显式质量异常，不与 unknown 混合', hasNumber(anomalyCount) && anomalyCount > 0 ? 'danger' : 'ok')}
      </div>

      <div class="overview-layout">
        <article class="panel panel--truth">
          <header class="panel-header">
            <div><p class="panel-kicker">FOUR-VALUED MATRIX</p><h2>四值语义分布</h2></div>
            <span class="panel-note">unknown ≠ false · out_of_scope ≠ 未发生</span>
          </header>
          ${renderTruthDistribution(truthDistribution)}
        </article>

        <article class="panel panel--generation">
          <header class="panel-header">
            <div><p class="panel-kicker">PIPELINE HEALTH</p><h2>生成与校验</h2></div>
            <span class="status-badge status-badge--${statusTone(generation.status)}">${escape(generation.status)}</span>
          </header>
          <div class="generation-gauge">
            <div class="gauge-value">${formatCount(generation.completed_episodes)}<small> / ${formatCount(generation.total_episodes)}</small></div>
            <div class="progress-track" aria-label="生成进度"><i style="width:${generationProgress}%"></i></div>
            <span>${PERCENT_FORMAT.format(generationProgress)}%</span>
          </div>
          <dl class="definition-grid compact">
            <div><dt>校验状态</dt><dd>${escape(generation.validation_status)}</dd></div>
            <div><dt>最近更新</dt><dd>${escape(formatDate(generation.updated_at))}</dd></div>
          </dl>
          <button class="button button--secondary button--full" type="button" data-route="generate">查看生成与打包接口</button>
        </article>

        <article class="panel panel--closure">
          <header class="panel-header">
            <div><p class="panel-kicker">EPI ACCEPTANCE</p><h2>闭合状态</h2></div>
            <button class="text-button" type="button" data-route="closure">查看阶段证据 →</button>
          </header>
          <div class="status-stack">${renderClosureCounts(corpus.closure_status_counts, corpus.episode_count)}</div>
          <p class="panel-footnote">闭合仅由阶段契约与实际证据评定，展示层不使用 EPI 名称辅助检测。</p>
        </article>

        <article class="panel panel--families">
          <header class="panel-header">
            <div><p class="panel-kicker">OBSERVED EVENT FAMILIES</p><h2>事件族活性</h2></div>
            <strong>${formatCount(eventFamilies.length)} 族</strong>
          </header>
          <ol class="family-list">${renderEventFamilies(eventFamilies, escape)}</ol>
        </article>
      </div>
    </section>`;

  root.querySelectorAll('[data-route]').forEach((button) => {
    button.addEventListener('click', () => ctx.navigate(button.dataset.route));
  });
}
