const NUMBER_FORMAT = new Intl.NumberFormat('zh-CN');

const EMPTY_FILTERS = Object.freeze({
  epi: '',
  seed: '',
  event_family: '',
  closure: '',
});

const TRUTH_KEYS = [
  ['true', 'T'],
  ['false', 'F'],
  ['out_of_scope', 'OOS'],
  ['unknown', 'UNK'],
];

function hasNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function formatCount(value) {
  return hasNumber(value) ? NUMBER_FORMAT.format(value) : '—';
}

function closureTone(status) {
  if (status === 'PASS') return 'ok';
  if (status === 'FAIL') return 'danger';
  return 'muted';
}

function renderLoading(root) {
  root.innerHTML = `
    <section class="view view-episodes" aria-busy="true">
      <div class="view-loading">
        <span class="loading-pulse" aria-hidden="true"></span>
        <span>正在加载 Episode 索引…</span>
      </div>
    </section>`;
}

function renderFailure(root, escape, error) {
  root.innerHTML = `
    <section class="view view-episodes">
      <header class="view-header"><div><p class="eyebrow">EPISODE INDEX</p><h1>Episode 选择</h1></div></header>
      <div class="panel error-panel" role="alert">
        <strong>Episode 索引读取失败</strong>
        <p>${escape(error instanceof Error ? error.message : String(error))}</p>
      </div>
    </section>`;
}

function renderOptions(facets, selected, escape) {
  return facets.map((facet) => `
    <option value="${escape(facet.value)}" ${facet.value === selected ? 'selected' : ''}>
      ${escape(facet.label)} (${formatCount(facet.count)})
    </option>`).join('');
}

function renderTruthStrip(distribution) {
  const available = TRUTH_KEYS.filter(([key]) => hasNumber(distribution[key]));
  const total = available.reduce((sum, [key]) => sum + distribution[key], 0);
  if (!available.length) return '<span class="data-missing">未返回四值汇总</span>';

  const segments = TRUTH_KEYS.map(([key]) => {
    const value = distribution[key];
    const width = hasNumber(value) && total > 0 ? (value / total) * 100 : 0;
    return `<i class="truth-${key}" style="width:${width}%" title="${key}: ${formatCount(value)}"></i>`;
  }).join('');
  return `<span class="truth-mini-bar" aria-label="四值分布">${segments}</span>`;
}

function renderFamilyTags(families, escape, maximum = 3) {
  if (!families.length) return '<span class="data-missing">无已观测事件族</span>';
  const visible = families.slice(0, maximum);
  const hiddenCount = families.length - visible.length;
  return `${visible.map((family) => `<span class="tag tag--event">${escape(family)}</span>`).join('')}${hiddenCount > 0 ? `<span class="tag">+${hiddenCount}</span>` : ''}`;
}

function renderEpisodeRows(items, selectedIds, escape) {
  if (!items.length) {
    return `
      <tr>
        <td colspan="7"><div class="empty-state compact"><strong>没有匹配的 Episode</strong><span>请放宽 EPI、seed、事件族或闭合状态条件。</span></div></td>
      </tr>`;
  }

  return items.map((item) => `
    <tr class="episode-row ${selectedIds.has(item.episode_id) ? 'is-selected' : ''}">
      <td class="compare-check-cell">
        <label class="check-only" title="加入并排比较">
          <input type="checkbox" data-compare-id="${escape(item.episode_id)}" ${selectedIds.has(item.episode_id) ? 'checked' : ''}>
          <span aria-hidden="true"></span><span class="sr-only">选择 ${escape(item.episode_id)}</span>
        </label>
      </td>
      <td>
        <strong class="episode-id">${escape(item.episode_id)}</strong>
        <span class="episode-seed">${escape(item.seed)}</span>
      </td>
      <td><span class="status-badge status-badge--${closureTone(item.closure_status)}">${escape(item.closure_status)}</span></td>
      <td class="numeric"><strong>${formatCount(item.event_count)}</strong><small>事件</small></td>
      <td class="numeric"><strong>${formatCount(item.transition_count)}</strong><small>转换</small></td>
      <td><div class="event-tags">${renderFamilyTags(item.event_families, escape)}</div></td>
      <td>
        <div class="row-actions">
          <button class="button button--small button--primary" type="button" data-open-episode="${escape(item.episode_id)}">打开时序</button>
          <button class="button button--small button--ghost" type="button" data-compare-seeds="${escape(item.epi_id)}">三 seed</button>
        </div>
      </td>
    </tr>`).join('');
}

function renderComparisonEpisode(episode, escape) {
  const truthRows = TRUTH_KEYS.map(([key, shortLabel]) => `
    <li class="truth-stat--${key}"><span>${shortLabel}</span><strong>${formatCount(episode.truth_distribution[key])}</strong></li>`).join('');

  return `
    <article class="compare-column" data-seed="${escape(episode.seed)}">
      <header>
        <div><span class="seed-label">${escape(episode.seed)}</span><h3>${escape(episode.episode_id)}</h3></div>
        <span class="status-badge status-badge--${closureTone(episode.closure_status)}">${escape(episode.closure_status)}</span>
      </header>
      <div class="compare-truth-strip">${renderTruthStrip(episode.truth_distribution)}</div>
      <div class="compare-metric-grid">
        <div><span>事件</span><strong>${formatCount(episode.event_count)}</strong></div>
        <div><span>谓词转换</span><strong>${formatCount(episode.transition_count)}</strong></div>
        <div><span>异常</span><strong>${formatCount(episode.anomaly_count)}</strong></div>
      </div>
      <ul class="compare-truth-values">${truthRows}</ul>
      <div class="event-tags">${renderFamilyTags(episode.event_families, escape, 6)}</div>
      <button class="button button--secondary button--full" type="button" data-open-episode="${escape(episode.episode_id)}">在时序主界面查看</button>
    </article>`;
}

function renderComparison(comparison, compareError, escape) {
  if (compareError) {
    return `
      <div class="inline-error" role="alert">
        <strong>并排比较读取失败</strong><span>${escape(compareError instanceof Error ? compareError.message : String(compareError))}</span>
      </div>`;
  }
  if (!comparison) {
    return `
      <div class="empty-state compare-empty">
        <strong>选择 2–3 个 Episode 进行并排比较</strong>
        <span>也可点击任意一行的“三 seed”，按同一 EPI 的 seed00 / seed01 / seed02 展开。</span>
      </div>`;
  }
  return `<div class="compare-grid compare-grid--${comparison.episodes.length}">${comparison.episodes.map((episode) => renderComparisonEpisode(episode, escape)).join('')}</div>`;
}

export async function renderEpisodes(ctx) {
  const { root, api, store, ui } = ctx;
  const escape = (value) => ui.escape(String(value ?? ''));
  const filters = { ...EMPTY_FILTERS, ...(store.filters ?? {}) };
  const compareIds = Array.isArray(store.compareIds) ? store.compareIds.slice(0, 3) : [];
  renderLoading(root);

  const episodesRequest = api.episodes(filters);
  const comparisonRequest = compareIds.length > 0 ? api.compare(compareIds) : Promise.resolve(null);
  const [episodeResult, comparisonResult] = await Promise.allSettled([episodesRequest, comparisonRequest]);

  if (episodeResult.status === 'rejected') {
    renderFailure(root, escape, episodeResult.reason);
    return;
  }

  const listing = episodeResult.value;
  const comparison = comparisonResult.status === 'fulfilled' ? comparisonResult.value : null;
  const compareError = comparisonResult.status === 'rejected' ? comparisonResult.reason : null;
  const selectedIds = new Set(compareIds);

  root.innerHTML = `
    <section class="view view-episodes">
      <header class="view-header episode-header">
        <div>
          <p class="eyebrow">EPISODE INDEX / 3-SEED COMPARISON</p>
          <h1>Episode 选择</h1>
          <p>事件族仅用于已检测结果筛选，不会反向参与事件检测。</p>
        </div>
        <div class="result-counter"><strong>${formatCount(listing.total)}</strong><span>个匹配 Episode</span></div>
      </header>

      <form class="panel filter-ribbon" data-filter-form>
        <label><span>EPI</span><select name="epi"><option value="">全部 EPI</option>${renderOptions(listing.facets.epis, filters.epi, escape)}</select></label>
        <label><span>SEED</span><select name="seed"><option value="">全部 seed</option>${renderOptions(listing.facets.seeds, filters.seed, escape)}</select></label>
        <label><span>事件族</span><select name="event_family"><option value="">全部事件族</option>${renderOptions(listing.facets.event_families, filters.event_family, escape)}</select></label>
        <label><span>闭合状态</span><select name="closure"><option value="">全部状态</option>${renderOptions(listing.facets.closure_statuses, filters.closure, escape)}</select></label>
        <div class="filter-actions">
          <button class="button button--primary" type="submit">应用筛选</button>
          <button class="button button--ghost" type="button" data-reset-filters>重置</button>
        </div>
      </form>

      <section class="panel episode-index-panel">
        <header class="panel-header table-toolbar">
          <div>
            <p class="panel-kicker">ON-DEMAND EPISODE METADATA</p>
            <h2>单集索引</h2>
          </div>
          <div class="compare-toolbar">
            <span>已选 <strong data-compare-count>${selectedIds.size}</strong> / 3</span>
            <button class="button button--secondary" type="button" data-run-compare ${selectedIds.size < 2 ? 'disabled' : ''}>并排比较</button>
            <button class="text-button" type="button" data-clear-compare ${selectedIds.size === 0 ? 'disabled' : ''}>清空</button>
          </div>
        </header>
        <div class="table-scroll">
          <table class="data-table episode-table">
            <thead><tr><th></th><th>EPISODE / SEED</th><th>闭合</th><th>事件</th><th>转换</th><th>已观测事件族</th><th>操作</th></tr></thead>
            <tbody>${renderEpisodeRows(listing.items, selectedIds, escape)}</tbody>
          </table>
        </div>
      </section>

      <section class="panel comparison-panel">
        <header class="panel-header">
          <div><p class="panel-kicker">SEED00 / SEED01 / SEED02</p><h2>三 Seed 并排</h2></div>
          <span class="panel-note">同一数据口径 · 缺失值显示为 “—”</span>
        </header>
        ${renderComparison(comparison, compareError, escape)}
      </section>
    </section>`;

  const filterForm = root.querySelector('[data-filter-form]');
  filterForm.addEventListener('submit', (event) => {
    event.preventDefault();
    const formData = new FormData(filterForm);
    store.patch({
      filters: {
        epi: formData.get('epi'),
        seed: formData.get('seed'),
        event_family: formData.get('event_family'),
        closure: formData.get('closure'),
      },
    });
    renderEpisodes(ctx);
  });

  root.querySelector('[data-reset-filters]').addEventListener('click', () => {
    store.patch({ filters: { ...EMPTY_FILTERS } });
    renderEpisodes(ctx);
  });

  root.querySelectorAll('[data-compare-id]').forEach((checkbox) => {
    checkbox.addEventListener('change', () => {
      const current = new Set(Array.isArray(store.compareIds) ? store.compareIds : []);
      if (checkbox.checked && current.size >= 3) {
        checkbox.checked = false;
        return;
      }
      if (checkbox.checked) current.add(checkbox.dataset.compareId);
      else current.delete(checkbox.dataset.compareId);
      const nextIds = [...current];
      store.patch({ compareIds: nextIds });
      root.querySelector('[data-compare-count]').textContent = String(nextIds.length);
      root.querySelector('[data-run-compare]').disabled = nextIds.length < 2;
      root.querySelector('[data-clear-compare]').disabled = nextIds.length === 0;
      checkbox.closest('tr').classList.toggle('is-selected', checkbox.checked);
    });
  });

  root.querySelector('[data-run-compare]').addEventListener('click', () => renderEpisodes(ctx));
  root.querySelector('[data-clear-compare]').addEventListener('click', () => {
    store.patch({ compareIds: [] });
    renderEpisodes(ctx);
  });

  root.querySelectorAll('[data-compare-seeds]').forEach((button) => {
    button.addEventListener('click', () => {
      const ids = ['seed00', 'seed01', 'seed02'].map((seed) => `${button.dataset.compareSeeds}__${seed}`);
      store.patch({ compareIds: ids });
      renderEpisodes(ctx);
    });
  });

  root.querySelectorAll('[data-open-episode]').forEach((button) => {
    button.addEventListener('click', () => {
      store.patch({ episodeId: button.dataset.openEpisode });
      ctx.navigate('semantic');
    });
  });
}
