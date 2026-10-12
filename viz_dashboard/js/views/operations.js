const NUMBER_FORMAT = new Intl.NumberFormat('zh-CN');
const PERCENT_FORMAT = new Intl.NumberFormat('zh-CN', {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

function hasNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function formatCount(value) {
  return hasNumber(value) ? NUMBER_FORMAT.format(value) : '—';
}

function formatBytes(value) {
  if (!hasNumber(value)) return '—';
  if (value < 1024) return `${NUMBER_FORMAT.format(value)} B`;
  const units = ['KiB', 'MiB', 'GiB', 'TiB'];
  let size = value;
  let unitIndex = -1;
  do {
    size /= 1024;
    unitIndex += 1;
  } while (size >= 1024 && unitIndex < units.length - 1);
  return `${size.toFixed(size >= 100 ? 0 : size >= 10 ? 1 : 2)} ${units[unitIndex]}`;
}

function formatDate(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? String(value) : date.toLocaleString('zh-CN', { hour12: false });
}

function ratio(completed, total) {
  if (!hasNumber(completed) || !hasNumber(total) || total <= 0) return null;
  return Math.min(100, Math.max(0, (completed / total) * 100));
}

function statusTone(status) {
  if (['ready', 'complete', 'completed', 'verified', 'PASS'].includes(status)) return 'ok';
  if (['running', 'building', 'uploading', 'validating'].includes(status)) return 'active';
  if (['failed', 'error', 'FAIL', 'mismatch'].includes(status)) return 'danger';
  return 'muted';
}

function renderLoading(root, label) {
  root.innerHTML = `
    <section class="view view-operations" aria-busy="true">
      <div class="view-loading"><span class="loading-pulse" aria-hidden="true"></span><span>${label}</span></div>
    </section>`;
}

function renderFailure(root, escape, title, error) {
  root.innerHTML = `
    <section class="view view-operations">
      <header class="view-header"><div><p class="eyebrow">CONTROL PLANE</p><h1>${title}</h1></div></header>
      <div class="panel error-panel" role="alert">
        <strong>状态接口读取失败</strong>
        <p>${escape(error instanceof Error ? error.message : String(error))}</p>
      </div>
    </section>`;
}

function setBusy(root, busy) {
  root.querySelectorAll('[data-operation-action]').forEach((button) => {
    button.disabled = busy || button.dataset.contractDisabled === 'true';
  });
  root.setAttribute('aria-busy', String(busy));
}

function showActionError(root, escape, error) {
  const feedback = root.querySelector('[data-operation-feedback]');
  feedback.hidden = false;
  feedback.className = 'operation-feedback operation-feedback--error';
  feedback.innerHTML = escape(error instanceof Error ? error.message : String(error));
}

function renderInterfaceNotice(state, escape, actionName) {
  const available = state.available === true;
  return `
    <div class="interface-notice interface-notice--${available ? 'available' : 'shell'}" role="status">
      <span class="interface-notice__icon" aria-hidden="true">${available ? '●' : '◇'}</span>
      <div>
        <strong>${available ? `${actionName}接口已开放` : `${actionName}接口壳已就绪`}</strong>
        <p>${escape(state.message)}</p>
      </div>
      <span class="status-badge status-badge--${statusTone(state.status)}">${escape(state.status)}</span>
    </div>`;
}

function renderJob(job, escape) {
  if (!job) {
    return `
      <div class="empty-state operation-empty">
        <strong>当前没有语义转换任务</strong>
        <span>任务启动后，此处仅展示服务端返回的阶段和进度。</span>
      </div>`;
  }
  const progress = ratio(job.completed, job.total);
  return `
    <div class="job-monitor">
      <div class="job-monitor__top">
        <div><span class="mono-label">${escape(job.id)}</span><strong>${escape(job.phase)}</strong></div>
        <span class="status-badge status-badge--${statusTone(job.status)}">${escape(job.status)}</span>
      </div>
      <div class="progress-track progress-track--large" aria-label="转换进度"><i style="width:${progress ?? 0}%"></i></div>
      <div class="progress-caption">
        <span>${formatCount(job.completed)} / ${formatCount(job.total)} episode</span>
        <strong>${progress === null ? '—' : `${PERCENT_FORMAT.format(progress)}%`}</strong>
      </div>
      <dl class="definition-grid compact">
        <div><dt>开始时间</dt><dd>${escape(formatDate(job.started_at))}</dd></div>
        <div><dt>最近更新</dt><dd>${escape(formatDate(job.updated_at))}</dd></div>
      </dl>
    </div>`;
}

function renderPackage(packageInfo, escape) {
  if (!packageInfo) {
    return `
      <div class="empty-state operation-empty">
        <strong>尚未构建唯一上传包</strong>
        <span>纯语义转换成功后才能请求服务端打包。</span>
      </div>`;
  }
  return `
    <div class="package-receipt">
      <div class="package-receipt__icon" aria-hidden="true">PKG</div>
      <div class="package-receipt__body">
        <span class="mono-label">${escape(packageInfo.id)}</span>
        <h3>${escape(packageInfo.name)}</h3>
        <dl class="definition-grid compact">
          <div><dt>文件数</dt><dd>${formatCount(packageInfo.file_count)}</dd></div>
          <div><dt>包大小</dt><dd>${formatBytes(packageInfo.size_bytes)}</dd></div>
          <div><dt>生成时间</dt><dd>${escape(formatDate(packageInfo.created_at))}</dd></div>
        </dl>
        <div class="digest-box"><span>SHA-256</span><code>${escape(packageInfo.sha256)}</code></div>
      </div>
    </div>`;
}

export async function renderGenerate(ctx) {
  const { root, api, ui } = ctx;
  const escape = (value) => ui.escape(String(value ?? ''));
  renderLoading(root, '正在读取转换与打包状态…');

  let state;
  try {
    state = await api.generationStatus();
  } catch (error) {
    renderFailure(root, escape, '生成与打包', error);
    return;
  }

  const inputs = state.available_inputs ?? [];
  const canStart = state.available === true && inputs.length > 0 && !['running', 'building'].includes(state.job?.status);
  const canPackage = state.available === true && state.job?.status === 'complete';

  root.innerHTML = `
    <section class="view view-operations view-generate">
      <header class="view-header operation-header">
        <div>
          <p class="eyebrow">SEMANTIC BUILD CONTROL</p>
          <h1>生成与打包</h1>
          <p>从服务端允许列表中选择已有输入，启动纯语义转换，并生成一个可校验上传包。</p>
        </div>
        <button class="button button--ghost" type="button" data-refresh data-operation-action>刷新状态</button>
      </header>

      ${renderInterfaceNotice(state, escape, '生成')}
      <div class="operation-feedback" data-operation-feedback hidden></div>

      <div class="operation-layout">
        <section class="panel operation-command-panel">
          <header class="panel-header"><div><p class="panel-kicker">INPUT / MODE / CONTRACT</p><h2>转换命令</h2></div><span class="step-number">01</span></header>
          <form data-generation-form>
            <label class="field-stack">
              <span>已有输入目录</span>
              <select name="input_id" required ${state.available !== true ? 'disabled' : ''}>
                ${inputs.length ? inputs.map((input) => `<option value="${escape(input.id)}">${escape(input.label)} · ${escape(input.path)}</option>`).join('') : '<option value="">服务端未提供可用输入</option>'}
              </select>
              <small>只提交目录 ID，服务端必须再次校验路径允许列表。</small>
            </label>
            <div class="locked-mode">
              <span>执行模式</span><strong>objective_semantic_only</strong><i>固定</i>
            </div>
            <div class="contract-grid">
              <div><span>TICK RANGE</span><strong>0–900</strong></div>
              <div><span>FORMAL STEP</span><strong>5 ticks</strong></div>
              <div><span>CAPTURE</span><strong>forbidden</strong></div>
            </div>
            <div class="safety-callout">
              <strong>服务边界</strong>
              <span>只生成语义真值与图工件；不启动 UE，不写入 RGB / depth / SEG 正式采集。</span>
            </div>
            <button class="button button--primary button--full" type="submit" data-operation-action data-contract-disabled="${!canStart}" ${canStart ? '' : 'disabled'}>启动纯语义转换</button>
          </form>
        </section>

        <section class="panel operation-progress-panel">
          <header class="panel-header"><div><p class="panel-kicker">SERVER-SIDE PROGRESS</p><h2>执行进度</h2></div><span class="step-number">02</span></header>
          ${renderJob(state.job, escape)}
        </section>

        <section class="panel operation-package-panel">
          <header class="panel-header">
            <div><p class="panel-kicker">IMMUTABLE UPLOAD ARTIFACT</p><h2>唯一上传包</h2></div><span class="step-number">03</span>
          </header>
          ${renderPackage(state.package, escape)}
          <button class="button button--secondary button--full" type="button" data-build-package data-operation-action data-contract-disabled="${!canPackage}" ${canPackage ? '' : 'disabled'}>生成唯一上传包</button>
        </section>
      </div>
    </section>`;

  root.querySelector('[data-refresh]').addEventListener('click', () => renderGenerate(ctx));
  root.querySelector('[data-generation-form]').addEventListener('submit', async (event) => {
    event.preventDefault();
    setBusy(root, true);
    try {
      const formData = new FormData(event.currentTarget);
      await api.startGeneration({ input_id: formData.get('input_id'), mode: 'objective_semantic_only' });
      await renderGenerate(ctx);
    } catch (error) {
      setBusy(root, false);
      showActionError(root, escape, error);
    }
  });
  root.querySelector('[data-build-package]').addEventListener('click', async () => {
    setBusy(root, true);
    try {
      await api.buildPackage({ job_id: state.job.id });
      await renderGenerate(ctx);
    } catch (error) {
      setBusy(root, false);
      showActionError(root, escape, error);
    }
  });
}

function renderTargets(targets, escape) {
  if (!targets.length) return '<option value="">服务端未提供允许的上传目标</option>';
  return targets.map((target) => `<option value="${escape(target.id)}">${escape(target.label)} · ${escape(target.directory)}</option>`).join('');
}

function renderPackages(packages, escape) {
  if (!packages.length) return '<option value="">尚无可上传包</option>';
  return packages.map((packageInfo) => `<option value="${escape(packageInfo.id)}">${escape(packageInfo.name)} · ${formatBytes(packageInfo.size_bytes)}</option>`).join('');
}

function renderTransfer(transfer, escape) {
  if (!transfer) {
    return `
      <div class="empty-state operation-empty">
        <strong>当前没有上传任务</strong>
        <span>选择唯一上传包和服务端允许的目标后，才能发起传输。</span>
      </div>`;
  }
  const progress = ratio(transfer.bytes_sent, transfer.total_bytes);
  const hashesPresent = Boolean(transfer.local_sha256 && transfer.remote_sha256);
  const hashesMatch = hashesPresent && transfer.local_sha256 === transfer.remote_sha256;
  const checksumLabel = !hashesPresent ? '等待服务端校验' : hashesMatch ? '校验一致' : '校验不一致';
  return `
    <div class="transfer-monitor">
      <div class="job-monitor__top">
        <div><span class="mono-label">${escape(transfer.id)}</span><strong>${escape(transfer.package_id)} → ${escape(transfer.target_id)}</strong></div>
        <span class="status-badge status-badge--${statusTone(transfer.status)}">${escape(transfer.status)}</span>
      </div>
      <div class="progress-track progress-track--large" aria-label="上传进度"><i style="width:${progress ?? 0}%"></i></div>
      <div class="progress-caption"><span>${formatBytes(transfer.bytes_sent)} / ${formatBytes(transfer.total_bytes)}</span><strong>${progress === null ? '—' : `${PERCENT_FORMAT.format(progress)}%`}</strong></div>
      <dl class="definition-grid compact">
        <div><dt>校验结果</dt><dd><span class="status-badge status-badge--${statusTone(transfer.verification_status)}">${escape(transfer.verification_status)}</span></dd></div>
        <div><dt>最近更新</dt><dd>${escape(formatDate(transfer.updated_at))}</dd></div>
      </dl>
      <div class="checksum-comparison ${hashesMatch ? 'is-match' : hashesPresent ? 'is-mismatch' : ''}">
        <div><span>LOCAL SHA-256</span><code>${escape(transfer.local_sha256)}</code></div>
        <div><span>REMOTE SHA-256</span><code>${escape(transfer.remote_sha256)}</code></div>
        <strong>${checksumLabel}</strong>
      </div>
    </div>`;
}

export async function renderUpload(ctx) {
  const { root, api, ui } = ctx;
  const escape = (value) => ui.escape(String(value ?? ''));
  renderLoading(root, '正在读取上传与远端校验状态…');

  let state;
  try {
    state = await api.uploadStatus();
  } catch (error) {
    renderFailure(root, escape, '上传状态', error);
    return;
  }

  const targets = state.targets ?? [];
  const packages = state.packages ?? [];
  const canUpload = state.available === true && targets.length > 0 && packages.length > 0 && state.transfer?.status !== 'uploading';

  root.innerHTML = `
    <section class="view view-operations view-upload">
      <header class="view-header operation-header">
        <div>
          <p class="eyebrow">VERIFIED DELIVERY</p>
          <h1>上传状态</h1>
          <p>仅传递服务端签发的包 ID 与目标 ID，并通过本地/远端 SHA-256 一致性完成交付。</p>
        </div>
        <button class="button button--ghost" type="button" data-refresh data-operation-action>刷新状态</button>
      </header>

      ${renderInterfaceNotice(state, escape, '上传')}
      <div class="credential-boundary">
        <span class="credential-boundary__icon" aria-hidden="true">🛡</span>
        <div><strong>认证边界在服务端</strong><p>浏览器不采集、不保存、不传输 SSH 密码、私钥或私钥口令；连接配置由部署端管理。</p></div>
      </div>
      <div class="operation-feedback" data-operation-feedback hidden></div>

      <div class="operation-layout operation-layout--upload">
        <section class="panel operation-command-panel">
          <header class="panel-header"><div><p class="panel-kicker">ALLOWLISTED DESTINATION</p><h2>传输命令</h2></div><span class="step-number">01</span></header>
          <form data-upload-form>
            <label class="field-stack">
              <span>唯一上传包</span>
              <select name="package_id" required ${state.available !== true ? 'disabled' : ''}>${renderPackages(packages, escape)}</select>
            </label>
            <label class="field-stack">
              <span>目标服务器目录</span>
              <select name="target_id" required ${state.available !== true ? 'disabled' : ''}>${renderTargets(targets, escape)}</select>
              <small>目标路径只能由服务端允许列表给出，前端不提供自由路径输入。</small>
            </label>
            <div class="safety-callout">
              <strong>交付边界</strong>
              <span>上传内容必须是原样的 <code>aw_data/render_ready_episodes_capture_filtered</code> 交接产物，不改写 episode 内容。</span>
            </div>
            <button class="button button--primary button--full" type="submit" data-operation-action data-contract-disabled="${!canUpload}" ${canUpload ? '' : 'disabled'}>启动校验上传</button>
          </form>
        </section>

        <section class="panel operation-progress-panel">
          <header class="panel-header"><div><p class="panel-kicker">TRANSFER / VERIFY</p><h2>上传与校验</h2></div><span class="step-number">02</span></header>
          ${renderTransfer(state.transfer, escape)}
        </section>
      </div>
    </section>`;

  root.querySelector('[data-refresh]').addEventListener('click', () => renderUpload(ctx));
  root.querySelector('[data-upload-form]').addEventListener('submit', async (event) => {
    event.preventDefault();
    setBusy(root, true);
    try {
      const formData = new FormData(event.currentTarget);
      await api.startUpload({ package_id: formData.get('package_id'), target_id: formData.get('target_id') });
      await renderUpload(ctx);
    } catch (error) {
      setBusy(root, false);
      showActionError(root, escape, error);
    }
  });
}
