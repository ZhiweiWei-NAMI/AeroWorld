export const escape = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

export const shortId = (value, length = 18) => {
  const text = String(value ?? "—");
  return text.length > length ? `${text.slice(0, length - 7)}…${text.slice(-6)}` : text;
};

export const formatNumber = (value, maximumFractionDigits = 3) => {
  if (typeof value !== "number" || !Number.isFinite(value)) return String(value ?? "—");
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits }).format(value);
};

export const truthBadge = (truth) => {
  const value = String(truth ?? "unknown").toLowerCase();
  const labels = {
    true: "TRUE",
    false: "FALSE",
    unknown: "UNKNOWN",
    out_of_scope: "OUT OF SCOPE",
    pass: "PASS",
    fail: "FAIL",
    succeeded: "SUCCEEDED",
    occurred: "OCCURRED",
  };
  return `<span class="truth-badge truth-${escape(value.replaceAll("_", "-"))}">${escape(labels[value] || value.toUpperCase())}</span>`;
};

export const empty = (message = "暂无已物化记录") => `
  <div class="empty-state">
    <span class="empty-state__glyph">∅</span>
    <div>${escape(message)}</div>
  </div>`;

export const loading = (message = "正在按需读取真实语义数据…") => `
  <div class="loading-state" role="status">
    <span class="loader"></span>
    <span>${escape(message)}</span>
  </div>`;

export const errorPanel = (error) => `
  <section class="error-panel" role="alert">
    <div class="error-panel__title">数据链路异常</div>
    <p>${escape(error?.message || error)}</p>
    <p class="muted">系统不会把缺失或读取失败显示为“未发生”。请检查服务终端中的原始错误。</p>
  </section>`;

export const keyValueRows = (object = {}) => Object.entries(object).map(([key, value]) => `
  <div class="kv-row"><span>${escape(key)}</span><code>${escape(typeof value === "object" ? JSON.stringify(value) : value)}</code></div>
`).join("");

// DOM 版本（供模块视图直接 replaceChildren，不做 innerHTML 拼接）
export const errorPanelDOM = (error) => {
  const box = document.createElement("section");
  box.className = "error-panel";
  box.setAttribute("role", "alert");
  const title = document.createElement("div");
  title.className = "error-panel__title";
  title.textContent = "数据链路异常";
  const p = document.createElement("p");
  p.textContent = error?.message || String(error);
  const muted = document.createElement("p");
  muted.className = "muted";
  muted.textContent = "系统不会把缺失或读取失败显示为“未发生”。请检查服务终端中的原始错误。";
  box.append(title, p, muted);
  return box;
};

export const ui = Object.freeze({
  escape,
  shortId,
  formatNumber,
  truthBadge,
  empty,
  loading,
  errorPanel,
  errorPanelDOM,
  keyValueRows,
});

