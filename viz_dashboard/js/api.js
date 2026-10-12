const encodeQuery = (params = {}) => {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") return;
    query.set(key, Array.isArray(value) ? value.join(",") : String(value));
  });
  const suffix = query.toString();
  return suffix ? `?${suffix}` : "";
};

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error(`数据服务返回了非 JSON 响应（HTTP ${response.status}）`, { cause: error });
  }
  if (!response.ok && response.status !== 501) {
    throw new Error(payload.error || `数据请求失败（HTTP ${response.status}）`);
  }
  return payload;
}

export const api = Object.freeze({
  health: () => request("/api/health"),
  summary: () => request("/api/summary"),
  episodes: (params = {}) => request(`/api/episodes${encodeQuery(params)}`),
  compare: (ids) => request(`/api/episodes/compare${encodeQuery({ ids })}`),
  episode: (episodeId) => request(`/api/episodes/${encodeURIComponent(episodeId)}`),
  timeline: (episodeId, tick) =>
    request(`/api/episodes/${encodeURIComponent(episodeId)}/timeline${encodeQuery({ tick })}`),
  sceneMap: (episodeId) =>
    request(`/api/episodes/${encodeURIComponent(episodeId)}/scene-map`),
  eventChain: (episodeId) =>
    request(`/api/episodes/${encodeURIComponent(episodeId)}/event-chain`),
  semanticContext: (episodeId, selection = {}) =>
    request(`/api/episodes/${encodeURIComponent(episodeId)}/semantic-context${encodeQuery(selection)}`),
  catalogs: () => request("/api/catalogs"),
  closure: (episodeId) => request(`/api/episodes/${encodeURIComponent(episodeId)}/closure`),
  armsCatalog: () => request("/api/arms"),
  armProfile: (profile, seed) =>
    request(`/api/arms/profile${encodeQuery({ profile, seed })}`),
  armGroup: (profile, seed, group) =>
    request(`/api/arms/group${encodeQuery({ profile, seed, group })}`),
  armViz: (profile, seed, group, tick) =>
    request(`/api/arms/viz${encodeQuery({ profile, seed, group, tick })}`),
  startGeneration: (payload) =>
    request("/api/operations/generation/start", { method: "POST", body: JSON.stringify(payload) }),
  buildPackage: (payload) =>
    request("/api/operations/package/build", { method: "POST", body: JSON.stringify(payload) }),
  startUpload: (payload) =>
    request("/api/operations/upload/start", { method: "POST", body: JSON.stringify(payload) }),
});
