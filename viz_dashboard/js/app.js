import { api } from "./api.js";
import { store } from "./store.js";
import { ui, loading, errorPanel, escape } from "./ui.js";
import { renderOverview } from "./views/overview.js";
import { renderEpisodes } from "./views/episodes.js";
import { renderSemanticWorkbench } from "./views/semantic-workbench.js";
import { renderArms } from "./views/arms.js";
import { renderInterventions } from "./views/interventions.js";
import { renderPredicateCatalog, renderEventCatalog } from "./views/catalogs.js";
import { renderClosure } from "./views/closure.js";
import { renderGenerate, renderUpload } from "./views/operations.js";

const root = document.querySelector("#view");
const activeEpisode = document.querySelector("#active-episode");
const activeTick = document.querySelector("#active-tick");
const dataHealth = document.querySelector("#data-health");
const sidebar = document.querySelector(".sidebar");
const mobileMenu = document.querySelector("#mobile-menu");
const mobileViewport = window.matchMedia("(max-width: 860px)");

const ROUTES = Object.freeze({
  overview: renderOverview,
  episodes: renderEpisodes,
  semantic: renderSemanticWorkbench,
  arms: renderArms,
  interventions: renderInterventions,
  predicates: renderPredicateCatalog,
  "event-rules": renderEventCatalog,
  closure: renderClosure,
  generate: renderGenerate,
  upload: renderUpload,
});

const DEFAULT_ROUTE = "arms";

let renderSequence = 0;
let disposeCurrentView = null;

function setMobileNavigation(open, restoreFocus = false) {
  const expanded = mobileViewport.matches && open;
  sidebar.classList.toggle("is-open", expanded);
  sidebar.inert = mobileViewport.matches && !expanded;
  mobileMenu.setAttribute("aria-expanded", String(expanded));
  mobileMenu.setAttribute("aria-label", expanded ? "关闭导航" : "打开导航");
  if (restoreFocus) mobileMenu.focus();
}

const toast = (message, tone = "info") => {
  const region = document.querySelector("#toast-region");
  const item = document.createElement("div");
  item.className = `toast toast--${tone}`;
  item.innerHTML = `<span>${escape(message)}</span>`;
  region.append(item);
  requestAnimationFrame(() => item.classList.add("is-visible"));
  setTimeout(() => {
    item.classList.remove("is-visible");
    setTimeout(() => item.remove(), 220);
  }, 3600);
};

const appUi = Object.freeze({ ...ui, toast });

function currentRoute() {
  const route = location.hash.replace(/^#\/?/, "").split("?")[0];
  return ROUTES[route] ? route : DEFAULT_ROUTE;
}

function navigate(route) {
  if (!ROUTES[route]) throw new Error(`未知界面路由: ${route}`);
  if (currentRoute() === route) {
    renderRoute();
    return;
  }
  location.hash = `#/${route}`;
}

function updateChrome() {
  activeEpisode.textContent = store.episodeId;
  activeTick.textContent = String(store.tick).padStart(3, "0");
  const route = currentRoute();
  document.querySelectorAll("#primary-nav [data-route]").forEach((link) => {
    link.classList.toggle("is-active", link.dataset.route === route);
  });
}

async function renderRoute() {
  const sequence = ++renderSequence;
  if (typeof disposeCurrentView === "function") disposeCurrentView();
  disposeCurrentView = null;
  updateChrome();
  setMobileNavigation(false);
  root.innerHTML = loading();
  root.focus({ preventScroll: true });
  const renderer = ROUTES[currentRoute()];
  const mount = document.createElement("div");
  mount.className = "view-mount";
  try {
    const dispose = await renderer({ root: mount, api, store, navigate, ui: appUi });
    if (sequence !== renderSequence) {
      if (typeof dispose === "function") dispose();
      return;
    }
    root.replaceChildren(mount);
    disposeCurrentView = typeof dispose === "function" ? dispose : null;
  } catch (error) {
    if (sequence !== renderSequence) return;
    console.error(error);
    root.innerHTML = errorPanel(error);
  }
}

store.subscribe(() => updateChrome());
window.addEventListener("hashchange", renderRoute);
mobileMenu.addEventListener("click", () => {
  setMobileNavigation(!sidebar.classList.contains("is-open"));
});
mobileViewport.addEventListener("change", () => setMobileNavigation(false));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && sidebar.classList.contains("is-open")) {
    setMobileNavigation(false, true);
  }
});
setMobileNavigation(false);

const clock = document.querySelector("#system-clock");
const updateClock = () => {
  clock.textContent = new Intl.DateTimeFormat("zh-CN", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date());
};
updateClock();
setInterval(updateClock, 1000);

api.health()
  .then(() => dataHealth.classList.add("is-online"))
  .catch((error) => {
    dataHealth.classList.add("is-offline");
    dataHealth.innerHTML = "<i></i>数据连接失败";
    toast(error.message, "error");
  });

if (!location.hash || !ROUTES[currentRoute()]) {
  location.replace(`#/${DEFAULT_ROUTE}`);
} else {
  renderRoute();
}
