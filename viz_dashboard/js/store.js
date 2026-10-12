const STORAGE_KEY = "aeroworld-semantic-dashboard-state-v1";

const DEFAULT_STATE = Object.freeze({
  episodeId: "L4-1_v1__seed00",
  tick: 385,
  selection: null,
  compareIds: [],
  filters: {},
  events: [],
  transitions: [],
  frame: null,
});

export class DashboardStore extends EventTarget {
  constructor() {
    super();
    const persisted = this.#readPersisted();
    this.state = { ...DEFAULT_STATE, ...persisted };
  }

  #readPersisted() {
    try {
      const value = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "{}");
      if (!value || typeof value !== "object" || Array.isArray(value)) return {};
      return {
        episodeId: typeof value.episodeId === "string" ? value.episodeId : DEFAULT_STATE.episodeId,
        tick: Number.isInteger(value.tick) ? value.tick : DEFAULT_STATE.tick,
        selection: value.selection && typeof value.selection === "object" ? value.selection : null,
        compareIds: Array.isArray(value.compareIds) ? value.compareIds.slice(0, 3) : [],
      };
    } catch {
      sessionStorage.removeItem(STORAGE_KEY);
      return {};
    }
  }

  get episodeId() { return this.state.episodeId; }
  get tick() { return this.state.tick; }
  get selection() { return this.state.selection; }
  get compareIds() { return this.state.compareIds; }
  get filters() { return this.state.filters; }
  get events() { return this.state.events; }
  get transitions() { return this.state.transitions; }
  get frame() { return this.state.frame; }

  patch(partial) {
    if (!partial || typeof partial !== "object" || Array.isArray(partial)) {
      throw new TypeError("store.patch expects an object");
    }
    const previous = this.state;
    this.state = { ...previous, ...partial };
    const persisted = {
      episodeId: this.state.episodeId,
      tick: this.state.tick,
      selection: this.state.selection,
      compareIds: this.state.compareIds,
    };
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(persisted));
    this.dispatchEvent(new CustomEvent("store:change", { detail: { previous, current: this.state, partial } }));
  }

  subscribe(listener) {
    const handler = (event) => listener(event.detail.current, event.detail);
    this.addEventListener("store:change", handler);
    return () => this.removeEventListener("store:change", handler);
  }
}

export const store = new DashboardStore();

