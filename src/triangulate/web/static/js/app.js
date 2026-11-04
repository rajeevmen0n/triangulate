import { el, api, STAGES, ACTIVE } from "./util.js";
import { newAnalysis } from "./form.js";
import { showJob, stopJobView } from "./job.js";

export const state = { config: null, jobs: [] };
const main = document.getElementById("main");

async function boot() {
  try { state.config = await api("/config"); } catch (e) {
    main.replaceChildren(el("p", { class: "err", role: "alert" }, "Cannot reach the server: " + e.message)); return;
  }
  const c = state.config;
  document.getElementById("use-notice").textContent = c.use_notice;
  const at = document.getElementById("attribution");
  c.attribution.forEach((a) => at.append(el("li", {}, a)));
  document.getElementById("net-note").textContent = c.basemap_enabled
    ? "Network: map tiles are fetched from the configured basemap server in your browser. Audio and analysis stay on this machine; flight data, terrain and OSM lookups are fetched by the server."
    : "Basemap disabled: this page makes no third-party requests. Flight data, terrain and OSM lookups are fetched by the server.";
  window.addEventListener("hashchange", route);
  await refreshJobs();
  route();
  setInterval(refreshJobs, 4000);
}

export async function refreshJobs() {
  try { state.jobs = await api("/jobs"); } catch { return; }
  renderJobs();
}

function currentId() { const m = location.hash.match(/^#\/job\/([\w-]+)/); return m ? m[1] : null; }

function renderJobs() {
  const ul = document.getElementById("job-list");
  document.getElementById("job-empty").hidden = state.jobs.length > 0;
  const cur = currentId();
  ul.replaceChildren(...state.jobs.map((j) => {
    const when = new Date(j.created * (j.created < 1e11 ? 1000 : 1)).toLocaleString();
    const label = `${j.params?.constraints?.date_start ?? ""} ${j.id.slice(0, 8)}`.trim();
    const acts = el("div", { class: "acts" });
    if (ACTIVE.has(j.status)) acts.append(btn("Cancel", () => act(() => api(`/jobs/${j.id}/cancel`, { method: "POST" }))));
    else {
      acts.append(btn("Rerun", async () => { const n = await api(`/jobs/${j.id}/rerun`, { method: "POST", headers: { "content-type": "application/json" }, body: "{}" }); await refreshJobs(); location.hash = `#/job/${n.id}`; }));
      acts.append(btn("Delete", () => { if (confirm("Delete this analysis and its uploaded file?")) act(async () => { await api(`/jobs/${j.id}`, { method: "DELETE" }); if (cur === j.id) location.hash = "#/new"; }); }, "danger"));
    }
    return el("li", { class: "job" + (cur === j.id ? " sel" : ""), "data-status": j.status },
      el("a", { class: "open", href: `#/job/${j.id}` }, label),
      el("div", { class: "meta" }, `${j.status}${j.status === "running" ? ` ${Math.round(j.progress * 100)}%` : ""} (${when})`), acts);
  }));
}
const btn = (t, fn, cls = "") => el("button", { type: "button", class: `btn small ${cls}`, onclick: fn }, t);
async function act(fn) {
  try { await fn(); } catch (e) { alert(e.message); }
  await refreshJobs();
}

function route() {
  stopJobView();
  renderJobs();
  const id = currentId();
  if (id) showJob(main, id, state);
  else newAnalysis(main, state, (job) => { refreshJobs(); location.hash = `#/job/${job.id}`; });
  main.focus({ preventScroll: true });
}
void STAGES;
boot();
