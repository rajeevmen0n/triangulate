import { el, api, STAGES, STAGE_LABEL, ACTIVE } from "./util.js";
import { renderResults } from "./results.js";

let es = null;
let token = 0;
export function stopJobView() { token++; if (es) { es.close(); es = null; } }

export async function showJob(root, id, state) {
  const my = ++token;
  root.replaceChildren(el("p", { class: "muted" }, "Loading..."));
  let job;
  try { job = await api(`/jobs/${id}`); } catch (e) {
    root.replaceChildren(el("p", { class: "err", role: "alert" }, e.message)); return;
  }
  if (my !== token) return;
  const head = el("section", { class: "card" });
  const body = el("section");
  root.replaceChildren(head, body);

  const bar = el("div", { class: "bar big", role: "progressbar", "aria-valuemin": 0, "aria-valuemax": 100, "aria-label": "Overall progress" }, el("i"));
  const msg = el("p", { "aria-live": "polite" });
  const stageEls = {};
  const list = el("ol", { class: "stages", style: "list-style:none" }, STAGES.map((s) => {
    const fill = el("i"); const li = el("li", { "data-s": "wait" }, el("span", {}, STAGE_LABEL[s]), el("div", { class: "bar" }, fill), el("small", { class: "muted" }, "waiting"));
    stageEls[s] = { li, fill, note: li.lastChild }; return li;
  }));
  const log = el("ul", { class: "log", "aria-label": "Log" });
  let lastStage = null;

  function setProgress(p, stage, message, status) {
    const pc = Math.round(Math.min(1, Math.max(0, p > 1 ? p / 100 : p)) * 100);
    bar.firstChild.style.width = pc + "%"; bar.setAttribute("aria-valuenow", pc);
    msg.textContent = `${pc}%  ${message || ""}`;
    const idx = STAGES.indexOf(stage);
    STAGES.forEach((s, i) => {
      const e = stageEls[s]; let st = "wait";
      if (status === "done") st = "done"; else if (idx >= 0 && i < idx) st = "done"; else if (i === idx) st = "active";
      e.li.dataset.s = st; e.fill.style.width = st === "done" ? "100%" : st === "active" ? "50%" : "0";
      e.note.textContent = st === "done" ? "done" : st === "active" ? (message || "running") : "waiting";
    });
  }

  function paintHead(j) {
    const c = j.params?.constraints || {};
    const acts = el("div", { class: "row" });
    if (ACTIVE.has(j.status)) acts.append(el("button", { class: "btn danger", type: "button", onclick: () => api(`/jobs/${id}/cancel`, { method: "POST" }).catch((e) => alert(e.message)) }, "Cancel"));
    head.replaceChildren(
      el("div", { class: "row", style: "justify-content:space-between" }, el("h2", {}, `Analysis ${id.slice(0, 8)}`), acts),
      el("p", { class: "muted small" }, `Region ${c.bbox ? c.bbox.map((v) => (+v).toFixed(3)).join(", ") : "n/a"}; date ${c.date_start ?? "n/a"}${c.date_end ? " to " + c.date_end : ""}${j.params?.visual ? "; visual module on (frames sent to Claude API)" : ""}`),
      el("p", {}, el("strong", {}, j.status), j.error ? el("span", { class: "err", role: "alert", style: "margin-left:10px" }, j.error.split("\n").pop()) : null));
  }
  paintHead(job);

  async function finish() {
    if (es) { es.close(); es = null; }
    const j = await api(`/jobs/${id}`).catch(() => job);
    if (my !== token) return;
    job = j; paintHead(j); setProgress(j.progress, j.stage, j.message, j.status);
    if (j.status === "done" || j.has_result) {
      const res = el("section"); body.replaceChildren(res);
      try { await renderResults(res, j, state, (n) => { location.hash = `#/job/${n.id}`; }); }
      catch (e) { res.replaceChildren(el("p", { class: "err", role: "alert" }, "Could not show results: " + e.message)); }
    } else {
      body.append(el("p", { class: "warn" }, `This analysis ended as ${j.status}. Partial files: ${j.artifacts.join(", ") || "none"}.`));
    }
  }

  if (ACTIVE.has(job.status)) {
    body.replaceChildren(el("div", { class: "card" }, el("h3", {}, "Progress"), bar, msg, list, log));
    setProgress(job.progress, job.stage, job.message, job.status);
    es = new EventSource(`/api/jobs/${id}/stream`);
    const onEv = (ev) => {
      if (my !== token) return;
      const d = JSON.parse(ev.data);
      setProgress(d.progress ?? 0, d.stage, d.message, "running");
      if (d.stage !== lastStage || ev.type === "error") {
        lastStage = d.stage;
        log.append(el("li", { class: ev.type }, `${STAGE_LABEL[d.stage] || d.stage}: ${d.message}`));
        log.scrollTop = log.scrollHeight;
      }
    };
    es.addEventListener("status", onEv); es.addEventListener("progress", onEv); es.addEventListener("error", (ev) => { if (ev.data) onEv(ev); });
    es.addEventListener("end", () => finish());
  } else {
    await finish();
  }
}
