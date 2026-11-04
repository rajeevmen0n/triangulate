// Small helpers. All dynamic text goes through textContent (video-derived strings are untrusted).
export function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === false || v == null) continue;
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k === "text") n.textContent = v;
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) n.append(c.nodeType ? c : document.createTextNode(String(c)));
  return n;
}
export function svg(tag, attrs = {}, ...kids) {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  for (const c of kids.flat()) if (c) n.append(c);
  return n;
}
export async function api(path, opts = {}) {
  const r = await fetch("/api" + path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j); } catch { /* not JSON */ }
    const e = new Error(`${r.status}: ${msg}`); e.status = r.status; throw e;
  }
  return r.status === 204 ? null : r.json();
}
export const artifactUrl = (id, name) => `/api/jobs/${encodeURIComponent(id)}/artifacts/${name.split("/").map(encodeURIComponent).join("/")}`;
export const fmt = (x, d = 1) => (x == null || !Number.isFinite(+x) ? "n/a" : (+x).toFixed(d));
export function fmtDist(m) { return m == null ? "n/a" : m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${Math.round(m)} m`; }
export const pct = (p) => (p == null ? "n/a" : `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`);
export const coordText = (lat, lon) => `${(+lat).toFixed(6)}, ${(+lon).toFixed(6)}`;
export async function copy(text, btn) {
  try { await navigator.clipboard.writeText(text); } catch {
    const t = el("textarea", { "aria-hidden": "true" }); t.value = text; document.body.append(t); t.select();
    try { document.execCommand("copy"); } catch { /* ignore */ } t.remove();
  }
  if (btn) { const o = btn.textContent; btn.textContent = "Copied"; setTimeout(() => (btn.textContent = o), 1200); }
}
export const STAGES = ["ingest", "metadata", "spectrogram", "events", "adsb", "dem", "cpa", "match", "score", "visual", "fusion", "render"];
export const STAGE_LABEL = {
  ingest: "Read file", metadata: "Metadata hints", spectrogram: "Spectrogram", events: "Find aircraft events",
  adsb: "Flight tracks", dem: "Terrain", cpa: "Closest approach", match: "Match events to flights",
  score: "Score locations", visual: "Visual clues", fusion: "Combine evidence", render: "Render maps",
};
export const ACTIVE = new Set(["queued", "running"]);
