import { el, svg, api, artifactUrl, fmt } from "./util.js";

// Evidence panel: spectrogram PNG + SVG overlay (fitted curves, ridge points, event markers) and the event editor.
export async function evidencePanel(job, state, onRerun, getJson) {
  const id = job.id;
  const name = job.artifacts.find((a) => /^spectrogram.*\.json$/.test(a)) || job.artifacts.find((a) => /spectrogram.*\.json$/.test(a));
  const ov = name ? await getJson(id, name) : null;
  const card = el("div", { class: "card", style: "margin-top:16px" }, el("h3", {}, "Evidence: what the audio shows"));
  if (!ov || !ov.image) {
    card.append(el("p", { class: "muted" }, "No spectrogram was rendered for this analysis."));
    return card;
  }
  const W = ov.image.width_px, H = ov.image.height_px, win = ov.window;
  const imgName = ov.image.file.includes("/") || job.artifacts.includes(ov.image.file) ? ov.image.file : name.replace(/\.json$/, ".png");
  const toT = (x) => win.t0 + (x / W) * (win.t1 - win.t0);
  const g = (f) => (win.scale === "log" ? Math.log(f) : f);
  const gi = (v) => (win.scale === "log" ? Math.exp(v) : v);
  const toF = (y) => gi(g(win.fmax) - (y / H) * (g(win.fmax) - g(win.fmin)));
  const toX = (t) => ((t - win.t0) / (win.t1 - win.t0)) * W;
  const toY = (f) => ((g(win.fmax) - g(f)) / (g(win.fmax) - g(win.fmin))) * H;

  // editable events (full Event objects from events.json; the overlay alone is not enough to rerun)
  const raw = job.artifacts.includes("events.json") ? await getJson(id, "events.json") : null;
  const events = (Array.isArray(raw) ? raw : raw?.events) || null;
  const edit = events ? JSON.parse(JSON.stringify(events)) : null;
  let tracing = null;
  const show = { curves: true, ridge: true, events: true };

  const svgEl = svg("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", "aria-hidden": "true" });
  const wrap = el("div", { class: "spec" }, el("img", { src: artifactUrl(id, imgName), alt: "Spectrogram of the recording: time across, frequency up. Fitted aircraft Doppler curves are drawn on top." }), svgEl);
  const pl = (pts, attrs) => svg("polyline", { points: pts.map((p) => p.join(",")).join(" "), fill: "none", ...attrs });

  function draw() {
    svgEl.replaceChildren();
    if (show.events) for (const e of ov.events || []) {
      const sel = edit ? edit.find((x) => x.id === e.id)?.selected !== false : e.selected;
      const [x0, x1] = e.t_h_range_px;
      svgEl.append(svg("rect", { x: x0, y: 0, width: Math.max(1, x1 - x0), height: H, fill: "#fff", opacity: sel ? 0.12 : 0.04 }),
        svg("line", { x1: e.t_h_px, x2: e.t_h_px, y1: 0, y2: H, stroke: "#fff", "stroke-width": 1.5, "stroke-dasharray": "5 4", opacity: sel ? 1 : 0.4 }));
      const t = svg("text", { x: e.t_h_px + 4, y: 14, fill: "#fff", "font-size": 12, "font-family": "sans-serif" }); t.textContent = e.id; svgEl.append(t);
    }
    if (show.curves) for (const c of ov.curves || []) svgEl.append(pl(c.points_px, { stroke: "#ffd23f", "stroke-width": c.harmonic > 1 ? 1.2 : 2, opacity: c.harmonic > 1 ? 0.7 : 1 }));
    if (show.ridge) for (const r of ov.ridge_points || []) for (const [x, y] of r.points_px) svgEl.append(svg("circle", { cx: x, cy: y, r: 2, fill: "#3de0ff" }));
    for (const r of ov.traced_lines || []) svgEl.append(pl(r.points_px, { stroke: "#ff6bd6", "stroke-width": 2 }));
    for (const e of edit || []) if (e.traced_line?.length) {
      svgEl.append(pl(e.traced_line.map(([t, f]) => [toX(t), toY(f)]), { stroke: "#ff6bd6", "stroke-width": 2, "stroke-dasharray": "3 2" }));
      for (const [t, f] of e.traced_line) svgEl.append(svg("circle", { cx: toX(t), cy: toY(f), r: 3.5, fill: "#ff6bd6", stroke: "#000" }));
    }
  }
  wrap.addEventListener("click", (ev) => {
    if (!tracing) return;
    const r = svgEl.getBoundingClientRect();
    const x = ((ev.clientX - r.left) / r.width) * W, y = ((ev.clientY - r.top) / r.height) * H;
    const e = edit.find((q) => q.id === tracing);
    (e.traced_line ||= []).push([+toT(x).toFixed(3), +toF(y).toFixed(1)]);
    draw();
  });

  const tog = (key, label) => el("label", { class: "inline" }, el("input", { type: "checkbox", checked: true, onchange: (e) => { show[key] = e.target.checked; draw(); } }), label);
  const legend = el("div", { class: "legend" }, el("span", {}, el("i", { style: "border-color:#ffd23f" }), "fitted curve"), el("span", {}, el("i", { style: "border-color:#3de0ff;border-top-style:dotted" }), "detected ridge"),
    el("span", {}, el("i", { style: "border-color:#555;border-top-style:dashed" }), "event time and range"), el("span", {}, el("i", { style: "border-color:#ff6bd6" }), "your trace"));
  card.append(el("p", { class: "small muted" }, `Audio ${fmt(win.t0, 1)} to ${fmt(win.t1, 1)} s, ${fmt(win.fmin, 0)} to ${fmt(win.fmax, 0)} Hz (${win.scale} frequency axis).`),
    wrap, legend, el("div", { class: "row" }, tog("curves", "Fitted curves"), tog("ridge", "Ridge points"), tog("events", "Event markers")));

  // event parameter table (read-only values from the overlay) and the editor
  const evRows = (ov.events || []).map((e) => el("tr", {}, el("td", {}, e.id), el("td", { class: "r num" }, fmt(e.t_h, 1)), el("td", { class: "r num" }, fmt(e.f0, 0)), el("td", { class: "r num" }, fmt(e.v, 0)),
    el("td", { class: "r num" }, fmt(e.d, 0)), el("td", { class: "r num" }, fmt(e.snr, 1)), el("td", {}, (e.quality_flags || []).join(", "))));
  card.append(el("table", { style: "margin-top:10px" }, el("caption", { class: "small muted", style: "text-align:left" }, "Detected events"),
    el("thead", {}, el("tr", {}, ["Event", "Heard at (s)", "f0 (Hz)", "Speed (m/s)", "Closest dist (m)", "SNR", "Flags"].map((h, i) => el("th", { class: i && i < 6 ? "r" : "" }, h)))), el("tbody", {}, evRows)));

  if (!edit || !edit.length) {
    card.append(el("p", { class: "small muted" }, "Event editing is unavailable: this analysis has no events.json."));
    draw(); return card;
  }
  const traceBtns = {};
  const numIn = (e, key, i, label) => el("input", { type: "number", step: "any", value: e[key][i === 0 ? "lo" : "hi"], "aria-label": label, style: "width:90px",
    onchange: (ev) => { e[key][i === 0 ? "lo" : "hi"] = parseFloat(ev.target.value); } });
  const rows = edit.map((e) => {
    const tb = el("button", { type: "button", class: "btn small", "aria-pressed": "false", onclick: () => {
      tracing = tracing === e.id ? null : e.id;
      for (const [k, b] of Object.entries(traceBtns)) { b.setAttribute("aria-pressed", String(k === tracing)); b.textContent = k === tracing ? "Stop tracing" : "Trace"; }
      wrap.classList.toggle("tracing", !!tracing);
    } }, "Trace");
    traceBtns[e.id] = tb;
    return el("tr", {}, el("td", {}, el("input", { type: "checkbox", checked: e.selected !== false, "aria-label": `Use event ${e.id}`, onchange: (ev) => { e.selected = ev.target.checked; draw(); } })),
      el("td", {}, e.id),
      el("td", {}, numIn(e, "t_h_range", 0, `${e.id} earliest heard time (s)`), " to ", numIn(e, "t_h_range", 1, `${e.id} latest heard time (s)`)),
      el("td", {}, numIn(e, "f0_range", 0, `${e.id} lowest f0 (Hz)`), " to ", numIn(e, "f0_range", 1, `${e.id} highest f0 (Hz)`)),
      el("td", {}, tb, " ", el("button", { type: "button", class: "btn small", onclick: () => { e.traced_line = null; draw(); } }, "Clear")));
  });
  const status = el("span", { class: "small", role: "status" });
  const go = el("button", { type: "button", class: "btn primary", onclick: async () => {
    go.disabled = true; status.textContent = "Starting rerun...";
    try {
      const n = await api(`/jobs/${id}/rerun`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ events: edit }) });
      onRerun(n);
    } catch (x) { status.textContent = x.message; go.disabled = false; }
  } }, "Rerun with these events");
  card.append(el("h3", { style: "margin-top:16px" }, "Edit events and rerun"),
    el("p", { class: "small muted" }, "Untick events that are not aircraft, adjust the heard-time and f0 ranges, or press Trace and click along the Doppler glide on the spectrogram. Cached spectrogram, flight and terrain data are reused."),
    el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "Use"), el("th", {}, "Event"), el("th", {}, "Heard time range (s)"), el("th", {}, "f0 range (Hz)"), el("th", {}, "Trace"))), el("tbody", {}, rows)),
    el("div", { class: "row", style: "margin-top:10px" }, go, status));
  draw();
  return card;
}
