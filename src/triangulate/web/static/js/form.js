import { el, api } from "./util.js";
import { makeMap } from "./map.js";

const field = (label, input, hint) => el("label", {}, label, hint ? el("span", { class: "hint" }, hint) : null, input);
const num = (id, attrs = {}) => el("input", { type: "number", id, name: id, step: "any", ...attrs });

export function newAnalysis(root, state, onCreated) {
  const cfg = state.config;
  const bb = {};
  for (const k of ["s", "w", "n", "e"]) bb[k] = num("bb-" + k, { required: true, "aria-describedby": "bbox-help" });
  const mapDiv = el("div", { class: "map", id: "pick-map", role: "application", "aria-label": "Map for choosing the search region. The numeric fields below are an equivalent keyboard alternative." });
  const err = el("div", { class: "err", role: "alert", hidden: true });
  const file = el("input", { type: "file", id: "file", name: "file", required: true, accept: "audio/*,video/*,.wav,.mp3,.m4a,.mp4,.mov,.mkv,.webm,.flac,.ogg" });
  const date0 = el("input", { type: "date", id: "d0", required: true });
  const date1 = el("input", { type: "date", id: "d1" });
  const tw0 = el("input", { type: "time", id: "tw0" });
  const tw1 = el("input", { type: "time", id: "tw1" });
  const tz = el("input", { type: "text", id: "tz", placeholder: "auto from region", autocomplete: "off", spellcheck: "false" });
  const hobs = num("hobs", { value: cfg.default_h_obs_m ?? 5, min: 0 });
  const temp = num("temp", { placeholder: "default 343 m/s" });
  const kPos = num("kpos", { min: 0 });
  const kUtc = el("input", { type: "datetime-local", id: "kutc", step: 1 });
  const kSig = num("ksig", { value: 60, min: 1 });
  const visual = el("input", { type: "checkbox", id: "visual" });
  const consent = el("input", { type: "checkbox", id: "consent" });
  const consentBox = el("div", { class: "consent warn", hidden: true },
    el("p", { id: "vnotice" }, cfg.visual_notice),
    el("p", { class: "small" }, `Spending is capped at $${cfg.llm_budget_usd} per analysis.` + (cfg.llm_enabled ? "" : " No API key is configured on this server, so the visual module will be skipped.")),
    el("label", { class: "inline" }, consent, "I understand and consent to sending frames and OCR text from this upload to the Claude API."));
  const submit = el("button", { class: "btn primary", type: "submit" }, "Start analysis");
  const sync = () => {
    consentBox.hidden = !visual.checked;
    if (!visual.checked) consent.checked = false;
    submit.disabled = visual.checked && !consent.checked;
  };
  visual.addEventListener("change", sync);
  consent.addEventListener("change", sync);

  // --- map + bbox
  const map = makeMap(mapDiv, cfg);
  let rect = null;
  let drawing = false;
  let a0 = null;
  const setRect = (s, w, n, e) => {
    const b = [[s, w], [n, e]];
    if (rect) rect.setBounds(b); else rect = L.rectangle(b, { color: "#0b5a8c", weight: 2, fillOpacity: 0.08 }).addTo(map);
  };
  const readBox = () => ["s", "w", "n", "e"].map((k) => parseFloat(bb[k].value));
  const fromInputs = () => { const v = readBox(); if (v.every(Number.isFinite) && v[0] < v[2] && v[1] < v[3]) setRect(...v); };
  const fill = (s, w, n, e) => { [s, w, n, e].forEach((v, i) => (bb["swne"[i]].value = (+v).toFixed(5))); fromInputs(); };
  Object.values(bb).forEach((i) => i.addEventListener("input", fromInputs));
  const drawBtn = el("button", { type: "button", class: "btn", "aria-pressed": "false" }, "Draw box on map");
  const setDrawing = (on) => {
    drawing = on;
    drawBtn.setAttribute("aria-pressed", String(on));
    drawBtn.textContent = on ? "Drag on the map..." : "Draw box on map";
    mapDiv.style.cursor = on ? "crosshair" : "";
  };
  drawBtn.onclick = () => setDrawing(!drawing);
  map.on("mousedown", (e) => { if (drawing) { map.dragging.disable(); a0 = e.latlng; } });
  map.on("mousemove", (e) => { if (a0) { const b = L.latLngBounds(a0, e.latlng); setRect(b.getSouth(), b.getWest(), b.getNorth(), b.getEast()); } });
  map.on("mouseup", (e) => {
    if (!a0) return;
    const b = L.latLngBounds(a0, e.latlng);
    a0 = null;
    map.dragging.enable();
    if (b.getNorth() - b.getSouth() > 1e-4 && b.getEast() - b.getWest() > 1e-4) fill(b.getSouth(), b.getWest(), b.getNorth(), b.getEast());
    setDrawing(false);
  });
  const useView = el("button", { type: "button", class: "btn" }, "Use current map view");
  useView.onclick = () => { const b = map.getBounds(); fill(b.getSouth(), b.getWest(), b.getNorth(), b.getEast()); };

  function fail(m) { err.textContent = m; err.hidden = false; err.scrollIntoView({ block: "nearest" }); }
  async function onSubmit(ev) {
    ev.preventDefault();
    err.hidden = true;
    const [s, w, n, e] = readBox();
    if (!(s < n && w < e)) return fail("The search region needs south < north and west < east.");
    const c = { bbox: [s, w, n, e], date_start: date0.value, h_obs_m: parseFloat(hobs.value) };
    if (date1.value) c.date_end = date1.value;
    if (tz.value.trim()) c.tz = tz.value.trim();
    if (tw0.value) c.window_start = tw0.value;
    if (tw1.value) c.window_end = tw1.value;
    if (temp.value !== "") c.temperature_c = parseFloat(temp.value);
    if (kPos.value !== "" && kUtc.value) {
      c.known_audio_time = { audio_pos_s: parseFloat(kPos.value), utc: kUtc.value + (kUtc.value.length === 16 ? ":00" : "") + "Z", sigma_s: parseFloat(kSig.value) || 60 };
    }
    const fd = new FormData();
    fd.append("file", file.files[0]);
    fd.append("constraints", JSON.stringify(c));
    fd.append("visual", String(visual.checked));
    fd.append("visual_consent", String(visual.checked && consent.checked));
    submit.disabled = true;
    submit.textContent = "Uploading...";
    try { onCreated(await api("/jobs", { method: "POST", body: fd })); }
    catch (x) { fail(x.message); }
    finally { submit.textContent = "Start analysis"; sync(); }
  }

  const form = el("form", { onsubmit: onSubmit },
    el("h2", {}, "New analysis"),
    el("p", { class: "muted" }, "Tell us roughly where and when the recording was made. The tighter the region and date, the better and faster the answer."),
    el("fieldset", {}, el("legend", {}, "Recording"), field("Audio or video file", file, `Up to ${Math.round((cfg.max_upload_bytes || 0) / 1e6)} MB. The file stays on this machine.`)),
    el("fieldset", {}, el("legend", {}, "Search region (required)"),
      mapDiv,
      el("div", { class: "row", style: "margin:8px 0" }, drawBtn, useView),
      el("p", { id: "bbox-help", class: "small muted", style: "margin:0 0 8px" }, "Draw a box, pan and zoom then use the current view, or type decimal degrees."),
      el("div", { class: "grid" }, field("South (lat)", bb.s), field("West (lon)", bb.w), field("North (lat)", bb.n), field("East (lon)", bb.e))),
    el("fieldset", {}, el("legend", {}, "When"),
      el("div", { class: "grid" }, field("Date (required)", date0, "Local date of the recording"), field("Last date", date1, "Optional, up to 7 days"),
        field("Earliest local time", tw0), field("Latest local time", tw1), field("Time zone", tz, "IANA name, e.g. America/New_York"))),
    el("fieldset", {}, el("legend", {}, "Optional details"),
      el("div", { class: "grid" }, field("Observer height (m)", hobs), field("Air temperature (C)", temp),
        field("Audio position (s)", kPos, "Known clip time: seconds into the file"), field("UTC at that position", kUtc), field("Uncertainty (s)", kSig))),
    el("fieldset", {}, el("legend", {}, "Visual module"),
      el("label", { class: "inline" }, visual, "Also analyse video frames for visual clues (off by default)"), consentBox),
    err, el("div", { class: "row" }, submit));
  root.replaceChildren(el("section", {}, form));
  sync();
  setTimeout(() => map.invalidateSize(), 0);
}
