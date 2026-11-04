import { el, api, artifactUrl, fmt, fmtDist, pct, coordText, copy } from "./util.js";
import { makeMap } from "./map.js";
import { evidencePanel } from "./evidence.js";

const getJson = async (id, name) => { try { const r = await fetch(artifactUrl(id, name)); return r.ok ? await r.json() : null; } catch { return null; } };

export async function renderResults(root, job, state, onRerun) {
  const id = job.id, has = (n) => job.artifacts.includes(n);
  const result = await api(`/jobs/${id}/result`).catch(() => ({}));
  const gj = has("candidates.geojson") ? await getJson(id, "candidates.geojson") : null;
  const hm = has("heatmap.json") ? await getJson(id, "heatmap.json") : null;
  const feats = gj?.features || [];
  let cands = feats.filter((f) => f.properties?.kind === "candidate").map((f) => ({ ...f.properties, lon: f.geometry.coordinates[0], lat: f.geometry.coordinates[1] }));
  if (!cands.length && Array.isArray(result.candidates)) cands = result.candidates;
  cands.sort((a, b) => (a.rank ?? 0) - (b.rank ?? 0));
  const hpdPolys = feats.filter((f) => f.properties?.kind === "hpd");
  const hpdSummary = result.hpd || [];
  const attribution = [...new Set([...(state.config.attribution || []), ...(gj?.attribution || []), ...(result.attribution || [])])];

  // --- uncertainty warnings (plan section 7: never a pin without its region; warn on single-event/low margin)
  const warns = [];
  if (!cands.length) warns.push("No candidate locations were produced. Check the notes below and the evidence panel.");
  if (cands[0]?.margin_to_runner_up != null && cands[0].margin_to_runner_up < 0.5) warns.push("Low margin: the top two candidates are nearly equally likely.");
  const nSel = (await getJson(id, "events.json"));
  const evList = Array.isArray(nSel) ? nSel : nSel?.events;
  if (evList && evList.filter((e) => e.selected !== false).length === 1) warns.push("Only one aircraft event was used. Single-event results are much less certain.");
  // results.json (triangulate.results/1) carries its own warnings; the single-event/low-margin ones are already derived above
  for (const n of result.warnings || []) if (!/one aircraft event|Low margin/i.test(n)) warns.push(String(n));
  const flightInfo = result.flights || {};

  const banner = el("div", { class: "warn", role: "note" }, el("strong", {}, "Read the region, not the pin. "),
    "Every candidate below is a probability, not a fix. The shaded areas hold 50% and 90% of the probability.",
    warns.length ? el("ul", { style: "margin:6px 0 0" }, warns.map((w) => el("li", {}, w))) : null);

  // --- map
  const mapDiv = el("div", { class: "map tall", role: "application", "aria-label": "Result map. Candidates are listed in the table as an alternative." });
  const bbox = job.params?.constraints?.bbox;
  const map = makeMap(mapDiv, state.config);
  const layerGroup = L.layerGroup().addTo(map);
  if (bbox) L.rectangle([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], { color: "#14202a", weight: 1.5, dashArray: "6 4", fill: false, interactive: false }).addTo(map);
  let overlay = null;
  if (hm && has("heatmap.png")) {
    overlay = L.imageOverlay(artifactUrl(id, "heatmap.png"), hm.leaflet_bounds, { opacity: 0.7, alt: "Probability heatmap" }).addTo(map);
  }
  for (const f of hpdPolys) {
    const m = f.properties.mass;
    L.geoJSON(f, { style: { color: m <= 0.5 ? "#7a0f3a" : "#14202a", weight: 2, dashArray: m <= 0.5 ? null : "5 4", fillOpacity: 0 }, interactive: false }).addTo(layerGroup);
  }
  const markers = [];
  let selected = 0;
  const table = el("tbody");
  const detail = el("div", { class: "card", "aria-live": "polite" });

  function select(i, pan = true) {
    selected = i;
    markers.forEach((m, k) => m.getElement()?.querySelector(".cmark")?.classList.toggle("sel", k === i));
    [...table.children].forEach((tr, k) => tr.classList.toggle("sel", k === i));
    if (cands[i]) { drawDetail(cands[i]); if (pan) map.panTo([cands[i].lat, cands[i].lon]); }
  }
  cands.forEach((c, i) => {
    const m = L.marker([c.lat, c.lon], { keyboard: false, title: `Candidate ${c.rank}`, icon: L.divIcon({ className: "", html: `<div class="cmark">${c.rank}</div>`, iconSize: [24, 24] }) }).addTo(map);
    m.on("click", () => select(i, false)); markers.push(m);
    if (c.mass_radius_m) L.circle([c.lat, c.lon], { radius: c.mass_radius_m, color: "#b0245a", weight: 1, fillOpacity: 0.05, interactive: false }).addTo(layerGroup);
    const cp = el("button", { type: "button", class: "btn small", "aria-label": `Copy coordinates of candidate ${c.rank}`, onclick: (e) => { e.stopPropagation(); copy(coordText(c.lat, c.lon), e.currentTarget); } }, "Copy");
    const pick = el("button", { type: "button", class: "btn small", onclick: () => select(i) }, "Show");
    table.append(el("tr", { class: "cand", onclick: () => select(i) },
      el("td", { class: "num" }, String(c.rank)),
      el("td", { class: "num" }, el("span", { class: "mass", "aria-hidden": "true" }, el("i", { style: `width:${Math.round((c.prob_mass ?? 0) * 100)}%` })), pct(c.prob_mass)),
      el("td", { class: "num" }, c.mass_radius_m ? `±${fmtDist(c.mass_radius_m)}` : "n/a"),
      el("td", {}, el("span", { class: "coord" }, coordText(c.lat, c.lon))),
      el("td", {}, el("div", { class: "row" }, pick, cp))));
  });
  const fit = bbox ? [[bbox[0], bbox[1]], [bbox[2], bbox[3]]] : hm?.leaflet_bounds;

  const opacity = el("input", { type: "range", min: 0, max: 100, value: 70, id: "op", "aria-label": "Heatmap opacity", oninput: (e) => overlay?.setOpacity(e.target.value / 100) });
  const toggleHpd = el("input", { type: "checkbox", id: "hpd", checked: true, onchange: (e) => (e.target.checked ? layerGroup.addTo(map) : map.removeLayer(layerGroup)) });
  const mapCard = el("div", { class: "card" }, el("h3", {}, "Where it might be"), mapDiv,
    el("div", { class: "row", style: "margin-top:8px" },
      overlay ? el("label", { class: "inline" }, "Heatmap opacity", opacity) : el("span", { class: "muted small" }, "No heatmap in this result."),
      el("label", { class: "inline" }, toggleHpd, "Show probability regions (solid 50%, dashed 90%)")));

  function drawDetail(c) {
    const contribs = Object.entries(c.contributions || {});
    detail.replaceChildren(...[el("h3", {}, `Candidate ${c.rank}`),
      el("dl", { class: "kv" },
        el("dt", {}, "Coordinates"), el("dd", {}, el("span", { class: "coord" }, coordText(c.lat, c.lon)), " ", el("button", { type: "button", class: "btn small", onclick: (e) => copy(coordText(c.lat, c.lon), e.currentTarget) }, "Copy")),
        el("dt", {}, "Probability mass"), el("dd", {}, pct(c.prob_mass)),
        el("dt", {}, "Uncertainty radius"), el("dd", {}, c.mass_radius_m ? `±${fmtDist(c.mass_radius_m)}` : "n/a"),
        el("dt", {}, "Smallest region holding it"), el("dd", {}, c.hpd_level != null ? pct(c.hpd_level) + " HPD" : "n/a"),
        el("dt", {}, "Margin over next"), el("dd", {}, c.margin_to_runner_up != null ? `${fmt(c.margin_to_runner_up, 2)} nats` : "n/a"),
        el("dt", {}, "Recording start (UTC)"), el("dd", {}, c.t_s ? String(c.t_s) : "n/a"),
        el("dt", {}, "Flights"), el("dd", {}, c.flights?.length ? el("span", { class: "chips" }, c.flights.map((f) => el("span", { class: "chip" }, flightInfo[f]?.reg ? `${f} (${flightInfo[f].reg}${flightInfo[f].typecode ? ", " + flightInfo[f].typecode : ""})` : f))) : "none matched")),
      contribs.length ? el("table", {}, el("caption", { class: "small muted", style: "text-align:left" }, "Evidence by module (weighted log-likelihood)"),
        el("tbody", {}, contribs.map(([k, v]) => el("tr", {}, el("td", {}, k), el("td", { class: "r num" }, fmt(v, 2)))))) : null,
      c.notes?.length ? el("ul", {}, c.notes.map((n) => el("li", {}, String(n)))) : null].filter(Boolean));
  }

  const ledger = el("div", { class: "card" }, el("h3", {}, "Ranked candidates"),
    cands.length ? el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "#"), el("th", {}, "Probability"), el("th", {}, "Radius"), el("th", {}, "Coordinates"), el("th", {}, el("span", { class: "sr" }, "Actions")))), table)
      : el("p", { class: "muted" }, "No candidates."),
    hpdSummary.length ? el("p", { class: "small muted" }, hpdSummary.map((h) => `${Math.round(h.mass * 100)}% region: ${fmt(h.area_km2, 2)} km2`).join("; ")) : null);

  // --- extras: module status, metadata hints, visual clues
  const extras = [];
  if (result.module_status) extras.push(el("p", { class: "small" }, "Modules: ", Object.entries(result.module_status).map(([k, v]) => `${k} ${v}`).join("; ")));
  const meta = result.metadata || (has("metadata.json") ? await getJson(id, "metadata.json") : null);
  if (meta && !meta.stub && !meta.skipped) {
    // WP13 MetadataReport: gps[], times[], make/model/encoder, reencode_markers
    const hints = Array.isArray(meta.hints) ? meta.hints : [
      ...(meta.gps || []).map((g) => ({ key: `GPS (${g.source})`, value: `${coordText(g.lat, g.lon)}${g.accuracy_m ? ` ±${fmtDist(g.accuracy_m)}` : ""}` })),
      ...(meta.times || []).map((t) => ({ key: `Time (${t.field})`, value: `${t.naive}${t.offset_min != null ? ` offset ${t.offset_min} min` : " (no time zone)"}` })),
      ...["make", "model", "encoder"].filter((k) => meta[k]).map((k) => ({ key: k, value: meta[k] })),
      ...((meta.reencode_markers || []).length ? [{ key: "re-encode markers", value: meta.reencode_markers.join(", ") }] : []),
    ];
    extras.push(el("div", { class: "card" }, el("h3", {}, "Metadata hints ", el("span", { class: "badge untrusted" }, "untrusted")),
      el("p", { class: "small muted" }, "Embedded file metadata can be wrong or forged. It is never treated as proof."),
      el("dl", { class: "kv" }, hints.slice(0, 40).flatMap((h) => [el("dt", {}, String(h.key ?? h.name ?? "hint")), el("dd", {}, typeof (h.value ?? h) === "object" ? JSON.stringify(h.value ?? h) : String(h.value ?? h))]))));
  }
  const clues = result.clues || result.visual?.clues || (has("clues.json") ? await getJson(id, "clues.json") : null);
  const clueList = Array.isArray(clues) ? clues : (clues?.clues || clues?.report?.clues);
  if (clueList?.length) {
    extras.push(el("div", { class: "card" }, el("h3", {}, "Visual clues"), el("p", { class: "small muted" }, "Read by a language model from sampled frames. Treat each as a lead to verify."),
      clueList.map((c) => {
        const crop = c.crop || c.frame || c.image;
        const src = typeof crop === "string" && has(crop) ? artifactUrl(id, crop) : null;
        return el("div", { class: "clue" }, src ? el("img", { src, alt: `Frame crop for ${c.text || c.kind || "clue"}` }) : el("span"),
          el("div", {}, el("strong", {}, String(c.kind || c.type || "clue")), c.confidence != null ? el("span", { class: "muted" }, ` (confidence ${fmt(c.confidence, 2)})`) : null,
            el("div", {}, String(c.text || c.description || c.value || JSON.stringify(c)))));
      })));
  }

  const dl = el("p", { class: "small" }, "Downloads: ", job.artifacts.filter((a) => /\.(json|geojson|png|tif)$/.test(a)).map((a) => [el("a", { href: artifactUrl(id, a), download: a }, a), " "]));
  const ev = await evidencePanel(job, state, onRerun, getJson);

  root.replaceChildren(banner,
    el("div", { class: "results", style: "margin-top:16px" }, mapCard, el("div", { style: "display:grid;gap:16px" }, ledger, detail)),
    ev, ...extras.map((x) => el("div", { style: "margin-top:16px" }, x)),
    el("div", { class: "card", style: "margin-top:16px" }, el("h3", {}, "Data and attribution"), el("ul", {}, attribution.map((a) => el("li", {}, a))), dl));
  map.invalidateSize();
  // fit only once the map div is in the DOM: on a detached (0x0) container Leaflet computes a NaN centre
  if (fit) map.fitBounds(fit, { padding: [20, 20] });
  if (cands.length) select(0, false);
}
