// Leaflet helpers. The basemap is optional (config.basemap_enabled); without it we draw on a plain ground.
export function makeMap(container, config, view = { c: [39.5, -98.35], z: 4 }) {
  const map = L.map(container, { zoomSnap: 0.25, attributionControl: true }).setView(view.c, view.z);
  map.attributionControl.setPrefix(false);
  if (config.basemap_enabled && config.basemap_url) {
    L.tileLayer(config.basemap_url, { maxZoom: 19, attribution: "&copy; OpenStreetMap contributors (ODbL)" }).addTo(map);
  } else {
    container.classList.add("nobase");
    map.attributionControl.addAttribution("No basemap (offline mode)");
  }
  return map;
}
