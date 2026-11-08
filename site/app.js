/* Valencia flood map
 *
 * The right-hand map carries the model layers; the left exists only for the
 * optional radar swipe. Compare is OFF by default, so the map opens as a
 * single post-event view over labelled basemap tiles.
 *
 * Things that differ from the build brief, because the brief did not match
 * what is on disk or on the network:
 *   - footprint attributes are prob_mean / conf_best / area_m2 /
 *     landcover_name, not mean_prob / conf_class / area_ha;
 *   - image bounds live at stats.json -> layers[file].bounds_wgs84 as
 *     [W, S, E, N], not at layers.bounds as SW/NE lat-lng pairs;
 *   - the compare plugin is @maplibre/maplibre-gl-compare@0.5.0; the
 *     unscoped maplibre-gl-compare@0.6.0 in the brief does not exist on npm;
 *   - MapLibre has no raster-color at any version, so the radar swath is
 *     masked with a 1-bit PNG composited onto a canvas (see maskedCanvas).
 */
'use strict';

const WEBMAP = 'webmap/';
const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------- basemap */
/* OpenFreeMap "positron": light neutral vector tiles, genuinely keyless and
 * unmetered. Both options in the brief were dead ends -- Stadia needs an API
 * key on any non-localhost domain, and CARTO's basemaps now stamp
 * "API KEY REQUIRED" across the tiles. Esri World Light Gray is the keyless
 * raster fallback if OpenFreeMap is unreachable. */
const BASEMAP_URL = 'https://tiles.openfreemap.org/styles/positron';

const BASEMAP_FALLBACK = {
  version: 8,
  sources: {
    esri: {
      type: 'raster',
      tiles: ['https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/' +
              'World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'],
      tileSize: 256, maxzoom: 16, attribution: 'Tiles &copy; Esri'
    }
  },
  layers: [{ id: 'esri', type: 'raster', source: 'esri' }]
};

const EXTRA_ATTRIB =
  'Copernicus Sentinel-1 2024 | ' +
  '<a href="https://rapidmapping.emergency.copernicus.eu/EMSR773/">EMS EMSR773</a>';

const CONF_COLOR = { high: '#0d9488', medium: '#f59e0b',
                     low: '#ea580c', unclassed: '#9ca3af' };
const CONF_LABEL = { high: 'High', medium: 'Medium', low: 'Low' };
const CONF_ERR = { high: '1%', medium: '46%', low: '66%' };

/* Bounds arrive as [W, S, E, N]; an image source wants the four corners as
 * [[TL],[TR],[BR],[BL]] in [lng, lat]. */
const corners = ([w, s, e, n]) => [[w, n], [e, n], [e, s], [w, s]];
const CANVAS_MAX_W = window.matchMedia('(max-width:720px)').matches ? 2048 : 3852;
const isPhone = () => window.matchMedia('(max-width:720px)').matches;
const fmt = (n, d = 0) => n.toLocaleString('en-GB', { minimumFractionDigits: d,
                                                      maximumFractionDigits: d });

/* layers are restacked in this order after anything is added, so a lazily
 * loaded source never ends up covering the overlays above it */
const STACK_AFTER = ['radar-post', 'confidence', 'footprint-fill', 'footprint-line',
                     'reference-hit', 'reference-line', 'roads-line',
                     'buildings-circle', 'study-area-line'];
const STACK_BEFORE = ['radar-pre', 'study-area-line'];

const state = { masked: true, loaded: {}, busy: new Set(), radar: null };
let beforeMap, afterMap, compare, STATS = null, STUDY = null;

function busy(key, text) {
  if (text) { state.busy.add(key); $('loading-text').textContent = text; }
  else state.busy.delete(key);
  $('loading').hidden = state.busy.size === 0;
}

function fail(msg, err) {
  console.error(msg, err || '');
  $('loading-text').textContent = msg;
  $('loading').hidden = false;
  const sp = $('loading').querySelector('.spinner');
  if (sp) sp.style.display = 'none';
}

function restack() {
  STACK_AFTER.forEach((id) => { if (afterMap.getLayer(id)) afterMap.moveLayer(id); });
  STACK_BEFORE.forEach((id) => { if (beforeMap.getLayer(id)) beforeMap.moveLayer(id); });
}

/* ------------------------------------------------- reset-view control */
class ResetViewControl {
  constructor(bounds) { this._b = bounds; }
  onAdd() {
    const c = document.createElement('div');
    c.className = 'maplibregl-ctrl maplibregl-ctrl-group';
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'reset-view-btn';
    b.title = 'Reset view to the study area';
    b.setAttribute('aria-label', 'Reset view to the study area');
    b.innerHTML =
      '<svg viewBox="0 0 24 24" fill="none" stroke="#333" stroke-width="2" ' +
      'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      '<path d="M3 10.5 12 3l9 7.5"/><path d="M5.5 9.5V20h13V9.5"/>' +
      '<path d="M9.5 20v-6h5v6"/></svg>';
    b.addEventListener('click', () => {
      [afterMap, beforeMap].forEach((m) =>
        m.fitBounds(this._b, { padding: 24, duration: 800 }));
    });
    c.appendChild(b);
    this._c = c;
    return c;
  }
  onRemove() { this._c.remove(); }
}

/* ---------------------------------------------------------------- boot */
(async function init() {
  let style = BASEMAP_FALLBACK;
  try {
    const r = await fetch(BASEMAP_URL, { mode: 'cors' });
    if (r.ok) style = BASEMAP_URL;
    else console.warn('OpenFreeMap returned', r.status, '— using Esri fallback');
  } catch (e) {
    console.warn('OpenFreeMap unreachable — using Esri fallback:', e.message);
  }

  const common = {
    style, center: [-0.40, 39.45], zoom: 10.4,
    minZoom: 8, maxZoom: 17, attributionControl: false
  };
  beforeMap = new maplibregl.Map({ container: 'before', ...common });
  afterMap = new maplibregl.Map({ container: 'after', ...common });

  afterMap.addControl(new maplibregl.AttributionControl({
    compact: true, customAttribution: EXTRA_ATTRIB }));
  afterMap.addControl(
    new maplibregl.NavigationControl({ showCompass: false }), 'top-left');
  afterMap.addControl(new maplibregl.ScaleControl({ maxWidth: 110, unit: 'metric' }));

  compare = new maplibregl.Compare(beforeMap, afterMap, '#comparison-container', {});
  window.__map = { beforeMap, afterMap, compare, state };   // debugging handle

  /* Resolve as soon as the style is usable. Waiting on 'load' alone is
   * fragile: it also waits on sprites and glyphs from the remote style, and
   * never fires if one of those 404s, which would leave the whole map blank. */
  const ready = (m, capMs = 12000) => new Promise((res) => {
    const t0 = Date.now();
    m.once('load', res);
    (function tick() {
      if (m.isStyleLoaded() || Date.now() - t0 > capMs) return res();
      setTimeout(tick, 120);
    })();
  });

  try {
    const [stats] = await Promise.all([
      fetch(WEBMAP + 'stats.json').then((r) => r.json()),
      ready(beforeMap), ready(afterMap)
    ]);
    STATS = stats;
    build(stats);
  } catch (err) {
    fail('Failed to load map data — see console.', err);
  }
})();

/* ------------------------------------------------------------- helpers */
const loadImg = (url) => new Promise((res, rej) => {
  const i = new Image();
  i.onload = () => res(i);
  i.onerror = () => rej(new Error('could not load ' + url));
  i.src = url;
});

/** JPEG + 1-bit mask -> an RGBA canvas with the off-swath area transparent. */
async function maskedCanvas(jpg, maskPng) {
  const [img, mask] = await Promise.all([
    loadImg(WEBMAP + jpg), loadImg(WEBMAP + maskPng)]);
  const sc = Math.min(1, CANVAS_MAX_W / img.naturalWidth);
  const w = Math.max(1, Math.round(img.naturalWidth * sc));
  const h = Math.max(1, Math.round(img.naturalHeight * sc));

  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  const base = ctx.getImageData(0, 0, w, h);

  const mc = document.createElement('canvas');
  mc.width = w; mc.height = h;
  const mctx = mc.getContext('2d', { willReadFrequently: true });
  mctx.imageSmoothingEnabled = false;          // keep the swath edge crisp
  mctx.drawImage(mask, 0, 0, w, h);
  const md = mctx.getImageData(0, 0, w, h).data;

  const d = base.data;
  for (let i = 0; i < d.length; i += 4) d[i + 3] = md[i] > 127 ? 255 : 0;
  ctx.putImageData(base, 0, 0);
  return cv;
}

async function addRadar(map, id, jpg, maskPng, bnds) {
  let canvas;
  try {
    canvas = await maskedCanvas(jpg, maskPng);
  } catch (err) {
    console.warn('swath mask failed, showing the raw JPEG:', err.message);
    state.masked = false;
    map.addSource(id, { type: 'image', url: WEBMAP + jpg, coordinates: corners(bnds) });
    map.addLayer({ id, type: 'raster', source: id,
                   layout: { visibility: 'none' },
                   paint: { 'raster-fade-duration': 0 } });
    return;
  }
  map.addSource(id, { type: 'canvas', canvas, coordinates: corners(bnds), animate: false });
  map.addLayer({ id, type: 'raster', source: id,
                 layout: { visibility: 'none' },
                 paint: { 'raster-fade-duration': 0 } });
}

/* Two full-resolution RGBA canvases cost ~90 MB, and neither radar is on
 * screen at load now that compare defaults off, so they are built on first
 * demand rather than at init. */
function ensureRadar() {
  if (state.radar) return state.radar;
  const L = STATS.layers;
  busy('radar', 'Preparing radar imagery…');
  state.radar = Promise.all([
    addRadar(beforeMap, 'radar-pre', 'pre_radar.jpg', 'pre_radar_mask.png',
             L['pre_radar.jpg'].bounds_wgs84),
    addRadar(afterMap, 'radar-post', 'post_radar.jpg', 'post_radar_mask.png',
             L['post_radar.jpg'].bounds_wgs84)
  ]).then(() => {
    restack();
    busy('radar');
    console.log(`radar ready, swath mask ${state.masked ? 'applied' : 'FAILED'}`);
  }).catch((err) => {
    busy('radar');
    state.radar = null;
    throw err;
  });
  return state.radar;
}

function setVisible(map, ids, on) {
  ids.forEach((id) => {
    if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', on ? 'visible' : 'none');
  });
}

const confMatch = (prop) => [
  'match', ['get', prop],
  'high', CONF_COLOR.high, 'medium', CONF_COLOR.medium,
  'low', CONF_COLOR.low, CONF_COLOR.unclassed
];

/* Point-sampling of confidence.png.
 *
 * The footprint's conf_best attribute is the best confidence class anywhere in
 * the polygon, and polygons run to 145 km², so using it in the popup would
 * label a whole floodplain "high confidence" on the strength of one pixel.
 * Sampling the raster at the clicked coordinate answers the question the user
 * is actually asking: how much can I trust THIS spot.
 *
 * The raster is EPSG:4326 with a linear transform, so lng/lat maps to pixel
 * arithmetically. Sampled at 2048 px wide (~19 m) to keep the extra texture
 * to ~12 MB rather than the 44 MB a native-resolution copy would cost. */
const confSampler = { ready: false };

async function buildConfSampler(bnds) {
  const img = await loadImg(WEBMAP + 'confidence.png');
  const w = Math.min(2048, img.naturalWidth);
  const h = Math.round(img.naturalHeight * (w / img.naturalWidth));
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = false;            // never blend class colours
  ctx.drawImage(img, 0, 0, w, h);
  Object.assign(confSampler, { ready: true, ctx, w, h,
    W: bnds[0], S: bnds[1], E: bnds[2], N: bnds[3] });
}

function sampleConfidence(lngLat) {
  if (!confSampler.ready) return null;
  const { ctx, w, h, W, S, E, N } = confSampler;
  const x = Math.floor((lngLat.lng - W) / (E - W) * w);
  const y = Math.floor((N - lngLat.lat) / (N - S) * h);
  if (x < 0 || y < 0 || x >= w || y >= h) return null;
  const [r, g, b, a] = ctx.getImageData(x, y, 1, 1).data;
  if (a < 40) return 'none';
  let best = null, bd = Infinity;
  for (const [k, hex] of Object.entries(CONF_COLOR)) {
    if (k === 'unclassed') continue;
    const cr = parseInt(hex.slice(1, 3), 16), cg = parseInt(hex.slice(3, 5), 16),
          cb = parseInt(hex.slice(5, 7), 16);
    const d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2;
    if (d < bd) { bd = d; best = k; }
  }
  return bd < 3000 ? best : 'none';
}

/* --------------------------------------------------------------- build */
function build(s) {
  const L = s.layers;
  const bb = s.study_area.bbox_wgs84;                   // [W, S, E, N]
  STUDY = [[bb[0], bb[1]], [bb[2], bb[3]]];

  /* legend swatches are painted from stats.json so they cannot drift away
   * from the colours actually baked into confidence.png */
  const cls = (L['confidence.png'] && L['confidence.png'].classes) || {};
  document.querySelectorAll('.key[data-conf]').forEach((el) => {
    el.style.background = cls[el.dataset.conf] || CONF_COLOR[el.dataset.conf];
  });

  afterMap.addControl(new ResetViewControl(STUDY), 'top-right');

  /* study-area rectangle, on both maps, always on */
  const rect = {
    type: 'Feature', properties: {},
    geometry: { type: 'Polygon', coordinates: [[
      [bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]], [bb[0], bb[1]]
    ]] }
  };
  [afterMap, beforeMap].forEach((m) => {
    m.addSource('study-area', { type: 'geojson', data: rect });
    m.addLayer({
      id: 'study-area-line', type: 'line', source: 'study-area',
      paint: { 'line-color': '#334155', 'line-width': 1.5,
               'line-dasharray': [6, 4] }
    });
  });

  afterMap.addSource('confidence', {
    type: 'image', url: WEBMAP + 'confidence.png',
    coordinates: corners(L['confidence.png'].bounds_wgs84)
  });
  afterMap.addLayer({
    id: 'confidence', type: 'raster', source: 'confidence',
    layout: { visibility: 'none' },
    paint: { 'raster-opacity': 0.85, 'raster-fade-duration': 0,
             'raster-resampling': 'nearest' }
  });
  buildConfSampler(L['confidence.png'].bounds_wgs84)
    .catch((e) => console.warn('confidence sampler unavailable:', e.message));

  wireControls();
  renderExposure(s);
  renderValidation(s);
  renderMethods(s);
  setCompare(false);

  busy('vectors', 'Loading flood footprint…');
  Promise.all([
    fetch(WEBMAP + 'flood_footprint.geojson').then((r) => r.json()),
    fetch(WEBMAP + 'flood_reference.geojson').then((r) => r.json())
  ]).then(([foot, ref]) => {
    afterMap.addSource('footprint', { type: 'geojson', data: foot });
    afterMap.addLayer({
      id: 'footprint-fill', type: 'fill', source: 'footprint',
      paint: {
        'fill-color': '#3b82f6',
        'fill-opacity': ['interpolate', ['linear'], ['get', 'prob_mean'],
                         0.5, 0.25, 1.0, 0.5]
      }
    });
    afterMap.addLayer({
      id: 'footprint-line', type: 'line', source: 'footprint',
      paint: { 'line-color': '#1d4ed8', 'line-width': 0.4, 'line-opacity': 0.55 }
    });

    afterMap.addSource('reference', { type: 'geojson', data: ref });
    /* an invisible fill so a click can be tested against the reference:
     * queryRenderedFeatures cannot hit a line layer reliably, and a layer set
     * to visibility:none is not queryable at all */
    afterMap.addLayer({
      id: 'reference-hit', type: 'fill', source: 'reference',
      paint: { 'fill-opacity': 0 }
    });
    afterMap.addLayer({
      id: 'reference-line', type: 'line', source: 'reference',
      layout: { visibility: 'none' },
      paint: { 'line-color': '#c026d3', 'line-width': 2, 'line-dasharray': [4, 3] }
    });

    restack();
    setVisible(afterMap, ['footprint-fill', 'footprint-line'], $('t-footprint').checked);
    setVisible(afterMap, ['reference-line'], $('t-reference').checked);
    busy('vectors');
    console.log(`footprint ${foot.features.length} polygons | ` +
                `reference ${ref.features.length} polygons`);
  }).catch((err) => fail('Failed to load flood layers — see console.', err));

  wirePopups();
}

/* ------------------------------------------------- compare on/off mode */
function setCompare(on) {
  const cc = $('comparison-container');
  cc.classList.toggle('no-compare', !on);
  $('mapwrap').classList.toggle('compare-on', on);
  $('row-radar').hidden = on;

  if (on) {
    ensureRadar().then(() => {
      setVisible(beforeMap, ['radar-pre'], true);
      setVisible(afterMap, ['radar-post'], true);
      compare.setSlider(cc.getBoundingClientRect().width / 2);
      beforeMap.resize(); afterMap.resize();
    }).catch((err) => {
      fail('Could not prepare the radar imagery — see console.', err);
      $('t-compare').checked = false;
      setCompare(false);
    });
  } else {
    compare.setSlider(0);                      // after map fills the container
    setVisible(beforeMap, ['radar-pre'], false);
    setVisible(afterMap, ['radar-post'], $('t-radar').checked);
    afterMap.resize();
  }
}

/* --------------------------------------------- lazy buildings and roads */
const LAZY = {
  buildings: {
    file: 'buildings_affected.geojson',
    note: 'Loading buildings (4 MB)…',
    layers: ['buildings-circle'],
    add(data) {
      afterMap.addSource('buildings', { type: 'geojson', data });
      afterMap.addLayer({
        id: 'buildings-circle', type: 'circle', source: 'buildings',
        paint: {
          'circle-color': confMatch('conf_class'),
          'circle-radius': ['interpolate', ['linear'], ['zoom'],
                            9, 1.6, 12, 2.6, 15, 5, 17, 7],
          'circle-stroke-width': ['interpolate', ['linear'], ['zoom'], 12, 0, 14, 0.6],
          'circle-stroke-color': 'rgba(255,255,255,.85)',
          'circle-opacity': 0.9
        }
      });
    }
  },
  roads: {
    file: 'roads_affected.geojson',
    note: 'Loading road network (12 MB)…',
    layers: ['roads-line'],
    add(data) {
      afterMap.addSource('roads', { type: 'geojson', data });
      afterMap.addLayer({
        id: 'roads-line', type: 'line', source: 'roads',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: {
          'line-color': confMatch('conf_class'),
          'line-width': ['interpolate', ['linear'], ['zoom'],
                         9, 1, 12, 1.5, 15, 3, 17, 4.5],
          'line-opacity': 0.95
        }
      });
    }
  }
};

async function toggleLazy(key, on, input) {
  const spec = LAZY[key];
  if (!on) { setVisible(afterMap, spec.layers, false); syncAssetLegend(); return; }
  if (state.loaded[key]) { setVisible(afterMap, spec.layers, true); syncAssetLegend(); return; }

  input.disabled = true;
  busy(key, spec.note);
  try {
    const data = await fetch(WEBMAP + spec.file).then((r) => r.json());
    spec.add(data);
    restack();
    state.loaded[key] = true;
    console.log(`${key}: ${data.features.length.toLocaleString()} features`);
  } catch (err) {
    console.error(key, err);
    input.checked = false;
    fail(`Could not load ${key} — see console.`);
    setTimeout(() => { $('loading').hidden = true; }, 2600);
  } finally {
    input.disabled = false;
    busy(key);
    syncAssetLegend();
  }
}

function syncAssetLegend() {
  $('legend-assets').hidden = !($('t-buildings').checked || $('t-roads').checked);
}

/* -------------------------------------------------------------- popups */
function wirePopups() {
  const popup = new maplibregl.Popup({ closeButton: true, maxWidth: '290px' });

  afterMap.on('mousemove', (e) => {
    const hit = afterMap.getLayer('footprint-fill') &&
      afterMap.queryRenderedFeatures(e.point, { layers: ['footprint-fill'] }).length;
    afterMap.getCanvas().style.cursor = hit ? 'pointer' : '';
  });

  afterMap.on('click', (e) => {
    if (!afterMap.getLayer('footprint-fill')) return;
    const f = afterMap.queryRenderedFeatures(e.point, { layers: ['footprint-fill'] })[0];
    if (!f) { popup.remove(); return; }
    const p = f.properties;

    const inRef = afterMap.getLayer('reference-hit') &&
      afterMap.queryRenderedFeatures(e.point, { layers: ['reference-hit'] }).length > 0;

    const prob = Number(p.prob_mean);
    const here = sampleConfidence(e.lngLat);
    const conf = here && here !== 'none' ? here : null;
    const m2 = Number(p.area_m2);
    const area = m2 > 1e6 ? `${fmt(m2 / 1e6, 2)} km²` : `${fmt(m2 / 1e4, 1)} ha`;

    const extra = [];
    if (state.loaded.buildings) {
      const b = afterMap.queryRenderedFeatures(e.point, { layers: ['buildings-circle'] })[0];
      if (b) extra.push(['Building here', b.properties.building_type || 'yes']);
    }
    if (state.loaded.roads) {
      const r = afterMap.queryRenderedFeatures(e.point, { layers: ['roads-line'] })[0];
      if (r) extra.push(['Road here', r.properties.highway]);
    }

    const sw = conf ? CONF_COLOR[conf] : CONF_COLOR.unclassed;
    const confText = conf
      ? `${CONF_LABEL[conf]} — this class is wrong ${CONF_ERR[conf]} of the time`
      : 'None — below 70% probability, no measurable skill here';

    popup.setLngLat(e.lngLat).setHTML(`
      <div class="pop">
        <h4><span class="key" style="background:${sw}"></span>
            Confidence at this point</h4>
        <p class="sub">${confText}</p>
        <div class="bar"><i style="width:${Math.round(prob * 100)}%"></i></div>
        <dl>
          <dt>Flood probability</dt><dd>${(prob * 100).toFixed(1)}%</dd>
          <dt>Land cover</dt><dd>${p.landcover_name}</dd>
          <dt>Patch area</dt><dd>${area}</dd>
          ${extra.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}
        </dl>
        <p class="sub">Probability and land cover are averages for the whole
           connected patch.</p>
        <div class="verdict ${inRef ? 'agree' : 'disagree'}">
          ${inRef ? '✓ Agrees with EMS reference' : '✗ Outside EMS reference'}
        </div>
      </div>`).addTo(afterMap);
  });
}

/* ------------------------------------------------------ exposure panel */
const EXPO_ROWS = [
  ['lower (HIGH only)', 'High confidence'],
  ['moderate (HIGH+MEDIUM)', 'High + Medium'],
  ['central (full RF footprint)', 'Full model footprint'],
  ['EMS reference (EMSR773)', 'EMS reference']
];

function renderExposure(s) {
  const byName = Object.fromEntries(s.exposure.estimates.map((e) => [e.estimate, e]));
  const prec = Object.fromEntries(s.exposure.vs_reference.map((v) => [v.estimate, v]));
  const pc = (v) => (v === undefined ? '—' : Math.round(v * 100) + '%');

  /* precision is one column holding both figures: five columns do not fit the
   * sidebar, and a horizontally scrolled table hides the number that matters */
  const body = EXPO_ROWS.map(([key, label]) => {
    const e = byName[key], v = prec[key];
    const isRef = key.startsWith('EMS');
    return `<tr class="${isRef ? 'ref' : ''}">
      <td>${label}</td>
      <td class="num">${fmt(e.buildings)}</td>
      <td class="num">${fmt(e.road_km, 1)}</td>
      <td class="num">${isRef ? '—'
        : `${pc(v && v.buildings.precision)} · ${pc(v && v.roads.precision)}`}</td>
      </tr>`;
  }).join('');

  $('exposure-table').innerHTML = `
    <thead><tr><th>Estimate</th><th>Bldgs</th><th>Road km</th>
      <th>Precision</th></tr></thead>
    <tbody>${body}</tbody>`;

  $('exposure-caveat').innerHTML =
    '<b>Precision is shown as buildings · roads</b>, against the official EMS ' +
    'reference. 90% of the footprint’s buildings fall in the 0.50–0.70 ' +
    'probability band, which carries no measurable skill above the base rate.';
}

/* ---------------------------------------------------- validation panel */
function renderValidation(s) {
  const by = Object.fromEntries(s.validation.by_method.map((m) => [m.method, m]));
  const r = by.random, sp = by.spatial_2km;
  const varRatio = Math.round(sp.sd.f1 / r.sd.f1);

  const W = 290, H = 150, PAD = 30, BASE = H - 26, TOP = 14;
  const scale = (v) => BASE - (v / 0.62) * (BASE - TOP);
  const bars = [
    { x: 54, v: r.mean.f1, sd: r.sd.f1, c: '#ef4444', l: 'Standard test' },
    { x: 162, v: sp.mean.f1, sd: sp.sd.f1, c: '#3b82f6', l: 'Stricter spatial test' }
  ];

  $('cv-chart').innerHTML = `
  <svg viewBox="0 0 ${W} ${H}" role="img"
       aria-label="Standard testing gives F1 ${r.mean.f1.toFixed(2)}; the stricter
       spatial test gives F1 ${sp.mean.f1.toFixed(2)} with much wider spread">
    ${[0, 0.2, 0.4, 0.6].map((g) => `
      <line x1="${PAD}" x2="${W - 8}" y1="${scale(g)}" y2="${scale(g)}"
            stroke="#e5e7eb" stroke-width="1"/>
      <text x="${PAD - 5}" y="${scale(g) + 3.5}" font-size="9" fill="#6b7280"
            text-anchor="end">${g.toFixed(1)}</text>`).join('')}
    ${bars.map((b) => `
      <rect x="${b.x}" y="${scale(b.v)}" width="74"
            height="${BASE - scale(b.v)}" fill="${b.c}" rx="2"/>
      <line x1="${b.x + 37}" x2="${b.x + 37}" y1="${scale(b.v + b.sd)}"
            y2="${scale(Math.max(0, b.v - b.sd))}" stroke="#111827" stroke-width="1.4"/>
      <line x1="${b.x + 27}" x2="${b.x + 47}" y1="${scale(b.v + b.sd)}"
            y2="${scale(b.v + b.sd)}" stroke="#111827" stroke-width="1.4"/>
      <line x1="${b.x + 27}" x2="${b.x + 47}" y1="${scale(Math.max(0, b.v - b.sd))}"
            y2="${scale(Math.max(0, b.v - b.sd))}" stroke="#111827" stroke-width="1.4"/>
      <text x="${b.x + 37}" y="${scale(b.v + b.sd) - 5}" font-size="11.5"
            font-weight="700" text-anchor="middle" fill="#111827">
        ${b.v.toFixed(2)}</text>
      <text x="${b.x + 37}" y="${BASE + 13}" font-size="9.5" text-anchor="middle"
            fill="#4b5563">${b.l}</text>`).join('')}
    <text x="4" y="10" font-size="9" fill="#6b7280">F1</text>
  </svg>`;

  $('cv-text').innerHTML =
    'The accuracy was tested in two ways. Standard testing gives ' +
    `<b>F1 = ${r.mean.f1.toFixed(2)}</b> — but this overstates real-world ` +
    `performance. A stricter spatial test gives <b>F1 = ${sp.mean.f1.toFixed(2)}</b>, ` +
    `with <b>${varRatio}× more uncertainty</b>. The model works best on wetland ` +
    'and cropland. It has no reliable signal in built-up areas.';

  $('cv-note').textContent =
    'F1 score measures how well predicted floods match the official reference ' +
    '(1.0 = perfect match).';
}

/* ------------------------------------------------------- methods panel */
function renderMethods(s) {
  const prov = s.provenance || [];
  const optical = (100 - s.reference.sentinel1_share_pct).toFixed(1);

  const sources = prov.map((p) => `
    <li><b>${p.name}</b>
      <span>${p.provider}${p.url
        ? ` · <a href="${p.url}" target="_blank" rel="noopener">link</a>` : ''}</span>
    </li>`).join('');

  $('methods-body').innerHTML = `
    <ul>
      <li>Sentinel-1 radar images from 20 Oct and 1 Nov 2024, processed to show
          backscatter change</li>
      <li>Random Forest classifier trained on radar change, terrain height,
          slope, and distance to rivers</li>
      <li>Validated against Copernicus Emergency Management Service
          (EMSR773) — ${optical}% derived from optical satellite imagery</li>
      <li>${s.imagery.lag_after_peak_hours}-hour gap between the reference and
          the radar image limits agreement in fast-draining areas</li>
    </ul>

    <h3>Data sources</h3>
    <ul class="src">${sources}</ul>

    <h3>More</h3>
    <ul>
      <li><a href="#">Full report (placeholder)</a></li>
      <li><a href="#">Source code (placeholder)</a></li>
    </ul>`;
}

/* ------------------------------------------------------------ controls */
function wireControls() {
  $('t-compare').addEventListener('change', (e) => setCompare(e.target.checked));

  $('t-radar').addEventListener('change', (e) => {
    if (!e.target.checked) { setVisible(afterMap, ['radar-post'], false); return; }
    e.target.disabled = true;
    ensureRadar()
      .then(() => setVisible(afterMap, ['radar-post'], $('t-radar').checked))
      .catch((err) => {
        fail('Could not prepare the radar imagery — see console.', err);
        e.target.checked = false;
      })
      .finally(() => { e.target.disabled = false; });
  });

  $('t-footprint').addEventListener('change', (e) =>
    setVisible(afterMap, ['footprint-fill', 'footprint-line'], e.target.checked));

  $('t-confidence').addEventListener('change', (e) => {
    setVisible(afterMap, ['confidence'], e.target.checked);
    $('legend-confidence').hidden = !e.target.checked;
  });

  $('t-reference').addEventListener('change', (e) =>
    setVisible(afterMap, ['reference-line'], e.target.checked));

  $('t-buildings').addEventListener('change', (e) =>
    toggleLazy('buildings', e.target.checked, e.target));

  $('t-roads').addEventListener('change', (e) =>
    toggleLazy('roads', e.target.checked, e.target));

  document.querySelectorAll('.card-head').forEach((btn) => {
    btn.addEventListener('click', () => {
      const open = btn.getAttribute('aria-expanded') === 'true';
      btn.setAttribute('aria-expanded', String(!open));
      $(btn.getAttribute('aria-controls')).hidden = open;
    });
  });

  const handle = $('sheet-handle'), sidebar = $('sidebar');
  handle.addEventListener('click', () => {
    const open = sidebar.classList.toggle('open');
    handle.setAttribute('aria-expanded', String(open));
  });
  afterMap.on('dragstart', () => {
    if (sidebar.classList.contains('open') && isPhone()) {
      sidebar.classList.remove('open');
      handle.setAttribute('aria-expanded', 'false');
    }
  });
}
