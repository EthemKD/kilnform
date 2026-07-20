import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { Viewer } from './viewer.js';
import { generateFromText } from './textTo3d.js';
import { generateFromImage } from './imageTo3d.js';
import { applyMaterial } from './materials.js';
import { voxelize, lowpoly, toon } from './stylize.js';
import { rememberBase, applyTransform, setSmoothShading } from './meshOps.js';
import { saveModel, listModels, getModel, deleteModel, renameModel } from './library.js';
import { exportModel, toGLB, download } from './exporter.js';
import * as ai from './aiClient.js';

const $ = (id) => document.getElementById(id);

const viewer = new Viewer($('viewport'));
const state = {
  model: null, pristine: null, prompt: '', variation: 0,
  engine: 'proc', aiReady: false, busy: false,
  // 128 | 256 | 320 (marching-cubes resolution) or 'ultra' (Hunyuan3D tier)
  detail: ((v) => (v === 'ultra' ? v : Number(v) || 256))(localStorage.getItem('kilnform-detail')),
  lastMake: null,   // { via, label, seconds } shown in the Model card
  lastSaveId: null, // history record the "Apply name" button renames
};

/* ---- helpers ---- */
function cloneModel(group) {
  const c = group.clone(true);
  c.traverse((o) => {
    if (o.isMesh) {
      o.geometry = o.geometry.clone();
      o.material = Array.isArray(o.material) ? o.material.map((m) => m.clone()) : o.material.clone();
    }
  });
  c.userData = JSON.parse(JSON.stringify({ parsed: group.userData.parsed || null }));
  return c;
}

function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { t.hidden = true; }, 2600);
}

function updateStats() {
  const n = viewer.polyCount();
  $('viewport-stats').textContent = n > 0 ? `${n.toLocaleString('en-US')} triangles` : '— triangles';
  $('empty-state').style.display = state.model ? 'none' : 'flex';
  updateModelInfo();
}

function updateModelInfo() {
  const el = $('model-info');
  if (!state.model) { el.textContent = 'The bench is empty.'; return; }
  let verts = 0;
  state.model.traverse((o) => { if (o.isMesh) verts += o.geometry?.attributes?.position?.count || 0; });
  const m = state.lastMake;
  const src = m ? `${m.via} · ${m.label}${m.seconds ? ` · ${m.seconds}s` : ''}` : '';
  el.innerHTML =
    `<b>${viewer.polyCount().toLocaleString('en-US')}</b> triangles · <b>${verts.toLocaleString('en-US')}</b> vertices` +
    (src ? `<br/>${src}` : '');
}

/** Every successful make lands in History automatically (capped at 40). */
async function autoSave(name) {
  try {
    const glb = await toGLB(state.model);
    const thumb = viewer.captureThumbnail();
    state.lastSaveId = await saveModel({ name, thumb, glb });
    $('save-name').value = name;
    const items = await listModels();
    for (const it of items.slice(40)) await deleteModel(it.id);
    refreshLibrary();
  } catch (err) {
    console.error(err); // history is best-effort; the bench still has the model
  }
}

function resetTransformUI() {
  $('tr-scale').value = 1; $('tr-scale-val').textContent = '1.0';
  $('tr-roty').value = 0; $('tr-roty-val').textContent = '0°';
  $('tr-posy').value = 0; $('tr-posy-val').textContent = '0.0';
  $('tr-smooth').checked = false;
}

/** Puts a new model on the bench; keeps a pristine copy for undo. */
function setActiveModel(group, { keepPristine = false } = {}) {
  if (!group) { toast('Nothing came out of that — try different settings.'); return; }
  if (!keepPristine) state.pristine = cloneModel(group);
  rememberBase(group);
  state.model = group;
  viewer.setModel(group);
  viewer.frameModel();
  resetTransformUI();
  updateStats();
  const label = group.userData.parsed?.typeLabel;
  if (label) $('save-name').value = label;
}

function restorePristine() {
  if (!state.pristine) return;
  setActiveModel(cloneModel(state.pristine), { keepPristine: true });
  toast('Back to the original.');
}

/* ---- AI engine ---- */
function setEngine(engine, remember = true) {
  state.engine = engine;
  $('engine-ai').classList.toggle('active', engine === 'ai');
  $('engine-proc').classList.toggle('active', engine === 'proc');
  $('variations').hidden = !state.prompt;
  $('proc-image-opts').hidden = engine === 'ai';
  $('ai-detail').hidden = engine !== 'ai';
  if (remember) localStorage.setItem('kilnform-engine', engine);
}
$('engine-ai').addEventListener('click', () => setEngine('ai'));
$('engine-proc').addEventListener('click', () => setEngine('proc'));

/* detail segment: three marching-cubes tiers plus the Hunyuan3D 'ultra' tier */
document.querySelectorAll('.seg-btn').forEach((b) => {
  b.classList.toggle('active', (b.dataset.res === 'ultra' ? 'ultra' : Number(b.dataset.res)) === state.detail);
  b.addEventListener('click', () => {
    state.detail = b.dataset.res === 'ultra' ? 'ultra' : Number(b.dataset.res);
    localStorage.setItem('kilnform-detail', String(state.detail));
    document.querySelectorAll('.seg-btn').forEach((x) => x.classList.toggle('active', x === b));
  });
});

async function pollAi(first = false) {
  const h = await ai.health();
  const badge = $('ai-badge');
  if (h) {
    state.aiReady = true;
    $('engine-ai').disabled = false;
    badge.className = 'hint ready';
    const dev = h.cuda ? h.device.replace('NVIDIA GeForce ', '') : 'CPU';
    badge.textContent = h.models === 'ready' ? `AI ready · ${dev}` : `AI found · ${dev} · models load on first make`;
    if (first) {
      const pref = localStorage.getItem('kilnform-engine');
      setEngine(pref === 'proc' ? 'proc' : 'ai', false);
      ai.warmup();
    }
  } else {
    state.aiReady = false;
    $('engine-ai').disabled = true;
    badge.className = 'hint err';
    badge.textContent = 'AI backend is off (run start.bat) — instant mode active';
    // a make can starve the 2s health timeout (xatlas pegs every core) —
    // never demote the engine mid-make on a single missed poll
    if (state.engine === 'ai' && !state.busy) setEngine('proc', false);
  }
}
pollAi(true);
setInterval(() => pollAi(false), 10000);

const STAGE_LABELS = {
  idle: 'getting ready',
  loading: 'loading models (first run takes a minute)',
  translating: 'translating the prompt',
  painting: 'painting the reference image',
  cutting: 'cutting out the subject',
  sculpting: 'sculpting the mesh',
  'sculpting-ultra': 'sculpting the mesh (Ultra, takes a while)',
  extracting: 'extracting geometry',
  texturing: 'baking the texture',
};
let statusTimer = null;
function showAiStatus() {
  const el = $('ai-status');
  el.hidden = false;
  const t0 = Date.now();
  let stage = 'idle';
  let tick = 0;
  clearInterval(statusTimer);
  statusTimer = setInterval(async () => {
    if (tick++ % 3 === 0) {
      const s = await ai.progress();
      if (s) stage = s;
    }
    const label = STAGE_LABELS[stage] || stage;
    el.innerHTML = `<span class="spin"></span>${label} · ${Math.round((Date.now() - t0) / 1000)}s`;
  }, 300);
}
function hideAiStatus() {
  clearInterval(statusTimer);
  $('ai-status').hidden = true;
}

let currentAbort = null;

function setBusy(busy, cancellable = false) {
  state.busy = busy;
  const g = $('btn-generate');
  const gi = $('btn-image-generate');
  if (busy && cancellable) {
    g.disabled = false; g.textContent = 'Cancel'; g.classList.add('cancel');
    gi.disabled = false; gi.textContent = 'Cancel'; gi.classList.add('cancel');
  } else {
    g.disabled = busy; g.textContent = 'Make it'; g.classList.remove('cancel');
    gi.disabled = busy || !(loadedImage || (loadedFile && state.engine === 'ai'));
    gi.textContent = 'Convert to model'; gi.classList.remove('cancel');
  }
}

/** Turns an AI GLB into a scene group: center, ground, scale to the bench. */
function loadGlbIntoScene(glbBuffer, label) {
  return new Promise((resolve, reject) => {
    new GLTFLoader().parse(glbBuffer, '', (gltf) => {
      const group = gltf.scene;
      const box = new THREE.Box3().setFromObject(group);
      const size = box.getSize(new THREE.Vector3());
      const scale = 2.2 / (Math.max(size.x, size.y, size.z) || 1);
      group.scale.setScalar(scale);
      const box2 = new THREE.Box3().setFromObject(group);
      const c = box2.getCenter(new THREE.Vector3());
      group.position.set(-c.x, -box2.min.y, -c.z);
      group.userData.parsed = { typeLabel: label, colorName: null, variation: 0 };
      resolve(group);
    }, reject);
  });
}

async function generateTextAI(prompt, seed) {
  currentAbort = new AbortController();
  setBusy(true, true);
  showAiStatus();
  try {
    const res = await ai.textTo3d(prompt, seed, state.detail, currentAbort.signal);
    const group = await loadGlbIntoScene(res.glb, prompt);
    state.lastMake = {
      via: state.detail === 'ultra' ? 'AI · Ultra' : 'AI',
      label: `${prompt} · V${seed + 1}`,
      seconds: res.seconds,
    };
    setActiveModel(group);
    $('ai-preview').src = res.previewUrl;
    $('ai-preview-wrap').hidden = false;
    $('variations').hidden = false;
    document.querySelectorAll('.var-card').forEach((c) =>
      c.classList.toggle('active', Number(c.dataset.seed) === seed));
    const pr = $('parse-result');
    pr.hidden = false;
    pr.innerHTML = `AI made: <b>${prompt}</b>${res.promptEn && res.promptEn !== prompt ? ` → <i>${res.promptEn}</i>` : ''} · V${seed + 1} · ${res.seconds}s`;
    toast(`AI model ready (${res.seconds}s).`);
    autoSave(`${prompt} · V${seed + 1}`);
  } catch (err) {
    if (err.name === 'AbortError') {
      toast('Cancelled — the bench is all yours.');
    } else {
      console.error(err);
      toast(`AI error: ${err.message}. Instant mode still works.`);
    }
  } finally {
    currentAbort = null;
    hideAiStatus();
    setBusy(false);
  }
}

async function generateImageAI(file) {
  currentAbort = new AbortController();
  setBusy(true, true);
  showAiStatus();
  try {
    const res = await ai.imageTo3d(file, state.detail, currentAbort.signal);
    const name = file.name.replace(/\.[^.]+$/, '') || 'image piece';
    const group = await loadGlbIntoScene(res.glb, name);
    state.lastMake = {
      via: state.detail === 'ultra' ? 'AI · image · Ultra' : 'AI · image',
      label: name,
      seconds: res.seconds,
    };
    setActiveModel(group);
    // show what the AI actually sculpted: the subject, cut from the background
    $('cutout-preview').src = res.previewUrl;
    $('cutout-wrap').hidden = false;
    toast(`AI model ready (${res.seconds}s).`);
    autoSave(name);
  } catch (err) {
    if (err.name === 'AbortError') {
      toast('Cancelled — the bench is all yours.');
    } else {
      console.error(err);
      toast(`AI error: ${err.message}`);
    }
  } finally {
    currentAbort = null;
    hideAiStatus();
    setBusy(false);
  }
}

/* ---- tabs ---- */
document.querySelectorAll('.tab').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((b) => b.classList.toggle('active', b === btn));
    $('page-text').hidden = btn.dataset.tab !== 'text';
    $('page-image').hidden = btn.dataset.tab !== 'image';
  });
});

/* ---- text to 3D ---- */
function generateText(variation) {
  const prompt = $('prompt-input').value.trim();
  if (!prompt) { toast('Type something first — say, "red house".'); return; }
  if (state.busy) return;
  state.prompt = prompt;
  state.variation = variation;

  if (state.engine === 'ai' && state.aiReady) {
    generateTextAI(prompt, variation);
    return;
  }

  const group = generateFromText(prompt, variation);
  const p = group.userData.parsed;
  state.lastMake = { via: 'Instant', label: `${p.typeLabel} · V${variation + 1}` };
  setActiveModel(group);
  autoSave(`${p.typeLabel} · V${variation + 1}`);
  $('variations').hidden = false;
  document.querySelectorAll('.var-card').forEach((c) =>
    c.classList.toggle('active', Number(c.dataset.seed) === variation));
  const pr = $('parse-result');
  pr.hidden = false;
  let html = `Understood: <b>${p.typeLabel}</b>${p.colorName ? ` · color: <b>${p.colorName}</b>` : ''} · variation <b>V${variation + 1}</b>`;
  if (p.typeLabel === 'freeform sculpture' && state.aiReady) {
    html += `<br/>Not in the instant dictionary — switch to the <b>AI engine</b> for a real take on it.`;
  }
  pr.innerHTML = html;
}
$('btn-generate').addEventListener('click', () => {
  if (state.busy) { currentAbort?.abort(); return; } // button reads "Cancel" while busy
  generateText(0);
});
document.querySelectorAll('.chip').forEach((c) =>
  c.addEventListener('click', () => {
    $('prompt-input').value = c.textContent;
    $('prompt-input').focus();
  }));
$('prompt-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); generateText(0); }
});
document.querySelectorAll('.var-card').forEach((c) =>
  c.addEventListener('click', () => generateText(Number(c.dataset.seed))));

/* ---- image to 3D ---- */
let loadedImage = null;
let loadedFile = null;

const IMAGE_EXTENSIONS = /\.(png|jpe?g|webp|gif|bmp|avif|heic|heif|tiff?)$/i;

function looksLikeImage(file) {
  // phone photos (HEIC) and some clipboard files arrive with an empty MIME
  // type — fall back to the extension instead of silently ignoring them
  return file.type.startsWith('image/') || (!file.type && IMAGE_EXTENSIONS.test(file.name || ''));
}

/** Central intake for images from the picker, drag & drop, or paste. */
function loadImageFile(file) {
  if (!file) return;
  if (!looksLikeImage(file)) {
    toast(`"${file.name || 'that file'}" doesn't look like an image — PNG, JPG, WEBP or HEIC work.`);
    return;
  }
  loadedFile = file;
  loadedImage = null;
  const reader = new FileReader();
  reader.onload = () => {
    const img = $('image-preview');
    img.onload = () => {
      loadedImage = img;
      $('dz-idle').hidden = true;
      img.hidden = false;
      $('btn-image-generate').disabled = state.busy;
    };
    img.onerror = () => {
      // browser can't decode it (typically HEIC) — the AI backend still can
      $('dz-idle').hidden = true;
      img.hidden = true;
      $('image-preview-note').hidden = false;
      $('image-preview-note').textContent = `${file.name} — no browser preview for this format; the AI engine can still convert it.`;
      $('btn-image-generate').disabled = state.busy || state.engine !== 'ai';
    };
    $('image-preview-note').hidden = true;
    img.src = reader.result; // data URL, memory only
  };
  reader.readAsDataURL(file);
  // photos want the AI engine: it cuts out the subject and sculpts a real mesh
  if (state.aiReady && state.engine !== 'ai') {
    setEngine('ai');
    toast('Switched to the AI engine — it cuts out the subject and sculpts it properly.');
  }
  // bring the image tab forward so the preview is visible
  document.querySelector('.tab[data-tab="image"]').click();
}

$('image-input').addEventListener('change', (e) => loadImageFile(e.target.files?.[0]));

// dropzone card opens the (hidden) picker; guard against the input's own bubbled click
const dz = $('dropzone');
dz.addEventListener('click', (e) => {
  if (e.target !== $('image-input')) $('image-input').click();
});
dz.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('image-input').click(); }
});

// drag & drop anywhere onto the app
document.addEventListener('dragover', (e) => {
  e.preventDefault();
  document.body.classList.add('dropping');
});
document.addEventListener('dragleave', (e) => {
  if (!e.relatedTarget) document.body.classList.remove('dropping');
});
document.addEventListener('drop', (e) => {
  e.preventDefault();
  document.body.classList.remove('dropping');
  loadImageFile(e.dataTransfer?.files?.[0]);
});

// paste an image from the clipboard
document.addEventListener('paste', (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
  if (item) loadImageFile(item.getAsFile());
});
$('img-depth').addEventListener('input', (e) => { $('img-depth-val').textContent = Number(e.target.value).toFixed(1); });
$('img-res').addEventListener('input', (e) => { $('img-res-val').textContent = e.target.value; });
$('btn-image-generate').addEventListener('click', () => {
  if (state.busy) { currentAbort?.abort(); return; } // button reads "Cancel" while busy
  if (state.engine === 'ai' && state.aiReady && loadedFile) {
    generateImageAI(loadedFile);
    return;
  }
  if (!loadedImage) {
    toast('This format needs the AI engine — the browser can\'t read its pixels for Instant mode.');
    return;
  }
  const group = generateFromImage(loadedImage, $('img-mode').value, {
    res: Number($('img-res').value),
    depth: Number($('img-depth').value),
  });
  const name = `${(loadedFile?.name || 'image').replace(/\.[^.]+$/, '')} · ${$('img-mode').value}`;
  state.lastMake = { via: 'Instant · image', label: name };
  setActiveModel(group);
  autoSave(name);
});

/* ---- viewport toolbar ---- */
document.querySelectorAll('[data-view]').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-view]').forEach((b) => b.classList.toggle('active', b === btn));
    viewer.applyViewMode(btn.dataset.view);
  });
});
$('btn-autorotate').addEventListener('click', (e) => {
  const on = e.target.classList.toggle('active');
  viewer.setAutoRotate(on);
});
$('btn-grid').addEventListener('click', (e) => {
  const on = e.target.classList.toggle('active');
  viewer.setGrid(on);
});
const BG_MODES = ['studio', 'light', 'night'];
let bgIdx = 0;
$('btn-bg').addEventListener('click', (e) => {
  bgIdx = (bgIdx + 1) % BG_MODES.length;
  viewer.setBackground(BG_MODES[bgIdx]);
  e.target.classList.toggle('active', bgIdx !== 0);
});
$('btn-shot').addEventListener('click', async () => {
  const blob = await viewer.captureScreenshot();
  if (!blob) { toast('Screenshot failed.'); return; }
  download(blob, 'kilnform-shot.png');
  toast('Screenshot downloaded.');
});

/* ---- material ---- */
$('mat-metal').addEventListener('input', (e) => { $('mat-metal-val').textContent = Number(e.target.value).toFixed(2); });
$('mat-rough').addEventListener('input', (e) => { $('mat-rough-val').textContent = Number(e.target.value).toFixed(2); });
$('btn-mat-apply').addEventListener('click', () => {
  if (!state.model) { toast('Make a model first.'); return; }
  applyMaterial(state.model, {
    color: $('mat-color').value,
    metalness: Number($('mat-metal').value),
    roughness: Number($('mat-rough').value),
    texture: $('mat-texture').value,
  });
  toast('Material applied.');
});
$('btn-mat-reset').addEventListener('click', restorePristine);

/* ---- stylize ---- */
$('btn-voxelize').addEventListener('click', () => {
  if (!state.model) { toast('Make a model first.'); return; }
  const v = voxelize(state.model);
  if (v) { setActiveModel(v, { keepPristine: true }); toast('Voxelized.'); }
});
$('btn-lowpoly').addEventListener('click', () => {
  if (!state.model) { toast('Make a model first.'); return; }
  lowpoly(state.model);
  updateStats();
  toast('Low-poly applied.');
});
$('btn-toon').addEventListener('click', () => {
  if (!state.model) { toast('Make a model first.'); return; }
  toon(state.model);
  toast('Toon shading applied.');
});
$('btn-unstyle').addEventListener('click', restorePristine);

/* ---- transform ---- */
function onTransform() {
  $('tr-scale-val').textContent = Number($('tr-scale').value).toFixed(1);
  $('tr-roty-val').textContent = `${$('tr-roty').value}°`;
  $('tr-posy-val').textContent = Number($('tr-posy').value).toFixed(1);
  applyTransform(state.model, {
    scale: Number($('tr-scale').value),
    rotYDeg: Number($('tr-roty').value),
    posY: Number($('tr-posy').value),
  });
}
['tr-scale', 'tr-roty', 'tr-posy'].forEach((id) => $(id).addEventListener('input', onTransform));
$('tr-smooth').addEventListener('change', (e) => setSmoothShading(state.model, e.target.checked));

/* ---- export ---- */
async function doExport(format) {
  if (!state.model) { toast('Make a model first.'); return; }
  try {
    const name = ($('save-name').value.trim() || 'kilnform-piece').replace(/[^\wçğıöşüÇĞİÖŞÜ-]+/g, '-');
    const { blob, filename } = await exportModel(state.model, format, name);
    download(blob, filename);
    toast(`${format.toUpperCase()} downloaded (${(blob.size / 1024).toFixed(0)} KB).`);
  } catch (err) {
    console.error(err);
    toast(`Export failed: ${err.message}`);
  }
}
$('btn-export-glb').addEventListener('click', () => doExport('glb'));
$('btn-export-obj').addEventListener('click', () => doExport('obj'));
$('btn-export-stl').addEventListener('click', () => doExport('stl'));

/* ---- library ---- */
async function refreshLibrary() {
  const items = await listModels();
  $('lib-count').textContent = `(${items.length})`;
  const wrap = $('library-items');
  wrap.innerHTML = '';
  if (items.length === 0) {
    wrap.innerHTML = '<span class="lib-empty">Nothing made yet — every make lands here automatically.</span>';
    return;
  }
  for (const it of items) {
    const el = document.createElement('div');
    el.className = 'lib-item';
    el.title = `${it.name} — click to load`;
    el.innerHTML = `<img src="${it.thumb}" alt="${it.name}" /><div class="lib-name">${it.name}</div><button class="lib-del" title="Delete">✕</button>`;
    el.addEventListener('click', () => loadFromLibrary(it.id));
    el.querySelector('.lib-del').addEventListener('click', async (e) => {
      e.stopPropagation();
      await deleteModel(it.id);
      refreshLibrary();
      toast(`"${it.name}" removed.`);
    });
    wrap.appendChild(el);
  }
}

$('btn-save').addEventListener('click', async () => {
  if (!state.model) { toast('Make a model first.'); return; }
  const name = $('save-name').value.trim() || 'untitled piece';
  try {
    if (state.lastSaveId != null) {
      await renameModel(state.lastSaveId, name);
      await refreshLibrary();
      toast(`Named "${name}".`);
      return;
    }
    const glb = await toGLB(state.model);
    const thumb = viewer.captureThumbnail();
    state.lastSaveId = await saveModel({ name, thumb, glb });
    await refreshLibrary();
    toast(`"${name}" saved to History.`);
  } catch (err) {
    console.error(err);
    toast(`Save failed: ${err.message}`);
  }
});

async function loadFromLibrary(id) {
  const rec = await getModel(id);
  if (!rec) return;
  new GLTFLoader().parse(rec.glb, '', (gltf) => {
    const group = gltf.scene;
    group.userData.parsed = { typeLabel: rec.name, colorName: null, variation: 0 };
    state.lastMake = { via: 'History', label: rec.name };
    state.lastSaveId = id; // "Apply name" now renames this record
    setActiveModel(group);
    toast(`"${rec.name}" loaded.`);
  }, (err) => {
    console.error(err);
    toast('Could not load that piece.');
  });
}

/* ---- privacy sentinel: did anything leave localhost? ---- */
function checkNetwork() {
  const bad = performance.getEntriesByType('resource').filter((r) => {
    try {
      const u = new URL(r.name);
      return u.hostname !== '127.0.0.1' && u.hostname !== 'localhost';
    } catch { return false; }
  });
  if (bad.length > 0) {
    $('net-dot').classList.add('warn');
    $('net-label').textContent = 'outside request detected!';
  }
}
setInterval(checkNetwork, 3000);

/* ---- test hook (end-to-end verification) ---- */
window.__kilnform = {
  viewer,
  hasModel: () => !!state.model,
  polyCount: () => viewer.polyCount(),
  parsed: () => state.model?.userData?.parsed || null,
  async exportInfo(format) {
    const { blob } = await exportModel(state.model, format);
    const head = new Uint8Array(await blob.slice(0, 8).arrayBuffer());
    return { size: blob.size, head: [...head] };
  },
  async libraryCount() { return (await listModels()).length; },
};

refreshLibrary();
updateStats();
