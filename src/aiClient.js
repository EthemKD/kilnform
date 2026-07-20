/* Client for the local AI backend — talks to 127.0.0.1:8000 and nowhere else. */

const BASE = 'http://127.0.0.1:8000';

function b64ToArrayBuffer(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

function unpack(json) {
  return {
    glb: b64ToArrayBuffer(json.glb_b64),
    previewUrl: `data:image/png;base64,${json.preview_b64}`,
    promptEn: json.prompt_en,
    seconds: json.seconds,
  };
}

/** Backend status: null (unreachable) or {cuda, device, models}. */
export async function health() {
  try {
    const r = await fetch(`${BASE}/api/health`, { signal: AbortSignal.timeout(2000) });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

/** Preloads the models so the first make is fast. Allowed to fail quietly. */
export function warmup() {
  fetch(`${BASE}/api/warmup`, { method: 'POST' }).catch(() => {});
}

/** Current pipeline stage while a make is running, or null if unreachable. */
export async function progress() {
  try {
    const r = await fetch(`${BASE}/api/progress`, { signal: AbortSignal.timeout(1000) });
    return (await r.json()).stage;
  } catch {
    return null;
  }
}

/* `detail` is a marching-cubes resolution (number) or the string 'ultra'. */
function tierOf(detail) {
  return detail === 'ultra'
    ? { resolution: 320, tier: 'ultra' }
    : { resolution: detail, tier: 'mc' };
}

export async function textTo3d(prompt, seed = null, detail = 256, signal = null) {
  const { resolution, tier } = tierOf(detail);
  const r = await fetch(`${BASE}/api/text-to-3d`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, seed, resolution, tier }),
    signal,
  });
  if (!r.ok) throw new Error((await r.text()).slice(0, 200) || `HTTP ${r.status}`);
  return unpack(await r.json());
}

export async function imageTo3d(file, detail = 256, signal = null) {
  const { resolution, tier } = tierOf(detail);
  const form = new FormData();
  form.append('file', file);
  form.append('resolution', String(resolution));
  form.append('tier', tier);
  const r = await fetch(`${BASE}/api/image-to-3d`, { method: 'POST', body: form, signal });
  if (!r.ok) throw new Error((await r.text()).slice(0, 200) || `HTTP ${r.status}`);
  return unpack(await r.json());
}
