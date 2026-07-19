import * as THREE from 'three';

/* Procedural textures, painted on a canvas right here. */
function makeCanvas(draw) {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  draw(c.getContext('2d'));
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(3, 3);
  return tex;
}

export function proceduralTexture(kind) {
  if (kind === 'checker') {
    return makeCanvas((ctx) => {
      for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
        ctx.fillStyle = (x + y) % 2 ? '#ffffff' : '#c9c9c9';
        ctx.fillRect(x * 32, y * 32, 32, 32);
      }
    });
  }
  if (kind === 'stripes') {
    return makeCanvas((ctx) => {
      for (let i = 0; i < 8; i++) {
        ctx.fillStyle = i % 2 ? '#ffffff' : '#cfcfcf';
        ctx.fillRect(0, i * 32, 256, 32);
      }
    });
  }
  if (kind === 'noise') {
    return makeCanvas((ctx) => {
      const img = ctx.createImageData(256, 256);
      for (let i = 0; i < img.data.length; i += 4) {
        const v = 190 + Math.random() * 65;
        img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
        img.data[i + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
    });
  }
  return null;
}

/** Applies material settings to every mesh in the model. */
export function applyMaterial(model, { color, metalness, roughness, texture }) {
  if (!model) return;
  const tex = texture && texture !== 'none' ? proceduralTexture(texture) : null;
  model.traverse((o) => {
    if (!o.isMesh) return;
    const m = o.material;
    if (color != null) { m.color = new THREE.Color(color); m.vertexColors = false; }
    if (metalness != null) m.metalness = metalness;
    if (roughness != null) m.roughness = roughness;
    m.map = tex;
    m.needsUpdate = true;
  });
}
