import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

/* Image → 3D, procedurally. Everything happens in the browser via Canvas; the image goes nowhere. */

/** Samples the image at the given resolution, keeping aspect ratio. */
function sampleImage(img, res) {
  const aspect = img.naturalWidth / img.naturalHeight;
  const w = aspect >= 1 ? res : Math.max(8, Math.round(res * aspect));
  const h = aspect >= 1 ? Math.max(8, Math.round(res / aspect)) : res;
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  return { data: ctx.getImageData(0, 0, w, h).data, w, h };
}

function pixel(data, w, x, y) {
  const i = (y * w + x) * 4;
  const r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255, a = data[i + 3] / 255;
  return { r, g, b, a, lum: 0.2126 * r + 0.7152 * g + 0.0722 * b };
}

function makeMaterial() {
  return new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.8, metalness: 0.05 });
}

/** Relief: a standing colored plaque where brightness becomes height. */
function relief(img, res, depth) {
  const { data, w, h } = sampleImage(img, res);
  const W = 2.5, H = (2.5 * h) / w;
  const geo = new THREE.PlaneGeometry(W, H, w - 1, h - 1);
  const pos = geo.attributes.position;
  const colors = new Float32Array(pos.count * 3);
  const col = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    const x = i % w, y = Math.floor(i / w);
    const p = pixel(data, w, x, y);
    pos.setZ(i, 0.06 + p.lum * depth * 0.45 * p.a);
    col.setRGB(p.r, p.g, p.b, THREE.SRGBColorSpace);
    colors[i * 3] = col.r; colors[i * 3 + 1] = col.g; colors[i * 3 + 2] = col.b;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geo.computeVertexNormals();
  const front = new THREE.Mesh(geo, makeMaterial());
  // backing slab gives the relief some body
  const back = new THREE.Mesh(
    new THREE.BoxGeometry(W, H, 0.06),
    new THREE.MeshStandardMaterial({ color: 0x2a2f38, roughness: 0.9 })
  );
  back.position.z = 0.03;
  const g = new THREE.Group();
  g.add(back, front);
  return g;
}

/** Grid-of-cubes core shared by extrude and voxel modes. */
function cubeGrid(img, res, depth, mode) {
  const capped = Math.min(res, 72); // keeps the cube count sane
  const { data, w, h } = sampleImage(img, capped);
  const cell = 2.5 / Math.max(w, h);
  const col = new THREE.Color();
  const parts = [];

  // white-background heuristic: in extrude mode, very bright pixels count as background
  let skipWhite = false;
  if (mode === 'extrude') {
    let bright = 0, total = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const p = pixel(data, w, x, y);
      if (p.a > 0.5) { total++; if (p.lum > 0.93) bright++; }
    }
    skipWhite = total > 0 && bright / total > 0.15 && bright / total < 0.85;
  }

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = pixel(data, w, x, y);
      if (p.a < 0.5) continue;
      if (mode === 'extrude' && skipWhite && p.lum > 0.93) continue;
      const zScale = mode === 'voxel' ? (0.15 + (1 - p.lum) * depth * 0.85) : depth * 0.5;
      const box = new THREE.BoxGeometry(cell, cell, cell * 4 * zScale);
      box.translate((x - w / 2 + 0.5) * cell, (h / 2 - y - 0.5) * cell, 0);
      col.setRGB(p.r, p.g, p.b, THREE.SRGBColorSpace);
      const cArr = new Float32Array(box.attributes.position.count * 3);
      for (let i = 0; i < box.attributes.position.count; i++) {
        cArr[i * 3] = col.r; cArr[i * 3 + 1] = col.g; cArr[i * 3 + 2] = col.b;
      }
      box.setAttribute('color', new THREE.BufferAttribute(cArr, 3));
      parts.push(box);
    }
  }
  if (parts.length === 0) return null;
  const merged = mergeGeometries(parts);
  parts.forEach((p) => p.dispose());
  const g = new THREE.Group();
  g.add(new THREE.Mesh(merged, makeMaterial()));
  return g;
}

/**
 * Entry point: HTMLImageElement + mode → THREE.Group.
 * mode: 'relief' | 'extrude' | 'voxel'
 */
export function generateFromImage(img, mode, { res = 96, depth = 1 } = {}) {
  let group;
  if (mode === 'relief') group = relief(img, Math.min(res, 160), depth);
  else group = cubeGrid(img, res, depth, mode);
  if (!group) return null;

  // ground and center
  const box = new THREE.Box3().setFromObject(group);
  const c = box.getCenter(new THREE.Vector3());
  group.position.set(-c.x, -box.min.y, -c.z);
  group.name = `image-${mode}`;
  group.userData.parsed = { typeLabel: `image (${mode})`, colorName: null, variation: 0 };
  return group;
}
