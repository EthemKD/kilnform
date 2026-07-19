import * as THREE from 'three';
import { MeshSurfaceSampler } from 'three/addons/math/MeshSurfaceSampler.js';
import { SimplifyModifier } from 'three/addons/modifiers/SimplifyModifier.js';
import { mergeGeometries, mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';

/** Voxelize: samples the surface and rebuilds it from cubes, brick-toy style. */
export function voxelize(model, resolution = 26) {
  model.updateMatrixWorld(true);
  const bbox = new THREE.Box3().setFromObject(model);
  if (bbox.isEmpty()) return null;
  const size = bbox.getSize(new THREE.Vector3());
  const cell = Math.max(size.x, size.y, size.z) / resolution;
  if (cell <= 0) return null;

  const voxels = new Map(); // "x,y,z" -> THREE.Color
  const pos = new THREE.Vector3();
  const colTarget = new THREE.Color();

  model.traverse((o) => {
    if (!o.isMesh || !o.geometry?.attributes?.position) return;
    let sampler;
    try { sampler = new MeshSurfaceSampler(o).build(); } catch { return; }
    const hasVertexColor = !!o.geometry.attributes.color;
    const matColor = o.material?.color ? o.material.color.clone() : new THREE.Color(0xaaaaaa);
    const count = Math.min(24000, Math.max(2500, o.geometry.attributes.position.count * 4));
    for (let i = 0; i < count; i++) {
      if (hasVertexColor) sampler.sample(pos, undefined, colTarget);
      else { sampler.sample(pos); colTarget.copy(matColor); }
      pos.applyMatrix4(o.matrixWorld);
      const kx = Math.floor((pos.x - bbox.min.x) / cell);
      const ky = Math.floor((pos.y - bbox.min.y) / cell);
      const kz = Math.floor((pos.z - bbox.min.z) / cell);
      const key = `${kx},${ky},${kz}`;
      if (!voxels.has(key)) voxels.set(key, colTarget.clone());
    }
  });

  if (voxels.size === 0) return null;
  const parts = [];
  for (const [key, color] of voxels) {
    const [kx, ky, kz] = key.split(',').map(Number);
    const box = new THREE.BoxGeometry(cell, cell, cell);
    box.translate(
      bbox.min.x + (kx + 0.5) * cell,
      bbox.min.y + (ky + 0.5) * cell,
      bbox.min.z + (kz + 0.5) * cell
    );
    const cArr = new Float32Array(box.attributes.position.count * 3);
    for (let i = 0; i < box.attributes.position.count; i++) {
      cArr[i * 3] = color.r; cArr[i * 3 + 1] = color.g; cArr[i * 3 + 2] = color.b;
    }
    box.setAttribute('color', new THREE.BufferAttribute(cArr, 3));
    parts.push(box);
  }
  const merged = mergeGeometries(parts);
  parts.forEach((p) => p.dispose());

  const g = new THREE.Group();
  g.add(new THREE.Mesh(merged, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, metalness: 0.02 })));
  const nb = new THREE.Box3().setFromObject(g);
  const c = nb.getCenter(new THREE.Vector3());
  g.position.set(-c.x, -nb.min.y, -c.z);
  g.name = (model.name || 'model') + '-voxel';
  g.userData.parsed = { ...(model.userData.parsed || {}), typeLabel: ((model.userData.parsed?.typeLabel) || 'model') + ' · voxel' };
  return g;
}

/** Low-poly: simplifies meshes and switches to faceted shading. */
export function lowpoly(model) {
  const modifier = new SimplifyModifier();
  model.traverse((o) => {
    if (!o.isMesh || !o.geometry) return;
    try {
      let geo = mergeVertices(o.geometry.clone(), 1e-4);
      const vCount = geo.attributes.position.count;
      if (vCount > 60) {
        const target = Math.floor(vCount * 0.5);
        const simplified = modifier.modify(geo, target);
        if (simplified?.attributes?.position?.count > 3) geo = simplified;
      }
      // SimplifyModifier drops the color attribute; carry the average color to the material
      if (o.material.vertexColors && !geo.attributes.color) {
        const src = o.geometry.attributes.color;
        const avg = new THREE.Color(0, 0, 0);
        if (src) {
          for (let i = 0; i < src.count; i++) { avg.r += src.getX(i); avg.g += src.getY(i); avg.b += src.getZ(i); }
          avg.multiplyScalar(1 / src.count);
        } else avg.set(0x8a8f98);
        o.material = o.material.clone();
        o.material.vertexColors = false;
        o.material.color = avg;
      }
      o.geometry = geo;
    } catch { /* meshes that refuse to simplify stay as they are */ }
    o.material.flatShading = true;
    o.material.needsUpdate = true;
    o.geometry.computeVertexNormals();
  });
  return model;
}

/** Toon look: swaps materials for stepped toon shading. */
export function toon(model) {
  model.traverse((o) => {
    if (!o.isMesh) return;
    const old = o.material;
    o.material = new THREE.MeshToonMaterial({
      color: old.color ? old.color.clone() : new THREE.Color(0xaaaaaa),
      vertexColors: !!(old.vertexColors && o.geometry.attributes.color),
      emissive: old.emissive ? old.emissive.clone() : new THREE.Color(0),
    });
  });
  return model;
}
