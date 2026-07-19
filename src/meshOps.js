/* Transform and shading operations. */

/** Remembers the model's transform at creation time (sliders multiply on top of it). */
export function rememberBase(model) {
  model.userData.base = {
    sx: model.scale.x, sy: model.scale.y, sz: model.scale.z,
    posY: model.position.y,
    rotY: model.rotation.y,
  };
}

export function applyTransform(model, { scale = 1, rotYDeg = 0, posY = 0 }) {
  if (!model) return;
  const b = model.userData.base || { sx: 1, sy: 1, sz: 1, posY: 0, rotY: 0 };
  model.scale.set(b.sx * scale, b.sy * scale, b.sz * scale);
  model.rotation.y = b.rotY + (rotYDeg * Math.PI) / 180;
  model.position.y = b.posY + posY;
}

export function setSmoothShading(model, smooth) {
  if (!model) return;
  model.traverse((o) => {
    if (!o.isMesh) return;
    o.material.flatShading = !smooth;
    o.material.needsUpdate = true;
    if (smooth) o.geometry.computeVertexNormals();
  });
}
