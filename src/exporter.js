import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { OBJExporter } from 'three/addons/exporters/OBJExporter.js';
import { STLExporter } from 'three/addons/exporters/STLExporter.js';

/** Serializes the model to a GLB ArrayBuffer (shared by library saves and GLB downloads). */
export function toGLB(model) {
  return new Promise((resolve, reject) => {
    new GLTFExporter().parse(
      model,
      (result) => resolve(result),
      (err) => reject(err),
      { binary: true }
    );
  });
}

/** format: 'glb' | 'obj' | 'stl' → { blob, filename } */
export async function exportModel(model, format, baseName = 'atolye3d-model') {
  if (format === 'glb') {
    const buf = await toGLB(model);
    return { blob: new Blob([buf], { type: 'model/gltf-binary' }), filename: `${baseName}.glb` };
  }
  if (format === 'obj') {
    const text = new OBJExporter().parse(model);
    return { blob: new Blob([text], { type: 'text/plain' }), filename: `${baseName}.obj` };
  }
  if (format === 'stl') {
    const data = new STLExporter().parse(model, { binary: true });
    return { blob: new Blob([data], { type: 'model/stl' }), filename: `${baseName}.stl` };
  }
  throw new Error(`Unknown format: ${format}`);
}

/** Downloads a blob to local disk. */
export function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}
