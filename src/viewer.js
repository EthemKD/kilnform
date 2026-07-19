import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class Viewer {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = null;

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 200);
    this.camera.position.set(4.2, 3.2, 5.4);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0.8, 0);
    this.controls.autoRotateSpeed = 2.2;

    // Studio lighting: warm key, cool fill, hemisphere ambient
    const key = new THREE.DirectionalLight(0xfff0dd, 2.6);
    key.position.set(5, 7, 4);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -6; key.shadow.camera.right = 6;
    key.shadow.camera.top = 6; key.shadow.camera.bottom = -6;
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x8fa8cc, 0.9);
    fill.position.set(-6, 3, -4);
    this.scene.add(fill);
    this.scene.add(new THREE.HemisphereLight(0x3a4356, 0x14110d, 1.1));

    // Floor: shadow-catcher disk + grid
    const ground = new THREE.Mesh(
      new THREE.CircleGeometry(9, 64),
      new THREE.ShadowMaterial({ opacity: 0.35 })
    );
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    this.scene.add(ground);

    this.grid = new THREE.GridHelper(10, 20, 0x3a4356, 0x232a36);
    this.grid.position.y = 0.001;
    this.scene.add(this.grid);

    this.modelRoot = new THREE.Group();
    this.modelRoot.name = 'modelRoot';
    this.scene.add(this.modelRoot);

    this.viewMode = 'solid';
    this._savedMaterials = new Map();

    this._resize = this._resize.bind(this);
    window.addEventListener('resize', this._resize);
    new ResizeObserver(this._resize).observe(container);
    this._resize();

    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  _resize() {
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  /** Replaces the model in the scene; disposes the old one. */
  setModel(group) {
    this.clearModel();
    if (!group) return;
    group.traverse((o) => {
      if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; }
    });
    this.modelRoot.add(group);
    this.applyViewMode(this.viewMode);
  }

  clearModel() {
    for (const child of [...this.modelRoot.children]) {
      this.modelRoot.remove(child);
      child.traverse((o) => {
        if (o.isMesh) {
          o.geometry?.dispose();
          const mats = Array.isArray(o.material) ? o.material : [o.material];
          mats.forEach((m) => m?.dispose());
        }
      });
    }
    this._savedMaterials.clear();
  }

  getModel() {
    return this.modelRoot.children[0] || null;
  }

  polyCount() {
    let tris = 0;
    this.modelRoot.traverse((o) => {
      if (o.isMesh && o.geometry) {
        const g = o.geometry;
        tris += g.index ? g.index.count / 3 : (g.attributes.position?.count || 0) / 3;
      }
    });
    return Math.round(tris);
  }

  applyViewMode(mode) {
    this.viewMode = mode;
    this.modelRoot.traverse((o) => {
      if (!o.isMesh) return;
      if (mode === 'solid') {
        if (this._savedMaterials.has(o.uuid)) {
          o.material = this._savedMaterials.get(o.uuid);
          this._savedMaterials.delete(o.uuid);
        }
        o.material.wireframe = false;
      } else {
        if (!this._savedMaterials.has(o.uuid)) this._savedMaterials.set(o.uuid, o.material);
        if (mode === 'wireframe') {
          o.material = new THREE.MeshBasicMaterial({ color: 0xe8965a, wireframe: true });
        } else if (mode === 'normals') {
          o.material = new THREE.MeshNormalMaterial();
        }
      }
    });
  }

  setAutoRotate(on) { this.controls.autoRotate = on; }
  setGrid(on) { this.grid.visible = on; }

  /** 'studio' = CSS gradient (transparent canvas), or a flat scene color. */
  setBackground(mode) {
    const colors = { studio: null, light: 0xd8dde3, night: 0x08090c };
    const c = colors[mode];
    this.scene.background = c == null ? null : new THREE.Color(c);
  }

  /** Full-resolution PNG of the current view (transparent in studio mode). */
  captureScreenshot() {
    this.renderer.render(this.scene, this.camera);
    return new Promise((resolve) => this.renderer.domElement.toBlob(resolve, 'image/png'));
  }

  /** Frames the camera on the current model. */
  frameModel() {
    const model = this.getModel();
    if (!model) return;
    const box = new THREE.Box3().setFromObject(model);
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length() || 1;
    this.controls.target.copy(center);
    const dir = new THREE.Vector3(0.75, 0.55, 1).normalize();
    this.camera.position.copy(center).addScaledVector(dir, size * 1.35);
    this.controls.update();
  }

  /** Captures a thumbnail for the library shelf. */
  captureThumbnail(w = 168, h = 120) {
    this.renderer.render(this.scene, this.camera);
    const src = this.renderer.domElement;
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#161a21';
    ctx.fillRect(0, 0, w, h);
    const scale = Math.max(w / src.width, h / src.height);
    const dw = src.width * scale, dh = src.height * scale;
    ctx.drawImage(src, (w - dw) / 2, (h - dh) / 2, dw, dh);
    return canvas.toDataURL('image/png');
  }
}
