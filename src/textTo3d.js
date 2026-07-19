import * as THREE from 'three';

/* Seeded (deterministic) randomness — the same prompt + seed always makes the same model. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function hashString(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

const COLORS = {
  'kırmızı': 0xc0392b, red: 0xc0392b,
  'mavi': 0x2e6da4, blue: 0x2e6da4,
  'yeşil': 0x3d8b4f, green: 0x3d8b4f,
  'sarı': 0xe3b52c, yellow: 0xe3b52c,
  'mor': 0x7d5ba6, purple: 0x7d5ba6,
  'turuncu': 0xd9772b, orange: 0xd9772b,
  'pembe': 0xd977a8, pink: 0xd977a8,
  'siyah': 0x2b2b30, black: 0x2b2b30,
  'beyaz': 0xe8e6e0, white: 0xe8e6e0,
  'kahverengi': 0x7a5230, brown: 0x7a5230,
  'gri': 0x8a8f98, gray: 0x8a8f98, grey: 0x8a8f98,
  'altın': 0xd4a017, gold: 0xd4a017,
  'gümüş': 0xb8bcc4, silver: 0xb8bcc4,
};

const TYPES = [
  { id: 'house',   kws: ['ev', 'house', 'bina', 'kulübe'], label: 'house' },
  { id: 'tree',    kws: ['ağaç', 'tree', 'çam'], label: 'tree' },
  { id: 'car',     kws: ['araba', 'car', 'araç', 'otomobil'], label: 'car' },
  { id: 'robot',   kws: ['robot'], label: 'robot' },
  { id: 'rocket',  kws: ['roket', 'rocket', 'füze'], label: 'rocket' },
  { id: 'table',   kws: ['masa', 'table'], label: 'table' },
  { id: 'chair',   kws: ['sandalye', 'chair', 'koltuk'], label: 'chair' },
  { id: 'tower',   kws: ['kule', 'tower', 'kale'], label: 'tower' },
  { id: 'ship',    kws: ['gemi', 'ship', 'tekne', 'kayık', 'boat'], label: 'ship' },
  { id: 'snowman', kws: ['kardan', 'snowman'], label: 'snowman' },
  { id: 'mug',     kws: ['kupa', 'fincan', 'bardak', 'mug', 'cup'], label: 'mug' },
  { id: 'lamp',    kws: ['lamba', 'lamp', 'abajur'], label: 'lamp' },
];

/** Parses the prompt: type, color, size, material. Turkish and English keywords both count. */
export function parsePrompt(prompt) {
  const norm = prompt.toLocaleLowerCase('tr');
  const words = norm.split(/[^a-zçğıöşüâîû0-9]+/).filter(Boolean);
  const startsWith = (kw) => words.some((w) => (kw.length >= 3 ? w.startsWith(kw) : w === kw));

  let type = null;
  for (const t of TYPES) {
    if (t.kws.some(startsWith)) { type = t; break; }
  }

  let color = null, colorName = null;
  for (const [name, hex] of Object.entries(COLORS)) {
    if (startsWith(name)) { color = hex; colorName = name; break; }
  }

  const size = {
    scale: startsWith('büyük') || startsWith('dev') || startsWith('big') || startsWith('large') ? 1.45
      : startsWith('küçük') || startsWith('mini') || startsWith('small') || startsWith('tiny') ? 0.6 : 1,
    tall: startsWith('uzun') || startsWith('tall') || startsWith('yüksek'),
    wide: startsWith('geniş') || startsWith('wide'),
  };

  const material = {
    metalness: startsWith('metal') || startsWith('çelik') || startsWith('altın') || startsWith('gümüş') ? 0.85 : 0.05,
    roughness: startsWith('parlak') || startsWith('shiny') || startsWith('cilalı') ? 0.15
      : words.includes('mat') || startsWith('matte') ? 0.95 : 0.75,
  };

  return { type, color, colorName, size, material, prompt };
}

/* ---- helpers ---- */
let _parsed = null; // material settings of the build in progress

function M(color, opts = {}) {
  return new THREE.MeshStandardMaterial({
    color,
    roughness: opts.roughness ?? _parsed.material.roughness,
    metalness: opts.metalness ?? _parsed.material.metalness,
    emissive: opts.emissive ?? 0x000000,
    emissiveIntensity: opts.emissiveIntensity ?? 1,
    flatShading: opts.flat ?? false,
  });
}
function mesh(geom, color, opts) {
  const m = new THREE.Mesh(geom, M(color, opts));
  return m;
}
function place(m, x, y, z, group) { m.position.set(x, y, z); group.add(m); return m; }

/** Grounds and centers the model. */
function ground(group) {
  const box = new THREE.Box3().setFromObject(group);
  const c = box.getCenter(new THREE.Vector3());
  group.position.x -= c.x; group.position.z -= c.z;
  group.position.y -= box.min.y;
  return group;
}

/* ---- makers ---- */
const GEN = {
  house(r, c) {
    const g = new THREE.Group();
    const w = 2 + r() * 0.8, d = 1.6 + r() * 0.6, h = 1.1 + r() * 0.5;
    const wall = c ?? 0xd8c9a8;
    place(mesh(new THREE.BoxGeometry(w, h, d), wall), 0, h / 2, 0, g);
    const roof = mesh(new THREE.CylinderGeometry(0, Math.sqrt(w * w + d * d) / 2 * 1.08, 0.8 + r() * 0.4, 4, 1), 0x9c4a2f, { flat: true });
    roof.rotation.y = Math.PI / 4;
    roof.scale.set(w / Math.sqrt(w * w + d * d) * 1.45, 1, d / Math.sqrt(w * w + d * d) * 1.45);
    place(roof, 0, h + (0.8 + r() * 0.4) / 2, 0, g);
    place(mesh(new THREE.BoxGeometry(0.34, 0.62, 0.05), 0x5a3a22), 0, 0.31, d / 2 + 0.02, g);
    const win = () => mesh(new THREE.BoxGeometry(0.3, 0.3, 0.05), 0x9ec8e0, { emissive: 0x223a4a, roughness: 0.2 });
    place(win(), -w / 4, h * 0.6, d / 2 + 0.02, g);
    place(win(), w / 4, h * 0.6, d / 2 + 0.02, g);
    place(mesh(new THREE.BoxGeometry(0.22, 0.7, 0.22), 0x8a5a40), w / 3.2, h + 0.55, -d / 5, g);
    return g;
  },
  tree(r, c) {
    const g = new THREE.Group();
    const th = 0.8 + r() * 0.6;
    place(mesh(new THREE.CylinderGeometry(0.12, 0.17, th, 8), 0x6b4a2c), 0, th / 2, 0, g);
    const leaf = c ?? 0x3d8b4f;
    const pine = r() > 0.5;
    if (pine) {
      let y = th, rad = 0.85 + r() * 0.3;
      for (let i = 0; i < 3; i++) {
        const ch = 0.8 - i * 0.12;
        place(mesh(new THREE.ConeGeometry(rad, ch, 9), leaf, { flat: true }), 0, y + ch / 2, 0, g);
        y += ch * 0.55; rad *= 0.72;
      }
    } else {
      place(mesh(new THREE.IcosahedronGeometry(0.75 + r() * 0.3, 1), leaf, { flat: true }), 0, th + 0.55, 0, g);
      place(mesh(new THREE.IcosahedronGeometry(0.45 + r() * 0.2, 1), leaf, { flat: true }), 0.45, th + 0.85, 0.15, g);
    }
    return g;
  },
  car(r, c) {
    const g = new THREE.Group();
    const body = c ?? 0xc0392b;
    const L = 2.3 + r() * 0.5;
    place(mesh(new THREE.BoxGeometry(L, 0.5, 1.1), body), 0, 0.55, 0, g);
    place(mesh(new THREE.BoxGeometry(L * 0.5, 0.42, 0.95), body), -L * 0.06, 1.0, 0, g);
    const wheel = () => {
      const w = mesh(new THREE.CylinderGeometry(0.28, 0.28, 0.18, 18), 0x23232a, { roughness: 0.9 });
      w.rotation.x = Math.PI / 2;
      return w;
    };
    for (const [x, z] of [[-L / 3, 0.56], [L / 3, 0.56], [-L / 3, -0.56], [L / 3, -0.56]])
      place(wheel(), x, 0.28, z, g);
    place(mesh(new THREE.BoxGeometry(0.06, 0.12, 0.24), 0xfff2c8, { emissive: 0xffe9a0, emissiveIntensity: 0.7 }), L / 2, 0.6, 0.3, g);
    place(mesh(new THREE.BoxGeometry(0.06, 0.12, 0.24), 0xfff2c8, { emissive: 0xffe9a0, emissiveIntensity: 0.7 }), L / 2, 0.6, -0.3, g);
    return g;
  },
  robot(r, c) {
    const g = new THREE.Group();
    const body = c ?? 0x8a8f98;
    place(mesh(new THREE.BoxGeometry(0.9, 1.1, 0.55), body), 0, 1.25, 0, g);
    place(mesh(new THREE.BoxGeometry(0.55, 0.5, 0.5), body), 0, 2.1, 0, g);
    const eye = () => mesh(new THREE.SphereGeometry(0.07, 10, 10), 0x66d9ef, { emissive: 0x2ab8d8, emissiveIntensity: 1.6 });
    place(eye(), -0.13, 2.16, 0.26, g); place(eye(), 0.13, 2.16, 0.26, g);
    place(mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.3, 6), 0x555a63), 0, 2.5, 0, g);
    place(mesh(new THREE.SphereGeometry(0.06, 8, 8), 0xe8965a, { emissive: 0xe8965a, emissiveIntensity: 1.2 }), 0, 2.68, 0, g);
    const arm = () => mesh(new THREE.CylinderGeometry(0.09, 0.09, 0.85, 10), 0x555a63);
    const a1 = place(arm(), -0.62, 1.25, 0, g); a1.rotation.z = 0.15 + r() * 0.35;
    const a2 = place(arm(), 0.62, 1.25, 0, g); a2.rotation.z = -(0.15 + r() * 0.35);
    const leg = () => mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.7, 10), 0x555a63);
    place(leg(), -0.25, 0.35, 0, g); place(leg(), 0.25, 0.35, 0, g);
    return g;
  },
  rocket(r, c) {
    const g = new THREE.Group();
    const body = c ?? 0xe8e6e0;
    const H = 2 + r() * 0.8;
    place(mesh(new THREE.CylinderGeometry(0.42, 0.5, H, 20), body), 0, H / 2 + 0.35, 0, g);
    place(mesh(new THREE.ConeGeometry(0.42, 0.9, 20), 0xc0392b), 0, H + 0.8, 0, g);
    const port = mesh(new THREE.TorusGeometry(0.14, 0.045, 10, 20), 0x2e6da4);
    place(port, 0, H * 0.62 + 0.35, 0.44, g);
    for (let i = 0; i < 3; i++) {
      const fin = mesh(new THREE.BoxGeometry(0.08, 0.75, 0.5), 0xc0392b);
      const a = (i / 3) * Math.PI * 2;
      place(fin, Math.sin(a) * 0.52, 0.55, Math.cos(a) * 0.52, g);
      fin.rotation.y = a;
    }
    place(mesh(new THREE.ConeGeometry(0.3, 0.5, 14), 0xe8965a, { emissive: 0xd96b2b, emissiveIntensity: 1.4 }), 0, 0.12, 0, g);
    return g;
  },
  table(r, c) {
    const g = new THREE.Group();
    const wood = c ?? 0x8a5a36;
    const w = 1.8 + r() * 0.6, d = 1 + r() * 0.4, h = 0.9;
    place(mesh(new THREE.BoxGeometry(w, 0.09, d), wood), 0, h, 0, g);
    for (const [x, z] of [[w / 2 - 0.08, d / 2 - 0.08], [-w / 2 + 0.08, d / 2 - 0.08], [w / 2 - 0.08, -d / 2 + 0.08], [-w / 2 + 0.08, -d / 2 + 0.08]])
      place(mesh(new THREE.BoxGeometry(0.09, h, 0.09), wood), x, h / 2, z, g);
    return g;
  },
  chair(r, c) {
    const g = new THREE.Group();
    const wood = c ?? 0x9c6b40;
    place(mesh(new THREE.BoxGeometry(0.85, 0.08, 0.8), wood), 0, 0.85, 0, g);
    place(mesh(new THREE.BoxGeometry(0.85, 0.95, 0.07), wood), 0, 1.36, -0.37, g);
    for (const [x, z] of [[0.36, 0.34], [-0.36, 0.34], [0.36, -0.34], [-0.36, -0.34]])
      place(mesh(new THREE.BoxGeometry(0.07, 0.85, 0.07), wood), x, 0.42, z, g);
    return g;
  },
  tower(r, c) {
    const g = new THREE.Group();
    const stone = c ?? 0x9a9484;
    const H = 2.4 + r() * 1;
    place(mesh(new THREE.CylinderGeometry(0.62, 0.78, H, 14), stone), 0, H / 2, 0, g);
    place(mesh(new THREE.CylinderGeometry(0.82, 0.82, 0.3, 14), stone), 0, H + 0.15, 0, g);
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2;
      place(mesh(new THREE.BoxGeometry(0.18, 0.24, 0.18), stone), Math.sin(a) * 0.74, H + 0.42, Math.cos(a) * 0.74, g);
    }
    place(mesh(new THREE.BoxGeometry(0.28, 0.5, 0.06), 0x3a3026), 0, 0.25, 0.76, g);
    place(mesh(new THREE.CylinderGeometry(0.018, 0.018, 0.7, 6), 0x5a4a35), 0, H + 0.65, 0, g);
    const flag = mesh(new THREE.BoxGeometry(0.4, 0.22, 0.02), 0xc0392b);
    place(flag, 0.2, H + 0.85, 0, g);
    return g;
  },
  ship(r, c) {
    const g = new THREE.Group();
    const hullC = c ?? 0x6b4a2c;
    const L = 2.6 + r() * 0.6;
    const hull = mesh(new THREE.CylinderGeometry(0.55, 0.28, L, 4, 1), hullC, { flat: true });
    hull.rotation.z = Math.PI / 2; hull.rotation.y = Math.PI / 4;
    hull.scale.set(1, 1, 0.72);
    place(hull, 0, 0.5, 0, g);
    place(mesh(new THREE.BoxGeometry(L * 0.82, 0.1, 0.62), 0x8a6a44), 0, 0.82, 0, g);
    place(mesh(new THREE.CylinderGeometry(0.045, 0.045, 1.7, 8), 0x5a4a35), 0, 1.7, 0, g);
    const sail = mesh(new THREE.BoxGeometry(0.04, 1.05, 0.9), 0xe8e6e0, { roughness: 1 });
    place(sail, 0.06, 1.85, 0, g);
    const flag = mesh(new THREE.BoxGeometry(0.26, 0.14, 0.02), 0xc0392b);
    place(flag, 0.13, 2.62, 0, g);
    return g;
  },
  snowman(r, c) {
    const g = new THREE.Group();
    const snow = c ?? 0xeef2f5;
    place(mesh(new THREE.SphereGeometry(0.62, 22, 18), snow, { roughness: 0.95 }), 0, 0.62, 0, g);
    place(mesh(new THREE.SphereGeometry(0.45, 20, 16), snow, { roughness: 0.95 }), 0, 1.5, 0, g);
    place(mesh(new THREE.SphereGeometry(0.32, 18, 14), snow, { roughness: 0.95 }), 0, 2.18, 0, g);
    const nose = mesh(new THREE.ConeGeometry(0.07, 0.32, 10), 0xd9772b);
    nose.rotation.x = Math.PI / 2;
    place(nose, 0, 2.2, 0.42, g);
    place(mesh(new THREE.SphereGeometry(0.045, 8, 8), 0x23232a), -0.11, 2.3, 0.28, g);
    place(mesh(new THREE.SphereGeometry(0.045, 8, 8), 0x23232a), 0.11, 2.3, 0.28, g);
    place(mesh(new THREE.CylinderGeometry(0.24, 0.24, 0.06, 16), 0x23232a), 0, 2.48, 0, g);
    place(mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.26, 16), 0x23232a), 0, 2.62, 0, g);
    const arm = () => {
      const a = mesh(new THREE.CylinderGeometry(0.03, 0.02, 0.75, 6), 0x6b4a2c);
      a.rotation.z = Math.PI / 2.6;
      return a;
    };
    const a1 = place(arm(), -0.68, 1.62, 0, g);
    const a2 = place(arm(), 0.68, 1.62, 0, g); a2.rotation.z *= -1;
    for (let i = 0; i < 3; i++)
      place(mesh(new THREE.SphereGeometry(0.05, 8, 8), 0x23232a), 0, 1.4 + i * 0.22, 0.43 - i * 0.03, g);
    return g;
  },
  mug(r, c) {
    const g = new THREE.Group();
    const col = c ?? 0x2e6da4;
    const H = 0.9 + r() * 0.25, R = 0.42;
    const wall = mesh(new THREE.CylinderGeometry(R, R * 0.92, H, 26, 1, true), col);
    wall.material.side = THREE.DoubleSide;
    place(wall, 0, H / 2, 0, g);
    place(mesh(new THREE.CylinderGeometry(R * 0.93, R * 0.93, 0.05, 26), col), 0, 0.03, 0, g);
    place(mesh(new THREE.CylinderGeometry(R * 0.88, R * 0.88, 0.03, 26), 0x4a3222, { roughness: 0.4 }), 0, H * 0.82, 0, g);
    const handle = mesh(new THREE.TorusGeometry(0.24, 0.055, 12, 24), col);
    place(handle, R + 0.14, H / 2, 0, g);
    return g;
  },
  lamp(r, c) {
    const g = new THREE.Group();
    const col = c ?? 0xd9772b;
    place(mesh(new THREE.CylinderGeometry(0.34, 0.42, 0.08, 20), 0x3a3f48), 0, 0.04, 0, g);
    place(mesh(new THREE.CylinderGeometry(0.035, 0.035, 1.4, 10), 0x3a3f48), 0, 0.78, 0, g);
    place(mesh(new THREE.SphereGeometry(0.14, 14, 12), 0xfff2c8, { emissive: 0xffdf94, emissiveIntensity: 2 }), 0, 1.6, 0, g);
    const shade = mesh(new THREE.CylinderGeometry(0.3, 0.55, 0.55, 20, 1, true), col);
    shade.material.side = THREE.DoubleSide;
    place(shade, 0, 1.68, 0, g);
    return g;
  },
  abstract(r, c) {
    const g = new THREE.Group();
    const base = c ?? 0xe8965a;
    const n = 4 + Math.floor(r() * 4);
    const geoms = [
      () => new THREE.IcosahedronGeometry(0.3 + r() * 0.35, 0),
      () => new THREE.BoxGeometry(0.3 + r() * 0.5, 0.3 + r() * 0.5, 0.3 + r() * 0.5),
      () => new THREE.TorusGeometry(0.3 + r() * 0.2, 0.09 + r() * 0.07, 10, 22),
      () => new THREE.ConeGeometry(0.25 + r() * 0.2, 0.5 + r() * 0.5, 5 + Math.floor(r() * 8)),
      () => new THREE.CylinderGeometry(0.12 + r() * 0.2, 0.12 + r() * 0.25, 0.4 + r() * 0.8, 5 + Math.floor(r() * 12)),
    ];
    const c2 = new THREE.Color(base).offsetHSL(0.5, 0, 0).getHex();
    let y = 0.2;
    for (let i = 0; i < n; i++) {
      const m = mesh(geoms[Math.floor(r() * geoms.length)](), r() > 0.65 ? c2 : base, { flat: r() > 0.5 });
      m.rotation.set(r() * Math.PI, r() * Math.PI, r() * Math.PI);
      place(m, (r() - 0.5) * 0.8, y, (r() - 0.5) * 0.8, g);
      y += 0.3 + r() * 0.35;
    }
    return g;
  },
};

/**
 * Entry point: prompt + variation seed → THREE.Group.
 * The parse result rides along in .userData.parsed.
 */
export function generateFromText(prompt, variation = 0) {
  const parsed = parsePrompt(prompt);
  _parsed = parsed;
  const seed = hashString(parsed.prompt.trim().toLocaleLowerCase('tr')) + variation * 7919;
  const r = mulberry32(seed);
  const genId = parsed.type ? parsed.type.id : 'abstract';
  const group = GEN[genId](r, parsed.color);
  group.name = genId;

  // size adjectives
  group.scale.multiplyScalar(parsed.size.scale);
  if (parsed.size.tall) group.scale.y *= 1.5;
  if (parsed.size.wide) { group.scale.x *= 1.35; group.scale.z *= 1.35; }

  ground(group);
  group.userData.parsed = {
    typeLabel: parsed.type ? parsed.type.label : 'freeform sculpture',
    colorName: parsed.colorName,
    variation,
  };
  return group;
}
