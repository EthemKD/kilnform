/* Asset library backed by IndexedDB — every record lives in this browser only. */

const DB_NAME = 'atolye3d';
const STORE = 'models';

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx(db, mode, fn) {
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const store = t.objectStore(STORE);
    const req = fn(store);
    t.oncomplete = () => resolve(req?.result);
    t.onerror = () => reject(t.error);
  });
}

/** @param {{name:string, thumb:string, glb:ArrayBuffer}} rec */
export async function saveModel(rec) {
  const db = await openDb();
  return tx(db, 'readwrite', (s) => s.add({ ...rec, date: Date.now() }));
}

export async function listModels() {
  const db = await openDb();
  const all = await tx(db, 'readonly', (s) => s.getAll());
  return (all || []).sort((a, b) => b.date - a.date);
}

export async function getModel(id) {
  const db = await openDb();
  return tx(db, 'readonly', (s) => s.get(id));
}

export async function renameModel(id, name) {
  const db = await openDb();
  const rec = await tx(db, 'readonly', (s) => s.get(id));
  if (!rec) return null;
  rec.name = name;
  return tx(db, 'readwrite', (s) => s.put(rec));
}

export async function deleteModel(id) {
  const db = await openDb();
  return tx(db, 'readwrite', (s) => s.delete(id));
}
