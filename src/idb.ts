// Minimal IndexedDB helpers for saving/loading the model + scaler
const DB_NAME = 'elm-voice';
const STORE = 'models';
const KEY = 'default';

type SavePayload = { modelJSON: any; scaler: any; savedAt: number };

function openDB(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, 1);
        req.onupgradeneeded = () => {
            const db = req.result;
            if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

export async function saveModelToIDB(modelJSON: any, scaler: any): Promise<void> {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE, 'readwrite');
        const store = tx.objectStore(STORE);
        const data: SavePayload = { modelJSON, scaler, savedAt: Date.now() };
        const req = store.put(data, KEY);
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
    });
}

export async function loadModelFromIDB(): Promise<SavePayload | null> {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE, 'readonly');
        const store = tx.objectStore(STORE);
        const req = store.get(KEY);
        req.onsuccess = () => resolve((req.result as SavePayload) || null);
        req.onerror = () => reject(req.error);
    });
}

export async function deleteModelFromIDB(): Promise<void> {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE, 'readwrite');
        const store = tx.objectStore(STORE);
        const req = store.delete(KEY);
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
    });
}
