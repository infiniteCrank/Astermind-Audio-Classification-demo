// src/main.ts
import { extractFeaturesMonoFloat32 } from './mfcc';
import WorkerCtor from './worker.ts?worker';
import { saveModelToIDB, loadModelFromIDB, deleteModelFromIDB } from './idb';
import { InfiniteRunnerGame } from './game';

const $ = <T extends Element = Element>(sel: string) => document.querySelector(sel) as T;

const logEl = $('#log') as HTMLDivElement;

const micBtn = $('#micBtn') as HTMLButtonElement;
const micStatus = $('#micStatus') as HTMLSpanElement;
const recLeftBtn = $('#recLeft') as HTMLButtonElement;
const recRightBtn = $('#recRight') as HTMLButtonElement;
const leftCountEl = $('#leftCount') as HTMLSpanElement;
const rightCountEl = $('#rightCount') as HTMLSpanElement;
const trainBtn = $('#trainBtn') as HTMLButtonElement;
const trainStatus = $('#trainStatus') as HTMLSpanElement;
const listenBtn = $('#listenBtn') as HTMLButtonElement;
const predEl = $('#pred') as HTMLSpanElement;

const saveBtn = $('#saveBtn') as HTMLButtonElement;
const loadBtn = $('#loadBtn') as HTMLButtonElement;
const resetModelBtn = $('#resetModelBtn') as HTMLButtonElement;

// Game controls
const startGameBtn = $('#startGameBtn') as HTMLButtonElement;
const pauseGameBtn = $('#pauseGameBtn') as HTMLButtonElement;
const resetGameBtn = $('#resetGameBtn') as HTMLButtonElement;
const gameCanvas = $('#gameCanvas') as HTMLCanvasElement;
const game = new InfiniteRunnerGame(gameCanvas);

// Debug HUD
const dbg = (() => {
  const host = document.createElement('div');
  host.style.cssText =
    'position:fixed;right:16px;bottom:16px;z-index:9999;background:rgba(0,0,0,0.75);' +
    'color:#fff;font:12px/1.2 system-ui;padding:10px 12px;border-radius:10px;min-width:240px';
  host.innerHTML = `
    <div style="opacity:.9;margin-bottom:6px">🔎 Debug</div>
    <div id="dbg-rms">RMS: —</div>
    <div id="dbg-prob">P(left,right): —</div>
    <div id="dbg-note" style="opacity:.8"></div>
  `;
  document.body.appendChild(host);
  const rmsEl = host.querySelector('#dbg-rms') as HTMLDivElement;
  const probEl = host.querySelector('#dbg-prob') as HTMLDivElement;
  const noteEl = host.querySelector('#dbg-note') as HTMLDivElement;
  return {
    setRMS(v: number) { rmsEl.textContent = `RMS: ${v.toFixed(4)}`; },
    setProb(p: number[]) { probEl.textContent = `P(left,right): [${p.map(n => n.toFixed(2)).join(', ')}]`; },
    note(s: string) { noteEl.textContent = s; }
  };
})();

function log(s: string) {
  logEl.textContent += s + '\n';
  logEl.scrollTop = logEl.scrollHeight;
}

/** ───────────────────────── Config knobs ───────────────────────── **/
// Hysteretic VAD so we don't flicker
const VAD_ENTER = 0.005;
const VAD_EXIT = 0.003;
let CONF_THRESHOLD = 0.40;
const PRED_WINDOW = 5;
const REQUIRE_VAD = true;
const MFCC_OPTS = { trimThreshold: 0.003, allowUntrimFallback: true, padToFrame: true };

/** ───────────────────────── State ───────────────────────── **/
let audioCtx: AudioContext | null = null;
let sourceNode: MediaStreamAudioSourceNode | null = null;
let processor: ScriptProcessorNode | null = null;
let stream: MediaStream | null = null;
let sampleRate = 44100;

const captureBuf: Float32Array[] = []; // rolling capture (~5s)
const leftSamples: Float32Array[] = [];
const rightSamples: Float32Array[] = [];

let worker: Worker | null = null;
let listenTimer: number | null = null;

// smoothing + segmentation
const lastPreds: number[] = [];
let inSpeech = false;
let lastStable: 'LEFT' | 'RIGHT' | null = null;
// NEW: fire one command per utterance
let utteranceFired = false;

function resetVotes() {
  lastPreds.length = 0;
  predEl.textContent = '…';
  utteranceFired = false;
  dbg.note('speech start (cleared votes)');
}

/** ───────────────────────── Audio capture ───────────────────────── **/
function attachAudio() {
  if (!audioCtx || !sourceNode || processor) return;
  processor = audioCtx.createScriptProcessor(2048, 1, 1);
  processor.onaudioprocess = (e: AudioProcessingEvent) => {
    const ch0 = (e.inputBuffer as AudioBuffer).getChannelData(0);
    captureBuf.push(new Float32Array(ch0));
    // keep buffer bounded (~5s)
    const maxSamples = Math.ceil(5 * sampleRate);
    let total = 0;
    for (let i = captureBuf.length - 1; i >= 0; i--) {
      total += captureBuf[i].length;
      if (total > maxSamples) {
        captureBuf.splice(0, i);
        break;
      }
    }
  };
  sourceNode.connect(processor);
  processor.connect(audioCtx.destination); // keeps node alive reliably
}

async function enableMic() {
  if (stream) return;
  stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
  sourceNode = audioCtx.createMediaStreamSource(stream);
  sampleRate = audioCtx.sampleRate;
  attachAudio();
  micStatus.textContent = 'mic: on';
  log('Mic enabled.');
}

function beginSample(seconds = 1.2) {
  if (!processor) throw new Error('mic not enabled');
  captureBuf.length = 0; // reset rolling window for the prompted recording
  window.setTimeout(() => { /* window elapsed */ }, Math.max(200, seconds * 1000));
}

function takeCapturedMono(): Float32Array {
  let total = 0;
  for (const c of captureBuf) total += c.length;
  const mono = new Float32Array(total);
  let off = 0;
  for (const c of captureBuf) { mono.set(c, off); off += c.length; }
  return mono;
}

/** ───────────────────────── Utils ───────────────────────── **/
function oneHot(index: number, classes = 2): number[] {
  const y = new Array(classes).fill(0);
  y[index] = 1;
  return y;
}
function rms(x: Float32Array): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) { const v = x[i]; s += v * v; }
  return Math.sqrt(s / Math.max(1, x.length));
}
function argmax(v: number[]): { idx: number, val: number } {
  let bi = 0, bv = -Infinity;
  for (let i = 0; i < v.length; i++) if (v[i] > bv) { bv = v[i]; bi = i; }
  return { idx: bi, val: bv };
}
function pushPred(idx: number) {
  lastPreds.push(idx);
  if (lastPreds.length > PRED_WINDOW) lastPreds.shift();
}
function majority(): number | null {
  if (!lastPreds.length) return null;
  const counts = new Map<number, number>();
  for (const i of lastPreds) counts.set(i, (counts.get(i) || 0) + 1);
  let best = -1, bestC = -1;
  counts.forEach((c, k) => { if (c > bestC) { bestC = c; best = k; } });
  return best;
}

// worker bridge
function ensureWorker(): Promise<void> {
  if (worker) return Promise.resolve();
  worker = new WorkerCtor();
  return callWorker({ action: 'init' }).then(() => { });
}
function callWorker(msg: any): Promise<any> {
  if (!worker) throw new Error('worker not ready');
  return new Promise((resolve, reject) => {
    const id = Math.random().toString(36).slice(2);
    const onMsg = (e: MessageEvent<any>) => {
      const { id: rid, ok, result, error } = e.data || {};
      if (rid !== id) return;
      worker!.removeEventListener('message', onMsg);
      ok ? resolve(result) : reject(new Error(error || 'Worker error'));
    };
    worker!.addEventListener('message', onMsg);
    worker!.postMessage({ ...msg, id });
  });
}

// train
async function trainModel() {
  await ensureWorker();

  const X: number[][] = [];
  const y: number[][] = [];
  leftSamples.forEach(x => { X.push(Array.from(x)); y.push(oneHot(0)); });
  rightSamples.forEach(x => { X.push(Array.from(x)); y.push(oneHot(1)); });
  if (X.length < 8) { log('Need 4 left + 4 right before training.'); return; }

  trainStatus.textContent = 'training...';
  const modelOptions = { task: 'classification', outputDim: 2, kernel: { type: 'rbf', sigma: 1.0 }, ridgeLambda: 1e-2 };
  await callWorker({ action: 'train', payload: { X, y, modelOptions } });
  trainStatus.textContent = 'trained ✔';
  listenBtn.disabled = false;
  saveBtn.disabled = false;
  log('Training complete.');
}

// live listen
async function livePredictTick() {
  if (!worker) return;

  // Take the last ~0.8s of audio
  const framesNeeded = Math.ceil(0.8 * sampleRate);
  const mono = takeCapturedMono();
  const slice = mono.length > framesNeeded ? mono.subarray(mono.length - framesNeeded) : mono;

  const e = rms(slice);
  dbg.setRMS(e);

  // hysteretic VAD
  const speechNow = !REQUIRE_VAD || (inSpeech ? e >= VAD_EXIT : e >= VAD_ENTER);

  // VAD edges
  if (!inSpeech && speechNow) {
    inSpeech = true;
    resetVotes();                 // clear old votes ONLY
    // continue; we still extract features this tick
  } else if (inSpeech && !speechNow) {
    inSpeech = false;
    predEl.textContent = '…';
    dbg.note('silence');
    return;
  }

  if (!inSpeech) {
    predEl.textContent = '…';
    dbg.note('silence (below VAD threshold)');
    return;
  }

  const feats = await extractFeaturesMonoFloat32(slice, sampleRate, MFCC_OPTS);

  // guard: zero features
  let nonZero = false;
  for (let i = 0; i < feats.length; i++) { if (feats[i] !== 0) { nonZero = true; break; } }
  if (!nonZero) {
    predEl.textContent = '…';
    dbg.note('no features (zeros)');
    return;
  }

  const { y } = await callWorker({ action: 'predict', payload: { x: Array.from(feats) } }) as { y: number[] };

  if (!y || !Array.isArray(y) || y.length < 2 || !isFinite(y[0]) || !isFinite(y[1])) {
    predEl.textContent = '…';
    dbg.setProb([NaN, NaN]);
    dbg.note('bad prediction vector');
    return;
  }

  // Normalize to probabilities if they don't sum to 1
  const sum = y.reduce((a, b) => a + (isFinite(b) ? b : 0), 0);
  const probs = sum > 1e-6 ? y.map(v => v / sum) : y;
  dbg.setProb(probs);

  const { idx, val } = argmax(probs);

  // Ephemeral label for responsiveness
  const ephemeralLabel = idx === 0 ? 'LEFT' : 'RIGHT';
  predEl.textContent = ephemeralLabel;

  // Build stable majority only with confident frames
  if (val >= CONF_THRESHOLD) pushPred(idx);

  const maj = majority();
  if (maj !== null) {
    const label = maj === 0 ? 'LEFT' : 'RIGHT';
    predEl.textContent = label;

    // FIRE ONCE PER UTTERANCE (even if same as previous label)
    if (!utteranceFired && val >= CONF_THRESHOLD) {
      utteranceFired = true;
      window.dispatchEvent(new CustomEvent('voice-command', { detail: { label } }));
      lastStable = label;
    }
  } else {
    dbg.note(`ephemeral: ${ephemeralLabel} (conf=${val.toFixed(2)})`);
  }
}

function startListening() {
  if (listenTimer) return;
  // reset session state
  lastPreds.length = 0;
  inSpeech = false;
  lastStable = null;
  utteranceFired = false;

  listenTimer = window.setInterval(livePredictTick, 250);
  predEl.textContent = '…';
  dbg.note('listening…');
  log('Live listening started.');
}
function stopListening() {
  if (!listenTimer) return;
  window.clearInterval(listenTimer);
  listenTimer = null;
  inSpeech = false;
  dbg.note('stopped');
  log('Live listening stopped.');
}

// persistence
async function saveModel() {
  try {
    const { modelJSON, scaler } = await callWorker({ action: 'export' });
    await saveModelToIDB(modelJSON, scaler);
    log('Model saved to IndexedDB.');
  } catch (e: any) {
    log('Save error: ' + e.message);
  }
}
async function loadModel() {
  try {
    await ensureWorker();
    const payload = await loadModelFromIDB();
    if (!payload) { log('No saved model in IndexedDB.'); return; }
    await callWorker({ action: 'import', payload });
    trainStatus.textContent = 'loaded ✔';
    listenBtn.disabled = false;
    saveBtn.disabled = false;
    log('Model loaded from IndexedDB.');
  } catch (e: any) {
    log('Load error: ' + e.message);
  }
}
async function resetModel() {
  try {
    await ensureWorker();
    await callWorker({ action: 'reset' });
    await deleteModelFromIDB();
    trainStatus.textContent = 'not trained';
    listenBtn.disabled = true;
    saveBtn.disabled = true;
    lastStable = null;
    utteranceFired = false;
    log('Model reset (worker + IndexedDB cleared).');
  } catch (e: any) {
    log('Reset error: ' + e.message);
  }
}

// UI wiring
micBtn.onclick = () => enableMic().catch(e => log('Mic error: ' + e.message));

recLeftBtn.onclick = async () => {
  if (!audioCtx) return log('Enable mic first.');
  if (leftSamples.length >= 4) return log('Already have 4 "left" samples.');
  log('Say “left”…');
  beginSample(1.2);
  setTimeout(async () => {
    const mono = takeCapturedMono();
    const feats = await extractFeaturesMonoFloat32(mono, sampleRate, MFCC_OPTS);
    leftSamples.push(feats);
    leftCountEl.textContent = `${leftSamples.length} / 4`;
    trainBtn.disabled = !(leftSamples.length >= 4 && rightSamples.length >= 4);
    log('Captured "left".');
  }, 1300);
};

recRightBtn.onclick = async () => {
  if (!audioCtx) return log('Enable mic first.');
  if (rightSamples.length >= 4) return log('Already have 4 "right" samples.');
  log('Say “right”…');
  beginSample(1.2);
  setTimeout(async () => {
    const mono = takeCapturedMono();
    const feats = await extractFeaturesMonoFloat32(mono, sampleRate, MFCC_OPTS);
    rightSamples.push(feats);
    rightCountEl.textContent = `${rightSamples.length} / 4`;
    trainBtn.disabled = !(leftSamples.length >= 4 && rightSamples.length >= 4);
    log('Captured "right".');
  }, 1300);
};

trainBtn.onclick = () => trainModel().catch(e => log('Train error: ' + e.message));
listenBtn.onclick = () => {
  if (listenTimer) { stopListening(); listenBtn.textContent = '🔎 Start Live Listen'; }
  else { ensureWorker().then(startListening).catch(e => log('Listen error: ' + e.message)); listenBtn.textContent = '⏹ Stop Live Listen'; }
};

saveBtn.onclick = () => saveModel();
loadBtn.onclick = () => loadModel();
resetModelBtn.onclick = () => resetModel();

// game controls
startGameBtn.onclick = () => { game.start(); startGameBtn.disabled = true; pauseGameBtn.disabled = false; };
pauseGameBtn.onclick = () => { game.pause(); startGameBtn.disabled = false; pauseGameBtn.disabled = true; };
resetGameBtn.onclick = () => { game.reset(); startGameBtn.disabled = false; pauseGameBtn.disabled = true; };

// Try auto-load a saved model on startup
loadModel().catch(() => { });
