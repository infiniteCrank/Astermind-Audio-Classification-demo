// src/main.ts
import { extractFeaturesMonoFloat32 } from './mfcc';
import WorkerCtor from './worker.ts?worker';

const $ = <T extends Element = Element>(sel: string) =>
  document.querySelector(sel) as T;

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

function log(s: string) {
  logEl.textContent += s + '\n';
  logEl.scrollTop = logEl.scrollHeight;
}

/** ───────────────────────── Config knobs ───────────────────────── **/
let VAD_RMS_THRESHOLD = 0.006;  // lower lets more speech through
let CONF_THRESHOLD = 0.50;      // allow slightly lower confidence
const PRED_WINDOW = 5;          // majority vote window (frames)
const REQUIRE_VAD = true;       // set false to test without RMS gating

// MFCC extraction options (matches mfcc.ts robust behavior)
const MFCC_OPTS = {
  trimThreshold: 0.003,
  allowUntrimFallback: true,
  padToFrame: true,
};

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

/** ───────────────────────── Debug panel ───────────────────────── **/
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
    setRMS(v: number) {
      rmsEl.textContent = `RMS: ${v.toFixed(4)}`;
    },
    setProb(p: number[]) {
      probEl.textContent = `P(left,right): [${p.map((n) => n.toFixed(2)).join(', ')}]`;
    },
    note(s: string) {
      noteEl.textContent = s;
    },
  };
})();

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
  captureBuf.length = 0; // reset rolling window
  window.setTimeout(() => { /* window elapsed */ }, Math.max(200, seconds * 1000));
}

function takeCapturedMono(): Float32Array {
  let total = 0;
  for (const c of captureBuf) total += c.length;
  const mono = new Float32Array(total);
  let off = 0;
  for (const c of captureBuf) {
    mono.set(c, off);
    off += c.length;
  }
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
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    s += v * v;
  }
  return Math.sqrt(s / Math.max(1, x.length));
}
function argmax(v: number[]): { idx: number; val: number } {
  let bi = 0,
    bv = -Infinity;
  for (let i = 0; i < v.length; i++) if (v[i] > bv) { bv = v[i]; bi = i; }
  return { idx: bi, val: bv };
}

/** ───────────────────────── Smoothing ───────────────────────── **/
const lastPreds: number[] = [];
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

/** ───────────────────────── Worker bridge ───────────────────────── **/
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

/** ───────────────────────── Train ───────────────────────── **/
async function trainModel() {
  await ensureWorker();

  const X: number[][] = [];
  const y: number[][] = [];
  leftSamples.forEach((x) => {
    X.push(Array.from(x));
    y.push(oneHot(0));
  });
  rightSamples.forEach((x) => {
    X.push(Array.from(x));
    y.push(oneHot(1));
  });
  if (X.length < 8) {
    log('Need 4 left + 4 right before training.');
    return;
  }

  trainStatus.textContent = 'training...';
  const modelOptions = {
    task: 'classification',
    outputDim: 2,
    kernel: { type: 'rbf', sigma: 1.0 },
    ridgeLambda: 1e-2,
  };
  await callWorker({ action: 'train', payload: { X, y, modelOptions } });
  trainStatus.textContent = 'trained ✔';
  listenBtn.disabled = false;
  log('Training complete.');
}

/** ───────────────────────── Live listen loop ───────────────────────── **/
async function livePredictTick() {
  if (!worker) return;

  // Take the last ~0.8s of audio
  const framesNeeded = Math.ceil(0.8 * sampleRate);
  const mono = takeCapturedMono();
  const slice =
    mono.length > framesNeeded ? mono.subarray(mono.length - framesNeeded) : mono;

  const e = rms(slice);
  dbg.setRMS(e);

  if (REQUIRE_VAD && e < VAD_RMS_THRESHOLD) {
    predEl.textContent = '…'; // silence
    dbg.note('silence (below VAD threshold)');
    return;
  }

  const feats = await extractFeaturesMonoFloat32(
    slice,
    sampleRate,
    MFCC_OPTS
  );

  // Guard: empty/zero features → treat as silence
  let nonZero = false;
  for (let i = 0; i < feats.length; i++) {
    if (feats[i] !== 0) { nonZero = true; break; }
  }
  if (!nonZero) {
    predEl.textContent = '…';
    dbg.note('no features (zeros)');
    return;
  }

  const { y } = (await callWorker({
    action: 'predict',
    payload: { x: Array.from(feats) },
  })) as { y: number[] };

  // Guard unexpected outputs
  if (!y || !Array.isArray(y) || y.length < 2 || !isFinite(y[0]) || !isFinite(y[1])) {
    predEl.textContent = '…';
    dbg.setProb([NaN, NaN]);
    dbg.note('bad prediction vector');
    return;
  }

  // Normalize to probabilities if they don't sum to 1
  const sum = y.reduce((a, b) => a + (isFinite(b) ? b : 0), 0);
  const probs = sum > 1e-6 ? y.map((v) => v / sum) : y;
  dbg.setProb(probs);

  const { idx, val } = argmax(probs);

  // Confidence gating (soft): show ephemeral label even if < threshold,
  // but only feed smoothing buffer when confident.
  if (val < CONF_THRESHOLD) {
    predEl.textContent = idx === 0 ? 'LEFT' : 'RIGHT';
    dbg.note(`low confidence (${val.toFixed(2)})`);
    return;
  }

  pushPred(idx);
  const maj = majority();
  const finalIdx = maj !== null ? maj : idx;
  predEl.textContent = finalIdx === 0 ? 'LEFT' : 'RIGHT';
  dbg.note(`conf=${val.toFixed(2)} (maj=${finalIdx === 0 ? 'L' : 'R'})`);
}

/** ───────────────────────── Controls ───────────────────────── **/
function startListening() {
  if (listenTimer) return;
  listenTimer = window.setInterval(livePredictTick, 250);
  predEl.textContent = '…';
  lastPreds.length = 0;
  dbg.note('listening…');
  log('Live listening started.');
}
function stopListening() {
  if (!listenTimer) return;
  window.clearInterval(listenTimer);
  listenTimer = null;
  dbg.note('stopped');
  log('Live listening stopped.');
}

/** ───────────────────────── UI wiring ───────────────────────── **/
micBtn.onclick = () => enableMic().catch((e) => log('Mic error: ' + e.message));

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

trainBtn.onclick = () =>
  trainModel().catch((e) => log('Train error: ' + e.message));

listenBtn.onclick = () => {
  if (listenTimer) {
    stopListening();
    listenBtn.textContent = '🔎 Start Live Listen';
  } else {
    ensureWorker().then(startListening).catch((e) => log('Listen error: ' + e.message));
    listenBtn.textContent = '⏹ Stop Live Listen';
  }
};
