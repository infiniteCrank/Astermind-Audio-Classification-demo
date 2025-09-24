// Standardizes features (z-score); supports KernelELM + ELM; adds export/import/reset.
import { ELM, KernelELM } from '@astermind/astermind-elm';

type TrainMsg = { action: 'train', payload: { X: number[][], y: number[][] | number[], modelOptions?: any } };
type PredictMsg = { action: 'predict', payload: { x: number[] } };
type Action = 'train' | 'predict' | 'init' | 'export' | 'import' | 'reset';
type InMsg = { id: string; action: Action; payload?: any };
let model: any = null;
let scaler: { mu: number[], sigma: number[] } | null = null;

function post(id: string, ok: boolean, result?: any, error?: string) {
  (self as any).postMessage({ id, ok, result, error });
}

function deepMerge(base: any, extra: any): any {
  const out: any = Array.isArray(base) ? base.slice() : { ...base };
  if (!extra) return out;
  for (const k of Object.keys(extra)) {
    const v = (extra as any)[k];
    if (v && typeof v === 'object' && !Array.isArray(v)) out[k] = deepMerge(out[k] ?? {}, v);
    else out[k] = v;
  }
  return out;
}

// ---- z-score scaler ----
function fitScaler(X: number[][]): { mu: number[], sigma: number[] } {
  const n = X.length, d = X[0].length;
  const mu = new Array(d).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mu[j] += X[i][j];
  for (let j = 0; j < d; j++) mu[j] /= n;

  const sd = new Array(d).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) {
    const diff = X[i][j] - mu[j];
    sd[j] += diff * diff;
  }
  for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j] / Math.max(1, n - 1)) || 1e-6;
  return { mu, sigma: sd };
}
function transformOne(x: number[], sc: { mu: number[], sigma: number[] }): number[] {
  const y = new Array(x.length);
  for (let j = 0; j < x.length; j++) y[j] = (x[j] - sc.mu[j]) / (sc.sigma[j] || 1e-6);
  return y;
}
function transformX(X: number[][], sc: { mu: number[], sigma: number[] }): number[][] {
  return X.map(r => transformOne(r, sc));
}

// ---- labels ----
function oneHotify(y: number[] | number[][], k: number): number[][] {
  if (Array.isArray(y) && typeof y[0] === 'number') {
    return (y as number[]).map((lab) => {
      const row = new Array(k).fill(0);
      row[Math.max(0, Math.min(k - 1, lab))] = 1;
      return row;
    });
  }
  return y as number[][];
}

function probs1D(m: any, x: number[]): number[] {
  if (typeof m.predictProbaFromVectors === 'function') {
    const P = m.predictProbaFromVectors([x]);
    return Array.isArray(P) && Array.isArray(P[0]) ? (P[0] as number[]) : [];
  }
  if (typeof m.predictLogitsFromVectors === 'function') {
    const L = m.predictLogitsFromVectors([x]) as number[][];
    const logits = Array.isArray(L) && Array.isArray(L[0]) ? (L[0] as number[]) : [];
    const max = Math.max(...logits);
    const exps = logits.map(v => Math.exp(v - max));
    const s = exps.reduce((a, b) => a + b, 0) || 1;
    return exps.map(v => v / s);
  }
  if (typeof m.predict === 'function') {
    const out = m.predict([x]) as number[][];
    return Array.isArray(out) && Array.isArray(out[0]) ? (out[0] as number[]) : [];
  }
  throw new Error('Model has no predict method');
}

// ---- export/import helpers ----
function modelToJSON(m: any): any {
  if (!m) throw new Error('No model to export');
  if (typeof m.toJSON === 'function') return m.toJSON();
  if (typeof m.toJSONString === 'function') return JSON.parse(m.toJSONString());
  // last resort: shallow dump known fields (may not work on all versions)
  const obj: any = {};
  for (const k of Object.keys(m)) {
    const v = (m as any)[k];
    if (typeof v !== 'function') obj[k] = v;
  }
  return obj;
}
function loadModelFromJSON(json: any): any {
  const ctor = (typeof KernelELM !== 'undefined' && KernelELM) ? KernelELM : ELM;
  const cfg = json?.config || json?.cfg || {};
  const m = new ctor(cfg);
  if (typeof (m as any).loadModelFromJSON === 'function') {
    (m as any).loadModelFromJSON(JSON.stringify(json));
    return m;
  }
  if (typeof (m as any).fromJSON === 'function') {
    (m as any).fromJSON(json);
    return m;
  }
  throw new Error('Model lacks loadModelFromJSON/fromJSON');
}

self.onmessage = async (e: MessageEvent<InMsg>) => {
  const { id, action, payload } = e.data || ({} as InMsg);

  try {
    if (action === 'init') {
      post(id, true, { ready: true });
      return;
    }

    if (action === 'train') {
      const { X, y, modelOptions } = (payload as TrainMsg['payload']) || ({} as any);
      if (!X?.length || !y?.length) throw new Error('train: missing X or y');

      const outDim = Array.isArray(y[0]) ? (y[0] as number[]).length : 1;
      scaler = fitScaler(X);
      const Xs = transformX(X, scaler);
      const Y = oneHotify(y as any, outDim);

      if (typeof KernelELM !== 'undefined' && KernelELM) {
        const defaults = { task: 'classification', outputDim: outDim, kernel: { type: 'rbf' }, ridgeLambda: 1e-2 };
        const cfg = deepMerge(defaults, modelOptions || {});
        model = new KernelELM(cfg);
        model.fit(Xs, Y);
      } else {
        const defaults = { categories: Array.from({ length: outDim }, (_, i) => String(i)), outputDim: outDim, hiddenUnits: 128, activation: 'leakyrelu', ridgeLambda: 1e-2 };
        const cfg = deepMerge(defaults, modelOptions || {});
        model = new ELM(cfg);
        if (typeof model.trainFromData === 'function') model.trainFromData(Xs, Y);
        else if (typeof model.train === 'function') model.train({ X: Xs, y: Y });
        else throw new Error('ELM: no train/trainFromData method');
      }

      post(id, true, { trained: true });
      return;
    }

    if (action === 'predict') {
      if (!model) throw new Error('Model not trained yet');
      const { x } = (payload as PredictMsg['payload']) || ({} as any);
      const xs = scaler ? transformOne(x, scaler) : x;
      const yhat = probs1D(model, xs);
      post(id, true, { y: yhat });
      return;
    }

    if (action === 'export') {
      if (!model) throw new Error('No model to export');
      const modelJSON = modelToJSON(model);
      post(id, true, { modelJSON, scaler });
      return;
    }

    if (action === 'import') {
      const { modelJSON, scaler: sc } = (payload as any) || {};
      if (!modelJSON) throw new Error('Missing modelJSON');
      model = loadModelFromJSON(modelJSON);
      scaler = sc || null;
      post(id, true, { imported: true });
      return;
    }

    if (action === 'reset') {
      model = null;
      scaler = null;
      post(id, true, { reset: true });
      return;
    }

    post(id, false, null, `Unknown action: ${String(action)}`);
  } catch (err: any) {
    post(id, false, null, err?.message || String(err));
  }
};
