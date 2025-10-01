// src/mfcc.ts
import Meyda from 'meyda';

export type MFCCOptions = {
  /** Silence trim threshold (lower = less aggressive). Default: 0.003 */
  trimThreshold?: number;
  /** If trimmed is too short, fall back to untrimmed signal. Default: true */
  allowUntrimFallback?: boolean;
  /** If still short, zero-pad to frame size. Default: true */
  padToFrame?: boolean;
};

// Meyda requires bufferSize to be a power of two (e.g., 512, 1024, 2048).
function isPowerOfTwo(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

// Pick a power-of-two frame size near ~25ms of audio.
function pickFrameSize(sampleRate: number): number {
  const target = Math.max(1, Math.floor(0.025 * sampleRate)); // ~25ms
  const lowerPow = 1 << Math.floor(Math.log2(target));
  const upperPow = 1 << Math.ceil(Math.log2(target));
  let chosen = Math.abs(upperPow - target) < Math.abs(lowerPow - target) ? upperPow : lowerPow;
  if (chosen < 256) chosen = 256;
  if (chosen > 4096) chosen = 4096;
  return chosen;
}

/**
 * Compute 13 MFCCs per frame; aggregate to [means(13), stds(13)] = 26-dim feature vector.
 * Robust to short/quiet inputs by falling back and padding as needed.
 */
export async function extractFeaturesMonoFloat32(
  float32: Float32Array,
  sampleRate: number,
  opts: MFCCOptions = {}
): Promise<Float32Array> {
  const {
    trimThreshold = 0.003,
    allowUntrimFallback = true,
    padToFrame = true,
  } = opts;

  // 1) Normalize amplitude
  const norm = normalize(float32);

  // 2) Trim silence (conservative)
  let sig = trimSilence(norm, trimThreshold);

  const frameSize = pickFrameSize(sampleRate);
  const hop = Math.floor(frameSize / 2);

  // 3) If trimming made it too short, fall back to untrimmed
  if (sig.length < frameSize && allowUntrimFallback) {
    sig = norm;
  }

  // 4) If STILL too short, pad with zeros (tail-pad)
  if (sig.length < frameSize && padToFrame) {
    const padded = new Float32Array(frameSize);
    const copyLen = Math.min(sig.length, frameSize);
    padded.set(sig.subarray(sig.length - copyLen), frameSize - copyLen);
    sig = padded;
  }

  // If we still can't make a frame, bail with zeros
  if (sig.length < frameSize) {
    return new Float32Array(26);
  }

  if (!isPowerOfTwo(frameSize)) {
    throw new Error('Internal error: non-power-of-two frame size selected');
  }

  const mfccs: number[][] = [];

  // 5) Slide windows; collect MFCC arrays (Meyda returns number[])
  for (let i = 0; i + frameSize <= sig.length; i += hop) {
    const frame = sig.subarray(i, i + frameSize);
    try {
      const mf = (Meyda as any).extract('mfcc', frame, {
        sampleRate,
        bufferSize: frameSize,
        melBands: 26,
        numberOfMFCCCoefficients: 13,
        windowingFunction: 'hamming',
      }) as number[] | undefined;

      if (Array.isArray(mf) && mf.length > 0) {
        mfccs.push(mf);
      }
    } catch {
      // skip problematic frame
    }
  }

  // 6) If none collected, try a single tail window
  if (mfccs.length === 0) {
    const frame = sig.subarray(sig.length - frameSize);
    try {
      const mf = (Meyda as any).extract('mfcc', frame, {
        sampleRate,
        bufferSize: frameSize,
        melBands: 26,
        numberOfMFCCCoefficients: 13,
        windowingFunction: 'hamming',
      }) as number[] | undefined;

      if (Array.isArray(mf) && mf.length > 0) {
        mfccs.push(mf);
      }
    } catch {
      // ignore
    }
  }

  // 7) Still nothing? return zeros to avoid undefined access
  if (mfccs.length === 0 || !Array.isArray(mfccs[0]) || mfccs[0].length === 0) {
    return new Float32Array(26);
  }

  // 8) Aggregate mean and std across time
  const T = mfccs.length;
  const D = mfccs[0].length;
  const means = new Array(D).fill(0);
  for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) means[d] += mfccs[t][d];
  for (let d = 0; d < D; d++) means[d] /= T;

  const stds = new Array(D).fill(0);
  for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) {
    const diff = mfccs[t][d] - means[d];
    stds[d] += diff * diff;
  }
  for (let d = 0; d < D; d++) stds[d] = Math.sqrt(stds[d] / Math.max(1, T - 1));

  const out = new Float32Array(D * 2);
  out.set(means, 0);
  out.set(stds, D);
  return out;
}

// ── helpers ──────────────────────────────────────────────────────────
function normalize(x: Float32Array): Float32Array {
  let max = 1e-9;
  for (let i = 0; i < x.length; i++) max = Math.max(max, Math.abs(x[i]));
  if (max < 1e-6) return x.slice(0);
  const y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) y[i] = x[i] / max;
  return y;
}

function trimSilence(x: Float32Array, thr = 0.003): Float32Array {
  let s = 0, e = x.length - 1;
  while (s < x.length && Math.abs(x[s]) < thr) s++;
  while (e > s && Math.abs(x[e]) < thr) e--;
  return x.subarray(s, e + 1);
}
