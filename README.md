# Left/Right Voice Classifier (Vite + TypeScript)

Tiny on-device classifier using **@astermind/astermind-elm** and **Meyda**.
Users record 4 samples of “left” and 4 samples of “right”, we compute MFCC features,
train an ELM/KELM in a Web Worker, and then live-classify the mic stream.

## Quickstart
```bash
npm install
npm run dev
# open the printed local URL (usually http://localhost:5173)
```

## How it works
- **Audio capture**: Web Audio API captures mono PCM.
- **Features**: We compute 13 MFCCs per frame (~25ms windows) and aggregate with mean+std → 26-dim vector per utterance.
- **Training**: Dataset (8 samples) is sent to a Web Worker that instantiates **KernelELM (RBF)** if available or falls back to **ELM**.
- **Inference**: Every ~250ms, we grab ~0.8s of recent audio, compute features, and run `predict` for LEFT/RIGHT.

## Notes
- If your editor complains about `meyda` types, see `src/types.d.ts` (ambient module).
- You can tweak the kernel by editing `modelOptions` in `src/main.ts` before training
  (e.g., change `sigma`, or switch to `kernel: { type: 'linear' }`).

## Ideas to extend
- Voice activity detection to auto-start stop capture
- Save/load the trained model JSON
- Add more commands/classes
# Astermind-Audio-Classification-demo
