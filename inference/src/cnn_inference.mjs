/**
 * CNN classifier inference.
 *
 * Uses ONNX Runtime Web (UMD build) loaded from a CDN at runtime, so the
 * repository ships no vendor binaries. Each crossing is extracted as a
 * 40x40 two-channel patch (V=max(RGB), M=min(RGB)) from the warped image
 * and classified into empty / black / white. Channel order and NCHW layout
 * must match train_cnn.py precompute_patches.
 *
 * The ONNX model is produced by training/cnn/train_cnn.py (TRAINING.md)
 * and checked into the repository at training/out-cnn/cnn.onnx.
 */
import { warpPerspective } from "./warp.mjs";

/** Must match the cell size used when training (train_cnn.py --cell). */
export const CELL_DEFAULT = 40;
export const PATCH_SIZE = 40;

const ORT_VERSION = "1.19.2";
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

let _ortReady = null;

function loadOrt() {
  if (_ortReady) return _ortReady;
  _ortReady = new Promise((resolve, reject) => {
    if (typeof ort !== "undefined") {
      resolve(ort);
      return;
    }
    const script = document.createElement("script");
    script.src = ORT_CDN + "ort.min.js";
    script.onload = () => {
      if (typeof ort === "undefined") {
        reject(new Error("ORT loaded but ort global not found"));
        return;
      }
      // wasm assets come from the same CDN directory as the UMD bundle.
      ort.env.wasm.wasmPaths = ORT_CDN;
      // Single-threaded build avoids the COOP/COEP (SharedArrayBuffer)
      // requirement, so the demo works from a plain static server.
      ort.env.wasm.numThreads = 1;
      resolve(ort);
    };
    script.onerror = () => reject(new Error("Failed to load ORT runtime from CDN"));
    document.head.appendChild(script);
  });
  return _ortReady;
}

/**
 * Extract 40x40 two-channel patches from warped image at each crossing.
 * NCHW layout [P, 2, 40, 40]: channel 0 = V (max RGB), channel 1 = M (min RGB).
 * @param {Uint8Array} V  max-channel plane (Wd*Wd)
 * @param {Uint8Array} M  min-channel plane (Wd*Wd)
 */
export function extractPatches(V, M, Wd, sites, cell) {
  const P = sites.length;
  const half = PATCH_SIZE >> 1;
  const PLANE = PATCH_SIZE * PATCH_SIZE;
  const CH = 2;
  const patches = new Float32Array(P * CH * PLANE);

  for (let p = 0; p < P; p++) {
    const cx = cell + sites[p].col * cell;
    const cy = cell + sites[p].row * cell;
    const x0 = Math.round(cx) - half;
    const y0 = Math.round(cy) - half;
    const base = p * CH * PLANE;

    for (let dy = 0; dy < PATCH_SIZE; dy++) {
      for (let dx = 0; dx < PATCH_SIZE; dx++) {
        const px = x0 + dx;
        const py = y0 + dy;
        let v = 0;
        let m = 0;
        if (px >= 0 && px < Wd && py >= 0 && py < Wd) {
          const idx = py * Wd + px;
          v = V[idx];
          m = M[idx];
        }
        const o = dy * PATCH_SIZE + dx;
        patches[base + 0 * PLANE + o] = v / 255.0;
        patches[base + 1 * PLANE + o] = m / 255.0;
      }
    }
  }
  return patches;
}

/**
 * Run CNN inference on all crossings.
 */
export async function runCnnReading(bench, surveyResult, bundle) {
  const lines = surveyResult.lines;
  const sites = surveyResult.sites;
  const cell = CELL_DEFAULT;

  // Warp
  const xy = (i, j) => [sites[i * lines + j].x, sites[i * lines + j].y];
  const quad = [
    xy(0, 0), xy(0, lines - 1),
    xy(lines - 1, lines - 1), xy(lines - 1, 0),
  ];
  const warped = warpPerspective(bench, quad, lines, cell);
  // `M` is already used below for the classification result matrix, so alias
  // the min-channel plane from the warp to a distinct name.
  const { V, M: MinPlane, Wd } = warped;

  // Extract patches (2-channel NCHW)
  const patches = extractPatches(V, MinPlane, Wd, sites, cell);

  // Load ORT + create session
  const ort = await loadOrt();
  const session = await ort.InferenceSession.create(bundle.onnxModel);
  const input = new ort.Tensor("float32", patches, [sites.length, 2, PATCH_SIZE, PATCH_SIZE]);
  const results = await session.run({ patch: input });
  const logits = results.logits.data;

  // Argmax + softmax
  const P = lines * lines;
  const M = new Int8Array(P);
  const C = new Float32Array(P);
  for (let i = 0; i < P; i++) {
    let best = 0;
    let bestV = logits[i * 3];
    for (let c = 1; c < 3; c++) {
      if (logits[i * 3 + c] > bestV) {
        bestV = logits[i * 3 + c];
        best = c;
      }
    }
    M[i] = best;
    const mx = Math.max(logits[i*3], logits[i*3+1], logits[i*3+2]);
    const sum = Math.exp(logits[i*3]-mx) + Math.exp(logits[i*3+1]-mx) + Math.exp(logits[i*3+2]-mx);
    C[i] = Math.exp(logits[i*3+best]-mx) / sum;
  }

  const occupied = M.reduce((a, v) => a + (v !== 0 ? 1 : 0), 0);
  const meanConf = C.reduce((a, v) => a + v, 0) / P;

  return {
    ok: true,
    P: null,
    M,
    C,
    quality: meanConf,
    refused: false,
    evidence: { descriptors: 1, sites: P },
    gates: {},
    diagnostics: {
      occupied,
      meanConfidence: meanConf,
      cell,
      patchSize: PATCH_SIZE,
      method: "cnn",
    },
  };
}
