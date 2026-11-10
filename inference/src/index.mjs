/**
 * weiqi-photo — public library entry.
 *
 * This is the single module a host application (e.g. weiqi-bot) should import.
 * The repository is NOT an npm package: it ships inference code, training code,
 * docs and the trained ONNX model (training/out-cnn/cnn.onnx). Hosts consume
 * these native ES modules directly (via a bundler alias or a relative path)
 * and supply the ONNX model bytes themselves.
 *
 * Public surface (mirror this list in the host's `declare module` .d.ts):
 *   decodeFile, paintVisual, digest                       (host.mjs)
 *   unitFromQuad, rasterFromUnit, crossingSites,
 *   gridSpacing, geometrySanity                          (homography.mjs)
 *   unitFromRaster, warpPerspective                      (warp.mjs)
 *   CELL_DEFAULT, PATCH_SIZE, extractPatches,
 *   runCnnReading                                        (cnn_inference.mjs)
 *   transcribe                                           (this file)
 *
 * Zero runtime dependencies. ONNX Runtime Web is loaded from a CDN at runtime
 * (src/cnn_inference.mjs). The .onnx model is produced by
 * training/cnn/train_cnn.py (TRAINING.md), checked into the repository, and
 * never embedded in this library.
 */
export { decodeFile, paintVisual, digest } from "./host.mjs";
export {
  unitFromQuad, rasterFromUnit, crossingSites, gridSpacing, geometrySanity,
} from "./homography.mjs";
export { unitFromRaster, warpPerspective } from "./warp.mjs";
export {
  CELL_DEFAULT, PATCH_SIZE, extractPatches, runCnnReading,
} from "./cnn_inference.mjs";

import { unitFromQuad, rasterFromUnit, crossingSites, gridSpacing } from "./homography.mjs";
import { runCnnReading } from "./cnn_inference.mjs";

/**
 * Transcribe a photographed Go board into a stone matrix.
 *
 * Wraps the exact orchestration used by inference/app/main.js so a host does
 * not need to know about homographies, crossing sites or survey objects:
 *   unitFromQuad -> rasterFromUnit -> crossingSites -> gridSpacing -> runCnnReading
 *
 * @param {{data:Uint8Array|Uint8ClampedArray,width:number,height:number}} imageData
 *        RGBA raster in ORIGINAL photo pixel coordinates. Use a canvas'
 *        getImageData(...), or decodeFile(file).decoded.
 * @param {{x:number,y:number}[]} quad
 *        Four corners of the PLAYABLE grid in raster (photo) pixel coords,
 *        cyclic order [top-left, top-right, bottom-right, bottom-left] —
 *        same order as the demo's four clicks.
 * @param {number} lines  Board line count (9 / 13 / 19).
 * @param {Uint8Array} onnxModel  Bytes of cnn.onnx (see TRAINING.md).
 * @returns {Promise<{ok:boolean,size:number,stones:number[],M:Int8Array,
 *           C:Float32Array,quality:number,diagnostics:object}>}
 *        stones is a lines*lines flat array, 0=empty / 1=black / 2=white —
 *        the same encoding as weiqi-bot's existing result.stones[].
 * @throws {Error} on invalid input, a degenerate quad or a singular projection.
 */
export async function transcribe(imageData, quad, lines, onnxModel) {
  if (!imageData || !imageData.data || !imageData.width || !imageData.height) {
    throw new Error("transcribe: imageData must be an RGBA raster {data,width,height}");
  }
  if (!Array.isArray(quad) || quad.length !== 4) {
    throw new Error("transcribe: quad must be four points [TL,TR,BR,BL]");
  }
  if (!Number.isInteger(lines) || lines < 2) {
    throw new Error("transcribe: lines must be an integer >= 2");
  }
  if (!onnxModel || typeof onnxModel.byteLength !== "number") {
    throw new Error("transcribe: onnxModel (Uint8Array of cnn.onnx) is required");
  }

  const q = unitFromQuad(quad);
  if (!q.ok) throw new Error("transcribe: geometry: " + q.reason);
  const hInv = rasterFromUnit(q.h);
  const built = crossingSites(hInv, lines);
  if (!built.ok) throw new Error("transcribe: geometry: " + built.reason);
  const sites = built.sites;
  const spacing = gridSpacing(sites, lines);
  const survey = { sites, spacing, lines, coverage: 1 };

  const res = await runCnnReading(imageData, survey, { onnxModel });
  return {
    ok: res.ok,
    size: lines,
    stones: Array.from(res.M),
    M: res.M,
    C: res.C,
    quality: res.quality,
    diagnostics: res.diagnostics,
  };
}
