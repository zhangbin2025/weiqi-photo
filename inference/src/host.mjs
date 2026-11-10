/**
 * Browser host adapter.
 *
 * Thin adapter around the platform's own decoder (createImageBitmap + canvas).
 * It performs NO recognition and NO parameter choices; it only turns a File
 * into a byte array + RGBA raster for the inference pipeline.
 *
 * The raster is returned in the decoder's own coordinate system, together
 * with its width/height.
 */
export async function decodeFile(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const bitmap = await createImageBitmap(file);
  const surface = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = surface.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("host: 2d context unavailable");
  ctx.drawImage(bitmap, 0, 0);
  const img = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close();
  return {
    bytes,
    decoded: { data: img.data, width: img.width, height: img.height },
  };
}

/** Draw the four-corner selection overlay + crossing sites for human review. */
export function paintVisual(canvas, visual, cornerDraft) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!visual) return;
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(255,255,255,0.35)";
  for (const s of visual.sites ?? []) {
    ctx.beginPath();
    ctx.arc(s.x, s.y, 1.5, 0, Math.PI * 2);
    ctx.stroke();
  }
  const pts = cornerDraft?.length ? cornerDraft : visual.corners;
  if (pts?.length >= 2) {
    ctx.strokeStyle = "rgba(255,220,80,0.95)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i += 1) ctx.lineTo(pts[i].x, pts[i].y);
    if (pts.length === 4) ctx.closePath();
    ctx.stroke();
    for (const p of pts) {
      ctx.fillStyle = "rgba(255,80,80,0.95)";
      ctx.fillRect(p.x - 3, p.y - 3, 6, 6);
    }
  }
}

/** Compact, dependency-free digest of bytes for reproducible run stamping. */
export async function digest(bytes) {
  const h = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
