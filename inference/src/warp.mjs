/**
 * Perspective warp (inverse homography + bilinear), pure ES module.
 *
 * Maps the playable-grid quadrilateral into a square image where every grid
 * cell is `cell` pixels, with a `cell`-pixel margin on each side. Two channels
 * are produced from the SAME bilinearly interpolated R/G/B: `V` = max(R,G,B)
 * and `M` = min(R,G,B), both uint8 0..255. The CNN consumes them as a
 * 2-channel patch (V, M). No hand-crafted features, thresholds or geometry
 * constants are introduced.
 *
 * JS↔Python parity: the homography is solved independently (8×8 Gaussian
 * elimination vs numpy.linalg.solve), but both target the same unit-square
 * parameterisation and bilinear interpolation formula. Both channels are
 * checked against training/cnn/warp.py pixel-by-pixel.
 */

/** Solve an n×n linear system by Gaussian elimination with partial pivoting. */
function gaussianSolve(rows, rhs) {
  const n = rows.length;
  const M = rows.map((r, i) => [...r, rhs[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    if (Math.abs(M[c][c]) < 1e-12) throw new Error("weiqi-photo: singular matrix");
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

/** 3×3 homography H mapping unit-square (u,v) → raster (x,y). */
export function unitFromRaster(quad) {
  const dst = [[0, 0], [1, 0], [1, 1], [0, 1]];
  const rows = [];
  const rhs = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = quad[i];
    const [u, v] = dst[i];
    rows.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); rhs.push(u);
    rows.push([0, 0, 0, x, y, 1, -v * x, -v * y]); rhs.push(v);
  }
  const sol = gaussianSolve(rows, rhs);
  return [
    [sol[0], sol[1], sol[2]],
    [sol[3], sol[4], sol[5]],
    [sol[6], sol[7], 1.0],
  ];
}

/** Invert a 3×3 matrix. */
function invert3(m) {
  const [a, b, c] = m[0];
  const [d, e, f] = m[1];
  const [g, h, i] = m[2];
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (Math.abs(det) < 1e-15) throw new Error("weiqi-photo: singular homography");
  const inv = 1 / det;
  return [
    [(e * i - f * h) * inv, (c * h - b * i) * inv, (b * f - c * e) * inv],
    [(f * g - d * i) * inv, (a * i - c * g) * inv, (c * d - a * f) * inv],
    [(d * h - e * g) * inv, (b * g - a * h) * inv, (a * e - b * d) * inv],
  ];
}

/**
 * Perspective warp of the board region into a square image with `cell` pixels
 * per grid cell and a `cell`-pixel margin.
 *
 * @param {{data:ArrayLike<number>,width:number,height:number}} bench
 * @param {number[][]} quad — four [x,y] raster corners (TL, TR, BR, BL)
 * @param {number} nLines — board size (19 / 13 / 9)
 * @param {number} cell — pixels per grid cell in the warped image
 * @returns {{V:Uint8Array, M:Uint8Array, Wd:number, cell:number}}
 */
export function warpPerspective(bench, quad, nLines, cell) {
  const { data, width: sW, height: sH } = bench;
  const stride = data.length === sW * sH * 4 ? 4 : 3;
  const margin = cell;
  const S = cell * (nLines - 1);
  const Wd = S + 2 * margin;

  const H = unitFromRaster(quad);
  const Hi = invert3(H);

  const V = new Uint8Array(Wd * Wd);
  const M = new Uint8Array(Wd * Wd);

  for (let gy = 0; gy < Wd; gy++) {
    const v = (gy - margin) / S;
    for (let gx = 0; gx < Wd; gx++) {
      const u = (gx - margin) / S;
      const w = Hi[2][0] * u + Hi[2][1] * v + Hi[2][2];
      const sx = (Hi[0][0] * u + Hi[0][1] * v + Hi[0][2]) / w;
      const sy = (Hi[1][0] * u + Hi[1][1] * v + Hi[1][2]) / w;

      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const fx = sx - x0;
      const fy = sy - y0;
      const x0c = Math.max(0, Math.min(x0, sW - 1));
      const x1c = Math.max(0, Math.min(x0 + 1, sW - 1));
      const y0c = Math.max(0, Math.min(y0, sH - 1));
      const y1c = Math.max(0, Math.min(y0 + 1, sH - 1));

      const w00 = (1 - fx) * (1 - fy);
      const w10 = fx * (1 - fy);
      const w01 = (1 - fx) * fy;
      const w11 = fx * fy;

      const i00 = (y0c * sW + x0c) * stride;
      const i10 = (y0c * sW + x1c) * stride;
      const i01 = (y1c * sW + x0c) * stride;
      const i11 = (y1c * sW + x1c) * stride;

      const r = data[i00] * w00 + data[i10] * w10 + data[i01] * w01 + data[i11] * w11;
      const g = data[i00 + 1] * w00 + data[i10 + 1] * w10 + data[i01 + 1] * w01 + data[i11 + 1] * w11;
      const b = data[i00 + 2] * w00 + data[i10 + 2] * w10 + data[i01 + 2] * w01 + data[i11 + 2] * w11;
      V[gy * Wd + gx] = Math.max(r, g, b);
      M[gy * Wd + gx] = Math.min(r, g, b);
    }
  }

  return { V, M, Wd, cell };
}
