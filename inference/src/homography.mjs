/**
 * Board geometry and planar homography utilities.
 *
 * Computes a projective map from a user-supplied quadrilateral (four board
 * corners, in raster coordinates) to the unit board square, and back.
 * This is pure linear algebra: solving an 8x8 system from four point
 * correspondences (the direct linear transform, a standard textbook method
 * for planar homography estimation).
 *
 * No thresholds, no weights, no tuned constants. A degenerate quad is
 * reported as such so the delivery stage can refuse.
 */

/** Solve A h = b by Gaussian elimination with partial pivoting. */
function solveLinear(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let r = col + 1; r < n; r += 1) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    if (Math.abs(M[pivot][col]) < Number.EPSILON) return null;
    [M[col], M[pivot]] = [M[pivot], M[col]];
    for (let r = col + 1; r < n; r += 1) {
      const f = M[r][col] / M[col][col];
      for (let c = col; c <= n; c += 1) M[r][c] -= f * M[col][c];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r -= 1) {
    let s = M[r][n];
    for (let c = r + 1; c < n; c += 1) s -= M[r][c] * x[c];
    x[r] = s / M[r][r];
  }
  return x;
}

/**
 * Build the map from raster coordinates → unit board coordinates (u,v in [0,1]).
 * @param {{x:number,y:number}[]} quad - four corners in cyclic order:
 *        [top-left, top-right, bottom-right, bottom-left] of the PLAYABLE grid.
 * @returns {{ h: number[9], ok: boolean, reason: string }}
 */
export function unitFromQuad(quad) {
  if (!Array.isArray(quad) || quad.length !== 4
      || quad.some((p) => !Number.isFinite(p.x) || !Number.isFinite(p.y))) {
    return { h: null, ok: false, reason: "quad must be four finite points" };
  }
  const dst = [[0, 0], [1, 0], [1, 1], [0, 1]];
  const A = [];
  const b = [];
  for (let k = 0; k < 4; k += 1) {
    const { x, y } = quad[k];
    const [u, v] = dst[k];
    // h = [a,b,c,d,e,f,g,i,j] with j fixed to 1.
    // u = (a x + b y + c) / (g x + i y + 1)
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    // v = (d x + e y + f) / (g x + i y + 1)
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  }
  const sol = solveLinear(A, b);
  if (!sol) return { h: null, ok: false, reason: "degenerate quadrilateral (singular system)" };
  return { h: [...sol, 1], ok: true, reason: "ok" };
}

function project(h, x, y) {
  const den = h[6] * x + h[7] * y + h[8];
  if (Math.abs(den) < Number.EPSILON) return null;
  return {
    x: (h[0] * x + h[1] * y + h[2]) / den,
    y: (h[3] * x + h[4] * y + h[5]) / den,
  };
}

/** Inverse map (unit board → raster) using the adjugate of the 3x3 matrix. */
export function rasterFromUnit(h) {
  const [a, b, c, d, e, f, g, i, j] = h;
  return [
    e * j - f * i, c * i - b * j, b * f - c * e,
    f * g - d * j, a * j - c * g, c * d - a * f,
    d * i - e * g, b * g - a * i, a * e - b * d,
  ];
}

/**
 * Sample the N×N crossing points in raster coordinates.
 * @param {number[9]} hInv - result of rasterFromUnit()
 * @param {number} lines - board line count (9 / 13 / 19)
 */
export function crossingSites(hInv, lines) {
  const sites = [];
  for (let row = 0; row < lines; row += 1) {
    for (let col = 0; col < lines; col += 1) {
      const u = lines === 1 ? 0 : col / (lines - 1);
      const v = lines === 1 ? 0 : row / (lines - 1);
      const p = project(hInv, u, v);
      if (!p) return { sites: null, ok: false, reason: "projection singular at a crossing" };
      sites.push({ row, col, u, v, x: p.x, y: p.y });
    }
  }
  return { sites, ok: true, reason: "ok" };
}

/** Mean distance between adjacent crossing points, in raster pixels.
 *  This is a MEASURED geometric quantity (from the supplied quad), not a
 *  configured parameter — it is the "grid spacing" unit used by later stages. */
export function gridSpacing(sites, lines) {
  let sum = 0;
  let n = 0;
  for (let row = 0; row < lines; row += 1) {
    for (let col = 0; col < lines - 1; col += 1) {
      const a = sites[row * lines + col];
      const c = sites[row * lines + col + 1];
      sum += Math.hypot(c.x - a.x, c.y - a.y);
      n += 1;
    }
  }
  return n ? sum / n : null;
}

/** Check the quad is convex and the projected grid stays within the raster. */
export function geometrySanity(quad, sites, width, height) {
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const signs = [];
  for (let k = 0; k < 4; k += 1) {
    signs.push(Math.sign(cross(quad[k], quad[(k + 1) % 4], quad[(k + 2) % 4])));
  }
  const convex = signs.every((s) => s === signs[0] && s !== 0);
  let inside = 0;
  for (const s of sites) {
    if (s.x >= 0 && s.x <= width && s.y >= 0 && s.y <= height) inside += 1;
  }
  return {
    convex,
    coverage: sites.length ? inside / sites.length : 0,
    ok: convex,
    reason: convex ? "ok" : "non-convex or self-intersecting quadrilateral",
  };
}
