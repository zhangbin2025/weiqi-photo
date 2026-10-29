/**
 * 视差场拟合。
 *
 * 拍摄角度倾斜时，子的实际中心会系统性偏离格点。
 * 这里用二次曲面拟合出每个格点的偏移量，然后按偏移重新采样特征。
 * 默认关闭。
 */

import type { Warped } from "../geometry/warp.js";
import type { OffsetField } from "./feats.js";
import { hypot2 } from "../math/stats.js";

const CT: number[] = [];
const ST: number[] = [];
for (let t = 0; t < 48; t++) {
  CT.push(Math.cos((2 * Math.PI * t) / 48));
  ST.push(Math.sin((2 * Math.PI * t) / 48));
}

export interface ShiftParams {
  sr: number; cmin: number; min: number; tol: number;
}

/** 6 项二次曲面：1, u, v, u², v², uv */
function solve6(M: number[][], y: number[]): number[] {
  const n = 6;
  for (let a = 0; a < n; a++) M[a][a] += 1e-3;
  for (let a = 0; a < n; a++) {
    let pv = a;
    for (let b = a + 1; b < n; b++) if (Math.abs(M[b][a]) > Math.abs(M[pv][a])) pv = b;
    const tm = M[a]; M[a] = M[pv]; M[pv] = tm;
    const ty = y[a]; y[a] = y[pv]; y[pv] = ty;
    for (let b = a + 1; b < n; b++) {
      const f = M[b][a] / M[a][a];
      for (let c = a; c < n; c++) M[b][c] -= f * M[a][c];
      y[b] -= f * y[a];
    }
  }
  const x = [0, 0, 0, 0, 0, 0];
  for (let a = n - 1; a >= 0; a--) {
    let sm = y[a];
    for (let b = a + 1; b < n; b++) sm -= M[a][b] * x[b];
    x[a] = sm / M[a][a];
  }
  return x;
}

export function shiftField(
  Wp: Warped, L: number[][], N: number, P: ShiftParams, rs: { eo: number },
): OffsetField | null {
  const cell = Wp.cell, W = Wp.W;
  const GX = Wp.GX, GY = Wp.GY;
  const S = Math.round(P.sr * cell);
  const Pts: number[][] = [];

  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      if (!L[j][i]) continue;
      const cx = cell + i * cell, cy = cell + j * cell;
      let best = 0, bx = 0, by = 0;
      const E = Math.round(rs.eo * cell);
      const x0 = i === 0 ? -E : -S, x1 = i === N - 1 ? E : S;
      const y0 = j === 0 ? -E : -S, y1 = j === N - 1 ? E : S;
      for (let dy = y0; dy <= y1; dy += 2) {
        for (let dx = x0; dx <= x1; dx += 2) {
          for (let ri = 0; ri < 3; ri++) {
            const r = [17, 19, 21][ri] * (cell / 40);
            let c = 0;
            for (let t = 0; t < 48; t++) {
              let xs = Math.trunc(cx + dx + r * CT[t]);
              let ys = Math.trunc(cy + dy + r * ST[t]);
              if (xs < 0) xs = 0; else if (xs >= W) xs = W - 1;
              if (ys < 0) ys = 0; else if (ys >= W) ys = W - 1;
              const k = ys * W + xs;
              if (Math.abs(GX[k] * CT[t] + GY[k] * ST[t]) > 40) c++;
            }
            if (c > best) { best = c; bx = dx; by = dy; }
          }
        }
      }
      if (best / 48 >= P.cmin) Pts.push([i, j, bx / cell, by / cell]);
    }
  }

  if (Pts.length < P.min) return null;

  const ev = (x: number[], i: number, j: number): number => {
    const u = i / (N - 1), v = j / (N - 1);
    return x[0] + x[1] * u + x[2] * v + x[3] * u * u + x[4] * v * v + x[5] * u * v;
  };

  function fit(k: number, use: boolean[]): number[] {
    const n = 6;
    const M: number[][] = [], y: number[] = [];
    for (let a = 0; a < n; a++) { y.push(0); M.push([0, 0, 0, 0, 0, 0]); }
    Pts.forEach((p, q) => {
      if (!use[q]) return;
      const u = p[0] / (N - 1), v = p[1] / (N - 1);
      const r = [1, u, v, u * u, v * v, u * v];
      for (let a = 0; a < n; a++) {
        y[a] += r[a] * p[k];
        for (let b = 0; b < n; b++) M[a][b] += r[a] * r[b];
      }
    });
    return solve6(M, y);
  }

  let use = Pts.map(() => true);
  let ax: number[] = [], ay: number[] = [];
  for (let it = 0; it < 3; it++) {
    ax = fit(2, use);
    ay = fit(3, use);
    use = Pts.map((p) => hypot2(p[2] - ev(ax, p[0], p[1]), p[3] - ev(ay, p[0], p[1])) < P.tol);
  }

  const out: OffsetField = [];
  for (let j = 0; j < N; j++) {
    out.push([]);
    for (let i = 0; i < N; i++) {
      const ox = Math.max(-0.5, Math.min(0.5, ev(ax, i, j)));
      const oy = Math.max(-0.5, Math.min(0.5, ev(ay, i, j)));
      out[j].push([ox, oy]);
    }
  }
  return out;
}
