/**
 * 去重。
 * 两个相邻格点若检测到同一个圆（距离过近），保留更合理的一个。
 * 默认关闭。
 */

import type { Warped } from "../geometry/warp.js";

const CT: number[] = [];
const ST: number[] = [];
for (let t = 0; t < 48; t++) {
  CT.push(Math.cos((2 * Math.PI * t) / 48));
  ST.push(Math.sin((2 * Math.PI * t) / 48));
}

export interface DedupeParams {
  r: number; st: number; d: number; col: boolean;
}

export function dedupe(Wp: Warped, L: number[][], N: number, P: DedupeParams): number[][] {
  const cell = Wp.cell, W = Wp.W;
  const GX = Wp.GX, GY = Wp.GY;
  const S = Math.round(P.r * cell);
  const C: Array<Array<[number, number, number] | null>> = [];

  for (let j = 0; j < N; j++) {
    C.push([]);
    for (let i = 0; i < N; i++) {
      if (!L[j][i]) { C[j].push(null); continue; }
      const cx = cell + i * cell, cy = cell + j * cell;
      let best = 0, bx = 0, by = 0;
      for (let dy = -S; dy <= S; dy += P.st) {
        for (let dx = -S; dx <= S; dx += P.st) {
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
      C[j].push([i * cell + bx, j * cell + by, best / 48]);
    }
  }

  const drop: Array<[number, number, number, number]> = [];
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const a = C[j][i];
      if (!a) continue;
      for (const q of [[1, 0], [0, 1]]) {
        const i2 = i + q[0], j2 = j + q[1];
        if (i2 >= N || j2 >= N) continue;
        const b = C[j2][i2];
        if (!b) continue;
        if (Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) >= P.d * cell) continue;
        const m: [number, number] = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
        const da = Math.sqrt((m[0] - i * cell) ** 2 + (m[1] - j * cell) ** 2);
        const db = Math.sqrt((m[0] - i2 * cell) ** 2 + (m[1] - j2 * cell) ** 2);
        drop.push(da > db ? [i, j, i2, j2] : [i2, j2, i, j]);
      }
    }
  }

  for (const p of drop) {
    if (L[p[1]][p[0]] && L[p[3]][p[2]]) {
      if (L[p[1]][p[0]] === 1 && L[p[3]][p[2]] === 2 && P.col) L[p[3]][p[2]] = 1;
      L[p[1]][p[0]] = 0;
    }
  }
  return L;
}
