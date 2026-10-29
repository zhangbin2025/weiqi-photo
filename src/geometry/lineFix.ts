/**
 * 棋盘线偏移修正。
 *
 * 问题：木纹、盘面印刷线本身是暗的，会让"格点"跟真实交点错位。
 * 做法：沿每条棋盘线做剖面，找到暗线峰的位置，
 *      再用加权最小二乘拟合出整数格点的实际像素偏移。
 */

import type { CvLike } from "../cv/types.js";
import { inv3, ptH, fromCvMat, type Mat3 } from "../math/homog.js";
import { median } from "../math/stats.js";
import type { CellFeature } from "../types.js";
import type { LfParams } from "../params.js";
import type { Warped } from "./warp.js";

export interface LineFixResult {
  /** 四角的 x 偏移（单位：格） */
  cx: number[];
  /** 四角的 y 偏移（单位：格） */
  cy: number[];
  /** 'lin' = 线性拟合；'hom' = 单应拟合 */
  mode: string;
}

function homog(cv: CvLike, src: Array<[number, number]>, dst: Array<[number, number]>, th: number): Mat3 | null {
  const n = src.length;
  if (n < 4) return null;
  const a: number[] = [], b: number[] = [];
  for (let i = 0; i < n; i++) { a.push(src[i][0], src[i][1]); b.push(dst[i][0], dst[i][1]); }
  const A = cv.matFromArray(n, 1, cv.CV_32FC2, a);
  const B = cv.matFromArray(n, 1, cv.CV_32FC2, b);
  const msk = new cv.Mat();
  const Hm = cv.findHomography(A, B, cv.RANSAC, th, msk);
  const H = Hm.rows === 3 ? fromCvMat(Hm.data64F) : null;
  for (const o of [A, B, msk, Hm]) o.delete();
  return H;
}

/**
 * @param c0/r0 棋盘在扩展格点中的起始列/行
 * @returns null 表示修正量过小或不可靠，不修正
 */
export function lineFix(
  _cv: CvLike, Wp: Warped, Fh: CellFeature[][], c0: number, r0: number, N: number, LF: LfParams,
): LineFixResult | null {  // eslint-disable-line @typescript-eslint/no-unused-vars
  const cell = Wp.cell, W = Wp.W;
  const G = Wp.G, V = Wp.V;
  const R = Math.round(LF.r * cell);
  const vs: number[] = [];
  for (let j = r0; j < r0 + N; j++) for (let i = c0; i < c0 + N; i++) vs.push(Fh[j][i].v);
  vs.sort((a, b) => a - b);
  const bv = vs[Math.floor(vs.length * 0.5)];
  const lo = bv * LF.lo;
  const hi = Math.min(bv * LF.hi, 250);
  const ok = (v: number): boolean => v > lo && v < hi;

  /** 沿一个轴测量每条线的偏移 */
  function measure(ax: number): { p: number; q: number; m: number[][] } | null {
    const out: number[][] = [];
    for (let k = 0; k < N; k++) {
      const p0 = cell * (1 + (ax ? r0 : c0) + k);
      const s0 = cell * (1 + (ax ? c0 : r0));
      const s1 = cell * (1 + (ax ? c0 : r0) + N - 1);
      const prof: number[] = [];
      const cnt: number[] = [];
      for (let d = -R; d <= R; d++) {
        let acc = 0, n = 0;
        for (let t = s0; t <= s1; t += 2) {
          const x = ax ? t : p0 + d;
          const y = ax ? p0 + d : t;
          const xm = ax ? x : x - 4;
          const xp = ax ? x : x + 4;
          const ym = ax ? y - 4 : y;
          const yp = ax ? y + 4 : y;
          if (xm < 0 || ym < 0 || xp >= W || yp >= W) continue;
          const g = G[y * W + x];
          const g1 = G[ym * W + xm];
          const g2 = G[yp * W + xp];
          if (!ok(V[y * W + x]) || !ok(V[ym * W + xm]) || !ok(V[yp * W + xp])) continue;
          const dk = (g1 + g2) / 2 - g;
          if (dk > 0) acc += dk;
          n++;
        }
        prof.push(n ? acc / n : 0);
        cnt.push(n);
      }
      let bi = 0;
      for (let d = 1; d < prof.length; d++) if (prof[d] > prof[bi]) bi = d;
      if (cnt[bi] < (LF.n * cell) / 2 || prof[bi] < LF.t || bi < 2 || bi > prof.length - 3 || prof[bi] < LF.pk * median(prof)) continue;
      let off = bi - R;
      if (bi > 0 && bi < prof.length - 1) {
        const a = prof[bi - 1], b = prof[bi], c = prof[bi + 1];
        const den = a - 2 * b + c;
        if (den < 0) off += (0.5 * (a - c)) / den;
      }
      out.push([k, off, prof[bi]]);
    }
    if (out.length < LF.min) return null;

    // 鲁棒直线拟合（迭代剔除离群点）
    const sl: number[] = [];
    for (let a = 0; a < out.length; a++) {
      for (let b = a + 1; b < out.length; b++) sl.push((out[b][1] - out[a][1]) / (out[b][0] - out[a][0]));
    }
    let q = median(sl);
    let p = median(out.map((o) => o[1] - q * o[0]));
    let use = out.map((o) => Math.abs(o[1] - (p + q * o[0])) < LF.tol * cell);
    for (let it = 0; it < 3; it++) {
      let sw = 0, sk = 0, so = 0, skk = 0, sko = 0;
      out.forEach((o, m) => {
        if (!use[m]) return;
        const w = o[2];
        sw += w; sk += w * o[0]; so += w * o[1]; skk += w * o[0] * o[0]; sko += w * o[0] * o[1];
      });
      const den2 = sw * skk - sk * sk;
      if (Math.abs(den2) < 1e-9) { q = 0; p = sw ? so / sw : 0; }
      else { q = (sw * sko - sk * so) / den2; p = (so - q * sk) / sw; }
      use = out.map((o) => Math.abs(o[1] - (p + q * o[0])) < LF.tol * cell);
    }
    const nu = use.filter((v) => v).length;
    if (nu < LF.min) return null;
    return { p: p / cell, q: q / cell, m: out };
  }

  const fx = measure(0);
  const fy = measure(1);
  if (!fx && !fy) return null;

  let cx = [0, 0, 0, 0], cy = [0, 0, 0, 0];
  let mode = "lin";
  if (fx) cx = [fx.p, fx.p + fx.q * (N - 1), fx.p, fx.p + fx.q * (N - 1)];
  if (fy) cy = [fy.p, fy.p, fy.p + fy.q * (N - 1), fy.p + fy.q * (N - 1)];

  // 可选：用单应拟合二阶畸变
  if (LF.hom && fx && fy && fx.m.length >= LF.hn && fy.m.length >= LF.hn) {
    const src: Array<[number, number]> = [], dst: Array<[number, number]> = [];
    for (const a of fx.m) {
      for (const b of fy.m) {
        const x0 = cell * (1 + c0 + a[0]);
        const y0 = cell * (1 + r0 + b[0]);
        src.push([x0, y0]);
        dst.push([x0 + a[1], y0 + b[1]]);
      }
    }
    const Hc = homog(_cv, src, dst, LF.ht);
    if (Hc) {
      let ok2 = true;
      const cx2: number[] = [], cy2: number[] = [];
      for (const q of [[0, 0], [N - 1, 0], [0, N - 1], [N - 1, N - 1]]) {
        const x0 = cell * (1 + c0 + q[0]);
        const y0 = cell * (1 + r0 + q[1]);
        const pp = ptH(Hc, x0, y0);
        const dx = (pp[0] - x0) / cell;
        const dy = (pp[1] - y0) / cell;
        if (!(Math.abs(dx) < LF.max && Math.abs(dy) < LF.max)) ok2 = false;
        cx2.push(dx); cy2.push(dy);
      }
      if (ok2) { cx = cx2; cy = cy2; mode = "hom"; }
    }
  }

  let mx = 0;
  for (let m = 0; m < 4; m++) {
    mx = Math.max(mx, Math.abs(cx[m]), Math.abs(cy[m]));
    if (Math.abs(cx[m]) > LF.max || Math.abs(cy[m]) > LF.max) return null;
  }
  if (mx < LF.minc) return null;
  return { cx, cy, mode };
}

export { inv3 };
