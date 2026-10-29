/**
 * 分类：把每个交叉点的特征判成 黑/白/空。
 * 自适应阈值 + 黑白判子 + 黄色偏差修正。
 */

import type { CellFeature } from "../types.js";
import { median, pct } from "../math/stats.js";

export interface LabDbg {
  /** 自适应阈值 */
  T: number;
  /** 全盘亮度中位数（初始） */
  bv0: number;
  /** 非黑点的亮度 30% 分位 */
  bv: number;
  /** 最大间隙信息 */
  gap: [number, number, number];
}

/**
 * 自适应阈值：对全盘 vq 排序，找最大间隙。
 * 用来应对不同曝光/不同盘面材质。
 */
export function adaptT(F: CellFeature[][], blk: {
  t: number; ad: boolean; gap: number; amax: number;
}): number {
  const a: number[] = [];
  for (const r of F) for (const x of r) a.push(x.vq);
  a.sort((p, q) => p - q);
  const hi = a[Math.floor(a.length * 0.75)];
  let g = 0;
  let m = blk.t;
  for (let k = 0; k + 1 < a.length; k++) {
    if (a[k] < blk.t * 0.8 || a[k] > blk.amax * hi) continue;
    const d = a[k + 1] - a[k];
    if (d > g) { g = d; m = (a[k] + a[k + 1]) / 2; }
  }
  return blk.ad && g >= blk.gap && m > blk.t ? m - blk.t : 0;
}

export interface ClassifyParams {
  blk: {
    on: boolean; t: number; sure: number; ln: number; cc: number;
    ad: boolean; gap: number; amax: number; oz: boolean;
  };
  gl: {
    on: boolean; wc: number; ws: number; wc2: number; ws2: number;
    a: number; b: number; nd: number; ns: number; s: number;
  };
  wb: {
    on: boolean; hi: number; d: number; cc: number;
    c2: number; r2: number; ln: number; lcc: number; lv: number; sh: number;
  };
  lr: { w: [number, number, number]; b: number };
}

/**
 * 主分类：特征 → 黑/白/空矩阵。
 *
 * 判定逻辑：
 *  1. 明显暗 → 黑
 *  2. 否则用 3 权重逻辑回归判白：z = b + 1.641·circ·10 − 1.114·arm/10 + 0.988·sd/10
 *  3. 黄棋修正（亮但低饱和 → 白；环形暗比例高 → 黑）
 */
export function lab(F: CellFeature[][], N: number, P: ClassifyParams, dbg: LabDbg): number[][] {
  const L: number[][] = [];
  const dt = adaptT(F, P.blk);
  const T = P.blk.t + dt;
  const SU = P.blk.sure + dt;
  dbg.T = T;
  dbg.gap = [Math.round(dt), Math.round(T), 0];

  const nb0: number[] = [];
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) nb0.push(F[j][i].v);
  nb0.sort((p, q) => p - q);
  dbg.bv0 = nb0[Math.floor(nb0.length * 0.5)];

  for (let j = 0; j < N; j++) {
    L.push([]);
    for (let i = 0; i < N; i++) {
      const x = F[j][i];
      const dark = P.blk.on
        ? (x.vq < SU || (x.vq < T && (x.line < P.blk.ln || x.circ >= P.blk.cc)))
        : x.v < 100;
      if (dark && !(P.blk.oz && x.vq < 1 && x.v > T)) {
        // 明显暗：默认黑，但亮且低饱和的（黄棋/玻璃子）改判白
        if (P.gl.on && ((x.vc >= P.gl.wc && x.sat < P.gl.ws) || (x.vc >= P.gl.wc2 && x.sat < P.gl.ws2))) {
          L[j].push(2);
          continue;
        }
        L[j].push(1);
        continue;
      }
      // 逻辑回归判白
      const z = P.lr.b + P.lr.w[0] * x.circ * 10 + P.lr.w[1] * (x.arm / 10) + P.lr.w[2] * (x.sd / 10);
      let wv = z > 0 &&
        !(P.wb.ln && x.line > P.wb.ln && x.circ < P.wb.lcc && x.v < dbg.bv0 + P.wb.lv && x.vc - x.v < P.wb.sh)
        ? 2 : 0;
      if (wv === 2 && P.gl.on && x.rv && x.rv.length) {
        let dk = 0;
        for (let q = 0; q < x.rv.length; q++) if (x.rv[q] < T) dk++;
        dk /= x.rv.length;
        if (dk >= P.gl.b || (dk >= P.gl.a && x.sat >= P.gl.s) || (x.nsd >= P.gl.nd && x.sat >= P.gl.ns)) wv = 1;
      }
      L[j].push(wv);
    }
  }

  // 四角若是白子但圆形度低 → 清空（多半是盘外高光）
  const corners: Array<[number, number]> = [[0, 0], [0, N - 1], [N - 1, 0], [N - 1, N - 1]];
  for (const q of corners) {
    if (L[q[0]][q[1]] === 2 && F[q[0]][q[1]].circ < P.wb.cc) L[q[0]][q[1]] = 0;
  }

  const nb: number[] = [];
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) if (L[j][i] !== 1) nb.push(F[j][i].v);
  nb.sort((p, q) => p - q);
  const BV = nb[Math.floor(nb.length * 0.3)];
  dbg.bv = BV;

  if (P.wb.on && BV >= P.wb.hi) {
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        if (L[j][i] === 2 && F[j][i].v < BV - P.wb.d) L[j][i] = 0;
      }
    }
  }
  if (P.wb.c2) {
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        if (L[j][i] === 2 && F[j][i].v < BV * P.wb.r2 && F[j][i].circ < P.wb.c2) L[j][i] = 0;
      }
    }
  }
  return L;
}

export interface YcParams {
  on: boolean; n: number; min: number; add: number; del: number;
  c: number; ct: number; rel: number; hn: number;
}

/**
 * 黄色偏差修正。
 * 木盘 + 暖光下，白子会偏黄、黑子会边界发亮，这里用黄蓝度中位数归一化后统一修正。
 */
export function ycol(L: number[][], F: CellFeature[][], N: number, P: YcParams, T: number, gl: {
  on: boolean; a: number; b: number;
}, dbg: Record<string, unknown>): number[][] {
  if (!P.on) return L;
  const ys: number[] = [];
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      if (!L[j][i] && F[j][i].yb0 != null) ys.push(F[j][i].yb0);
    }
  }
  if (ys.length < P.n) return L;
  const wy = median(ys);
  dbg.wy = wy;
  if (wy < P.min) return L;

  const vw: number[] = [], vh: number[] = [];
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      if (!L[j][i]) vw.push(F[j][i].v);
      else if (L[j][i] === 2) vh.push(F[j][i].v);
    }
  }
  const WV = vw.length ? median(vw) : 0;
  const HV = vh.length >= P.hn ? median(vh) : WV;
  const CTv = HV - WV;
  dbg.yct = CTv;

  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const f = F[j][i];
      const m = (wy - f.yb0) / Math.max(wy, 15);
      if (!L[j][i] && m >= P.add && f.v >= T && f.circ >= P.c &&
          (CTv < P.ct || f.v - WV >= P.rel * CTv)) {
        let dk = 0;
        if (f.rv && f.rv.length) {
          for (let q = 0; q < f.rv.length; q++) if (f.rv[q] < T) dk++;
          dk /= f.rv.length;
        }
        L[j][i] = dk >= gl.a ? 1 : 2;
      } else if (L[j][i] === 2 && m < P.del) {
        L[j][i] = 0;
      }
    }
  }
  return L;
}

export { pct };
