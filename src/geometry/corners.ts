/**
 * 求棋盘四角 — 整套算法最核心的部分。
 *
 * 思路（比"Hough 找完整网格"鲁棒得多）：
 *   1. 把线段按方向分横竖两族
 *   2. 每族求消失点（M = Σ n̂n̂ᵀ 的最小特征向量）
 *   3. 把线段中点沿消失点方向投影成一维均匀坐标 u
 *   4. 用连续周期的"DFT"扫频求出格距 d 和相位 ph
 *   5. 用整数格点 ↔ 实际线段端点求单应矩阵 H，迭代 refine
 *   6. 在 (N+8)² 的扩展格点上滑 N×N 窗，取分最高的位置
 *   7. lineFix 修正木纹/格线造成的偏移
 */

import type { CvLike, MatLike } from "../cv/types.js";
import {
  cross, inv3, ptH, fromCvMat, vanishingPoint,
  type Mat3, type Vec3,
} from "../math/homog.js";
import { familySplit, period, spanOf, orderQuad } from "../math/period.js";
import { pct, rnd, sum } from "../math/stats.js";
import type { Pt, Quad, Seg } from "../types.js";
import type { Params } from "../params.js";
import { feats } from "../classify/feats.js";
import { warp } from "./warp.js";
import { lineFix } from "./lineFix.js";
import { segs } from "../preprocess.js";

export interface CornerDbg {
  /** 滑窗得分 0~1 */
  win: number;
  /** 粗定位四边形 */
  q: Pt[];
  /** 线段数 */
  n: number;
  /** 分族后每族数量 */
  fam: [number, number] | null;
  /** 被方向过滤掉的线段数 */
  drop: number;
  /** 周期日志 [span, d, phase] */
  per: Array<[number, number, number]>;
  /** 粗四边形贴在边界上的角数 */
  nb: number;
  /** 扩展格点的"像不像有内容"评分 */
  E: number[][] | null;
  /** 最佳窗口位置 [score, row, col] */
  bb: [number, number, number] | null;
  /** 重试日志 */
  tried: Array<[number, number, number | null]>;
  /** 两族周期备选 */
  rq: [number, number] | null;
  /** 兜底：初始单应 */
  H0: Mat3 | null;
  /** 兜底：refine 后的单应 */
  H: Mat3 | null;
}

/** 单应矩阵求解：src/dst 均为 2D 点数组 */
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
 * 迭代 refine：把线段吸附到最近的整数格点，重新 fitLine 求交，再解 H。
 * 每轮都会让格点更贴合真实棋盘线。
 */
function refine(cv: CvLike, H: Mat3, S: Seg[], it: number): Mat3 {
  let Hc = H;
  for (let r = 0; r < it; r++) {
    const Hi = inv3(Hc);
    const V: Record<number, Array<[number, number]>> = {};
    const Hz: Record<number, Array<[number, number]>> = {};
    for (const sg of S) {
      const p: [number, number] = [sg[0], sg[1]];
      const q: [number, number] = [sg[2], sg[3]];
      const a = ptH(Hi, p[0], p[1]);
      const b = ptH(Hi, q[0], q[1]);
      for (const AD of [[0, V], [1, Hz]] as Array<[number, Record<number, Array<[number, number]>>]>) {
        const ax = AD[0], D = AD[1];
        const k1 = a[ax], k2 = b[ax];
        const k = rnd((k1 + k2) / 2);
        if (Math.abs(k1 - k) < 0.2 && Math.abs(k2 - k) < 0.2 && Math.abs(a[1 - ax] - b[1 - ax]) > 0.5) {
          (D[k] = D[k] || []).push(p, q);
        }
      }
    }
    const lines: Array<{ ax: number; k: number; l: Vec3 }> = [];
    for (const AD of [[0, V], [1, Hz]] as Array<[number, Record<number, Array<[number, number]>>]>) {
      const D = AD[1];
      for (const k of Object.keys(D)) {
        const pts = D[+k];
        if (pts.length < 2) continue;
        const pm = cv.matFromArray(pts.length, 1, cv.CV_32FC2, ([] as number[]).concat(...pts));
        const ln = new cv.Mat();
        cv.fitLine(pm, ln, cv.DIST_HUBER, 0, 0.01, 0.01);
        const f = ln.data32F;
        lines.push({
          ax: AD[0],
          k: +k,
          l: cross([f[2], f[3], 1], [f[2] + f[0], f[3] + f[1], 1]),
        });
        pm.delete();
        ln.delete();
      }
    }
    const src: Array<[number, number]> = [], dst: Array<[number, number]> = [];
    for (const A of lines) {
      if (A.ax) continue;
      for (const B of lines) {
        if (!B.ax) continue;
        const p = cross(A.l, B.l);
        if (Math.abs(p[2]) < 1e-12) continue;
        src.push([A.k, B.k]);
        dst.push([p[0] / p[2], p[1] / p[2]]);
      }
    }
    if (src.length < 8) break;
    const H2 = homog(cv, src, dst, 2.0);
    if (!H2) break;
    Hc = H2;
  }
  return Hc;
}

interface FamilyData {
  S: Seg[];
  lines: Vec3[];
  vp: Vec3;
  u: number[];
  w: number[];
  d: number;
  ph: number;
  alts: Array<[number, number]>;
}

/**
 * 求四角的主入口。
 *
 * @param gated 是否启用"深色桌面"掩膜分支
 */
export function corners1(
  cv: CvLike, img: MatLike, N: number, gated: boolean,
  SM: { s: number; g: MatLike; sm: MatLike }, rough: Pt[], P: Params, dbg: CornerDbg,
): Quad {
  const RQ = P.rq;
  const s = SM.s;
  const q = rough;
  const S = segs(cv, SM as never, q);

  if (S.length < 20) throw new Error("nolines");

  const c: [number, number] = [0, 0];
  for (const p of q) { c[0] += (p[0] * s) / 4; c[1] += (p[1] * s) / 4; }
  const lab = familySplit(S);
  const F: FamilyData[] = [];

  // 按主方向过滤掉偏离过大的线段
  (() => {
    const cs: Array<[number, number]> = [[0, 0], [0, 0]];
    S.forEach((r, i) => {
      const a = 2 * Math.atan2(r[3] - r[1], r[2] - r[0]);
      const w = Math.sqrt((r[2] - r[0]) ** 2 + (r[3] - r[1]) ** 2);
      cs[lab[i]][0] += w * Math.cos(a);
      cs[lab[i]][1] += w * Math.sin(a);
    });
    const ca = cs.map((v) => Math.atan2(v[1], v[0]));
    const S2: Seg[] = [], L2: number[] = [];
    S.forEach((r, i) => {
      const a = 2 * Math.atan2(r[3] - r[1], r[2] - r[0]);
      const d = Math.abs(Math.atan2(Math.sin(a - ca[lab[i]]), Math.cos(a - ca[lab[i]])));
      if (d < Math.PI / 6) { S2.push(r); L2.push(lab[i]); }
    });
    dbg.drop = S.length - S2.length;
    S.length = 0; S.push(...S2);
    lab.length = 0; lab.push(...L2);
  })();

  dbg.q = q;
  dbg.n = S.length;
  dbg.fam = [lab.filter((v) => !v).length, lab.filter((v) => v).length];

  for (let f = 0; f < 2; f++) {
    const Sf = S.filter((_r, i) => lab[i] === f);
    const lines = Sf.map((r): Vec3 => cross([r[0], r[1], 1], [r[2], r[3], 1]));
    F.push({
      S: Sf, lines,
      vp: vanishingPoint(lines),
      u: [], w: [], d: 0, ph: 0, alts: [],
    });
  }

  /** 把点沿消失点方向投影成一维均匀坐标 */
  function ucoord(f: number, pts: Array<[number, number]>): number[] {
    const vf = F[f].vp, vo = F[1 - f].vp;
    const ref = cross([c[0], c[1], 1], vo);
    const nr = Math.sqrt(ref[0] ** 2 + ref[1] ** 2) || 1;
    const d: [number, number] = [ref[0] / nr, ref[1] / nr];
    const t: [number, number] = [-d[1], d[0]];
    const tp = pts.map((p) => {
      const l = cross([p[0], p[1], 1], vf);
      const x = cross(l, ref);
      return ((x[0] / x[2] - c[0]) * t[0] + (x[1] / x[2] - c[1]) * t[1]);
    });
    if (Math.abs(vo[2]) > 1e-6 * Math.sqrt(vo[0] ** 2 + vo[1] ** 2)) {
      const tv = ((vo[0] / vo[2] - c[0]) * t[0] + (vo[1] / vo[2] - c[1]) * t[1]);
      return tp.map((v) => 1 / (v - tv));
    }
    return tp;
  }

  for (let f = 0; f < 2; f++) {
    const Ff = F[f];
    const mid = Ff.S.map((r): [number, number] => [(r[0] + r[2]) / 2, (r[1] + r[3]) / 2]);
    Ff.u = ucoord(f, mid);
    Ff.w = Ff.S.map((r) => Math.sqrt((r[2] - r[0]) ** 2 + (r[3] - r[1]) ** 2));
    const span = spanOf(Ff.u);
    const pd = period(Ff.u, Ff.w, span, N, { alt: RQ.alt, na: RQ.na });
    Ff.d = pd.d; Ff.ph = pd.phase; Ff.alts = pd.alts;
    (dbg.per = dbg.per || []).push([span, pd.d, pd.phase]);
  }

  function solve(): Quad {
    const R_: number[][] = [];
    for (let f = 0; f < 2; f++) {
      const o = 1 - f;
      const e: Array<[number, number]> = F[f].S.map((r): [number, number] => [r[0], r[1]])
        .concat(F[f].S.map((r): [number, number] => [r[2], r[3]]));
      R_[o] = ucoord(o, e).map((v) => (v - F[o].ph) / F[o].d);
    }

    const idx: Array<{ k: number[]; sel: boolean[]; lines: Vec3[] }> = [];
    for (let f = 0; f < 2; f++) {
      const Ff = F[f];
      const ke = R_[f];
      const wm = sum(Ff.w) / Ff.w.length;
      const k = Ff.u.map((v) => rnd((v - Ff.ph) / Ff.d));
      const ok = Ff.u.map((v, i) => Math.abs((v - Ff.ph) / Ff.d - k[i]) < 0.2);
      const lo = pct(ke, 1), hi = pct(ke, 99);
      const e0 = rnd((lo + hi - (N - 1)) / 2);
      let best: [number, number] | null = null;
      for (let k0 = e0 - 3; k0 <= e0 + 3; k0++) {
        let a = 0, b = 0, out = 0, edge = 0;
        for (let i = 0; i < k.length; i++) {
          if (!ok[i]) continue;
          if (k[i] >= k0 && k[i] < k0 + N) a += Ff.w[i];
          else b += Ff.w[i];
        }
        for (const v of ke) {
          if (v < k0 - 0.3 || v > k0 + N - 1 + 0.3) out++;
          if (Math.abs(v - k0) < 0.15 || Math.abs(v - (k0 + N - 1)) < 0.15) edge++;
        }
        const sc = a / wm - 1.5 * (b / wm) - 0.5 * out + 0.5 * edge;
        if (!best || sc > best[0]) best = [sc, k0];
      }
      const k0 = (best as [number, number])[1];
      idx.push({
        k: k.map((v) => v - k0),
        sel: k.map((v, i) => ok[i] && v >= k0 && v < k0 + N),
        lines: Ff.lines,
      });
    }

    const src: Array<[number, number]> = [], dst: Array<[number, number]> = [];
    const A = idx[0], B = idx[1];
    for (let i = 0; i < A.k.length; i++) {
      if (!A.sel[i]) continue;
      for (let j = 0; j < B.k.length; j++) {
        if (!B.sel[j]) continue;
        const p = cross(A.lines[i], B.lines[j]);
        if (Math.abs(p[2]) < 1e-12) continue;
        src.push([A.k[i], B.k[j]]);
        dst.push([p[0] / p[2], p[1] / p[2]]);
      }
    }
    const H0 = homog(cv, src, dst, 3.0);
    if (!H0) throw new Error("nohomog");
    dbg.H0 = H0;
    const H = refine(cv, H0, S, 3);
    dbg.H = H;

    // 滑窗：在扩展格点上找真正的棋盘位置
    const M = 4;
    const NN = N + 2 * M;
    const lat = (a: number, b: number): [number, number] => {
      const p = ptH(H, a, b);
      return [p[0] / s, p[1] / s];
    };
    const cw: Quad = [
      lat(-M, -M), lat(N - 1 + M, -M), lat(-M, N - 1 + M), lat(N - 1 + M, N - 1 + M),
    ];
    const Wp = warp(cv, img, cw, NN, 40, false);
    const Fh = feats(Wp, null, P.blk, P.gl);
    const E: number[][] = [];
    for (let j = 0; j < NN; j++) {
      E.push([]);
      for (let i = 0; i < NN; i++) {
        const pp = lat(i - M, j - M);
        const ins = pp[0] >= 0 && pp[0] < img.cols && pp[1] >= 0 && pp[1] < img.rows;
        const x = Fh[j][i];
        const ev = (((P.blk.on ? x.vq < P.blk.t : x.v < 100) && x.circ >= P.rq.ec) || x.line > 30 || x.circ > 0.64) && ins;
        E[j].push(ins ? (ev ? 1 : -1) : -0.2);
      }
    }

    let bb: [number, number, number] | null = null;
    for (let r0 = 0; r0 <= NN - N; r0++) {
      for (let c0 = 0; c0 <= NN - N; c0++) {
        let sc = 0;
        for (let j = r0; j < r0 + N; j++) for (let i = c0; i < c0 + N; i++) sc += E[j][i];
        if (!bb || sc > bb[0]) bb = [sc, r0, c0];
      }
    }
    const bbr = bb as [number, number, number];
    const a0 = bbr[2] - M, b0 = bbr[1] - M;
    dbg.win = bbr[0] / (N * N);
    dbg.E = E;
    dbg.bb = bbr;

    // 格线偏移修正
    let cx = [0, 0, 0, 0], cy = [0, 0, 0, 0];
    if (P.lf.on) {
      const lf = lineFix(cv, Wp, Fh, bbr[2], bbr[1], N, P.lf);
      if (lf) { cx = lf.cx; cy = lf.cy; }
    }
    return orderQuad([
      lat(a0 + cx[0], b0 + cy[0]),
      lat(a0 + N - 1 + cx[1], b0 + cy[1]),
      lat(a0 + cx[2], b0 + N - 1 + cy[2]),
      lat(a0 + N - 1 + cx[3], b0 + N - 1 + cy[3]),
    ]) as Quad;
  }

  const r0 = solve();
  const D0 = { ...dbg };

  // 得分不够好时，用备选周期组合重试
  if (RQ.rw && D0.win < RQ.rw) {
    let best = r0;
    let bw = D0.win;
    const pr: Array<[number, number]> = [[F[0].d, F[0].ph], [F[1].d, F[1].ph]];
    const c0 = [pr[0]].concat(F[0].alts || []);
    const c1 = [pr[1]].concat(F[1].alts || []);
    const combos: Array<[number, number]> = [];
    c0.forEach((_a, ia) => c1.forEach((_b, ib) => { if (ia || ib) combos.push([ia, ib]); }));
    combos.sort((p, q) => p[0] + p[1] - (q[0] + q[1]));

    const tried: Array<[number, number, number | null]> = [];
    for (let ci = 0; ci < combos.length && bw < RQ.rs; ci++) {
      const ia = combos[ci][0], ib = combos[ci][1];
      F[0].d = c0[ia][0]; F[0].ph = c0[ia][1];
      F[1].d = c1[ib][0]; F[1].ph = c1[ib][1];
      dbg.per = D0.per; dbg.q = D0.q; dbg.n = D0.n; dbg.fam = D0.fam; dbg.drop = D0.drop; dbg.nb = D0.nb;
      let r: Quad | null = null;
      try { r = solve(); } catch { /* ignore */ }
      tried.push([ia, ib, r ? Math.round(dbg.win * 100) / 100 : null]);
      if (r && dbg.win > bw + RQ.rm) { best = r; bw = dbg.win; }
    }
    dbg.tried = tried;
    return best;
  }
  return r0;
}


