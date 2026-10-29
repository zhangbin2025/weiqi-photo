/**
 * 主流程编排：检测路数 → 求四角 → 判子。
 *
 * 完整链路：
 *   缩放 → 摩尔纹检测 → 粗定位 → 霍夫线段 → 消失点+周期 → 四角
 *   → 透视校正 → 特征提取 → 分类 → 补漏 → 黄色修正 → 去重/阴影剔除
 */

import type { CvLike, MatLike } from "./cv/types.js";
import type { BoardResult, DebugInfo, Quad, Stone } from "./types.js";
import type { Params } from "./params.js";
import { moire, roughQuad, small, type SmallImage, type RoughQuadDbg } from "./preprocess.js";
import { corners1, type CornerDbg } from "./geometry/corners.js";
import { warp } from "./geometry/warp.js";
import { feats, type OffsetField } from "./classify/feats.js";
import { lab, ycol, type LabDbg } from "./classify/lab.js";
import { rescue, type RescueDbg } from "./classify/rescue.js";
import { shiftField } from "./classify/shift.js";
import { dedupe } from "./classify/dedupe.js";
import { median } from "./math/stats.js";
import { makeParams } from "./params.js";

/** 检测四角（含自动路数） */
export function detectCorners(
  cv: CvLike, img: MatLike, N: number, P: Params, dbg: DebugInfo,
): { c: Quad; N: number; auto: boolean } {
  // 摩尔纹抑制：屏幕翻拍会严重干扰霍夫直线
  const mo = moire(cv, img);
  dbg.moire = mo;
  const sc = Math.max(img.cols, img.rows) / 2000;
  let work: MatLike = img;
  let blurred: MatLike | null = null;
  if (P.mo.on && mo > P.mo.t) {
    blurred = new cv.Mat();
    cv.GaussianBlur(img, blurred, new cv.Size(0, 0), P.mo.s * sc);
    work = blurred;
  }

  try {
    if (N) {
      const c = cornersFixed(cv, work, N, P, dbg);
      return { c, N, auto: false };
    }
    const A = autoN(cv, work, P, dbg);
    dbg.auto = true;
    if (blurred) { blurred.delete(); blurred = null; }
    return { c: A.c, N: A.N, auto: true };
  } finally {
    if (blurred) blurred.delete();
  }
}

/** 指定路数求四角（含 gate 回退） */
function cornersFixed(cv: CvLike, img: MatLike, N: number, P: Params, dbg: DebugInfo): Quad {
  const SM = small(cv, img);
  const rdbg: RoughQuadDbg = { nb: 0 };
  let r0: Quad | null = null;
  let w0 = -9;
  let e0: unknown = null;
  const cdbg0 = newCornerDbg();
  try {
    const q0 = roughQuad(cv, SM, false, P, rdbg);
    r0 = corners1(cv, img, N, false, SM, q0, P, cdbg0);
    w0 = cdbg0.win;
    copyDbg(cdbg0, dbg);
  } catch (e) { e0 = e; }

  if (!P.rq.on || (r0 && cdbg0.nb < 2)) {
    cleanupSM(SM);
    if (r0) return r0;
    throw e0 instanceof Error ? e0 : new Error(String(e0));
  }

  const cdbg1 = newCornerDbg();
  let r1: Quad | null = null;
  let w1 = -9;
  try {
    const q1 = roughQuad(cv, SM, true, P, rdbg);
    r1 = corners1(cv, img, N, true, SM, q1, P, cdbg1);
    w1 = cdbg1.win;
  } catch { /* ignore */ }
  dbg.rq = [w0, w1];

  cleanupSM(SM);
  if (r1 && w1 > w0 + P.rq.m) return r1;
  if (r0) return r0;
  if (r1) return r1;
  throw e0 instanceof Error ? e0 : new Error(String(e0));
}

/** 自动路数：按 19→15→13→9 试，用滑窗得分 + 外圈占比判定 */
function autoN(cv: CvLike, img: MatLike, P: Params, dbg: DebugInfo): { N: number; c: Quad; win: number } {
  let best: { N: number; c: Quad; win: number; sc: number } | null = null;
  const rw = P.rq.rw;
  const log: Array<[number, number, number | null, number | null]> = [];
  for (let pass = 0; pass < 2; pass++) {
    P.rq.rw = pass ? rw : 0;
    try {
      for (const n of P.au.ns) {
        const cdbg = newCornerDbg();
        let c: Quad | null = null;
        try { c = cornersFixed(cv, img, n, P, dbg); } catch { c = null; }
        const w = cdbg.win ?? 0;
        const rg = ringOf(cdbg, n);
        log.push([pass, n, +w.toFixed(2), +rg.toFixed(2)]);
        if (w >= P.au.w && rg <= P.au.r) {
          dbg.aulog = log;
          copyDbg(cdbg, dbg);
          return { N: n, c: c as Quad, win: w };
        }
        if (pass && (!best || w - rg > best.sc)) best = { N: n, c: c as Quad, win: w, sc: w - rg };
      }
    } finally {
      P.rq.rw = rw;
    }
  }
  if (best) { dbg.aulog = log; return { N: best.N, c: best.c, win: best.win }; }
  dbg.aulog = log;
  throw new Error("noboard");
}

/** 外圈有子说明棋盘框小了（路数猜大了） */
function ringOf(d: CornerDbg, N: number): number {
  const E = d.E, bb = d.bb;
  if (!E || !bb) return 1;
  const NN = E.length;
  const r0 = bb[1], c0 = bb[2];
  let n = 0, p = 0;
  for (let j = 0; j < NN; j++) {
    for (let i = 0; i < NN; i++) {
      const dj = j < r0 ? r0 - j : j >= r0 + N ? j - (r0 + N - 1) : 0;
      const di = i < c0 ? c0 - i : i >= c0 + N ? i - (c0 + N - 1) : 0;
      const dd = Math.max(dj, di);
      if (dd === 1) { n++; if (E[j][i] > 0) p++; }
    }
  }
  return n ? p / n : 1;
}

function newCornerDbg(): CornerDbg {
  return { win: -9, q: [], n: 0, fam: null, drop: 0, per: [], nb: 0, E: null, bb: null, tried: [], rq: null, H0: null, H: null } as CornerDbg;
}

function copyDbg(from: CornerDbg, to: DebugInfo): void {
  to.win = from.win;
  to.quad = from.q;
  to.nSegs = from.n;
  to.fam = from.fam;
  to.per = from.per;
}

function cleanupSM(SM: SmallImage): void {
  SM.sm.delete();
  SM.g.delete();
}

/**
 * 从已求得的四角识别棋盘矩阵。
 *
 * @param keepFeatures 为 true 时把每交叉点的特征矩阵挂到 dbg.features 上，
 *   供调参/训练导出使用。默认 false（不占内存）。
 */
export function recognize(
  cv: CvLike, img: MatLike, c: Quad, N: number, P: Params, dbg: DebugInfo,
  keepFeatures = false,
): Stone[] {
  const Wp = warp(cv, img, c, N, 40, P.blk.mk !== 0);
  const F0 = feats(Wp, null, P.blk, P.gl);
  if (keepFeatures) dbg.features = F0;

  const labDbg: LabDbg = { T: 0, bv0: 0, bv: 0, gap: [0, 0, 0] };
  let L = lab(F0, N, P, labDbg);
  dbg.threshold = labDbg.T;
  dbg.bv = labDbg.bv;

  // 阴影方向剔除：白子若紧邻同色且处于阴影方向，判空
  if (P.pb.on) {
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        if (L[j][i] !== 2 || F0[j][i].circ >= P.pb.c) continue;
        const dirs: Array<[number, number]> = [[1, 0], [-1, 0], [0, 1], [0, -1]];
        for (const q of dirs) {
          const i2 = i + q[0], j2 = j + q[1];
          if (i2 < 0 || j2 < 0 || i2 >= N || j2 >= N || !L[j2][i2]) continue;
          const f = F0[j2][i2];
          const tw = -(q[0] * f.ox + q[1] * f.oy);
          if (tw >= P.pb.t) { L[j][i] = 0; break; }
        }
      }
    }
  }

  // 补漏子
  if (P.rs.on) {
    const L0 = L.map((r) => r.slice());
    const rdbg: RescueDbg = { rs: [], rsdrop: [], circles: [], rrej: [], CTf: null, bv: labDbg.bv };
    L = rescue(Wp, L, N, F0, P.rs, P.wb, { bv: labDbg.bv, T: labDbg.T }, rdbg);
    if (P.rs.lnx) {
      for (let j = 0; j < N; j++) {
        for (let i = 0; i < N; i++) {
          if (L[j][i] === 2 && L0[j][i] === 0 && F0[j][i].line > P.rs.lnx) L[j][i] = 0;
        }
      }
    }
    dbg.rescue = rdbg.rs;
  }

  if (P.dd.on) L = dedupe(Wp, L, N, P.dd);

  // 视差场（默认关闭）
  if (P.par.on) {
    const off = shiftField(Wp, L, N, P.par, P.rs);
    if (off) {
      const F1 = feats(Wp, off, P.blk, P.gl);
      L = lab(F1, N, P, labDbg);
      L = ycol(L, F1, N, P.yc, labDbg.T, P.gl, dbg);
      return flatten(L, N);
    }
  }
  L = ycol(L, F0, N, P.yc, labDbg.T, P.gl, dbg);
  return flatten(L, N);
}

function flatten(L: number[][], N: number): Stone[] {
  const out: Stone[] = new Array(N * N);
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) out[j * N + i] = L[j][i] as Stone;
  return out;
}

/** 完整入口：图 → 棋盘矩阵 */
export function runPipeline(
  cv: CvLike, img: MatLike, opt: { size?: number; corners?: Quad; params?: Partial<Params>; keepFeatures?: boolean },
): BoardResult {
  const P = makeParams(opt.params);
  const dbg: DebugInfo = {
    moire: 0, win: 0, auto: false, aulog: [], quad: null,
    nSegs: 0, fam: null, per: [], threshold: 0, features: null,
  };

  let N = opt.size ?? 0;
  let c: Quad;
  if (opt.corners) {
    c = opt.corners;
    dbg.auto = false;
    if (!N) throw new Error("size required when corners given");
  } else {
    const d = detectCorners(cv, img, N, P, dbg);
    c = d.c;
    N = d.N;
    dbg.auto = d.auto;
  }

  const stones = recognize(cv, img, c, N, P, dbg, opt.keepFeatures ?? false);
  return { N, corners: c, stones, debug: dbg };
}

export { median };
