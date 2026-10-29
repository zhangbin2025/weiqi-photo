/**
 * 预处理：缩放、摩尔纹检测、黑帽形态学、粗定位、霍夫线段。
 * 预处理：缩放 / 摩尔纹 / 黑帽 / 粗四边形 / 线段检测。
 */

import type { CvLike, MatLike, MatWithConvert } from "./cv/types.js";
import { quadArea, orderQuad } from "./math/period.js";
import { pct } from "./math/stats.js";
import type { Pt, Seg } from "./types.js";
import type { Params } from "./params.js";

export interface SmallImage {
  /** 缩放比例（原图 → 小图） */
  s: number;
  /** 缩放后的 RGBA 图 */
  sm: MatLike;
  /** 缩放后的灰度图 */
  g: MatLike;
}

/** 缩放到最长边 1000px 并转灰度 */
export function small(cv: CvLike, img: MatLike): SmallImage {
  const s = 1000 / Math.max(img.rows, img.cols);
  const sm = new cv.Mat();
  cv.resize(
    img, sm,
    new cv.Size(Math.round(img.cols * s), Math.round(img.rows * s)),
    0, 0, cv.INTER_AREA,
  );
  const g = new cv.Mat();
  cv.cvtColor(sm, g, cv.COLOR_RGBA2GRAY);
  return { s, sm, g };
}

/** 黑帽形态学：提取比背景暗的小区域（棋子、棋盘线） */
export function blackhat(cv: CvLike, g: MatLike, k: number): MatLike {
  const o = new cv.Mat();
  const K = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(k, k));
  cv.morphologyEx(g, o, cv.MORPH_BLACKHAT, K);
  K.delete();
  return o;
}

/**
 * 摩尔纹检测。
 * 原理：转到 Lab 空间，用原图减高斯模糊后的残差衡量高频能量。
 * 屏幕翻拍的照片摩尔纹很强，会干扰霍夫直线，需要先模糊。
 *
 * 注意：这里原实现用 cv.absdiff 的等价写法（手动求 |a-b| 之和）。
 */
export function moire(cv: CvLike, img: MatLike): number {
  const s = 600 / Math.max(img.rows, img.cols);
  const sm = new cv.Mat();
  cv.resize(img, sm, new cv.Size(Math.round(img.cols * s), Math.round(img.rows * s)), 0, 0, cv.INTER_AREA);
  const rgb = new cv.Mat();
  cv.cvtColor(sm, rgb, cv.COLOR_RGBA2RGB);
  const lab = new cv.Mat();
  cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);
  const f = new cv.Mat();
  (lab as MatWithConvert).convertTo(f, cv.CV_32F);
  const g = new cv.Mat();
  cv.GaussianBlur(f, g, new cv.Size(0, 0), (2 * 600) / 1000);
  const w = f.cols, h = f.rows;
  const F = f.data32F, G = g.data32F;
  let t = 0, n = 0;
  for (let y = Math.floor(h / 4); y < (3 * h) / 4; y++) {
    for (let x = Math.floor(w / 4); x < (3 * w) / 4; x++) {
      const i = (y * w + x) * 3;
      t += Math.abs(F[i + 1] - G[i + 1]) + Math.abs(F[i + 2] - G[i + 2]);
      n++;
    }
  }
  for (const o of [sm, rgb, lab, f, g]) o.delete();
  return n ? (t / n) * (255 / 256) : 0;
}

export interface RoughQuadDbg {
  nb: number;
  gate?: boolean;
}

/**
 * 粗定位：在小图上找一个包住棋盘的四边形。
 *
 * 流程：
 *  1. 黑帽(7×7) 找暗区 → boxFilter(41) → 阈值 0.04 → 开运算(15×15)
 *  2. 连通域筛选（面积 > 2%，质心靠中心 35%）
 *  3. 凸包 → 抽稀到 40 点 → 暴力搜最大面积四边形
 *
 * useGate=true 时先做一个"够黑够大"的掩膜，处理深色桌面/深色背景。
 */
export function roughQuad(
  cv: CvLike, SM: SmallImage, useGate: boolean, P: Params, dbg: RoughQuadDbg,
): Pt[] {
  const RQ = P.rq;
  const g = SM.g, sm = SM.sm;
  const h = g.rows, w = g.cols;
  const bh = blackhat(cv, g, 7);
  const n = h * w;
  const lm = new cv.Mat(h, w, cv.CV_32F);
  const L = lm.data32F;
  const B = bh.data;
  const Px = sm.data;

  let gate: MatLike | null = null;
  if (RQ.on && useGate) {
    const l1 = new cv.Mat(h, w, cv.CV_32F);
    const L1 = l1.data32F;
    for (let i = 0; i < n; i++) {
      const V1 = Math.max(Px[4 * i], Px[4 * i + 1], Px[4 * i + 2]);
      L1[i] = B[i] > 25 && V1 > 60 ? 1 : 0;
    }
    const d1 = new cv.Mat();
    cv.boxFilter(l1, d1, -1, new cv.Size(41, 41));
    const m1 = new cv.Mat(h, w, cv.CV_8U);
    const D1 = d1.data32F, M1 = m1.data;
    for (let i = 0; i < n; i++) M1[i] = D1[i] > 0.04 ? 1 : 0;
    const K1 = cv.Mat.ones(15, 15, cv.CV_8U);
    const o1 = new cv.Mat();
    cv.morphologyEx(m1, o1, cv.MORPH_OPEN, K1);
    const lb1 = new cv.Mat(), st1 = new cv.Mat(), cn1 = new cv.Mat();
    const n1 = cv.connectedComponentsWithStats(o1, lb1, st1, cn1);
    let bk = 0;
    let ba = 0;
    const gk1: Record<number, number> = {};
    for (let k = 1; k < n1; k++) {
      const a1x = st1.intAt ? st1.intAt(k, 4) : st1.data32S[k * 5 + 4];
      if (a1x > RQ.ma * h * w && Math.abs(cn1.doubleAt(k, 0) - w / 2) < RQ.c * w && Math.abs(cn1.doubleAt(k, 1) - h / 2) < RQ.c * h) {
        ba += a1x;
        gk1[k] = 1;
        bk = 1;
      }
    }
    if (bk && ba > 0.08 * h * w) {
      const gp: number[] = [];
      const LB1 = lb1.data32S;
      for (let y1 = 0; y1 < h; y1++) {
        for (let x1 = 0; x1 < w; x1++) if (gk1[LB1[y1 * w + x1]]) gp.push(x1, y1);
      }
      const gm = cv.matFromArray(gp.length / 2, 1, cv.CV_32SC2, gp);
      const gh = new cv.Mat();
      cv.convexHull(gm, gh, false, true);
      gate = cv.Mat.zeros(h, w, cv.CV_8U);
      const gv = new cv.MatVector();
      gv.push_back(gh);
      cv.fillPoly(gate, gv, new cv.Scalar(1));
      const gk = Math.round(RQ.g * Math.sqrt(ba));
      const GK = cv.Mat.ones(gk, gk, cv.CV_8U);
      cv.dilate(gate, gate, GK);
      for (const o of [gm, gh, gv, GK]) o.delete();
    }
    for (const o of [l1, d1, m1, K1, o1, lb1, st1, cn1]) o.delete();
  }

  const GD = gate ? gate.data : null;
  for (let i = 0; i < n; i++) {
    const V = Math.max(Px[4 * i], Px[4 * i + 1], Px[4 * i + 2]);
    L[i] = (B[i] > 25 && V > 60) || (V < 50 && (!GD || GD[i])) ? 1 : 0;
  }
  if (gate) gate.delete();

  const d = new cv.Mat();
  cv.boxFilter(lm, d, -1, new cv.Size(41, 41));
  const m = new cv.Mat(h, w, cv.CV_8U);
  const D = d.data32F, Mm = m.data;
  for (let i = 0; i < n; i++) Mm[i] = D[i] > 0.04 ? 1 : 0;
  const K = cv.Mat.ones(15, 15, cv.CV_8U);
  const m2 = new cv.Mat();
  cv.morphologyEx(m, m2, cv.MORPH_OPEN, K);

  const lab = new cv.Mat(), st = new cv.Mat(), cen = new cv.Mat();
  const nl = cv.connectedComponentsWithStats(m2, lab, st, cen);
  const keep: Record<number, number> = {};
  let any = false;
  let big = 1;
  const areaOf = (k: number): number => (st.intAt ? st.intAt(k, 4) : st.data32S[k * 5 + 4]);
  for (let k = 1; k < nl; k++) {
    const ar = areaOf(k);
    if (areaOf(k) > areaOf(big)) big = k;
    if (ar > 0.02 * h * w && Math.abs(cen.doubleAt(k, 0) - w / 2) < 0.35 * w && Math.abs(cen.doubleAt(k, 1) - h / 2) < 0.35 * h) {
      keep[k] = 1;
      any = true;
    }
  }
  if (!any) keep[big] = 1;

  const Lb = lab.data32S;
  const pts: number[] = [];
  for (let y = 0; y < h; y++) {
    let x0 = -1, x1 = -1;
    for (let x = 0; x < w; x++) {
      if (keep[Lb[y * w + x]]) { if (x0 < 0) x0 = x; x1 = x; }
    }
    if (x0 >= 0) { pts.push(x0, y); if (x1 !== x0) pts.push(x1, y); }
  }
  const pm = cv.matFromArray(pts.length / 2, 1, cv.CV_32SC2, pts);
  const hull = new cv.Mat();
  cv.convexHull(pm, hull, false, true);

  const hp: Pt[] = [];
  for (let i = 0; i < hull.rows; i++) hp.push([hull.data32S[2 * i], hull.data32S[2 * i + 1]]);
  if (hp.length > 40) {
    const st2 = hp.length / 40;
    const h2: Pt[] = [];
    for (let i = 0; i < 40; i++) h2.push(hp[Math.floor(i * st2)]);
    hp.length = 0;
    hp.push(...h2);
  }

  let best: [number, number, number, number, number] | null = null;
  const nh = hp.length;
  for (let a = 0; a < nh; a++) {
    for (let b = a + 1; b < nh; b++) {
      for (let c = b + 1; c < nh; c++) {
        for (let dd = c + 1; dd < nh; dd++) {
          const ar = quadArea(hp[a], hp[b], hp[c], hp[dd]);
          if (!best || ar > best[0]) best = [ar, a, b, c, dd];
        }
      }
    }
  }
  const bb = best as [number, number, number, number, number];
  const q: Pt[] = [1, 2, 3, 4].map((k): Pt => {
    const p = hp[bb[k]];
    return [p[0] / SM.s, p[1] / SM.s];
  });

  for (const o of [bh, lm, d, m, K, m2, lab, st, cen, pm, hull]) o.delete();
  return orderQuad(q);
}

/**
 * 在粗定位区域内提取棋盘线段。
 * ROI 是粗四边形相对中心外扩 1.2 倍。
 */
export function segs(cv: CvLike, SM: SmallImage, q: Pt[]): Seg[] {
  const g = SM.g, s = SM.s;
  const bh = blackhat(cv, g, 5);
  const h = g.rows, w = g.cols, n = h * w;
  const m = new cv.Mat(h, w, cv.CV_8U);
  const M = m.data, B = bh.data, G = g.data;
  for (let i = 0; i < n; i++) M[i] = B[i] > 20 && G[i] > 60 ? 255 : 0;

  let cx = 0, cy = 0;
  for (const p of q) { cx += p[0] * s; cy += p[1] * s; }
  cx /= 4; cy /= 4;
  const qq: Pt[] = [0, 1, 3, 2].map((k): Pt => [
    Math.trunc(cx + (q[k][0] * s - cx) * 1.2),
    Math.trunc(cy + (q[k][1] * s - cy) * 1.2),
  ]);

  const roi = cv.Mat.zeros(h, w, cv.CV_8U);
  const pm = cv.matFromArray(4, 1, cv.CV_32SC2, ([] as number[]).concat(...qq));
  cv.fillConvexPoly(roi, pm, new cv.Scalar(255));
  const R = roi.data;
  for (let i = 0; i < n; i++) M[i] &= R[i];

  const ls = new cv.Mat();
  cv.HoughLinesP(m, ls, 1, Math.PI / 720, 30, 35, 4);
  const S: Seg[] = [];
  for (let i = 0; i < ls.rows; i++) {
    S.push([ls.data32S[4 * i], ls.data32S[4 * i + 1], ls.data32S[4 * i + 2], ls.data32S[4 * i + 3]]);
  }
  for (const o of [bh, m, roi, pm, ls]) o.delete();
  return S;
}

/** 从 48 点等分圆采样表（cos/sin），多处复用 */
export function circleTable(n = 48): { ct: number[]; st: number[] } {
  const ct: number[] = [], st: number[] = [];
  for (let t = 0; t < n; t++) {
    ct.push(Math.cos((2 * Math.PI * t) / n));
    st.push(Math.sin((2 * Math.PI * t) / n));
  }
  return { ct, st };
}

/** 供外部复用的百分位（避免重复 import） */
export { pct };
