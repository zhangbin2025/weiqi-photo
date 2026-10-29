/**
 * 一维周期估计 + 线段分族。
 *
 * 核心思路：把 2D 网格问题降成 1D 周期问题。
 * 把每条棋盘线沿消失点方向投影成一维坐标 u（间距均匀），
 * 然后用连续周期 d 的"DFT"扫频找最优周期。
 */

import { pct } from "./stats.js";

/** 周期估计结果：[周期, 相位, 备选周期列表] */
export interface PeriodResult {
  /** 最优周期 d */
  d: number;
  /** 相位（网格原点偏移） */
  phase: number;
  /** 次优备选（供回退重试） */
  alts: Array<[number, number]>;
}

export interface PeriodOptions {
  /** 备选峰的相对幅度阈值（相对最强峰） */
  alt?: number;
  /** 保留的备选数量 */
  na?: number;
  /** 二倍频接受阈值 */
  dbl?: number;
  /** 超过 hf*(N-1) 条线时才尝试半周期 */
  hf?: number;
  /** 半周期接受阈值 */
  dbl2?: number;
  /** 扫频采样点数 */
  samples?: number;
}

/**
 * 用加权复数求和估计一维点列的周期。
 *
 * z(d) = Σ w_i · exp(2πi·u_i/d)
 * 当 d 命中真实间距时，所有相位对齐，|z| 出现峰值。
 *
 * @param u 一维坐标（沿消失点投影后的均匀坐标）
 * @param w 权重（线段长度）
 * @param span u 的跨度，用于限定搜索范围 [span/26, span/6]
 * @param N 期望路数（可为 0 表示未知）
 */
export function period(
  u: number[],
  w: number[],
  span: number,
  N: number,
  opt: PeriodOptions = {},
): PeriodResult {
  const ALT = opt.alt ?? 0.5;
  const NA = opt.na ?? 3;
  const DBL = opt.dbl ?? 0.9;
  const HF = opt.hf ?? 1.45;
  const DBL2 = opt.dbl2 ?? 0.5;
  const SAMPLES = opt.samples ?? 1200;

  /** 复数求和：Σ w_i · exp(2πi·u_i/d)，返回 [模, d, 相位] */
  function z(d: number): [number, number, number] {
    let re = 0, im = 0;
    for (let i = 0; i < u.length; i++) {
      const a = (2 * Math.PI * u[i]) / d;
      re += w[i] * Math.cos(a);
      im += w[i] * Math.sin(a);
    }
    return [Math.sqrt(re * re + im * im), d, (Math.atan2(im, re) * d) / (2 * Math.PI)];
  }

  /** 在 [lo,hi] 上细扫 n 点取最强峰 */
  function peak(lo: number, hi: number, n: number): [number, number, number] {
    let best: [number, number, number] | null = null;
    for (let t = 0; t < n; t++) {
      const r = z(lo + ((hi - lo) * t) / (n - 1));
      if (!best || r[0] > best[0]) best = r;
    }
    return best as [number, number, number];
  }

  const all: Array<[number, number, number]> = [];
  const lo0 = span / 26;
  const hi0 = span / 6;
  for (let t = 0; t < SAMPLES; t++) all.push(z(lo0 + ((hi0 - lo0) * t) / (SAMPLES - 1)));

  let best = all[0];
  for (const r of all) if (r[0] > best[0]) best = r;
  const top = best[0];

  // 收集局部极大作为备选
  const alts: Array<[number, number, number]> = [];
  for (let t = 1; t + 1 < all.length; t++) {
    if (all[t][0] > all[t - 1][0] && all[t][0] >= all[t + 1][0] && all[t][0] >= ALT * top) {
      alts.push(all[t]);
    }
  }
  alts.sort((p, q) => q[0] - p[0]);

  // 防"每两条线当成一条"：若 2d 峰也够强，取 2d
  for (let k = 0; k < 3; k++) {
    const d2 = best[1] * 2;
    if (d2 > span / 3) break;
    const c2 = peak(d2 * 0.96, d2 * 1.04, 60);
    if (c2[0] >= DBL * top) best = c2;
    else break;
  }

  // 防漏线：若线数远超 N-1，尝试半周期
  if (N && span / best[1] > HF * (N - 1)) {
    const d3 = best[1] * 2;
    const c3 = peak(d3 * 0.96, d3 * 1.04, 60);
    if (c3[0] >= DBL2 * top) best = c3;
  }

  return {
    d: best[1],
    phase: best[2],
    alts: alts
      .filter((r) => Math.abs(r[1] - best[1]) > 0.02 * best[1])
      .slice(0, NA)
      .map((r): [number, number] => [r[1], r[2]]),
  };
}

/**
 * 把线段按方向分成横竖两族。
 * 角度乘 2 后做 2-means（5 个初值取最优），避免 π 周期性问题。
 *
 * @param segs 线段 [x0,y0,x1,y1]
 * @returns 每条线段的族标签（0 或 1）
 */
export function familySplit(segs: Array<[number, number, number, number]>): number[] {
  const z = segs.map((r) => {
    const a = Math.atan2(r[3] - r[1], r[2] - r[0]);
    return [Math.cos(2 * a), Math.sin(2 * a)] as [number, number];
  });
  let best: { in_: number; lab: number[] } | null = null;
  for (let t = 0; t < 5; t++) {
    const c0 = z[Math.floor((t * z.length) / 5)];
    let far = 0, fd = -1;
    z.forEach((p, i) => {
      const d = (p[0] - c0[0]) * (p[0] - c0[0]) + (p[1] - c0[1]) * (p[1] - c0[1]);
      if (d > fd) { fd = d; far = i; }
    });
    const C: Array<[number, number]> = [c0.slice() as [number, number], z[far].slice() as [number, number]];
    const lab: number[] = [];
    for (let it = 0; it < 50; it++) {
      const sm = [[0, 0, 0], [0, 0, 0]];
      let ch = false;
      z.forEach((p, i) => {
        const d0 = (p[0] - C[0][0]) ** 2 + (p[1] - C[0][1]) ** 2;
        const d1 = (p[0] - C[1][0]) ** 2 + (p[1] - C[1][1]) ** 2;
        const l = d1 < d0 ? 1 : 0;
        if (lab[i] !== l) { lab[i] = l; ch = true; }
        sm[l][0] += p[0]; sm[l][1] += p[1]; sm[l][2]++;
      });
      for (let k = 0; k < 2; k++) {
        if (sm[k][2]) C[k] = [sm[k][0] / sm[k][2], sm[k][1] / sm[k][2]];
      }
      if (!ch) break;
    }
    let in_ = 0;
    z.forEach((p, i) => {
      const c = C[lab[i]];
      in_ += (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2;
    });
    if (!best || in_ < best.in_) best = { in_, lab: lab.slice() };
  }
  return (best as { in_: number; lab: number[] }).lab;
}

/**
 * 把 4 个点按左上/右上/右下/左下 排序（先极角，再旋转到起点为"最小和"点）。
 * orderQuad()：极角排序后按 index [0,1,3,2] 重排。
 */
export function orderQuad(q: Array<[number, number]>): Array<[number, number]> {
  let cx = 0, cy = 0;
  for (const p of q) { cx += p[0]; cy += p[1]; }
  cx /= q.length; cy /= q.length;
  const sorted = q
    .slice()
    .sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
  let k = 0;
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i][0] + sorted[i][1] < sorted[k][0] + sorted[k][1]) k = i;
  }
  const rotated = sorted.slice(k).concat(sorted.slice(0, k));
  return [rotated[0], rotated[1], rotated[3], rotated[2]];
}

/** 多边形面积（用于最大面积四边形搜索） */
export function quadArea(a: [number, number], b: [number, number], c: [number, number], d: [number, number]): number {
  const P = [a, b, c, d];
  let s = 0;
  for (let k = 0; k < 4; k++) {
    const u = P[k], v = P[(k + 1) % 4];
    s += u[0] * v[1] - v[0] * u[1];
  }
  return Math.abs(s) / 2;
}

/** u 坐标跨度的鲁棒估计（2%~98% 分位差） */
export function spanOf(u: number[]): number {
  return Math.abs(pct(u, 98) - pct(u, 2));
}
