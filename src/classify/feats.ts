/**
 * 每个交叉点的特征提取。
 * 逐格特征计算。
 *
 * 最重要的特征是 circ（圆形度）：在 ±10 偏移内搜 3 个半径，
 * 统计圆周 48 点上「梯度方向与半径方向一致」的比例。
 * 这是判白子的核心依据，比单纯亮度阈值鲁棒得多。
 */

import type { Warped } from "../geometry/warp.js";
import { median, pct, sum } from "../math/stats.js";
import type { CellFeature } from "../types.js";
import type { BlkParams, GlParams } from "../params.js";

/** 48 点等分圆采样表 */
const CT: number[] = [];
const ST: number[] = [];
for (let t = 0; t < 48; t++) {
  CT.push(Math.cos((2 * Math.PI * t) / 48));
  ST.push(Math.sin((2 * Math.PI * t) / 48));
}

export type OffsetField = Array<Array<[number, number]>>;

export function feats(Wp: Warped, off: OffsetField | null, blk: BlkParams, gl: GlParams): CellFeature[][] {
  const N = Wp.N, cell = Wp.cell, W = Wp.W;
  const V = Wp.V, G = Wp.G, MK = Wp.MK;
  const h = cell >> 1;
  const r2 = (0.3 * cell) * (0.3 * cell);
  const a0 = Math.floor(0.15 * cell);
  const a1 = Math.floor(0.45 * cell);
  const b = 5;

  /** 沿某轴做剖面均值：axis=1 逐行（返回长度 y1-y0），axis=0 逐列 */
  function prof(x0: number, x1: number, y0: number, y1: number, axis: number): number[] | null {
    x0 = Math.max(0, x0); y0 = Math.max(0, y0);
    x1 = Math.min(W, x1); y1 = Math.min(W, y1);
    if (x1 <= x0 || y1 <= y0) return null;
    const p: number[] = [];
    if (axis === 1) {
      for (let y = y0; y < y1; y++) {
        let s = 0;
        for (let x = x0; x < x1; x++) s += G[y * W + x];
        p.push(s / (x1 - x0));
      }
    } else {
      for (let x = x0; x < x1; x++) {
        let s = 0;
        for (let y = y0; y < y1; y++) s += G[y * W + x];
        p.push(s / (y1 - y0));
      }
    }
    return p;
  }

  const out: CellFeature[][] = [];
  for (let j = 0; j < N; j++) {
    const row: CellFeature[] = [];
    for (let i = 0; i < N; i++) {
      const cx0 = cell + i * cell;
      const cy0 = cell + j * cell;
      const of = off ? off[j][i] : [0, 0];
      const cxReal = cx0 + of[0] * cell;
      const cyReal = cy0 + of[1] * cell;
      const cx = Math.round(cxReal);
      const cy = Math.round(cyReal);

      // 中心圆盘采样
      const vs: number[] = [], gs: number[] = [];
      for (let dy = -h; dy <= h; dy++) {
        for (let dx = -h; dx <= h; dx++) {
          if (dx * dx + dy * dy > r2) continue;
          const y = cy + dy, x = cx + dx;
          if (y < 0 || x < 0 || y >= W || x >= W) continue;
          if (MK && !MK[y * W + x]) continue;
          vs.push(V[y * W + x]);
          gs.push(G[y * W + x]);
        }
      }
      if (!vs.length) { vs.push(255); gs.push(255); }

      // 环形采样（抗中心高光）
      const rs: number[] = [];
      for (let dy = -h; dy <= h; dy++) {
        for (let dx = -h; dx <= h; dx++) {
          const dd = dx * dx + dy * dy;
          if (dd < blk.r0 * blk.r0 * cell * cell || dd > blk.r1 * blk.r1 * cell * cell) continue;
          const y = cy + dy, x = cx + dx;
          if (y < 0 || x < 0 || y >= W || x >= W) continue;
          if (MK && !MK[y * W + x]) continue;
          rs.push(V[y * W + x]);
        }
      }
      if (!rs.length) rs.push(...vs);

      const vq = pct(blk.ring ? rs : vs, blk.q);
      const mv = median(vs);
      const mg = sum(gs) / gs.length;
      let sd = 0;
      for (const v of gs) sd += (v - mg) * (v - mg);
      sd = Math.sqrt(sd / gs.length);

      // 四个方向「手臂」凹陷度：棋线造成的十字暗线
      const arms: number[] = [], rawv: number[] = [];
      const dirs: Array<[number, number]> = [[1, 0], [-1, 0], [0, 1], [0, -1]];
      for (const d of dirs) {
        let p: number[] | null;
        if (d[0]) {
          p = prof(d[0] > 0 ? cx0 + a0 : cx0 - a1, d[0] > 0 ? cx0 + a1 : cx0 - a0, cy0 - b, cy0 + b + 1, 1);
        } else {
          p = prof(cx0 - b, cx0 + b + 1, d[1] > 0 ? cy0 + a0 : cy0 - a1, d[1] > 0 ? cy0 + a1 : cy0 - a0, 0);
        }
        if (!p) { arms.push(0); rawv.push(0); continue; }
        const md = median(p);
        const mn = Math.min.apply(null, p);
        arms.push(((md - mn) / Math.max(md, 1)) * 100);
        rawv.push(md - mn);
      }
      arms.sort((x, y) => x - y);
      rawv.sort((x, y) => x - y);

      // 圆形度：搜最优子中心偏移
      let best = 0, obx = 0, oby = 0;
      const GX = Wp.GX, GY = Wp.GY;
      for (let dy = -10; dy <= 10; dy += 2) {
        for (let dx = -10; dx <= 10; dx += 2) {
          for (let ri = 0; ri < 3; ri++) {
            const r = [17, 19, 21][ri] * (cell / 40);
            let c = 0;
            for (let t = 0; t < 48; t++) {
              let xs = Math.trunc(cxReal + (dx * cell) / 40 + r * CT[t]);
              let ys = Math.trunc(cyReal + (dy * cell) / 40 + r * ST[t]);
              if (xs < 0) xs = 0; else if (xs >= W) xs = W - 1;
              if (ys < 0) ys = 0; else if (ys >= W) ys = W - 1;
              const k = ys * W + xs;
              if (Math.abs(GX[k] * CT[t] + GY[k] * ST[t]) > 40) c++;
            }
            if (c / 48 > best) { best = c / 48; obx = (dx * cell) / 40; oby = (dy * cell) / 40; }
          }
        }
      }

      // 暗像素质心偏移
      let sx = 0, sy = 0, sn = 0;
      const R5 = Math.round(0.55 * cell);
      for (let dy = -R5; dy <= R5; dy++) {
        for (let dx = -R5; dx <= R5; dx++) {
          if (dx * dx + dy * dy > R5 * R5) continue;
          const y = cy0 + dy, x = cx0 + dx;
          if (y < 0 || x < 0 || y >= W || x >= W) continue;
          if (V[y * W + x] < blk.t) { sx += dx; sy += dy; sn++; }
        }
      }
      const dc = sn ? Math.sqrt((sx / sn) ** 2 + (sy / sn) ** 2) / cell : 0;

      // 子中心处亮度
      const vcs: number[] = [];
      const hc = Math.round(0.3 * cell);
      for (let dy = -hc; dy <= hc; dy++) {
        for (let dx = -hc; dx <= hc; dx++) {
          if (dx * dx + dy * dy > hc * hc) continue;
          const x = Math.round(cxReal + obx) + dx, y = Math.round(cyReal + oby) + dy;
          if (x < 0 || y < 0 || x >= W || y >= W) continue;
          vcs.push(V[y * W + x]);
        }
      }

      // 环形采样（判「外圈黑中间亮」的玻璃黄子）
      const rv: number[] = [];
      for (const rr of [0.34, 0.38, 0.42]) {
        for (let t = 0; t < 48; t++) {
          const xs2 = Math.round(cxReal + obx + rr * cell * CT[t]);
          const ys2 = Math.round(cyReal + oby + rr * cell * ST[t]);
          if (xs2 < 0 || ys2 < 0 || xs2 >= W || ys2 >= W) continue;
          rv.push(V[ys2 * W + xs2]);
        }
      }

      // 高光饱和占比
      let sat = 0, sn2 = 0;
      for (let dy = -hc; dy <= hc; dy++) {
        for (let dx = -hc; dx <= hc; dx++) {
          if (dx * dx + dy * dy > hc * hc) continue;
          const x5 = Math.round(cxReal + obx) + dx, y5 = Math.round(cyReal + oby) + dy;
          if (x5 < 0 || y5 < 0 || x5 >= W || y5 >= W) continue;
          sn2++;
          if (V[y5 * W + x5] >= gl.sv) sat++;
        }
      }

      // 黄蓝度（大范围）
      const ybs: number[] = [];
      for (let dy = -hc; dy <= hc; dy++) {
        for (let dx = -hc; dx <= hc; dx++) {
          if (dx * dx + dy * dy > hc * hc) continue;
          const x7 = Math.round(cxReal + obx) + dx, y7 = Math.round(cyReal + oby) + dy;
          if (x7 < 0 || y7 < 0 || x7 >= W || y7 >= W) continue;
          if (MK && !MK[y7 * W + x7]) continue;
          ybs.push(Wp.YB[y7 * W + x7]);
        }
      }

      // 环形暗像素比例
      let nsd = 0, nsn = 0;
      const r42 = Math.round(0.42 * cell);
      for (let dy = -r42; dy <= r42; dy++) {
        for (let dx = -r42; dx <= r42; dx++) {
          if (dx * dx + dy * dy > r42 * r42) continue;
          const x9 = Math.round(cxReal + obx) + dx, y9 = Math.round(cyReal + oby) + dy;
          if (x9 < 0 || y9 < 0 || x9 >= W || y9 >= W) continue;
          if (MK && !MK[y9 * W + x9]) continue;
          const v9 = V[y9 * W + x9];
          if (v9 >= gl.sv - 8) continue;
          nsn++;
          if (v9 < blk.t) nsd++;
        }
      }

      // 黄蓝度（核心区）
      const yb0: number[] = [];
      const h25 = Math.round(0.25 * cell);
      for (let dy = -h25; dy <= h25; dy++) {
        for (let dx = -h25; dx <= h25; dx++) {
          if (dx * dx + dy * dy > h25 * h25) continue;
          const y8 = cy + dy, x8 = cx + dx;
          if (y8 < 0 || x8 < 0 || y8 >= W || x8 >= W) continue;
          if (MK && !MK[y8 * W + x8]) continue;
          yb0.push(Wp.YB[y8 * W + x8]);
        }
      }

      row.push({
        ox: obx, oy: oby, dc,
        v: mv, vq,
        arm: arms[2], line: rawv[2],
        sd, circ: best,
        vc: vcs.length ? median(vcs) : mv,
        rv,
        sat: sn2 ? sat / sn2 : 0,
        nsd: nsn ? nsd / nsn : 0,
        nsn,
        yb: ybs.length ? median(ybs) : 0,
        yb0: yb0.length ? median(yb0) : 0,
      });
    }
    out.push(row);
  }
  return out;
}
