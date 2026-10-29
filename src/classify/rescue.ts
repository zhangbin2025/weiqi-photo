/**
 * 补漏子 rescue — 流水线里最大的一块。
 *
 * 作用：lab() 用固定格点采样，会漏掉两类子：
 *   1. 偏离格点较远的子（透视残余 / 拍摄角度）
 *   2. 边缘处只有部分可见的子
 * 这里对空格做穷举搜索，用圆形度找回漏检子，再定色、去重、分配。
 */

import type { Warped } from "../geometry/warp.js";
import { median } from "../math/stats.js";
import type { CellFeature } from "../types.js";
import type { RsParams, WbParams } from "../params.js";

const CT: number[] = [];
const ST: number[] = [];
for (let t = 0; t < 48; t++) {
  CT.push(Math.cos((2 * Math.PI * t) / 48));
  ST.push(Math.sin((2 * Math.PI * t) / 48));
}

export interface RescueDbg {
  /** 补上的子 [i,j,color,...] */
  rs: number[][];
  /** 被去重丢弃的 */
  rsdrop: number[][];
  /** 圆形检测结果 */
  circles: number[][];
  /** 被拒绝的候选及原因 */
  rrej: number[][];
  /** 子中心修正量 */
  CTf: unknown;
  /** 基线亮度 */
  bv: number;
}

/** 独立圆形检测命中的候选子 */
interface Circle {
  x: number; y: number; q: number; col: number; f: number[];
  /** 可分配到的空格（[距离, i, j]） */
  opt: number[][];
}

export interface RescueCtx {
  /** 当前亮度中位数基线（空格亮度） */
  bv: number;
  /** 黑子阈值 */
  T: number;
}

/**
 * 沿棋盘线检测暗线强度。
 * 用来判断候选位置是否落在棋盘线上（线上不是子）。
 */
function lineIn(
  Wp: Warped, gx: number, gy: number, sx: number, sy: number, R: number, mn: boolean,
): number {
  const W = Wp.W, G = Wp.G, cell = Wp.cell;
  let tot = 0, n = 0;
  const g = (x: number, y: number): number | null => {
    x = Math.round(x); y = Math.round(y);
    if (x < 0 || y < 0 || x >= W || y >= W) return null;
    return G[y * W + x];
  };
  for (let m = -1; m <= 1; m++) {
    const lx = gx + m * cell, ly = gy + m * cell;
    if (Math.abs(lx - sx) < R) {
      const hh = Math.sqrt(R * R - (lx - sx) * (lx - sx));
      let best = 0;
      for (let o = -2; o <= 2; o++) {
        let c = 0, cn = 0;
        for (let t = -hh; t <= hh; t += 1) {
          const a = g(lx + o, sy + t);
          const l2 = g(lx + o - 4, sy + t);
          const r2 = g(lx + o + 4, sy + t);
          if (a == null || l2 == null || r2 == null) continue;
          c += (mn ? Math.min(l2, r2) : (l2 + r2) / 2) - a;
          cn++;
        }
        if (cn && c / cn > best) best = c / cn;
      }
      tot += best; n++;
    }
    if (Math.abs(ly - sy) < R) {
      const hh2 = Math.sqrt(R * R - (ly - sy) * (ly - sy));
      let best = 0;
      for (let o = -2; o <= 2; o++) {
        let c = 0, cn = 0;
        for (let t = -hh2; t <= hh2; t += 1) {
          const a = g(sx + t, ly + o);
          const u2 = g(sx + t, ly + o - 4);
          const d2 = g(sx + t, ly + o + 4);
          if (a == null || u2 == null || d2 == null) continue;
          c += (mn ? Math.min(u2, d2) : (u2 + d2) / 2) - a;
          cn++;
        }
        if (cn && c / cn > best) best = c / cn;
      }
      tot += best; n++;
    }
  }
  return n ? tot / n : 0;
}

export function rescue(
  Wp: Warped, L: number[][], N: number, F: CellFeature[][] | null,
  P: RsParams, wb: WbParams, ctx: RescueCtx, dbg: RescueDbg,
): number[][] {
  const cell = Wp.cell, W = Wp.W;
  const V = Wp.V, GX = Wp.GX, GY = Wp.GY;
  const S0 = Math.round(P.r * cell);
  const S = Math.round(P.r2 * cell);
  const E = Math.round(P.eo * cell);

  /** 圆形度：在 3 个半径上取最优 */
  function circ(x: number, y: number): number {
    let bc = 0;
    for (let ri = 0; ri < 3; ri++) {
      const r = [17, 19, 21][ri] * (cell / 40);
      let c = 0;
      for (let t = 0; t < 48; t++) {
        let xs = Math.trunc(x + r * CT[t]);
        let ys = Math.trunc(y + r * ST[t]);
        if (xs < 0) xs = 0; else if (xs >= W) xs = W - 1;
        if (ys < 0) ys = 0; else if (ys >= W) ys = W - 1;
        const k = ys * W + xs;
        if (Math.abs(GX[k] * CT[t] + GY[k] * ST[t]) > 40) c++;
      }
      if (c > bc) bc = c;
    }
    return bc / 48;
  }

  /** 小圆盘亮度中位数 */
  function inner(x: number, y: number): number {
    const vs: number[] = [];
    const h = Math.round(0.3 * cell);
    for (let yy = -h; yy <= h; yy++) {
      for (let xx = -h; xx <= h; xx++) {
        if (xx * xx + yy * yy > h * h) continue;
        const y4 = y + yy, x4 = x + xx;
        if (y4 < 0 || x4 < 0 || y4 >= W || x4 >= W) continue;
        vs.push(V[y4 * W + x4]);
      }
    }
    return vs.length ? median(vs) : 0;
  }

  const WT = Math.max(ctx.T, Math.min(P.pw * ctx.bv, 250));

  /** 迭代求质心 */
  function centroid(x: number, y: number, col: number): [number, number] {
    for (let it = 0; it < 3; it++) {
      let sx = 0, sy = 0, n = 0;
      const h = Math.round(0.5 * cell);
      const X = Math.round(x), Y = Math.round(y);
      for (let yy = -h; yy <= h; yy++) {
        for (let xx = -h; xx <= h; xx++) {
          if (xx * xx + yy * yy > h * h) continue;
          const y4 = Y + yy, x4 = X + xx;
          if (y4 < 0 || x4 < 0 || y4 >= W || x4 >= W) continue;
          const v = V[y4 * W + x4];
          if (col === 1 ? v < ctx.T : v >= WT) { sx += xx; sy += yy; n++; }
        }
      }
      if (!n) break;
      x = X + sx / n;
      y = Y + sy / n;
    }
    return [x, y];
  }

  const edgeDist = (i: number, j: number): number => Math.min(i, N - 1 - i, j, N - 1 - j);

  function purity(x: number, y: number): [number, number] {
    let nb = 0, nw = 0, n = 0;
    const h = Math.round(P.pr * cell);
    for (let yy = -h; yy <= h; yy++) {
      for (let xx = -h; xx <= h; xx++) {
        if (xx * xx + yy * yy > h * h) continue;
        const y4 = y + yy, x4 = x + xx;
        if (y4 < 0 || x4 < 0 || y4 >= W || x4 >= W) continue;
        const v = V[y4 * W + x4];
        n++;
        if (v < ctx.T) nb++;
        else if (v >= WT) nw++;
      }
    }
    return n ? [nb / n, nw / n] : [0, 0];
  }

  /** 精细圆形检测：在 R 范围内搜最优圆心 */
  function refineC(x0: number, y0: number, R: number, step: number): [number, number, number] {
    let rb = -1, rx = x0, ry = y0;
    for (let ey = -R; ey <= R; ey += step) {
      for (let ex = -R; ex <= R; ex += step) {
        let bc = 0;
        for (let ri = 0; ri < 5; ri++) {
          const r = (17 + ri) * (cell / 40);
          let c = 0;
          for (let t = 0; t < 48; t++) {
            let xs = Math.trunc(x0 + ex + r * CT[t]);
            let ys = Math.trunc(y0 + ey + r * ST[t]);
            if (xs < 0) xs = 0; else if (xs >= W) xs = W - 1;
            if (ys < 0) ys = 0; else if (ys >= W) ys = W - 1;
            const k = ys * W + xs;
            if (Math.abs(GX[k] * CT[t] + GY[k] * ST[t]) > 40) c++;
          }
          if (c > bc) bc = c;
        }
        if (bc > rb) { rb = bc; rx = x0 + ex; ry = y0 + ey; }
      }
    }
    return [rx, ry, rb / 48];
  }

  const stoneAtLast: number[] = [0, 0, 0];
  /** 判断 (x,y) 处是黑/白/无子 */
  function stoneAt(x: number, y: number): number {
    if (lineIn(Wp, cell * Math.round(x / cell), cell * Math.round(y / cell), x, y, P.ilr * cell, true) > P.lc) return 0;
    const iv: number[] = [], ov: number[] = [];
    const ih = Math.round(0.3 * cell);
    const o0 = 0.6 * cell, o1 = 0.8 * cell;
    const oR = Math.round(o1);
    const wlo = 0.6 * ctx.bv, whi = 1.15 * ctx.bv;
    for (let qy = -oR; qy <= oR; qy++) {
      for (let qx = -oR; qx <= oR; qx++) {
        const dd = Math.sqrt(qx * qx + qy * qy);
        const y5 = Math.round(y) + qy, x5 = Math.round(x) + qx;
        if (y5 < 0 || x5 < 0 || y5 >= W || x5 >= W) continue;
        const v5 = V[y5 * W + x5];
        if (dd <= ih) iv.push(v5);
        else if (dd >= o0 && dd <= o1 && v5 >= wlo && v5 <= whi) ov.push(v5);
      }
    }
    const im = median(iv);
    const om = ov.length >= 10 ? median(ov) : ctx.bv;

    let tx = 0, tn = 0;
    const th = Math.round(0.25 * cell);
    for (let qy = -th; qy <= th; qy++) {
      for (let qx = -th; qx <= th; qx++) {
        if (qx * qx + qy * qy > th * th) continue;
        const y6 = Math.round(y) + qy, x6 = Math.round(x) + qx;
        if (y6 < 0 || x6 < 0 || y6 >= W || x6 >= W) continue;
        const k = y6 * W + x6;
        tx += Math.sqrt(GX[k] * GX[k] + GY[k] * GY[k]);
        tn++;
      }
    }
    tx = tn ? tx / tn : 0;
    stoneAtLast[0] = Math.round(im);
    stoneAtLast[1] = Math.round(om);
    stoneAtLast[2] = Math.round(tx);
    if (im < ctx.T) return tx <= P.tx ? 1 : 0;
    return im >= Math.min(P.wb * ctx.bv, 250) && im - om >= P.ct && tx <= P.tx ? 2 : 0;
  }

  // 子中心修正表（可选）
  const CTf: Array<Array<[number, number] | null>> = [];
  if (P.rc) {
    for (let j = 0; j < N; j++) {
      CTf.push([]);
      for (let i = 0; i < N; i++) {
        if (!L[j][i]) { CTf[j].push(null); continue; }
        const c0x = cell + i * cell, c0y = cell + j * cell;
        const o0x = F ? F[j][i].ox : 0, o0y = F ? F[j][i].oy : 0;
        if (P.rc === 2) {
          let bs = -1, bx = 0, by = 0;
          for (let dy = -S; dy <= S; dy += 2) {
            for (let dx = -S; dx <= S; dx += 2) {
              if (dx * dx + dy * dy > S * S) continue;
              const s0 = circ(c0x + dx, c0y + dy) - (P.rcp * Math.sqrt(dx * dx + dy * dy)) / cell;
              if (s0 > bs) { bs = s0; bx = dx; by = dy; }
            }
          }
          const cc = centroid(c0x + o0x, c0y + o0y, L[j][i]);
          const gx0 = cc[0] - c0x, gy0 = cc[1] - c0y;
          const rt = P.rct * cell;
          const pu = P.rcpu ? purity(Math.round(c0x + bx), Math.round(c0y + by)) : [0, 0];
          const pure = bs >= P.rcm && (L[j][i] === 1 ? pu[0] : pu[1]) >= P.rcpu;
          if (bs >= P.rch || pure || (bs >= P.rcm && Math.sqrt((bx - gx0) ** 2 + (by - gy0) ** 2) <= rt)) {
            CTf[j].push([bx, by]);
          } else if (Math.sqrt((o0x - gx0) ** 2 + (o0y - gy0) ** 2) <= rt) {
            CTf[j].push([gx0, gy0]);
          } else {
            CTf[j].push([o0x, o0y]);
          }
        } else {
          const cc = centroid(c0x + o0x, c0y + o0y, L[j][i]);
          let ex = cc[0] - c0x, ey = cc[1] - c0y;
          if (Math.sqrt(ex * ex + ey * ey) > S) { ex = o0x; ey = o0y; }
          CTf[j].push([ex, ey]);
        }
      }
    }
  }
  dbg.CTf = CTf;

  const add: number[][] = [];
  const circles: Circle[] = [];

  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      if (L[j][i]) continue;
      const cx = cell + i * cell, cy = cell + j * cell;
      const cand: number[][] = [];

      const outw = (x: number, y: number): boolean =>
        (i === 0 && x < -S0) || (i === N - 1 && x > S0) || (j === 0 && y < -S0) || (j === N - 1 && y > S0);
      const x0 = i === 0 ? -E : -S, x1 = i === N - 1 ? E : S;
      const y0 = j === 0 ? -E : -S, y1 = j === N - 1 ? E : S;

      for (let dy = y0; dy <= y1; dy += 2) {
        for (let dx = x0; dx <= x1; dx += 2) {
          const ow = outw(dx, dy);
          if (!ow && dx * dx + dy * dy > S * S) continue;
          if (ow && Math.abs(dx) > S0 && Math.abs(dy) > S0) continue;
          const sc = circ(cx + dx, cy + dy);
          if (sc >= (ow || dx * dx + dy * dy <= S0 * S0 ? P.c : P.c2)) cand.push([sc, dx, dy]);
        }
      }

      // 相邻空格之间的候选（子落在两格中间）
      const mt = Math.round(P.mrm * cell);
      if (P.mid) {
        const M2 = Math.round(P.mw * cell);
        const dirs: Array<[number, number]> = [[1, 0], [-1, 0], [0, 1], [0, -1]];
        for (const d of dirs) {
          const i3 = i + d[0], j3 = j + d[1];
          if (i3 < 0 || j3 < 0 || i3 >= N || j3 >= N || L[j3][i3]) continue;
          for (let my = -M2; my <= M2; my += 2) {
            for (let mx = -M2; mx <= M2; mx += 2) {
              const px = (d[0] * cell) / 2 + mx, py = (d[1] * cell) / 2 + my;
              if ((d[0] * px > cell / 2 + mt) || (d[1] * py > cell / 2 + mt) || px * px + py * py <= S * S) continue;
              const sc = circ(cx + px, cy + py);
              if (sc >= P.c2) cand.push([sc, px, py]);
            }
          }
        }
      }

      cand.sort((p, q) => q[0] - p[0]);
      if (i === P.di && j === P.dj) dbg.rrej.push([i, j, -1, cand.length, 0]);

      // 独立圆形检测（mt 模式）
      if (P.mt) {
        let nrec = 0;
        for (let cj = 0; cj < cand.length && nrec < P.mk; cj++) {
          const cq = cand[cj];
          if (cq[0] < P.sc || Math.abs(cq[1]) > S0 || Math.abs(cq[2]) > S0) continue;
          let dup = false;
          for (const cc of circles) {
            if (Math.sqrt((cc.x - (cx + cq[1])) ** 2 + (cc.y - (cy + cq[2])) ** 2) < P.dup * cell) { dup = true; break; }
          }
          if (dup) continue;
          const rc = refineC(cx + cq[1], cy + cq[2], Math.round(P.rf * cell), 1);
          nrec++;
          if (rc[2] < P.sc) continue;
          let dup2 = false;
          for (const cc of circles) {
            if (Math.sqrt((cc.x - rc[0]) ** 2 + (cc.y - rc[1]) ** 2) < P.dup * cell) { dup2 = true; break; }
          }
          if (dup2) continue;
          const col = stoneAt(rc[0], rc[1]);
          if (col) circles.push({ x: rc[0], y: rc[1], q: rc[2], col, f: stoneAtLast.slice(), opt: [] });
        }
      }

      const corner = (i === 0 || i === N - 1) && (j === 0 || j === N - 1);
      let got: number[] | null = null;
      for (let ci = 0; ci < cand.length && ci < P.k && !got; ci++) {
        const bestv = cand[ci][0];
        let bx = cand[ci][1], by = cand[ci][2];
        let ok = true;
        const ow2 = outw(bx, by);
        if (ci > 0 && !ow2) continue;
        const far = !ow2 && bx * bx + by * by > S0 * S0;

        if (far && P.rf) {
          const mv0 = inner(cx + bx, cy + by);
          if (mv0 >= ctx.T && mv0 < WT) { dbg.rrej.push([i, j, 0, bx, by, Math.round(mv0)]); continue; }
          const rc = centroid(cx + bx, cy + by, mv0 < ctx.T ? 1 : 2);
          bx = Math.round(rc[0]) - cx;
          by = Math.round(rc[1]) - cy;
          if (bx * bx + by * by > (S + mt) * (S + mt)) { dbg.rrej.push([i, j, 1, Math.round(bx), Math.round(by)]); continue; }
        }

        // 邻子冲突检测
        for (let b = -1; b <= 1 && ok; b++) {
          for (let a = -1; a <= 1; a++) {
            if (!a && !b) continue;
            const i2 = i + a, j2 = j + b;
            if (i2 < 0 || j2 < 0 || i2 >= N || j2 >= N || !L[j2][i2]) continue;
            let nx = a * cell + (P.nf && F ? F[j2][i2].ox : 0);
            let ny = b * cell + (P.nf && F ? F[j2][i2].oy : 0);
            if (P.rc && !P.nb2) { nx = a * cell + (CTf[j2][i2] as [number, number])[0]; ny = b * cell + (CTf[j2][i2] as [number, number])[1]; }
            if (Math.sqrt((nx - bx) ** 2 + (ny - by) ** 2) < P.d * cell) { ok = false; break; }
            if (P.rc && P.nb2 && Math.sqrt((a * cell + (CTf[j2][i2] as [number, number])[0] - bx) ** 2 + (b * cell + (CTf[j2][i2] as [number, number])[1] - by) ** 2) < P.d * cell) { ok = false; break; }
          }
        }
        if (!ok) { dbg.rrej.push([i, j, 2, bx, by, Math.round(bestv * 100) / 100]); continue; }
        if (corner && bestv < wb.cc) continue;

        if (ow2) {
          if (lineIn(Wp, cx, cy, cx + bx, cy + by, P.or * cell, false) > P.olc) continue;
          if (inner(cx + bx, cy + by) < Math.min(P.ob * ctx.bv, 250)) continue;
        }

        let lc = 0;
        if (ow2 || bx * bx + by * by <= S0 * S0) {
          lc = lineIn(Wp, cx, cy, cx + bx, cy + by, P.lr * cell, false);
          if (lc > P.lc) { dbg.rrej.push([i, j, 3, bx, by, Math.round(bestv * 100) / 100, Math.round(lc)]); continue; }
        } else {
          lc = lineIn(Wp, cx, cy, cx + bx, cy + by, P.lr * cell, false);
          const pu = purity(cx + bx, cy + by);
          if (lc > P.lc && Math.max(pu[0], pu[1]) < P.pb) {
            dbg.rrej.push([i, j, 4, bx, by, Math.round(bestv * 100) / 100, Math.round(lc), Math.round(pu[0] * 100), Math.round(pu[1] * 100)]);
            continue;
          }
        }
        got = [bx, by, lc, bestv, ow2 ? lineIn(Wp, cx, cy, cx + bx, cy + by, P.or * cell, false) : -1];
      }
      if (!got) continue;
      const mvs = inner(cx + got[0], cy + got[1]);
      if (mvs >= ctx.T && mvs < Math.min(P.wb * ctx.bv, 250)) {
        dbg.rrej.push([i, j, 5, got[0], got[1], Math.round(mvs)]);
        continue;
      }
      add.push([i, j, mvs < ctx.T ? 1 : 2, got[2], got[3], got[4], got[0] / cell, got[1] / cell, mvs, ctx.bv]);
    }
  }

  function sideNb(A: number[], B: number[]): number {
    const dx = B[0] - A[0], dy = B[1] - A[1];
    let n = 0;
    for (const q of [[dy, dx], [-dy, -dx]]) {
      const i5 = A[0] + q[0], j5 = A[1] + q[1];
      if (i5 >= 0 && j5 >= 0 && i5 < N && j5 < N && L[j5][i5]) n++;
    }
    return n;
  }

  // 去重：两个候选太近时保留更合理的一个
  const drop: Record<number, number> = {};
  for (let p = 0; p < add.length; p++) {
    for (let q = p + 1; q < add.length; q++) {
      const A = add[p], B = add[q];
      if (Math.sqrt((A[0] + A[6] - (B[0] + B[6])) ** 2 + (A[1] + A[7] - (B[1] + B[7])) ** 2) < P.d) {
        const cc = centroid(cell + (A[0] + A[6]) * cell, cell + (A[1] + A[7]) * cell, A[2]);
        const da = Math.sqrt((cc[0] - cell - A[0] * cell) ** 2 + (cc[1] - cell - A[1] * cell) ** 2);
        const db = Math.sqrt((cc[0] - cell - B[0] * cell) ** 2 + (cc[1] - cell - B[1] * cell) ** 2);
        let keepA: boolean;
        if (Math.abs(da - db) > P.tie * cell) keepA = da < db;
        else {
          const na = P.nbt ? sideNb(A, B) : 0;
          const nb = P.nbt ? sideNb(B, A) : 0;
          const ea = edgeDist(A[0], A[1]);
          const eb = edgeDist(B[0], B[1]);
          keepA = ea !== eb ? ea > eb : na !== nb ? na > nb : da <= db;
        }
        drop[keepA ? q : p] = 1;
        dbg.rrej.push([A[0], A[1], B[0], B[1], Math.round(da), Math.round(db)]);
      }
    }
  }
  dbg.rsdrop = add.filter((_x, k) => drop[k]);
  const kept = add.filter((_x, k) => !drop[k]);
  for (const p of kept) L[p[1]][p[0]] = p[2];
  dbg.rs = kept;

  // 把独立检测到的圆分配到最近空格
  if (P.mt && circles.length) {
    const occ: number[][] = [];
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        if (!L[j][i]) continue;
        let px = cell + i * cell + (F ? F[j][i].ox : 0);
        let py = cell + j * cell + (F ? F[j][i].oy : 0);
        const ro = refineC(px, py, Math.round(P.nr * cell), 2);
        if (Math.sqrt((ro[0] - cell - i * cell) ** 2 + (ro[1] - cell - j * cell) ** 2) <= P.mr * cell) {
          px = ro[0]; py = ro[1];
        }
        occ.push([px, py, i, j, L[j][i], ro[2]]);
      }
    }
    dbg.circles = circles.map((c) => [c.col, Math.round(((c.x - cell) / cell) * 100) / 100, Math.round(((c.y - cell) / cell) * 100) / 100, Math.round(c.q * 100) / 100, ...c.f]);

    const left = circles.filter((c) => {
      for (const o of occ) {
        if (Math.sqrt((o[0] - c.x) ** 2 + (o[1] - c.y) ** 2) < P.same * cell) return false;
        if (o[5] < P.sc && Math.sqrt((cell + o[2] * cell - c.x) ** 2 + (cell + o[3] * cell - c.y) ** 2) <= P.mr * cell) return false;
      }
      return true;
    });

    const eye = (ii: number, jj: number): boolean => {
      let e = 0;
      for (const q of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const a = ii + q[0], b = jj + q[1];
        if (a < 0 || b < 0 || a >= N || b >= N) return false;
        if (!L[b][a]) e++;
      }
      return e === 0;
    };

    for (const c of left) {
      c.opt = [];
      const i0 = Math.round((c.x - cell) / cell), j0 = Math.round((c.y - cell) / cell);
      for (let dj = -1; dj <= 1; dj++) {
        for (let di = -1; di <= 1; di++) {
          const ii = i0 + di, jj = j0 + dj;
          if (ii < 0 || jj < 0 || ii >= N || jj >= N || L[jj][ii] || eye(ii, jj)) continue;
          const d = Math.sqrt((c.x - cell - ii * cell) ** 2 + (c.y - cell - jj * cell) ** 2);
          if (d <= P.mr * cell) c.opt.push([d, ii, jj]);
        }
      }
      c.opt.sort((p, q) => p[0] - q[0]);
    }
    const left2 = left.filter((c) => c.opt.length);
    left2.sort((p, q) => p.opt[0][0] - q.opt[0][0]);

    const owner: Record<string, number> = {};
    const tryAssign = (ci: number, seen: Record<string, number>): boolean => {
      const c = left2[ci];
      for (const o of c.opt) {
        const key = `${o[1]},${o[2]}`;
        if (seen[key]) continue;
        seen[key] = 1;
        if (owner[key] === undefined || tryAssign(owner[key], seen)) {
          owner[key] = ci;
          return true;
        }
      }
      return false;
    };
    for (let ci = 0; ci < left2.length; ci++) tryAssign(ci, {});

    for (const key of Object.keys(owner)) {
      const c = left2[owner[key]];
      const ij = key.split(",");
      L[+ij[1]][+ij[0]] = c.col;
    }
  }

  return L;
}
