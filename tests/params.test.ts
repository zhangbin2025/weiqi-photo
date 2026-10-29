/**
 * 参数与纯数学层单测 — 不依赖 opencv，可快速验证移植正确性。
 */

import { describe, it, expect } from "vitest";
import { pct, median, rnd, sum } from "../src/math/stats.js";
import { cross, inv3, ptH, eigMin3, vanishingPoint } from "../src/math/homog.js";
import type { Mat3, Vec3 } from "../src/math/homog.js";
import { period, familySplit, orderQuad, quadArea, spanOf } from "../src/math/period.js";
import { DEFAULT_PARAMS, makeParams } from "../src/params.js";
import { stonesToText, textToStones } from "../src/index.js";

describe("stats", () => {
  it("pct 边界与插值", () => {
    expect(pct([1, 2, 3, 4, 5], 0)).toBe(1);
    expect(pct([1, 2, 3, 4, 5], 100)).toBe(5);
    expect(pct([1, 2, 3, 4, 5], 50)).toBe(3);
    expect(pct([1, 2, 3, 4], 50)).toBeCloseTo(2.5, 6);
    expect(pct([], 50)).toBe(0);
    expect(pct([7], 50)).toBe(7);
  });

  it("median / sum", () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(sum([1, 2, 3])).toBe(6);
    expect(sum([])).toBe(0);
  });

  it("rnd 采用 banker's rounding", () => {
    expect(rnd(2.5)).toBe(2);
    expect(rnd(3.5)).toBe(4);
    expect(rnd(-2.5)).toBe(-2);
    expect(rnd(2.4)).toBe(2);
    expect(rnd(2.6)).toBe(3);
  });

  it("pct 不修改原数组（内部复制）", () => {
    const a = [3, 1, 2];
    pct(a, 50);
    expect(a).toEqual([3, 1, 2]);
  });
});

describe("homog", () => {
  it("cross 叉积", () => {
    expect(cross([1, 0, 0], [0, 1, 0])).toEqual([0, 0, 1]);
    expect(cross([1, 2, 3], [4, 5, 6])).toEqual([-3, 6, -3]);
  });

  it("inv3 求逆后乘回单位阵", () => {
    const m: Mat3 = [[4, 7, 2], [3, 6, 1], [2, 5, 1]];
    const inv = inv3(m);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const s = m[i][0] * inv[0][j] + m[i][1] * inv[1][j] + m[i][2] * inv[2][j];
        expect(s).toBeCloseTo(i === j ? 1 : 0, 6);
      }
    }
  });

  it("ptH 齐次变换", () => {
    const I: Mat3 = [[1, 0, 3], [0, 1, 4], [0, 0, 1]];
    expect(ptH(I, 0, 0)).toEqual([3, 4]);
  });

  it("eigMin3 求最小特征向量", () => {
    const M: Mat3 = [[5, 0, 0], [0, 3, 0], [0, 0, 1]];
    const v = eigMin3(M);
    const n = Math.hypot(v[0], v[1], v[2]);
    expect(Math.abs(v[0] / n)).toBeLessThan(1e-6);
    expect(Math.abs(v[1] / n)).toBeLessThan(1e-6);
    expect(Math.abs(Math.abs(v[2] / n) - 1)).toBeLessThan(1e-6);
  });

  it("vanishingPoint: 竖直线族收敛到 [0,1,0] 方向的无穷远点", () => {
    // 竖直线 x = c 的齐次式是 [1, 0, -c]。
    // 它们的公共无穷远点 V 满足 l·V = 0，即 V[0] = 0 → V = [0, 1, 0]。
    const lines: Vec3[] = [[1, 0, 0], [1, 0, -10], [1, 0, -20]];
    const v = vanishingPoint(lines);
    const n = Math.hypot(v[0], v[1], v[2]);
    expect(Math.abs(v[0] / n)).toBeLessThan(1e-3);   // x 分量为 0
    expect(Math.abs(v[1] / n)).toBeCloseTo(1, 3);    // 主导分量是 y
    // 验证：每条线与消失点的点积为 0（确实在直线上）
    for (const l of lines) {
      expect(l[0] * v[0] + l[1] * v[1] + l[2] * v[2]).toBeCloseTo(0, 6);
    }
  });

  it("vanishingPoint: 水平线族", () => {
    // 水平线 y = c 的齐次式 [0, 1, -c] → 消失点 [1, 0, 0]
    const lines: Vec3[] = [[0, 1, 0], [0, 1, -10], [0, 1, -20]];
    const v = vanishingPoint(lines);
    const n = Math.hypot(v[0], v[1], v[2]);
    expect(Math.abs(v[0] / n)).toBeCloseTo(1, 3);
    for (const l of lines) {
      expect(l[0] * v[0] + l[1] * v[1] + l[2] * v[2]).toBeCloseTo(0, 6);
    }
  });
});

describe("period", () => {
  // 注意：搜索范围是 [span/26, span/6]，所以要让真值落在这个区间内。
  // 19 条线、间距 40 → span≈720，范围 [27.7, 120]，真值 40 在区间内 ✓

  it("19 条均匀线能求出正确周期（真实场景）", () => {
    const u: number[] = [];
    for (let i = 0; i < 19; i++) u.push(i * 40);
    const w = u.map(() => 1);
    const r = period(u, w, spanOf(u), 19);
    expect(Math.abs(r.d - 40) / 40).toBeLessThan(0.02);
  });

  it("缺线时仍鲁棒", () => {
    const u: number[] = [];
    for (let i = 0; i < 19; i++) if (i % 7 !== 3) u.push(i * 40);
    const w = u.map(() => 1);
    const r = period(u, w, spanOf(u), 19);
    expect(Math.abs(r.d - 40) / 40).toBeLessThan(0.05);
  });

  it("9 路小棋盘同样能求出周期", () => {
    const u: number[] = [];
    for (let i = 0; i < 9; i++) u.push(i * 40);
    const w = u.map(() => 1);
    const r = period(u, w, spanOf(u), 9);
    expect(Math.abs(r.d - 40) / 40).toBeLessThan(0.05);
  });

  it("权重越大（线越长）的线越主导结果", () => {
    const u: number[] = [];
    const w: number[] = [];
    for (let i = 0; i < 19; i++) { u.push(i * 40); w.push(i < 3 ? 50 : 1); }
    const r = period(u, w, spanOf(u), 19);
    expect(r.d).toBeGreaterThan(0);
  });

  it("备选中不含与最优过于接近的周期", () => {
    const u: number[] = [];
    for (let i = 0; i < 19; i++) u.push(i * 40);
    const w = u.map(() => 1);
    const r = period(u, w, spanOf(u), 19);
    for (const [d] of r.alts) {
      expect(Math.abs(d - r.d) / r.d).toBeGreaterThan(0.02);
    }
  });
});

describe("familySplit", () => {
  // 注意：不要假设输出标签的绝对值（0=横还是 0=竖取决于 2-means 初值），
  // 只验证"同类聚合、异类分开"这个性质。

  it("横竖线段（交替 push）分成两族", () => {
    const segs: Array<[number, number, number, number]> = [];
    for (let i = 0; i < 10; i++) {
      segs.push([0, i * 20, 400, i * 20]);   // 横
      segs.push([i * 20, 0, i * 20, 400]);   // 竖
    }
    const lab = familySplit(segs);
    expect(lab.length).toBe(20);
    expect(lab.filter((v) => v === 0).length).toBe(10);
    expect(lab.filter((v) => v === 1).length).toBe(10);
    // 交替 push：偶数下标全同族，奇数下标全另一族
    for (let i = 0; i < 10; i++) {
      expect(lab[2 * i]).toBe(lab[0]);
      expect(lab[2 * i + 1]).toBe(lab[1]);
    }
    expect(lab[0]).not.toBe(lab[1]);
  });

  it("连续 push 也能正确分族", () => {
    const segs: Array<[number, number, number, number]> = [];
    for (let i = 0; i < 10; i++) segs.push([0, i * 20, 400, i * 20]);
    for (let i = 0; i < 10; i++) segs.push([i * 20, 0, i * 20, 400]);
    const lab = familySplit(segs);
    expect(lab.filter((v) => v === 0).length).toBe(10);
    expect(lab.filter((v) => v === 1).length).toBe(10);
    for (let i = 0; i < 10; i++) {
      expect(lab[i]).toBe(lab[0]);
      expect(lab[10 + i]).toBe(lab[10]);
    }
    expect(lab[0]).not.toBe(lab[10]);
  });

  it("旋转 45 度后仍能分族", () => {
    const segs: Array<[number, number, number, number]> = [];
    const c = Math.cos(Math.PI / 4), s = Math.sin(Math.PI / 4);
    for (let i = 0; i < 10; i++) {
      segs.push([c * 0 - s * (i * 20), s * 0 + c * (i * 20), c * 400 - s * (i * 20), s * 400 + c * (i * 20)]);
      segs.push([c * (i * 20) - s * 0, s * (i * 20) + c * 0, c * (i * 20) - s * 400, s * (i * 20) + c * 400]);
    }
    const lab = familySplit(segs);
    expect(lab.filter((v) => v === 0).length).toBeGreaterThan(0);
    expect(lab.filter((v) => v === 1).length).toBeGreaterThan(0);
  });
});

describe("orderQuad", () => {
  it("四点排序后不自交", () => {
    const q: Array<[number, number]> = [[10, 10], [0, 0], [10, 0], [0, 10]];
    const o = orderQuad(q);
    expect(o).toHaveLength(4);
    expect(quadArea(o[0], o[1], o[3], o[2])).toBeCloseTo(100, 6);
  });

  it("quadArea", () => {
    expect(quadArea([0, 0], [10, 0], [10, 10], [0, 10])).toBeCloseTo(100, 6);
  });
});

describe("params", () => {
  it("默认值（基线快照，待重新标定）", () => {
    expect(DEFAULT_PARAMS.blk.t).toBe(110);
    expect(DEFAULT_PARAMS.blk.q).toBe(40);
    expect(DEFAULT_PARAMS.lr.w).toEqual([1.641, -1.114, 0.988]);
    expect(DEFAULT_PARAMS.lr.b).toBe(-9.43);
    expect(DEFAULT_PARAMS.au.ns).toEqual([19, 15, 13, 9]);
    expect(DEFAULT_PARAMS.au.w).toBe(0.9);
    expect(DEFAULT_PARAMS.au.r).toBe(0.42);
    expect(DEFAULT_PARAMS.par.on).toBe(false);
    expect(DEFAULT_PARAMS.blk.ring).toBe(true);
    expect(DEFAULT_PARAMS.blk.oz).toBe(true);
  });

  it("makeParams 可覆盖且不污染默认值", () => {
    const p = makeParams({ blk: { t: 150 } });
    expect(p.blk.t).toBe(150);
    expect(DEFAULT_PARAMS.blk.t).toBe(110);   // 不污染默认值
    expect(p.blk.q).toBe(40);                  // 其余保持默认
  });
});

describe("stonesToText / textToStones", () => {
  it("往返一致", () => {
    const s = [0, 1, 2, 0];
    expect(stonesToText(s as never, 2)).toBe(". B\nW .");
    const back = textToStones(". B\nW .");
    expect(back.N).toBe(2);
    expect(back.stones).toEqual([0, 1, 2, 0]);
  });
});
