/**
 * weiqi-photo — 纯前端拍照还原围棋棋盘矩阵库。
 *
 * 核心思路（均为公开的传统 CV / 几何方法）：
 *   消失点 + 一维周期估计求棋盘四角，再用圆形度等特征判黑白空。
 *   全程传统 CV，无机器学习，不上传服务器。
 *
 * 用法：
 *   const cv = await loadCv();
 *   const img = cv.imread(canvas);
 *   const { N, corners, stones } = recognizeBoard(cv, img);
 */

import type { CvLike, MatLike } from "./cv/types.js";
import type { BoardResult, BoardSize, Pt, Quad, RecognizeOptions, Stone } from "./types.js";
import type { Params } from "./params.js";
import { makeParams, DEFAULT_PARAMS } from "./params.js";
import { runPipeline } from "./pipeline.js";

export { loadCv, wrapCv, DEFAULT_CV_URL } from "./cv/adapter.js";
export type { LoadCvOptions } from "./cv/adapter.js";
export type { CvLike, MatLike } from "./cv/types.js";
export type {
  BoardResult, BoardSize, DebugInfo, Params, Pt, Quad, RecognizeOptions, Stone, CellFeature,
} from "./types.js";
export { DEFAULT_PARAMS, makeParams } from "./params.js";
export { runPipeline, detectCorners, recognize } from "./pipeline.js";

// 纯数学（可脱离 opencv 使用）
export { pct, median, rnd, sum } from "./math/stats.js";
export { cross, inv3, ptH, eigMin3, vanishingPoint } from "./math/homog.js";
export type { Mat3, Vec3 } from "./math/homog.js";
export { period, familySplit, orderQuad, quadArea } from "./math/period.js";
export type { PeriodResult } from "./math/period.js";

/**
 * 识别棋盘：从图像矩阵还原黑白空矩阵。
 *
 * @param cv 已就绪的 opencv 实例
 * @param img cv.imread() 得到的图像矩阵
 * @param opt 选项；size 省略时自动检测路数（19/15/13/9）
 */
export function recognizeBoard(
  cv: CvLike,
  img: MatLike,
  opt: RecognizeOptions = {},
): BoardResult {
  return runPipeline(cv, img, {
    size: opt.size ?? 0,
    corners: opt.corners,
    params: opt.params,
    keepFeatures: opt.keepFeatures,
  });
}

/**
 * 只检测棋盘四角（不判子）。
 * 用于需要手动校正四角的 UI 流程。
 */
export function findBoard(
  cv: CvLike,
  img: MatLike,
  opt: { size?: BoardSize | 0; params?: Partial<Params> } = {},
): { N: number; corners: Quad; debug: BoardResult["debug"] } {
  const r = runPipeline(cv, img, {
    size: opt.size ?? 0,
    params: opt.params,
  });
  return { N: r.N, corners: r.corners, debug: r.debug };
}

/**
 * 已知四角时直接判子（跳过检测）。
 * 用户手动微调四角后走这条路径，最快。
 */
export function readStones(
  cv: CvLike,
  img: MatLike,
  corners: Quad,
  size: BoardSize,
  opt: { params?: Partial<Params> } = {},
): Stone[] {
  return runPipeline(cv, img, {
    size,
    corners,
    params: opt.params,
  }).stones;
}

/** 把矩阵转成易读的字符串（调试用）：. 空 / B 黑 / W 白 */
export function stonesToText(stones: Stone[], N: number): string {
  const rows: string[] = [];
  for (let j = 0; j < N; j++) {
    let s = "";
    for (let i = 0; i < N; i++) {
      const v = stones[j * N + i];
      s += v === 1 ? "B" : v === 2 ? "W" : ".";
      s += i < N - 1 ? " " : "";
    }
    rows.push(s);
  }
  return rows.join("\n");
}

/** 从文本解析回矩阵（测试用） */
export function textToStones(text: string): { N: number; stones: Stone[] } {
  const rows = text.trim().split("\n").map((r) => r.trim().split(/\s+/));
  const N = rows.length;
  const stones: Stone[] = new Array(N * N);
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const t = rows[j][i];
      stones[j * N + i] = t === "B" ? 1 : t === "W" ? 2 : 0;
    }
  }
  return { N, stones };
}
