/**
 * 透视校正：把原图中的棋盘拉正成规则网格。
 * 透视校正：把四边形区域拉成正方形棋盘图。
 */

import type { CvLike } from "../cv/types.js";
import type { Quad } from "../types.js";

/** 透视校正后的棋盘张量（纯 TypedArray，不持有 cv.Mat） */
export interface Warped {
  /** 输出图像边长（像素） */
  W: number;
  /** 每格边长 */
  cell: number;
  /** 路数 */
  N: number;
  /** max(R,G,B)，长度 W*W */
  V: Uint8Array;
  /** 灰度，长度 W*W */
  G: Uint8Array;
  /** 黄蓝度 (R+G)/2 - B，长度 W*W */
  YB: Int16Array;
  /** 有效掩膜（在棋盘内为 1），可能为 null */
  MK: Uint8Array | null;
  /** x 方向 Sobel，长度 W*W */
  GX: Float32Array;
  /** y 方向 Sobel，长度 W*W */
  GY: Float32Array;
}

/**
 * 把四角 c 透视变换到 W×W 的正棋盘图。
 *
 * @param cell 每格像素数（默认 40）
 * @param withMask 是否生成有效掩膜（用于排除棋盘外的像素）
 */
export function warp(cv: CvLike, img: unknown, c: Quad, N: number, cell: number, withMask: boolean): Warped {
  const S = cell * (N - 1);
  const W = S + 2 * cell;
  const A = cv.matFromArray(4, 1, cv.CV_32FC2, [c[0][0], c[0][1], c[1][0], c[1][1], c[2][0], c[2][1], c[3][0], c[3][1]]);
  const B = cv.matFromArray(4, 1, cv.CV_32FC2, [cell, cell, cell + S, cell, cell, cell + S, cell + S, cell + S]);
  const M = cv.getPerspectiveTransform(A, B);
  const o = new cv.Mat();
  cv.warpPerspective(img as never, o, M, new cv.Size(W, W), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());

  const g = new cv.Mat();
  cv.cvtColor(o, g, cv.COLOR_RGBA2GRAY);

  const n = W * W;
  const V = new Uint8Array(n);
  const G = new Uint8Array(n);
  const YB = new Int16Array(n);
  const P = o.data;
  const Gd = g.data;
  for (let i = 0; i < n; i++) {
    V[i] = Math.max(P[4 * i], P[4 * i + 1], P[4 * i + 2]);
    G[i] = Gd[i];
    YB[i] = ((P[4 * i] + P[4 * i + 1]) >> 1) - P[4 * i + 2];
  }

  let MK: Uint8Array | null = null;
  if (withMask) {
    const src = img as { rows: number; cols: number };
    const mi = new cv.Mat(src.rows, src.cols, cv.CV_8UC1, new cv.Scalar(255));
    const mo = new cv.Mat();
    cv.warpPerspective(mi, mo, M, new cv.Size(W, W), cv.INTER_NEAREST, cv.BORDER_CONSTANT, new cv.Scalar(0));
    MK = new Uint8Array(mo.data);
    mi.delete();
    mo.delete();
  }

  // Sobel 需要在 float 灰度上做，先构造临时 Mat
  const gf = new cv.Mat();
  // opencv-js 的 convertTo 存在，但类型定义不全，这里用受控的鸭子类型调用
  (g as unknown as { convertTo(dst: unknown, rtype: number): void }).convertTo(gf, cv.CV_32F);
  const gb = new cv.Mat();
  cv.GaussianBlur(gf, gb, new cv.Size(3, 3), 0);
  const gx = new cv.Mat(), gy = new cv.Mat();
  cv.Sobel(gb, gx, cv.CV_32F, 1, 0);
  cv.Sobel(gb, gy, cv.CV_32F, 0, 1);
  const GX = new Float32Array(gx.data32F);
  const GY = new Float32Array(gy.data32F);

  for (const x of [A, B, M, o, g, gf, gb, gx, gy]) x.delete();
  return { W, cell, N, V, G, YB, MK, GX, GY };
}
