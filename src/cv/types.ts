/**
 * opencv.js 的最小接口契约。
 *
 * 设计目的：让上层算法只依赖这个接口，不依赖具体 opencv 实现。
 * 好处有两点：纯数学部分可脱离 opencv 单测（无需初始化 wasm），
 * 且库本身不强制打包 10MB wasm，由宿主决定加载方式。
 */

export interface MatLike {
  readonly rows: number;
  readonly cols: number;
  readonly data: Uint8Array;
  readonly data32S: Int32Array;
  readonly data32F: Float32Array;
  readonly data64F: Float64Array;
  /** connectedComponentsWithStats 的 stats 取 int（opencv 提供，d.ts 未声明） */
  intAt(row: number, col: number): number;
  /** connectedComponentsWithStats 的 centroids 取 double（opencv 提供，d.ts 未声明） */
  doubleAt(row: number, col: number): number;
  delete(): void;
}

/** 带 convertTo 的 Mat（opencv 实际提供，官方 d.ts 未声明） */
export interface MatWithConvert extends MatLike {
  convertTo(dst: MatLike, rtype: number): void;
}

export interface SizeLike { readonly width: number; readonly height: number; }

export interface ScalarLike {
  readonly 0: number; readonly 1: number; readonly 2: number; readonly 3: number;
}

export interface MatVectorLike {
  push_back(m: MatLike): void;
  delete(): void;
}

/** 本项目实际用到的 opencv 能力子集 */
export interface CvLike {
  readonly CV_8U: number;
  readonly CV_8UC1: number;
  readonly CV_8UC4: number;
  readonly CV_32F: number;
  readonly CV_32FC2: number;
  readonly CV_32SC2: number;
  readonly MORPH_RECT: number;
  readonly MORPH_BLACKHAT: number;
  readonly MORPH_OPEN: number;
  readonly COLOR_RGBA2GRAY: number;
  readonly COLOR_RGBA2RGB: number;
  readonly COLOR_RGB2Lab: number;
  readonly INTER_AREA: number;
  readonly INTER_LINEAR: number;
  readonly INTER_NEAREST: number;
  readonly BORDER_CONSTANT: number;
  readonly BORDER_REPLICATE: number;
  readonly RANSAC: number;
  readonly DIST_HUBER: number;

  readonly Mat: {
    new (): MatLike;
    new (rows: number, cols: number, type: number): MatLike;
    new (rows: number, cols: number, type: number, s: ScalarLike): MatLike;
    ones(rows: number, cols: number, type: number): MatLike;
    zeros(rows: number, cols: number, type: number): MatLike;
  };
  readonly Size: { new (width: number, height: number): SizeLike };
  readonly Scalar: {
    new (): ScalarLike;
    new (a: number): ScalarLike;
    new (a: number, b: number, c: number): ScalarLike;
    new (a: number, b: number, c: number, d: number): ScalarLike;
  };
  readonly MatVector: { new (): MatVectorLike };

  matFromArray(rows: number, cols: number, type: number, array: ArrayLike<number>): MatLike;
  getStructuringElement(shape: number, ksize: SizeLike): MatLike;
  morphologyEx(src: MatLike, dst: MatLike, op: number, kernel: MatLike): void;
  cvtColor(src: MatLike, dst: MatLike, code: number): void;
  resize(src: MatLike, dst: MatLike, size: SizeLike, fx: number, fy: number, interp: number): void;
  GaussianBlur(src: MatLike, dst: MatLike, ksize: SizeLike, sigma: number): void;
  boxFilter(src: MatLike, dst: MatLike, ddepth: number, ksize: SizeLike): void;
  Sobel(src: MatLike, dst: MatLike, ddepth: number, dx: number, dy: number): void;
  HoughLinesP(src: MatLike, lines: MatLike, rho: number, theta: number, threshold: number, minLineLength: number, maxLineGap: number): void;
  connectedComponentsWithStats(src: MatLike, labels: MatLike, stats: MatLike, centroids: MatLike): number;
  convexHull(points: MatLike, hull: MatLike, clockwise: boolean, returnPoints: boolean): void;
  fillPoly(img: MatLike, pts: MatVectorLike, color: ScalarLike): void;
  fillConvexPoly(img: MatLike, pts: MatLike, color: ScalarLike): void;
  dilate(src: MatLike, dst: MatLike, kernel: MatLike): void;
  findHomography(srcPoints: MatLike, dstPoints: MatLike, method: number, ransacReprojThreshold: number, mask?: MatLike): MatLike;
  getPerspectiveTransform(src: MatLike, dst: MatLike): MatLike;
  warpPerspective(src: MatLike, dst: MatLike, M: MatLike, dsize: SizeLike, flags?: number, borderMode?: number, borderValue?: ScalarLike): void;
  fitLine(points: MatLike, line: MatLike, distType: number, param: number, reps: number, aeps: number): void;
  imread(canvas: unknown): MatLike;
  imshow(canvas: unknown, mat: MatLike): void;
}
