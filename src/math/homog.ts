/**
 * 齐次坐标 / 单应矩阵运算。纯数学，无 opencv 依赖。
 */

export type Vec3 = [number, number, number];
/** 3x3 行主序矩阵 */
export type Mat3 = [[number, number, number], [number, number, number], [number, number, number]];

/** 三维叉积（齐次坐标下求两线交点 / 两点的连线） */
export function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

/** 3x3 矩阵求逆 */
export function inv3(m: Mat3): Mat3 {
  const a = m[0][0], b = m[0][1], c = m[0][2];
  const d = m[1][0], e = m[1][1], f = m[1][2];
  const g = m[2][0], h = m[2][1], i = m[2][2];
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
}

/** 用单应矩阵 H 变换点 (x,y) */
export function ptH(H: Mat3, x: number, y: number): [number, number] {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  return [
    (H[0][0] * x + H[0][1] * y + H[0][2]) / w,
    (H[1][0] * x + H[1][1] * y + H[1][2]) / w,
  ];
}

/** 从 cv.findHomography 返回的 Mat（data64F，9 元素）提取 3x3 */
export function fromCvMat(data: ArrayLike<number>): Mat3 {
  const d = data;
  return [
    [d[0], d[1], d[2]],
    [d[3], d[4], d[5]],
    [d[6], d[7], d[8]],
  ];
}

/**
 * 3x3 对称矩阵的最小特征值对应的特征向量（Jacobi 旋转法，60 次迭代）。
 * 用于求消失点：M = Σ n̂n̂ᵀ，其最小特征向量即消失点。
 */
export function eigMin3(M: Mat3): Vec3 {
  const a: number[][] = [
    [M[0][0], M[0][1], M[0][2]],
    [M[1][0], M[1][1], M[1][2]],
    [M[2][0], M[2][1], M[2][2]],
  ];
  const v: number[][] = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let it = 0; it < 60; it++) {
    let p = 0, q = 1;
    let mx = Math.abs(a[0][1]);
    if (Math.abs(a[0][2]) > mx) { p = 0; q = 2; mx = Math.abs(a[0][2]); }
    if (Math.abs(a[1][2]) > mx) { p = 1; q = 2; mx = Math.abs(a[1][2]); }
    if (mx < 1e-15) break;
    const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
    const sgn = th >= 0 ? 1 : -1;
    const t = sgn / (Math.abs(th) + Math.sqrt(th * th + 1));
    const c = 1 / Math.sqrt(t * t + 1);
    const s = t * c;
    for (let k = 0; k < 3; k++) {
      const akp = a[k][p], akq = a[k][q];
      a[k][p] = c * akp - s * akq;
      a[k][q] = s * akp + c * akq;
    }
    for (let k = 0; k < 3; k++) {
      const akp = a[p][k], akq = a[q][k];
      a[p][k] = c * akp - s * akq;
      a[q][k] = s * akp + c * akq;
    }
    for (let k = 0; k < 3; k++) {
      const akp = v[k][p], akq = v[k][q];
      v[k][p] = c * akp - s * akq;
      v[k][q] = s * akp + c * akq;
    }
  }
  let j = 0;
  for (let i = 1; i < 3; i++) if (a[i][i] < a[j][j]) j = i;
  return [v[0][j], v[1][j], v[2][j]];
}

/**
 * 从一组直线（齐次表示 [a,b,c]，ax+by+c=0）求消失点。
 * 构造 M = Σ n̂n̂ᵀ（法向量归一化后外积和），取最小特征向量。
 */
export function vanishingPoint(lines: Vec3[]): Vec3 {
  const M: Mat3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const l of lines) {
    const n = Math.sqrt(l[0] * l[0] + l[1] * l[1]) || 1;
    const r: Vec3 = [l[0] / n, l[1] / n, l[2] / n];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) M[i][j] += r[i] * r[j];
    }
  }
  return eigMin3(M);
}
