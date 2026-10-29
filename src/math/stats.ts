/**
 * 统计辅助函数。纯函数，无 opencv 依赖，可完整单测。
 */

/**
 * 百分位数。q ∈ [0,100]，线性插值。
 * 注意：会复制并排序入参（原实现行为），调用方无需预排序。
 */
export function pct(arr: ArrayLike<number>, q: number): number {
  const a: number[] = Array.prototype.slice.call(arr).sort((x: number, y: number) => x - y);
  if (!a.length) return 0;
  const p = ((a.length - 1) * q) / 100;
  const i = Math.floor(p);
  const f = p - i;
  return i + 1 < a.length ? a[i] * (1 - f) + a[i + 1] * f : a[i];
}

/** 中位数 */
export function median(a: ArrayLike<number>): number {
  return pct(a, 50);
}

/** banker's rounding（四舍六入五取偶）*/
export function rnd(x: number): number {
  let r = Math.round(x);
  if (Math.abs(x % 1) === 0.5 && r % 2 !== 0) r -= 1;
  return r;
}

export function sum(a: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s;
}

export function hypot2(x: number, y: number): number {
  return Math.sqrt(x * x + y * y);
}
