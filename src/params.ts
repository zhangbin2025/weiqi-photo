/**
 * 调参对象 — 整条流水线的可调数值集中在这里。
 *
 * 这些数字是从真实照片标定出来的，改动前请先在测试集上验证。
 *
 * ⚠️ 待标定：当前默认值为初始基线，建议基于自己的测试集重新标定。
 */

/** 视差场（默认关闭） */
export interface ParParams {
  on: boolean;
  sr: number; cmin: number; min: number; tol: number;
}

/** 黄棋 / 玻璃子修正 */
export interface GlParams {
  on: boolean;
  wc: number; ws: number; wc2: number; ws2: number;
  nd: number; ns: number; a: number; b: number; s: number; sv: number;
}

/** 黄色偏差修正（木盘 / 暖光） */
export interface YcParams {
  on: boolean;
  n: number; min: number; add: number; del: number; c: number; ct: number; rel: number; hn: number;
}

/** 黑子判定 / 环形采样 */
export interface BlkParams {
  on: boolean; mk: number; q: number; t: number;
  ring: boolean; r0: number; r1: number;
  dbl: number; hf: number; dbl2: number;
  sure: number; ln: number; cc: number;
  ad: boolean; gap: number; amax: number; oz: boolean;
}

/** 白子 / 盘外高光抑制 */
export interface WbParams {
  on: boolean;
  hi: number; d: number; cc: number;
  c2: number; r2: number;
  ln: number; lcc: number; lv: number; sh: number;
}

/** 补漏子 rescue（最大的一块） */
export interface RsParams {
  nbt: boolean; on: boolean; lnx: number; rcpu: number;
  mt: boolean; mk: number; tx: number; sc: number; rf: number; nr: number; mr: number;
  dup: number; same: number; ilr: number; ct: number; wb: number; nf: boolean; k: number;
  eo: number; or: number; olc: number; ob: number;
  r: number; r2: number; c: number; c2: number; d: number; lc: number; lr: number;
  mid: boolean; mw: number; mrm: number; pr: number; pb: number; pw: number;
  rc: number; rcp: number; rcm: number; rch: number; rct: number;
  tie: number; rrf: boolean; nb2: boolean;
  dbg: boolean; di: number; dj: number;
}

/** 棋盘线偏移修正 */
export interface LfParams {
  on: boolean;
  r: number; lo: number; hi: number; n: number; t: number; pk: number;
  min: number; tol: number; max: number; minc: number;
  hom: boolean; hn: number; ht: number;
}

/** 粗定位 roughQuad */
export interface RqParams {
  on: boolean;
  g: number; ma: number; c: number; m: number; ec: number;
  alt: number; na: number; rw: number; rm: number; rs: number;
}

/** 摩尔纹抑制 */
export interface MoParams { on: boolean; t: number; s: number; s1: number; }

/** 自动路数 */
export interface AuParams { w: number; r: number; ns: number[]; }

/** 死子剔除（阴影方向） */
export interface PbParams { on: boolean; c: number; t: number; }

/** 白子逻辑回归权重 */
export interface LrParams { w: [number, number, number]; b: number; }

/** 去重（默认关闭） */
export interface DdParams { on: boolean; r: number; st: number; d: number; col: boolean; }

export interface Params {
  par: ParParams;
  gl: GlParams;
  yc: YcParams;
  blk: BlkParams;
  wb: WbParams;
  rs: RsParams;
  lf: LfParams;
  rq: RqParams;
  mo: MoParams;
  au: AuParams;
  pb: PbParams;
  lr: LrParams;
  dd: DdParams;
}

export const DEFAULT_PARAMS: Params = {
  par: { on: false, sr: 0.5, cmin: 0.6, min: 12, tol: 0.18 },
  gl: { on: true, wc: 205, ws: 0.1, wc2: 195, ws2: 0.05, nd: 0.43, ns: 0.1, a: 0.4, b: 0.6, s: 0.05, sv: 248 },
  yc: { on: true, n: 10, min: 12, add: 0.85, del: 0.2, c: 0.4, ct: 40, rel: 0.5, hn: 5 },
  blk: {
    on: true, mk: 1, q: 40, t: 110,
    ring: true, r0: 0.15, r1: 0.42,
    dbl: 0.9, hf: 1.45, dbl2: 0.5,
    sure: 70, ln: -1, cc: 0.55,
    ad: true, gap: 30, amax: 0.75, oz: true,
  },
  wb: { on: true, hi: 225, d: 4, cc: 0.75, c2: 0.6, r2: 0.94, ln: 40, lcc: 0.8, lv: 40, sh: 15 },
  rs: {
    nbt: true, on: true, lnx: 150, rcpu: 0.95,
    mt: false, mk: 3, tx: 80, sc: 0.8, rf: 0.15, nr: 0.3, mr: 0.6,
    dup: 0.4, same: 0.5, ilr: 0.3, ct: 20, wb: 1.04, nf: true, k: 200,
    eo: 0.6, or: 0.6, olc: 28, ob: 1.08,
    r: 0.4, r2: 0.5, c: 0.55, c2: 0.75, d: 0.85, lc: 15, lr: 0.45,
    mid: true, mw: 0.1, mrm: 0.05, pr: 0.3, pb: 0.85, pw: 1.06,
    rc: 2, rcp: 0.2, rcm: 0.5, rch: 0.8, rct: 0.2,
    tie: 0.1, rrf: false, nb2: false,
    dbg: false, di: -1, dj: -1,
  },
  lf: { on: true, r: 0.3, lo: 0.55, hi: 1.12, n: 0.5, t: 3, pk: 2.5, min: 4, tol: 0.08, max: 0.45, minc: 0.04, hom: false, hn: 4, ht: 2.0 },
  rq: { on: true, g: 0.08, ma: 0.005, c: 0.35, m: 0.02, ec: 0.4, alt: 0.5, na: 3, rw: 0.9, rm: 0.03, rs: 0.97 },
  mo: { on: true, t: 3.0, s: 3, s1: 1 },
  au: { w: 0.9, r: 0.42, ns: [19, 15, 13, 9] },
  pb: { on: true, c: 0.53, t: 6 },
  lr: { w: [1.641, -1.114, 0.988], b: -9.43 },
  dd: { on: false, r: 0.6, st: 4, d: 0.5, col: false },
};

/** 深拷贝默认参数并应用覆盖 */
export function makeParams(over?: Partial<Params>): Params {
  const p: Params = JSON.parse(JSON.stringify(DEFAULT_PARAMS)) as Params;
  if (over) {
    for (const k of Object.keys(over) as Array<keyof Params>) {
      const v = over[k];
      if (v) Object.assign(p[k] as object, v);
    }
  }
  return p;
}
