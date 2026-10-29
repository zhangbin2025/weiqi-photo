/**
 * 公共类型定义。
 */

/** 棋盘交叉点状态：0=空，1=黑，2=白 */
export type Stone = 0 | 1 | 2;

/** 支持的棋盘路数 */
export type BoardSize = 9 | 13 | 15 | 19;

/** 平面 2D 点 */
export type Pt = [number, number];

/** 齐次直线 [a,b,c]，满足 ax+by+c=0 */
export type Line3 = [number, number, number];

/** 线段 [x0,y0,x1,y1] */
export type Seg = [number, number, number, number];

/** 棋盘四角，顺序：左上、右上、左下、右下 */
export type Quad = [Pt, Pt, Pt, Pt];

/** 识别结果 */
export interface BoardResult {
  /** 路数 */
  N: number;
  /** 棋盘四角（原图坐标系） */
  corners: Quad;
  /** N×N 的黑白空矩阵，行主序，索引 j*N+i */
  stones: Stone[];
  /** 调试信息（可选导出） */
  debug: DebugInfo;
}

/** 调试/诊断信息，用于调参和问题定位 */
export interface DebugInfo {
  /** 检测到的摩尔纹强度 */
  moire: number;
  /** 四角求解的滑窗得分（0~1，越高越可信） */
  win: number;
  /** 是否为自动路数检测 */
  auto: boolean;
  /** 自动检测时的各路数尝试日志 */
  aulog: Array<[number, number, number | null, number | null]>;
  /** 粗定位的四边形 */
  quad: Pt[] | null;
  /** 线段数量 */
  nSegs: number;
  /** 两族的线段数量 */
  fam: [number, number] | null;
  /** 周期估计日志 [span, d, phase] */
  per: Array<[number, number, number]>;
  /** 分类阈值 */
  threshold: number;
  /** 每交叉点特征（可选） */
  features: CellFeature[][] | null;
  /** 附加的任意诊断数据 */
  [key: string]: unknown;
}

/** 单个交叉点的特征向量 */
export interface CellFeature {
  /** 最优子中心相对格点的偏移 */
  ox: number;
  oy: number;
  /** 暗像素质心到格点的归一化距离 */
  dc: number;
  /** 中心圆盘亮度中位数 */
  v: number;
  /** 环形采样亮度分位数（抗高光） */
  vq: number;
  /** 十字暗线"手臂"凹陷度（排序后第 3 个） */
  arm: number;
  /** 原始手臂凹陷度 */
  line: number;
  /** 圆盘灰度标准差 */
  sd: number;
  /** 圆形度 0~1，判子的核心特征 */
  circ: number;
  /** 子中心处亮度 */
  vc: number;
  /** 高光饱和像素占比 */
  sat: number;
  /** 环形采样值（判黄棋） */
  rv: number[];
  /** 环形暗像素比例 */
  nsd: number;
  /** 环形有效采样数 */
  nsn: number;
  /** 黄蓝度中位数（大范围） */
  yb: number;
  /** 黄蓝度中位数（核心区） */
  yb0: number;
}

import type { Params } from "./params.js";
export type { Params } from "./params.js";

/** 识别选项 */
export interface RecognizeOptions {
  /** 指定路数；0 或省略表示自动检测 */
  size?: BoardSize | 0;
  /** 手动指定四角（跳过检测） */
  corners?: Quad;
  /** 自动检测的候选路数，默认 [19,15,13,9] */
  autoSizes?: BoardSize[];
  /** 是否输出调试信息 */
  debug?: boolean;
  /** 参数覆盖 */
  params?: Partial<Params>;
  /** 是否在 debug.features 里输出每交叉点特征（调参用，默认 false） */
  keepFeatures?: boolean;
}
