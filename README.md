# weiqi-photo

纯前端 TypeScript 库：**一张围棋棋盘照片 → 黑白空矩阵**。

核心是**传统 CV + 几何，零机器学习训练、零服务器上传** —— 整条流水线都在浏览器里跑。

方法均为公开的传统 CV / 几何技术：消失点估计、一维周期（扫频）检测、
单应变换校正、圆形度等几何特征分类。

> 需要 opencv.js（约 10MB）。本库不内置打包，运行时通过 CDN 加载或外部注入。

---

## 快速开始

```ts
import { loadCv, recognizeBoard, stonesToText } from "weiqi-photo";

const cv = await loadCv();                    // 从 CDN 加载 opencv.js
const img = cv.imread(canvas);                // canvas / <img> / ImageData
const r = recognizeBoard(cv, img);            // 自动检测路数
console.log(r.N, stonesToText(r.stones, r.N));
img.delete();
```

`r.stones` 是行主序的 `Uint8Array`（长度 `N*N`，索引 `j*N+i`）：

| 值 | 含义 |
|----|------|
| 0  | 空 |
| 1  | 黑 |
| 2  | 白 |

**不含死活判定** —— 只还原照片上看到的黑白子，不做地域/死子分析。

---

## API

| 函数 | 说明 |
|------|------|
| `loadCv(opt?)` | 加载 opencv.js（fetch 文本 → script 注入） |
| `wrapCv(cv)` | 包装已有的 opencv 实例（自己加载时用） |
| `recognizeBoard(cv, img, opt?)` | 一步到位：检测 + 判子 |
| `findBoard(cv, img, opt?)` | 只检测四角（手动校正 UI 用） |
| `readStones(cv, img, corners, size, opt?)` | 已知四角时直接判子（最快） |
| `stonesToText(stones, N)` / `textToStones(text)` | 矩阵 ↔ 文本 |

### 指定 / 自动检测路数

```ts
recognizeBoard(cv, img);                  // 自动：依次试 19/15/13/9
recognizeBoard(cv, img, { size: 13 });    // 指定 13 路
recognizeBoard(cv, img, { size: 0, autoSizes: [19, 13] });
```

自动检测要求得分 ≥ `au.w`(0.9) 且外圈占比 ≤ `au.r`(0.42)，否则换下一个候选。

### 手动校正四角

用户拖拽微调后走这条路径，跳过检测，最快：

```ts
const { N, corners } = findBoard(cv, img);
const stones = readStones(cv, img, userAdjustedCorners, N as BoardSize);
```

---

## 算法流程

```
原图
 ├─ small()          最长边缩到 1000（保证 640×480 小图也能处理）
 ├─ moire()          Lab 空间高频残差 > 3.0 → 判定摩尔纹，先高斯模糊
 └─ roughQuad()      黑帽(7×7) → boxFilter(41) → 阈值 0.04 → 开运算(15×15)
                     → 连通域 → 凸包 → 暴力搜最大面积四边形（粗定位）
        ↓
    segs()           在粗四边形外扩 1.2 倍的 ROI 内跑 HoughLinesP
        ↓
    familySplit()    线段方向 ×2 后做 2-means，分横竖两族
        ↓
    vanishingPoint() M = Σ n̂n̂ᵀ，取最小特征值的特征向量
        ↓
    period()         沿消失点投影成一维坐标 u，扫频 z(d)=Σ wᵢ·exp(2πi·uᵢ/d)
                     在 [span/26, span/6] 扫 1200 点取 |z| 最大
        ↓
    corners()        findHomography(RANSAC,3.0) + refine×3（线段吸附整数格点）
                     → 扩成 (N+8)² 滑 N×N 窗取最高分，确定棋盘偏移
        ↓
    lineFix()        沿每条线做剖面修正（木纹/格线偏移）
        ↓
    warp()           透视变换成 (N+2)² 的正视图，算 Sobel 梯度场
        ↓
    feats()          每格算特征：circ(圆形度)、arm、sd、vq、yb …
        ↓
    classify()       白子：3 权重逻辑回归  z = -9.43 + 1.641·circ·10 - 1.114·arm/10 + 0.988·sd/10
                     黑子：自适应阈值（基准 110）
        ↓
    rescue()         补漏子（最大一块逻辑，138 行）
        ↓
    stones[]         黑(1)/白(2)/空(0)
```

### 两个最值钱的设计

**1. 找棋盘 = 消失点 + 一维周期估计，而不是 Hough 找完整网格。**

Hough 在实拍照片上必然漏线、多线。这里的做法是先求两族的消失点，
把 2D 网格降成 1D 周期问题 —— 少数几条线就能定出间距，鲁棒性质变好。

**2. 判子核心特征是圆形度 `circ`，不是亮度阈值。**

在 ±10 偏移内搜 3 个半径（17/19/21 · cell/40），统计圆周 48 点中
梯度方向与半径方向一致的比例。白子判定只用了 3 个权重，说明特征选得好。

---

## 文件结构

```
src/
  math/
    stats.ts      pct / median / rnd(banker's rounding) / sum
    homog.ts      cross / inv3 / ptH / eigMin3(Jacobi 3×3) / vanishingPoint
    period.ts     period(扫频) / familySplit / orderQuad / quadArea
  cv/
    types.ts      CvLike — opencv 的最小接口（层间只传这个，不直接 import opencv）
    adapter.ts    loadCv / wrapCv
  params.ts       14 个调参对象（默认值待基于本地测试集重新标定）
  preprocess.ts   small / moire / blackhat / roughQuad / segs
  geometry/       warp / corners / lineFix
  classify/       feats / lab / rescue / shift / dedupe
  pipeline.ts     串联
  index.ts        公共入口
```

### 为什么要 `CvLike`

opencv.js 在 Node 里跑不起来：它由 emscripten 编译而来，依赖浏览器 API，
且 10MB JS 内嵌 base64 wasm，初始化开销大。所以：

- 算法层**不直接 import opencv**，只依赖 `CvLike` 接口
- 端到端验证放浏览器里跑（`tools/bench.html`）

---

## 开发

```bash
npm install
npm run typecheck     # tsc --noEmit，必须零错误
npm test              # vitest
npm run build         # 产出 dist/
```

> `dist/` 不在仓库里（已 gitignore）。clone 后必须先 `npm run build`，
> 否则 demo 与 `tools/*.html` 里的 `/dist/...` 引用会 404。

### 端到端测试（需要自备测试集）

测试集是若干张实拍照片 + 同名 `.txt` 期望矩阵（19×19，`skip/` 目录跳过）。
矩阵每行 19 个空格分隔的 token：`.`/`0` 为空，`B`/`1` 黑子，`W`/`2` 白子。

```bash
node tools/serve.mjs <你的测试集目录>
# 浏览器打开 http://localhost:8899/bench.html
```

单张调试用 `demo/index.html`（`http://localhost:8899/`）。

> opencv.js 首次加载要 1-3 分钟，之后走 HTTP 缓存。

### `?local=1`：改用本地预剥离版

默认从 CDN 拉 9.9MB 的完整 opencv.js（wasm 以 base64 内嵌）。加 `?local=1` 改用本地文件：

```bash
node tools/extract-wasm.mjs     # 生成 dist/opencv.wasm + dist/opencv.stripped.js
# 浏览器打开 http://localhost:8899/?local=1
```

省掉 base64 解码，并让 emscripten 走 `instantiateStreaming`。某次实测（chromium）：

| 模式 | 冷缓存 | 有缓存 |
|---|---|---|
| 默认（CDN，9.9MB 内嵌版） | 14.2s | 5.8s |
| `?local=1`（同源 223KB + 7.6MB wasm） | 0.33s | 0.35s |

差距主要来自外网带宽 —— 从 CDN 拉 9.9MB 通常要几十秒，**不是**编译时间
（纯 wasm 编译实测只要 ~30ms）。外网快或已缓存时两者差异很小，用默认即可。

> `dist/opencv.wasm` 与 `dist/opencv.stripped.js` 共约 7.8MB，由 `extract-wasm.mjs`
> 生成，被 `.gitignore` 忽略。缺这两个文件时请走默认模式。

---

## 调参

```ts
import { makeParams, recognizeBoard } from "weiqi-photo";

const P = makeParams({ blk: { t: 130 } });     // 黑子阈值 110 → 130
recognizeBoard(cv, img, { params: P });
```

`makeParams` 是深拷贝，不污染默认值。主要参数组：

| 组 | 用途 |
|----|------|
| `blk` | 黑子判定（阈值 `t`、环带 `r0/r1`、过曝 `oz`） |
| `lr` | 白子逻辑回归权重 |
| `gl` | 木盘/暖光修正 |
| `yc` | 黄棋黄蓝度修正 |
| `rs` | `rescue()` 补漏子 |
| `wb` | 白子亮部 |
| `lf` | 格线偏移修正 |
| `rq` | 粗定位四边形 |
| `mo` | 摩尔纹 |
| `au` | 自动路数（候选 `ns`、得分阈值 `w`） |

---

## 设计取舍

| 项 | 选择 |
|---|---|
| 判子 | 圆形度等几何特征 |
| 数子/死活 | ❌ **不做**，只出黑白空矩阵 |
| 路数 | 19/15/13/9 + 自动检测 |
| 语言 | TypeScript strict |
| opencv 依赖 | `CvLike` 接口注入，不内置打包 |

---

## License

MIT
