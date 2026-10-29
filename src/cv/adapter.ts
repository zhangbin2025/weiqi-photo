/**
 * OpenCV 加载逻辑。
 *
 * 策略：fetch 文本 → script.text=code 注入 → 轮询 cv.Mat 就绪。
 * 这样做（而非 `<script src>`）是为了能在注入前后做 patch。
 * 额外做两件事：
 *
 *  1) wasm 外置（可选）
 *     实测 opencv.js 源码：emscripten 的 instantiateAsync() 判断
 *       if(!wasmBinary && instantiateStreaming && !isDataURI(wasmBinaryFile) && fetch)
 *         → 流式编译（边下边编）
 *       else → instantiateArrayBuffer（先备齐全部字节再一次性编译）
 *     而 @techstark/opencv-js 的 wasm 是 base64 内嵌在 data: URI 里的，
 *     isDataURI() 恒为真 → 永远走第二条路。
 *     把 wasmBinaryFile 换成真实 URL 后即可走流式编译。
 *
 *     实测收益主要来自「省掉 base64 解码 + 不必从外网拉整包」，
 *     而非编译更快（纯编译实测 ~30ms）。若 opencv.js 与 .wasm 都在同源
 *     本地提供，冷启动可从十几秒降到亚秒级；外网快的机器上差异会小很多。
 *
 *  2) 防 window.cv 被 DOM 抢占
 *     浏览器会把带 id 的 DOM 元素暴露成同名全局变量（named access）。
 *     页面里若有 <canvas id="cv">，window.cv 就是那个 canvas
 *     —— truthy 但没有 Mat，只看 truthy 会提前返回，导致不发起 fetch。
 */

import type { CvLike } from "./types.js";
import { localizeWasm } from "./localize.js";

export const DEFAULT_CV_URL =
  "https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js";

/** Math.abs 的替身，仅在 Math.abs 不可用时启用 */
function fallbackAbs(x: number): number {
  return x < 0 ? -x : x;
}

// ── 诊断日志 ─────────────────────────────────────────
// setCvDebug(true) 后，loader 每一步都会打点。
// 关键：pre-inject / post-inject 这对日志能区分
//   · 死循环在 opencv.js 的同步执行里 → post-inject 不出现
//   · 死循环在我的轮询里             → post-inject 出现，HB 继续但 cv 不就绪
let DEBUG = false;
export function setCvDebug(v: boolean): void {
  DEBUG = v;
}
function dbg(...a: unknown[]): void {
  if (!DEBUG) return;
  const msg = "cv: " + a.map(String).join(" ");
  const L = (globalThis as unknown as { L?: (s: string) => void }).L;
  if (typeof L === "function") L(msg);
  else console.log("[" + (performance.now() / 1000).toFixed(3) + "s] " + msg);
}

function isReady(v: unknown): boolean {
  return (
    !!v &&
    typeof v === "object" &&
    typeof (v as { Mat?: unknown }).Mat === "function"
  );
}

export interface LoadProgress {
  stage: "fetch-js" | "patch" | "inject" | "compile" | "ready";
  loaded: number;
  total: number;
}

/**
 * 把 emscripten Module 的 thenable 转成 Promise —— 但**绝不 await 那个 thenable**。
 *
 * ⚠️⚠️ 这个工程的头号坑（2026-10-29 定位，占 98% CPU 采样）：
 * emscripten 的 `Module.then(cb)` **返回 Module 自身**，而 Module 自己就是 thenable。
 * 于是 `await module` 的内部机制会：
 *     调 module.then(res) → res(Module) → 发现又是个 thenable → 再调 .then → …
 * 形成**无限微任务循环**：单核 100%、主线程冻结、连 setInterval 心跳都停。
 * 且与 wasm 编译完全无关（实测编译只要 35ms），所以之前"编译慢"的判断全错。
 *
 * 正确写法是不要 `await c`，而是 `c.then(function(m){...})` ——
 * 直接回调，不触发 thenable 同化。
 */
/**
 * 摘掉 emscripten Module 的 "thenable" 身份。
 *
 * ⚠️⚠️ 这才是本工程真正的死因（2026-10-29 用 CDP Debugger.pause 抓栈确认：
 *     冻结时栈顶帧就是 `Module.then`，CPU 采样 98% tick 都在这里）。
 * 只要 Module 上还挂着 `then`，任何 Promise 解析都会触发**thenable 同化**：
 *     resolve(Module) → 发现是 thenable → 调 Module.then(res,rej)
 *     → res(Module) → 又是 thenable → 再调 .then → 无限微任务循环
 * 表现：单核 100%、主线程冻结、连 setInterval 心跳都停。
 * 且与 wasm 编译毫无关系（实测实例化只要 35ms）。
 *
 * 正确做法是不要让 Module 直接进 Promise，而是 `res({cv:c})` —— 包一层普通
 * 对象，天然不是 thenable。这里沿用该思路（删 then + 包一层）。
 */
function neutralizeThenable(m: unknown): void {
  try {
    const o = m as Record<string, unknown>;
    if (o && typeof o.then === "function") {
      try {
        delete o.then;
      } catch {
        /* ignore */
      }
      if (typeof o.then === "function") {
        Object.defineProperty(o, "then", { value: undefined, configurable: true, writable: true });
      }
    }
  } catch {
    /* ignore */
  }
}

/** 取 Module.then 回调出的模块；包一层普通对象再 resolve（绝不拿 thenable 去 resolve） */

/**
 * 取 Module.then 回调出的模块。
 * 关键：包一层 { mod } 再 resolve —— 绝不拿 thenable 本身去 resolve。
 */
function thenableToPromise(v: { then: (cb: (m: unknown) => void) => unknown }): Promise<unknown> {
  return new Promise((resolve) => {
    let done = false;
    try {
      v.then((m: unknown) => {
        if (done) return;
        done = true;
        neutralizeThenable(m);
        resolve({ mod: m });
      });
    } catch {
      /* then 抛错就放弃这条路，回退到轮询 */
    }
    // 兜底：then 一直不回调时不要永久挂起
    setTimeout(() => {
      if (!done) {
        done = true;
        resolve(undefined);
      }
    }, 10000);
  });
}

/**
 * 等待 opencv 就绪。
 *
 * ⚠️ 必须传"取值的函数"，不能传"值"：
 *   轮询里每次都要重新读 window.cv，
 *   因为 opencv.js 是异步赋值的 —— 传进来的旧值永远是 undefined。
 *   我第一版写成 awaitCv(g.cv) 就是踩了这个坑，导致必然超时。
 */
async function awaitCv(get: () => unknown, timeoutMs = 120000): Promise<CvLike> {
  const v0 = get();
  if (isReady(v0)) {
    neutralizeThenable(v0);
    return v0 as CvLike;
  }

  // 绝不能 `await v0`：那是 thenable，会无限同化。只取它 then 回调出来的模块。
  if (v0 && typeof (v0 as Promise<unknown>).then === "function") {
    const w = (await thenableToPromise(v0 as { then: (cb: (m: unknown) => void) => unknown })) as
      | { mod?: unknown }
      | undefined;
    const m = w?.mod;
    if (isReady(m)) return m as CvLike;
    const g1 = get();
    if (isReady(g1)) {
      neutralizeThenable(g1);
      return g1 as CvLike;
    }
  }

  const t0 = Date.now();
  let n = 0;
  let thenTried = !!(v0 && typeof (v0 as Promise<unknown>).then === "function");
  while (Date.now() - t0 < timeoutMs) {
    await new Promise((r) => setTimeout(r, 100));
    n++;
    // 每 2 秒打一次点（≈20 次循环），便于看到轮询仍在推进
    if (DEBUG && n % 20 === 0) {
      dbg("poll " + n + "  elapsed=" + Math.round(Date.now() - t0) + "ms  typeof cv=" + typeof get());
    }
    const v = get();
    if (isReady(v)) {
      neutralizeThenable(v);
      return v as CvLike;
    }
    // 只尝试一次 then，且走 thenableToPromise，绝不 await 这个 thenable 本身
    if (!thenTried && v && typeof (v as Promise<unknown>).then === "function") {
      thenTried = true;
      const w = (await thenableToPromise(v as { then: (cb: (m: unknown) => void) => unknown })) as
        | { mod?: unknown }
        | undefined;
      const m = w?.mod;
      if (isReady(m)) return m as CvLike;
      const g2 = get();
      if (isReady(g2)) {
        neutralizeThenable(g2);
        return g2 as CvLike;
      }
    }
  }
  throw new Error("opencv init timeout");
}

/** 带进度的文本下载 */
async function fetchText(url: string, onProgress?: (p: LoadProgress) => void): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`opencv fetch failed: ${res.status}`);
  const total = Number(res.headers.get("content-length") || 0);

  if (!res.body || typeof res.body.getReader !== "function") {
    const t = await res.text();
    onProgress?.({ stage: "fetch-js", loaded: t.length, total: total || t.length });
    return t;
  }

  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  const dec = new TextDecoder();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.({ stage: "fetch-js", loaded, total });
  }
  let s = "";
  for (const c of chunks) s += dec.decode(c, { stream: true });
  s += dec.decode();
  return s;
}

/** HEAD 探测资源是否存在（决定能不能用外置 wasm） */
async function headOk(url: string): Promise<boolean> {
  try {
    const r = await fetch(url, { method: "HEAD" });
    return r.ok;
  } catch {
    return false;
  }
}

export interface LoadCvOptions {
  /** opencv.js 的 URL，默认 jsdelivr CDN */
  url?: string;
  /** 等待就绪的超时（毫秒），默认 120s */
  timeoutMs?: number;
  /** 已存在的实例（例如页面已加载过），直接使用 */
  existing?: unknown;
  /** 强制重新加载，忽略已有的 window.cv */
  force?: boolean;
  /**
   * 独立的 .wasm 文件 URL。给了就会把代码里内嵌的 base64 换掉，
   * 让 emscripten 走 instantiateStreaming（流式编译）。
   *
   * 主要收益是省掉 base64 解码、以及不必从外网拉 9.9MB 整包；
   * 不是为了让编译变快（纯编译实测 ~30ms）。
   * 若该 URL 不可用会自动回退到内嵌版。
   */
  wasmUrl?: string;
  onProgress?: (p: LoadProgress) => void;
}

export async function loadCv(opt: LoadCvOptions = {}): Promise<CvLike> {
  const t0 = performance.now();
  const el = () => Math.round(performance.now() - t0) + "ms";
  const url = opt.url ?? DEFAULT_CV_URL;
  const timeoutMs = opt.timeoutMs ?? 120000;
  const g = globalThis as unknown as { cv?: unknown };

  dbg("loadCv start  url=" + url + "  wasmUrl=" + (opt.wasmUrl || "(none)"));

  if (opt.existing) return awaitCv(() => opt.existing, timeoutMs);

  // ⚠️ 必须 isReady()，不能只判 truthy —— 见文件头说明 (2)
  if (!opt.force && g.cv && isReady(g.cv)) {
    dbg("reuse existing window.cv");
    return awaitCv(() => g.cv, timeoutMs);
  }

  let code = await fetchText(url, opt.onProgress);
  dbg("fetched js  " + code.length + " chars  @" + el());

  // wasm 外置：换成真实 URL → 走流式编译。
  // ⚠️ 预剥离版（opencv.stripped.js）里的 __WASM_URL__ 占位必须无条件替换，
  // 否则 emscripten 会去 fetch 字面量 "__WASM_URL__" 而失败。
  // 只有「原始内嵌版」才需要先探测 wasmUrl 是否可用（不可用时保留内嵌作回退）。
  if (opt.wasmUrl) {
    const r = localizeWasm(code, opt.wasmUrl);
    if (r.replaced) {
      code = r.code;
      dbg("localizeWasm replaced -> " + opt.wasmUrl + "  (js now " + code.length + " chars, was " + (code.length + r.wasmBytes) + ")");
      opt.onProgress?.({ stage: "patch", loaded: r.wasmBytes, total: r.wasmBytes });
    } else {
      dbg("localizeWasm: no placeholder/data-uri matched (using embedded)");
      if (await headOk(opt.wasmUrl)) {
        const r2 = localizeWasm(code, opt.wasmUrl);
        if (r2.replaced) {
          code = r2.code;
          dbg("localizeWasm (2nd try) replaced");
          opt.onProgress?.({ stage: "patch", loaded: r2.wasmBytes, total: r2.wasmBytes });
        }
      }
    }
  }
  dbg("localize done  @" + el());

  // 注入前清掉被 DOM 抢占的 window.cv（named access）
  if (g.cv && !isReady(g.cv)) {
    const el = g.cv as { nodeType?: number; id?: string; removeAttribute?: (n: string) => void };
    if (typeof el.nodeType === "number" && el.id === "cv" && el.removeAttribute) {
      el.removeAttribute("id");
    }
    try {
      delete (globalThis as unknown as Record<string, unknown>).cv;
    } catch {
      /* 某些环境下 window 属性不可删；opencv.js 自身的赋值仍会覆盖 */
    }
  }

  const script = document.createElement("script");
  script.text = code;

  // 注入期间临时替换 Math.abs，注入完还原
  const oldAbs = Math.abs;
  if (typeof oldAbs !== "function") {
    (globalThis as unknown as { Math: { abs: (x: number) => number } }).Math.abs = fallbackAbs;
  }
  dbg("pre-inject  @" + el() + "  — 若下面这行之后不再输出，说明死循环在 opencv.js 同步执行里");
  try {
    opt.onProgress?.({ stage: "inject", loaded: code.length, total: code.length });
    document.head.appendChild(script);
  } finally {
    Math.abs = oldAbs;
  }
  dbg("post-inject  @" + el() + "  window.cv=" + typeof g.cv);

  opt.onProgress?.({ stage: "compile", loaded: 0, total: 0 });
  dbg("polling for cv.Mat …");
  const cv = await awaitCv(() => g.cv, timeoutMs);
  dbg("*** READY @" + el() + "  cv.Mat=" + typeof cv.Mat + " ***");
  opt.onProgress?.({ stage: "ready", loaded: 0, total: 0 });
  return cv;
}

/** 包装已存在的 opencv 实例（宿主自己加载的情况） */
export async function wrapCv(raw: unknown, timeoutMs = 120000): Promise<CvLike> {
  return awaitCv(() => raw, timeoutMs);
}
