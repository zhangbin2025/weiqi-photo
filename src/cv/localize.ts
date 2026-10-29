/**
 * 把内嵌 base64 的 wasm 拆出来，让 opencv.js 走流式编译。
 *
 * 背景（实测 opencv.js 源码所得）：
 *   emscripten 的 instantiateAsync() 判断：
 *     if(!wasmBinary && WebAssembly.instantiateStreaming && !isDataURI(wasmBinaryFile) && fetch)
 *       → 流式编译（边下边编）
 *     else
 *       → instantiateArrayBuffer（先拿全部字节再一次性编译）
 *
 *   @techstark/opencv-js 的 wasm 是 base64 内嵌在 data: URI 里的，
 *   所以 isDataURI() 恒为真 → 永远走第二条路。
 *
 * 做法：在注入前把代码里的 data: URI 替换成一个真实的 .wasm URL。
 *   → isDataURI() 为假 → 走 instantiateStreaming → 边下边编。
 *
 * 收益来自两处，按实测占比排序：
 *   1) 主要：省掉 9.9MB → 7.6MB 的 base64 解码与整包下载
 *      （从 CDN 拉完整 JS 通常要几十秒，改走同源本地文件后可降到亚秒级）
 *   2) 次要：wasm 能被 HTTP 缓存，改代码不必重下
 *
 * ⚠️ 别把这里和「wasm 编译慢」混为一谈 —— 实测纯编译只要 ~30ms。
 *    早前基于该错误前提还做过 Web Worker 方案，已撤回；
 *    wasm 外置是两回事，确有收益，保留。
 */

/** 匹配 wasmBinaryFile = "data:application/octet-stream;base64,..." */
const DATA_URI_RE = /(wasmBinaryFile\s*=\s*")data:application\/octet-stream;base64,([A-Za-z0-9+/=]+)(")/;
/** 匹配预剥离版里的占位标记：wasmBinaryFile = "__WASM_URL__" */
const PLACEHOLDER_RE = /(wasmBinaryFile\s*=\s*)"__WASM_URL__"/;

export interface LocalizeResult {
  code: string;
  /** 是否成功把内嵌 wasm 换成了独立 URL */
  replaced: boolean;
  /** wasm 字节数（内嵌时） */
  wasmBytes: number;
}

/**
 * 把 opencv.js 代码里的内嵌 wasm 换成外部 URL。
 *
 * @param code opencv.js 源码文本
 * @param wasmUrl 独立 .wasm 文件的 URL（同源，或带 CORS 头）
 */
export function localizeWasm(code: string, wasmUrl: string): LocalizeResult {
  // 1) 预剥离版：只替换占位标记（JS 已是 223KB，最省）
  if (PLACEHOLDER_RE.test(code)) {
    return {
      code: code.replace(PLACEHOLDER_RE, `$1"${wasmUrl}"`),
      replaced: true,
      wasmBytes: 0,
    };
  }

  // 2) 原始版：摘掉内嵌 base64
  const m = code.match(DATA_URI_RE);
  if (!m) {
    return { code, replaced: false, wasmBytes: 0 };
  }
  const bytes = Math.floor((m[2].length * 3) / 4);
  return {
    code: code.replace(DATA_URI_RE, `$1${wasmUrl}$3`),
    replaced: true,
    wasmBytes: bytes,
  };
}
