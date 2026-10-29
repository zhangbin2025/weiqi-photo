/**
 * 从 opencv.js 里抽出内嵌的 wasm，写成独立文件。
 *
 * 为什么：@techstark/opencv-js 的 dist 只有 opencv.js，wasm 以 base64 内嵌在
 * data: URI 里。浏览器必须先把 10MB JS 整个下载+解析，再把 base64 解码回二进制，
 * 最后才能编译 —— 多一次大内存拷贝，且无法流式编译。
 *
 * 抽出独立 .wasm 后，可以：
 *   1. 让 emscripten 走 instantiateStreaming（边下边编）
 *   2. 让浏览器/HTTP 缓存它（改代码不用重新下 7.6MB）
 *
 * 用法：node tools/extract-wasm.mjs [opencv.js路径] [输出.wasm路径]
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const src = process.argv[2] ||
  "https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js";
const out = process.argv[3] || path.resolve(__dirname, "../dist/opencv.wasm");
/** 同时产出“剥离掉内嵌 base64”的 JS，避免每次都要下载 9.9MB */
const outJs = process.argv[4] || out.replace(/\.wasm$/, ".stripped.js");

let code;
if (src.startsWith("http")) {
  process.stdout.write(`fetching ${src} ...\n`);
  const r = await fetch(src);
  if (!r.ok) throw new Error(`fetch failed: ${r.status}`);
  code = await r.text();
} else {
  code = fs.readFileSync(src, "utf8");
}

const m = code.match(/wasmBinaryFile\s*=\s*"data:application\/octet-stream;base64,([A-Za-z0-9+/=]+)"/);
if (!m) throw new Error("找不到内嵌的 wasm data URI（包结构变了？）");

const bin = Buffer.from(m[1], "base64");
if (bin.slice(0, 4).toString("binary") !== "\x00asm") {
  throw new Error("解出来的不是合法 wasm（magic 不对）");
}

fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, bin);

// 剥离版 JS：把 data: URI 换成一个占位标记（运行时由 wasmUrl 覆盖）
const stripped = code.replace(
  /(wasmBinaryFile\s*=\s*")data:application\/octet-stream;base64,[A-Za-z0-9+/=]+(")/,
  `$1__WASM_URL__$2`,
);
if (stripped.length < code.length) {
  fs.writeFileSync(outJs, stripped);
  console.log(`stripped js: ${(stripped.length / 1048576).toFixed(2)} MB → ${outJs}  (原 ${(code.length / 1048576).toFixed(2)} MB)`);
}

console.log(`wasm: ${bin.length.toLocaleString()} bytes → ${out}`);
console.log(`相比内嵌 base64（${m[1].length.toLocaleString()} 字符）省掉约 ${Math.round((1 - bin.length / m[1].length) * 100)}% 体积`);
console.log(`\n使用：loadCv({ url: "<stripped.js>", wasmUrl: "<opencv.wasm>" })`);
