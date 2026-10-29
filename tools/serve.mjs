/**
 * 本地静态服务 + 测试集接口。
 *
 *   node tools/serve.mjs [photoDir]
 *
 * 提供：
 *   /             → demo/index.html
 *   /dist/*       → 编译产物
 *   /photo/*      → 测试集图片
 *   /bench/list   → 测试集清单（JSON：{dir, items:[{name,img,txt}]}）
 *
 * ⚠️ 这里**不提供** /bench/run：跑测试集需要 opencv，而 opencv 只存在于浏览器
 *    （node 端没有 cv，也不可内置）。测试集由前端拿 /bench/list 后逐张在浏览器里跑。
 *    早期注释曾承诺 POST /bench/run 但从未实现，导致前端请求它 → 落到静态处理器
 *    → 返回 404 纯文本 "not found" → res.json() 报 "Unexpected token 'o'"。
 */

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
// 测试集目录：不设默认值 —— 各人照片存放位置不同，硬编码某台机器的路径没意义。
// 用法：node tools/serve.mjs <测试集目录>
const PHOTO_DIR = process.argv[2] || "";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
  ".png": "image/png", ".svg": "image/svg+xml",
  ".css": "text/css; charset=utf-8",
  ".wasm": "application/wasm",
};

/** 递归收集测试集文件（跳过 skip 目录） */
function collect(dir) {
  const out = [];
  if (!fs.existsSync(dir)) return out;
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) {
      if (e.name === "skip") continue;
      out.push(...collect(p));
    } else if (e.name.endsWith(".jpg")) {
      const txt = p.replace(/\.jpg$/, ".txt");
      if (fs.existsSync(txt)) out.push({ jpg: p, txt });
    }
  }
  return out.sort((a, b) => a.jpg.localeCompare(b.jpg));
}

const DIAG_LOG = process.env.DIAG_LOG || "./weiqi-photo-diag.log";

const server = http.createServer((req, res) => {
  const url = new URL(req.url, "http://x");
  const p = url.pathname;

  // 诊断日志：页面用 sendBeacon 实时上报，页面冻住也不影响（冻结前的行都已发出）
  if (p === "/diaglog" && req.method === "POST") {
    let body = "";
    req.on("data", (c) => { body += c; if (body.length > 1e6) req.destroy(); });
    req.on("end", () => {
      try { fs.appendFileSync(DIAG_LOG, body.replace(/\n*$/, "\n")); } catch { /* ignore */ }
      res.writeHead(204); res.end();
    });
    return;
  }
  if (p === "/diaglog") {
    let txt = "(空)";
    try { txt = fs.existsSync(DIAG_LOG) ? fs.readFileSync(DIAG_LOG, "utf8") : "(尚无日志)"; } catch { /* ignore */ }
    res.writeHead(200, { "content-type": "text/plain; charset=utf-8", "cache-control": "no-store" });
    res.end(txt);
    return;
  }
  if (p === "/diagclear" && req.method === "POST") {
    try { fs.writeFileSync(DIAG_LOG, ""); } catch { /* ignore */ }
    res.writeHead(204); res.end();
    return;
  }

  if (p === "/bench/list") {
    const items = collect(PHOTO_DIR).map((f) => ({
      name: path.basename(f.jpg),
      img: "/photo/" + path.relative(PHOTO_DIR, f.jpg),
      txt: "/photo/" + path.relative(PHOTO_DIR, f.txt),
    }));
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ dir: PHOTO_DIR, items }));
    return;
  }

  // 测试集「运行」接口：只回传清单 + 期望矩阵文本。
  //
  // ⚠️ 这里只做数据分发，**不做识别**：opencv 只在浏览器里有（node 端没有 cv，
  //    也不可能为了跑分在服务端内置 10MB wasm）。识别由前端逐张跑，
  //    结果汇总成 {pass,total,avgAcc,items} 再渲染。
  //
  // 早期这里只声明了 POST /bench/run 却没实现，请求落到静态文件处理器 →
  // 返回 404 的 "not found" 纯文本 → 前端 res.json() 解析它 →
  // "Unexpected token 'o', \"not found\" is not valid JSON"。
  //
  // 期望矩阵是 19 行、每行 19 个空格分隔的 '.', 'B', 'W'。
  if (p === "/bench/run") {
    const items = collect(PHOTO_DIR).map((f) => {
      const st = fs.statSync(f.jpg);
      let expect = "";
      try {
        expect = fs.readFileSync(f.txt, "utf8");
      } catch { /* 缺期望文件就留空，前端计为错误项 */ }
      return {
        name: path.basename(f.jpg),
        img: "/photo/" + path.relative(PHOTO_DIR, f.jpg),
        bytes: st.size,
        expect,
      };
    });
    res.writeHead(200, { "content-type": "application/json; charset=utf-8" });
    res.end(JSON.stringify({ dir: PHOTO_DIR, items }));
    return;
  }

  if (p.startsWith("/photo/")) {
    const fp = path.join(PHOTO_DIR, p.slice("/photo/".length));
    if (!fp.startsWith(PHOTO_DIR) || !fs.existsSync(fp)) {
      res.writeHead(404); res.end("not found"); return;
    }
    res.writeHead(200, { "content-type": MIME[path.extname(fp)] || "application/octet-stream" });
    fs.createReadStream(fp).pipe(res);
    return;
  }

  // 静态文件
  let fp = p === "/" ? path.join(ROOT, "demo", "index.html") : path.join(ROOT, p.slice(1));
  if (!fp.startsWith(ROOT)) { res.writeHead(403); res.end("forbidden"); return; }
  if (!fs.existsSync(fp) || fs.statSync(fp).isDirectory()) {
    res.writeHead(404); res.end("not found"); return;
  }
  res.writeHead(200, {
    "content-type": MIME[path.extname(fp)] || "application/octet-stream",
    "cache-control": "no-store",
  });
  fs.createReadStream(fp).pipe(res);
});

const PORT = process.env.PORT || 8899;
server.listen(PORT, () => {
  const items = collect(PHOTO_DIR);
  console.log(`weiqi-photo demo: http://localhost:${PORT}/`);
  if (!PHOTO_DIR) {
    console.log("测试集: 未指定 —— 用法 node tools/serve.mjs <测试集目录>");
  } else {
    console.log(`测试集: ${PHOTO_DIR} — ${items.length} 张`);
  }
});
