#!/usr/bin/env node
/**
 * Local static demo server for the weiqi-photo browser app.
 *
 * Zero third-party dependencies (Node built-ins only). Serves the repository
 * tree read-only over plain HTTP so an operator can open the demo at
 * http://127.0.0.1:8899/ and do the photo-corner exercise: import an image,
 * click the four board corners, transcribe the board into a stone matrix.
 *
 * This server does NOT run recognition and contains no dataset content.
 * It only maps URL paths to static files. ES modules must get a
 * JavaScript MIME type.
 */
import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = fileURLToPath(new URL(".", import.meta.url));
const ROOT = resolve(SCRIPT_DIR, "..", ".."); // repo root (this file lives in inference/tools/)
const ENTRY = "/inference/app/index.html";
const PORT = Number(process.env.DEMO_PORT ?? 8899);
const BIND = process.env.DEMO_BIND ?? "0.0.0.0";

const MIME = Object.freeze({
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".wasm": "application/wasm",
  ".txt": "text/plain; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
});

function safePath(urlPath) {
  let decoded;
  try {
    decoded = decodeURIComponent(urlPath.split("?")[0].split("#")[0]);
  } catch {
    return null;
  }
  // A bare "/" must REDIRECT to the app directory, not serve its HTML
  // inline: relative refs like ./main.js then resolve against the correct
  // base. Serving the file inline at "/" makes them resolve to /main.js (404).
  if (decoded === "/" || decoded === "") return { redirect: ENTRY.replace(/[^/]*$/, "") };
  // Also land /inference/app (no trailing slash) on the directory root.
  if (decoded === "/inference/app") return { redirect: "/inference/app/" };
  if (decoded.endsWith("/")) decoded += "index.html";
  const rel = normalize(decoded).replace(/^([/\\])+/, "");
  const full = resolve(ROOT, rel);
  // Never serve anything outside the repository root.
  if (full !== ROOT && !full.startsWith(ROOT + sep)) return null;
  return full;
}

const server = createServer(async (req, res) => {
  const p = safePath(req.url ?? "/");
  if (p && typeof p === "object" && p.redirect) {
    res.writeHead(302, { Location: p.redirect, "Cache-Control": "no-store" });
    res.end();
    return;
  }
  if (!p) {
    res.writeHead(400, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("bad path");
    return;
  }
  let st;
  try {
    st = await stat(p);
  } catch {
    res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("not found");
    return;
  }
  if (!st.isFile()) {
    res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("not found");
    return;
  }
  const type = MIME[extname(p).toLowerCase()] ?? "application/octet-stream";
  res.writeHead(200, {
    "Content-Type": type,
    "Content-Length": st.size,
    "Cache-Control": "no-store",
  });
  if (req.method === "HEAD") {
    res.end();
    return;
  }
  createReadStream(p).pipe(res);
});

server.listen(PORT, BIND, () => {
  console.log(`weiqi-photo demo: http://127.0.0.1:${PORT}${ENTRY}`);
  console.log(`serving root: ${ROOT}  (read-only static files, Node built-ins only)`);
});
