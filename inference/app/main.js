// weiqi-photo demo — import an image, click four corners, run the CNN.
import { decodeFile } from "../src/host.mjs";
import { runCnnReading } from "../src/cnn_inference.mjs";
import { unitFromQuad, rasterFromUnit, crossingSites, gridSpacing } from "../src/homography.mjs";

const CLASS_CHAR = [".", "B", "W"];
const $ = (id) => document.getElementById(id);
const canvas = $("view");
const ctx = canvas.getContext("2d");
const out = $("out");
const badge = $("badge");
const btnRecognize = $("recognize");
const LINES = () => Number($("lines").value);

let bitmap = null;
let frame = null;
let corners = [];
let scale = 1;
let offsetX = 0, offsetY = 0;
let busy = false;
let cnnModel = null;
let fileName = "";

function setBadge(text, kind) {
  badge.textContent = text;
  badge.className = "badge " + kind;
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (bitmap) {
    scale = Math.min(canvas.width / bitmap.width, canvas.height / bitmap.height);
    offsetX = (canvas.width - bitmap.width * scale) / 2;
    offsetY = (canvas.height - bitmap.height * scale) / 2;
    ctx.drawImage(bitmap, offsetX, offsetY, bitmap.width * scale, bitmap.height * scale);
  }
  if (corners.length) {
    ctx.strokeStyle = "rgba(255,220,80,0.95)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    corners.forEach((p, i) => {
      const x = offsetX + p.x * scale, y = offsetY + p.y * scale;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    if (corners.length === 4) ctx.closePath();
    ctx.stroke();
    ctx.fillStyle = "rgba(255,80,80,0.95)";
    for (let i = 0; i < corners.length; i++) {
      const p = corners[i];
      const x = offsetX + p.x * scale, y = offsetY + p.y * scale;
      ctx.fillRect(x - 4, y - 4, 8, 8);
      ctx.fillStyle = "#000";
      ctx.font = "11px sans-serif";
      ctx.fillText(String(i + 1), x + 5, y - 5);
      ctx.fillStyle = "rgba(255,80,80,0.95)";
    }
  }
}

function matrixText(M, lines) {
  if (!Array.isArray(M)) return "  （无矩阵）";
  const rows = [];
  for (let r = 0; r < lines; r++) {
    rows.push(M.slice(r * lines, (r + 1) * lines).map((v) => CLASS_CHAR[v] ?? "?").join(""));
  }
  return rows.join("\n");
}

function printResult(res, lines, name) {
  const L = [];
  L.push(`文件      : ${name}`);
  L.push(`识别      : CNN (ONNX / browser)`);
  L.push(`路数      : ${lines}`);
  if (res.diagnostics?.spacing) L.push(`格距(px)  : ${res.diagnostics.spacing?.toFixed?.(2) ?? res.diagnostics.spacing}`);
  if (typeof res.quality === "number") L.push(`平均置信度: ${res.quality.toFixed(4)}`);
  L.push("");
  L.push("识别矩阵（. 空 / B 黑 / W 白）：");
  const M = Array.isArray(res.M) ? res.M : Array.from(res.M ?? []);
  L.push(matrixText(M, lines));
  const b = M.reduce((a, v) => (a[v] = (a[v] || 0) + 1, a), {});
  L.push("");
  L.push(`类别计数  : . ${b[0] || 0} / B ${b[1] || 0} / W ${b[2] || 0}`);
  setBadge("识别完成", "ok");
  out.textContent = L.join("\n");
}

async function recognize() {
  if (busy || !frame || corners.length !== 4) return;
  busy = true;
  setBadge("识别中…", "wait");
  out.textContent = "正在识别（CNN）…";
  btnRecognize.disabled = true;
  try {
    const lines = LINES();
    const q = unitFromQuad(corners);
    if (!q.ok) throw new Error("geometry: " + q.reason);
    const hInv = rasterFromUnit(q.h);
    const built = crossingSites(hInv, lines);
    if (!built.ok) throw new Error("geometry: " + built.reason);
    const sites = built.sites;
    const spacing = gridSpacing(sites, lines);
    const survey = { sites, spacing, lines, coverage: 1 };

    if (!cnnModel) {
      setBadge("加载 CNN 模型…", "wait");
      const resp = await fetch("/training/out-cnn/cnn.onnx");
      if (!resp.ok) throw new Error(`ONNX 加载失败: ${resp.status}（请检查 training/out-cnn/cnn.onnx 是否随仓库签入）`);
      cnnModel = new Uint8Array(await resp.arrayBuffer());
    }

    const res = await runCnnReading(frame.decoded, survey, { onnxModel: cnnModel });
    printResult(res, lines, fileName);
  } catch (e) {
    setBadge("出错", "no");
    out.textContent = `识别异常：${e?.message ?? e}\n${e?.stack ?? ""}`;
  } finally {
    busy = false;
    btnRecognize.disabled = corners.length !== 4;
  }
}

$("file").addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  fileName = file.name;
  setBadge("载入中…", "wait");
  out.textContent = "正在解码图片…";
  try {
    bitmap = await createImageBitmap(file);
    frame = await decodeFile(file);
    corners = [];
    btnRecognize.disabled = true;
    draw();
    setBadge("图片已载入 · 请点 4 个角 (0/4)", "wait");
    out.textContent = "图片已显示。在图上依次点：左上 → 右上 → 右下 → 左下。点满 4 下后点「识别」。";
  } catch (err) {
    bitmap = null; frame = null;
    setBadge("载入失败", "no");
    out.textContent = `图片解码失败：${err?.message ?? err}`;
    draw();
  }
});

canvas.addEventListener("pointerdown", (e) => {
  if (!bitmap || busy) return;
  const r = canvas.getBoundingClientRect();
  const sx = canvas.width / r.width, sy = canvas.height / r.height;
  const cx = (e.clientX - r.left) * sx, cy = (e.clientY - r.top) * sy;
  const p = { x: (cx - offsetX) / scale, y: (cy - offsetY) / scale };
  if (corners.length >= 4) { corners = []; btnRecognize.disabled = true; }
  corners.push(p);
  draw();
  if (corners.length === 4) {
    setBadge("4 角已选 · 点「识别」", "wait");
    btnRecognize.disabled = false;
  } else {
    setBadge(`图片已载入 · 请点 4 个角 (${corners.length}/4)`, "wait");
  }
});

$("reset").addEventListener("click", () => {
  corners = [];
  btnRecognize.disabled = true;
  draw();
  if (bitmap) setBadge("图片已载入 · 请点 4 个角 (0/4)", "wait");
  out.textContent = "四角已清除。依次点：左上 → 右上 → 右下 → 左下。";
});

btnRecognize.addEventListener("click", recognize);

draw();
