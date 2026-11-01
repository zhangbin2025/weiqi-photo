/**
 * 读 JPEG 的 EXIF Orientation（tag 0x0112）。
 *
 * 测试集真值 .txt 存的是 **RAW 朝向**（未应用 EXIF），而识别管线走浏览器
 * `Image.src` → canvas，浏览器会自动应用 EXIF 把图转正。两者对不上，
 * 于是带 EXIF 的图（t1 全组）差异全是旋转错位造成的假失败。
 *
 * 这里只解析 orientation，不做任何图像解码 —— 服务端没有 cv，
 * 也不该为了读一个 tag 去引 exif 解析库。
 *
 * @returns {number} 1~8；无 EXIF / 无 tag / 非 JPEG 时返回 1（等同不旋转）
 */
export function readOrientation(buf) {
  if (buf.length < 4 || buf[0] !== 0xff || buf[1] !== 0xd8) return 1; // not JPEG
  let i = 2;
  while (i + 4 <= buf.length) {
    if (buf[i] !== 0xff) { i++; continue; }
    const m = buf[i + 1];
    // 无长度的标记：SOI/EOI/TEM/RSTn
    if (m === 0xd8 || m === 0xd9 || m === 0x01 || (m >= 0xd0 && m <= 0xd7)) { i += 2; continue; }
    const len = buf.readUInt16BE(i + 2);
    if (len < 2) break;
    if (m === 0xe1 && i + 10 <= buf.length &&
        buf[i + 4] === 0x45 && buf[i + 5] === 0x78 && buf[i + 6] === 0x69 &&
        buf[i + 7] === 0x66 && buf[i + 8] === 0x00 && buf[i + 9] === 0x00) {
      // APP1 / Exif\0\0 → TIFF header 从 i+10 开始
      const tiff = buf.subarray(i + 10, Math.min(buf.length, i + 2 + len));
      if (tiff.length < 8) return 1;
      const le = tiff[0] === 0x49 && tiff[1] === 0x49; // "II" little-endian
      const e = le ? "<" : ">";
      if (!le && !(tiff[0] === 0x4d && tiff[1] === 0x4d)) return 1; // 既不是 II 也不是 MM
      const get16 = (o) => tiff.readUInt16BE ? (le ? tiff.readUInt16LE(o) : tiff.readUInt16BE(o)) : 0;
      const get32 = (o) => (le ? tiff.readUInt32LE(o) : tiff.readUInt32BE(o));
      if (tiff.length < 8) return 1;
      const off = get32(4);
      if (off + 2 > tiff.length) return 1;
      const n = get16(off);
      for (let k = 0; k < n; k++) {
        const p = off + 2 + k * 12;
        if (p + 12 > tiff.length) break;
        if (get16(p) === 0x0112) {
          const v = get16(p + 8);
          return v >= 1 && v <= 8 ? v : 1;
        }
      }
      return 1;
    }
    i += 2 + len;
  }
  return 1;
}

/**
 * orientation → 顺时针旋转 90° 的次数（对齐到浏览器显示帧）。
 * 1 不转、3 转 2 次(180°)、6 转 1 次(90°)、8 转 3 次(270°)。
 * 2/4/5/7 含镜像，测试集里不会出现；保守按不旋转处理。
 */
export function rotSteps(orientation) {
  return { 1: 0, 3: 2, 6: 1, 8: 3 }[orientation] ?? 0;
}
