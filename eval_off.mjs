// kigen on the Open Food Facts pilot (datasets/off_pilot, labelled by vlm_collect.py): photos
// where both vision models read the same expiry date. A camera-guide frame (7:1, the date about
// 2/3 of its height) is cut around the box the first model gave, and readRegion runs on it.
//   node eval_off.mjs            MODEL=web/other.bin   SHOW=1 (every photo)
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import jpeg from "jpeg-js";
import * as K from "./web/kigen-ocr.js";

const ROOT = dirname(fileURLToPath(import.meta.url)), DIR = join(ROOT, "datasets/off_pilot");
const load = async (p) => { const b = readFileSync(join(ROOT, p)); return K.loadModel(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const model = await load(process.env.MODEL || "web/kigen.bin"), det = await load("web/kigen-det.bin");
const labels = JSON.parse(readFileSync(join(DIR, "labels.json"), "utf8"));
const norm = (s) => (s || "").replace(/[\s.\/\-:]/g, "");

// RGBA crop of [x, y, w, h] (clamped), box-filtered down to at most 160 px tall like the app's grab()
function grab(img, x, y, w, h, maxH = 160) {
  x = Math.max(0, Math.round(x)); y = Math.max(0, Math.round(y));
  w = Math.min(img.width - x, Math.round(w)); h = Math.min(img.height - y, Math.round(h));
  const s = Math.min(1, maxH / h), nw = Math.max(8, Math.round(w * s)), nh = Math.max(8, Math.round(h * s));
  const out = new Uint8ClampedArray(nw * nh * 4);
  for (let j = 0; j < nh; j++)
    for (let i = 0; i < nw; i++) {
      const ax = x + Math.floor(i * w / nw), bx = Math.max(ax + 1, x + Math.floor((i + 1) * w / nw));
      const ay = y + Math.floor(j * h / nh), by = Math.max(ay + 1, y + Math.floor((j + 1) * h / nh));
      let r = 0, g = 0, b = 0, c = 0;
      for (let v = ay; v < by; v++) for (let u = ax; u < bx; u++) { const p = (v * img.width + u) * 4; r += img.data[p]; g += img.data[p + 1]; b += img.data[p + 2]; c++; }
      const o = (j * nw + i) * 4; out[o] = r / c; out[o + 1] = g / c; out[o + 2] = b / c; out[o + 3] = 255;
    }
  return { data: out, width: nw, height: nh };
}

const st = { n: 0, right: 0, wrong: 0, none: 0, noGt: 0 };
for (const [file, r] of Object.entries(labels)) {
  const a = r.a ?? {}, b = r.b ?? {};
  if (!(a.has_date && b.has_date && norm(a.text) && norm(a.text) === norm(b.text))) continue;
  const gt = K.parseDate(a.text, { strict: false })?.iso;
  if (!gt) { st.noGt++; continue; }
  const img = jpeg.decode(readFileSync(join(DIR, "images", file)), { useTArray: true, formatAsRGBA: true, maxMemoryUsageInMB: 2048 });
  const [x0, y0, x1, y1] = (a.box ?? [0, 0, 1000, 1000]).map((v, i) => v / 1000 * (i % 2 ? img.height : img.width));
  const bh = Math.max(4, y1 - y0), th = bh * 1.5, tw = Math.max(th * 7, (x1 - x0) * 1.15), cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const res = K.readRegion(model, grab(img, cx - tw / 2, cy - th / 2, tw, th), { det });
  const out = res.date ? (res.date.iso === gt ? "right" : "wrong") : "none";
  st.n++; st[out]++;
  if (process.env.SHOW || out !== "right") console.log(`${out.padEnd(5)} ${file.padEnd(22)} vlm "${a.text}" (${a.label || "-"}) kigen "${res.text}"${res.rejected ? " " + res.rejected.reason : ""}`);
}
console.log(`== both models agree, parseable: n=${st.n}  kigen right ${st.right} wrong ${st.wrong} none ${st.none}  (agreed but not a date to the parser: ${st.noGt})`);
