// Whole-view reading (web/kigen-scene.js) on ExpDate product photos, no guide frame: each photo
// is scaled so the date line is PX pixels tall (what a glasses / wide camera view would give).
//   node bench_scene.mjs                 PX=12,16,24,32  OFFSET=150 N=100
//   DET=web/kigen-det.bin (default)      SHOW=1 lists misses   KEEP=0.3 detector window share
//   LINES=1  skip the detector scan: read every annotated text line (exp, due, prod, code ...), as
//            a platform text detector (Apple Vision) would hand them over — the upper bound of that path
// found = one of the boxes read covers the date's centre (the finder did its job)
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import jpeg from "jpeg-js";
import * as K from "./web/kigen-ocr.js";
import { readScene, readLines } from "./web/kigen-scene.js";

const ROOT = dirname(fileURLToPath(import.meta.url));
const load = async (p) => { const b = readFileSync(join(ROOT, p)); return K.loadModel(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const model = await load("web/kigen.bin"), det = await load(process.env.DET || "web/kigen-det.bin");
const base = join(ROOT, "datasets/Products-Real/evaluation");
const ann = JSON.parse(readFileSync(join(base, "annotations.json"), "utf8"));

function resize(img, s) {
  const nw = Math.max(8, Math.round(img.width * s)), nh = Math.max(8, Math.round(img.height * s));
  const out = new Uint8ClampedArray(nw * nh * 4), { data, width } = img;
  for (let j = 0; j < nh; j++) {
    const ay = Math.floor(j / s), by = Math.max(ay + 1, Math.floor((j + 1) / s));
    for (let i = 0; i < nw; i++) {
      const ax = Math.floor(i / s), bx = Math.max(ax + 1, Math.floor((i + 1) / s));
      let r = 0, g = 0, b = 0, c = 0;
      for (let v = ay; v < by; v++) for (let u = ax; u < bx; u++) { const p = (v * width + u) * 4; r += data[p]; g += data[p + 1]; b += data[p + 2]; c++; }
      const o = (j * nw + i) * 4; out[o] = r / c; out[o + 1] = g / c; out[o + 2] = b / c; out[o + 3] = 255;
    }
  }
  return { data: out, width: nw, height: nh };
}

const PX = (process.env.PX || "12,16,24,32").split(",").map(Number);
const photos = [];
let skip = Number(process.env.OFFSET ?? 150);
for (const [file, meta] of Object.entries(ann)) {
  const a = (meta.ann || []).find((a) => a.cls === "exp" && a.transcription);
  if (!a) continue;
  let [x0, y0, x1, y1] = a.bbox;
  if (x1 < x0) [x0, x1] = [x1, x0];
  if (y1 < y0) [y0, y1] = [y1, y0];
  if (y1 - y0 < 4) continue;
  if (skip-- > 0) continue;
  const gt = K.parseDate(a.transcription, { strict: false })?.iso ?? null;
  const lines = (meta.ann || []).map((q) => { const [a, b, c, d] = q.bbox; return [Math.min(a, c), Math.min(b, d), Math.max(a, c), Math.max(b, d)]; });
  photos.push({ file, box: [x0, y0, x1, y1], gt, lines });
  if (photos.length >= Number(process.env.N || 100)) break;
}

for (const px of PX) {
  let right = 0, wrong = 0, found = 0, n = 0, ms = 0, msFind = 0, size = 0;
  for (const p of photos) {
    const raw = jpeg.decode(readFileSync(join(base, "images", p.file)), { useTArray: true, formatAsRGBA: true });
    const s = px / (p.box[3] - p.box[1]);
    if (s > 1.001) continue; // would need upscaling: the photo has no smaller date to offer
    const img = resize(raw, s), box = p.box.map((v) => v * s);
    const r = process.env.LINES
      ? { ...readLines(model, det, img, p.lines.map((b) => ({ box: b.map((v) => v * s), score: 1 })), { top: 99 }), msFind: 0 }
      : readScene(model, det, img, process.env.KEEP ? { keep: Number(process.env.KEEP) } : {});
    n++; ms += r.ms; msFind += r.msFind; size += img.width * img.height;
    const cx = (box[0] + box[2]) / 2, cy = (box[1] + box[3]) / 2;
    const hit = r.reads.some((q) => q.box[0] <= cx && cx <= q.box[2] && q.box[1] <= cy && cy <= q.box[3]);
    if (hit) found++;
    if (r.date) r.date.iso === p.gt ? right++ : wrong++;
    if (process.env.SHOW && r.date?.iso !== p.gt)
      console.log(`  ${px}px ${p.file} want ${p.gt} got ${r.date?.iso ?? "-"} found ${hit} ${img.width}x${img.height}  ` +
        r.reads.map((q) => `[${q.box.map((v) => v.toFixed(0))} ${q.score.toFixed(2)} "${q.text}"]`).join(" "));
  }
  const pct = (v) => (100 * v / Math.max(n, 1)).toFixed(0).padStart(3) + "%";
  console.log(`date ${String(px).padStart(2)}px  n=${n}  right ${pct(right)} wrong ${pct(wrong)} none ${pct(n - right - wrong)}  found ${pct(found)}  ` +
    `${(ms / n).toFixed(0)}ms (find ${(msFind / n).toFixed(0)}ms)  view ~${(size / n / 1e6).toFixed(2)}MP`);
}
