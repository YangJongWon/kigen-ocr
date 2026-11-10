/** Evaluate updated web/kigen.bin + kigen-ocr.js on ExpDate real photos. */
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import jpeg from "jpeg-js";
import { loadModel, readRegion, toGray, toInput, recognize, parseDate } from "./web/kigen-ocr.js";

const ROOT = dirname(fileURLToPath(import.meta.url));
const model = await loadModel(readFileSync(join(ROOT, "web/kigen.bin")).buffer);

function loadJpeg(path) {
  const raw = jpeg.decode(readFileSync(path), { useTArray: true, formatAsRGBA: true });
  return { data: raw.data, width: raw.width, height: raw.height };
}

function crop(img, bbox, padRatio = 0.04) {
  let [x0, y0, x1, y1] = bbox.map((v) => Math.round(v));
  if (x1 < x0) [x0, x1] = [x1, x0];
  if (y1 < y0) [y0, y1] = [y1, y0];
  const pad = Math.max(1, Math.round(Math.max(1, y1 - y0) * padRatio));
  const left = Math.max(0, x0 - pad), top = Math.max(0, y0 - pad);
  const right = Math.min(img.width, x1 + pad), bot = Math.min(img.height, y1 + pad);
  const w = right - left, h = bot - top;
  if (w < 2 || h < 2) return null;
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    const src = ((top + y) * img.width + left) * 4;
    data.set(img.data.subarray(src, src + w * 4), y * w * 4);
  }
  return { data, width: w, height: h };
}

function digits(s) {
  return String(s || "").replace(/\D+/g, "");
}

function scoreSuite(name, items, mode) {
  let exact = 0, dig = 0, iso = 0, n = 0;
  const misses = [];
  for (const it of items) {
    n++;
    let r;
    if (mode === "crop-line") {
      // Date-Real / tight crop: whole image is the date line
      const g = toGray(it.img);
      const inp = toInput(g, it.img.width, it.img.height, model);
      r = recognize(model, inp.x, inp.w);
      r.date = parseDate(r.text);
    } else {
      r = readRegion(model, it.img);
    }
    const text = r.text || "";
    const gotIso = r.date?.iso || null;
    if (text === it.gt) exact++;
    if (digits(text) === digits(it.gt) && digits(it.gt)) dig++;
    if (it.iso && gotIso === it.iso) iso++;
    if (digits(text) !== digits(it.gt) || (it.iso && gotIso !== it.iso)) {
      if (misses.length < 15) misses.push({ file: it.file, gt: it.gt, iso: it.iso, text, gotIso, conf: r.conf });
    }
  }
  console.log(
    `== ${name} n=${n} text-exact=${(exact / n).toFixed(4)} digit=${(dig / n).toFixed(4)}` +
      (items[0]?.iso != null ? ` iso=${(iso / n).toFixed(4)}` : "")
  );
  for (const m of misses) {
    console.log(
      `   miss ${m.file} gt=${JSON.stringify(m.gt)} -> ${JSON.stringify(m.text)}` +
        (m.iso ? ` iso ${m.iso}->${m.gotIso}` : "") +
        ` conf=${(m.conf * 100).toFixed(0)}%`
    );
  }
}

function loadDateReal(limit = 0) {
  const base = join(ROOT, "datasets/Date-Real");
  const ann = JSON.parse(readFileSync(join(base, "annotations.json"), "utf8"));
  const items = [];
  for (const [file, meta] of Object.entries(ann)) {
    const parts = [...(meta.ann || [])].filter((a) => a.transcription).sort((a, b) => a.bbox[0] - b.bbox[0]);
    if (!parts.length) continue;
    let gt = parts[0].transcription;
    for (let i = 1; i < parts.length; i++) {
      const gap = parts[i].bbox[0] - parts[i - 1].bbox[2];
      if (gap >= 3) gt += ".";
      gt += parts[i].transcription;
    }
    items.push({ file, gt, img: loadJpeg(join(base, "images", file)) });
    if (limit && items.length >= limit) break;
  }
  return items;
}

function ymdIso(s) {
  // best-effort GT iso from ExpDate transcriptions like 2021.08.03 / 11/11/21 / JUN 2023
  const d = parseDate(s);
  return d?.iso ?? null;
}

function loadProducts(limit = 0, full = false) {
  const base = join(ROOT, "datasets/Products-Real/evaluation");
  const ann = JSON.parse(readFileSync(join(base, "annotations.json"), "utf8"));
  const items = [];
  for (const [file, meta] of Object.entries(ann)) {
    const fullImg = loadJpeg(join(base, "images", file));
    for (const a of meta.ann || []) {
      if (a.cls !== "exp" || !a.transcription) continue;
      const img = full ? fullImg : crop(fullImg, a.bbox);
      if (!img) continue;
      items.push({ file, gt: a.transcription, iso: ymdIso(a.transcription), img });
      if (limit && items.length >= limit) return items;
    }
  }
  return items;
}

console.log("model loaded");
scoreSuite("Date-Real (line crop)", loadDateReal(), "crop-line");
scoreSuite("Products-Real exp-crop + readRegion", loadProducts(0, false), "region");
scoreSuite("Products-Real FULL label + readRegion", loadProducts(80, true), "region");
