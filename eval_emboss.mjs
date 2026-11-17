// readRegion on the hand-picked ExpDate crops of datasets/emboss_set.json (kind emboss / overlap).
//   node eval_emboss.mjs                 web/kigen.bin
//   MODEL=web/other.bin node eval_emboss.mjs
//   SHOW=1                               every crop, not only the misses
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import jpeg from "jpeg-js";
import * as K from "./web/kigen-ocr.js";

const ROOT = dirname(fileURLToPath(import.meta.url));
const b = readFileSync(join(ROOT, process.env.MODEL || "web/kigen.bin"));
const model = await K.loadModel(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
const set = JSON.parse(readFileSync(join(ROOT, "datasets/emboss_set.json"), "utf8")); // ExpDate crops: not published
const st = {};
for (const it of set.items) {
  const raw = jpeg.decode(readFileSync(join(ROOT, "datasets", it.split, "images", it.file)), { useTArray: true, formatAsRGBA: true });
  const [x0, y0, x1, y1] = it.bbox, p = Math.round((y1 - y0) * 0.1);
  const X0 = Math.max(0, x0 - p), Y0 = Math.max(0, y0 - p), X1 = Math.min(raw.width, x1 + p), Y1 = Math.min(raw.height, y1 + p);
  const w = X1 - X0, h = Y1 - Y0, data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) data.set(raw.data.subarray(((Y0 + y) * raw.width + X0) * 4, ((Y0 + y) * raw.width + X1) * 4), y * w * 4);
  // the label of one crop is a typo: score against what is printed
  const truth = it.note?.match(/reads (\S+)/)?.[1] ?? it.text;
  const gt = K.parseDate(truth, { strict: false })?.iso;
  const r = K.readRegion(model, { data, width: w, height: h });
  const out = r.date ? (r.date.iso === gt ? "right" : "wrong") : "none";
  // text match ignoring separators/spaces: does the recognizer see the characters?
  const norm = (s) => s.replace(/[\s.\/\-:]/g, "");
  const seen = norm(r.text).includes(norm(truth));
  const s = (st[it.kind] ??= { n: 0, right: 0, wrong: 0, none: 0, seen: 0 });
  s.n++; s[out]++; s.seen += seen;
  if (process.env.SHOW || out !== "right") console.log(`${it.kind.padEnd(7)} ${out.padEnd(5)} ${it.file.padEnd(15)} ${truth.padEnd(12)} "${r.text}"${r.rejected ? " " + r.rejected.reason : ""}`);
}
for (const [k, s] of Object.entries(st))
  console.log(`== ${k} n=${s.n} right ${s.right} wrong ${s.wrong} none ${s.none} | characters seen ${s.seen}`);
