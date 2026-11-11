// Profile readRegion on camera-like frames from ExpDate Products-Real.
// node bench_region.mjs [n]
import { readFileSync, writeFileSync, unlinkSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import jpeg from "jpeg-js";

const ROOT = dirname(fileURLToPath(import.meta.url));
const N = Number(process.argv[2] || 150);

// instrumented copy: every listed function is wrapped with an (inclusive) timer + call counter
const FNS = ["toGray", "estimateSkew", "rotateGray", "textLines", "textBand", "inkCols", "inkClusters",
  "toInput", "resizeArea", "recognize", "parseDate", "dateConf", "readGray", "readTurns"];
let src = readFileSync(process.env.KIGEN_SRC || join(ROOT, "web/kigen-ocr.js"), "utf8");
let wrap = "\nexport const __T = {}, __C = {};\n";
for (const f of FNS) {
  const exp = src.includes(`export function ${f}(`);
  src = src.replace(new RegExp(`(export )?function ${f}\\(`), `function ${f}__raw(`);
  wrap += `${exp ? "export " : ""}function ${f}(...a) { const t = performance.now(); try { return ${f}__raw(...a); } finally { __T.${f} = (__T.${f} || 0) + performance.now() - t; __C.${f} = (__C.${f} || 0) + 1; } }\n`;
}
const tmp = join(ROOT, ".prof-kigen-ocr.mjs");
writeFileSync(tmp, src + wrap);
const K = await import(pathToFileURL(tmp).href);
unlinkSync(tmp);

const buf = readFileSync(join(ROOT, "web/kigen.bin"));
const model = await K.loadModel(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));

const base = join(ROOT, "datasets/Products-Real/evaluation");
const ann = JSON.parse(readFileSync(join(base, "annotations.json"), "utf8"));

// RGBA crop, resized (box filter) so the height is at most maxH — like grab() in index.html
function grab(img, x, y, w, h, maxH = 160) {
  x = Math.max(0, Math.round(x)); y = Math.max(0, Math.round(y));
  w = Math.min(img.width - x, Math.round(w)); h = Math.min(img.height - y, Math.round(h));
  const s = Math.min(1, maxH / h), nw = Math.max(8, Math.round(w * s)), nh = Math.max(8, Math.round(h * s));
  const out = new Uint8ClampedArray(nw * nh * 4);
  for (let j = 0; j < nh; j++)
    for (let i = 0; i < nw; i++) {
      // box filter over the source block (closer to canvas drawImage with smoothing "high")
      const ax = x + Math.floor(i * w / nw), bx = Math.max(ax + 1, x + Math.floor((i + 1) * w / nw));
      const ay = y + Math.floor(j * h / nh), by = Math.max(ay + 1, y + Math.floor((j + 1) * h / nh));
      let r = 0, g = 0, b = 0, c = 0;
      for (let v = ay; v < by; v++)
        for (let u = ax; u < bx; u++) {
          const p = (v * img.width + u) * 4;
          r += img.data[p]; g += img.data[p + 1]; b += img.data[p + 2]; c++;
        }
      const o = (j * nw + i) * 4;
      out[o] = r / c; out[o + 1] = g / c; out[o + 2] = b / c; out[o + 3] = 255;
    }
  return { data: out, width: nw, height: nh };
}

const tight = [], frame = [], above = [], thin = [], thinAbove = [], gts = [];
for (const [file, meta] of Object.entries(ann)) {
  const a = (meta.ann || []).find((a) => a.cls === "exp" && a.transcription);
  if (!a) continue;
  const raw = jpeg.decode(readFileSync(join(base, "images", file)), { useTArray: true, formatAsRGBA: true });
  let [x0, y0, x1, y1] = a.bbox;
  if (x1 < x0) [x0, x1] = [x1, x0];
  if (y1 < y0) [y0, y1] = [y1, y0];
  const bh = y1 - y0, bw = x1 - x0, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  if (bh < 4 || bw < 8) continue;
  tight.push(grab(raw, x0 - bh * 0.1, y0 - bh * 0.1, bw + bh * 0.2, bh * 1.2));
  // camera guide box: 4.5:1, date ~45% of its height, a bit wider than the date
  const fh = bh * 2.2, fw = Math.max(fh * 4.5, bw * 1.15);
  frame.push(grab(raw, cx - fw / 2, cy - fh / 2, fw, fh));
  // same guide aimed a little high: the line above the date (賞味期限) comes in too
  // (shifted up until the date's bottom edge is 0.1 x its height above the guide's edge)
  above.push(grab(raw, cx - fw / 2, cy - fh / 2 - bh * 0.5, fw, fh));
  // thinner guide (7:1, date ~2/3 of its height), centred and aimed high the same way
  const th = bh * 1.5, tw = Math.max(th * 7, bw * 1.15);
  thin.push(grab(raw, cx - tw / 2, cy - th / 2, tw, th));
  thinAbove.push(grab(raw, cx - tw / 2, cy - th / 2 - bh * 0.15, tw, th));
  gts.push(K.parseDate(a.transcription, { strict: false })?.iso ?? null);
  if (tight.length >= N) break;
}

function run(name, imgs, opts) {
  for (const k in K.__T) { delete K.__T[k]; delete K.__C[k]; }
  for (let i = 0; i < 5; i++) K.readRegion(model, imgs[i], opts); // warm up
  for (const k in K.__T) { delete K.__T[k]; delete K.__C[k]; }
  const ms = [];
  let ok = 0, turned = 0, weakFirst = 0, right = 0, wrong = 0, unstableRight = 0;
  const rej = {};
  const before = () => K.__C.readGray || 0;
  for (const [idx, img] of imgs.entries()) {
    const g0 = before(), t = performance.now();
    const r = K.readRegion(model, img, opts);
    ms.push(performance.now() - t);
    if (process.env.SHOW_WRONG && r.date && r.date.iso !== gts[idx])
      console.log(`     wrong #${idx}: got ${r.date.iso} want ${gts[idx]}  text "${r.text}"  band ${r.band}/${img.height}  lines ${r.lines?.length}`);
    if (r.date) { ok++; r.date.iso === gts[idx] ? right++ : wrong++; }
    else rej[r.rejected?.reason ?? "no date"] = (rej[r.rejected?.reason ?? "no date"] || 0) + 1;
    if (r.rejected?.reason === "unstable" && r.rejected.date.iso === gts[idx]) unstableRight++;
    if ((K.__C.readGray || 0) - g0 > 1) weakFirst++;
    if (r.turn) turned++;
  }
  ms.sort((a, b) => a - b);
  const tot = ms.reduce((a, b) => a + b, 0), n = imgs.length;
  const avgW = imgs.reduce((s, i) => s + i.width, 0) / n, avgH = imgs.reduce((s, i) => s + i.height, 0) / n;
  console.log(`\n== ${name}  n=${n}  avg ${avgW.toFixed(0)}x${avgH.toFixed(0)}px  date=${(ok / n * 100).toFixed(0)}%`);
  console.log(`   per call: mean ${(tot / n).toFixed(1)} ms  p50 ${ms[n >> 1].toFixed(1)}  p90 ${ms[Math.floor(n * 0.9)].toFixed(1)}  max ${ms[n - 1].toFixed(1)}`);
  console.log(`   rotation retries (180/90/270) triggered: ${(weakFirst / n * 100).toFixed(0)}% of calls`);
  console.log(`   no date returned: ${JSON.stringify(rej)}  (unstable but actually right: ${unstableRight})`);
  console.log(`   returned dates: right ${right} (${(right / n * 100).toFixed(0)}%), wrong ${wrong} (${(wrong / n * 100).toFixed(1)}%)`);
  console.log(`   per call: recognize x${(K.__C.recognize / n).toFixed(1)}, readGray x${(K.__C.readGray / n).toFixed(2)}, parseDate x${(K.__C.parseDate / n).toFixed(0)}`);
  const rows = Object.keys(K.__T).filter((k) => k !== "readTurns" && k !== "readGray")
    .map((k) => [k, K.__T[k] / n, K.__C[k] / n]).sort((a, b) => b[1] - a[1]);
  for (const [k, t, c] of rows)
    console.log(`   ${k.padEnd(13)} ${t.toFixed(2).padStart(7)} ms/call  ${(t / (tot / n) * 100).toFixed(0).padStart(3)}%  (x${c.toFixed(1)})`);
}

console.log("wasm:", !!model.wa, "int8:", !!model.wa?.int8);
run("tight date crop", tight);
run("camera guide frame (4.5:1)", frame);
run("camera guide frame, recheck off", frame, { recheck: false });
run("guide aimed high (line above in frame), recheck off", above, { recheck: false });
run("thin guide 7:1, recheck off", thin, { recheck: false });
run("thin guide 7:1 aimed high, recheck off", thinAbove, { recheck: false });
