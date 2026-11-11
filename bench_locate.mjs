// Does readRegion find the date line inside a camera guide frame? Products-Real eval photos,
// cropped like camera frames (guide 4.5:1 / 7:1, aimed high, held far, tilted with ROT=1).
//   CMP=1        right/wrong per frame set, strips off vs on
//   TIGHT=1      was a candidate band snug around the date? (vertical / horizontal)
//   NOSNUG=<set> frames with no snug band: row edge profiles, full width vs date columns
//   NEWWRONG=<set> frames that turned wrong with the current options
//   OFFSET, N    photo range (0-149 was used while tuning; 150+ is held out)
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import jpeg from "jpeg-js";
import * as K from "./web/kigen-ocr.js";

const ROOT = dirname(fileURLToPath(import.meta.url));
const buf = readFileSync(join(ROOT, "web/kigen.bin"));
const model = await K.loadModel(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const base = join(ROOT, "datasets/Products-Real/evaluation");
const ann = JSON.parse(readFileSync(join(base, "annotations.json"), "utf8"));

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
  return { img: { data: out, width: nw, height: nh }, x, y, sx: nw / w, sy: nh / h };
}

// Axis-aligned guide of fw x fh centred on (cx, cy) over a label rotated by deg around that point
// (2x2 supersampled bilinear, edges clamped), resized so the height is at most maxH.
function grabRot(img, cx, cy, fw, fh, deg, maxH = 160) {
  const s = Math.min(1, maxH / fh), nw = Math.max(8, Math.round(fw * s)), nh = Math.max(8, Math.round(fh * s));
  const c = Math.cos(deg * Math.PI / 180), sn = Math.sin(deg * Math.PI / 180), out = new Uint8ClampedArray(nw * nh * 4);
  const px = (x, y, k) => img.data[(Math.min(img.height - 1, Math.max(0, y)) * img.width + Math.min(img.width - 1, Math.max(0, x))) * 4 + k];
  for (let j = 0; j < nh; j++)
    for (let i = 0; i < nw; i++) {
      const acc = [0, 0, 0];
      for (const [di, dj] of [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]) {
        const u = (i + di) / s - fw / 2, v = (j + dj) / s - fh / 2;
        const sx = cx + u * c - v * sn, sy = cy + u * sn + v * c, x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
        for (let k = 0; k < 3; k++)
          acc[k] += (px(x0, y0, k) * (1 - fx) + px(x0 + 1, y0, k) * fx) * (1 - fy) + (px(x0, y0 + 1, k) * (1 - fx) + px(x0 + 1, y0 + 1, k) * fx) * fy;
      }
      const o = (j * nw + i) * 4; out[o] = acc[0] / 4; out[o + 1] = acc[1] / 4; out[o + 2] = acc[2] / 4; out[o + 3] = 255;
    }
  return { img: { data: out, width: nw, height: nh }, x: 0, y: 0, sx: s, sy: s };
}

const sets = { "4.5:1": [], "7:1": [], "7:1 high": [] };
let n = 0, skip = Number(process.env.OFFSET || 0);
for (const [file, meta] of Object.entries(ann)) {
  const a = (meta.ann || []).find((a) => a.cls === "exp" && a.transcription);
  if (!a) continue;
  const raw = jpeg.decode(readFileSync(join(base, "images", file)), { useTArray: true, formatAsRGBA: true });
  let [x0, y0, x1, y1] = a.bbox;
  if (x1 < x0) [x0, x1] = [x1, x0];
  if (y1 < y0) [y0, y1] = [y1, y0];
  const bh = y1 - y0, bw = x1 - x0, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  if (bh < 4 || bw < 8) continue;
  if (skip-- > 0) continue;
  const gt = K.parseDate(a.transcription, { strict: false })?.iso ?? null;
  const add = (k, fx, fy, fw, fh) => sets[k].push({ ...grab(raw, fx, fy, fw, fh), box: [x0, y0, x1, y1], gt, file });
  (sets.tight ??= []).push({ ...grab(raw, x0 - bh * 0.1, y0 - bh * 0.1, bw + bh * 0.2, bh * 1.2), box: [x0, y0, x1, y1], gt, file });
  const fh = bh * 2.2, fw = Math.max(fh * 4.5, bw * 1.15);
  add("4.5:1", cx - fw / 2, cy - fh / 2, fw, fh);
  const th = bh * 1.5, tw = Math.max(th * 7, bw * 1.15);
  add("7:1", cx - tw / 2, cy - th / 2, tw, th);
  add("7:1 high", cx - tw / 2, cy - th / 2 - bh * 0.15, tw, th);
  // phone held farther away: the date fills only 1/2.5 or 1/3.5 of the guide's height
  for (const k of [2.5, 3.5]) {
    const fh2 = bh * k, fw2 = fh2 * 7;
    (sets[`7:1 far x${k}`] ??= []).push({ ...grab(raw, cx - fw2 / 2, cy - fh2 / 2, fw2, fh2), box: [x0, y0, x1, y1], gt, file });
  }
  // ROT=1: label tilted against the guide (box: not meaningful for these, only right/wrong)
  if (process.env.ROT) {
    const r = (k, fw, fh, deg) => (sets[`rot ${k} ${deg}°`] ??= []).push({ ...grabRot(raw, cx, cy, fw, fh, deg), box: [0, 0, 0, 0], gt, file });
    for (const d of [2, 5]) r("7:1", tw, th, d);
    for (const d of [2, 5, 10]) r("far2.5", bh * 2.5 * 7, bh * 2.5, d);
    for (const d of [15, 30]) r("2:1", bw * 1.6, bw * 0.8, d);
    if (process.env.ROT === "only") for (const k of Object.keys(sets)) if (!k.startsWith("rot")) sets[k] = [];
  }
  if (++n >= Number(process.env.N || 150)) break;
}

const pct2 = (v) => (v * 100).toFixed(0).padStart(3) + "%";
// NEWWRONG=<set>: frames that focus turned wrong (were right or empty without it)
if (process.env.NEWWRONG) {
  for (const f of sets[process.env.NEWWRONG]) {
    const a = K.readRegion(model, f.img, { strips: false }), b = K.readRegion(model, f.img);
    if (!b.date || b.date.iso === f.gt || (a.date && a.date.iso !== f.gt)) continue;
    console.log(`want ${f.gt}  before ${a.date?.iso ?? "-"} "${a.text}"  focus ${b.date.iso} "${b.text}" band ${b.band} cols ${b.cols} agree ${b.agree}\n    ${b.candidates.map((c) => `[${c.band}] "${c.text}"`).join("  ")}`);
  }
  process.exit(0);
}
if (process.env.CMP) {
  for (const [name, frames] of Object.entries(sets)) {
    if (!frames.length) continue;
    const line = [];
    for (const [label, opts] of [["strips off", { strips: false }], ["strips", {}]]) {
      let right = 0, wrong = 0, t = performance.now();
      for (const f of frames) { const r = K.readRegion(model, f.img, opts); if (r.date) r.date.iso === f.gt ? right++ : wrong++; }
      const n = frames.length;
      line.push(`${label} ${pct2(right / n)} / ${((wrong / n) * 100).toFixed(1).padStart(4)}%  ${((performance.now() - t) / n).toFixed(0)}ms`);
    }
    console.log(`${name.padEnd(13)} right/wrong  ${line.join("   |   ")}`);
  }
  process.exit(0);
}

const ov = (a0, a1, b0, b1) => Math.max(0, Math.min(a1, b1) - Math.max(a0, b0));
const pct = (v) => (v * 100).toFixed(0).padStart(3) + "%";

// NOSNUG=<set>: frames with no candidate band snug around the date: what textLines found, and the
// row edge profile over the full width vs over just the date's columns (sparkline, top -> bottom)
if (process.env.NOSNUG) {
  const bars = " .:-=+*#%@";
  const prof = (g, w, y0, y1, x0, x1) => {
    const e = [];
    for (let y = y0; y < y1; y++) { let s = 0; for (let x = Math.max(1, x0); x < x1; x++) s += Math.abs(g[y * w + x] - g[y * w + x - 1]); e.push(s); }
    const mx = Math.max(...e) || 1;
    return e.map((v) => bars[Math.min(9, Math.floor(v / mx * 9.99))]).join("");
  };
  let shown = 0;
  for (const f of sets[process.env.NOSNUG]) {
    const r = K.readRegion(model, f.img, { recheck: false });
    if (Math.abs(r.skew) >= 3 || r.turn) continue;
    const [g0, g1] = [(f.box[1] - f.y) * f.sy, (f.box[3] - f.y) * f.sy].map(Math.round);
    const [h0, h1] = [(f.box[0] - f.x) * f.sx, (f.box[2] - f.x) * f.sx].map(Math.round);
    const snug = ([a, b]) => ov(a, b, g0, g1) >= 0.95 * (g1 - g0) && b - a <= 1.7 * (g1 - g0);
    if (r.candidates.some((x) => snug(x.band))) continue;
    const g = K.toGray(f.img), w = f.img.width, h = f.img.height;
    const mark = Array.from({ length: h }, (_, y) => (y >= g0 && y < g1 ? "D" : " ")).join("");
    const lin = Array.from({ length: h }, (_, y) => { const i = r.lines.findIndex(([a, b]) => y >= a && y < b); return i < 0 ? " " : String(i); }).join("");
    console.log(`want ${f.gt} got ${r.date?.iso ?? "-"} h=${h} date rows ${g0}-${g1} cols ${h0}-${h1} of ${w}  lines ${JSON.stringify(r.lines)}`);
    console.log(`  full  |${prof(g, w, 0, h, 0, w)}|\n  date  |${prof(g, w, 0, h, h0, h1)}|\n  truth |${mark}|\n  lines |${lin}|`);
    if (++shown >= 14) break;
  }
  process.exit(0);
}

// TIGHT=1: was a candidate band snug around the date (covers >= 95% of it, at most 1.7x its height)?
if (process.env.TIGHT) {
  for (const [name, frames] of Object.entries(sets)) {
    const c = {};
    for (const f of frames) {
      const r = K.readRegion(model, f.img, { recheck: false });
      if (Math.abs(r.skew) >= 3 || r.turn) continue;
      const [g0, g1] = [(f.box[1] - f.y) * f.sy, (f.box[3] - f.y) * f.sy];
      const snug = ([a, b]) => ov(a, b, g0, g1) >= 0.95 * (g1 - g0) && b - a <= 1.7 * (g1 - g0);
      const out = r.date ? (r.date.iso === f.gt ? "right" : "wrong") : "none";
      const k = `${out}/${r.candidates.some((x) => snug(x.band)) ? (snug(r.band) ? "picked snug" : "snug exists, not picked") : "no snug band"}`;
      c[k] = (c[k] || 0) + 1;
      if (snug(r.band)) { // horizontal: cols width / date width
        const [h0, h1] = [(f.box[0] - f.x) * f.sx, (f.box[2] - f.x) * f.sx];
        (c["~w " + out] ??= []).push((r.cols[1] - r.cols[0]) / (h1 - h0));
        if (process.env.SHOWW && out !== "right") console.log(`   ${out} want ${f.gt} "${r.text}" cols ${r.cols.map(Math.round)} date ${[h0, h1].map(Math.round)} of ${f.img.width}`);
      }
    }
    const ws = Object.entries(c).filter(([k]) => k.startsWith("~w")).map(([k, v]) => {
      v.sort((a, b) => a - b); return `${k.slice(3)}: cols/date width median ${v[v.length >> 1].toFixed(2)}, >1.5x ${v.filter((x) => x > 1.5).length}/${v.length}`;
    });
    console.log(`${name.padEnd(13)} ` + Object.entries(c).filter(([k]) => !k.startsWith("~")).sort().map(([k, v]) => `${k} ${v}`).join("  |  ") + "\n              " + ws.join("  |  "));
  }
  process.exit(0);
}
for (const [name, frames] of Object.entries(sets)) {
  const rows = [];
  for (const f of frames) {
    const r = K.readRegion(model, f.img);
    if (Math.abs(r.skew) >= 3 || r.turn) continue;
    const [g0, g1] = [(f.box[1] - f.y) * f.sy, (f.box[3] - f.y) * f.sy];
    const [h0, h1] = [(f.box[0] - f.x) * f.sx, (f.box[2] - f.x) * f.sx];
    const [b0, b1] = r.band, [c0, c1] = r.cols;
    rows.push({
      out: r.date ? (r.date.iso === f.gt ? "right" : "wrong") : "none",
      vCov: ov(b0, b1, g0, g1) / (g1 - g0),      // share of the date's height inside the band
      vTall: (b1 - b0) / (g1 - g0),              // band height / date height (1 = tight)
      hCov: ov(c0, c1, h0, h1) / (h1 - h0),      // share of the date's width inside the cols
      whole: b0 === 0 && b1 === f.img.height,     // read the whole box, no line picked
      nLines: r.lines.length, file: f.file, text: r.text, gt: f.gt, gy: [g0, g1].map(Math.round), h: f.img.height,
      rej: r.rejected?.reason, cands: r.candidates.map((c) => `[${c.band}] "${c.text}" ${c.conf.toFixed(2)}`),
    });
  }
  console.log(`\n== ${name}  (${rows.length} frames without rotation)`);
  for (const out of ["right", "wrong", "none"]) {
    const s = rows.filter((r) => r.out === out);
    if (!s.length) continue;
    const m = (k) => s.reduce((a, r) => a + r[k], 0) / s.length;
    const cutV = s.filter((r) => r.vCov < 0.9).length, cutH = s.filter((r) => r.hCov < 0.95).length;
    const fat = s.filter((r) => r.vTall > 1.8).length, whole = s.filter((r) => r.whole).length;
    console.log(`  ${out.padEnd(5)} n=${String(s.length).padStart(3)}  height in band ${pct(m("vCov"))}  band/date height ${m("vTall").toFixed(2)}`
      + `  width in cols ${pct(m("hCov"))} | cut top/bottom ${cutV}  cut sides ${cutH}  band >1.8x date ${fat}  whole box ${whole}`);
  }
  if (process.env.LIST && name.includes(process.env.LIST)) {
    const none = rows.filter((r) => r.out === "none");
    const rej = {};
    for (const r of none) rej[r.rej ?? "no date"] = (rej[r.rej ?? "no date"] || 0) + 1;
    // would any candidate have parsed to the right date?
    console.log(`    reasons ${JSON.stringify(rej)}; right date in some candidate: ${none.filter((r) => r.cands.some((c) => c.includes(r.gt?.slice(2, 4) ?? "@@"))).length}`);
    for (const r of none.slice(0, 12))
      console.log(`    ${r.rej ?? "no date"} want ${r.gt} date rows ${r.gy}/${r.h} lines ${r.nLines}\n      ${r.cands.join("\n      ")}`);
  }
}
