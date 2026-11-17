// Whole-scene reading: find the expiry-date line anywhere in a camera view that has no guide
// frame (smart glasses, a photo of a whole pack), then read it like a framed box.
//
//   import { readScene } from "./kigen-scene.js";
//   const r = readScene(model, det, img);      // img: RGBA ImageData-like, whole view
//   r.date, r.box (where it was found, image pixels), r.rejected, r.ms
//
// When the platform already finds text lines (Apple Vision on iOS), skip the scan: readLines()
// takes those boxes, or readCrops() takes guide-shaped crops the caller cut itself (guideAround)
// plus the platform's own reading of each line, so a date is confirmed by two engines.
//
// The date-line detector was trained on guide-shaped frames (aspect 2:1 to 8.5:1, the date
// filling 1/1.3 to 1/3.5 of the height). So the view is scanned with windows of that shape at
// several sizes; the strongest boxes are cut out like a camera guide and read with readRegion.
import { toGray, detectDate, readRegion, parseDate } from "./kigen-ocr.js";

const ASPECT = 6;        // window width / height
const BAND_MAX = 0.4;    // largest window height, share of the view's shorter side
const BAND_MIN = 28;     // smallest window height in pixels (a date ~10 px tall)
const BAND_STEP = 0.62;  // next window height
const MIN_SCORE = 0.6;   // detector score worth reading
const TOP = 3;           // boxes read per view
const KEEP = 1;          // share of windows (busiest first) the detector looks at; lower is faster but
                         // misses faint dot / etched dates (0.3: 2-3x faster, found 90% -> 77% on ExpDate)

function cropGray(g, w, x0, y0, cw, ch) {
  const out = new Float32Array(cw * ch);
  for (let y = 0; y < ch; y++) out.set(g.subarray((y0 + y) * w + x0, (y0 + y) * w + x0 + cw), y * cw);
  return out;
}

// RGBA crop resized (box filter) so its height is at most maxH, like a camera guide frame
function cropRGBA(img, x0, y0, cw, ch, maxH = 160) {
  const s = Math.min(1, maxH / ch), nw = Math.max(8, Math.round(cw * s)), nh = Math.max(8, Math.round(ch * s));
  const out = new Uint8ClampedArray(nw * nh * 4), { data, width } = img;
  for (let j = 0; j < nh; j++) {
    const ay = y0 + Math.floor(j * ch / nh), by = Math.max(ay + 1, y0 + Math.floor((j + 1) * ch / nh));
    for (let i = 0; i < nw; i++) {
      const ax = x0 + Math.floor(i * cw / nw), bx = Math.max(ax + 1, x0 + Math.floor((i + 1) * cw / nw));
      let r = 0, gg = 0, b = 0, c = 0;
      for (let v = ay; v < by; v++)
        for (let u = ax; u < bx; u++) { const p = (v * width + u) * 4; r += data[p]; gg += data[p + 1]; b += data[p + 2]; c++; }
      const o = (j * nw + i) * 4;
      out[o] = r / c; out[o + 1] = gg / c; out[o + 2] = b / c; out[o + 3] = 255;
    }
  }
  return { data: out, width: nw, height: nh };
}

const iou = (a, b) => {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy;
  return u > 0 ? ix * iy / u : 0;
};

// Candidate date-line boxes in the whole view: [{box: [x0, y0, x1, y1], score}], strongest first.
export function findDateLines(det, g, w, h, opts = {}) {
  const found = [];
  const bandMin = opts.bandMin ?? BAND_MIN, minScore = opts.minScore ?? MIN_SCORE, keep = opts.keep ?? KEEP;
  // integral image of |horizontal gradient|: text (many vertical strokes) is where it is high, so
  // only the busiest windows of each size go through the detector (it costs ~7 ms a window)
  const I = new Float64Array((w + 1) * (h + 1));
  for (let y = 0; y < h; y++) {
    let row = 0;
    for (let x = 0; x < w; x++) {
      row += x ? Math.abs(g[y * w + x] - g[y * w + x - 1]) : 0;
      I[(y + 1) * (w + 1) + x + 1] = I[y * (w + 1) + x + 1] + row;
    }
  }
  const energy = (x0, y0, cw, ch) =>
    (I[(y0 + ch) * (w + 1) + x0 + cw] - I[y0 * (w + 1) + x0 + cw] - I[(y0 + ch) * (w + 1) + x0] + I[y0 * (w + 1) + x0]) / (cw * ch);
  for (let bh = Math.round(Math.min(w, h) * BAND_MAX); bh >= bandMin; bh = Math.round(bh * BAND_STEP)) {
    const bw = Math.min(w, Math.round(bh * ASPECT));
    const ys = Math.max(1, Math.round(bh / 2)), xs = Math.max(1, Math.round(bw / 2));
    const wins = [];
    for (let y = 0; ; y += ys) {
      const y0 = Math.min(y, h - bh);
      for (let x = 0; ; x += xs) {
        const x0 = Math.min(x, w - bw);
        wins.push([x0, y0, energy(x0, y0, bw, bh)]);
        if (x0 + bw >= w) break;
      }
      if (y0 + bh >= h) break;
    }
    wins.sort((a, b) => b[2] - a[2]);
    for (const [x0, y0] of wins.slice(0, Math.max(2, Math.ceil(wins.length * keep)))) {
      const d = detectDate(det, cropGray(g, w, x0, y0, bw, bh), bw, bh);
      if (d.box && d.score >= minScore) {
        const [a, b, c, e] = d.box;
        // a box touching the window's top or bottom is a date cut by the window: the
        // overlapping window holds it whole
        if (b > 0.5 && e < bh - 0.5) found.push({ box: [x0 + a, y0 + b, x0 + c, y0 + e], score: d.score, band: bh });
      }
    }
  }
  found.sort((p, q) => q.score - p.score);
  const kept = [];
  for (const f of found) if (!kept.some((k) => iou(k.box, f.box) > 0.3)) kept.push(f);
  return kept;
}

// A guide-shaped frame around a line box [x0, y0, x1, y1]: the line filling ~1/2.2 of its height,
// room on both sides (where readRegion expects the user to have framed it).
export function guideAround(box, w, h) {
  const [x0, y0, x1, y1] = box, bh = y1 - y0, bw = x1 - x0;
  const fh = bh * 2.2, fw = Math.max(bw + bh * 2, fh * 4.5), cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const fx0 = Math.max(0, Math.round(cx - fw / 2)), fy0 = Math.max(0, Math.round(cy - fh / 2));
  return [fx0, fy0, Math.min(w, Math.round(cx + fw / 2)) - fx0, Math.min(h, Math.round(cy + fh / 2)) - fy0];
}

// Several dates in one view (a manufacturing date beside the expiry date, a misread lot code):
// guessing which one is the expiry date is how a wrong date gets spoken. So:
//   - reads of the same line (overlapping boxes) count once: the widest read with a date
//   - production-labelled dates never; yearless month-days not here (no framing to trust them)
//   - one date left: that one. Several different dates: none, rejected "several" (the caller can
//     say so and look again), never the latest or the most confident of them
function pickExpiry(reads) {
  const dated = reads.filter((q) => q.r.date && q.r.date.type !== "production" && q.r.date.order !== "MD");
  const area = (b) => (b[2] - b[0]) * (b[3] - b[1]);
  const same = (a, b) => {
    const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
    const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    return ix * iy > 0.5 * Math.min(area(a), area(b));
  };
  const groups = [];
  for (const q of dated) {
    const g = groups.find((g) => same(g.c.box, q.c.box));
    if (!g) groups.push(q);
    else if (q.c.box[2] - q.c.box[0] > g.c.box[2] - g.c.box[0]) groups[groups.indexOf(g)] = q;
  }
  const isos = [...new Set(groups.map((q) => q.r.date.iso))];
  if (isos.length !== 1) return { best: null, several: isos.length > 1 };
  return { best: groups.sort((a, b) => (b.r.dateConf ?? 0) - (a.r.dateConf ?? 0))[0], several: false };
}

// Read line boxes found by anything (the detector scan below, or the platform's own text
// detector): each is cut out like a camera guide and read with readRegion.
// boxes: [{box: [x0, y0, x1, y1], score?, text?}] in img pixels, best first.
export function readLines(model, det, img, boxes, opts = {}) {
  const { width: w, height: h } = img, crops = [];
  for (const c of boxes.slice(0, opts.top ?? TOP)) {
    const [fx, fy, fw, fh] = guideAround(c.box, w, h);
    if (fw >= 8 && fh >= 8) crops.push({ ...c, img: cropRGBA(img, fx, fy, fw, fh) });
  }
  return readCrops(model, det, crops, opts);
}

// crops: [{img, box?, score?, text?}] — img a guide-shaped RGBA crop around one line (guideAround,
// height ~160 px is plenty), text what another OCR engine read on that line (optional).
// result.agree: the other engine's text parses to the same date (two engines, one answer).
export function readCrops(model, det, crops, opts = {}) {
  const t0 = performance.now(), reads = [];
  for (const c of crops) {
    const r = readRegion(model, c.img, det ? { ...opts, det } : opts);
    const other = c.text ? parseDate(c.text, { strict: true, today: opts.today })?.iso ?? null : null;
    reads.push({ c, r, other });
  }
  const { best, several } = pickExpiry(reads);
  const dates = [...new Set(reads.filter((q) => q.r.date).map((q) => q.r.date.iso))];
  return {
    date: best?.r.date ?? null, dateConf: best?.r.dateConf ?? null, text: best?.r.text ?? null,
    box: best?.c.box ?? null, agree: !!best && best.other === best.r.date.iso, dates,
    rejected: best ? null : several ? { reason: "several", dates } : reads.find((q) => q.r.rejected)?.r.rejected ?? null,
    reads: reads.map(({ c, r, other }) => ({ box: c.box ?? null, score: c.score ?? null, text: r.text, date: r.date?.iso ?? null,
      type: r.date?.type ?? null, rejected: r.rejected?.reason ?? null, other })),
    ms: performance.now() - t0,
  };
}

// Read the expiry date anywhere in a whole view, finding the lines with the detector.
// opts go to readRegion (strict, minConf, ...); opts.top: how many boxes to read (default 3).
export function readScene(model, det, img, opts = {}) {
  const t0 = performance.now();
  const boxes = findDateLines(det, toGray(img), img.width, img.height, opts);
  const msFind = performance.now() - t0;
  const r = readLines(model, det, img, boxes, opts);
  return { ...r, boxes, msFind, ms: performance.now() - t0 };
}
