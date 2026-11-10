// kigen-ocr — tiny OCR for Japanese expiry dates (賞味期限 / 消費期限). MIT License.
// No dependencies. Works in browsers and Node (ES module).

// ------------------------------------------------------------------ model
export async function loadModel(src) {
  const buf = src instanceof ArrayBuffer ? src : await (await fetch(src)).arrayBuffer();
  const hl = new DataView(buf).getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
  const base = 4 + hl;
  for (const l of meta.layers) {
    if (!l.q) continue;
    const q = new Int8Array(buf, base + l.q[0], l.q[1]);
    const s = new Float32Array(buf.slice(base + l.s[0], base + l.s[0] + l.s[1] * 4));
    l.b = new Float32Array(buf.slice(base + l.b[0], base + l.b[0] + l.b[1] * 4));
    const per = q.length / s.length;
    l.w = new Float32Array(q.length);
    for (let i = 0; i < q.length; i++) l.w[i] = q[i] * s[(i / per) | 0];
  }
  return meta;
}

// out[O][P] = W[O4][K] * col[K][P4] + b, register-tiled 4x4. W rows padded to O4, col to P4.
function gemm(wp, b, O, K, col, P, P4) {
  const O4 = wp.length / K, out = new Float32Array(O * P);
  for (let o = 0; o < O4; o += 4) {
    const r0 = o * K, r1 = r0 + K, r2 = r1 + K, r3 = r2 + K;
    for (let p = 0; p < P4; p += 4) {
      let a00 = 0, a01 = 0, a02 = 0, a03 = 0, a10 = 0, a11 = 0, a12 = 0, a13 = 0;
      let a20 = 0, a21 = 0, a22 = 0, a23 = 0, a30 = 0, a31 = 0, a32 = 0, a33 = 0;
      for (let k = 0, ci = p; k < K; k++, ci += P4) {
        const c0 = col[ci], c1 = col[ci + 1], c2 = col[ci + 2], c3 = col[ci + 3];
        const w0 = wp[r0 + k], w1 = wp[r1 + k], w2 = wp[r2 + k], w3 = wp[r3 + k];
        a00 += w0 * c0; a01 += w0 * c1; a02 += w0 * c2; a03 += w0 * c3;
        a10 += w1 * c0; a11 += w1 * c1; a12 += w1 * c2; a13 += w1 * c3;
        a20 += w2 * c0; a21 += w2 * c1; a22 += w2 * c2; a23 += w2 * c3;
        a30 += w3 * c0; a31 += w3 * c1; a32 += w3 * c2; a33 += w3 * c3;
      }
      if (o + 3 < O && p + 3 < P) {
        let ob = o * P + p, bo = b[o];
        out[ob] = a00 + bo; out[ob + 1] = a01 + bo; out[ob + 2] = a02 + bo; out[ob + 3] = a03 + bo;
        ob += P; bo = b[o + 1];
        out[ob] = a10 + bo; out[ob + 1] = a11 + bo; out[ob + 2] = a12 + bo; out[ob + 3] = a13 + bo;
        ob += P; bo = b[o + 2];
        out[ob] = a20 + bo; out[ob + 1] = a21 + bo; out[ob + 2] = a22 + bo; out[ob + 3] = a23 + bo;
        ob += P; bo = b[o + 3];
        out[ob] = a30 + bo; out[ob + 1] = a31 + bo; out[ob + 2] = a32 + bo; out[ob + 3] = a33 + bo;
      } else { // edge tile
        const acc = [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33];
        for (let i = 0; i < 4 && o + i < O; i++) {
          const bo = b[o + i], ob = (o + i) * P;
          for (let j = 0; j < 4 && p + j < P; j++) out[ob + p + j] = acc[i * 4 + j] + bo;
        }
      }
    }
  }
  return out;
}

function padW(L, K) { // weights padded to a multiple of 4 output rows (cached)
  if (!L.wp) {
    const O = L.shape[0], O4 = Math.ceil(O / 4) * 4;
    L.wp = new Float32Array(O4 * K);
    L.wp.set(L.w);
  }
  return L.wp;
}

function conv2d(inp, C, H, W, L) {
  const [O, , kh, kw] = L.shape, [ph, pw] = L.pad;
  const Ho = H + 2 * ph - kh + 1, Wo = W + 2 * pw - kw + 1, P = Ho * Wo, P4 = Math.ceil(P / 4) * 4;
  const K = C * kh * kw, col = new Float32Array(K * P4);
  for (let c = 0; c < C; c++)
    for (let i = 0; i < kh; i++)
      for (let j = 0; j < kw; j++) {
        const row = ((c * kh + i) * kw + j) * P4;
        const x0 = Math.max(0, pw - j), x1 = Math.min(Wo, W + pw - j);
        for (let y = 0; y < Ho; y++) {
          const sy = y + i - ph;
          if (sy < 0 || sy >= H) continue;
          const so = (c * H + sy) * W - pw + j, ro = row + y * Wo;
          for (let x = x0; x < x1; x++) col[ro + x] = inp[so + x];
        }
      }
  const out = gemm(padW(L, K), L.b, O, K, col, P, P4);
  for (let i = 0; i < out.length; i++) if (out[i] < 0) out[i] = 0;
  return [out, O, Ho, Wo];
}

function maxpool(a, C, H, W, kh, kw) {
  const Ho = (H / kh) | 0, Wo = (W / kw) | 0, out = new Float32Array(C * Ho * Wo);
  for (let c = 0; c < C; c++)
    for (let y = 0; y < Ho; y++)
      for (let x = 0; x < Wo; x++) {
        let m = -Infinity;
        for (let i = 0; i < kh; i++)
          for (let j = 0; j < kw; j++) {
            const v = a[(c * H + y * kh + i) * W + x * kw + j];
            if (v > m) m = v;
          }
        out[(c * Ho + y) * Wo + x] = m;
      }
  return [out, C, Ho, Wo];
}

function conv1d(a, C, T, L, dil, residual) { // kernel 3 (res) or 1 (fc)
  const [O, , k] = L.shape, T4 = Math.ceil(T / 4) * 4, K = C * k, col = new Float32Array(K * T4);
  for (let c = 0; c < C; c++)
    for (let t = 0; t < k; t++) {
      const off = (t - (k - 1) / 2) * dil, row = (c * k + t) * T4;
      const lo = Math.max(0, -off), hi = Math.min(T, T - off);
      for (let x = lo; x < hi; x++) col[row + x] = a[c * T + x + off];
    }
  const out = gemm(padW(L, K), L.b, O, K, col, T, T4);
  if (residual) for (let i = 0; i < out.length; i++) { const v = out[i] + a[i]; out[i] = v > 0 ? v : 0; }
  return out;
}

// x: Float32Array(32*w) in [0,1] -> {text, conf, probs}
export function recognize(model, x, w) {
  let a = x, C = 1, H = model.h, W = w;
  for (const l of model.layers) {
    if (l.op === "conv2d") [a, C, H, W] = conv2d(a, C, H, W, l);
    else if (l.op === "maxpool") [a, C, H, W] = maxpool(a, C, H, W, l.k[0], l.k[1]);
    else if (l.op === "res1d") a = conv1d(a, C, W, l, l.dil, true);
    else { a = conv1d(a, C, W, l, 1, false); C = l.shape[0]; }
  }
  // a: (V,T) logits -> greedy CTC
  const V = C, T = W, vocab = model.vocab;
  let s = "", prev = 0, confSum = 0, n = 0, minConf = 1;
  for (let t = 0; t < T; t++) {
    let best = 0, bv = -Infinity, sum = 0;
    for (let v = 0; v < V; v++) if (a[v * T + t] > bv) { bv = a[v * T + t]; best = v; }
    for (let v = 0; v < V; v++) sum += Math.exp(a[v * T + t] - bv);
    const p = 1 / sum;
    if (best !== 0 && best !== prev) { s += vocab[best]; confSum += p; n++; if (p < minConf) minConf = p; }
    prev = best;
  }
  return { text: s.split(/\s+/).filter(Boolean).join(" "), conf: n ? confSum / n : 0, minConf: n ? minConf : 0 };
}

// ------------------------------------------------------------------ preprocessing
// RGBA (ImageData-like) -> gray Float32Array (0..255)
export function toGray(img) {
  const { data, width, height } = img, g = new Float32Array(width * height);
  for (let i = 0; i < g.length; i++) g[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
  return g;
}

function resizeArea(g, w, h, nw, nh) { // box-filter resize (≈ cv2.INTER_AREA)
  const tmp = new Float32Array(nw * h), out = new Float32Array(nw * nh);
  const sx = w / nw, sy = h / nh;
  for (let x = 0; x < nw; x++) {
    const x0 = x * sx, x1 = x0 + sx;
    for (let y = 0; y < h; y++) {
      let acc = 0;
      for (let u = Math.floor(x0); u < Math.ceil(x1) && u < w; u++)
        acc += g[y * w + u] * (Math.min(x1, u + 1) - Math.max(x0, u));
      tmp[y * nw + x] = acc / sx;
    }
  }
  for (let y = 0; y < nh; y++) {
    const y0 = y * sy, y1 = y0 + sy;
    for (let x = 0; x < nw; x++) {
      let acc = 0;
      for (let v = Math.floor(y0); v < Math.ceil(y1) && v < h; v++)
        acc += tmp[v * nw + x] * (Math.min(y1, v + 1) - Math.max(y0, v));
      out[y * nw + x] = acc / sy;
    }
  }
  return out;
}

// gray crop (w x h) -> {x, w} model input; matches gen.to_input
export function toInput(g, w, h, model) {
  const H = model.h, nw = Math.min(model.maxw, Math.max(8, Math.round(w * H / h)));
  const r = resizeArea(g, w, h, nw, H);
  const s = Float32Array.from(r).sort();
  const lo = s[Math.floor(0.02 * (s.length - 1))], hi = s[Math.floor(0.98 * (s.length - 1))];
  const d = Math.max(hi - lo, 1);
  const W4 = Math.ceil(nw / 4) * 4, x = new Float32Array(H * W4);
  for (let y = 0; y < H; y++)
    for (let i = 0; i < W4; i++) {
      const v = (r[y * nw + Math.min(i, nw - 1)] - lo) / d;
      x[y * W4 + i] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
  return { x, w: W4 };
}

// Find the text band inside a user-framed box (rows with many vertical edges).
function textBand(g, w, h) {
  const e = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    let s = 0;
    for (let x = 1; x < w; x++) s += Math.abs(g[y * w + x] - g[y * w + x - 1]);
    e[y] = s;
  }
  const k = Math.max(1, Math.round(h / 30)), sm = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    let s = 0, n = 0;
    for (let d = -k; d <= k; d++) if (y + d >= 0 && y + d < h) { s += e[y + d]; n++; }
    sm[y] = s / n;
  }
  let my = 0;
  for (let y = 1; y < h; y++) if (sm[y] > sm[my]) my = y;
  const sorted = Float32Array.from(sm).sort(), floor = sorted[Math.floor(h * 0.2)];
  const th = floor + (sm[my] - floor) * 0.3;
  let y0 = my, y1 = my;
  while (y0 > 0 && sm[y0 - 1] > th) y0--;
  while (y1 < h - 1 && sm[y1 + 1] > th) y1++;
  const m = Math.round((y1 - y0 + 1) * 0.25);
  return [Math.max(0, y0 - m), Math.min(h, y1 + 1 + m)];
}

// Split a framed box into text lines (e.g. a 賞味期限 line sitting right above the date).
// Rows with many vertical edges form runs; a run is split again at a deep valley so
// lines with almost no gap between them are still separated. Returns [y0, y1] bands, top first.
function textLines(g, w, h) {
  const e = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    let s = 0;
    for (let x = 1; x < w; x++) s += Math.abs(g[y * w + x] - g[y * w + x - 1]);
    e[y] = s;
  }
  const smooth = (k) => {
    const o = new Float32Array(h);
    for (let y = 0; y < h; y++) {
      let s = 0, n = 0;
      for (let d = -k; d <= k; d++) if (y + d >= 0 && y + d < h) { s += e[y + d]; n++; }
      o[y] = s / n;
    }
    return o;
  };
  const sm = smooth(Math.max(1, Math.round(h / 60)));
  const sorted = Float32Array.from(sm).sort(), floor = sorted[Math.floor(h * 0.2)];
  let peak = 0;
  for (let y = 0; y < h; y++) if (sm[y] > peak) peak = sm[y];
  const th = floor + (peak - floor) * 0.25;
  let runs = [];
  for (let y = 0; y < h; y++) {
    if (sm[y] <= th) continue;
    let y1 = y;
    while (y1 + 1 < h && sm[y1 + 1] > th) y1++;
    runs.push([y, y1 + 1]);
    y = y1;
  }
  // re-join pieces of one line (dot-matrix rows, thin strokes)
  for (let i = 1; i < runs.length; i++) {
    const [a0, a1] = runs[i - 1], [b0, b1] = runs[i];
    if (b0 - a1 < 0.35 * Math.min(a1 - a0, b1 - b0) || b0 - a1 <= 1) { runs[i - 1] = [a0, b1]; runs.splice(i--, 1); }
  }
  // split a run at a deep valley (two lines with ~no gap)
  const fine = smooth(1), out = [];
  for (const [y0, y1] of runs) {
    const H = y1 - y0;
    let my = -1;
    if (H >= 12)
      for (let y = y0 + Math.round(H * 0.35); y < y1 - Math.round(H * 0.35); y++) if (my < 0 || fine[y] < fine[my]) my = y;
    if (my > 0) {
      let la = 0, lb = 0;
      for (let y = y0; y < my; y++) la = Math.max(la, fine[y]);
      for (let y = my; y < y1; y++) lb = Math.max(lb, fine[y]);
      if (fine[my] < 0.6 * Math.min(la, lb)) { out.push([y0, my], [my, y1]); continue; }
    }
    out.push([y0, y1]);
  }
  // drop specks: too thin or too little edge mass compared to the strongest line
  const mass = out.map(([a, b]) => { let s = 0; for (let y = a; y < b; y++) s += e[y]; return s; });
  const maxH = Math.max(0, ...out.map(([a, b]) => b - a)), maxM = Math.max(0, ...mass);
  runs = out.filter(([a, b], i) => b - a >= Math.max(4, 0.3 * maxH) && mass[i] >= 0.15 * maxM);
  // pad each line, but never into its neighbour
  return runs.slice(0, 4).map(([a, b], i) => {
    const m = Math.round((b - a) * 0.25);
    const lo = i ? Math.ceil((runs[i - 1][1] + a) / 2) : 0, hi = i + 1 < runs.length ? Math.floor((b + runs[i + 1][0]) / 2) : h;
    return [Math.max(lo, a - m), Math.min(hi, b + m)];
  });
}

// Horizontal ink extent of rows y0..y1, padded by 0.5 x band height. The model was trained
// with side margins up to ~1.5 x text height; wide blank areas make it hallucinate ("賞味期限" -> "2").
function inkCols(g, w, y0, y1) {
  const col = new Float32Array(w);
  for (let y = y0; y < y1; y++) for (let x = 1; x < w; x++) col[x] += Math.abs(g[y * w + x] - g[y * w + x - 1]);
  const k = Math.max(1, Math.round((y1 - y0) / 8)), sm = new Float32Array(w);
  let mx = 0;
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let d = -k; d <= k; d++) if (x + d >= 0 && x + d < w) s += col[x + d];
    sm[x] = s;
    if (s > mx) mx = s;
  }
  let x0 = 0, x1 = w - 1;
  while (x0 < w - 1 && sm[x0] < mx * 0.15) x0++;
  while (x1 > x0 && sm[x1] < mx * 0.15) x1--;
  const pad = Math.round((y1 - y0) * 0.5);
  return [Math.max(0, x0 - pad), Math.min(w, x1 + 1 + pad)];
}

// Ink clusters along a band (columns with edges), split at gaps wider than 0.25 x band height.
function inkClusters(g, w, y0, y1, x0, x1) {
  const H = y1 - y0, col = new Float32Array(x1 - x0);
  let mx = 0;
  for (let x = x0 + 1; x < x1; x++) {
    let s = 0;
    for (let y = y0; y < y1; y++) s += Math.abs(g[y * w + x] - g[y * w + x - 1]);
    col[x - x0] = s;
    if (s > mx) mx = s;
  }
  const on = (i) => col[i] > mx * 0.12, out = [];
  for (let i = 0; i < col.length; i++) {
    if (!on(i)) continue;
    let j = i;
    while (j + 1 < col.length && on(j + 1)) j++;
    const a = x0 + i, b = x0 + j + 1;
    if (out.length && a - out[out.length - 1][1] < 0.25 * H) out[out.length - 1][1] = b; else out.push([a, b]);
    i = j;
  }
  return out;
}

function crop(g, w, y0, y1, x0 = 0, x1 = w) {
  if (x0 === 0 && x1 === w) return g.subarray(y0 * w, y1 * w);
  const cw = x1 - x0, out = new Float32Array(cw * (y1 - y0));
  for (let y = y0; y < y1; y++) out.set(g.subarray(y * w + x0, y * w + x1), (y - y0) * cw);
  return out;
}

function kindOf(text) {
  const s = (text || "").normalize("NFKC");
  return /消費|USE ?BY/.test(s) ? "消費期限" : /賞味|EXP|BB|BEST/.test(s) ? "賞味期限" : null;
}

// Skew of the text in degrees (-90..90): the angle at which strong-edge points, projected
// onto the vertical axis, pile up into the sharpest row histogram (text lines horizontal).
const SKEW_MAX = 90; // line direction is defined modulo 180°; up/down is settled by readRegion
export function estimateSkew(g, w, h) {
  const mag = [];
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x, m = Math.abs(g[i + 1] - g[i - 1]) + Math.abs(g[i + w] - g[i - w]);
      if (m > 0) mag.push(m);
    }
  if (mag.length < 50) return 0;
  const th = Float32Array.from(mag).sort()[Math.floor(mag.length * 0.85)];
  const step = Math.max(1, Math.round(Math.sqrt(mag.length * 0.15 / 4000)));
  const px = [], py = [], cx = w / 2, cy = h / 2;
  for (let y = 1; y < h - 1; y += step)
    for (let x = 1; x < w - 1; x += step) {
      const i = y * w + x, m = Math.abs(g[i + 1] - g[i - 1]) + Math.abs(g[i + w] - g[i - w]);
      if (m >= th) { px.push(x - cx); py.push(y - cy); }
    }
  const R = Math.ceil(Math.hypot(w, h) / 2) + 1, hist = new Float32Array(2 * R + 1);
  const sharp = (deg) => {
    const a = deg * Math.PI / 180, s = Math.sin(a), c = Math.cos(a);
    hist.fill(0);
    for (let k = 0; k < px.length; k++) hist[Math.round(-px[k] * s + py[k] * c) + R]++;
    let q = 0;
    for (let k = 0; k < hist.length; k++) q += hist[k] * hist[k];
    return q;
  };
  let best = 0, bq = sharp(0);
  for (let d = -SKEW_MAX; d <= SKEW_MAX; d += 3) { const q = sharp(d); if (q > bq * 1.02) { bq = q; best = d; } }
  const coarse = best;
  for (let d = coarse - 2.5; d <= coarse + 2.5; d += 0.5) { const q = sharp(d); if (q > bq) { bq = q; best = d; } }
  return best;
}

// Rotate a gray image so that a line at `deg` becomes horizontal (canvas grows, edges replicated;
// a flat median fill was tried and read slightly worse on tilted product photos).
function rotateGray(g, w, h, deg) {
  const a = deg * Math.PI / 180, s = Math.sin(a), c = Math.cos(a);
  const nw = Math.ceil(w * Math.abs(c) + h * Math.abs(s)), nh = Math.ceil(w * Math.abs(s) + h * Math.abs(c));
  const out = new Float32Array(nw * nh), cx = w / 2, cy = h / 2, ncx = nw / 2, ncy = nh / 2;
  for (let y = 0; y < nh; y++)
    for (let x = 0; x < nw; x++) {
      const u = x - ncx, v = y - ncy;
      let sx = u * c - v * s + cx - 0.5, sy = u * s + v * c + cy - 0.5;
      sx = Math.min(w - 1, Math.max(0, sx)); sy = Math.min(h - 1, Math.max(0, sy));
      const x0 = Math.floor(sx), y0 = Math.floor(sy), x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1);
      const fx = sx - x0, fy = sy - y0;
      out[y * nw + x] = (g[y0 * w + x0] * (1 - fx) + g[y0 * w + x1] * fx) * (1 - fy)
        + (g[y1 * w + x0] * (1 - fx) + g[y1 * w + x1] * fx) * fy;
    }
  return { g: out, w: nw, h: nh };
}

// Read a framed region (RGBA ImageData-like). Straightens tilted text first (any angle),
// then tries the whole box, the detected text band and, when the box holds several lines,
// each line on its own. If the date line has no 賞味/消費 label, the label is taken from a
// neighbouring line. If nothing parses as a date, the box may be upside down or sideways:
// retry at 180° (and ±90° when the box is taller than wide) — extra time only on failed reads.
// Result: {text, conf, date, band, cols, input, lines, skew, turn, candidates}
// A read is "weak" when it may come from text seen sideways/upside down: no date, a yearless
// month-day ("2/8" is easy to hallucinate), or low confidence. Weak reads trigger the turn retry;
// a turned read replaces the original only if it is not weak itself and scores higher.
const weak = (r) => !r.date || r.date.order === "MD" || r.conf < 0.6;

// opts.strict (default true): drop readings OCR noise produces easily — broken years
// ("22027/03/31"), yearless "2/8", 2-digit-year "03/31" — and results below opts.minConf.
// A dropped date is kept in `rejected` ({date, reason}) so the UI can ask for a retake.
export function readRegion(model, img, opts = {}) {
  const o = { strict: opts.strict ?? true, minConf: opts.minConf ?? MIN_CONF };
  const best = readTurns(model, img, o);
  if (o.strict && best.date && best.conf < o.minConf) {
    best.rejected = { date: best.date, reason: "lowconf" };
    best.date = null;
  }
  // a digit misread with high confidence ("10.08" -> "10.03") rarely survives a second look
  if (o.strict && best.date && (opts.recheck ?? RECHECK)) {
    const again = best.recheck();
    if (again?.iso !== best.date.iso) { best.rejected = { date: best.date, reason: "unstable", again: again?.iso ?? null }; best.date = null; }
  }
  delete best.recheck;
  return best;
}

// mean char confidence cut: on ExpDate photos and wide-reticle crops it removes ~20% of the
// wrong dates for ~0.3% of the right ones (min-char confidence cost more right ones)
const MIN_CONF = 0.8;
// second look on by default: on ExpDate photos it cut wrong dates 4.1% -> 1.2% for 1.8% fewer
// right ones. In camera mode a missed frame is cheap (the next frame retries), a wrong date is not.
const RECHECK = true;

function readTurns(model, img, o) {
  const g = toGray(img), w = img.width, h = img.height;
  const skew = estimateSkew(g, w, h);
  const first = readGray(model, g, w, h, skew, o);
  if (!weak(first)) return first;
  let best = first;
  // the line direction is only known modulo 180°: try the other way up, then (tall box) ±90°.
  // Turns are taken relative to the estimated skew — re-estimating on a turned image can land
  // on the same upside-down solution again (+90° and -90° look alike).
  const turns = [180, ...(h > w * 0.8 ? [90, 270] : [])];
  for (const t of turns) {
    const alt = readGray(model, g, w, h, skew + t, o);
    alt.turn = t;
    if (!weak(alt) ? (weak(best) || alt.score > best.score) : alt.date && !best.date) best = alt;
  }
  return best;
}

// Read with the text straightened by `angle` degrees (any angle; |angle| < 3 is left as is).
function readGray(model, g, w, h, angle, o) {
  const skew = ((angle % 360) + 540) % 360 - 180; // -> [-180, 180)
  if (Math.abs(skew) >= 3) ({ g, w, h } = rotateGray(g, w, h, skew));
  const lines = textLines(g, w, h);
  const cands = [[0, h]];
  const [b0, b1] = textBand(g, w, h);
  if (b1 - b0 < h * 0.85) cands.push([b0, b1]);
  if (lines.length > 1)
    for (const l of lines) if (!cands.some(([a, b]) => Math.abs(a - l[0]) + Math.abs(b - l[1]) < 0.15 * (l[1] - l[0]))) cands.push(l);
  let best = null;
  const reads = [];
  const read = (y0, y1, x0, x1) => {
    const inp = toInput(crop(g, w, y0, y1, x0, x1), x1 - x0, y1 - y0, model);
    const r = recognize(model, inp.x, inp.w);
    r.date = parseDate(r.text, { strict: o.strict });
    r.band = [y0, y1];
    r.cols = [x0, x1];
    r.input = inp;
    reads.push(r);
    // a bare label line ("賞味期限") is never the answer
    r.score = r.conf + (r.date ? 0.5 : 0) - (/\d/.test(r.text) ? 0 : 0.5);
    if (!best || r.score > best.score) best = { ...r };
    return r;
  };
  // (stopping at the first confident date was tried: 10-18% faster, but more wrong dates and
  // fewer 賞味/消費 labels on two-line boxes, so every candidate is read)
  for (const [y0, y1] of cands) read(y0, y1, ...inkCols(g, w, y0, y1));
  // Junk at a line's ends (label border, cut neighbour glyphs, lot code) turns "2022.09.10"
  // into "22022.09.10" / "2021.11.17I". If no solid date came out, re-read the most confident
  // lines without their outermost ink cluster. Cutting a real part ("25 09 2021" -> "25 09")
  // leaves a yearless/broken date, which strict parsing drops, so the right variant wins.
  if (weak(best))
    for (const r of [...reads].filter((r) => /\d{2}/.test(r.text)).sort((p, q) => q.conf - p.conf).slice(0, 2)) {
      const [y0, y1] = r.band, cl = inkClusters(g, w, y0, y1, ...r.cols);
      if (cl.length < 2) continue;
      const pad = Math.round((y1 - y0) * 0.3), [x0, x1] = r.cols;
      const L = Math.max(x0, cl[1][0] - pad), R = Math.min(x1, cl[cl.length - 2][1] + pad);
      read(y0, y1, L, x1);
      read(y0, y1, x0, R);
      if (cl.length >= 3 && R - L > (y1 - y0)) read(y0, y1, L, R);
      if (!weak(best)) break;
    }
  // label from a neighbouring line; one telltale kanji is enough there ("2費期", "2味:")
  if (best.date && !best.date.kind) {
    // reference = the single line the date was read from (best may be the whole box)
    const isLine = (r) => lines.length > 1 && lines.some(([a, b]) => Math.abs(a - r.band[0]) + Math.abs(b - r.band[1]) < 0.15 * (b - a));
    const ref = reads.filter((r) => r.date && isLine(r)).sort((p, q) => q.conf - p.conf)[0] ?? best;
    for (const r of reads) {
      if (r === ref || !isLine(r)) continue; // only a separately split line can be the label line
      // a real label line sits beside the date line, about as tall, with ~no digits;
      // halves of a wrongly split date line overlap it or are much shorter
      const [a, b] = r.band, [c, d] = ref.band;
      if (Math.min(b, d) - Math.max(a, c) > 0.3 * (b - a) || b - a < 0.6 * (d - c)) continue;
      if ((r.text.match(/\d/g) || []).length > 2) continue;
      const k = kindOf(r.text) ?? (/[消費]/.test(r.text) ? "消費期限" : /[賞味]/.test(r.text) ? "賞味期限" : null);
      if (k) { best.date = { ...best.date, kind: k }; break; }
    }
  }
  best.lines = lines;
  best.skew = skew;
  best.turn = 0;
  // second look at the chosen line with slightly different vertical framing (for readRegion)
  // (taller framing; rows beyond the image are edge rows repeated — shrinking cut glyphs)
  const [vy0, vy1] = best.band, [vx0, vx1] = best.cols, e = Math.max(1, Math.round((vy1 - vy0) * 0.12));
  best.recheck = () => {
    const cw = vx1 - vx0, nh = vy1 - vy0 + 2 * e, buf = new Float32Array(cw * nh);
    for (let y = 0; y < nh; y++) {
      const sy = Math.min(h - 1, Math.max(0, vy0 - e + y));
      buf.set(g.subarray(sy * w + vx0, sy * w + vx1), y * cw);
    }
    const inp = toInput(buf, cw, nh, model);
    return parseDate(recognize(model, inp.x, inp.w).text, { strict: o.strict });
  };
  best.candidates = reads.map(({ text, conf, band }) => ({ text, conf, band }));
  return best;
}

// ------------------------------------------------------------------ parsing
const KNUM = { "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9 };

function kanjiToDigits(s) { // 二〇二五年十二月三十一日 -> 2025年12月31日
  return s.replace(/[〇一二三四五六七八九十]+/g, (m) => {
    if (!m.includes("十")) return [...m].map((c) => KNUM[c]).join("");
    const [a, b] = m.split("十");
    return String((a ? KNUM[a] : 1) * 10 + (b ? KNUM[b] : 0));
  });
}

// English month names -> "#MM#" marker. OCR slips (AU6, EEB, PPR) are accepted when exactly
// one month is one character away; digits only count as their look-alike letter.
const MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];
const LOOK = { 0: "O", 1: "I", 2: "Z", 5: "S", 6: "G", 8: "B" };
function markMonths(s) {
  let out = "", i = 0;
  const isL = (c) => c >= "A" && c <= "Z";
  while (i < s.length) {
    const t = s.slice(i, i + 3);
    if (t.length === 3 && !isL(s[i - 1] ?? "") && !isL(s[i + 3] ?? "") && (t.match(/[A-Z]/g) || []).length >= 2) {
      let mi = MONTHS.indexOf(t);
      if (mi < 0) {
        const u = [...t].map((c) => LOOK[c] ?? c).join("");
        if (/^[A-Z]{3}$/.test(u)) {
          const near = MONTHS.map((m, k) => [k, [...m].filter((c, j) => c !== u[j]).length]).filter(([, d]) => d <= 1);
          if (near.length === 1) mi = near[0][0];
        }
      }
      if (mi >= 0) { out += `#${pad2(mi + 1)}#`; i += 3; continue; }
    }
    out += s[i++];
  }
  return out;
}

// one-kanji forms (令7.12.31 / 平31.4.30 / 昭64.1.7) are common abbreviations on labels
const ERA = { R: 2018, H: 1988, S: 1925, "令和": 2018, "平成": 1988, "昭和": 1925, "令": 2018, "平": 1988, "昭": 1925 };
const pad2 = (n) => String(n).padStart(2, "0");

function mkDate(y, m, d) {
  if (y < 100) y += 2000;
  if (m < 1 || m > 12 || y < 1950 || y > 2099) return null;
  if (d == null) return { y, m };
  const dim = new Date(y, m, 0).getDate();
  if (d < 1 || d > dim) return null;
  return { y, m, d };
}

// "賞味期限 R7.12.31 A3" -> {kind:"賞味期限", iso:"2025-12-31", precision:"day", ...}
// opts.today: reference date for plausibility (default: now)
export function parseDate(text, opts = {}) {
  if (!text) return null;
  let s = kanjiToDigits(text.normalize("NFKC")).replace(/元年/g, "1年")
    .replace(/\s*([.\/\-:年月])\s*/g, "$1")          // "R8. 11. 16" -> "R8.11.16"
    .replace(/(^|[^A-Z])([RHS]) (?=\d)/g, "$1$2");   // "R 8.11.16"
  const kind = /消費|USE ?BY/.test(s) ? "消費期限" : /賞味|EXP|BB|BEST/.test(s) ? "賞味期限" : null;
  // drop label words so an era letter glued to them still parses: "EXPH1.12.12", "BBR8.3.1"
  s = markMonths(s.replace(/BEST ?BEFORE|USE ?BY|EXP\.?|BB/g, " "))
    .replace(/(?<=\d)O(?![A-Z])|(?<![A-Z])O(?=\d)/g, "0"); // "2022.09.1O", "202O" (letter O next to digits)
  // time: 14時 / 14時30分 / 午後2時 / 14:30
  let time = null;
  const tm = s.match(/(午前|午後)?(\d{1,2})時(?:(\d{1,2})分)?|(?:^|\D)(\d{1,2}):(\d{2})(?!\d)/);
  if (tm) {
    let h = +(tm[2] ?? tm[4]);
    const mi = +(tm[3] ?? tm[5] ?? 0);
    if (tm[1] === "午後" && h < 12) h += 12;
    if (tm[1] === "午前" && h === 12) h = 0;
    if (h <= 24 && mi < 60) time = `${pad2(h)}:${pad2(mi)}`;
  }
  s = s.replace(/(午前|午後)?\d{1,2}時(\d{1,2}分)?|(^|\D)\d{1,2}:\d{2}(?!\d)/g, "$3 "); // drop time
  // strict: a wrong expiry entered silently is worse than no read. A digit run of 5+ glued to a
  // separator means the year is broken ("22027/03/31": label edge read as "2") -> no date.
  if (opts.strict && /\d{5,}[.\/\-年]|[.\/\-]\d{5,}/.test(s)) return null;
  let cands = findCandidates(s, kind, opts.today || new Date(), !!time);
  if (opts.strict) cands = cands.filter((c) => !c.loose);
  if (!cands.length) return null;
  const r = cands[0];
  let part = null;
  // 上旬/初旬/初/月初 = 1-10, 中旬/中 = 11-20, 下旬/末/月末 = 21-end
  if (!r.d) {
    const pm = s.replace(/[A-Z]+/g, "").match(/上旬|初旬|月初|初|中旬|中|下旬|月末|末/);
    if (pm) part = /上|初/.test(pm[0]) ? "上" : /中/.test(pm[0]) ? "中" : "下";
  }
  const precision = r.d ? "day" : part ? "dekad" : "month";
  let iso = `${r.y}-${pad2(r.m)}` + (r.d ? `-${pad2(r.d)}` : "");
  // report the last day of the range (conservative for expiry)
  const last = part ? { 上: 10, 中: 20, 下: new Date(r.y, r.m, 0).getDate() }[part] : null;
  const latest = r.d ? iso : `${r.y}-${pad2(r.m)}-${pad2(last ?? new Date(r.y, r.m, 0).getDate())}`;
  const toIso = (c) => `${c.y}-${pad2(c.m)}` + (c.d ? `-${pad2(c.d)}` : "");
  const alternatives = [...new Set(cands.slice(1).filter((c) => c.score >= r.score - 2 && (!r.d || c.d)).map(toIso))]
    .filter((v) => v !== iso);
  return { kind, iso, precision, part, latest, time, order: r.order, alternatives };
}

// month-day without a year: this year, or next year if it already passed (>60 days ago)
function inferYear(m, d, today) {
  const y = today.getFullYear();
  return new Date(y, m - 1, d) < new Date(y, today.getMonth(), today.getDate() - 60) ? y + 1 : y;
}

// All readings of every date-like token, best first.
// Explicit forms (era, 年月日) beat bare digit groups; for bare digits every
// order (YMD / DMY / MDY) is tried and judged by validity and plausibility.
function findCandidates(s, kind, today, hasTime) {
  const out = [];
  const foreign = /EXP|BB|BEST|USE/.test(s);
  const t0 = Date.UTC(today.getFullYear(), today.getMonth(), today.getDate());
  // loose = a reading that OCR noise produces easily ("2/8", "03/31"); dropped in strict mode
  const add = (y, m, d, order, base, loose = false) => {
    const r = mkDate(y, m, d);
    if (!r) return;
    const days = (Date.UTC(r.y, r.m - 1, r.d || 15) - t0) / 864e5;
    // expiry dates sit near today, mostly in the future
    const plaus = days >= -400 && days <= 6 * 366 ? 3 : days >= -3 * 366 && days <= 12 * 366 ? 1 : 0;
    out.push({ ...r, order, score: base + plaus, loose });
  };
  const marked = /[旬末初中月]/.test(s); // "26.4月初" / "R8.3下旬": a two-part date with an explicit part
  let m;
  // trailing 日 is optional (OCR often drops it), but a bare day must not run into a lot code
  for (m of s.matchAll(/(令和|平成|昭和|令|平|昭)(\d{1,2})年(\d{1,2})月(?:(\d{1,2})(?:日|(?![\dA-Z])))?/g))
    add(ERA[m[1]] + +m[2], +m[3], m[4] && +m[4], "era", 10);
  // R8.3.31 / 令和9.3.31
  // (?!\d): "H05.05.2022" is not 平成5年5月20日
  for (m of s.matchAll(/(?:^|[^A-Z])(令和|平成|昭和|令|平|昭|[RHS])(\d{1,2})[.\/\-年](\d{1,2})(?:[.\/\-月](\d{1,2}))?(?![\d]|[.\/\-]\d)/g))
    add(ERA[m[1]] + +m[2], +m[3], m[4] && +m[4], "era", 9);
  for (m of s.matchAll(/(\d{2,4})年(\d{1,2})月(?:(\d{1,2})日)?/g))
    add(+m[1], +m[2], m[3] && +m[3], "YMD", 8);
  for (m of s.matchAll(/(?<![\d年])(\d{1,2})月(\d{1,2})日/g)) // 11月9日 (no year)
    add(inferYear(+m[1], +m[2], today), +m[1], +m[2], "MD", 7);
  // month names (marked #MM#): 24.APR.2021 / AUG 13 2021 / AUG292020 / 2022.JUN.14 / NOV.2021
  const S = "[.\\/\\- ]*";
  for (m of s.matchAll(new RegExp(`(?<!\\d)(\\d{1,2})${S}#(\\d\\d)#${S}(\\d{4}|\\d{2})(?!\\d)`, "g")))
    add(+m[3], +m[2], +m[1], "DMY", 9);
  for (m of s.matchAll(new RegExp(`#(\\d\\d)#${S}(\\d{1,2})(?:[.\\/\\- ]+(\\d{4}|\\d{2})|(\\d{4}))(?!\\d)`, "g")))
    add(+(m[3] ?? m[4]), +m[1], +m[2], "MDY", 9);
  for (m of s.matchAll(new RegExp(`(?<!\\d)(\\d{4})${S}#(\\d\\d)#${S}(\\d{1,2})(?!\\d)`, "g")))
    add(+m[1], +m[2], +m[3], "YMD", 9);
  for (m of s.matchAll(new RegExp(`#(\\d\\d)#${S}(\\d{4})(?!\\d)`, "g")))
    add(+m[2], +m[1], null, "MY", 8);
  if (out.length) return out.sort((a, b) => b.score - a.score);

  const P = { YMD: 3, DMY: foreign ? 2.5 : 1.5, MDY: foreign ? 1.5 : 0.5 };
  // separated: a.b.c or a.b
  for (m of s.matchAll(/(?<![\dA-Z])(\d{1,4})[.\/\- ](\d{1,2})(?:[.\/\- ](\d{1,4}))?(?![\dA-Z])/g)) {
    const [a, b, c] = [m[1], m[2], m[3]];
    if (c) {
      if (a.length !== 4 && (c.length === 4 || c.length === 2)) {
        // "7/2/10" with a 2-digit year is usually 令7/2/10 with the era lost, not 7 Feb 2010
        const lostEra = a.length === 1 && c.length === 2;
        add(+c, +b, +a, "DMY", P.DMY + (c.length === 4 ? 0.5 : 0), lostEra);
        add(+c, +a, +b, "MDY", P.MDY + (c.length === 4 ? 0.5 : 0), lostEra);
      }
      if (c.length <= 2 && (a.length === 4 || a.length === 2)) add(+a, +b, +c, "YMD", P.YMD + (a.length === 4 ? 1 : 0));
    } else {
      // a two-part date touching a separator is a cut-off three-part one ("2022.01.", "/12/2021")
      const cut = /[.\/\-]/.test(s[m.index - 1] ?? "") || /[.\/\-]/.test(s[m.index + m[0].length] ?? "");
      if (a.length === 4 || a.length === 2) add(+a, +b, null, "YM", 2 + (a.length === 4 ? 1 : 0), cut || (a.length === 2 && !marked));
      if (a.length <= 2 && b.length === 2) add(+b, +a, null, "MY", 1, cut || !marked);
      // 11.09 14:00 -> month-day without year (likely when a time follows)
      if (a.length <= 2) add(inferYear(+a, +b, today), +a, +b, "MD", hasTime ? 4 : 0.5, !hasTime);
    }
  }
  // MM.YYYY
  for (m of s.matchAll(/(?<!\d)(\d{1,2})[.\/\-](\d{4})(?!\d)/g))
    add(+m[2], +m[1], null, "MY", 2, /[.\/\-]/.test(s[m.index - 1] ?? "") || /[.\/\-]/.test(s[m.index + m[0].length] ?? ""));
  // compact digits: 20260811, 260917, 11082026, 170926
  for (m of s.matchAll(/(?<![\d])(\d{8}|\d{6})(?!\d)/g)) {
    const n = m[1], q = (i, l) => +n.slice(i, i + l);
    if (n.length === 8) {
      add(q(0, 4), q(4, 2), q(6, 2), "YMD", P.YMD + 1);
      add(q(4, 4), q(2, 2), q(0, 2), "DMY", P.DMY + 0.5);
      add(q(4, 4), q(0, 2), q(2, 2), "MDY", P.MDY + 0.5);
    } else {
      add(q(0, 2), q(2, 2), q(4, 2), "YMD", P.YMD);
      add(q(4, 2), q(2, 2), q(0, 2), "DMY", P.DMY);
      add(q(4, 2), q(0, 2), q(2, 2), "MDY", P.MDY);
      // a 6-digit run can also be YYYYMM (202608)
      add(q(0, 4), q(4, 2), null, "YM", 1.5);
    }
  }
  return out.sort((a, b) => b.score - a.score || (b.d ? 1 : 0) - (a.d ? 1 : 0));
}
