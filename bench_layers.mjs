// recognize() cost by input width and backend, plus per-op split (instrumented copy).
import { readFileSync, writeFileSync, unlinkSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = dirname(fileURLToPath(import.meta.url));
const FNS = ["conv2d", "maxpool", "conv1d", "wasmConv8", "runGemm"];
let src = readFileSync(join(ROOT, "web/kigen-ocr.js"), "utf8");
let wrap = "\nexport const __T = {}, __C = {};\n";
for (const f of FNS) {
  src = src.replace(new RegExp(`function ${f}\\(`), `function ${f}__raw(`);
  wrap += `function ${f}(...a) { const t = performance.now(); try { return ${f}__raw(...a); } finally { const k = ${f === "conv2d" || f === "conv1d" ? `"${f}:" + a[a.length - (${f === "conv2d" ? 1 : 3})].shape.join("x")` : `"${f}"`}; __T[k] = (__T[k] || 0) + performance.now() - t; __C[k] = (__C[k] || 0) + 1; } }\n`;
}
const tmp = join(ROOT, ".prof-layers.mjs");
writeFileSync(tmp, src + wrap);
const K = await import(pathToFileURL(tmp).href);
unlinkSync(tmp);

const buf = readFileSync(join(ROOT, "web/kigen.bin"));
const ab = () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const models = {
  "wasm int8": await K.loadModel(ab()),
  "wasm f32": await K.loadModel(ab(), { int8: false }),
  "plain JS": await K.loadModel(ab(), { wasm: false }),
};

const time = (f, n = 20) => { f(); f(); const t = performance.now(); for (let i = 0; i < n; i++) f(); return (performance.now() - t) / n; };
console.log("width  " + Object.keys(models).map((k) => k.padStart(10)).join(""));
for (const w of [64, 128, 192, 256, 384, 512]) {
  const x = new Float32Array(32 * w).map(() => Math.random());
  console.log(String(w).padEnd(7) + Object.values(models).map((m) => time(() => K.recognize(m, x, w)).toFixed(1).padStart(10)).join("") + " ms");
}

const w = 192, n = 30, x = new Float32Array(32 * w).map(() => Math.random());
const m = models["wasm int8"];
K.recognize(m, x, w);
for (const k in K.__T) { delete K.__T[k]; delete K.__C[k]; }
const t0 = performance.now();
for (let i = 0; i < n; i++) K.recognize(m, x, w);
const tot = (performance.now() - t0) / n;
console.log(`\nper op (wasm int8, w=${w}), total ${tot.toFixed(1)} ms:`);
const keys = Object.keys(K.__T).filter((k) => !["wasmConv8", "runGemm"].includes(k));
let sum = 0;
for (const k of keys) {
  const t = K.__T[k] / n; sum += t;
  console.log(`  ${k.padEnd(26)} ${t.toFixed(2).padStart(6)} ms  ${(t / tot * 100).toFixed(0).padStart(3)}%  x${(K.__C[k] / n).toFixed(0)}`);
}
console.log(`  ${"(decode + glue)".padEnd(26)} ${(tot - sum).toFixed(2).padStart(6)} ms`);
