// node web/test-det.mjs — JS detector == numpy int8 reference (fixture_det.json from detect.py export)
import { readFileSync } from "node:fs";
import * as K from "./kigen-ocr.js";
const buf = readFileSync(new URL("./kigen-det.bin", import.meta.url));
const fx = JSON.parse(readFileSync(new URL("./fixture_det.json", import.meta.url), "utf8"));
let bad = 0;
for (const wasm of [true, false]) {
  const det = await K.loadModel(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), { wasm });
  for (const f of fx) {
    const t = performance.now(), { z, Wc } = K.detectLogits(det, Float32Array.from(f.x), f.w), ms = performance.now() - t;
    let err = 0;
    for (let i = 0; i < f.logits.length; i++) err = Math.max(err, Math.abs(z[i] - f.logits[i]));
    // int8 activations (wasm) differ a little from the float reference
    const ok = err < (det.wa?.int8 ? 0.6 : 0.01);
    if (!ok) bad++;
    console.log(`${ok ? "ok  " : "FAIL"} ${det.wa ? (det.wa.int8 ? "wasm8" : "wasmf") : "js   "} w=${f.w} max|dz| ${err.toFixed(3)} ${ms.toFixed(1)}ms`);
  }
}
console.log(bad ? `${bad} failed` : "detector JS == numpy");
process.exit(bad ? 1 : 0);
