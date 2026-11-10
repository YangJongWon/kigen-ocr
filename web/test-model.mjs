// Checks the JS runtime against the numpy reference (fixture.json written by export.py).
import { readFileSync } from "node:fs";
import { loadModel, recognize, parseDate } from "./kigen-ocr.js";

const buf = readFileSync(new URL("./kigen.bin", import.meta.url));
const model = await loadModel(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const fx = JSON.parse(readFileSync(new URL("./fixture.json", import.meta.url), "utf8"));
let fail = 0;
for (const c of fx) {
  const t = performance.now();
  const r = recognize(model, Float32Array.from(c.x), c.w);
  const ms = performance.now() - t;
  const ok = r.text === c.pred;
  if (!ok) fail++;
  console.log(ok ? "ok  " : "FAIL", `w=${c.w} ${ms.toFixed(0)}ms`, JSON.stringify(r.text), "numpy:", JSON.stringify(c.pred),
    "truth:", JSON.stringify(c.text), "->", parseDate(r.text)?.iso ?? null);
}
console.log(fail ? `${fail} mismatches` : "JS == numpy on all fixtures");
process.exit(fail ? 1 : 0);
