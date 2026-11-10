import { readFileSync } from "node:fs";
import { loadModel, recognize } from "./kigen-ocr.js";
const buf = readFileSync(new URL("./kigen.bin", import.meta.url));
const model = await loadModel(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
for (const w of [128, 256, 512]) {
  const x = new Float32Array(32 * w).map(() => Math.random());
  recognize(model, x, w);
  const t = performance.now(); for (let i = 0; i < 10; i++) recognize(model, x, w);
  console.log(`w=${w}: ${((performance.now() - t) / 10).toFixed(1)} ms`);
}
