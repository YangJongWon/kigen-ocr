// Compiles web/gemm.wat and embeds it (base64) into web/kigen-ocr.js, so the runtime stays a
// single dependency-free file. Run after editing gemm.wat: node tools/build-wasm.mjs
import { readFileSync, writeFileSync } from "node:fs";
import wabtInit from "wabt";

const wabt = await wabtInit();
const src = new URL("../web/gemm.wat", import.meta.url), js = new URL("../web/kigen-ocr.js", import.meta.url);
const mod = wabt.parseWat("gemm.wat", readFileSync(src, "utf8"), { simd: true });
mod.validate();
const bin = Buffer.from(mod.toBinary({}).buffer);
const b64 = bin.toString("base64");
const code = readFileSync(js, "utf8");
const re = /const GEMM_WASM = "[^"]*";/;
if (!re.test(code)) throw new Error("GEMM_WASM line not found in kigen-ocr.js");
writeFileSync(js, code.replace(re, `const GEMM_WASM = "${b64}";`));
console.log(`gemm.wasm ${bin.length} bytes embedded`);
