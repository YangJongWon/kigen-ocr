// node web/test-guards.mjs — readRegion's dekad guards on small synthetic guide frames (web/guard/,
// date ~14 px tall, made with Windows fonts): a dekad mark cut off at the frame's right edge must
// never come out as the month's end ("2028.07初" -> "2028.07" = 07-31, 21 days too late), an
// unclear 初/末 is not handed out, and dates with room (or ending in a year) still read.
import { readFileSync } from "node:fs";
import jpeg from "jpeg-js";
import * as K from "./kigen-ocr.js";
const load = async (f) => { const b = readFileSync(new URL(f, import.meta.url)); return K.loadModel(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const model = await load("./kigen.bin"), det = await load("./kigen-det.bin");
// [frame, accepted results (latest date, or null = not handed out), dates that must never come out]
// A cut frame may still be read right (enough of 初 left); it must never come out as the month's end.
const cases = [
  ["sho_cut_meiryo", [null, "2028-07-10"], ["2028-07-31"]],
  ["sho_cut_msgothic", [null, "2028-07-10"], ["2028-07-31"]],
  ["sho_cut_YuGothB", [null, "2028-07-10"], ["2028-07-31"]],
  ["sho_room_msgothic", "2028-07-10"],
  ["sho_touch_msgothic", "2028-07-10"],
  ["jojun_cut_msgothic", "2026-11-10"],
  ["my_touch_msgothic", "2023-02-28"],
  ["my_touch_meiryo", "2023-02-28"],
];
let bad = 0;
for (const [f, want, never = []] of cases) {
  const img = jpeg.decode(readFileSync(new URL(`./guard/${f}.jpg`, import.meta.url)), { useTArray: true, formatAsRGBA: true });
  const r = K.readRegion(model, img, { det }), got = r.date ? (r.date.latest ?? r.date.iso) : null;
  const ok = (Array.isArray(want) ? want.includes(got) : got === want) && !never.includes(got);
  if (!ok) bad++;
  console.log(`${ok ? "ok  " : "FAIL"} ${f.padEnd(20)} "${r.text}" -> ${got ?? "-"}${r.rejected ? ` (${r.rejected.reason})` : ""}  want ${[want].flat().map((v) => v ?? "-").join(" or ")}`);
}
// the guards off: the cut frames do come out as the month's end (what the guards are for)
const img = jpeg.decode(readFileSync(new URL("./guard/sho_cut_msgothic.jpg", import.meta.url)), { useTArray: true, formatAsRGBA: true });
const off = K.readRegion(model, img, { det, dekadMargin: 0, edgeGap: -1e9 });
console.log(`     guards off: sho_cut_msgothic -> ${off.date?.latest ?? off.date?.iso ?? "-"}`);
if (bad) { console.log(`${bad} failed`); process.exit(1); }
console.log("all passed");
