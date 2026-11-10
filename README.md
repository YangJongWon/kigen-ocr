# kigen-ocr

Tiny, dependency-free OCR for **Japanese best-before / use-by dates** (賞味期限 / 消費期限) on product packaging. Runs in the browser or Node. The library is one ES module (`kigen-ocr.js`, ~34 KB) plus an int8 model (`kigen.bin`, ~770 KB).

日本の商品パッケージの賞味期限・消費期限を読む、依存ライブラリなしの小さな OCR です。ブラウザでも Node でも動きます。

- Reads printed, inkjet dot-matrix and faint laser-etched date lines
- Parses Japanese formats: `2026年3月31日`, `令和8年3月`, `R8.3.31`, `令8.3.31`, `平31.4.30`, `2026年3月下旬`, `二〇二六年…`, full-width digits, times (`14時30分`, `午後2時`)
- Also parses numeric and English formats: `2026.03.31`, `31/03/2026`, `260331`, `24 APR 2026`, `AUG292026`
- Straightens tilted text (any angle) and retries upside down / sideways reads
- Splits a box with several lines (e.g. a 賞味期限 line above the date) and takes the label from the neighbouring line
- **Strict by default**: it would rather return no date than a wrong one (see [Strict mode](#strict-mode))

## Quick start

```html
<script type="module">
import { loadModel, readRegion } from "./kigen-ocr.js";

const model = await loadModel("kigen.bin");

// img: ImageData of the area around the date line (e.g. from a canvas)
const r = readRegion(model, img);
if (r.date) {
  console.log(r.date.kind, r.date.iso);   // "賞味期限" "2026-03-31"
} else if (r.rejected) {
  console.log("not sure, ask for a retake:", r.rejected.reason);
}
</script>
```

Node:

```js
import { readFileSync } from "node:fs";
import { loadModel, readRegion, parseDate } from "./kigen-ocr.js";

const buf = readFileSync("kigen.bin");
const model = await loadModel(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const r = readRegion(model, { data: rgbaBytes, width, height }); // ImageData-like
```

`index.html` is a small demo (camera with a frame, or pick an image and drag a box). Serve the folder over HTTP(S); the camera needs HTTPS or `localhost`.

## API

### `loadModel(src) → Promise<model>`
`src` is a URL or an `ArrayBuffer` of `kigen.bin`.

### `readRegion(model, img, opts?) → result`
`img` is RGBA `ImageData` (or `{data, width, height}`) of the framed area. Frame **the date line as tightly as possible**: accuracy drops a lot when the box also holds other lines, label edges or packaging graphics.

| option | default | |
|---|---|---|
| `strict` | `true` | drop readings OCR noise produces easily (see below) |
| `minConf` | `0.8` | strict mode: reject results below this mean character confidence |
| `recheck` | `true` | strict mode: read the chosen line a second time with slightly different framing, reject if the date differs |

Result:

| field | |
|---|---|
| `date` | parsed date (see `parseDate`) or `null` |
| `rejected` | when strict mode dropped a reading: `{ date, reason }`, reason `"lowconf"` or `"unstable"` |
| `text`, `conf` | raw recognized text and mean character confidence of the chosen line |
| `skew`, `turn` | degrees the text was straightened by, and the extra 90/180/270° turn if one was needed |
| `lines`, `band`, `cols` | detected text lines and the area that was read (in the straightened image) |
| `candidates` | every area that was read, with its text and confidence |
| `input` | the 32 px model input that was read (`{x, w}`), useful for debugging |

### `parseDate(text, opts?) → date | null`
Parses OCR text. `opts.today` (default: now) is used to judge plausibility and to infer the year of month-day dates; `opts.strict` as above.

```js
parseDate("賞味期限 R8.3下旬")
// { kind: "賞味期限", iso: "2026-03", precision: "dekad", part: "下",
//   latest: "2026-03-31", time: null, order: "era", alternatives: [] }
```

- `precision`: `"day"`, `"dekad"` (上旬/中旬/下旬, 初/中/末) or `"month"`
- `latest`: the last day the date can mean (conservative for expiry)
- `order`: how digits were read (`era`, `YMD`, `DMY`, `MDY`, `YM`, `MY`, `MD`)
- `alternatives`: other plausible readings of ambiguous digit orders

### Low-level
`recognize(model, x, w)`, `toGray(img)`, `toInput(gray, w, h, model)`, `estimateSkew(gray, w, h)`.

## Strict mode

A wrong expiry date entered silently is worse than no read, so `readRegion` drops:

- broken years (`22027/03/31`: a label edge read as "2")
- yearless numeric month-day (`2/8`), unless a time follows (`11.09 14:00`)
- two-part dates with a 2-digit year (`03/31`, `26.3`) unless marked (`26.4月初`)
- cut-off dates (`2022.01.`, `/12/2021`) and dates whose era was likely lost (`7/2/10`)
- low-confidence and unstable reads (`minConf`, `recheck`)

Kanji month-day (`11月9日`) and anything with a 4-digit year or an era are kept. In camera mode a dropped frame is cheap: the next frame tries again.

## Accuracy (measured on a PC, strict mode)

| set | correct | wrong date | no date |
|---|---|---|---|
| ExpDate real photos, cropped to the date (1,173) | 84% | 1.3% | 14% |
| one-letter era `R8.3.31`, synthetic (300) | 82% | 2.7% | 15% |
| one-kanji era `令8.3.31`, synthetic (300) | 78% | 2.0% | 20% |
| product photos, wide box with neighbouring lines (199) | 68% | 3.5% | 28% |

Tilted / turned copies of the product photos (10°–270°) read at roughly 70–89% with 0–2.5% wrong dates (measured with the previous model). ExpDate is mostly numeric and English dates, so Japanese formats are only covered by synthetic tests; real Japanese label photos have not been benchmarked yet.

Speed (PC, single thread): ~250 ms for a tight crop, ~400–500 ms for a wide box, up to ~1 s when the text has to be turned. Not measured on phones.

## Limitations

- Expects the user to frame the date line; there is no detector for whole labels.
- Model input is 32 px high: very small or blurry text loses digits.
- Japanese-format accuracy is measured on synthetic images only.

## Training (optional)

The model is trained on **synthetic images only** (`gen.py`): text rendered with fonts in random styles (print, inkjet dots, faint etching) on packaging-like backgrounds. Real photos (ExpDate) were used for evaluation only.

```
uv run python run_train.py --steps 40000          # from scratch
uv run python run_train.py --init model.pt --lr 5e-4 --steps 6000 --out model_ft.pt
uv run python export.py --ckpt model.pt           # -> web/kigen.bin, web/fixture.json
node web/test-parse.mjs && node web/test-model.mjs
```

`gen.py` uses Windows fonts from `C:/Windows/Fonts` (MS Gothic/Mincho, Meiryo, Yu Gothic, BIZ UD, UD Digi Kyokasho, Noto Sans JP, and Latin fonts); adjust the paths on other systems. No font files are distributed.

## License

MIT, see [LICENSE](LICENSE).
