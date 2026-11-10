"""Evaluate model.pt on ExpDate real photos (Date-Real / Products-Real).

Downloads are expected under datasets/ (see LINKS.md). Example:

  uv run python eval_expdate.py
  uv run python eval_expdate.py --ckpt model.pt --limit 0
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

import gen
import train

ROOT = Path(__file__).resolve().parent
DATE_REAL = ROOT / "datasets" / "Date-Real"
PRODUCTS_EVAL = ROOT / "datasets" / "Products-Real" / "evaluation"

DIGITS = re.compile(r"\D+")
SEPS = re.compile(r"[.\-/ ]+")


def norm_digits(s: str) -> str:
    return DIGITS.sub("", s)


def norm_seps(s: str) -> str:
    """Collapse . / - space to a single '.', for format-tolerant scoring."""
    return SEPS.sub(".", s.strip()).strip(".")


def reconstruct_date_real_gt(ann: list[dict]) -> str:
    """Join year/month/day transcriptions left-to-right; insert '.' on gaps."""
    parts = sorted(
        (a for a in ann if a.get("transcription")),
        key=lambda a: a["bbox"][0],
    )
    if not parts:
        return ""
    out = parts[0]["transcription"]
    for prev, cur in zip(parts, parts[1:]):
        gap = cur["bbox"][0] - prev["bbox"][2]
        if gap >= 3:
            out += "."
        out += cur["transcription"]
    return out


def load_date_real(limit: int = 0):
    ann_path = DATE_REAL / "annotations.json"
    img_dir = DATE_REAL / "images"
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    items = []
    for name, meta in data.items():
        gt = reconstruct_date_real_gt(meta.get("ann", []))
        if not gt:
            continue
        path = img_dir / name
        if not path.exists():
            continue
        items.append((path, gt, None))
        if limit and len(items) >= limit:
            break
    return items


def load_products_real(limit: int = 0):
    ann_path = PRODUCTS_EVAL / "annotations.json"
    img_dir = PRODUCTS_EVAL / "images"
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    items = []
    for name, meta in data.items():
        path = img_dir / name
        if not path.exists():
            continue
        for a in meta.get("ann", []):
            if a.get("cls") != "exp" or not a.get("transcription"):
                continue
            items.append((path, a["transcription"], a["bbox"]))
            if limit and len(items) >= limit:
                return items
    return items


def read_crop(path: Path, bbox):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = bgr[:, :, ::-1]
    if bbox is None:
        return rgb
    x0, y0, x1, y1 = [int(v) for v in bbox]
    h, w = rgb.shape[:2]
    pad = max(1, int(0.04 * (y1 - y0)))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    return rgb[y0:y1, x0:x1]


def collate_xs(xs):
    W = max(x.shape[1] for x in xs)
    W = (W + 3) // 4 * 4
    return np.stack([np.pad(x, ((0, 0), (0, W - x.shape[1])), mode="edge") for x in xs])


def evaluate(model, items, device, batch_size=64):
    ok = sep_ok = dig_ok = errs = chars = 0
    oov = Counter()
    samples = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            xs, gts = [], []
            for path, gt, bbox in chunk:
                img = read_crop(path, bbox)
                if img.size == 0 or min(img.shape[:2]) < 2:
                    continue
                xs.append(gen.to_input(img))
                gts.append(gt)
                for ch in gt:
                    if ch not in gen.C2I:
                        oov[ch] += 1
            if not xs:
                continue
            x = torch.from_numpy(collate_xs(xs))[:, None].to(device)
            preds = train.decode(model(x))
            for p, t in zip(preds, gts):
                ok += p == t
                sep_ok += norm_seps(p) == norm_seps(t)
                dig_ok += norm_digits(p) == norm_digits(t) and bool(norm_digits(t))
                errs += train.cer(p, t)
                chars += max(len(t), 1)
                if len(samples) < 12 and norm_digits(p) != norm_digits(t):
                    samples.append((t, p))
    n = len(items)
    return {
        "n": n,
        "line_acc": ok / n if n else 0.0,
        "sep_acc": sep_ok / n if n else 0.0,
        "digit_acc": dig_ok / n if n else 0.0,
        "cer": errs / chars if chars else 0.0,
        "oov": oov,
        "samples": samples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model.pt")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = train.Net()
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    model.to(args.device)

    suites = []
    if (DATE_REAL / "annotations.json").exists():
        suites.append(("Date-Real", load_date_real(args.limit)))
    if (PRODUCTS_EVAL / "annotations.json").exists():
        suites.append(("Products-Real/evaluation", load_products_real(args.limit)))
    if not suites:
        raise SystemExit("No ExpDate data under datasets/. Download Date-Real / Products-Real first.")

    print(f"ckpt {args.ckpt} device {args.device}")
    for name, items in suites:
        m = evaluate(model, items, args.device)
        print(
            f"== {name}: n={m['n']} line-acc={m['line_acc']:.4f} "
            f"sep-acc={m['sep_acc']:.4f} digit-acc={m['digit_acc']:.4f} "
            f"CER={m['cer']:.4f}"
        )
        if m["oov"]:
            print("   OOV chars in GT:", dict(m["oov"]))
        for gt, pred in m["samples"]:
            line = f"   digit-miss  gt={gt!r}  pred={pred!r}"
            print(line.encode("utf-8", "backslashreplace").decode("utf-8"))


if __name__ == "__main__":
    main()
