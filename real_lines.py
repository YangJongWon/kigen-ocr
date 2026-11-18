"""Real date lines for training (KIGEN_REAL=1 in gen.py): Products-Real *train* split only — its
evaluation split, Date-Real and the camera bench stay held out.

Each line is cut with room around it (1.0 x height above/below, 2.0 x height left/right) so gen.py
can frame it at random. Labels go through gen.norm_label; lines with characters outside the
vocabulary are skipped.

uv run python real_lines.py            -> datasets/real_lines_train.pkl
"""
import json
import os
import pickle

import cv2
import numpy as np

import gen

BASE = "datasets/Products-Real/train"
OUT = "datasets/real_lines_train.pkl"


def main():
    ann = json.load(open(os.path.join(BASE, "annotations.json"), encoding="utf8"))
    out, skipped = [], 0
    for file, meta in ann.items():
        lines = [a for a in meta.get("ann", []) if a["cls"] == "date" and a.get("transcription")]
        if not lines:
            continue
        img = cv2.imread(os.path.join(BASE, "images", file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H, W = img.shape
        for a in lines:
            t = gen.norm_label(a["transcription"])
            if not t or any(c not in gen.C2I for c in t):
                skipped += 1
                continue
            x0, y0, x1, y1 = a["bbox"]
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            h = y1 - y0
            if h < 6 or x1 - x0 < 10:
                skipped += 1
                continue
            X0, Y0 = max(0, int(x0 - 2.0 * h)), max(0, int(y0 - 1.0 * h))
            X1, Y1 = min(W, int(x1 + 2.0 * h)), min(H, int(y1 + 1.0 * h))
            out.append({"g": img[Y0:Y1, X0:X1].copy(), "box": (x0 - X0, y0 - Y0, x1 - X0, y1 - Y0), "text": t, "file": file})
    pickle.dump(out, open(OUT, "wb"))
    print(len(out), "lines,", skipped, "skipped ->", OUT)


if __name__ == "__main__":
    main()
