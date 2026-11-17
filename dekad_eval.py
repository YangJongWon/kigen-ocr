"""Dekad lines (上旬/中旬/下旬/初/中/末) per condition, and the confusion pairs:
exact text accuracy per checkpoint, the same lines for every checkpoint.

  uv run python dekad_eval.py model_emb.pt model_dekad.pt [--n 300]

Sets: plain, rot (turned 15-90 deg in a 640 px frame and back), small (text 9-12 px),
blur (Gaussian sigma 1.2-2.0), clip (end glyphs cut 3-8%), conf (ordinary dates with 0/2/7 heavy
fields). Counts per checkpoint how often 初<->末, 上<->下, 0<->2 were swapped in a wrong read.
"""
import argparse
import difflib
import random

import cv2
import numpy as np
import torch

import gen
import train

PAIRS = [("初", "末"), ("上", "下"), ("0", "2"), ("2", "7"), ("中", "末"), ("初", "上")]


def dekad_text():
    gen.DEKAD_P, gen.CONF_P = 1.0, 0.0
    while True:
        vis, t = gen.rand_text()
        if any(c in t for c in "旬初末") or t.endswith("中"):
            return vis, t


def conf_text():
    gen.DEKAD_P, gen.CONF_P = 0.0, 1.0
    while True:
        vis, t = gen.rand_text()
        if sum(c in "027" for c in t) >= 3:
            return vis, t


def lines(n, mode, seed):
    random.seed(seed)
    np.random.seed(seed)
    gen.ML_P = 0.0
    gen.ROTRT_P, gen.CLIP_P = 0.0, (1.0 if mode == "clip" else 0.0)
    out = []
    while len(out) < n:
        vis, t = conf_text() if mode == "conf" else dekad_text()
        if mode in ("rot", "small", "blur"):
            orig = gen.photometric
            if mode == "rot":
                gen.ROTRT_P = 1.0
                rr = gen.rot_roundtrip
                gen_rot = lambda img, th: rr(img, th, ang=random.uniform(15, 90))
                gen.rot_roundtrip, keep = gen_rot, rr
            elif mode == "small":
                gen.ROTRT_P = 1.0
                rr = gen.rot_roundtrip
                gen.rot_roundtrip, keep = (lambda img, th: rr(img, th, ang=0.0, small=random.uniform(9, 12))), rr
            else:
                def blurred(img):
                    img = orig(img)
                    return cv2.GaussianBlur(img, (0, 0), random.uniform(1.2, 2.0))
                gen.photometric, keep = blurred, None
            img = gen.render(vis)
            if mode == "blur":
                gen.photometric = orig
            else:
                gen.rot_roundtrip = keep
                gen.ROTRT_P = 0.0
        else:
            img = gen.render(vis)
        if img is not None and t:
            out.append((gen.to_input(img), t))
    return out


import re
CORE = re.compile(r"(?:令和|[RH令])?\d[\d./\-年月]*(?:上旬|中旬|下旬|初旬|月末|月初|初|中|末|\d日|\d)")


def core(t):
    """The date itself (without prefix and lot code): what parseDate has to get right."""
    m = CORE.search(t)
    return m.group(0) if m else t


def swaps(p, t, cnt):
    for op, a0, a1, b0, b1 in difflib.SequenceMatcher(None, t, p).get_opcodes():
        if op == "replace" and a1 - a0 == b1 - b0:
            for x, y in zip(t[a0:a1], p[b0:b1]):
                for u, v in PAIRS:
                    if {x, y} == {u, v}:
                        cnt[f"{u}{v}"] = cnt.get(f"{u}{v}", 0) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    sets = {m: lines(a.n, m, 41 + i) for i, m in enumerate(("plain", "rot", "small", "blur", "clip", "conf"))}
    for ck in a.ckpts:
        m = train.Net()
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m.eval()
        res, cnt, shown = [], {}, 0
        for name, data in sets.items():
            ok = okc = 0
            for i in range(0, len(data), 64):
                chunk = data[i:i + 64]
                xb = train.collate(chunk)[0]
                with torch.no_grad():
                    pred = train.decode(m(xb))
                for p, (_, t) in zip(pred, chunk):
                    ok += p == t
                    okc += core(t) in p.replace(" ", "")
                    if p != t:
                        swaps(p, t, cnt)
                        if shown < a.show:
                            print(f"  {name:5s} want {t!r} got {p!r}")
                            shown += 1
            res.append(f"{name} {ok / len(data):.3f}/{okc / len(data):.3f}")
        print(f"{ck:18s} exact/date: " + "  ".join(res))
        print(f"{'':18s} swaps: " + "  ".join(f"{k} {v}" for k, v in sorted(cnt.items())))


if __name__ == "__main__":
    main()
