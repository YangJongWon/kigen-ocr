"""Embossed date lines with and without unlabelled dents / scratches (gen.dents), same lines and
seeds for every checkpoint: date digits right, a "." read where there is none (inserted), a real
"." lost (dropped). Half-width, spaced, full-width, full-width + U+3000, full-width letters.

  KIGEN_EDGE=0 uv run python scratch_eval.py model_real.pt model_new.pt [--n 600]
"""
import argparse
import random
import re

import numpy as np
import torch

import gen
import train

FW = {chr(c): chr(c + 0xFEE0) for c in range(0x21, 0x7F)}
STY = ["half", "half spaced", "full", "full+U3000", "FW letters"]


def text_for(i):
    y, m, d = random.randint(2025, 2031), random.randint(1, 12), random.randint(1, 28)
    k = i % 6
    if k == 0:
        return f"{y} {m:02d} {d:02d}"            # no dot anywhere: a dent would invent one
    if k == 1:
        return f"{y}{m:02d}{d:02d}"
    if k == 2:
        return f"R{y - 2018} {m} {d}"
    if k == 3:
        return f"{y}.{m:02d}.{d:02d}"            # real dots: must stay
    if k == 4:
        return f"{y % 100}.{m}.{d}"
    return f"{y}.{m:02d}"


def notation(t, st):
    pre = random.choice(["", "EXP ", "BB "]) if st == "FW letters" else ""
    s = pre + t
    if st == "half spaced":
        s = re.sub(r"\.(?=\d)", ". ", s)
    if st in ("full", "full+U3000"):
        s = "".join(FW.get(c, c) for c in s)
        s = s.replace(" ", "　")
        if st == "full+U3000":
            s = s.replace("．", "．　")
    if st == "FW letters":
        s = "".join(FW[c] if c.isascii() and c.isalpha() else c for c in s)
    return s, gen.norm_label(pre + t)


def lines(n, seed, scratch):
    out = []
    i = 0
    while len(out) < n:
        random.seed(seed * 100003 + i)
        np.random.seed((seed * 100003 + i) % 2 ** 32)
        st = STY[i % 5]
        t = text_for(i // 5)
        vis, lab = notation(t, st)
        gen.SCRATCH_P = 1.0 if scratch else 0.0
        img = gen.render(vis, style="emboss")
        i += 1
        if img is None:
            continue
        out.append((gen.to_input(img), lab, st))
    return out


def dots(s):
    return s.count(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    gen.ML_P = 0.0
    sets = {"clean": lines(a.n, 7, False), "dented": lines(a.n, 7, True)}
    for ck in a.ckpts:
        m = train.Net()
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m.eval()
        res = []
        for name, data in sets.items():
            ok = ins = drop = 0
            per = {}
            shown = 0
            for i in range(0, len(data), 64):
                ch = data[i:i + 64]
                with torch.no_grad():
                    pr = train.decode(m(train.collate([(x, t) for x, t, _ in ch])[0]))
                for p, (_, t, st) in zip(pr, ch):
                    good = re.findall(r"\d+", p) == re.findall(r"\d+", t) and ("R" in p) == ("R" in t)
                    ok += good
                    ins += dots(p) > dots(t)
                    drop += dots(p) < dots(t)
                    s = per.setdefault(st, [0, 0])
                    s[0] += good
                    s[1] += 1
                    if not good and shown < a.show:
                        print(f"   {name} {st} want {t!r} got {p!r}")
                        shown += 1
            res.append(f"{name}: date-digits {ok / len(data):.3f}  dot inserted {ins}  dot dropped {drop}  ("
                       + ", ".join(f"{k} {v[0] / v[1]:.2f}" for k, v in per.items()) + ")")
        print(f"== {ck}\n   " + "\n   ".join(res))


if __name__ == "__main__":
    main()
