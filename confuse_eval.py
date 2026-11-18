"""Targeted confusions on synthetic line crops (recognizer only), same lines for every checkpoint:
  r2    era dates "R9.3.31" / "Ｒ９．３．３１" / "R 9. 3. 31": R read as 2 (令和 abbreviation lost)
  fw50  full-width dates heavy in 5 and 0 ("２０２５．１１．１５"): 5 read as 0 and back
  end1  "27/3/31" style lines ending in 1 with a 15-45 deg tilted frame edge (straightening streaks)
        at the end: the final 1 dropped
Each set mixes half-width, full-width (letters too) and U+3000-spaced notation.

  KIGEN_EDGE=0 uv run python confuse_eval.py model_tilt.pt model_ksep.pt [--n 600]
"""
import argparse
import difflib
import random

import numpy as np
import torch

import gen
import train

FW = {chr(c): chr(c + 0xFEE0) for c in range(0x21, 0x7F)}


def notation(t, i):
    k = i % 4
    if k == 1:
        return "".join(FW.get(c, c) for c in t)                       # Ｒ９．３．３１
    if k == 2:                                                         # Ｒ９．　３．　３１
        return "".join(FW.get(c, c) + ("　" if c in "./" else "") for c in t)
    if k == 3:
        return "".join(c + (" " if c in "./" else "") for c in t)      # R9. 3. 31
    return t


def text_for(kind, i):
    y = random.randint(2025, 2031)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    if kind == "r2":
        sep = random.choice(".//")
        return f"R{random.randint(1, 16)}{sep}{m}{sep}{random.choice([d, 31, 15, 1])}"
    if kind == "fw50":
        d = random.choice([5, 10, 15, 20, 25, 30, 50 % 31, 5])
        m = random.choice([5, 10, 1, 11, m])
        return random.choice([f"{y}.{m:02d}.{d:02d}", f"{y % 100}.{m:02d}.{d:02d}", f"{y}/{m:02d}/{d:02d}", f"{y}年{m}月{d}日"])
    # end1
    d = random.choice([1, 11, 21, 31])
    return random.choice([f"{y % 100}/{m}/{d}", f"{y % 100}.{m}.{d}", f"{y}/{m:02d}/{d:02d}", f"R{y - 2018}.{m}.{d}"])


def lines(kind, n, seed):
    random.seed(seed)
    np.random.seed(seed)
    out = []
    i = 0
    while len(out) < n:
        t = text_for(kind, i)
        vis = notation(t, i)
        edge = {"side": "R", "cut": False, "partial": False, "phi": (45, 75)} if kind == "end1" else None
        img = gen.render(vis, edge=edge)
        i += 1
        if img is None or (edge is not None and not edge.get("done")):
            continue
        out.append((gen.to_input(img), gen.norm_label(t), vis))
    return out


def count(kind, p, t):
    if kind == "end1":
        return int(p == t[:-1] or p.rstrip() == t[:-1])
    n = 0
    for op, a0, a1, b0, b1 in difflib.SequenceMatcher(None, t, p).get_opcodes():
        if op == "replace" and a1 - a0 == b1 - b0:
            for x, y in zip(t[a0:a1], p[b0:b1]):
                if kind == "r2" and x == "R" and y == "2":
                    n += 1
                if kind == "fw50" and {x, y} == {"5", "0"}:
                    n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    gen.ML_P = 0.0
    sets = {k: lines(k, a.n, 70 + i) for i, k in enumerate(("r2", "fw50", "end1"))}
    for ck in a.ckpts:
        m = train.Net()
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m.eval()
        res = []
        for kind, data in sets.items():
            ok = bad = 0
            per = {}
            for i in range(0, len(data), 64):
                ch = [(x, t) for x, t, _ in data[i:i + 64]]
                with torch.no_grad():
                    pr = train.decode(m(train.collate(ch)[0]))
                for j, (p, (_, t)) in enumerate(zip(pr, ch)):
                    ok += p == t
                    c = count(kind, p, t)
                    bad += c
                    nt = ["half", "full", "full+U3000", "spaced"][(i + j) % 4]
                    per[nt] = per.get(nt, 0) + c
                    if c and a.show:
                        print(f"  {kind} want {t!r} shown {data[i + j][2]!r} got {p!r}")
            res.append(f"{kind}: exact {ok / len(data):.3f} confusions {bad} ({', '.join(f'{k} {v}' for k, v in per.items())})")
        print(f"== {ck}\n   " + "\n   ".join(res))


if __name__ == "__main__":
    main()
