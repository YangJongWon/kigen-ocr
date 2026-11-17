"""Synthetic lines of one hard style per set (emboss, text under the date, lines across it):
exact text accuracy per checkpoint, same lines for every checkpoint.

KIGEN_EDGE=0 uv run python style_eval.py model_plus.pt model_emb.pt [--n 400]
"""
import argparse
import random

import numpy as np
import torch

import gen
import train


def lines(n, mode, seed):
    random.seed(seed)
    np.random.seed(seed)
    gen.UNDER_P = 1.0 if mode == "under" else 0.0
    gen.LINES_P = 1.0 if mode == "lines" else 0.0
    out = []
    while len(out) < n:
        vis, t = gen.rand_text()
        img = gen.render(vis, style="emboss" if mode == "emboss" else None)
        if img is not None and t:
            out.append((gen.to_input(img), t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    sets = {m: lines(a.n, m, 21 + i) for i, m in enumerate(("emboss", "under", "lines", "plain"))}
    for ck in a.ckpts:
        m = train.Net()
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m.eval()
        res = []
        for name, data in sets.items():
            with torch.no_grad():
                ok = sum(train.decode(m(torch.from_numpy(x)[None, None]))[0] == t for x, t in data)
            res.append(f"{name} {ok / len(data):.3f}")
        print(f"{ck:16s} exact: " + "  ".join(res))


if __name__ == "__main__":
    main()
