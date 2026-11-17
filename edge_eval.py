"""Lines with a frame edge (gen.frame_edge): does a recognizer invent a cut glyph or read the
streaks as characters? Compares checkpoints on the same synthetic lines.

uv run python edge_eval.py model_plus.pt model_edge.pt [--n 600]
"""
import argparse
import random

import numpy as np
import torch

import gen
import train


def lines(n, mode, seed):
    """mode: "cut" (end glyph mostly cut, label drops it), "edge" (edge in margin / grazing),
    "touch" (whole end glyph touching the crop edge, no frame edge)."""
    random.seed(seed)
    np.random.seed(seed)
    cut = mode == "cut"
    gen.CUT_P, gen.EDGE_P = {"cut": (1.0, 0.0), "edge": (0.0, 1.0), "touch": (0.0, 0.0)}[mode]
    gen.TOUCH_P = 1.0 if mode == "touch" else 0.0
    out = []
    while len(out) < n:
        vis, t = gen.rand_text()
        if mode == "touch":
            img = gen.render(vis)
            if img is not None and t:
                out.append((gen.to_input(img), t, t))
            continue
        e = gen.edge_plan(vis, t)
        if not e:
            continue
        img = gen.render(vis, edge=e)
        if img is None or not e.get("done"):
            continue
        lab = " ".join((t[1:] if e["side"] == "L" else t[:-1]).split()) if cut else t
        out.append((gen.to_input(img), lab, t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    sets = {m: lines(a.n, m, 11 + i) for i, m in enumerate(("cut", "edge", "touch"))}
    for ck in a.ckpts:
        m = train.Net()
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m.eval()
        for name, data in sets.items():
            ok = invent = 0
            for x, lab, full in data:
                with torch.no_grad():
                    p = train.decode(m(torch.from_numpy(x)[None, None]))[0]
                ok += p == lab
                invent += (cut_invented(p, lab, full) if name == "cut" else len(p) > len(lab) if name == "edge"
                           else p in (lab[1:].strip(), lab[:-1].strip()))
            what = {"cut": "invented the cut glyph", "edge": "extra chars (streaks read)", "touch": "dropped a whole end glyph"}[name]
            print(f"{ck:16s} {name:5s} n={len(data)} exact {ok / len(data):.3f}  {what} {invent / len(data):.3f}")


def cut_invented(p, lab, full):
    """Output has a character where the cut glyph was: the full text, or lab with one more char."""
    return p == full or (len(p) == len(lab) + 1 and (p[1:] == lab or p[:-1] == lab))


if __name__ == "__main__":
    main()
