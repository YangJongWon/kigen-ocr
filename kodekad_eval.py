"""Korean 10-day parts (2026년 3월 하순 / 2026.03 상순 / 26년 2월말 / 2026.04 초), same lines for every
checkpoint: date digits + part right, and parts moved later (상순/초 read as 하순/말: the expiry
20 days late). Half-width, spaced, full-width, full-width + U+3000, full-width letters.

  KIGEN_EDGE=0 uv run python kodekad_eval.py model_ko.pt [--n 600]
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
PARTS = ["상순", "중순", "하순", "초순", "초", "말", "월말", "말일"]


def part_of(s):
    """early / mid / late / None from a Korean (or Japanese) part mark."""
    if re.search(r"상순|초순|월초|上旬|初旬|月初|(?<![가-힣])초(?![가-힣])|初", s):
        return "early"
    if re.search(r"중순|中旬|(?<![가-힣])중(?![가-힣])|中", s):
        return "mid"
    if re.search(r"하순|월말|말일|下旬|月末|(?<![가-힣])말(?![가-힣])|末", s):
        return "late"
    return None


def text_for(i):
    y, m = random.randint(2025, 2031), random.randint(1, 12)
    part = PARTS[i % len(PARTS)]
    f = (i // len(PARTS)) % 4
    sp = " " if random.random() < 0.6 else ""
    s = [f"{y}년 {m:02d}월{sp}{part}", f"{y}.{m:02d}{sp}{part}", f"{y % 100}년 {m}월{sp}{part}", f"{y}-{m:02d} {part}"][f]
    return random.choice(["", "소비기한 ", "유통기한 "]) + s


def notation(t, st):
    if st == "FW letters":
        t = random.choice(["EXP ", "BB "]) + t.split(" ", 1)[-1] if t.startswith(("소비", "유통")) else random.choice(["EXP ", "BB "]) + t
    lab = gen.norm_label(t)
    s = t
    if st == "half spaced":
        s = re.sub(r"([.\-])(?=\d)", r"\1 ", s)
    if st in ("full", "full+U3000"):
        s = "".join(FW.get(c, c) for c in s)
        if st == "full+U3000":
            s = s.replace(" ", "　")
    if st == "FW letters":
        s = "".join(FW[c] if c.isascii() and c.isalpha() else c for c in s)
    return s, lab


def lines(n, seed):
    out, i = [], 0
    while len(out) < n:
        random.seed(seed * 100003 + i)
        np.random.seed((seed * 100003 + i) % 2 ** 32)
        st = STY[i % 5]
        vis, lab = notation(text_for(i // 5), st)
        img = gen.render(vis)
        i += 1
        if img is not None:
            out.append((gen.to_input(img), lab, st))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    gen.ML_P = 0.0
    data = lines(a.n, 11)
    for ck in a.ckpts:
        m = train.Net()
        sd = torch.load(ck, map_location="cpu")
        if sd["fc.weight"].shape[0] != len(gen.VOCAB):
            print(f"== {ck}: vocabulary {sd['fc.weight'].shape[0]} (no Korean part marks), skipped")
            continue
        m.load_state_dict(sd)
        m.eval()
        ok = later = earlier = 0
        per, shown = {}, 0
        for i in range(0, len(data), 64):
            ch = data[i:i + 64]
            with torch.no_grad():
                pr = train.decode(m(train.collate([(x, t) for x, t, _ in ch])[0]))
            for p, (_, t, st) in zip(pr, ch):
                pt, pp = part_of(t), part_of(p)
                good = re.findall(r"\d+", p) == re.findall(r"\d+", t) and pt == pp
                ok += good
                order = {"early": 0, "mid": 1, "late": 2}
                if pt and pp and order[pp] > order[pt]:
                    later += 1
                if pt and pp and order[pp] < order[pt]:
                    earlier += 1
                s = per.setdefault(st, [0, 0])
                s[0] += good
                s[1] += 1
                if not good and shown < a.show:
                    print(f"   {st} want {t!r} got {p!r}")
                    shown += 1
        print(f"== {ck}\n   date+part {ok / len(data):.3f}  moved later {later}  moved earlier {earlier}  ("
              + ", ".join(f"{k} {v[0] / v[1]:.2f}" for k, v in per.items()) + ")")


if __name__ == "__main__":
    main()
