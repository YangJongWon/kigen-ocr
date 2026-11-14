"""Date-line detector: camera-guide frame -> where the expiry-date line is.

Fully convolutional. Input (1, 48, W) gray in [0,1] (detgen.to_input); output one logit per
4x4 cell (12 x W/4): the share of the cell covered by the date line's box. A box is decoded
from the cells around the strongest one (box_from_map, mirrored in kigen-ocr.js).

uv run python run_det.py train [--steps N]       (run_det keeps torch out of the data workers)
uv run python run_det.py export [--ckpt det.pt]  -> web/kigen-det.bin
uv run python run_det.py eval [--ckpt det.pt]    synthetic hit rate
"""
import argparse
import json
import random
import struct
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import detgen
from export import fold, quant


def cbr(i, o, k=(3, 3), d=(1, 1)):
    p = ((k[0] - 1) // 2 * d[0], (k[1] - 1) // 2 * d[1])
    return nn.Sequential(nn.Conv2d(i, o, k, 1, p, dilation=d, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Det(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.body = nn.Sequential(
            cbr(1, 12), nn.MaxPool2d(2),          # 24 x W/2
            cbr(12, 24), nn.MaxPool2d(2),         # 12 x W/4
            cbr(24, c), cbr(c, c),
            # wide horizontal context: is this run of glyphs shaped like a date?
            cbr(c, c, (1, 3), (1, 2)), cbr(c, c, (1, 3), (1, 4)), cbr(c, c, (1, 3), (1, 8)), cbr(c, c, (1, 3), (1, 16)),
            cbr(c, c))
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.out(self.body(x))[:, 0]  # B, H/4, W/4


def target(box, W):
    """Coverage of each 4x4 cell by box (x0, y0, x1, y1) -> (12, W/4)."""
    t = np.zeros((detgen.DET_H // 4, W // 4), np.float32)
    if box is None:
        return t
    x0, y0, x1, y1 = box
    cx = np.arange(W // 4) * 4
    cy = np.arange(detgen.DET_H // 4) * 4
    fx = np.clip(np.minimum(cx + 4, x1) - np.maximum(cx, x0), 0, 4) / 4
    fy = np.clip(np.minimum(cy + 4, y1) - np.maximum(cy, y0), 0, 4) / 4
    return fy[:, None] * fx[None, :]


def collate(batch):
    W = max(x.shape[1] for x, _ in batch)
    xs, ts, ms = [], [], []
    for x, b in batch:
        w = x.shape[1]
        xs.append(np.pad(x, ((0, 0), (0, W - w)), mode="edge"))
        m = np.zeros((detgen.DET_H // 4, W // 4), np.float32)
        m[:, : w // 4] = 1
        if b is not None and b[4:] == ("ignore",):
            m[target(b[:4], W) > 0] = 0  # a date of unknown kind: neither date nor background
            ts.append(np.zeros_like(m))
        elif b is not None and len(b) == 5:
            # loose box (real-photo label, drawn with margin): only its core is taught as date, the
            # margin is ignored, so how tightly a box hugs the glyphs is learnt from the synthetic data
            x0, y0, x1, y1 = b[:4]
            dy = 0.25 * (y1 - y0)
            inner = (x0 + dy, y0 + dy, x1 - dy, y1 - dy)
            t, outer = target(inner, W), target(b[:4], W)
            m[(outer > 0) & (t < 1)] = 0
            ts.append(t)
        else:
            ts.append(target(b, W))
        ms.append(m)
    return (torch.from_numpy(np.stack(xs))[:, None], torch.from_numpy(np.stack(ts)),
            torch.from_numpy(np.stack(ms)), [b for _, b in batch])


def batches(bs, workers, real=0.0):
    import collections
    import multiprocessing as mp
    pool = mp.get_context("spawn").Pool(workers)
    seed = int(time.time())
    q = collections.deque()
    while True:
        while len(q) < workers * 2:
            seed += 1
            if real > 0:  # mix in real-photo frames (realdet.py)
                import realdet
                q.append(pool.apply_async(realdet.make_batch, ((seed, bs, real),)))
            else:
                q.append(pool.apply_async(detgen.make_batch, ((seed, bs),)))
        yield collate(q.popleft().get())


def box_from_map(p, thr=0.5):
    """p: (12, Wc) cell probabilities -> (x0, y0, x1, y1) in input pixels, score; or (None, score).
    The component of cells > thr/2 holding the strongest cell (horizontal gaps of one cell bridged);
    its size from the summed coverage of its row / column profiles (sub-cell precise)."""
    Hc, Wc = p.shape
    i, j = np.unravel_index(np.argmax(p), p.shape)
    s = float(p[i, j])
    if s < thr:
        return None, s
    on = p > thr / 2
    seen = np.zeros_like(on)
    st, comp = [(i, j)], []
    seen[i, j] = True
    while st:
        a, b = st.pop()
        comp.append((a, b))
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 2), (0, -2)):
            u, v = a + da, b + db
            if 0 <= u < Hc and 0 <= v < Wc and on[u, v] and not seen[u, v]:
                seen[u, v] = True
                st.append((u, v))
    rs = sorted({a for a, _ in comp})
    cs = sorted({b for _, b in comp})
    r0, r1, c0, c1 = rs[0], rs[-1], cs[0], cs[-1]
    sub = p[r0:r1 + 1, c0:c1 + 1]
    rowv, colv = sub.max(1), sub.max(0)
    h = 4 * rowv.sum()
    w = 4 * colv.sum()
    cy = 4 * ((np.arange(r0, r1 + 1) + 0.5) * rowv).sum() / rowv.sum()
    cx = 4 * ((np.arange(c0, c1 + 1) + 0.5) * colv).sum() / colv.sum()
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), s


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / max(u, 1e-6)


def evaluate(model, val, dev):
    """hit: box with IoU >= 0.5 (date frames); fp: a box on a no-date frame; also the share of date
    frames whose box covers the date's full height (what the recognizer needs)."""
    model.eval()
    hit = cover = n = fp = neg = found = 0
    with torch.no_grad():
        for k in range(0, len(val), 64):
            x, _, _, boxes = collate(val[k:k + 64])
            p = torch.sigmoid(model(x.to(dev))).cpu().numpy()
            for pi, (xi, b) in zip(p, val[k:k + 64]):
                if b is not None and b[4:] == ("ignore",):
                    continue  # unknown kind: not scored
                b = None if b is None else b[:4]
                pb, _ = box_from_map(pi[:, : xi.shape[1] // 4])
                if b is None:
                    neg += 1
                    fp += pb is not None
                    continue
                n += 1
                if pb is not None:
                    hit += iou(pb, b) >= 0.5
                    # for loose (real) labels: centre inside the label, not taller than it
                    found += b[0] <= (pb[0] + pb[2]) / 2 <= b[2] and b[1] <= (pb[1] + pb[3]) / 2 <= b[3]                         and pb[3] - pb[1] <= 1.1 * (b[3] - b[1])
                    cover += pb[1] <= b[1] + 0.15 * (b[3] - b[1]) and pb[3] >= b[3] - 0.15 * (b[3] - b[1]) \
                        and pb[0] <= b[0] + 0.1 * (b[2] - b[0]) and pb[2] >= b[2] - 0.1 * (b[2] - b[0])
    model.train()
    return hit / max(n, 1), cover / max(n, 1), fp / max(neg, 1), found / max(n, 1)


def train(args):
    dev = "cuda"
    random.seed(4242)
    np.random.seed(4242)
    val = [detgen.sample() for _ in range(1500)]
    vreal = None
    if args.real > 0:
        import realdet
        vreal = [realdet.sample("val") for _ in range(1500)]
    model = Det().to(dev)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev))
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    best, t0, step = -1, time.time(), 0
    for x, t, m, _ in batches(args.bs, args.workers, args.real):
        x, t, m = x.to(dev, non_blocking=True), t.to(dev), m.to(dev)
        z = model(x)
        bce = (F.binary_cross_entropy_with_logits(z, t, reduction="none") * m).sum() / m.sum()
        # soft dice per frame: the date is a few cells among hundreds
        p = torch.sigmoid(z) * m
        inter, den = (p * t).sum((1, 2)), (p + t).sum((1, 2))
        dice = (1 - (2 * inter + 1) / (den + 1)).mean()
        loss = bce * 4 + dice
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        sched.step()
        step += 1
        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.3f} bce {bce.item():.3f} dice {dice.item():.3f} {time.time() - t0:.0f}s", flush=True)
        if step % 2000 == 0 or step == args.steps:
            hit, cov, fp, _ = evaluate(model, val, dev)
            msg = f"== step {step} hit {hit:.4f} cover {cov:.4f} fp-on-empty {fp:.3f}"
            score = hit
            if vreal:
                rh, rc, rf, rfd = evaluate(model, vreal, dev)
                msg += f" | real hit {rh:.4f} found {rfd:.4f} fp {rf:.3f}"
                score = (hit + rfd) / 2  # IoU with loose labels would favour loose boxes
            print(msg, flush=True)
            if score > best:
                best = score
                torch.save(model.state_dict(), args.out)
        if step >= args.steps:
            break
    print("best", best)


def layers(model):
    L = []
    for m in list(model.body):
        if isinstance(m, nn.Sequential):
            w, b = fold(m[0], m[1])
            L.append(dict(op="conv2d", w=w, b=b, pad=list(m[0].padding), dil=m[0].dilation[1], relu=True))
        else:
            L.append(dict(op="maxpool", k=[2, 2]))
    L.append(dict(op="conv2d", w=model.out.weight.detach().numpy(), b=model.out.bias.detach().numpy(), pad=[0, 0], dil=1, relu=False))
    return L


def np_forward(L, x):
    """Reference of the JS detector (int8-dequantized weights, float activations). x (48, W) -> (12, W/4) logits."""
    a = x[None]
    for l in L:
        if l["op"] == "maxpool":
            C, H, W = a.shape
            a = a[:, :H // 2 * 2, :W // 2 * 2].reshape(C, H // 2, 2, W // 2, 2).max((2, 4))
            continue
        w, b, (ph, pw), d = l["wq"], l["b"], l["pad"], l["dil"]
        ap = np.pad(a, ((0, 0), (ph, ph), (pw, pw)))
        O, C, kh, kw = w.shape
        H, W = ap.shape[1] - kh + 1, ap.shape[2] - d * (kw - 1)
        out = np.zeros((O, H, W), np.float32) + b[:, None, None]
        for i in range(kh):
            for j in range(kw):
                out += np.einsum("oc,chw->ohw", w[:, :, i, j], ap[:, i:i + H, j * d:j * d + W])
        a = np.maximum(out, 0) if l["relu"] else out
    return a[0]


def export(args):
    model = Det()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    L = layers(model)
    meta, blob = [], bytearray()
    for l in L:
        e = {k: v for k, v in l.items() if k not in ("w", "b")}
        if "w" in l:
            q, sc = quant(l["w"])
            l["wq"] = (q.astype(np.float32) * sc[:, None]).reshape(l["w"].shape)
            e["shape"] = list(l["w"].shape)
            for name, arr in (("q", q), ("s", sc), ("b", l["b"].astype(np.float32))):
                while len(blob) % 4:
                    blob.append(0)
                e[name] = [len(blob), arr.size]
                blob += arr.tobytes()
        meta.append(e)
    head = json.dumps({"kind": "det", "h": detgen.DET_H, "maxw": detgen.DET_MAXW, "layers": meta}).encode()
    head += b" " * ((4 - (len(head) + 4) % 4) % 4)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<I", len(head)) + head + bytes(blob))
    print(f"{args.out} {(4 + len(head) + len(blob)) / 1024:.0f} KB")
    # torch vs int8 reference, and a fixture for the JS test
    random.seed(99)
    np.random.seed(99)
    fx, agree = [], 0
    for k in range(40):
        x, b = detgen.sample()
        with torch.no_grad():
            pt = torch.sigmoid(model(torch.from_numpy(x)[None, None]))[0].numpy()
        z = np_forward(L, x)
        pq = 1 / (1 + np.exp(-z))
        bt, _ = box_from_map(pt)
        bq, sq = box_from_map(pq)
        agree += (bt is None) == (bq is None) and (bt is None or iou(bt, bq) > 0.8)
        if k < 6:
            fx.append({"x": np.round(x, 6).flatten().tolist(), "w": x.shape[1], "logits": np.round(z, 4).flatten().tolist(),
                       "box": None if bq is None else [float(v) for v in bq], "score": sq})
    print(f"torch vs int8 boxes agree {agree}/40")
    json.dump(fx, open("web/fixture_det.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "export", "eval"])
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--init", default=None)
    ap.add_argument("--real", type=float, default=0.0, help="share of real-photo frames (realdet.py)")
    ap.add_argument("--out", default="det.pt")
    ap.add_argument("--ckpt", default="det.pt")
    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "export":
        args.out = "web/kigen-det.bin" if args.out == "det.pt" else args.out
        export(args)
    else:
        model = Det().cuda()
        model.load_state_dict(torch.load(args.ckpt))
        random.seed(4242)
        np.random.seed(4242)
        val = [detgen.sample() for _ in range(1500)]
        print("hit / cover / fp-on-empty / found", evaluate(model, val, "cuda"))
        import realdet
        vreal = [realdet.sample("val") for _ in range(1500)]
        print("real:", evaluate(model, vreal, "cuda"))


if __name__ == "__main__":
    main()
