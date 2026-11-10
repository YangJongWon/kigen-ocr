"""Train a tiny CTC line recognizer on synthetic expiry-date lines.

uv run python train.py [--steps 40000]
"""
import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gen

V = len(gen.VOCAB)


def cbr(i, o, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, p, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Res1d(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.conv = nn.Conv1d(c, c, 3, padding=d, dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(c)

    def forward(self, x):
        return F.relu(x + self.bn(self.conv(x)))


class Net(nn.Module):
    """(B,1,IMG_H,W) -> (B,W/4,V) logits. Four 2x poolings leave IMG_H/16 rows; the last conv folds them."""

    def __init__(self, n=V, c=160):
        super().__init__()
        self.cnn = nn.Sequential(
            cbr(1, 32), nn.MaxPool2d(2),                        # 16 x W/2
            cbr(32, 64), nn.MaxPool2d(2),                       # 8 x W/4
            cbr(64, 96), cbr(96, 96), nn.MaxPool2d((2, 1)),     # 4
            cbr(96, 128), cbr(128, 128), nn.MaxPool2d((2, 1)),  # 2
            cbr(128, c, (gen.IMG_H // 16, 1), 1, 0))            # 1
        self.seq = nn.Sequential(Res1d(c, 1), Res1d(c, 2), Res1d(c, 4), Res1d(c, 1))
        self.fc = nn.Conv1d(c, n, 1)

    def forward(self, x):
        f = self.cnn(x).squeeze(2)          # B,C,T
        return self.fc(self.seq(f)).permute(0, 2, 1)  # B,T,V


def batches(bs, workers):
    """Endless batches from a torch-free process pool (bounded prefetch)."""
    import collections
    import multiprocessing as mp
    pool = mp.get_context("spawn").Pool(workers)
    seed = int(time.time())
    q = collections.deque()
    while True:
        while len(q) < workers * 2:
            seed += 1
            q.append(pool.apply_async(gen.make_batch, ((seed, bs),)))
        yield collate(q.popleft().get())


def collate(batch):
    W = max(x.shape[1] for x, _ in batch)
    W = (W + 3) // 4 * 4
    xs = np.stack([np.pad(x, ((0, 0), (0, W - x.shape[1])), mode="edge") for x, _ in batch])
    ys = [torch.tensor([gen.C2I[c] for c in t]) for _, t in batch]
    return (torch.from_numpy(xs)[:, None], torch.cat(ys), torch.tensor([len(y) for y in ys]),
            [t for _, t in batch])


def decode(logits):
    out = []
    for seq in logits.argmax(-1).cpu().numpy():
        s, prev = [], 0
        for k in seq:
            if k != prev and k != 0:
                s.append(gen.VOCAB[k])
            prev = k
        out.append(" ".join("".join(s).split()))
    return out


def cer(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]


def evaluate(model, val, dev):
    model.eval()
    ok = errs = tot = 0
    with torch.no_grad():
        for i in range(0, len(val), 256):
            x, _, _, txt = collate(val[i:i + 256])
            pred = decode(model(x.to(dev)))
            for p, t in zip(pred, txt):
                ok += p == t
                errs += cer(p, t)
                tot += len(t)
    model.train()
    return ok / len(val), errs / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--init", default=None)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="model.pt")
    args = ap.parse_args()
    dev = "cuda"
    random.seed(12345)
    np.random.seed(12345)
    val = [gen.sample() for _ in range(2000)]
    model = Net().to(dev)
    if args.init:
        # layers whose shape changed (e.g. the last conv when IMG_H changes) start fresh
        sd, own = torch.load(args.init, map_location=dev), model.state_dict()
        keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        print("init from", args.init, f"({len(keep)}/{len(own)} tensors)")
    print("params", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    ctc = nn.CTCLoss(zero_infinity=True)
    best, t0, step = -1, time.time(), 0
    for x, y, yl, _ in batches(args.bs, args.workers):
        x = x.to(dev, non_blocking=True)
        lp = model(x).log_softmax(-1).permute(1, 0, 2)  # T,B,V
        il = torch.full((x.shape[0],), lp.shape[0], dtype=torch.long)
        loss = ctc(lp.float(), y, il, yl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        sched.step()
        step += 1
        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        if step % 2000 == 0 or step == args.steps:
            acc, c = evaluate(model, val, dev)
            print(f"== step {step} val line-acc {acc:.4f} CER {c:.4f}", flush=True)
            if acc > best:
                best = acc
                torch.save(model.state_dict(), args.out)
        if step >= args.steps:
            break
    print("best", best)


if __name__ == "__main__":
    main()
