"""Fold BatchNorm, quantize weights to int8 (per output channel) and write
web/kigen.bin for the dependency-free JS runtime. Also verifies the numpy
reference of the quantized net against PyTorch on synthetic validation lines.

uv run python export.py
"""
import json
import random
import struct

import numpy as np
import torch

import gen
import train


@torch.no_grad()
def fold(conv, bn):
    w = conv.weight.detach().double()
    b = conv.bias.detach().double() if conv.bias is not None else torch.zeros(w.shape[0], dtype=torch.double)
    s = bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)
    w = w * s.view(-1, *[1] * (w.dim() - 1))
    b = (b - bn.running_mean.double()) * s + bn.bias.double()
    return w.float().numpy(), b.float().numpy()


def layers(model):
    L = []
    for m in model.cnn:
        if isinstance(m, torch.nn.Sequential):
            w, b = fold(m[0], m[1])
            L.append(dict(op="conv2d", w=w, b=b, pad=m[0].padding))
        else:
            L.append(dict(op="maxpool", k=m.kernel_size if isinstance(m.kernel_size, tuple) else (2, 2)))
    for r in model.seq:
        w, b = fold(r.conv, r.bn)
        L.append(dict(op="res1d", w=w, b=b, dil=r.conv.dilation[0]))
    L.append(dict(op="fc", w=model.fc.weight.detach().numpy(), b=model.fc.bias.detach().numpy()))
    return L


def quant(w):
    o = w.shape[0]
    flat = w.reshape(o, -1)
    sc = np.abs(flat).max(1) / 127 + 1e-12
    q = np.clip(np.round(flat / sc[:, None]), -127, 127).astype(np.int8)
    return q, sc.astype(np.float32)


def write(L, path):
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
    head = json.dumps({"vocab": gen.VOCAB, "h": gen.IMG_H, "maxw": gen.MAX_W, "layers": meta},
                      ensure_ascii=False).encode()
    head += b" " * ((4 - (len(head) + 4) % 4) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(head)) + head + bytes(blob))
    return 4 + len(head) + len(blob)


def np_forward(L, x):
    """Reference of the JS runtime. x: (32,W) float32 -> (T,V) logits."""
    a = x[None]
    for l in L:
        if l["op"] == "conv2d":
            w, b = l["wq"], l["b"]
            ph, pw = l["pad"]
            ap = np.pad(a, ((0, 0), (ph, ph), (pw, pw)))
            O, C, kh, kw = w.shape
            H, W = ap.shape[1] - kh + 1, ap.shape[2] - kw + 1
            out = np.zeros((O, H, W), np.float32) + b[:, None, None]
            for i in range(kh):
                for j in range(kw):
                    out += np.einsum("oc,chw->ohw", w[:, :, i, j], ap[:, i:i + H, j:j + W])
            a = np.maximum(out, 0)
        elif l["op"] == "maxpool":
            kh, kw = l["k"]
            C, H, W = a.shape
            a = a[:, :H // kh * kh, :W // kw * kw].reshape(C, H // kh, kh, W // kw, kw).max((2, 4))
        elif l["op"] == "res1d":
            if a.ndim == 3:
                a = a[:, 0]
            d = l["dil"]
            ap = np.pad(a, ((0, 0), (d, d)))
            T = a.shape[1]
            out = l["b"][:, None] + sum(l["wq"][:, :, k] @ ap[:, k * d:k * d + T] for k in range(3))
            a = np.maximum(a + out, 0)
        else:
            a = l["wq"][:, :, 0] @ a + l["b"][:, None]
    return a.T


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model.pt")
    ap.add_argument("--out", default="web/kigen.bin")
    ap.add_argument("--fixture", default="web/fixture.json")
    args = ap.parse_args()
    model = train.Net()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    L = layers(model)
    n = write(L, args.out)
    print(f"{args.out} {n / 1024:.0f} KB")
    random.seed(777)
    np.random.seed(777)
    ok_t = ok_q = 0
    N = 120
    for _ in range(N):
        x, t = gen.sample()
        with torch.no_grad():
            pt = train.decode(model(torch.from_numpy(x)[None, None]))[0]
        pq = train.decode(torch.from_numpy(np_forward(L, x))[None])[0]
        ok_t += pt == t
        ok_q += pq == t
    print(f"line-acc torch {ok_t / N:.3f}  int8 {ok_q / N:.3f}")
    # fixture for the JS test
    random.seed(5)
    np.random.seed(5)
    fx = []
    for _ in range(8):
        x, t = gen.sample()
        fx.append({"x": np.round(x, 6).flatten().tolist(), "w": x.shape[1], "text": t,
                   "pred": train.decode(torch.from_numpy(np_forward(L, x))[None])[0]})
    json.dump(fx, open(args.fixture, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
