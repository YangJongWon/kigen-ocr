"""Real-photo frames for the date-line detector (MIT datasets, boxes only — no transcriptions).

datasets/expiry-date-detection  Roboflow (MIT), 640x640 stretched, COCO boxes. Files named img_* / test_*
                                 are ExpDate photos (license not settled; test_* is our evaluation set):
                                 left out.
datasets/expiry-date-2          Roboflow (MIT), 152 phone photos, 2 boxes each (labels.json)

Each sample is a camera-guide-like crop (aspect 3.5-8, date filling 1/1.3-1/3.5 of its height, aimed
off-centre) around one box, degraded like detgen.photo and turned into detector input.
Boxes on these photos are not told apart (expiry vs manufacturing): a crop that also holds a second
box is skipped, so the detector is not taught to pick one at random.

uv run python realdet.py   -> samples/realdet_sheet.png
"""
import json
import os
import random

import cv2
import numpy as np

import detgen

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "datasets", "_cache_det")
_items = {}
LOOSE = ("loose",)  # 5th box element: label drawn with margin (detect.collate ignores its edges)
IGNORE = ("ignore",)  # 5th box element: a date of unknown kind (expiry or manufacturing): not taught


def _load(split):
    """[(path, [(x0, y0, x1, y1), ...])], split 'train' or 'val'."""
    if split in _items:
        return _items[split]
    out = []
    base = os.path.join(ROOT, "datasets", "expiry-date-detection")
    for s in (["train"] if split == "train" else ["valid", "test"]):
        d = json.load(open(os.path.join(base, s, "_annotations.coco.json")))
        boxes = {}
        for a in d["annotations"]:
            x, y, w, h = a["bbox"]
            boxes.setdefault(a["image_id"], []).append((x, y, x + w, y + h))
        for im in d["images"]:
            if im["file_name"].startswith(("img_", "test_")) or im["id"] not in boxes:
                continue
            out.append((os.path.join(base, s, im["file_name"]), boxes[im["id"]]))
    # expiry-date-2: 2250x4000 photos, cached at half size
    base = os.path.join(ROOT, "datasets", "expiry-date-2")
    d = json.load(open(os.path.join(base, "labels.json")))
    os.makedirs(CACHE, exist_ok=True)
    for im in d["images"]:
        if (im["split"] == "train") != (split == "train"):
            continue
        p = os.path.join(CACHE, im["name"])
        if not os.path.exists(p):
            g = cv2.imread(os.path.join(base, "images", im["name"]))
            cv2.imwrite(p, cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA))
        bs = [((b["x"] - b["w"] / 2) / 2, (b["y"] - b["h"] / 2) / 2, (b["x"] + b["w"] / 2) / 2, (b["y"] + b["h"] / 2) / 2)
              for b in im["boxes"]]
        out.append((p, bs))
    _items[split] = out
    return out


def _crop(img, x, y, w, h):
    """Crop with edge replication outside the photo."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
    pad = max(0, -x0, -y0, x1 - W, y1 - H)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return img[y0 + pad:y1 + pad, x0 + pad:x1 + pad]


def _map(M, box):
    x0, y0, x1, y1 = box
    p = np.float32([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]]) @ np.float32(M).T
    return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()


def tilted(img, box, cx, cy, fw, fh, th, loose=()):
    """What the detector sees when the label is tilted by th degrees against a fw x fh camera guide
    centred on (cx, cy): the camera frame (degraded like detgen.photo), then readGray's straightening
    (rotateGray clamps -> replicated edges, streaky corners) and its frame-size centre crop
    (detCentre). Below 3 degrees readGray does not straighten. Returns detector input, box."""
    fw, fh = int(round(fw)), int(round(fh))
    M1 = cv2.getRotationMatrix2D((cx, cy), th, 1.0)
    M1[:, 2] += (fw / 2 - cx, fh / 2 - cy)
    f = cv2.warpAffine(img, M1, (fw, fh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    f, s = detgen.photo(f)
    M = np.vstack([M1 * s, [0, 0, 1]])  # photo coords -> degraded frame (box corners mapped once)
    est = th + random.uniform(-2, 2)  # estimateSkew is not exact
    if abs(est) >= 3:
        h, w = f.shape[:2]
        M2 = cv2.getRotationMatrix2D((w / 2, h / 2), -est, 1.0)
        f = cv2.warpAffine(f, M2, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        M = np.vstack([M2, [0, 0, 1]]) @ M
    if loose:
        # a loose label is a box drawn around flat text: keep its size, move only its centre (its
        # rotated bounding box would be taller than the glyphs and teach tall boxes)
        cx2, cy2 = (M[:2] @ np.float32([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 1]))
        hw, hh = (box[2] - box[0]) * s / 2, (box[3] - box[1]) * s / 2
        b = (cx2 - hw, cy2 - hh, cx2 + hw, cy2 + hh)
    else:
        b = _map(M[:2], box)
    h, w = f.shape[:2]
    b = (max(0, b[0]), max(0, b[1]), min(w, b[2]), min(h, b[3]))
    if b[2] - b[0] < 4 or b[3] - b[1] < 2:
        return None
    x, ix, iy = detgen.to_input(f)
    return x, (b[0] * ix, b[1] * iy, b[2] * ix, b[3] * iy) + loose


def tilt_angle():
    return random.choice([-1, 1]) * random.choices([random.uniform(2, 8), random.uniform(8, 20), random.uniform(20, 40)], [3, 4, 3])[0]


def synth_tilted():
    """detgen scene seen through a tilted guide of the scene's own size."""
    while True:
        img, box = detgen.scene()
        if box is None:
            continue
        img = np.clip(img, 0, 255).astype(np.uint8)
        h, w = img.shape[:2]
        r = tilted(img, box, w / 2, h / 2, w, h, tilt_angle())
        if r:
            return r


def sample(split="train"):
    items = _load(split)
    while True:
        path, boxes = random.choice(items)
        # the 640 set was augmented with +-15 deg rotations: an axis-aligned box around tilted text is
        # tall and loose (teaches boxes that cut tilted dates). Keep only near-flat boxes there.
        lo = 3.0 if "expiry-date-detection" in path else 1.5
        hor = [b for b in boxes if b[2] - b[0] >= lo * (b[3] - b[1]) and b[3] - b[1] >= 6]
        if not hor:
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # the 640x640 set was stretched: undo / vary the aspect a little
        sx = random.uniform(0.75, 1.35)
        x0, y0, x1, y1 = random.choice(hor)
        # several dates on the photo (expiry + manufacturing, not told apart in these labels): the
        # one in the frame may be either, so it is neither taught as the date nor as background
        kind = IGNORE if len(boxes) > 1 else LOOSE
        bh, bw = y1 - y0, x1 - x0
        neg = random.random() < 0.1
        tilt = not neg and random.random() < 0.4
        fh = bh * random.choices([random.uniform(1.25, 1.8), random.uniform(1.8, 2.6), random.uniform(2.6, 3.6)], [4, 4, 2])[0]
        fw = max(fh * (random.uniform(1.8, 8) if tilt else random.uniform(3.5, 8)) / sx, bw * random.uniform(1.05, 1.3))
        cx = (x0 + x1) / 2 + random.uniform(-0.5, 0.5) * max(0, fw - bw * 1.05)
        cy = (y0 + y1) / 2 + random.uniform(-0.5, 0.5) * max(0, fh - bh * 1.15)
        if neg:  # same scale, shifted off every box (above or below)
            cy += random.choice([-1, 1]) * (fh + bh) * random.uniform(0.8, 1.5)
        # keep the frame inside the photo (edge replication streaks look nothing like a camera frame)
        H, W = img.shape[:2]
        fw, fh = min(fw, W), min(fh, H)
        if fh < bh * 1.1 or fw < bw * 0.9:
            continue
        fx = min(max(0, cx - fw / 2), W - fw)
        fy = min(max(0, cy - fh / 2), H - fh)
        # any box (other than the target) more than a third inside the frame: ambiguous, skip
        def inside(b):
            ix = max(0, min(b[2], fx + fw) - max(b[0], fx))
            iy = max(0, min(b[3], fy + fh) - max(b[1], fy))
            return ix * iy / max((b[2] - b[0]) * (b[3] - b[1]), 1)
        others = [b for b in boxes if b != (x0, y0, x1, y1)]
        if neg and any(inside(b) > 0.05 for b in boxes):
            continue
        if not neg and any(inside(b) > 0.33 for b in others):
            continue
        if tilt:
            im2 = cv2.resize(img, (max(16, int(img.shape[1] * sx)), img.shape[0]), interpolation=cv2.INTER_AREA)
            r = tilted(im2, (x0 * sx, y0, x1 * sx, y1), (fx + fw / 2) * sx, fy + fh / 2, fw * sx, fh, tilt_angle(), kind)
            if r:
                return r
            continue
        f = _crop(img, fx, fy, fw, fh)
        if f.size == 0 or f.shape[0] < 8 or f.shape[1] < 16:
            continue
        f = cv2.resize(f, (max(16, int(f.shape[1] * sx)), f.shape[0]), interpolation=cv2.INTER_AREA)
        box = None if neg else (max(0, (x0 - fx) * sx), max(0, y0 - fy), min(f.shape[1], (x1 - fx) * sx), min(f.shape[0], y1 - fy))
        if box and (box[2] - box[0] < 4 or box[3] - box[1] < 2):
            continue
        if random.random() < 0.5:
            f, box = detgen.warp(f, box)
        f, s = detgen.photo(f)
        x, ix, iy = detgen.to_input(f)
        if box:
            box = (box[0] * s * ix, box[1] * s * iy, box[2] * s * ix, box[3] * s * iy) + kind
        return x, box


def make_batch(args):
    """Mixed batch: share `real` of real-photo frames, the rest synthetic."""
    seed, n, real = args
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    out = []
    for _ in range(n):
        if random.random() < real:
            out.append(sample())
        else:  # synthetic; some through a tilted guide
            out.append(synth_tilted() if random.random() < 0.3 else detgen.sample())
    return out


if __name__ == "__main__":
    os.makedirs("samples", exist_ok=True)
    random.seed(3)
    np.random.seed(3)
    print("train", len(_load("train")), "val", len(_load("val")))
    rows = []
    for i in range(30):
        x, b = sample() if i < 20 else synth_tilted()
        v = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if b:
            col = (255, 0, 0) if b[4:] == IGNORE else (0, 0, 255)
            cv2.rectangle(v, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 1)
        rows.append(v)
    W = max(r.shape[1] for r in rows)
    sheet = np.full((len(rows) * (detgen.DET_H + 4), W, 3), 255, np.uint8)
    for i, r in enumerate(rows):
        sheet[i * (detgen.DET_H + 4):i * (detgen.DET_H + 4) + detgen.DET_H, :r.shape[1]] = r
    cv2.imwrite("samples/realdet_sheet.png", sheet)
