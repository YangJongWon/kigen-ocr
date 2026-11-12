"""Synthetic camera-guide frames for the date-line detector (detect.py).

A frame is what readRegion receives: the guide box over packaging. It holds one expiry-date line
(gen.rand_text, any style gen renders) somewhere inside, among things a rule cannot tell from it:
dense small print, label lines (賞味期限 on its own line), other numbers (price, weight, JAN code,
phone, lot), big brand text, barcodes, borders. ~10% of frames hold no date.

sample() -> (x float32 (DET_H, W) in [0,1], box (x0, y0, x1, y1) in x's pixels or None)
python detgen.py   writes samples/det_sheet.png
"""
import math
import random
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

import gen

DET_H = 48            # detector input height
DET_MAXW = DET_H * 10
SH = 96               # scene height while composing

KANJI = gen.CLUTTER + "賞味期限消費開封後お早めにお召し上がりください冷蔵要一〇度以下で保存してください本品製造工場では卵乳成分を含む製品を生産しています"
KATA = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワンガギグゲゴザジズゼゾダデドバビブベボパピプペポー"
LATIN = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def other_number():
    """A line with digits that is not an expiry date (hard negatives)."""
    r = random.random()
    d = lambda n: "".join(random.choice("0123456789") for _ in range(n))
    if r < 0.15:
        return d(13)                                       # JAN barcode digits
    if r < 0.27:
        return random.choice(["¥", "", "税込", "本体価格 "]) + str(random.randint(98, 1980)) + random.choice(["円", "", "円(税込)"])
    if r < 0.39:
        return random.choice(["内容量 ", "", "正味量 ", "NET "]) + str(random.randint(5, 999)) + random.choice(["g", "ml", "mL", "kg", "個"])
    if r < 0.49:
        return random.choice(["TEL ", "フリーダイヤル ", ""]) + random.choice(["0120", "03", "06"]) + "-" + d(3) + "-" + d(4)
    if r < 0.61:
        return random.choice(["LOT ", "LOT:", "", "製造所固有記号 ", "+"]) + gen.rand_code() + random.choice(["", d(2), "-" + d(3)])
    if r < 0.71:
        return random.choice(["エネルギー ", "熱量 ", "たんぱく質 ", "脂質 ", "炭水化物 ", "食塩相当量 "]) + f"{random.uniform(0, 400):.1f}" + random.choice(["kcal", "g"])
    if r < 0.79:
        return f"〒{d(3)}-{d(4)} " + "".join(random.choice(KANJI) for _ in range(random.randint(4, 10)))
    if r < 0.87:
        return d(random.randint(4, 8)) + random.choice(["", " ", "-"]) + d(random.randint(2, 6))  # numbers, no date shape
    return random.choice(["", "1日", "1食", "100g"]) + "当たり " + str(random.randint(1, 99)) + random.choice(["%", "g", "mg"])


def small_line(n):
    return "".join(random.choice(KANJI) for _ in range(n))


def brand():
    if random.random() < 0.5:
        return "".join(random.choice(KATA) for _ in range(random.randint(2, 7)))
    return "".join(random.choice(LATIN) for _ in range(random.randint(3, 8)))


def mask_of(text, h, dot=False, jp=False):
    """Tight white-on-black mask of `text` with glyph height ~h. jp: Japanese fonts only (gen picks
    Chinese ones for 保/存/有, which lack kana and render tofu)."""
    sf = gen.script_fonts
    if jp:
        gen.script_fonts = lambda t: gen.JP_FONTS
    try:
        m = gen.dot_matrix(text, int(h / 0.8)) if dot and gen.dot_ok(text) else gen.line_mask(text, int(h * 1.25))
    finally:
        gen.script_fonts = sf
    ys, xs = np.nonzero(m > 40)
    if len(ys) == 0:
        return None
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = h / m.shape[0]
    if abs(s - 1) > 0.03:
        m = cv2.resize(m, (max(1, int(m.shape[1] * s)), max(1, int(h))), interpolation=cv2.INTER_AREA)
    return m


MFG = re.compile(r"生产|製造|제조|MFG|ผลิต|PROD")


def is_date(label):
    return len(re.findall(r"\d", label)) >= 3 or bool(re.search(r"[〇一二三四五六七八九十]年", label))


class Scene:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.bg = gen.background(h, w)
        self.lum = self.bg.mean()
        self.taken = []  # rects that must not overlap

    def ink(self):
        if self.lum > 110 or random.random() < 0.15:
            return np.array([random.uniform(0, 80)] * 3) + np.random.uniform(0, 50, 3) * (random.random() < 0.3)
        return np.array([random.uniform(190, 255)] * 3)

    def free(self, r, gap=0):
        x0, y0, x1, y1 = r
        return all(x1 + gap <= a or x0 >= c + gap or y1 + gap <= b or y0 >= d + gap for a, b, c, d in self.taken)

    def put(self, m, x, y, ink=None, alpha=None, keep=True):
        """Blend mask m at (x, y) (may hang over the edges). Returns the visible rect or None."""
        mh, mw = m.shape
        x0, y0, x1, y1 = max(0, x), max(0, y), min(self.w, x + mw), min(self.h, y + mh)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        a = (m[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32) / 255)[:, :, None] * (alpha or random.uniform(0.8, 1))
        ink = self.ink() if ink is None else ink
        # legible: ink well apart from the background under it (else the opposite extreme)
        under = self.bg[y0:y1, x0:x1].mean()
        if abs(ink.mean() - under) < 60:
            ink = np.full(3, random.uniform(0, 50) if under > 128 else random.uniform(205, 255))
        self.bg[y0:y1, x0:x1] = self.bg[y0:y1, x0:x1] * (1 - a) + ink * a
        if keep:
            self.taken.append((x0, y0, x1, y1))
        return (x0, y0, x1, y1)


def place(sc, m, tries=20, gap=2, edge=False):
    """Put mask m at a random free spot; edge=True lets it hang over the frame edge."""
    mh, mw = m.shape
    for _ in range(tries):
        if edge:
            x, y = random.randint(-mw // 2, sc.w - mw // 2), random.randint(-mh // 2, sc.h - mh // 2)
        else:
            if mw > sc.w or mh > sc.h:
                return None
            x, y = random.randint(0, sc.w - mw), random.randint(0, sc.h - mh)
        r = (max(0, x), max(0, y), min(sc.w, x + mw), min(sc.h, y + mh))
        if sc.free(r, gap):
            return sc.put(m, x, y)
    return None


def small_block(sc, sh, x0, x1, y0, y1):
    """Rows of dense small print filling [x0,x1) x [y0,y1)."""
    ink, lead = sc.ink(), sh * random.uniform(1.15, 1.6)
    y = y0
    while y + sh <= y1:
        n = max(2, int((x1 - x0) / (sh * random.uniform(0.9, 1.1))))
        m = mask_of(small_line(n), sh, jp=True)
        if m is not None:
            m = m[:, :max(1, x1 - x0)]
            r = (x0, int(y), x0 + m.shape[1], int(y) + m.shape[0])
            if sc.free(r, 1):
                sc.put(m, x0, int(y), ink=ink)
        y += lead


def scene():
    aspect = random.choices([random.uniform(2.2, 4), random.uniform(4, 6), random.uniform(6, 8.5)], [2, 3, 5])[0]
    W, H = int(SH * aspect), SH
    sc = Scene(W, H)
    box = None
    has_date = random.random() > 0.1
    dh = H * random.choices([random.uniform(0.15, 0.3), random.uniform(0.3, 0.5), random.uniform(0.5, 0.8)], [3, 4, 3])[0]
    m = None
    for _ in range(20 if has_date else 0):
        vis, label = gen.rand_text()
        if not is_date(label) or MFG.search(vis):
            continue
        dot = random.random() < 0.4
        m = mask_of(vis, dh, dot)
        if m is not None and m.shape[1] > W * 0.95:  # too long for this frame: smaller print
            nw = int(W * random.uniform(0.4, 0.95))
            m = cv2.resize(m, (nw, max(4, int(m.shape[0] * nw / m.shape[1]))), interpolation=cv2.INTER_AREA)
        if m is not None and m.shape[0] >= 4:
            break
        m = None
    if m is not None:
        if dot and random.random() < 0.5:
            m = gen.dotify(m, vis)
        mh, mw = m.shape
        x = random.randint(0, W - mw) if random.random() > 0.05 else random.choice([-int(mw * 0.15), W - int(mw * 0.85)])
        y = random.randint(0, max(0, H - mh))
        box = sc.put(m, x, y)
    # distractors, in an order that lets big ones claim space first
    items = []
    if random.random() < 0.35:
        items.append("brand")
    items += ["small"] * random.choices([0, 1, 2, 3], [2, 3, 3, 2])[0]
    items += ["number"] * random.choices([0, 1, 2, 3], [3, 4, 2, 1])[0]
    if random.random() < 0.4:
        items.append("label")
    if random.random() < 0.15:
        items.append("barcode")
    if random.random() < 0.12:
        items.append("mfg")
    if random.random() < 0.3:
        items.append("rule")
    for it in items:
        if it == "brand":
            m = mask_of(brand(), H * random.uniform(0.35, 0.9), jp=True)
            if m is not None:
                place(sc, m, edge=True)
        elif it == "small":
            sh = max(5, dh * random.uniform(0.2, 0.45)) if has_date else H * random.uniform(0.06, 0.14)
            bw, bh = int(W * random.uniform(0.2, 1)), int(H * random.uniform(0.2, 1))
            for _ in range(10):
                x0, y0 = random.randint(-bw // 4, W - bw // 2), random.randint(-bh // 4, H - bh // 2)
                r = (max(0, x0), max(0, y0), min(W, x0 + bw), min(H, y0 + bh))
                if r[2] - r[0] > sh * 3 and r[3] - r[1] > sh and sc.free(r, int(sh * 0.4)):
                    small_block(sc, sh, r[0], r[2], r[1], r[3])
                    break
        elif it == "number":
            nh = (dh if has_date else H * 0.4) * random.uniform(0.3, 1.2)
            m = mask_of(other_number(), max(6, nh), dot=random.random() < 0.2, jp=True)
            if m is not None:
                place(sc, m, edge=random.random() < 0.3)
        elif it == "mfg":  # a manufacturing date is not the target
            for _ in range(30):
                vis, label = gen.rand_text()
                if MFG.search(vis):
                    break
            else:
                d = lambda: random.randint(1, 28)
                vis = random.choice(["製造日 ", "製造年月日 ", "MFG "]) + f"20{random.randint(20, 27)}.{d() % 12 + 1:02d}.{d():02d}"
            m = mask_of(vis, max(6, (dh if has_date else H * 0.4) * random.uniform(0.6, 1.1)))
            if m is not None:
                place(sc, m)
        elif it == "label":
            t = random.choice(gen.LABEL_ONLY + ["保存方法", "製造者", "原材料名", "名称", "販売者"])
            m = mask_of(t.strip(), max(6, (dh if has_date else H * 0.4) * random.uniform(0.5, 1.1)), jp=True)
            if m is not None:
                place(sc, m)
        elif it == "barcode":
            bw, bh = int(H * random.uniform(0.8, 2.5)), int(H * random.uniform(0.4, 1))
            m = np.zeros((bh, bw), np.uint8)
            x = 0
            while x < bw:
                s = random.choice([1, 1, 2, 3])
                m[:, x:x + s] = 255
                x += s + random.choice([1, 1, 2, 3])
            place(sc, m, edge=True)
        else:  # rule / border
            m = np.zeros((max(1, random.randint(1, 3)), int(W * random.uniform(0.3, 1))), np.uint8) + 255
            if random.random() < 0.5:
                m = m.T[: H]
            place(sc, m, gap=1, edge=True)
    return sc.bg, box


def warp(img, box):
    h, w = img.shape[:2]
    ang = random.uniform(-2.5, 2.5) * math.pi / 180
    c = np.float32([w / 2, h / 2])
    rot = np.float32([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    d = 0.04 * h
    dst = (src + np.random.uniform(-d, d, (4, 2)).astype(np.float32) - c) @ rot.T + c
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if box:
        x0, y0, x1, y1 = box
        p = cv2.perspectiveTransform(np.float32([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]), M)[0]
        box = (max(0, p[:, 0].min()), max(0, p[:, 1].min()), min(w, p[:, 0].max()), min(h, p[:, 1].max()))
    return img, box


def photo(img):
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    if random.random() < 0.35:
        cx, cy = random.uniform(0, w), random.uniform(-h, 2 * h)
        yy, xx = np.mgrid[0:h, 0:w]
        rr = ((xx - cx) / (h * random.uniform(0.5, 3))) ** 2 + ((yy - cy) / (h * random.uniform(0.3, 1.5))) ** 2
        f += (np.exp(-rr) * random.uniform(40, 110))[:, :, None]
    if random.random() < 0.3:
        f *= np.linspace(random.uniform(0.5, 1), random.uniform(0.5, 1.1), w, dtype=np.float32)[None, :, None]
    f = np.clip(f, 0, 255).astype(np.uint8)
    # camera frame resolution (grab() caps the guide at 160 px tall)
    th = random.randint(40, 160)
    s = th / h
    f = cv2.resize(f, (max(8, int(w * s)), th), interpolation=cv2.INTER_AREA)
    r = random.random()
    if r < 0.35:
        f = cv2.GaussianBlur(f, (0, 0), random.uniform(0.4, 1.5))
    if random.random() < 0.5:
        f = np.clip(f + np.random.randn(*f.shape) * random.uniform(2, 12), 0, 255).astype(np.uint8)
    if random.random() < 0.6:
        f = cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, random.randint(30, 90)])[1], cv2.IMREAD_COLOR)
    return f, s


def to_input(img):
    """Color/gray frame -> float32 (DET_H, W4) in [0,1]; matches toInput() in kigen-ocr.js."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    h, w = g.shape
    nw = min(DET_MAXW, max(8, int(round(w * DET_H / h))))
    g = cv2.resize(g, (nw, DET_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    g = np.clip((g - lo) / max(hi - lo, 1), 0, 1)
    w4 = (nw + 3) // 4 * 4
    if w4 > nw:
        g = np.pad(g, ((0, 0), (0, w4 - nw)), mode="edge")
    return g, nw / w, DET_H / h


def sample():
    while True:
        img, box = scene()
        img, box = warp(np.clip(img, 0, 255).astype(np.uint8), box)
        if box and (box[2] - box[0] < 4 or box[3] - box[1] < 2):
            continue
        img, s = photo(img)
        x, sx, sy = to_input(img)
        if box:
            box = (box[0] * s * sx, box[1] * s * sy, box[2] * s * sx, box[3] * s * sy)
        return x, box


def make_batch(args):
    seed, n = args
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    return [sample() for _ in range(n)]


if __name__ == "__main__":
    import os
    os.makedirs("samples", exist_ok=True)
    random.seed(3)
    np.random.seed(3)
    rows = []
    for i in range(24):
        x, b = sample()
        v = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if b:
            cv2.rectangle(v, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
        rows.append(v)
    W = max(r.shape[1] for r in rows)
    sheet = np.full((len(rows) * (DET_H + 4), W, 3), 255, np.uint8)
    for i, r in enumerate(rows):
        sheet[i * (DET_H + 4):i * (DET_H + 4) + DET_H, :r.shape[1]] = r
    cv2.imwrite("samples/det_sheet.png", sheet)
