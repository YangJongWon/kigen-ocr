"""Synthetic training data for Japanese expiry-date OCR (賞味期限 / 消費期限).

Generates one text line (date + optional prefix/suffix) rendered in a random
style: printed font, inkjet dot-matrix, faint laser/emboss, on packaging-like
backgrounds with curvature, glare, blur and JPEG damage.
"""
import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# index 0 is the CTC blank
CHARS = (" 0123456789〇一二三四五六七八九十年月日上中下旬初末令和平成昭元"
         "賞味期限消費時分午前後ABCDEFGHIJKLMNOPQRSTUVWXYZ./-:()"
         # appended for Korean / Chinese labels (indices of the characters above stay the same)
         "유통기한소비품질지제조일자년월까생산"
         "保质質有效至生产產製造到存民國国")
VOCAB = ["<b>"] + list(CHARS)
C2I = {c: i for i, c in enumerate(VOCAB)}
import os
IMG_H = int(os.environ.get("KIGEN_H", 32))  # model input height (32 = shipped model; 48 = experiment)
MAX_W = 512 * IMG_H // 32

FD = "C:/Windows/Fonts/"
JP_FONTS = [(FD + f, i) for f, i in [
    ("msgothic.ttc", 0), ("msgothic.ttc", 1), ("msmincho.ttc", 0),
    ("meiryo.ttc", 0), ("meiryob.ttc", 0), ("YuGothR.ttc", 0), ("YuGothM.ttc", 0),
    ("YuGothB.ttc", 0), ("YuGothL.ttc", 0), ("BIZ-UDGothicR.ttc", 0),
    ("BIZ-UDGothicB.ttc", 0), ("BIZ-UDMinchoM.ttc", 0), ("UDDigiKyokashoN-R.ttc", 0),
    ("UDDigiKyokashoN-B.ttc", 0), ("NotoSansJP-VF.ttf", 0)]]
LATIN_FONTS = [(FD + f, 0) for f in [
    "arial.ttf", "arialbd.ttf", "ARIALN.TTF", "bahnschrift.ttf", "consola.ttf",
    "consolab.ttf", "cour.ttf", "courbd.ttf", "impact.ttf", "lucon.ttf", "OCRAEXT.TTF",
    "segoeui.ttf", "segoeuib.ttf", "tahoma.ttf", "verdana.ttf", "calibri.ttf"]]
KR_FONTS = [(FD + f, i) for f, i in [
    ("malgun.ttf", 0), ("malgunbd.ttf", 0), ("gulim.ttc", 0), ("batang.ttc", 0),
    ("NotoSansKR-VF.ttf", 0), ("H2GTRM.TTF", 0)]]
SC_FONTS = [(FD + f, i) for f, i in [("msyh.ttc", 0), ("msyhbd.ttc", 0), ("simsun.ttc", 0)]]
TC_FONTS = [(FD + f, i) for f, i in [("msjh.ttc", 0), ("mingliub.ttc", 0)]]
TH_FONTS = [(FD + f, 0) for f in ["leelawui.ttf", "tahoma.ttf"]]


def script_fonts(text):
    """CJK font list that has the glyphs of `text` (Japanese fonts lack Hangul and most hanzi)."""
    if any("฀" <= c <= "๿" for c in text):
        return TH_FONTS
    if any("가" <= c <= "힣" for c in text):
        return KR_FONTS
    if any(c in "质产国" for c in text):
        return SC_FONTS
    if any(c in "質產國效" for c in text):
        return TC_FONTS + SC_FONTS
    if any(c in "保有至到存民" for c in text):
        return SC_FONTS + TC_FONTS
    return JP_FONTS


CLUTTER = "原材料名小麦粉砂糖植物油脂食塩保存方法直射日光高温多湿を避けて内容量製造者株式会社東京都栄養成分表示"

_font_cache = {}


def font(spec, size):
    key = (spec, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(spec[0], size, index=spec[1])
    return _font_cache[key]


# ---------------------------------------------------------------- text
KN = "〇一二三四五六七八九"


def kanji_num(n):  # 31 -> 三十一
    if n < 10:
        return KN[n]
    t, o = divmod(n, 10)
    return ("" if t == 1 else KN[t]) + "十" + (KN[o] if o else "")


def pad(n, p):
    return f"{n:02d}" if p else str(n)


def rand_code():
    n = random.randint(1, 5)
    al = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(random.choice(al) for _ in range(n))


MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
LABEL_ONLY = ["賞味期限", "消費期限", "賞味期限:", "消費期限:", "期限", "賞味期限 ", "消費期限 "]


ML_P = 0.3  # share of non-Japanese lines (Korean, Chinese, Taiwan, South-East Asia)


THAI_MFG, THAI_EXP, THAI_BB = "ผลิต", "หมดอายุ", "ควรบริโภคก่อน"


def ml_text():
    """Date lines as printed in Korea, China, Taiwan, Thailand, Vietnam, Indonesia/Malaysia.

    Returns (visual, label). Thai script is drawn but not in the vocabulary: it is left out of
    the label so the model learns to skip it and read the digits."""
    y, m, d = random.randint(2016, 2034), random.randint(1, 12), random.randint(1, 28)
    p, yy = random.random() < 0.8, f"{y % 100:02d}"
    lot = (" " + rand_code()) if random.random() < 0.2 else ""
    thai = ""
    lang = random.choices(["kr", "cn", "tw", "th", "vn", "id"], [4, 3, 1.5, 1, 1, 1])[0]
    if lang == "kr":
        # 소비기한 is the norm since 2024 (유통기한 on older / imported stock): "소비기한 2026.03.31까지"
        f = random.random()
        s = (f"{y}.{m:02d}.{d:02d}" if f < 0.4 else f"{y}. {m}. {d}." if f < 0.5 else f"{y}.{m:02d}.{d:02d}." if f < 0.55
             else f"{yy}.{m:02d}.{d:02d}" if f < 0.7 else f"{y}년 {pad(m, p)}월 {pad(d, p)}일" if f < 0.85
             else f"{y}-{m:02d}-{d:02d}")
        if random.random() < 0.1 and "년" not in s:
            s += f" {random.randint(0, 23):02d}:{random.choice(['00', '00', '30'])}"  # ready meals
        if random.random() < 0.4:
            s += random.choice(["까지", " 까지"])
        pre = random.choices(["소비기한 ", "소비기한:", "소비기한 : ", "유통기한 ", "유통기한:", "품질유지기한 ",
                              "제조일자 ", "제조일자:", "제조일 ", ""], [5, 3, 2, 2, 2, 1, 2, 1, 1, 4])[0]
    elif lang == "cn":
        # GB 7718: 生产日期 + 保质期 (often a duration); a date is given as 保质期至 / 有效期至
        f = random.random()
        s = (f"{y}{m:02d}{d:02d}" if f < 0.35 else f"{y}年{m:02d}月{d:02d}日" if f < 0.6
             else f"{y}/{m:02d}/{d:02d}" if f < 0.75 else f"{y}-{m:02d}-{d:02d}" if f < 0.9 else f"{y}.{m:02d}.{d:02d}")
        pre = random.choices(["生产日期:", "生产日期", "生产日期 ", "保质期至:", "保质期至", "有效期至:",
                              "有效期至", "到期日:", ""], [4, 2, 1, 2, 2, 2, 1, 1, 3])[0]
    elif lang == "tw":
        # mostly 西元; some products use the ROC year (民國115 = 2026)
        r, f = y - 1911, random.random()
        s = (f"{y}.{m:02d}.{d:02d}" if f < 0.3 else f"{y}/{m:02d}/{d:02d}" if f < 0.55 else f"{y}年{m:02d}月{d:02d}日" if f < 0.65
             else f"{r}.{m:02d}.{d:02d}" if f < 0.85 else f"{r}/{m:02d}/{d:02d}" if f < 0.92 else f"民國{r}年{m:02d}月{d:02d}日")
        pre = random.choices(["有效日期:", "有效日期", "製造日期:", "製造日期", "有效期限:", ""], [4, 2, 3, 1, 1, 2])[0]
    elif lang == "th":
        # Thai labels: ผลิต (MFG) / หมดอายุ (EXP) / ควรบริโภคก่อน (BB), Buddhist-era years (2569, or 69)
        f = random.random()
        yv = str(y) if f < 0.45 else str(y + 543) if f < 0.75 else f"{(y + 543) % 100:02d}"
        sep = random.choice("/./-")
        s = f"{d:02d}{sep}{m:02d}{sep}{yv}"
        tl = random.choices([("EXP ", ""), ("MFG ", ""), ("", THAI_EXP), ("", THAI_MFG), ("", THAI_BB), ("BBE ", "")],
                            [3, 2, 2, 1, 1, 1])[0]
        pre, thai = tl
    elif lang == "vn":
        # NSX (ngày sản xuất) and HSD (hạn sử dụng), often both on one line
        a = f"{d:02d}/{m:02d}/{y}" if random.random() < 0.7 else f"{d:02d}.{m:02d}.{yy}"
        y2 = y + random.randint(0, 2)
        b = f"{d:02d}/{m:02d}/{y2}" if random.random() < 0.7 else f"{d:02d}.{m:02d}.{y2 % 100:02d}"
        f = random.random()
        s, pre = ((f"NSX: {a} HSD: {b}", "") if f < 0.35 else (b, random.choice(["HSD: ", "HSD ", "HSD:"])) if f < 0.7
                  else (a, random.choice(["NSX: ", "NSX ", "NSX:"])))
    else:
        # Indonesia / Malaysia: BAIK DIGUNAKAN SEBELUM, EXP / BB, day first
        sep = random.choice(["/", ".", " ", "-", ""])
        s = f"{d:02d}{sep}{m:02d}{sep}{y if random.random() < 0.6 else yy}"
        pre = random.choices(["EXP ", "EXP:", "BB ", "BAIK DIGUNAKAN SEBELUM ", "GUNA SEBELUM ", "MFG ", ""],
                             [3, 2, 2, 1, 1, 1, 1])[0]
    if random.random() < 0.08 and pre.strip(" :"):
        t = pre.strip(" :")  # the label line alone
        return visual_variant(t), norm_label(t)
    t = " ".join((pre + s + lot).split())
    if len(t) > 26:
        t = " ".join((s + lot).split()) if len(s + lot) <= 26 else s
    vis = visual_variant(t)
    if thai:
        vis = thai + random.choice([" ", ": ", " "]) + vis
    return vis, norm_label(t)


def rand_text():
    if random.random() < ML_P:
        return ml_text()
    # a label line on its own (the line above the date once a box is split into lines)
    if random.random() < 0.06:
        t = random.choice(LABEL_ONLY).strip()
        return visual_variant(t), norm_label(t)
    y = random.randint(2016, 2034)
    m = random.randint(1, 12)
    d = random.randint(1, 31)
    p = random.random() < 0.75
    sep = random.choices([".", "/", "-", " ", ""], [10, 4, 2, 2, 4])[0]
    if sep == "":
        p = True
    k = random.choices(range(11), [22, 22, 6, 9, 5, 9, 4, 3, 1.5, 9, 7])[0]
    yy = pad(y % 100, True)
    if k == 0:    # 2025.12.31
        s = f"{y}{sep}{pad(m, p)}{sep}{pad(d, p)}"
    elif k == 1:  # 25.12.31
        s = f"{yy}{sep}{pad(m, p)}{sep}{pad(d, p)}"
    elif k == 2:  # 2026.03 / 26.3
        s = f"{random.choice([str(y), yy])}{sep or '.'}{pad(m, p)}"
    elif k == 3:  # 2025年12月31日
        s = f"{random.choice([str(y), yy])}年{pad(m, p)}月{pad(d, p)}日"
    elif k == 4:  # 2026年3月 (上旬) / 2026.03末 / 26年3月初旬 / R8.3 下旬
        PART = ["上旬", "中旬", "下旬", "初旬", "初", "中", "末"]
        r4 = random.random()
        ys = str(y) if random.random() < 0.6 else yy
        if r4 < 0.45:
            s = f"{ys}年{pad(m, p)}月" + random.choice(["", ""] + PART + ["末日"[:1]])
        elif r4 < 0.7:
            s = f"{ys}{random.choice('./-')}{pad(m, p)}" + random.choice(["", " "]) + random.choice(PART + ["月末", "月初"])
        elif r4 < 0.85:
            s = f"R{random.randint(1, 16)}.{pad(m, p)}" + random.choice(["", " "]) + random.choice(PART + ["月末"])
        else:
            s = f"令和{random.randint(1, 16)}年{m}月" + random.choice(PART)
    elif k == 5:  # R7.12.31  H31.4.30  S / one-kanji 令7.12.31 平31.4.30 昭64.1.7
        e, base = random.choices([("R", 2018), ("H", 1988), ("S", 1925), ("令", 2018), ("平", 1988), ("昭", 1925)],
                                 [10, 2, 0.3, 5, 1.5, 0.3])[0]
        n = random.randint(1, 31 if e != "S" else 64)
        s = f"{e}{pad(n, random.random() < 0.3)}{sep if sep not in ('', ' ') else '.'}{pad(m, p)}"
        if random.random() < 0.85:
            s += f"{sep if sep not in ('', ' ') else '.'}{pad(d, p)}"
    elif k == 6:  # 令和7年12月31日
        e = random.choices(["令和", "平成", "昭和", "令", "平", "昭"], [10, 3, 0.5, 4, 1.2, 0.2])[0]
        n = random.randint(1, 31)
        ys = "元" if n == 1 and random.random() < 0.6 else str(n)
        if random.random() < 0.3:  # 令和9.3.31 / 令和09.03.31
            sp = random.choice("../")
            s = f"{e}{pad(n, p)}{sp}{pad(m, p)}" + (f"{sp}{pad(d, p)}" if random.random() < 0.9 else "")
        else:
            s = f"{e}{ys}年{m}月" + (f"{d}日" if random.random() < 0.85 else "")
    elif k == 7:  # 31.12.2025 (imports)
        yv = str(y) if random.random() < 0.7 else yy
        a, b = (pad(d, p), pad(m, p)) if random.random() < 0.65 else (pad(m, p), pad(d, p))  # DMY / MDY
        s = f"{a}{sep}{b}{sep}{yv}" if sep else f"{pad(int(a), True)}{pad(int(b), True)}{yv}"
    elif k == 8:  # 二〇二五年十二月三十一日
        s = "".join(KN[int(c)] for c in str(y)) + "年" + kanji_num(m) + "月" + kanji_num(d) + "日"
        if random.random() < 0.3:
            s = "令和" + kanji_num(y - 2018) + "年" + kanji_num(m) + "月" + kanji_num(d) + "日"
    elif k == 10:  # imports with month names: 24.APR.2021 / AUG 13 2021 / 2022.JUN.14 / NOV.2021 / 04APR21
        mon = MONTHS[m - 1]
        yv = str(y) if random.random() < 0.7 else yy
        dd = pad(d, random.random() < 0.85)
        s1, s2 = (random.choices([".", " ", "/", "-", ""], [8, 7, 3, 2, 2])[0] for _ in range(2))
        if random.random() < 0.6:
            s2 = s1
        o = random.choices(["DMY", "MDY", "YMD", "MY"], [10, 9, 3, 2])[0]
        if o == "DMY":
            s = f"{dd}{s1}{mon}{s2}{yv}"
        elif o == "MDY":
            s = f"{mon}{s1}{pad(d, True) if not s2 else dd}{s2}{yv}"
        elif o == "YMD":
            s = f"{y}{s1}{mon}{s2}{dd}"
        else:
            s = f"{mon}{s1 or ' '}{y}"
    else:         # date + time (消費期限 of bento etc.)
        sp = sep if sep not in ("", " ") else "."
        dk = random.choices(range(7), [5, 4, 3, 5, 4, 1.5, 1.5])[0]
        if dk == 0:    # 25.11.09
            s = f"{yy}{sp}{pad(m, p)}{sp}{pad(d, p)}"
        elif dk == 1:  # 2025.11.09
            s = f"{y}{sp}{pad(m, p)}{sp}{pad(d, p)}"
        elif dk == 2:  # 2025年11月9日
            s = f"{random.choice([str(y), yy])}年{pad(m, p)}月{pad(d, p)}日"
        elif dk == 3:  # 11月9日 (no year)
            s = f"{pad(m, p)}月{pad(d, p)}日"
        elif dk == 4:  # 11.09 (no year)
            s = f"{pad(m, p)}{random.choice('./')}{pad(d, p)}"
        elif dk == 5:  # R7.11.9
            s = f"R{random.randint(1, 16)}.{pad(m, p)}.{pad(d, p)}"
        else:          # 251109 / 20251109
            s = f"{random.choice([str(y), yy])}{pad(m, True)}{pad(d, True)}"
        hh, mm = random.randint(0, 23), random.choice([0, 0, 0, 30, random.randint(0, 59)])
        h12 = random.randint(1, 12)
        tm = random.choice([
            f"{hh}時", f"{hh:02d}時", f"{hh}時{mm:02d}分", f"{hh}時{mm}分",
            f"午前{h12}時", f"午後{h12}時", f"午後{h12}時{mm}分", f"午前{h12}時{mm:02d}分",
            f"{hh:02d}:{mm:02d}", f"{hh}:{mm:02d}", f"{hh:02d}:{mm:02d}"])
        s += ("" if s.endswith("日") and random.random() < 0.5 else " ") + tm

    r = random.random()
    if r < 0.22:
        pre = random.choice(["賞味期限", "消費期限", "賞味期限:", "消費期限:", "賞味期限 ", "消費期限 "])
    elif r < 0.32:
        pre = random.choice(["EXP", "EXP.", "EXP ", "BB ", "BEST BEFORE ", "USE BY ", "期限"])
    else:
        pre = ""
    suf = ""
    if random.random() < 0.25:
        suf = " " + rand_code()
    t = " ".join((pre + s + suf).split())
    if len(t) > 24:
        t = s
    label = norm_label(t)
    return visual_variant(t), label


FW = {chr(c): chr(c + 0xFEE0) for c in range(0x21, 0x7F)}
FW[" "] = "　"
SEPS = "./-:年月"


def norm_label(t):
    """Training target: half-width, no spaces around separators."""
    import re
    t = " ".join(t.split())
    t = re.sub(r"\s*([./\-:年月])\s*", lambda m: m.group(1), t)
    t = re.sub(r"\s+(?=[上中下初末])", "", t)
    return re.sub(r"(?<![A-Z])([RHS]) (?=\d)", lambda m: m.group(1), t).strip()


def visual_variant(t):
    """What is actually printed: optional spaces after separators, full-width glyphs."""
    if random.random() < 0.25:  # "R8. 11. 16" / "2026 / 8 / 13" / "2026年 8月 13日"
        both = random.random() < 0.3
        out = ""
        for c in t:
            if c in SEPS:
                out += (" " if both else "") + c + " "
            else:
                out += c
        t = out
    if random.random() < 0.15:  # "R 8.11.16"
        import re
        t = re.sub(r"(?<![A-Z])([RHS])(?=\d)", lambda m: m.group(1) + " ", t, count=1)
    r = random.random()
    if r < 0.2:    # all full-width
        t = "".join(FW.get(c, c) for c in t)
    elif r < 0.3:  # mixed (e.g. full-width digits, half-width dots)
        p = random.random()
        t = "".join(FW[c] if c in FW and random.random() < p else c for c in t)
    return t

# ---------------------------------------------------------------- render
def line_mask(text, h=64):
    """White-on-black mask of the text line at height ~h."""
    size = int(h * random.uniform(0.7, 0.85))
    jp = random.choice(script_fonts(text))
    lat = random.choice(LATIN_FONTS) if random.random() < 0.55 else jp
    track = random.uniform(-0.05, 0.25) * size
    squash = random.uniform(0.75, 1.25)  # horizontal scale
    pieces, x = [], 0
    for ch in text:
        f = font(lat if ch.isascii() else jp, size)
        if ch == " ":
            x += size * random.uniform(0.3, 0.8)
            continue
        bb = f.getbbox(ch)
        w = max(1, bb[2] - bb[0])
        pieces.append((x, ch, f, bb))
        x += w + track + size * 0.06
    W = int(x + size)
    img = Image.new("L", (max(W, 8), h * 2), 0)
    dr = ImageDraw.Draw(img)
    base = int(h * 0.3)
    for px, ch, f, bb in pieces:
        dr.text((px - bb[0] + size * 0.3, base), ch, fill=255, font=f)
    m = np.array(img)
    if squash != 1:
        m = cv2.resize(m, (max(8, int(m.shape[1] * squash)), m.shape[0]))
    # stroke weight
    r = random.random()
    if r < 0.2:
        m = cv2.dilate(m, np.ones((2, 2), np.uint8))
    elif r < 0.3:
        m = cv2.erode(m, np.ones((2, 2), np.uint8))
    return m


def dotify(m, text):
    """Inkjet dot-matrix look: sample mask on a coarse grid, draw a dot per cell."""
    ys, xs = np.nonzero(m > 128)
    if len(ys) == 0:
        return m
    th = ys.max() - ys.min() + 1
    has_kanji = any(not c.isascii() for c in text)
    grid = random.randint(13, 18) if has_kanji else random.randint(9, 16)  # dots per line height
    if grid < 12:
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    cell = th / grid
    small = cv2.resize(m, (max(1, int(m.shape[1] / cell)), max(1, int(m.shape[0] / cell))),
                       interpolation=cv2.INTER_AREA)
    on = np.argwhere(small > random.uniform(70, 140))
    out = np.zeros_like(m)
    rad = cell * random.uniform(0.32, 0.6)
    jit = cell * random.uniform(0, 0.15)
    for (yy, xx) in on:
        if random.random() < 0.02:  # missing nozzle
            continue
        cx = (xx + 0.5) * cell + random.gauss(0, jit)
        cy = (yy + 0.5) * cell + random.gauss(0, jit)
        cv2.circle(out, (int(cx * 4), int(cy * 4)), max(1, int(rad * 4)), 255, -1,
                   lineType=cv2.LINE_AA, shift=2)
    return out


def background(h, w):
    c1 = np.array([random.randint(0, 255) for _ in range(3)], np.float32)
    if random.random() < 0.45:  # white / cream / silver packaging
        c1 = np.full(3, random.uniform(170, 255), np.float32) + np.random.uniform(-20, 10, 3)
    c2 = c1 + np.random.uniform(-60, 60, 3)
    t = np.linspace(0, 1, w if random.random() < 0.5 else h, dtype=np.float32)
    g = t[None, :, None] if len(t) == w else t[:, None, None]
    bg = c1 * (1 - g) + c2 * g
    bg = np.broadcast_to(bg, (h, w, 3)).copy()
    if random.random() < 0.4:  # printed graphics: random shapes
        for _ in range(random.randint(1, 4)):
            col = [random.randint(0, 255) for _ in range(3)]
            if random.random() < 0.5:
                cv2.rectangle(bg, (random.randint(-w, w), random.randint(-h, h)),
                              (random.randint(0, 2 * w), random.randint(0, 2 * h)), col, -1)
            else:
                cv2.circle(bg, (random.randint(0, w), random.randint(0, h)),
                           random.randint(h // 2, 3 * h), col, -1)
    if random.random() < 0.3:  # metallic / film texture
        n = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32), (0, 0), random.uniform(0.5, 3))
        bg += n[:, :, None] * random.uniform(5, 25)
    return np.clip(bg, 0, 255)


def clutter(canvas, mask_h, ink):
    """Partially visible unrelated text above/below the line."""
    h, w = canvas.shape[:2]
    for top in (True, False):
        if random.random() < 0.35:
            f = font(random.choice(JP_FONTS), max(6, int(mask_h * random.uniform(0.4, 0.8))))
            s = "".join(random.choice(CLUTTER) for _ in range(random.randint(4, 20)))
            img = Image.fromarray(canvas.astype(np.uint8))
            dr = ImageDraw.Draw(img)
            y = random.randint(-mask_h // 2, 0) if top else h - random.randint(2, max(3, mask_h // 2))
            dr.text((random.randint(-50, w // 2), y), s, fill=tuple(int(v) for v in ink), font=f)
            canvas[:] = np.array(img, np.float32)


NEIGHBOR_P = 0.2  # per side
NEIGHBOR = ["賞味期限", "消費期限", "賞味期限:", "消費期限:", "製造所固有記号", "保存方法", "LOT"]


def neighbor(canvas, th, mt, mb, ink):
    """Edge of a same-size adjacent line (e.g. 賞味期限 right above the date) cut by the crop.
    Only its outer 10-40% may show, and never inside the margin next to the target line."""
    h, w = canvas.shape[:2]
    for top in (True, False):
        room = mt if top else mb
        if random.random() > NEIGHBOR_P or room < 3:
            continue
        s = random.choice(NEIGHBOR + ["".join(random.choice("0123456789./") for _ in range(random.randint(6, 10)))])
        f = font(random.choice(JP_FONTS), max(8, int(th * random.uniform(0.95, 1.25))))
        bb = f.getbbox(s)
        gh = bb[3] - bb[1]
        vis = int(min(room - 1, gh * random.uniform(0.1, 0.4)))
        if vis < 2:
            continue
        y = vis - bb[3] if top else h - vis - bb[1]
        img = Image.fromarray(canvas.astype(np.uint8))
        ImageDraw.Draw(img).text((random.randint(-w // 4, w // 3), y), s, fill=tuple(int(v) for v in ink), font=f)
        canvas[:] = np.array(img, np.float32)


def render(text):
    m = line_mask(text)
    style = random.choices(["print", "dot", "faint"], [4, 5, 1.5])[0]
    if style == "dot":
        m = dotify(m, text)
    ys, xs = np.nonzero(m > 40)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    th = y1 - y0
    # margins (tight crops and loose crops)
    mt, mb = int(th * random.uniform(0.05, 0.6)), int(th * random.uniform(0.05, 0.6))
    ml, mr = int(th * random.uniform(0.05, 1.5)), int(th * random.uniform(0.05, 1.5))
    m = np.pad(m, ((mt + 2, mb + 2), (ml + 2, mr + 2)))[y0:y1 + mt + mb + 4, x0:x1 + ml + mr + 4]
    h, w = m.shape
    a = (m.astype(np.float32) / 255.0)[:, :, None]

    bg = background(h, w)
    lum = bg.mean()
    if style == "faint":  # laser etch / emboss: low contrast, edge highlight
        delta = random.choice([-1, 1]) * random.uniform(25, 70)
        ink = np.clip(bg.mean((0, 1)) + delta, 0, 255)
        sh = np.roll(a, (1, 1), (0, 1))
        img = bg * (1 - a) + ink * a
        img = img + (sh - a) * random.uniform(-40, 40)
    else:
        if lum > 110 or random.random() < 0.15:
            ink = np.array([random.uniform(0, 70)] * 3) + np.random.uniform(0, 40, 3) * (random.random() < 0.3)
        else:
            ink = np.array([random.uniform(200, 255)] * 3)
        clutter(bg, th, ink)
        neighbor(bg, th, mt + 2, mb + 2, ink)
        img = bg * (1 - a * random.uniform(0.75, 1.0)) + ink * a * random.uniform(0.75, 1.0)
    img = np.clip(img, 0, 255)
    on = a[:, :, 0] > 0.5
    if on.sum() < 10:
        return None
    # legibility: ink must differ from its own surrounding background
    bgl = cv2.blur(bg.mean(2), (9, 9))
    diff = np.abs(img.mean(2) - bgl)[on]
    if style != "faint" and np.mean(diff < 45) > 0.08:
        return None
    img = img.astype(np.uint8)
    img = geometric(img)
    img = photometric(img)
    return img


def geometric(img):
    h, w = img.shape[:2]
    if random.random() < 0.35:  # curved surface (bottle / cup)
        amp = random.uniform(0, 0.12) * h
        ph = random.uniform(0, math.pi)
        fx = np.arange(w, dtype=np.float32)
        mx = np.tile(fx, (h, 1))
        my = np.arange(h, dtype=np.float32)[:, None] + amp * np.sin(fx / w * math.pi + ph)[None, :]
        img = cv2.remap(img, mx, my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    d = 0.06
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32([[random.uniform(-d, d) * h * 3, random.uniform(-d, d) * h] for _ in range(4)])
    ang = random.uniform(-4, 4) * math.pi / 180
    c = np.float32([w / 2, h / 2])
    rot = np.float32([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    dst = (dst - c) @ rot.T + c
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def photometric(img):
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    if random.random() < 0.35:  # glare from shop lights on film
        cx, cy = random.uniform(0, w), random.uniform(-h, 2 * h)
        yy, xx = np.mgrid[0:h, 0:w]
        rr = ((xx - cx) / (h * random.uniform(0.5, 3))) ** 2 + ((yy - cy) / (h * random.uniform(0.3, 1.5))) ** 2
        f += (np.exp(-rr) * random.uniform(40, 110))[:, :, None]
    if random.random() < 0.3:  # shadow / uneven light
        g = np.linspace(random.uniform(0.5, 1), random.uniform(0.5, 1.1), w, dtype=np.float32)
        f *= g[None, :, None]
    f = np.clip(f, 0, 255).astype(np.uint8)
    # downscale to near target then degrade (simulates low-res camera crop)
    scale = random.uniform(0.6, 1.6) * IMG_H / h
    sw, sh = max(4, int(w * scale)), max(4, int(h * scale))
    f = cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA)
    r = random.random()
    if r < 0.3:
        k = random.choice([3, 5])
        f = cv2.GaussianBlur(f, (k, k), random.uniform(0.3, 1.3))
    elif r < 0.45:
        k = random.randint(3, 7)
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1 / k
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), random.uniform(0, 180), 1)
        ker = cv2.warpAffine(ker, M, (k, k))
        f = cv2.filter2D(f, -1, ker / max(ker.sum(), 1e-6))
    if random.random() < 0.5:
        f = np.clip(f + np.random.randn(*f.shape) * random.uniform(2, 14), 0, 255).astype(np.uint8)
    if random.random() < 0.6:
        q = random.randint(15, 90)
        f = cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    return f


def to_input(img_bgr_or_rgb):
    """Any color image -> float32 (32, W) in [0,1]. Must match the JS preprocessing."""
    g = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_RGB2GRAY) if img_bgr_or_rgb.ndim == 3 else img_bgr_or_rgb
    h, w = g.shape
    nw = min(MAX_W, max(8, int(round(w * IMG_H / h))))
    g = cv2.resize(g, (nw, IMG_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)  # per-line contrast stretch
    g = np.clip((g - lo) / max(hi - lo, 1), 0, 1)
    return g


def make_batch(args):
    """Pool worker: (seed, n) -> list of (x, text). Imports no torch."""
    seed, n = args
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    return [sample() for _ in range(n)]


def sample():
    while True:
        vis, t = rand_text()
        img = render(vis)
        if img is not None and t:
            return to_input(img), t


if __name__ == "__main__":
    import os
    os.makedirs("samples", exist_ok=True)
    random.seed(1)
    rows = []
    for i in range(40):
        x, t = sample()
        rows.append((x, t))
        print(i, t)
    W = max(r[0].shape[1] for r in rows)
    sheet = np.ones((len(rows) * (IMG_H + 4), W), np.float32)
    for i, (x, _) in enumerate(rows):
        sheet[i * (IMG_H + 4):i * (IMG_H + 4) + IMG_H, :x.shape[1]] = x
    cv2.imwrite("samples/sheet.png", (sheet * 255).astype(np.uint8))
