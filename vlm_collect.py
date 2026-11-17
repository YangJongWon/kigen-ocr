"""Collect expiry-date photos cheaply: Open Food Facts (Japan) package photos, read by two free
OpenRouter vision models. A date counts as labelled only when both models read the same date.

  uv run python vlm_collect.py fetch  --n 200            # photos -> datasets/off_pilot/
  uv run python vlm_collect.py label                     # both models, resumable
  uv run python vlm_collect.py report

The OpenRouter key is read from avataragent's data/office.db at run time (never written here).
Open Food Facts images are CC BY-SA: keep them for evaluation unless decided otherwise.
"""
import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "datasets", "off_pilot")
UA = "kigen-ocr research pilot (https://github.com/YangJongWon/kigen-ocr)"
# (Gemma 4 31B / 26B free answered only 429 on 2026-11-17)
MODELS = {"a": "nvidia/nemotron-3.5-super-vl:free", "b": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"}
PROMPT = """This is a photo of a food package. Find the EXPIRY date printed on it: 賞味期限, 消費期限, best before, use by, EXP, or a date printed on its own by an inkjet / stamp (often dot-matrix, often on the top, bottom or a flap). Do NOT report a manufacturing date (製造, 製造年月日, MFG, PROD) or a date that is part of the printed design.
Answer with JSON only:
{"has_date": true or false,
 "text": "the date characters exactly as printed (digits, separators, 年月日 etc.), without the label",
 "label": "the label printed next to it, or empty",
 "legible": true if every character of the date can be read with certainty, else false,
 "box": [x0, y0, x1, y1] of the date characters in 0-1000 coordinates of the image}
If no expiry date is visible, answer {"has_date": false}."""


def key():
    db = os.path.join(os.path.dirname(ROOT), "avataragent", "data", "office.db")
    m = re.search(rb"sk-or-v1-[0-9a-f]{20,}", open(db, "rb").read())
    if not m:
        sys.exit("no OpenRouter key in avataragent/data/office.db")
    return m.group().decode()


def get(url, timeout=60):
    """GET with retries: the Open Food Facts search API answers 503 / 429 when busy."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 502, 503, 504) or attempt == 5:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt == 5:
                raise
        time.sleep(15 * (attempt + 1))


def img_path(code):
    return "/".join([code[0:3], code[3:6], code[6:9], code[9:]]) if len(code) > 8 else code


def fetch(n):
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    meta_p = os.path.join(OUT, "photos.json")
    photos = json.load(open(meta_p, encoding="utf8")) if os.path.exists(meta_p) else []
    have = {p["file"] for p in photos}
    page = 1
    while len(photos) < n:
        url = ("https://world.openfoodfacts.org/api/v2/search?countries_tags_en=japan"
               f"&fields=code,images&page_size=50&page={page}&sort_by=last_modified_t")
        prods = json.loads(get(url)).get("products", [])
        if not prods:
            break
        for p in prods:
            code, imgs = p.get("code", ""), p.get("images") or {}
            # raw uploads (numeric ids); the ones picked as the front photo rarely show the date
            front = {v.get("imgid") for k, v in imgs.items() if k.startswith("front")}
            raw = [k for k in imgs if k.isdigit() and k not in front]
            for k in raw[:3]:
                full = imgs[k].get("sizes", {}).get("full", {})
                if min(full.get("w", 0), full.get("h", 0)) < 600:
                    continue
                f = f"{code}_{k}.jpg"
                if f in have:
                    continue
                try:
                    data = get(f"https://images.openfoodfacts.org/images/products/{img_path(code)}/{k}.jpg")
                except urllib.error.URLError as e:
                    print("skip", f, e)
                    continue
                open(os.path.join(OUT, "images", f), "wb").write(data)
                photos.append({"file": f, "code": code, "imgid": k,
                               "source": f"https://world.openfoodfacts.org/product/{code}", "license": "CC BY-SA 3.0 (Open Food Facts)"})
                have.add(f)
                if len(photos) >= n:
                    break
            if len(photos) >= n:
                break
        json.dump(photos, open(meta_p, "w", encoding="utf8"), indent=1)
        print(f"page {page}: {len(photos)} photos")
        page += 1
        time.sleep(7)  # search API: 10 requests / minute
    json.dump(photos, open(meta_p, "w", encoding="utf8"), indent=1)


def ask(k, model, path):
    img = cv2.imread(path)
    s = 1600 / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    b64 = base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()).decode()
    body = {"model": model, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": PROMPT}]}],
            "reasoning": {"enabled": False}}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(),
                                 {"Authorization": f"Bearer {k}", "Content-Type": "application/json",
                                  "HTTP-Referer": "https://github.com/YangJongWon/kigen-ocr", "X-Title": "kigen-ocr"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                txt = json.loads(r.read())["choices"][0]["message"]["content"] or ""
            m = re.search(r"\{.*\}", txt, re.S)
            return json.loads(m.group()) if m else {"error": "no json", "raw": txt[:300]}
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(20 * (attempt + 1))
                continue
            return {"error": f"http {e.code}"}
        except Exception as e:  # noqa: BLE001 (timeouts, bad json)
            return {"error": str(e)[:200]}
    return {"error": "429"}


def label():
    k = key()
    photos = json.load(open(os.path.join(OUT, "photos.json"), encoding="utf8"))
    res_p = os.path.join(OUT, "labels.json")
    res = json.load(open(res_p, encoding="utf8")) if os.path.exists(res_p) else {}
    for i, p in enumerate(photos):
        r = res.setdefault(p["file"], {})
        for tag, model in MODELS.items():
            if tag in r and "error" not in r[tag]:
                continue
            r[tag] = ask(k, model, os.path.join(OUT, "images", p["file"]))
            time.sleep(3)  # free models: ~20 requests / minute
        # (utf-8: the default Windows codec cannot write every kanji; write whole, then swap)
        tmp = res_p + ".tmp"
        with open(tmp, "w", encoding="utf8") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        os.replace(tmp, res_p)
        print(i, p["file"], {t: (v.get("text") if v.get("has_date") else v.get("error", "-")) for t, v in r.items()}, flush=True)


def report():
    sys.path.insert(0, ROOT)
    res = json.load(open(os.path.join(OUT, "labels.json"), encoding="utf8"))
    n = len(res)
    err = {t: sum("error" in r.get(t, {"error": 1}) for r in res.values()) for t in MODELS}
    has = {t: sum(bool(r.get(t, {}).get("has_date")) for r in res.values()) for t in MODELS}
    both, agree, rows = 0, 0, []
    norm = lambda s: re.sub(r"[\s.\/\-:]", "", s or "")
    for f, r in res.items():
        a, b = r.get("a", {}), r.get("b", {})
        if a.get("has_date") and b.get("has_date"):
            both += 1
            same = norm(a.get("text")) == norm(b.get("text")) and norm(a.get("text"))
            agree += bool(same)
            rows.append((f, a.get("text"), b.get("text"), "AGREE" if same else "differ"))
    print(f"photos {n}  errors {err}  has_date {has}  both say date {both}  same text {agree}")
    for row in rows:
        print("  ", *row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "label", "report"])
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    {"fetch": lambda: fetch(a.n), "label": label, "report": report}[a.cmd]()
