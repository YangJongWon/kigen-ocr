"""Crawl C&C Tajimaya product barcode IDs into sqlite.

The public WordPress REST list for products includes the barcode in the
SEO description, as ``名前(4901234567894)の商品情報``. Product pages under
``/products/`` are allowed by robots.txt; ``/catalog`` is not, and this
crawler does not touch it.

``--hours`` selects shards of the catalog: a product is in scope when
``product_id % 24`` is one of the given hours (0-23). ``1-6`` is hours
1, 2, 3, 4, 5 and 6. ``--max`` is how many new rows this run may insert.
A cursor in the database resumes the next run on the same hour set.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "https://www.tajimaya-cc.net/wp-json/wp/v2/products"
USER_AGENT = "kigenocr-barcode-id-crawl/0.1 (expiry-date research)"
PAGE_SIZE = 100
DELAY_S = 0.35
CODE_RE = re.compile(r"\((\d{6,14})\)の商品情報")

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    barcode TEXT NOT NULL,
    barcode_kind TEXT NOT NULL,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    published TEXT,
    hour INTEGER NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS products_barcode ON products (barcode);
CREATE INDEX IF NOT EXISTS products_hour ON products (hour);
CREATE TABLE IF NOT EXISTS crawl_cursor (
    hour_spec TEXT PRIMARY KEY,
    next_page INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def parse_hours(spec: str) -> set[int]:
    hours: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if not (0 <= lo <= hi <= 23):
                raise SystemExit(f"hour range out of 0-23: {part}")
            hours.update(range(lo, hi + 1))
        else:
            hour = int(part)
            if not 0 <= hour <= 23:
                raise SystemExit(f"hour out of 0-23: {part}")
            hours.add(hour)
    if not hours:
        raise SystemExit("no hours given")
    return hours


def ean_ok(code: str) -> bool:
    if len(code) not in (8, 13) or not code.isdigit():
        return False
    body = [int(c) for c in code[:-1]]
    total = sum(d * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(body)))
    return (10 - (total % 10)) % 10 == int(code[-1])


def barcode_kind(code: str) -> str:
    # Prefix 2 is GS1 restricted circulation (in-store), not a manufacturer JAN.
    if len(code) == 13 and ean_ok(code):
        if code.startswith(("45", "49")):
            return "jan"
        if code.startswith("2"):
            return "instore"
        return "ean13"
    if len(code) == 8 and ean_ok(code):
        return "instore" if code.startswith("2") else "ean8"
    return "other"


def barcode_from(item: dict) -> str | None:
    meta = item.get("aioseo_meta_data") or {}
    head = item.get("aioseo_head_json") or {}
    blobs = [
        meta.get("description") or "",
        meta.get("og_description") or "",
        head.get("description") or "",
        head.get("og:description") or "",
    ]
    for blob in blobs:
        match = CODE_RE.search(html.unescape(blob))
        if match:
            return match.group(1)
    return None


def product_name(item: dict) -> str:
    title = item.get("title") or {}
    raw = title.get("rendered") if isinstance(title, dict) else str(title)
    return html.unescape(raw or "").strip()


def fetch_page(page: int) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "per_page": PAGE_SIZE,
            "page": page,
            "orderby": "id",
            "order": "asc",
            "_fields": "id,date,link,title,aioseo_head_json",
        }
    )
    url = f"{API}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    delay = 1.0
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")
            if error.code == 400 and "rest_post_invalid_page_number" in body:
                return []
            if error.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(delay)
                delay *= 2
                continue
            raise SystemExit(f"HTTP {error.code} on page {page}: {body[:200]}") from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt < 4:
                time.sleep(delay)
                delay *= 2
                continue
            raise SystemExit(f"failed page {page}: {error}") from error
    return []


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    return connection


def crawl(db_path: Path, hour_spec: str, max_rows: int) -> None:
    if max_rows < 1:
        raise SystemExit("--max must be >= 1")
    hours = parse_hours(hour_spec)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    connection = open_db(db_path)
    row = connection.execute(
        "SELECT next_page FROM crawl_cursor WHERE hour_spec = ?",
        (hour_spec,),
    ).fetchone()
    page = row[0] if row else 1
    inserted = 0
    scanned = 0
    skipped = 0
    print(
        f"hours {hour_spec} -> {sorted(hours)}  page {page}  max {max_rows}  db {db_path}",
        flush=True,
    )
    while inserted < max_rows:
        items = fetch_page(page)
        if not items:
            print(f"page {page}: catalog end", flush=True)
            break
        page_inserted = 0
        hit_max = False
        with connection:
            for item in items:
                scanned += 1
                product_id = int(item["id"])
                if product_id % 24 not in hours:
                    continue
                exists = connection.execute(
                    "SELECT 1 FROM products WHERE product_id = ?",
                    (product_id,),
                ).fetchone()
                if exists:
                    continue
                barcode = barcode_from(item)
                if not barcode:
                    skipped += 1
                    continue
                connection.execute(
                    """
                    INSERT INTO products (
                        product_id, barcode, barcode_kind, name, url, published, hour, fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        product_id,
                        barcode,
                        barcode_kind(barcode),
                        product_name(item),
                        item.get("link") or f"https://www.tajimaya-cc.net/products/{product_id}",
                        item.get("date"),
                        product_id % 24,
                        now,
                    ),
                )
                inserted += 1
                page_inserted += 1
                if inserted >= max_rows:
                    hit_max = True
                    break
            next_page = page if hit_max else page + 1
            connection.execute(
                """
                INSERT INTO crawl_cursor (hour_spec, next_page, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT (hour_spec) DO UPDATE SET
                    next_page = excluded.next_page,
                    updated_at = excluded.updated_at
                """,
                (hour_spec, next_page, now),
            )
        print(
            f"page {page}: +{page_inserted}  inserted {inserted}  scanned {scanned}  no-code {skipped}",
            flush=True,
        )
        if hit_max:
            break
        page += 1
        time.sleep(DELAY_S)
    total = connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    by_kind = connection.execute(
        "SELECT barcode_kind, COUNT(*) FROM products GROUP BY barcode_kind ORDER BY barcode_kind"
    ).fetchall()
    connection.close()
    kinds = ", ".join(f"{kind} {count}" for kind, count in by_kind)
    print(f"done: inserted {inserted} this run, {total} rows in db ({kinds})", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Crawl C&C Tajimaya product barcode IDs into sqlite. "
        "--hours selects product_id % 24 shards (1-6 means remainders 1 through 6). "
        "--max is how many new rows this run may insert."
    )
    parser.add_argument("--db", required=True, type=Path, help="sqlite path, created if missing")
    parser.add_argument(
        "--hours",
        required=True,
        help="product_id %% 24 shards, e.g. 1-6 or 0,8,12-15",
    )
    parser.add_argument("--max", required=True, type=int, help="max new rows to insert this run")
    args = parser.parse_args(argv)
    try:
        crawl(args.db, args.hours, args.max)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
