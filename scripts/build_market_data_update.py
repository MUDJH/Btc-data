#!/usr/bin/env python3
"""Build overlap-safe market-data tails for ChatGPT backtests.

Sources:
- Binance Public Data daily 1m kline archives.
- Binance USD-M Futures API funding history.
- ff137/bitstamp-btcusd-minute-data latest overlap CSV.

All timestamps written by this script are normalized to Unix milliseconds.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAILS = ROOT / "tails"
TAILS.mkdir(parents=True, exist_ok=True)
UTC = timezone.utc
USER_AGENT = "MUDJH-Btc-data-updater/1.0"

SPOT_SYMBOLS = {
    "ADAUSDT": date(2026, 7, 10),
    "BNBUSDT": date(2026, 7, 10),
    "DOGEUSDT": date(2026, 7, 10),
    "ETHUSDT": date(2026, 7, 10),
    "LTCUSDT": date(2026, 7, 10),
    "SOLUSDT": date(2026, 7, 10),
    "XRPUSDT": date(2026, 7, 10),
}
BTC_CONFIGS = {
    "BTCUSDT_SPOT_1m.csv.gz": ("spot", "BTCUSDT", date(2026, 8, 5)),
    "BTCUSDT_USDM_PERP_1m.csv.gz": ("um", "BTCUSDT", date(2026, 8, 6)),
}
USDM_TAILS = {
    "XAUUSDT_BINANCE_USDM_PERP_1m.csv.gz": ("XAUUSDT", date(2026, 10, 1)),
    "SPYUSDT_BINANCE_USDM_PERP_1m.csv.gz": ("SPYUSDT", date(2026, 10, 1)),
}
FUNDING_TAILS = {
    "XAUUSDT_BINANCE_FUNDING.csv.gz": ("XAUUSDT", date(2026, 10, 1)),
    "SPYUSDT_BINANCE_FUNDING.csv.gz": ("SPYUSDT", date(2026, 10, 1)),
}
BITSTAMP_URL = (
    "https://raw.githubusercontent.com/ff137/"
    "bitstamp-btcusd-minute-data/main/data/updates/"
    "btcusd_bitstamp_1min_latest.csv"
)


def request_bytes(url: str, attempts: int = 4) -> bytes:
    last = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=90) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504):
                raise
            last = exc
            if attempt + 1 < attempts:
                retry_after = exc.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else 2 ** attempt)
        except Exception as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed after {attempts} attempts: {url}") from last


def norm_ms(value: str) -> int:
    n = int(value)
    if n > 100_000_000_000_000:
        n //= 1000
    return n


def iso_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def archive_url(market: str, symbol: str, day: date) -> str:
    stamp = day.isoformat()
    if market == "spot":
        return (
            f"https://data.binance.vision/data/spot/daily/klines/"
            f"{symbol}/1m/{symbol}-1m-{stamp}.zip"
        )
    return (
        f"https://data.binance.vision/data/futures/um/daily/klines/"
        f"{symbol}/1m/{symbol}-1m-{stamp}.zip"
    )


def read_archive(market: str, symbol: str, day: date) -> list[list[str]]:
    url = archive_url(market, symbol, day)
    payload = request_bytes(url)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise RuntimeError(f"expected one CSV in {url}, found {members}")
        with zf.open(members[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            rows = []
            for row in csv.reader(text):
                if not row or not row[0].strip().isdigit():
                    continue
                rows.append(row)
            return rows


def load_existing(path: Path) -> tuple[list[str] | None, dict[int, list[str]]]:
    if not path.exists():
        return None, {}
    rows: dict[int, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        for row in reader:
            if row:
                rows[norm_ms(row[0])] = row
    return header, rows


def start_for(path: Path, configured: date) -> date:
    _, rows = load_existing(path)
    if not rows:
        return configured
    return datetime.fromtimestamp(max(rows) / 1000, UTC).date()


def download_days(market: str, symbol: str, start: date, end: date) -> tuple[list[list[str]], list[str]]:
    days = []
    cursor = start
    while cursor <= end:
        days.append(cursor)
        cursor += timedelta(days=1)
    rows: list[list[str]] = []
    skipped: list[str] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(read_archive, market, symbol, day): day for day in days}
        for future in as_completed(futures):
            day = futures[future]
            try:
                rows.extend(future.result())
            except urllib.error.HTTPError as exc:
                if exc.code == 404 and day >= end - timedelta(days=2):
                    skipped.append(day.isoformat())
                else:
                    raise
    return rows, sorted(skipped)


def write_gzip(path: Path, header: list[str], rows: dict[int, list[str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="", compresslevel=6) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for key in sorted(rows):
            writer.writerow(rows[key])
    os.replace(tmp, path)


def build_alt(symbol: str, configured_start: date, end: date) -> dict:
    path = TAILS / f"{symbol}_1m_spot.csv.gz"
    header, merged = load_existing(path)
    start = start_for(path, configured_start)
    fresh, skipped = download_days("spot", symbol, start, end)
    for raw in fresh:
        ms = norm_ms(raw[0])
        merged[ms] = [str(ms), raw[1], raw[2], raw[3], raw[4], raw[5]]
    out_header = header or ["timestamp", "open", "high", "low", "close", "volume"]
    write_gzip(path, out_header, merged)
    return manifest_entry(path, merged, "Binance spot daily archives", skipped)


def build_btc(filename: str, market: str, symbol: str, configured_start: date, end: date) -> dict:
    path = TAILS / filename
    header, merged = load_existing(path)
    start = start_for(path, configured_start)
    fresh, skipped = download_days(market, symbol, start, end)
    for raw in fresh:
        open_ms = norm_ms(raw[0])
        close_ms = norm_ms(raw[6])
        merged[open_ms] = [
            str(open_ms),
            iso_ms(open_ms),
            raw[1],
            raw[2],
            raw[3],
            raw[4],
            raw[5],
            str(close_ms),
            iso_ms(close_ms),
            raw[7],
            raw[8],
            raw[9],
            raw[10],
            raw[11] if len(raw) > 11 else "0",
        ]
    out_header = header or [
        "open_time_ms", "open_time_utc", "open", "high", "low", "close",
        "volume_base", "close_time_ms", "close_time_utc", "quote_volume",
        "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ]
    write_gzip(path, out_header, merged)
    source = "Binance spot daily archives" if market == "spot" else "Binance USD-M futures daily archives"
    return manifest_entry(path, merged, source, skipped)


def build_usdm_tail(filename: str, symbol: str, configured_start: date, end: date) -> dict:
    path = TAILS / filename
    header, merged = load_existing(path)
    start = start_for(path, configured_start)
    fresh, skipped = download_days("um", symbol, start, end)
    for raw in fresh:
        open_ms = norm_ms(raw[0])
        close_ms = norm_ms(raw[6])
        merged[open_ms] = [
            str(open_ms),
            iso_ms(open_ms),
            raw[1],
            raw[2],
            raw[3],
            raw[4],
            raw[5],
            str(close_ms),
            raw[7],
            raw[8],
            raw[9],
            raw[10],
            raw[11] if len(raw) > 11 else "0",
        ]
    out_header = header or [
        "open_time_ms", "open_time_utc", "open", "high", "low", "close",
        "volume", "close_time_ms", "quote_volume", "trade_count",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ]
    keys = sorted(merged)
    gaps = sum(1 for left, right in zip(keys, keys[1:]) if right - left != 60_000)
    if gaps:
        raise RuntimeError(f"{filename} has {gaps} non-1m transitions")
    write_gzip(path, out_header, merged)
    return manifest_entry(path, merged, "Binance USD-M futures daily archives", skipped)


def funding_api_rows(symbol: str, start: date, end: date) -> list[dict]:
    start_ms = int(datetime.combine(start, datetime.min.time(), UTC).timestamp() * 1000)
    end_ms = int(datetime.combine(end + timedelta(days=1), datetime.min.time(), UTC).timestamp() * 1000) - 1
    rows: list[dict] = []
    cursor = start_ms
    while cursor <= end_ms:
        url = (
            "https://fapi.binance.com/fapi/v1/fundingRate"
            f"?symbol={symbol}&startTime={cursor}&endTime={end_ms}&limit=1000"
        )
        batch = json.loads(request_bytes(url))
        if not batch:
            break
        rows.extend(batch)
        last_ms = max(int(item["fundingTime"]) for item in batch)
        if len(batch) < 1000 or last_ms >= end_ms:
            break
        cursor = last_ms + 1
    return rows


def build_funding_tail(filename: str, symbol: str, configured_start: date, end: date) -> dict:
    path = TAILS / filename
    header, merged = load_existing(path)
    start = start_for(path, configured_start)
    for item in funding_api_rows(symbol, start, end):
        ms = int(item["fundingTime"])
        merged[ms] = [str(ms), iso_ms(ms), "", item["fundingRate"]]
    keys = sorted(merged)
    for index, ms in enumerate(keys):
        row = merged[ms]
        if len(row) >= 3 and row[2]:
            continue
        if index > 0:
            hours = (ms - keys[index - 1]) / 3_600_000
        elif len(keys) > 1:
            hours = (keys[1] - ms) / 3_600_000
        else:
            hours = 0
        row[2] = f"{hours:g}" if hours > 0 else ""
    out_header = header or [
        "funding_time_ms", "funding_time_utc", "funding_interval_hours", "last_funding_rate",
    ]
    write_gzip(path, out_header, merged)
    return manifest_entry(path, merged, "Binance USD-M Futures API funding history", [])


def build_bitstamp() -> dict:
    payload = request_bytes(BITSTAMP_URL)
    raw_path = TAILS / "BTCUSD_BITSTAMP_1m_latest.csv"
    raw_path.write_bytes(payload)
    rows: dict[int, list[str]] = {}
    with raw_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            first = int(float(row[0]))
            ms = first * 1000 if first < 100_000_000_000 else norm_ms(row[0])
            rows[ms] = row
    gz_path = TAILS / "BTCUSD_BITSTAMP_1m_latest.csv.gz"
    tmp = gz_path.with_suffix(gz_path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="", compresslevel=6) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for ms in sorted(rows):
            writer.writerow(rows[ms])
    os.replace(tmp, gz_path)
    raw_path.unlink()

    chunk_dir = TAILS / "BTCUSD_BITSTAMP_1m_latest.parts"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir()
    chunks = []
    with gz_path.open("rb") as source:
        index = 0
        while True:
            payload = source.read(700_000)
            if not payload:
                break
            part = chunk_dir / f"part_{index:03d}.bin"
            part.write_bytes(payload)
            chunks.append({
                "file": f"{chunk_dir.name}/{part.name}",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
            index += 1

    entry = manifest_entry(gz_path, rows, "ff137 Bitstamp BTCUSD latest overlap", [])
    entry["parts"] = chunks
    return entry


def manifest_entry(path: Path, rows: dict[int, list[str]], source: str, skipped: list[str]) -> dict:
    keys = sorted(rows)
    if not keys:
        raise RuntimeError(f"no rows produced for {path.name}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "file": path.name,
        "source": source,
        "rows": len(keys),
        "first_timestamp_ms": keys[0],
        "first_utc": iso_ms(keys[0]),
        "last_timestamp_ms": keys[-1],
        "last_utc": iso_ms(keys[-1]),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "recent_archives_not_yet_published": skipped,
    }


def main() -> None:
    end = datetime.now(UTC).date() - timedelta(days=1)
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "through_requested_utc_day": end.isoformat(),
        "timestamp_unit": "milliseconds (Binance outputs); source-native seconds in Bitstamp rows",
        "files": [],
    }
    for symbol, start in SPOT_SYMBOLS.items():
        manifest["files"].append(build_alt(symbol, start, end))
    for filename, (market, symbol, start) in BTC_CONFIGS.items():
        manifest["files"].append(build_btc(filename, market, symbol, start, end))
    for filename, (symbol, start) in USDM_TAILS.items():
        manifest["files"].append(build_usdm_tail(filename, symbol, start, end))
    for filename, (symbol, start) in FUNDING_TAILS.items():
        manifest["files"].append(build_funding_tail(filename, symbol, start, end))
    manifest["files"].append(build_bitstamp())
    manifest["files"].sort(key=lambda item: item["file"])
    (TAILS / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
