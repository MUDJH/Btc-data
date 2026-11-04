#!/usr/bin/env python3
"""Build overlap-safe market-data tails for ChatGPT backtests.

Sources:
- Binance Public Data daily 1m kline archives.
- ff137/bitstamp-btcusd-minute-data latest overlap CSV.

Binance timestamps are normalized to Unix milliseconds. Bitstamp preserves
its source-native seconds for compatibility, as declared in the manifest.
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
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAILS = ROOT / "tails"
TAILS.mkdir(parents=True, exist_ok=True)
UTC = timezone.utc
USER_AGENT = "MUDJH-Btc-data-updater/1.0"
ARCHIVE_FALLBACKS: list[dict] = []
RECOVERY_DIR = Path(__file__).resolve().parents[1] / "data_recovery"

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


def parse_archive(payload: bytes, url: str) -> list[list[str]]:
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


@lru_cache(maxsize=4)
def read_month_archive(market: str, symbol: str, month: str) -> list[list[str]]:
    prefix = "spot" if market == "spot" else "futures/um"
    url = (f"https://data.binance.vision/data/{prefix}/monthly/klines/"
           f"{symbol}/1m/{symbol}-1m-{month}.zip")
    return parse_archive(request_bytes(url), url)


def read_rest_day(symbol: str, day: date) -> tuple[list[list[str]], list[dict]]:
    """Read only closed 1m USD-M candles from the official public REST API."""
    start = int(datetime.combine(day, datetime.min.time(), UTC).timestamp() * 1000)
    stop = start + 86_400_000
    cursor = start
    rows = []
    responses = []
    while cursor < stop:
        url = (f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1m"
               f"&startTime={cursor}&endTime={stop - 1}&limit=1000")
        payload = request_bytes(url)
        data = json.loads(payload)
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"empty or invalid USD-M REST response for {symbol} {day}: {data}")
        responses.append({"url": url, "sha256": hashlib.sha256(payload).hexdigest()})
        for raw in data:
            if len(raw) != 12:
                raise RuntimeError("unexpected USD-M REST kline schema")
            row = [str(value) for value in raw]
            stamp = norm_ms(row[0])
            if stamp != cursor or stamp >= stop or norm_ms(row[6]) != stamp + 59_999:
                raise RuntimeError("USD-M REST missing, duplicate, unclosed, or out-of-range minute")
            opening, high, low, close, volume = map(Decimal, row[1:6])
            if not all(value.is_finite() for value in (opening, high, low, close, volume)) or min(opening, high, low, close) <= 0 or volume < 0 or high < max(opening, close) or low > min(opening, close):
                raise RuntimeError("invalid USD-M REST OHLCV values")
            rows.append(row)
            cursor = stamp + 60_000
    if len(rows) != 1440:
        raise RuntimeError(f"USD-M REST returned {len(rows)} minutes for a closed UTC day")
    return rows, responses


def load_approved_day(symbol: str, day: date):
    index = RECOVERY_DIR / "approved_snapshots.json"
    if not index.exists():
        return None
    entries = json.loads(index.read_text(encoding="utf-8"))["snapshots"]
    entry = next((item for item in entries if item["market"] == "um" and item["symbol"] == symbol and item["day"] == day.isoformat()), None)
    if entry is None:
        return None
    path = RECOVERY_DIR / entry["file"]
    if path.resolve().parent != RECOVERY_DIR.resolve():
        raise RuntimeError("invalid approved snapshot path")
    compressed = path.read_bytes()
    if hashlib.sha256(compressed).hexdigest() != entry["sha256"]:
        raise RuntimeError("approved snapshot checksum mismatch")
    payload = gzip.decompress(compressed)
    if hashlib.sha256(payload).hexdigest() != entry["csv_sha256"]:
        raise RuntimeError("approved snapshot CSV checksum mismatch")
    rows = list(csv.reader(io.StringIO(payload.decode("utf-8"))))
    start = int(datetime.combine(day, datetime.min.time(), UTC).timestamp() * 1000)
    if len(rows) != 1440:
        raise RuntimeError("approved snapshot must contain exactly 1440 minutes")
    for minute, row in enumerate(rows):
        if len(row) != 12 or norm_ms(row[0]) != start + minute * 60_000 or norm_ms(row[6]) != start + minute * 60_000 + 59_999:
            raise RuntimeError("approved snapshot minute coverage or schema failed")
        opening, high, low, close, volume = map(Decimal, row[1:6])
        if not all(value.is_finite() for value in (opening, high, low, close, volume)) or min(opening, high, low, close) <= 0 or volume < 0 or high < max(opening, close) or low > min(opening, close):
            raise RuntimeError("approved snapshot OHLCV validation failed")
    receipt = dict(entry["provenance"])
    receipt.update(source="Binance official USD-M REST saved snapshot", snapshot_file=entry["file"],
                   snapshot_sha256=entry["sha256"], csv_sha256=entry["csv_sha256"])
    return rows, receipt


def read_archive(market: str, symbol: str, day: date) -> list[list[str]]:
    url = archive_url(market, symbol, day)
    try:
        return parse_archive(request_bytes(url), url)
    except urllib.error.HTTPError as daily_error:
        if daily_error.code != 404:
            raise
        # Daily objects can be absent even after their data are published in
        # the official monthly archive. Never fill a missing day with prices
        # from another market, synthetic candles, or neighbouring days.
        month = day.strftime("%Y-%m")
        try:
            monthly = read_month_archive(market, symbol, month)
        except urllib.error.HTTPError as monthly_error:
            if monthly_error.code == 404:
                monthly = []
            else:
                raise
        start_ms = int(datetime.combine(day, datetime.min.time(), UTC).timestamp() * 1000)
        recovered = [row for row in monthly if start_ms <= norm_ms(row[0]) < start_ms + 86_400_000]
        if not recovered:
            if market == "um":
                approved = load_approved_day(symbol, day)
                if approved is not None:
                    rows, receipt = approved
                    ARCHIVE_FALLBACKS.append(receipt)
                    return rows
                try:
                    rest_rows, responses = read_rest_day(symbol, day)
                    ARCHIVE_FALLBACKS.append({"market": "um", "symbol": symbol,
                                              "day": day.isoformat(), "source": "Binance official USD-M REST 1m klines",
                                              "minute_rows": len(rest_rows), "responses": responses})
                    return rest_rows
                except urllib.error.HTTPError as rest_error:
                    if rest_error.code != 404:
                        raise
                    raise daily_error from rest_error
            raise daily_error
        ARCHIVE_FALLBACKS.append({"market": market, "symbol": symbol,
                                  "day": day.isoformat(), "source": "Binance monthly archive"})
        return recovered


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
    if skipped and rows:
        last_day = datetime.fromtimestamp(max(norm_ms(row[0]) for row in rows) / 1000, UTC).date()
        if any(date.fromisoformat(day) < last_day for day in skipped):
            raise RuntimeError(f"{symbol}: unavailable archive inside the requested interval: {sorted(skipped)}")
    return rows, sorted(skipped)


def gap_summary(rows: dict[int, list[str]]) -> dict:
    keys = sorted(rows)
    intervals = []
    missing_minutes = 0
    for left, right in zip(keys, keys[1:]):
        if right - left != 60_000:
            absent = max(0, (right - left) // 60_000 - 1)
            missing_minutes += absent
            intervals.append({"after_utc": iso_ms(left), "before_utc": iso_ms(right),
                              "missing_minutes": absent})
    return {"non_1m_transitions": len(intervals), "missing_minutes": missing_minutes,
            "intervals": intervals}


def require_continuous(rows: dict[int, list[str]], filename: str) -> None:
    gaps = gap_summary(rows)
    if gaps["non_1m_transitions"]:
        raise RuntimeError(f"{filename}: {gaps['non_1m_transitions']} non-1m transitions; no candles synthesized")


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
    require_continuous(merged, path.name)
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
    require_continuous(merged, path.name)
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
    # Preserve the source rows and report gaps explicitly. These separate
    # instruments must never prevent BTC/alt tails from being published.
    write_gzip(path, out_header, merged)
    return manifest_entry(path, merged, "Binance USD-M futures daily archives", skipped)


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
        if chunk_dir.resolve().parent != TAILS.resolve():
            raise RuntimeError("unsafe Bitstamp parts cleanup path")
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
        "continuity": gap_summary(rows),
    }


def run_isolated(filename: str, builder) -> dict:
    """A failed dataset leaves its previously published file untouched."""
    global TAILS
    destination = TAILS
    with tempfile.TemporaryDirectory(prefix="market-data-stage-", dir=ROOT) as temp:
        stage = Path(temp)
        if stage.resolve().parent != ROOT.resolve():
            raise RuntimeError("unsafe staging directory")
        existing = destination / filename
        if existing.exists():
            shutil.copy2(existing, stage / filename)
        TAILS = stage
        try:
            result = builder()
        finally:
            TAILS = destination
        for source in sorted(stage.rglob("*")):
            if source.is_file():
                target = destination / source.relative_to(stage)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
        if filename == "BTCUSD_BITSTAMP_1m_latest.csv.gz":
            parts = destination / "BTCUSD_BITSTAMP_1m_latest.parts"
            expected = {Path(item["file"]).name for item in result["parts"]}
            for part in parts.glob("part_*.bin"):
                if part.name not in expected:
                    part.unlink()
        return result


def main() -> None:
    end = datetime.now(UTC).date() - timedelta(days=1)
    previous_manifest = TAILS / "manifest.json"
    previous = {}
    previous_payload = {}
    if previous_manifest.exists():
        previous_payload = json.loads(previous_manifest.read_text(encoding="utf-8"))
        previous = {item["file"]: item for item in previous_payload["files"]}
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "through_requested_utc_day": end.isoformat(),
        "timestamp_unit": "milliseconds (Binance outputs); source-native seconds in Bitstamp rows",
        "files": [],
        "errors": [],
        "warnings": [],
    }
    jobs = []
    for symbol, start in SPOT_SYMBOLS.items():
        filename = f"{symbol}_1m_spot.csv.gz"
        jobs.append((filename, True, lambda s=symbol, d=start: build_alt(s, d, end)))
    for filename, (market, symbol, start) in BTC_CONFIGS.items():
        jobs.append((filename, True, lambda f=filename, m=market, s=symbol, d=start: build_btc(f, m, s, d, end)))
    for filename, (symbol, start) in USDM_TAILS.items():
        jobs.append((filename, False, lambda f=filename, s=symbol, d=start: build_usdm_tail(f, s, d, end)))
    jobs.append(("BTCUSD_BITSTAMP_1m_latest.csv.gz", True, build_bitstamp))
    for filename, required, builder in jobs:
        try:
            entry = run_isolated(filename, builder)
            entry["update_status"] = "updated"
        except Exception as exc:
            error = {"file": filename, "required_crypto": required,
                     "error": f"{type(exc).__name__}: {exc}"}
            if getattr(exc, "url", None):
                error["url"] = exc.url
            manifest["errors"].append(error)
            print(f"::warning file=scripts/build_market_data_update.py::{filename}: {error['error']}", flush=True)
            path = TAILS / filename
            if path.exists():
                _, retained = load_existing(path)
                entry = manifest_entry(path, retained, "Retained previous file; see error", [])
                old = previous.get(filename, {})
                if old.get("sha256") == entry["sha256"]:
                    entry["source"] = old.get("source", entry["source"])
                    if "parts" in old:
                        entry["parts"] = old["parts"]
                    entry["previous_manifest_generated_at_utc"] = previous_payload.get("generated_at_utc")
                entry["update_status"] = "error_preserved"
            else:
                entry = {"file": filename, "update_status": "error_no_file"}
            entry["error"] = error["error"]
        entry["required_crypto"] = required
        if entry.get("continuity", {}).get("non_1m_transitions", 0):
            warning = {"file": filename, "kind": "source_gaps", "continuity": entry["continuity"]}
            manifest["warnings"].append(warning)
            print(f"::warning::{filename}: source gaps recorded in manifest; no candles synthesized", flush=True)
        if "last_timestamp_ms" in entry:
            lag = max(0, (end - datetime.fromtimestamp(entry["last_timestamp_ms"] / 1000, UTC).date()).days)
            entry["lag_days_against_requested_day"] = lag
            if lag:
                manifest["warnings"].append({"file": filename, "kind": "source_lag", "lag_days": lag})
        manifest["files"].append(entry)
    successful = {filename for filename, _, _ in jobs
                  if next(entry for entry in manifest["files"] if entry["file"] == filename)["update_status"] == "updated"}
    symbol_files = {("spot", symbol): f"{symbol}_1m_spot.csv.gz" for symbol in SPOT_SYMBOLS}
    symbol_files.update({(market, symbol): filename for filename, (market, symbol, _) in BTC_CONFIGS.items()})
    symbol_files.update({("um", symbol): filename for filename, (symbol, _) in USDM_TAILS.items()})
    applied = [item for item in ARCHIVE_FALLBACKS if symbol_files.get((item["market"], item["symbol"])) in successful]
    history = {(item["market"], item["symbol"], item["day"]): item for item in previous_payload.get("archive_fallbacks", [])}
    history.update({(item["market"], item["symbol"], item["day"]): item for item in applied})
    manifest["archive_fallbacks"] = sorted(history.values(), key=lambda item: (item["symbol"], item["day"]))
    manifest["health"] = "degraded" if manifest["errors"] or manifest["warnings"] else "ok"
    manifest["required_crypto_failed"] = any(item["required_crypto"] for item in manifest["errors"])
    manifest["files"].sort(key=lambda item: item["file"])
    temporary_manifest = TAILS / "manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, TAILS / "manifest.json")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


