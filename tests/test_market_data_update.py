"""Failure cases that caused the 2026-10-31/11-03 update outage."""
import contextlib
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location("updater", Path(__file__).resolve().parents[1] / "scripts/build_market_data_update.py")
updater = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(updater)


def raw_row(ms):
    return [str(ms), "100", "102", "99", "101", "3", str(ms + 59_999), "303", "1", "1", "101", "0"]


def missing(url):
    return urllib.error.HTTPError(url, 404, "Not Found", {}, None)


def archive(rows):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        handle.writestr("candles.csv", "\n".join(",".join(row) for row in rows) + "\n")
    return buffer.getvalue()


class UpdateRecoveryTests(unittest.TestCase):
    def setUp(self):
        updater.read_month_archive.cache_clear()
        updater.ARCHIVE_FALLBACKS.clear()
        recovery_patch = patch.object(updater, "RECOVERY_DIR", Path(__file__).parent / "no_live_snapshots_in_unit_tests")
        recovery_patch.start()
        self.addCleanup(recovery_patch.stop)

    def test_daily_404_recovers_only_requested_day_from_monthly(self):
        day = date(2026, 10, 28)
        ms = int(datetime(2026, 10, 28, tzinfo=timezone.utc).timestamp() * 1000)
        payload = archive([raw_row(ms - 60_000), raw_row(ms), raw_row(ms + 60_000), raw_row(ms + 86_400_000)])

        def request(url):
            if "/daily/" in url:
                raise missing(url)
            self.assertIn("/monthly/", url)
            return payload

        with patch.object(updater, "request_bytes", side_effect=request):
            result = updater.read_archive("um", "BTCUSDT", day)
        self.assertEqual([int(row[0]) for row in result], [ms, ms + 60_000])
        self.assertEqual(updater.ARCHIVE_FALLBACKS[0]["day"], "2026-10-28")

    def test_monthly_spot_microseconds_are_filtered_as_milliseconds(self):
        ms = int(datetime(2026, 10, 28, tzinfo=timezone.utc).timestamp() * 1000)
        rows = [raw_row(ms * 1000), raw_row((ms + 86_400_000) * 1000)]
        with patch.object(updater, "request_bytes", side_effect=missing("daily")), patch.object(updater, "read_month_archive", return_value=rows):
            result = updater.read_archive("spot", "BTCUSDT", date(2026, 10, 28))
        self.assertEqual(len(result), 1)
        self.assertEqual(updater.norm_ms(result[0][0]), ms)

    def test_missing_monthly_does_not_invent_rows_or_swallow_original_url(self):
        with patch.object(updater, "request_bytes", side_effect=lambda url: (_ for _ in ()).throw(missing(url))):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                updater.read_archive("um", "BTCUSDT", date(2026, 10, 28))
        self.assertIn("/daily/", caught.exception.url)
        self.assertEqual(updater.ARCHIVE_FALLBACKS, [])

    def test_monthly_without_requested_day_does_not_substitute_another_day(self):
        ms = int(datetime(2026, 10, 27, tzinfo=timezone.utc).timestamp() * 1000)
        with patch.object(updater, "request_bytes", side_effect=missing("daily")), patch.object(updater, "read_month_archive", return_value=[raw_row(ms)]):
            with self.assertRaises(urllib.error.HTTPError):
                updater.read_archive("um", "BTCUSDT", date(2026, 10, 28))

    def test_unavailable_day_before_newer_data_is_a_failure(self):
        def read(market, symbol, day):
            if day.day == 2:
                raise missing("day2")
            ms = int(datetime.combine(day, datetime.min.time(), timezone.utc).timestamp() * 1000)
            return [raw_row(ms)]

        with patch.object(updater, "read_archive", side_effect=read):
            with self.assertRaisesRegex(RuntimeError, "inside the requested interval"):
                updater.download_days("spot", "BTCUSDT", date(2026, 11, 1), date(2026, 11, 3))

    def test_crypto_continuity_rejects_holes(self):
        with self.assertRaisesRegex(RuntimeError, "no candles synthesized"):
            updater.require_continuous({0: ["0"], 120_000: ["120000"]}, "BTC.csv.gz")

    def test_rest_fallback_paginates_and_preserves_all_1440_closed_minutes(self):
        day = date(2026, 10, 28)
        start = int(datetime(2026, 10, 28, tzinfo=timezone.utc).timestamp() * 1000)
        official = [raw_row(start + minute * 60_000) for minute in range(1440)]
        with patch.object(updater, "request_bytes", side_effect=[json.dumps(official[:1000]).encode(), json.dumps(official[1000:]).encode()]):
            result, receipts = updater.read_rest_day("BTCUSDT", day)
        self.assertEqual(result, official)
        self.assertEqual(len(receipts), 2)
        self.assertIn("fapi.binance.com/fapi/v1/klines", receipts[0]["url"])

    def test_rest_fallback_rejects_missing_minutes(self):
        day = date(2026, 10, 28)
        start = int(datetime(2026, 10, 28, tzinfo=timezone.utc).timestamp() * 1000)
        with patch.object(updater, "request_bytes", return_value=json.dumps([raw_row(start + 60_000)]).encode()):
            with self.assertRaisesRegex(RuntimeError, "missing, duplicate"):
                updater.read_rest_day("BTCUSDT", day)

    def test_rest_fallback_rejects_invalid_price_bounds(self):
        start = int(datetime(2026, 10, 28, tzinfo=timezone.utc).timestamp() * 1000)
        row = raw_row(start)
        row[2] = "99"
        with patch.object(updater, "request_bytes", return_value=json.dumps([row]).encode()):
            with self.assertRaisesRegex(RuntimeError, "invalid USD-M REST OHLCV"):
                updater.read_rest_day("BTCUSDT", date(2026, 10, 28))

    def test_verified_saved_day_works_without_live_api(self):
        import hashlib
        day = date(2026, 10, 28)
        start = int(datetime(2026, 10, 28, tzinfo=timezone.utc).timestamp() * 1000)
        candles = [raw_row(start + minute * 60_000) for minute in range(1440)]
        payload = ("\n".join(",".join(row) for row in candles) + "\n").encode()
        compressed = gzip.compress(payload, mtime=0)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            (root / "day.csv.gz").write_bytes(compressed)
            entry = {"market": "um", "symbol": "BTCUSDT", "day": day.isoformat(), "file": "day.csv.gz",
                     "sha256": hashlib.sha256(compressed).hexdigest(), "csv_sha256": hashlib.sha256(payload).hexdigest(),
                     "provenance": {"market": "um", "symbol": "BTCUSDT", "day": day.isoformat()}}
            (root / "approved_snapshots.json").write_text(json.dumps({"snapshots": [entry]}))
            with patch.object(updater, "RECOVERY_DIR", root), patch.object(updater, "request_bytes", side_effect=missing("daily")), patch.object(updater, "read_month_archive", return_value=[]), patch.object(updater, "read_rest_day", side_effect=AssertionError("live API must not be called")):
                actual = updater.read_archive("um", "BTCUSDT", day)
                self.assertEqual(actual, candles)
                (root / "day.csv.gz").write_bytes(compressed + b"tampered")
                with self.assertRaisesRegex(RuntimeError, "snapshot checksum"):
                    updater.load_approved_day("BTCUSDT", day)

    def test_failed_dataset_keeps_old_bytes_and_restores_destination(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            tails = root / "tails"
            tails.mkdir()
            old = tails / "BTC.csv.gz"
            old.write_bytes(b"original")

            def fail():
                (updater.TAILS / old.name).write_bytes(b"candidate")
                raise RuntimeError("incomplete update")

            with patch.object(updater, "ROOT", root), patch.object(updater, "TAILS", tails):
                with self.assertRaisesRegex(RuntimeError, "incomplete update"):
                    updater.run_isolated(old.name, fail)
                self.assertEqual(updater.TAILS, tails)
            self.assertEqual(old.read_bytes(), b"original")

    def test_gold_gap_is_described_without_filling_or_blocking_crypto(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp, patch.object(updater, "TAILS", Path(temp)):
            with patch.object(updater, "download_days", return_value=([raw_row(0), raw_row(120_000)], [])):
                result = updater.build_usdm_tail("XAU.csv.gz", "XAUUSDT", date(1970, 1, 1), date(1970, 1, 1))
            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["continuity"]["missing_minutes"], 1)
            with gzip.open(Path(temp) / "XAU.csv.gz", "rt") as handle:
                self.assertEqual(len(handle.readlines()), 3)

    def test_failed_optional_asset_still_publishes_required_files(self):
        self._test_isolation(required_failure=False)

    def test_required_failure_is_explicit_and_preserves_last_good_file(self):
        self._test_isolation(required_failure=True)

    def _test_isolation(self, required_failure):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            tails = root / "tails"
            tails.mkdir()
            end = datetime.now(timezone.utc).date() - timedelta(days=1)
            ms = int(datetime.combine(end, datetime.min.time(), timezone.utc).timestamp() * 1000)
            retained_name = "ADAUSDT_1m_spot.csv.gz" if required_failure else "XAU.csv.gz"
            with gzip.open(tails / retained_name, "wt") as handle:
                handle.write("timestamp,open,high,low,close,volume\n" + f"{ms},100,102,99,101,3\n")
            retained = (tails / retained_name).read_bytes()

            def success(filename):
                path = updater.TAILS / filename
                updater.write_gzip(path, ["timestamp", "open", "high", "low", "close", "volume"], {ms: [str(ms), "100", "102", "99", "101", "3"]})
                entry = updater.manifest_entry(path, {ms: [str(ms)]}, "fixture", [])
                if filename == "BTCUSD_BITSTAMP_1m_latest.csv.gz":
                    entry["parts"] = []
                return entry

            def alt(symbol, start, end):
                if required_failure:
                    raise RuntimeError("daily and monthly unavailable")
                return success(f"{symbol}_1m_spot.csv.gz")

            with patch.object(updater, "ROOT", root), patch.object(updater, "TAILS", tails), patch.object(updater, "SPOT_SYMBOLS", {"ADAUSDT": end}), patch.object(updater, "BTC_CONFIGS", {}), patch.object(updater, "USDM_TAILS", {"XAU.csv.gz": ("XAUUSDT", end)}), patch.object(updater, "build_alt", side_effect=alt), patch.object(updater, "build_usdm_tail", side_effect=RuntimeError("optional archive missing")), patch.object(updater, "build_bitstamp", side_effect=lambda: success("BTCUSD_BITSTAMP_1m_latest.csv.gz")), contextlib.redirect_stdout(io.StringIO()):
                updater.main()
            result = json.loads((tails / "manifest.json").read_text())
            self.assertEqual(result["required_crypto_failed"], required_failure)
            self.assertEqual(result["health"], "degraded")
            self.assertTrue((tails / "BTCUSD_BITSTAMP_1m_latest.csv.gz").exists())
            self.assertEqual(next(item for item in result["files"] if item["file"] == "BTCUSD_BITSTAMP_1m_latest.csv.gz")["update_status"], "updated")
            self.assertEqual((tails / retained_name).read_bytes(), retained)
            entry = next(item for item in result["files"] if item["file"] == retained_name)
            self.assertEqual(entry["update_status"], "error_preserved")


if __name__ == "__main__":
    unittest.main()

