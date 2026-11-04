# Market data updates

The `pinescript` branch publishes one-minute overlap tails under `tails/`.
These tails extend the complete histories held in the trading projects;
they are not themselves complete histories.

When a Binance daily archive returns 404, the updater checks the official
monthly archive for the same market, symbol, and UTC day. It never replaces
futures data with spot data or fills missing minutes with synthetic candles.
`archive_fallbacks` records the recovered days in `tails/manifest.json`.

If a USD-M day is absent from both archive families, the updater requests
that closed UTC day's 1m candles from Binance's official public REST API.
It requires all 1,440 ordered minutes, exact close timestamps, and valid OHLCV
bounds. Response URLs and SHA-256 digests remain in the manifest across runs.
`data_recovery/REST_KLINE_VALIDATION_2026-10-29.json` records a full comparison:
all 12 fields of all 1,440 official REST candles equal the checksum-verified
daily archive.

Raw-trade aggregation is not used. The negative control in
`data_recovery/RAW_TRADE_VALIDATION_2026-10-29.json` failed to reproduce the
official daily klines, so those reconstructed rows were rejected.

Each dataset is staged independently. A failed update retains that dataset's
last published bytes, records `error_preserved` and the exception, and allows
other datasets to publish. The final workflow health step fails if a required
crypto update failed, after publishing the other successful updates.

Source gaps for the separate XAU/SPY instruments are recorded with interval
boundaries and missing-minute counts. They are not assumed to be normal
market closures and do not block BTC or alt data. Missing historical days
inside a crypto update are rejected.

Read each file's `last_utc`, `update_status`, `continuity`, and
`lag_days_against_requested_day`. `generated_at_utc` is the time the manifest
was generated, not a promise that every source reached the requested day.
`health: degraded` exposes errors, source gaps, or delayed source coverage.

The outage beginning 2026-10-31 had two observed failure modes: an XAU gap
aborted the all-or-nothing update, then missing older BTC futures daily
archives exceeded the two-day grace period. The daily object for
2026-10-28 was absent. The October monthly archive was available but also
omitted that UTC day. See https://github.com/binance/binance-public-data/issues/525.

Run the recovery and publication regression checks with:

```sh
python -m unittest discover -s tests -p test_market_data_update.py -v
```

Updating this repository does not replace files uploaded to ChatGPT projects.
That separate action must be verified in each project's native file list.

