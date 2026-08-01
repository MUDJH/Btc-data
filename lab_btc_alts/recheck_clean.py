import io
import zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path('lab_btc_alts/recheck_clean_output')
OUT.mkdir(parents=True, exist_ok=True)

STARTS = {
    'BTCUSDT': '2017-08-17',
    'ETHUSDT': '2017-08-17',
    'LTCUSDT': '2017-12-13',
    'ADAUSDT': '2018-04-17',
    'XRPUSDT': '2018-05-04',
    'DOGEUSDT': '2019-07-05',
    'SOLUSDT': '2020-08-11',
}
ALTS = ['ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT', 'DOGEUSDT', 'LTCUSDT']
HORIZONS = {'4h': [1, 3, 6, 12], '1d': [1, 3, 5, 10]}


def utc_ts(x):
    t = pd.Timestamp(x)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def months_from(start_date, end_ym='2026-06'):
    s = utc_ts(start_date)
    y, m = s.year, s.month
    ey, em = map(int, end_ym.split('-'))
    out = []
    while (y, m) <= (ey, em):
        out.append(f'{y:04d}-{m:02d}')
        m += 1
        if m == 13:
            y += 1
            m = 1
    return out


def parse_zip_bytes(blob):
    cols = ['t','open','high','low','close','volume','ct','quote','trades','tb','tq','ignore']
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    q = pd.read_csv(io.BytesIO(raw), header=None, names=cols)
    t = pd.to_numeric(q['t'], errors='coerce')
    unit = 'us' if t.dropna().median() > 1e14 else 'ms'
    idx = pd.to_datetime(t, unit=unit, utc=True, errors='coerce')
    d = pd.DataFrame({
        'open': pd.to_numeric(q['open'], errors='coerce').to_numpy(),
        'high': pd.to_numeric(q['high'], errors='coerce').to_numpy(),
        'low': pd.to_numeric(q['low'], errors='coerce').to_numpy(),
        'close': pd.to_numeric(q['close'], errors='coerce').to_numpy(),
        'volume': pd.to_numeric(q['volume'], errors='coerce').to_numpy(),
    }, index=idx)
    return d[~d.index.isna()].dropna()


def download_one(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=45) as r:
            return parse_zip_bytes(r.read()), None
    except Exception as e:
        return None, f'{type(e).__name__}: {e}'


def load_4h(symbol):
    start = STARTS[symbol]
    urls = []
    for ym in months_from(start):
        urls.append((ym, f'https://data.binance.vision/data/spot/monthly/klines/{symbol}/4h/{symbol}-4h-{ym}.zip'))
    # July 2026 is not yet in the monthly archive; add the ten daily archives.
    for day in range(1, 11):
        ds = f'2026-07-{day:02d}'
        urls.append((ds, f'https://data.binance.vision/data/spot/daily/klines/{symbol}/4h/{symbol}-4h-{ds}.zip'))

    parts = []
    misses = []
    with ThreadPoolExecutor(max_workers=20) as ex:
        futs = {ex.submit(download_one, url): (tag, url) for tag, url in urls}
        for fut in as_completed(futs):
            tag, url = futs[fut]
            d, err = fut.result()
            if d is not None and not d.empty:
                parts.append(d)
            else:
                misses.append((tag, err))
    if not parts:
        raise RuntimeError(f'No Binance Vision data for {symbol}')
    x = pd.concat(parts).sort_index()
    x = x[~x.index.duplicated(keep='last')]
    x = x[x.index >= utc_ts(start)]
    print(symbol, '4H rows=', len(x), 'from=', x.index.min(), 'to=', x.index.max(), 'misses=', len(misses), flush=True)
    return x, misses


def to_daily(x):
    return x.resample('1D', label='left', closed='left').agg(
        open=('open','first'), high=('high','max'), low=('low','min'), close=('close','last'), volume=('volume','sum')
    ).dropna()


def features(d):
    z = d.copy()
    z['ret'] = z['close'] / z['open'] - 1.0
    z['red'] = z['ret'] < 0
    z['green'] = z['ret'] > 0
    z['abs_body'] = z['ret'].abs()
    # Causal: current bar excluded from both reference windows.
    z['body_ref50'] = z['abs_body'].rolling(50, min_periods=50).median().shift(1)
    z['strong'] = z['abs_body'] >= z['body_ref50']
    z['vol_ref20'] = z['volume'].rolling(20, min_periods=20).mean().shift(1)
    z['vol_rel20'] = z['volume'] / z['vol_ref20']
    z['high_vol'] = z['vol_rel20'] > 1.0
    return z


def add_forward(frame, prefix, close, horizons):
    for h in horizons:
        frame[f'{prefix}_f{h}'] = close.shift(-h) / close - 1.0


def event_stats(d, mask, horizons, prefix='btc'):
    out = {'n': int(mask.sum()), 'pct_bars': float(100 * mask.mean()) if len(mask) else np.nan}
    for h in horizons:
        x = d.loc[mask, f'{prefix}_f{h}'].dropna()
        out[f'mean_{h}'] = 100 * x.mean() if len(x) else np.nan
        out[f'median_{h}'] = 100 * x.median() if len(x) else np.nan
        out[f'pos_{h}'] = 100 * (x > 0).mean() if len(x) else np.nan
    return out


def onset(mask):
    return mask & ~mask.shift(1, fill_value=False)


def pair_rows(b, a, tf, alt, horizons):
    idx = b.index.intersection(a.index)
    if len(idx) == 0:
        return []
    b = b.loc[idx]
    a = a.loc[idx]
    d = pd.DataFrame(index=idx)
    d['btc_red'] = b['red']
    d['btc_strong'] = b['strong']
    d['btc_high_vol'] = b['high_vol']
    d['btc_vrel'] = b['vol_rel20']
    d['alt_green'] = a['green']
    d['alt_strong'] = a['strong']
    d['alt_high_vol'] = a['high_vol']
    d['alt_vrel'] = a['vol_rel20']
    add_forward(d, 'btc', b['close'], horizons)
    add_forward(d, 'alt', a['close'], horizons)

    masks = {
        'BTC red baseline': d['btc_red'],
        'BTC red + ALT green': d['btc_red'] & d['alt_green'],
        'strong opposite candles': d['btc_red'] & d['btc_strong'] & d['alt_green'] & d['alt_strong'],
        'strong + both high volume': d['btc_red'] & d['btc_strong'] & d['btc_high_vol'] & d['alt_green'] & d['alt_strong'] & d['alt_high_vol'],
        'strong + ALT volRel > BTC volRel': d['btc_red'] & d['btc_strong'] & d['alt_green'] & d['alt_strong'] & (d['alt_vrel'] > d['btc_vrel']),
    }
    baseline = event_stats(d, masks['BTC red baseline'], horizons)
    rows = []
    for event, raw_mask in masks.items():
        for sample, mask in [('bars', raw_mask), ('onsets', onset(raw_mask))]:
            r = {'tf': tf, 'alt': alt.replace('USDT',''), 'event': event, 'sample': sample,
                 'start': str(idx.min()), 'end': str(idx.max()), **event_stats(d, mask, horizons)}
            for h in horizons:
                aa = d.loc[mask, f'alt_f{h}'].dropna()
                joint = d.loc[mask, [f'btc_f{h}', f'alt_f{h}']].dropna()
                r[f'alt_mean_{h}'] = 100 * aa.mean() if len(aa) else np.nan
                r[f'alt_pos_{h}'] = 100 * (aa > 0).mean() if len(aa) else np.nan
                r[f'btc_outperforms_alt_{h}'] = 100 * (joint[f'btc_f{h}'] > joint[f'alt_f{h}']).mean() if len(joint) else np.nan
                r[f'pos_lift_vs_btc_red_{h}'] = r[f'pos_{h}'] - baseline[f'pos_{h}'] if np.isfinite(r[f'pos_{h}']) else np.nan
                r[f'mean_lift_vs_btc_red_{h}'] = r[f'mean_{h}'] - baseline[f'mean_{h}'] if np.isfinite(r[f'mean_{h}']) else np.nan
            rows.append(r)
    return rows


def breadth_rows(data, tf, horizons):
    common = data['BTCUSDT'].index
    for s in ALTS:
        common = common.intersection(data[s].index)
    if len(common) == 0:
        return [], pd.DataFrame()
    b = data['BTCUSDT'].loc[common]
    d = pd.DataFrame(index=common)
    d['btc_red'] = b['red']
    d['btc_strong'] = b['strong']
    d['btc_high_vol'] = b['high_vol']
    add_forward(d, 'btc', b['close'], horizons)

    green = []
    strong_green = []
    hv_strong_green = []
    for s in ALTS:
        a = data[s].loc[common]
        green.append(a['green'].astype(int))
        strong_green.append((a['green'] & a['strong']).astype(int))
        hv_strong_green.append((a['green'] & a['strong'] & a['high_vol']).astype(int))
    d['green_count'] = sum(green)
    d['strong_green_count'] = sum(strong_green)
    d['hv_strong_green_count'] = sum(hv_strong_green)

    baseline = event_stats(d, d['btc_red'], horizons)
    events = {'BTC red baseline': d['btc_red']}
    for k in range(1, 7):
        events[f'BTC red + >= {k}/6 alts green'] = d['btc_red'] & (d['green_count'] >= k)
        events[f'BTC red strong + >= {k}/6 alts green strong'] = d['btc_red'] & d['btc_strong'] & (d['strong_green_count'] >= k)
        events[f'BTC red strong highVol + >= {k}/6 alts green strong highVol'] = d['btc_red'] & d['btc_strong'] & d['btc_high_vol'] & (d['hv_strong_green_count'] >= k)

    rows = []
    for event, raw_mask in events.items():
        for sample, mask in [('bars', raw_mask), ('onsets', onset(raw_mask))]:
            r = {'tf': tf, 'event': event, 'sample': sample, 'start': str(common.min()), 'end': str(common.max()), **event_stats(d, mask, horizons)}
            for h in horizons:
                r[f'pos_lift_vs_btc_red_{h}'] = r[f'pos_{h}'] - baseline[f'pos_{h}'] if np.isfinite(r[f'pos_{h}']) else np.nan
                r[f'mean_lift_vs_btc_red_{h}'] = r[f'mean_{h}'] - baseline[f'mean_{h}'] if np.isfinite(r[f'mean_{h}']) else np.nan
            rows.append(r)
    return rows, d


def era_rows(d, tf, horizons):
    if d.empty:
        return []
    keys = {
        '>=3/6 strong highVol': d['btc_red'] & d['btc_strong'] & d['btc_high_vol'] & (d['hv_strong_green_count'] >= 3),
        '>=4/6 green': d['btc_red'] & (d['green_count'] >= 4),
        '6/6 green': d['btc_red'] & (d['green_count'] == 6),
    }
    eras = [
        ('2020-2022','2020-08-11','2022-12-31 23:59:59'),
        ('2023-2024','2023-01-01','2024-12-31 23:59:59'),
        ('2025-2026','2025-01-01','2026-07-10 23:59:59'),
    ]
    rows = []
    for key, raw in keys.items():
        for sample, base_mask in [('bars', raw), ('onsets', onset(raw))]:
            for era, a, b in eras:
                mask = base_mask & (d.index >= utc_ts(a)) & (d.index <= utc_ts(b))
                rows.append({'tf': tf, 'key': key, 'sample': sample, 'era': era, **event_stats(d, mask, horizons)})
    return rows


def main():
    raw4 = {}
    download_manifest = []
    for s in ['BTCUSDT'] + ALTS:
        x, misses = load_4h(s)
        raw4[s] = x
        download_manifest.append({'symbol': s, 'rows4h': len(x), 'start': str(x.index.min()), 'end': str(x.index.max()), 'missing_archives': len(misses)})

    pair_out = []
    breadth_out = []
    era_out = []
    tf_manifest = []

    for tf in ['4h', '1d']:
        raw = raw4 if tf == '4h' else {s: to_daily(x) for s, x in raw4.items()}
        data = {s: features(x) for s, x in raw.items()}
        horizons = HORIZONS[tf]
        for s, x in raw.items():
            tf_manifest.append({'tf': tf, 'symbol': s, 'rows': len(x), 'start': str(x.index.min()), 'end': str(x.index.max())})
        for alt in ALTS:
            pair_out += pair_rows(data['BTCUSDT'], data[alt], tf, alt, horizons)
        br, d = breadth_rows(data, tf, horizons)
        breadth_out += br
        if tf == '4h':
            era_out += era_rows(d, tf, horizons)

    P = pd.DataFrame(pair_out)
    B = pd.DataFrame(breadth_out)
    E = pd.DataFrame(era_out)
    M = pd.DataFrame(tf_manifest)
    D = pd.DataFrame(download_manifest)
    P.to_csv(OUT / 'PAIRWISE.csv', index=False)
    B.to_csv(OUT / 'BREADTH.csv', index=False)
    E.to_csv(OUT / 'ERA_4H.csv', index=False)
    M.to_csv(OUT / 'MANIFEST.csv', index=False)
    D.to_csv(OUT / 'DOWNLOAD_MANIFEST.csv', index=False)

    with open(OUT / 'REPORT.md', 'w', encoding='utf-8') as f:
        f.write('# RECHECK LIMPIO — BTC vs Alts\n\n')
        f.write('Fuente homogénea: Binance Spot vía Binance Vision. 4H nativo y diario agregado causalmente desde 4H UTC. ')
        f.write('Volumen relativo = volumen actual / media de las 20 velas cerradas anteriores. ')
        f.write('Vela fuerte = cuerpo absoluto actual >= mediana de los cuerpos de las 50 velas cerradas anteriores. ')
        f.write('Se reportan todas las velas y también solo los inicios de episodio (onsets).\n\n')
        f.write('## Descarga\n\n' + D.to_markdown(index=False) + '\n\n')
        f.write('## ADA 4H\n\n' + P[(P['alt']=='ADA') & (P['tf']=='4h')].to_markdown(index=False) + '\n\n')
        f.write('## Breadth 4H\n\n' + B[B['tf']=='4h'].to_markdown(index=False) + '\n\n')
        f.write('## Robustez 4H por era\n\n' + E.to_markdown(index=False) + '\n')
    print('RECHECK_CLEAN_DONE', flush=True)


if __name__ == '__main__':
    main()
