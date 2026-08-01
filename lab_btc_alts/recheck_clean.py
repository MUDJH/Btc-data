import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path('lab_btc_alts/recheck_clean_output')
OUT.mkdir(parents=True, exist_ok=True)
END = pd.Timestamp('2026-07-11T00:00:00Z')
END_MS = int(END.timestamp() * 1000)

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
CONFIG = {
    '4h': {'pair_start': '2018-04-17', 'breadth_start': '2020-08-11', 'h': [1, 3, 6, 12]},
    '1d': {'pair_start': '2018-04-17', 'breadth_start': '2020-08-11', 'h': [1, 3, 5, 10]},
}


def utc_ts(x):
    t = pd.Timestamp(x)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def fetch_klines(symbol, interval, start_date):
    start_ms = int(utc_ts(start_date).timestamp() * 1000)
    rows = []
    cursor = start_ms
    requests = 0
    while cursor < END_MS:
        params = urllib.parse.urlencode({
            'symbol': symbol,
            'interval': interval,
            'startTime': cursor,
            'endTime': END_MS - 1,
            'limit': 1000,
        })
        url = 'https://api.binance.com/api/v3/klines?' + params
        data = None
        for attempt in range(7):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = json.loads(r.read().decode('utf-8'))
                break
            except Exception:
                if attempt == 6:
                    raise
                time.sleep(1.0 + attempt * 1.5)
        requests += 1
        if not data:
            break
        rows.extend(data)
        nxt = int(data[-1][0]) + 1
        if nxt <= cursor:
            break
        cursor = nxt
        if len(data) < 1000:
            break
        time.sleep(0.04)

    cols = ['t','open','high','low','close','volume','ct','quote','trades','tb','tq','ignore']
    d = pd.DataFrame(rows, columns=cols)
    if d.empty:
        raise RuntimeError(f'No data for {symbol} {interval}')
    d = d.drop_duplicates('t').sort_values('t')
    d.index = pd.to_datetime(d['t'].astype('int64'), unit='ms', utc=True)
    for c in ['open','high','low','close','volume']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d[['open','high','low','close','volume']].dropna()
    print(interval, symbol, 'rows=', len(d), 'requests=', requests, 'from=', d.index.min(), 'to=', d.index.max(), flush=True)
    return d


def features(d):
    z = d.copy()
    z['ret'] = z['close'] / z['open'] - 1.0
    z['red'] = z['ret'] < 0
    z['green'] = z['ret'] > 0
    z['abs_body'] = z['ret'].abs()
    # Strictly causal references: the current candle is excluded.
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
            r = {
                'tf': tf, 'alt': alt.replace('USDT',''), 'event': event, 'sample': sample,
                'start': str(idx.min()), 'end': str(idx.max()),
                **event_stats(d, mask, horizons)
            }
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
    pair_out = []
    breadth_out = []
    era_out = []
    manifest = []

    for tf, cfg in CONFIG.items():
        data = {}
        # Fetch pair histories from their actual listings; BTC/ADA therefore starts in 2018.
        for s in ['BTCUSDT'] + ALTS:
            start = max(utc_ts(cfg['pair_start']), utc_ts(STARTS[s])).strftime('%Y-%m-%d')
            x = features(fetch_klines(s, tf, start))
            data[s] = x
            manifest.append({'tf': tf, 'symbol': s, 'rows': len(x), 'start': str(x.index.min()), 'end': str(x.index.max())})

        for alt in ALTS:
            pair_out += pair_rows(data['BTCUSDT'], data[alt], tf, alt, cfg['h'])

        # Breadth naturally starts when SOL exists because all six alts must be present.
        br_rows, d = breadth_rows(data, tf, cfg['h'])
        breadth_out += br_rows
        if tf == '4h':
            era_out += era_rows(d, tf, cfg['h'])

    P = pd.DataFrame(pair_out)
    B = pd.DataFrame(breadth_out)
    E = pd.DataFrame(era_out)
    M = pd.DataFrame(manifest)
    P.to_csv(OUT / 'PAIRWISE.csv', index=False)
    B.to_csv(OUT / 'BREADTH.csv', index=False)
    E.to_csv(OUT / 'ERA_4H.csv', index=False)
    M.to_csv(OUT / 'MANIFEST.csv', index=False)

    with open(OUT / 'REPORT.md', 'w', encoding='utf-8') as f:
        f.write('# RECHECK LIMPIO — BTC vs Alts\n\n')
        f.write('Fuente homogénea: Binance Spot. Estudio descriptivo por cierre de vela; no es un modelo de ejecución. ') 
        f.write('Volumen relativo = volumen actual / media de las 20 velas cerradas anteriores. ') 
        f.write('Vela fuerte = cuerpo absoluto actual >= mediana de los cuerpos de las 50 velas cerradas anteriores. ') 
        f.write('Se reportan tanto todas las velas como solo el inicio de cada episodio (onsets).\n\n')
        f.write('## Manifest\n\n' + M.to_markdown(index=False) + '\n\n')
        f.write('## ADA 4H\n\n' + P[(P['alt']=='ADA') & (P['tf']=='4h')].to_markdown(index=False) + '\n\n')
        f.write('## Breadth 4H\n\n' + B[B['tf']=='4h'].to_markdown(index=False) + '\n\n')
        f.write('## Robustez 4H por era\n\n' + E.to_markdown(index=False) + '\n')
    print('RECHECK_CLEAN_DONE', flush=True)


if __name__ == '__main__':
    main()
