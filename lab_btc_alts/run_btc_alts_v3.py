import io, zipfile, urllib.request
import pandas as pd
import run_btc_alts_v2 as v2


def get_month_15m(sym, ym):
    """Same Binance Spot history, but download native 15m monthly archives instead of 1m.
    This preserves every timeframe used by the study (15m and above) while reducing download size enormously.
    """
    url = f'https://data.binance.vision/data/spot/monthly/klines/{sym}/15m/{sym}-15m-{ym}.zip'
    try:
        with urllib.request.urlopen(url, timeout=90) as r:
            b = r.read()
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            raw = z.read(z.namelist()[0])
        cols = ['t','o','h','l','c','v','ct','qv','n','tb','tq','x']
        q = pd.read_csv(io.BytesIO(raw), header=None, names=cols)
        t = pd.to_numeric(q.t, errors='coerce')
        unit = 'us' if t.dropna().median() > 1e14 else 'ms'
        idx = pd.to_datetime(t, unit=unit, utc=True, errors='coerce')
        x = pd.DataFrame({
            'open': pd.to_numeric(q.o, errors='coerce').values,
            'high': pd.to_numeric(q.h, errors='coerce').values,
            'low': pd.to_numeric(q.l, errors='coerce').values,
            'close': pd.to_numeric(q.c, errors='coerce').values,
            'volume': pd.to_numeric(q.v, errors='coerce').values,
        }, index=idx).dropna()
        return x[~x.index.isna()]
    except Exception as e:
        print('MISS', sym, ym, type(e).__name__, str(e)[:100])
        return None


v2.get_month = get_month_15m
v2.main()
