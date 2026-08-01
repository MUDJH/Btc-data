import io
import types
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd

# Load the V2 engine after correcting the accidental use of the reserved Python
# keyword ``as`` as an attribute (d.as).  Keeping the correction here avoids
# rewriting the research engine while making the run fully reproducible.
engine_path = Path(__file__).with_name('run_btc_alts_v2.py')
src = engine_path.read_text(encoding='utf-8').replace('d.as', "d['as']")
v2 = types.ModuleType('run_btc_alts_v2_fixed')
v2.__file__ = str(engine_path)
exec(compile(src, str(engine_path), 'exec'), v2.__dict__)


def get_month_15m(sym, ym):
    """Download native Binance Spot 15m monthly archives.

    The study only uses 15m and higher timeframes, so native 15m archives are
    equivalent for OHLCV aggregation while being far smaller than the 1m
    archive set.
    """
    url = f'https://data.binance.vision/data/spot/monthly/klines/{sym}/15m/{sym}-15m-{ym}.zip'
    try:
        with urllib.request.urlopen(url, timeout=90) as r:
            b = r.read()
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            raw = z.read(z.namelist()[0])
        cols = ['t', 'o', 'h', 'l', 'c', 'v', 'ct', 'qv', 'n', 'tb', 'tq', 'x']
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
        print('MISS', sym, ym, type(e).__name__, str(e)[:120])
        return None


v2.get_month = get_month_15m
v2.main()
