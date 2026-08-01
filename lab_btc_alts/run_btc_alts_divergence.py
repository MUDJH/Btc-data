import io, os, zipfile, urllib.request, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path('lab_btc_alts/output')
OUT.mkdir(parents=True, exist_ok=True)
ASSETS = {
    'BTCUSDT': ('2017-08', '2026-07'),
    'ETHUSDT': ('2017-08', '2026-07'),
    'LTCUSDT': ('2017-12', '2026-07'),
    'ADAUSDT': ('2018-04', '2026-07'),
    'XRPUSDT': ('2018-05', '2026-07'),
    'DOGEUSDT': ('2019-07', '2026-07'),
    'SOLUSDT': ('2020-08', '2026-07'),
}
TFS = {'15m':'15min','30m':'30min','1h':'1h','4h':'4h','7h':'7h','1D':'1D'}


def ym_iter(start, end):
    y,m=map(int,start.split('-')); ey,em=map(int,end.split('-'))
    while (y,m) <= (ey,em):
        yield f'{y:04d}-{m:02d}'
        m += 1
        if m==13: y,m=y+1,1


def download_month(symbol, ym):
    url=f'https://data.binance.vision/data/spot/monthly/klines/{symbol}/1m/{symbol}-1m-{ym}.zip'
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            b=r.read()
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            name=z.namelist()[0]
            raw=z.read(name)
        # Binance 1m columns; open_time can be ms or microseconds in newer archives.
        cols=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_base','taker_quote','ignore']
        df=pd.read_csv(io.BytesIO(raw), header=None, names=cols)
        t=pd.to_numeric(df.open_time, errors='coerce')
        unit='us' if t.dropna().median() > 1e14 else 'ms'
        idx=pd.to_datetime(t, unit=unit, utc=True, errors='coerce')
        out=pd.DataFrame({
            'open':pd.to_numeric(df.open,errors='coerce').values,
            'high':pd.to_numeric(df.high,errors='coerce').values,
            'low':pd.to_numeric(df.low,errors='coerce').values,
            'close':pd.to_numeric(df.close,errors='coerce').values,
            'volume':pd.to_numeric(df.volume,errors='coerce').values,
        }, index=idx)
        out=out[~out.index.isna()].dropna()
        return out
    except Exception as e:
        print('MISS', symbol, ym, type(e).__name__, str(e)[:120])
        return None


def load_asset_15m(symbol, start, end):
    parts=[]
    for ym in ym_iter(start,end):
        d=download_month(symbol,ym)
        if d is None or d.empty: continue
        r=d.resample('15min', label='left', closed='left').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum')).dropna()
        parts.append(r)
        print(symbol, ym, len(d), '->', len(r))
    if not parts: raise RuntimeError(f'No data {symbol}')
    x=pd.concat(parts).sort_index()
    x=x[~x.index.duplicated(keep='last')]
    x.to_csv(OUT/f'{symbol}_15m.csv.gz', compression='gzip')
    return x


def resample(x, rule):
    if rule=='15min': return x.copy()
    return x.resample(rule, label='left', closed='left').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum')).dropna()


def add_features(x):
    z=x.copy()
    z['ret']=z.close/z.open-1
    z['bull']=z['ret']>0
    z['bear']=z['ret']<0
    z['body_abs']=z['ret'].abs()
    z['body_med50']=z['body_abs'].rolling(50,min_periods=25).median()
    z['body_strong']=z['body_abs']>=z['body_med50']
    z['vol_ma20']=z.volume.rolling(20,min_periods=10).mean().shift(1)
    z['vol_rel20']=z.volume/z['vol_ma20']
    z['vol_high']=z['vol_rel20']>1
    # compact ADX implementation using OHLC, Wilder RMA
    n=14
    up=z.high.diff(); dn=-z.low.diff()
    plus=np.where((up>dn)&(up>0),up,0.0); minus=np.where((dn>up)&(dn>0),dn,0.0)
    prev=z.close.shift(1)
    tr=pd.concat([(z.high-z.low),(z.high-prev).abs(),(z.low-prev).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    p=pd.Series(plus,index=z.index).ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    m=pd.Series(minus,index=z.index).ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    pdi=100*p/atr; mdi=100*m/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi)
    z['adx']=dx.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    z['di_dir']=np.where(pdi>mdi,1,np.where(mdi>pdi,-1,0))
    z['adx_active']=z.adx>=25
    return z


def fwd(series,h): return series.shift(-h)/series-1


def summarize_event(df, mask, horizons, prefix, rows):
    n=int(mask.sum())
    if n==0: return
    base={'event':prefix,'n':n,'pct_bars':100*n/len(df)}
    for h in horizons:
        s=df.loc[mask,f'btc_fwd_{h}'].dropna()
        base[f'btc_mean_{h}']=100*s.mean() if len(s) else np.nan
        base[f'btc_median_{h}']=100*s.median() if len(s) else np.nan
        base[f'btc_pos_{h}']=100*(s>0).mean() if len(s) else np.nan
    rows.append(base)


def analyze_tf(raw, tf):
    data={s:add_features(resample(x,TFS[tf])) for s,x in raw.items()}
    common=None
    for x in data.values(): common=x.index if common is None else common.intersection(x.index)
    d=pd.DataFrame(index=common)
    btc=data['BTCUSDT'].loc[common]
    d['btc_ret']=btc.ret; d['btc_bear']=btc.bear; d['btc_bull']=btc.bull; d['btc_strong']=btc.body_strong; d['btc_vrel']=btc.vol_rel20; d['btc_vhigh']=btc.vol_high
    d['btc_adx']=btc.adx; d['btc_didir']=btc.di_dir; d['btc_reg_bear']=btc.adx_active & (btc.di_dir==-1); d['btc_reg_bull']=btc.adx_active & (btc.di_dir==1)
    alts=[s for s in data if s!='BTCUSDT']
    for s in alts:
        a=data[s].loc[common]; k=s.replace('USDT','').lower()
        d[f'{k}_ret']=a.ret; d[f'{k}_bull']=a.bull; d[f'{k}_bear']=a.bear; d[f'{k}_strong']=a.body_strong; d[f'{k}_vrel']=a.vol_rel20; d[f'{k}_vhigh']=a.vol_high
        d[f'{k}_reg_bull']=a.adx_active & (a.di_dir==1); d[f'{k}_reg_bear']=a.adx_active & (a.di_dir==-1)
    bullcols=[s.replace('USDT','').lower()+'_bull' for s in alts]
    bearcols=[s.replace('USDT','').lower()+'_bear' for s in alts]
    strongbull=[(d[s.replace('USDT','').lower()+'_bull'] & d[s.replace('USDT','').lower()+'_strong']) for s in alts]
    highvolbull=[(d[s.replace('USDT','').lower()+'_bull'] & d[s.replace('USDT','').lower()+'_strong'] & d[s.replace('USDT','').lower()+'_vhigh']) for s in alts]
    regbull=[d[s.replace('USDT','').lower()+'_reg_bull'] for s in alts]
    d['alt_bull_count']=d[bullcols].sum(axis=1)
    d['alt_bear_count']=d[bearcols].sum(axis=1)
    d['alt_strong_bull_count']=sum(strongbull)
    d['alt_highvol_strong_bull_count']=sum(highvolbull)
    d['alt_reg_bull_count']=sum(regbull)
    # breadth mean relative volume among available alts
    vcols=[s.replace('USDT','').lower()+'_vrel' for s in alts]
    d['alt_vrel_mean']=d[vcols].mean(axis=1)
    # forward BTC close returns
    hs={'15m':[1,2,4,8,16,24,48,96], '30m':[1,2,3,6,12,24,48], '1h':[1,2,3,6,12,24,48], '4h':[1,2,3,6,12], '7h':[1,2,3,6], '1D':[1,2,3,5,10]}[tf]
    for h in hs: d[f'btc_fwd_{h}']=fwd(btc.close.loc[common],h)
    rows=[]
    summarize_event(d,d.btc_bear,hs,'BTC red baseline',rows)
    for k in range(1,len(alts)+1):
        summarize_event(d,d.btc_bear & (d.alt_bull_count>=k),hs,f'BTC red + >= {k}/{len(alts)} alts green',rows)
    for k in [1,2,3,4,5,6]:
        if k<=len(alts):
            summarize_event(d,d.btc_bear & d.btc_strong & (d.alt_strong_bull_count>=k),hs,f'BTC red strong + >= {k}/{len(alts)} alts green strong',rows)
            summarize_event(d,d.btc_bear & d.btc_strong & d.btc_vhigh & (d.alt_highvol_strong_bull_count>=k),hs,f'BTC red strong+highVol + >= {k}/{len(alts)} alts green strong+highVol',rows)
    for k in range(1,len(alts)+1):
        summarize_event(d,d.btc_reg_bear & (d.alt_reg_bull_count>=k),hs,f'BTC ADX bear + >= {k}/{len(alts)} alts ADX bull',rows)
    # ADA-focused
    if 'ada_bull' in d:
        summarize_event(d,d.btc_bear & d.ada_bull,hs,'BTC red + ADA green',rows)
        summarize_event(d,d.btc_bear & d.btc_strong & d.ada_bull & d.ada_strong,hs,'BTC red strong + ADA green strong',rows)
        summarize_event(d,d.btc_bear & d.btc_strong & d.btc_vhigh & d.ada_bull & d.ada_strong & d.ada_vhigh,hs,'BTC red strong highVol + ADA green strong highVol',rows)
        summarize_event(d,d.btc_reg_bear & d.ada_reg_bull,hs,'BTC ADX bear + ADA ADX bull',rows)
    summary=pd.DataFrame(rows)
    summary.insert(0,'tf',tf)
    summary.to_csv(OUT/f'summary_{tf}.csv',index=False)
    # regime era split on key event
    key=d.btc_bear & d.btc_strong & d.btc_vhigh & (d.alt_highvol_strong_bull_count>=max(2,len(alts)//2))
    era=[]
    for name,a,b in [('2018-2020','2018-01-01','2020-12-31 23:59'),('2021-2022','2021-01-01','2022-12-31 23:59'),('2023-2024','2023-01-01','2024-12-31 23:59'),('2025-2026','2025-01-01','2026-12-31 23:59')]:
        m=key & (d.index>=pd.Timestamp(a,tz='UTC')) & (d.index<=pd.Timestamp(b,tz='UTC'))
        row={'tf':tf,'era':name,'n':int(m.sum())}
        for h in hs[:4]:
            s=d.loc[m,f'btc_fwd_{h}'].dropna(); row[f'mean_{h}']=100*s.mean() if len(s) else np.nan; row[f'pos_{h}']=100*(s>0).mean() if len(s) else np.nan
        era.append(row)
    pd.DataFrame(era).to_csv(OUT/f'era_{tf}.csv',index=False)
    # save compact event rows for manual audit
    ev=d.loc[key].copy(); ev.to_csv(OUT/f'events_key_{tf}.csv.gz',compression='gzip')
    return summary, pd.DataFrame(era)


def main():
    raw={}
    manifest=[]
    for s,(a,b) in ASSETS.items():
        x=load_asset_15m(s,a,b); raw[s]=x
        manifest.append({'symbol':s,'rows15m':len(x),'start':str(x.index.min()),'end':str(x.index.max()),'duplicates':int(x.index.duplicated().sum()),'nulls':int(x.isna().sum().sum())})
    pd.DataFrame(manifest).to_csv(OUT/'manifest.csv',index=False)
    allsum=[]; allera=[]
    for tf in TFS:
        s,e=analyze_tf(raw,tf); allsum.append(s); allera.append(e)
    S=pd.concat(allsum,ignore_index=True); E=pd.concat(allera,ignore_index=True)
    S.to_csv(OUT/'SUMMARY_ALL.csv',index=False); E.to_csv(OUT/'ERA_ALL.csv',index=False)
    # concise markdown
    with open(OUT/'REPORT.md','w',encoding='utf-8') as f:
        f.write('# BTC vs ALT Divergence — First Full Pass\n\n')
        f.write('Universe: BTCUSDT vs ETH, SOL, ADA, XRP, DOGE, LTC; Binance Spot; causal candle-close analysis.\n\n')
        f.write('## Data manifest\n\n'+pd.DataFrame(manifest).to_markdown(index=False)+'\n\n')
        for tf in TFS:
            q=S[(S.tf==tf) & S.event.str.contains('BTC red strong\+highVol')].copy()
            f.write(f'## {tf} — high-volume strong divergence breadth\n\n')
            if len(q): f.write(q.to_markdown(index=False)+'\n\n')
        f.write('## ADA focused\n\n')
        q=S[S.event.str.contains('ADA')]
        f.write(q.to_markdown(index=False)+'\n')

if __name__=='__main__': main()
