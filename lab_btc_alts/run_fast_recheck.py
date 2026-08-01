import json, time, urllib.parse, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path('lab_btc_alts/fast_recheck_output')
OUT.mkdir(parents=True, exist_ok=True)
END_MS = int(pd.Timestamp('2026-07-11T00:00:00Z').timestamp()*1000)
STARTS = {
    'BTCUSDT':'2017-08-17','ETHUSDT':'2017-08-17','LTCUSDT':'2017-12-13',
    'ADAUSDT':'2018-04-17','XRPUSDT':'2018-05-04','DOGEUSDT':'2019-07-05','SOLUSDT':'2020-08-11'
}
ALTS = ['ETHUSDT','SOLUSDT','ADAUSDT','XRPUSDT','DOGEUSDT','LTCUSDT']
CONFIG = {
    '1h': {'start':'2020-08-11','h':[1,3,6,12,24]},
    '4h': {'start':'2018-04-17','h':[1,3,6,12]},
    '1d': {'start':'2018-04-17','h':[1,3,5,10]},
}


def fetch_klines(symbol, interval, start_date):
    start_ms = int(pd.Timestamp(start_date, tz='UTC').timestamp()*1000)
    rows=[]; cursor=start_ms; req=0
    while cursor < END_MS:
        q=urllib.parse.urlencode({'symbol':symbol,'interval':interval,'startTime':cursor,'endTime':END_MS-1,'limit':1000})
        url='https://api.binance.com/api/v3/klines?'+q
        for attempt in range(6):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data=json.loads(r.read().decode())
                break
            except Exception as e:
                if attempt==5: raise
                time.sleep(1.5*(attempt+1))
        req += 1
        if not data: break
        rows.extend(data)
        nxt=int(data[-1][0])+1
        if nxt<=cursor: break
        cursor=nxt
        if len(data)<1000: break
        time.sleep(0.03)
    cols=['t','open','high','low','close','volume','ct','qv','n','tb','tq','x']
    d=pd.DataFrame(rows,columns=cols)
    if d.empty: return d
    d=d.drop_duplicates('t').sort_values('t')
    d.index=pd.to_datetime(d.t.astype('int64'),unit='ms',utc=True)
    for c in ['open','high','low','close','volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d[['open','high','low','close','volume']].dropna()
    print(interval,symbol,'rows',len(d),'req',req,d.index.min(),d.index.max(), flush=True)
    return d


def features(d):
    z=d.copy()
    z['ret']=z.close/z.open-1
    z['red']=z.ret<0; z['green']=z.ret>0
    z['absbody']=z.ret.abs()
    z['body_ref50']=z.absbody.rolling(50,min_periods=50).median().shift(1)
    z['strong']=z.absbody>=z.body_ref50
    z['vol_ref20']=z.volume.rolling(20,min_periods=20).mean().shift(1)
    z['vrel']=z.volume/z.vol_ref20
    z['highvol']=z.vrel>1
    return z


def future_return(s,h): return s.shift(-h)/s-1


def stats_for_mask(frame,mask,horizons,prefix='btc'):
    out={'n':int(mask.sum()),'pct_bars':100*mask.mean()}
    for h in horizons:
        x=frame.loc[mask,f'{prefix}_f{h}'].dropna()
        out[f'mean_{h}']=100*x.mean() if len(x) else np.nan
        out[f'median_{h}']=100*x.median() if len(x) else np.nan
        out[f'pos_{h}']=100*(x>0).mean() if len(x) else np.nan
    return out


def pairwise(b,a,tf,alt,horizons):
    idx=b.index.intersection(a.index)
    b=b.loc[idx]; a=a.loc[idx]
    d=pd.DataFrame(index=idx)
    for name,src in [('btc',b),('alt',a)]:
        d[name+'_ret']=src.ret; d[name+'_red']=src.red; d[name+'_green']=src.green
        d[name+'_strong']=src.strong; d[name+'_highvol']=src.highvol; d[name+'_vrel']=src.vrel
        for h in horizons: d[f'{name}_f{h}']=future_return(src.close,h)
    masks={
      'BTC red baseline': d.btc_red,
      'BTC red + ALT green': d.btc_red & d.alt_green,
      'strong opposite candles': d.btc_red & d.btc_strong & d.alt_green & d.alt_strong,
      'strong + both high volume': d.btc_red & d.btc_strong & d.btc_highvol & d.alt_green & d.alt_strong & d.alt_highvol,
      'strong + ALT volRel > BTC volRel': d.btc_red & d.btc_strong & d.alt_green & d.alt_strong & (d.alt_vrel>d.btc_vrel),
    }
    base=stats_for_mask(d,masks['BTC red baseline'],horizons)
    rows=[]
    for ev,m in masks.items():
        r={'tf':tf,'alt':alt.replace('USDT',''),'event':ev,'start':str(idx.min()),'end':str(idx.max()),**stats_for_mask(d,m,horizons)}
        for h in horizons:
            aa=d.loc[m,f'alt_f{h}'].dropna()
            joint=d.loc[m,[f'btc_f{h}',f'alt_f{h}']].dropna()
            r[f'alt_mean_{h}']=100*aa.mean() if len(aa) else np.nan
            r[f'alt_pos_{h}']=100*(aa>0).mean() if len(aa) else np.nan
            r[f'btc_outperforms_alt_{h}']=100*(joint[f'btc_f{h}']>joint[f'alt_f{h}']).mean() if len(joint) else np.nan
            r[f'pos_lift_vs_btc_red_{h}']=r[f'pos_{h}']-base[f'pos_{h}'] if np.isfinite(r[f'pos_{h}']) else np.nan
            r[f'mean_lift_vs_btc_red_{h}']=r[f'mean_{h}']-base[f'mean_{h}'] if np.isfinite(r[f'mean_{h}']) else np.nan
        rows.append(r)
    return rows


def breadth(data,tf,horizons):
    common=data['BTCUSDT'].index
    for s in ALTS: common=common.intersection(data[s].index)
    b=data['BTCUSDT'].loc[common]
    d=pd.DataFrame(index=common)
    d['btc_red']=b.red; d['btc_strong']=b.strong; d['btc_highvol']=b.highvol
    for h in horizons: d[f'btc_f{h}']=future_return(b.close,h)
    green=[]; strong=[]; hv=[]
    for s in ALTS:
        a=data[s].loc[common]
        green.append(a.green.astype(int))
        strong.append((a.green&a.strong).astype(int))
        hv.append((a.green&a.strong&a.highvol).astype(int))
    d['green_count']=sum(green); d['strong_green_count']=sum(strong); d['hv_strong_green_count']=sum(hv)
    base=stats_for_mask(d,d.btc_red,horizons)
    rows=[]
    def add(ev,m):
        r={'tf':tf,'event':ev,'start':str(common.min()),'end':str(common.max()),**stats_for_mask(d,m,horizons)}
        for h in horizons:
            r[f'pos_lift_vs_btc_red_{h}']=r[f'pos_{h}']-base[f'pos_{h}'] if np.isfinite(r[f'pos_{h}']) else np.nan
            r[f'mean_lift_vs_btc_red_{h}']=r[f'mean_{h}']-base[f'mean_{h}'] if np.isfinite(r[f'mean_{h}']) else np.nan
        rows.append(r)
    add('BTC red baseline',d.btc_red)
    for k in range(1,7):
        add(f'BTC red + >= {k}/6 alts green',d.btc_red&(d.green_count>=k))
        add(f'BTC red strong + >= {k}/6 alts green strong',d.btc_red&d.btc_strong&(d.strong_green_count>=k))
        add(f'BTC red strong highVol + >= {k}/6 alts green strong highVol',d.btc_red&d.btc_strong&d.btc_highvol&(d.hv_strong_green_count>=k))
    return rows,d


def era_table(d,tf,horizons):
    rows=[]
    keys={
      '>=3/6 strong highVol': d.btc_red&d.btc_strong&d.btc_highvol&(d.hv_strong_green_count>=3),
      '>=4/6 green': d.btc_red&(d.green_count>=4),
      '6/6 green': d.btc_red&(d.green_count==6),
    }
    for key,m0 in keys.items():
      for era,a,b in [('2020-2022','2020-08-11','2022-12-31 23:59'),('2023-2024','2023-01-01','2024-12-31 23:59'),('2025-2026','2025-01-01','2026-07-10 23:59')]:
        m=m0&(d.index>=pd.Timestamp(a,tz='UTC'))&(d.index<=pd.Timestamp(b,tz='UTC'))
        r={'tf':tf,'key':key,'era':era,**stats_for_mask(d,m,horizons)}
        rows.append(r)
    return rows


def main():
    pairs=[]; br=[]; eras=[]; manifest=[]
    for tf,cfg in CONFIG.items():
        raw={}
        for s in ['BTCUSDT']+ALTS:
            start=max(pd.Timestamp(cfg['start']),pd.Timestamp(STARTS[s])).strftime('%Y-%m-%d')
            raw[s]=features(fetch_klines(s,tf,start))
            manifest.append({'tf':tf,'symbol':s,'rows':len(raw[s]),'start':str(raw[s].index.min()),'end':str(raw[s].index.max())})
        for a in ALTS: pairs += pairwise(raw['BTCUSDT'],raw[a],tf,a,cfg['h'])
        rr,d=breadth(raw,tf,cfg['h']); br += rr
        if tf=='4h': eras += era_table(d,tf,cfg['h'])
    P=pd.DataFrame(pairs); B=pd.DataFrame(br); E=pd.DataFrame(eras); M=pd.DataFrame(manifest)
    P.to_csv(OUT/'PAIRWISE.csv',index=False); B.to_csv(OUT/'BREADTH.csv',index=False); E.to_csv(OUT/'ERA_4H.csv',index=False); M.to_csv(OUT/'MANIFEST.csv',index=False)
    with open(OUT/'REPORT.md','w',encoding='utf-8') as f:
        f.write('# FAST RECHECK — BTC vs Alts\n\nAll assets: Binance Spot. Event study only. Volume reference = mean of previous 20 completed candles. Strong candle = absolute body >= median of previous 50 completed candle bodies.\n\n')
        f.write('## Manifest\n\n'+M.to_markdown(index=False)+'\n\n')
        f.write('## ADA\n\n'+P[(P.alt=='ADA')].to_markdown(index=False)+'\n\n')
        f.write('## 4H breadth\n\n'+B[(B.tf=='4h')].to_markdown(index=False)+'\n\n')
        f.write('## 4H robustness by era\n\n'+E.to_markdown(index=False)+'\n')
    print('FAST_RECHECK_DONE',OUT,flush=True)

if __name__=='__main__': main()
