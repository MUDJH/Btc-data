import io, zipfile, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('lab_btc_alts'); CACHE=ROOT/'cache15m'; OUT=ROOT/'output_v2'
CACHE.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
ASSETS={'BTCUSDT':('2017-08','2026-07'),'ETHUSDT':('2017-08','2026-07'),'LTCUSDT':('2017-12','2026-07'),'ADAUSDT':('2018-04','2026-07'),'XRPUSDT':('2018-05','2026-07'),'DOGEUSDT':('2019-07','2026-07'),'SOLUSDT':('2020-08','2026-07')}
TFS={'15m':'15min','30m':'30min','1h':'1h','4h':'4h','7h':'7h','1D':'1D'}
H={'15m':[1,2,4,8,16,24,48,96],'30m':[1,2,3,6,12,24,48],'1h':[1,2,3,6,12,24,48],'4h':[1,2,3,6,12],'7h':[1,2,3,6],'1D':[1,2,3,5,10]}


def months(a,b):
 y,m=map(int,a.split('-')); ey,em=map(int,b.split('-'))
 while (y,m)<=(ey,em):
  yield f'{y:04d}-{m:02d}'; m+=1
  if m==13:y,m=y+1,1


def get_month(sym,ym):
 url=f'https://data.binance.vision/data/spot/monthly/klines/{sym}/1m/{sym}-1m-{ym}.zip'
 try:
  with urllib.request.urlopen(url,timeout=90) as r:b=r.read()
  with zipfile.ZipFile(io.BytesIO(b)) as z:raw=z.read(z.namelist()[0])
  c=['t','o','h','l','c','v','ct','qv','n','tb','tq','x']; q=pd.read_csv(io.BytesIO(raw),header=None,names=c)
  t=pd.to_numeric(q.t,errors='coerce'); unit='us' if t.dropna().median()>1e14 else 'ms'; idx=pd.to_datetime(t,unit=unit,utc=True,errors='coerce')
  x=pd.DataFrame({'open':pd.to_numeric(q.o,errors='coerce').values,'high':pd.to_numeric(q.h,errors='coerce').values,'low':pd.to_numeric(q.l,errors='coerce').values,'close':pd.to_numeric(q.c,errors='coerce').values,'volume':pd.to_numeric(q.v,errors='coerce').values},index=idx).dropna()
  return x[~x.index.isna()]
 except Exception as e:
  print('MISS',sym,ym,type(e).__name__,str(e)[:100]); return None


def load15(sym,a,b):
 f=CACHE/f'{sym}_15m.csv.gz'
 if f.exists():
  x=pd.read_csv(f,index_col=0,parse_dates=True); x.index=pd.DatetimeIndex(x.index)
  if x.index.tz is None:x.index=x.index.tz_localize('UTC')
  return x
 parts=[]
 for ym in months(a,b):
  x=get_month(sym,ym)
  if x is None or x.empty:continue
  r=x.resample('15min',label='left',closed='left').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum')).dropna()
  parts.append(r); print(sym,ym,len(x),'->',len(r))
 if not parts:raise RuntimeError('NO DATA '+sym)
 x=pd.concat(parts).sort_index(); x=x[~x.index.duplicated(keep='last')]; x.to_csv(f,compression='gzip'); return x


def rs(x,rule):
 if rule=='15min':return x.copy()
 return x.resample(rule,label='left',closed='left').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum')).dropna()


def rma(s,n):return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def adx_components(x,n=14,sm=14):
 up=x.high.diff(); dn=-x.low.diff(); plus=pd.Series(np.where((up>dn)&(up>0),up,0.),index=x.index); minus=pd.Series(np.where((dn>up)&(dn>0),dn,0.),index=x.index)
 pc=x.close.shift(); tr=pd.concat([x.high-x.low,(x.high-pc).abs(),(x.low-pc).abs()],axis=1).max(axis=1); rr=rma(tr,n); pdi=100*rma(plus,n)/rr; mdi=100*rma(minus,n)/rr; dx=100*(pdi-mdi).abs()/(pdi+mdi); return rma(dx,sm),pdi,mdi


def dominant_mode(x):
 # Exact default logic of current Pine: main 14/14, mini 5/5, rearm <=21, escape >=25+rising,
 # direction close vs ADX-run start, mini must be rising; first opposite escape neutralizes; close breaks SL.
 adx,pdi,mdi=adx_components(x,14,14); mini,_,_=adx_components(x,5,5)
 n=len(x); mode=np.zeros(n,dtype=np.int8); ons=np.zeros(n,dtype=np.int8); cont=np.zeros(n,dtype=np.int8); neu=np.zeros(n,dtype=np.int8); slneu=np.zeros(n,dtype=np.int8)
 armed=False; run=False; rlow=rhigh=rstart=np.nan; md=0; pending=0; msl=np.nan; msldir=0
 for i in range(n):
  a=adx.iat[i]; ap=adx.iat[i-1] if i else np.nan; mn=mini.iat[i]; mnp=mini.iat[i-1] if i else np.nan; cl=x.close.iat[i]; lo=x.low.iat[i]; hi=x.high.iat[i]
  if np.isfinite(a) and a<=21:armed=True
  step=a-ap if np.isfinite(a) and np.isfinite(ap) else np.nan
  if armed and np.isfinite(step):
   if not run and step>0:run=True; rlow=lo; rhigh=hi; rstart=cl
   elif run:
    rlow=min(rlow,lo); rhigh=max(rhigh,hi)
    if a<=21 and step<0:run=False; rlow=rhigh=rstart=np.nan
  escape=armed and np.isfinite(a) and a>=25 and np.isfinite(ap) and a>ap
  miniok=np.isfinite(mn) and np.isfinite(mnp) and (mn-mnp)>0
  idir=1 if run and cl>rstart else (-1 if run and cl<rstart else 0)
  valid=escape and miniok and idir!=0 and run
  bullon=bearon=bullc=bearc=False
  if valid:
   if md==0:
    # neutral pending does not require repeat under default; next valid escape selects its own direction
    md=idir; pending=0; bullon=idir==1; bearon=idir==-1
   elif idir==md:
    pending=0; bullc=idir==1; bearc=idir==-1
   else:
    neu[i]=1 if md==1 else -1; md=0; pending=idir; msl=np.nan; msldir=0
   if bullon or bearon or bullc or bearc:
    if idir==1:msl=rlow; msldir=1
    else:msl=rhigh; msldir=-1
   armed=False; run=False; rlow=rhigh=rstart=np.nan
   ons[i]=1 if bullon else (-1 if bearon else 0); cont[i]=1 if bullc else (-1 if bearc else 0)
  # SL check occurs after signal engine, using close under defaults
  if md==1 and msldir==1 and np.isfinite(msl) and cl<msl:slneu[i]=1; md=0; pending=0; msl=np.nan; msldir=0
  elif md==-1 and msldir==-1 and np.isfinite(msl) and cl>msl:slneu[i]=-1; md=0; pending=0; msl=np.nan; msldir=0
  mode[i]=md
 return pd.DataFrame({'dom_mode':mode,'dom_on':ons,'dom_cont':cont,'dom_neu':neu,'dom_slneu':slneu,'adx':adx,'pdi':pdi,'mdi':mdi,'mini_adx':mini},index=x.index)


def feat(x):
 z=x.copy(); z['ret']=z.close/z.open-1; z['bull']=z.ret>0; z['bear']=z.ret<0; z['body_abs']=z.ret.abs(); z['body_med50']=z.body_abs.rolling(50,min_periods=25).median(); z['strong']=z.body_abs>=z.body_med50
 z['vol_ma20_prev']=z.volume.rolling(20,min_periods=10).mean().shift(1); z['vrel']=z.volume/z.vol_ma20_prev; z['vhigh']=z.vrel>1
 dm=dominant_mode(z); return z.join(dm)

def fwd(s,h):return s.shift(-h)/s-1


def pair_rows(b,a,tf,alt):
 idx=b.index.intersection(a.index); b=b.loc[idx]; a=a.loc[idx]; hs=H[tf]; d=pd.DataFrame(index=idx)
 d['br']=b.ret; d['ar']=a.ret; d['bb']=b.bear; d['ab']=a.bull; d['bs']=b.strong; d['as']=a.strong; d['bv']=b.vhigh; d['av']=a.vhigh; d['bvr']=b.vrel; d['avr']=a.vrel; d['bd']=b.dom_mode; d['ad']=a.dom_mode
 for h in hs:d[f'bf{h}']=fwd(b.close,h); d[f'af{h}']=fwd(a.close,h)
 events=[('BTC red + ALT green',d.bb&d.ab),('strong opposite candles',d.bb&d.bs&d.ab&d.as),('strong + both high volume',d.bb&d.bs&d.bv&d.ab&d.as&d.av),('strong + ALT volRel > BTC volRel',d.bb&d.bs&d.ab&d.as&(d.avr>d.bvr)),('BTC Dominant BEAR + ALT Dominant BULL',(d.bd==-1)&(d.ad==1)),('price+Dominant disagreement',d.bb&d.ab&(d.bd==-1)&(d.ad==1))]
 rows=[]
 for name,m in events:
  n=int(m.sum()); r={'tf':tf,'alt':alt,'event':name,'n':n,'pct_bars':100*n/len(d),'start':str(idx.min()),'end':str(idx.max())}
  for h in hs:
   bs=d.loc[m,f'bf{h}'].dropna(); aa=d.loc[m,f'af{h}'].dropna(); joint=d.loc[m,[f'bf{h}',f'af{h}']].dropna()
   r[f'btc_mean_{h}']=100*bs.mean() if len(bs) else np.nan; r[f'btc_pos_{h}']=100*(bs>0).mean() if len(bs) else np.nan; r[f'alt_mean_{h}']=100*aa.mean() if len(aa) else np.nan; r[f'alt_pos_{h}']=100*(aa>0).mean() if len(aa) else np.nan; r[f'btc_outperforms_{h}']=100*(joint[f'bf{h}']>joint[f'af{h}']).mean() if len(joint) else np.nan
  rows.append(r)
 return rows


def breadth_rows(data,tf):
 common=None
 for x in data.values():common=x.index if common is None else common.intersection(x.index)
 b=data['BTCUSDT'].loc[common]; alts=[s for s in data if s!='BTCUSDT']; d=pd.DataFrame(index=common); d['bb']=b.bear; d['bs']=b.strong; d['bv']=b.vhigh; d['bd']=b.dom_mode
 bull=[]; strong=[]; hv=[]; dom=[]
 for s in alts:
  a=data[s].loc[common]; bull.append(a.bull); strong.append(a.bull&a.strong); hv.append(a.bull&a.strong&a.vhigh); dom.append(a.dom_mode==1)
 d['bull']=sum(bull); d['strong']=sum(strong); d['hv']=sum(hv); d['dom']=sum(dom)
 for h in H[tf]:d[f'bf{h}']=fwd(b.close,h)
 rows=[]
 def add(name,m):
  n=int(m.sum()); r={'tf':tf,'event':name,'n':n,'pct_bars':100*n/len(d),'start':str(common.min()),'end':str(common.max())}
  for h in H[tf]:q=d.loc[m,f'bf{h}'].dropna(); r[f'btc_mean_{h}']=100*q.mean() if len(q) else np.nan; r[f'btc_pos_{h}']=100*(q>0).mean() if len(q) else np.nan
  rows.append(r)
 add('BTC red baseline',d.bb)
 for k in range(1,7):add(f'BTC red + >= {k}/6 alts green',d.bb&(d.bull>=k)); add(f'BTC red strong + >= {k}/6 alts green strong',d.bb&d.bs&(d.strong>=k)); add(f'BTC red strong highVol + >= {k}/6 alts green strong highVol',d.bb&d.bs&d.bv&(d.hv>=k)); add(f'BTC Dominant BEAR + >= {k}/6 alts Dominant BULL',(d.bd==-1)&(d.dom>=k))
 return rows,d


def main():
 raw={}; man=[]
 for s,(a,b) in ASSETS.items():
  x=load15(s,a,b); raw[s]=x; man.append({'symbol':s,'rows15m':len(x),'start':str(x.index.min()),'end':str(x.index.max()),'duplicates':int(x.index.duplicated().sum()),'nulls':int(x.isna().sum().sum())})
 pd.DataFrame(man).to_csv(OUT/'manifest.csv',index=False)
 pairs=[]; breadth=[]; eras=[]
 for tf,rule in TFS.items():
  data={s:feat(rs(x,rule)) for s,x in raw.items()}
  for alt in [s for s in data if s!='BTCUSDT']:pairs += pair_rows(data['BTCUSDT'],data[alt],tf,alt.replace('USDT',''))
  rr,d=breadth_rows(data,tf); breadth += rr
  key=d.bb&d.bs&d.bv&(d.hv>=3)
  for nm,a,b in [('2020-2022','2020-01-01','2022-12-31 23:59'),('2023-2024','2023-01-01','2024-12-31 23:59'),('2025-2026','2025-01-01','2026-12-31 23:59')]:
   m=key&(d.index>=pd.Timestamp(a,tz='UTC'))&(d.index<=pd.Timestamp(b,tz='UTC')); r={'tf':tf,'era':nm,'event':'BTC red strong highVol + >=3/6 alts green strong highVol','n':int(m.sum())}
   for h in H[tf][:4]:q=d.loc[m,f'bf{h}'].dropna(); r[f'mean_{h}']=100*q.mean() if len(q) else np.nan; r[f'pos_{h}']=100*(q>0).mean() if len(q) else np.nan
   eras.append(r)
 P=pd.DataFrame(pairs); B=pd.DataFrame(breadth); E=pd.DataFrame(eras); P.to_csv(OUT/'PAIRWISE_ALL.csv',index=False); B.to_csv(OUT/'BREADTH_ALL.csv',index=False); E.to_csv(OUT/'ERA_ALL.csv',index=False)
 with open(OUT/'REPORT.md','w',encoding='utf-8') as f:
  f.write('# BTC vs Alts — Full Historical Pass V2\n\n')
  f.write('Binance Spot for all seven assets. Candle-close descriptive/event study; no trading execution model. Relative volume uses current volume divided by the mean of the previous 20 completed candles (causal). Strong candle = absolute open-to-close return >= rolling median of previous/current 50 absolute bodies. Dominant regime reproduces the default stateful logic of the supplied ADX Dominante indicator.\n\n')
  f.write('## Manifest\n\n'+pd.DataFrame(man).to_markdown(index=False)+'\n\n')
  f.write('## ADA pairwise (full ADA/BTC overlap)\n\n'+P[P.alt.eq('ADA')].to_markdown(index=False)+'\n\n')
  f.write('## Breadth strong/high-volume\n\n'+B[B.event.str.contains('strong highVol')].to_markdown(index=False)+'\n\n')
  f.write('## Breadth Dominant disagreement\n\n'+B[B.event.str.contains('Dominant')].to_markdown(index=False)+'\n\n')
  f.write('## Era robustness\n\n'+E.to_markdown(index=False)+'\n')
 print('DONE',OUT)

if __name__=='__main__':main()
