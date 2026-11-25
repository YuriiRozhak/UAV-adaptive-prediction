"""Reproducible offline research POC: adaptive prediction of synthetic vehicle trajectories.
No flight-controller interfaces or operational control outputs.
"""
import argparse
import heapq
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

FS = 20
HORIZONS = (5, 10, 20)
FEATURES = ['x','y','z','vx','vy','vz','ax','ay','az','gx','gy','gz']
OUTPUTS = ['x','y','z','vx','vy','vz']


def synth_flight(seed, seconds=28, noise=1.0):
    rng = np.random.default_rng(seed)
    n = seconds * FS
    dt = 1/FS
    p = np.zeros((n,3)); v = np.zeros((n,3)); a = np.zeros((n,3)); gyro = np.zeros((n,3))
    p[0] = rng.uniform([-30,-30,4],[30,30,14]); v[0] = rng.uniform([-3,-3,-.5],[3,3,.5])
    periods = max(5, seconds // 4)
    ax_src = rng.uniform(-.55,.55,size=(periods,3))
    ax_src[:,2] *= .4
    # Each trajectory has its own smooth acceleration pattern.
    anchors = np.linspace(0,n-1,periods)
    acceleration = np.stack([np.interp(np.arange(n),anchors,ax_src[:,k]) for k in range(3)],axis=-1)
    acceleration += .12*np.sin(np.arange(n)[:,None]*dt*rng.uniform(.3,1.4,size=(1,3)) + rng.uniform(0,3,size=(1,3)))
    for i in range(1,n):
        a[i] = acceleration[i]
        v[i] = .9995*v[i-1] + a[i]*dt
        p[i] = p[i-1] + v[i-1]*dt + .5*a[i]*dt*dt
    # Yaw rate loosely represents turning of velocity vector; world-frame acceleration is already used.
    heading = np.unwrap(np.arctan2(v[:,1],v[:,0])); gyro[:,2] = np.gradient(heading,dt)
    gyro[:,0] = .015*np.sin(np.arange(n)*dt*.7); gyro[:,1] = .02*np.cos(np.arange(n)*dt*.5)
    truth = np.concatenate([p,v],axis=1)
    obs = np.concatenate([p,v,a,gyro],axis=1)
    noise_vec = np.array([.08]*3+[.05]*3+[.11]*3+[.025]*3)*noise
    obs += rng.normal(size=obs.shape)*noise_vec
    return {'seed':seed,'obs':obs.astype('float32'),'truth':truth.astype('float32')}


def make_samples(flights, window, horizon, xscaler=None, yscaler=None):
    xs=[]; ys=[]; metas=[]
    for flight in flights:
        obs=flight['obs']; truth=flight['truth']
        for i in range(window-1, len(obs)-horizon):
            xs.append(obs[i-window+1:i+1]);
            # Relative future displacement and velocity change to eliminate absolute-location shortcut.
            ys.append(truth[i+horizon] - obs[i,:6]);
            metas.append((flight['seed'], i, i+horizon))
    X=np.asarray(xs,dtype='float32'); y=np.asarray(ys,dtype='float32')
    if xscaler is not None:
        X=xscaler.transform(X.reshape(-1,12)).reshape(X.shape).astype('float32')
    if yscaler is not None:
        y=yscaler.transform(y).astype('float32')
    return X,y,metas


class Net(nn.Module):
    def __init__(self, hidden=48, layers=2):
        super().__init__()
        self.lstm=nn.LSTM(12,hidden,layers,batch_first=True,dropout=.15 if layers>1 else 0)
        self.head=nn.Linear(hidden,6)
    def forward(self,x):
        o,_=self.lstm(x)
        return self.head(o[:,-1])


def train_model(train_flights,val_flights, horizon, cfg, out):
    window=cfg.window
    # Fit only on train flights. Relative input coordinates to remove position origin variation.
    X0,y0,_=make_samples(train_flights,window,horizon)
    def relative_positions(X):
        X=X.copy()
        X[:,:,:3] -= X[:,-1:,:3]
        return X
    X0=relative_positions(X0)
    xsc=StandardScaler().fit(X0.reshape(-1,12))
    ysc=StandardScaler().fit(y0)
    Xtr=xsc.transform(X0.reshape(-1,12)).reshape(X0.shape).astype('float32')
    ytr=ysc.transform(y0).astype('float32')
    Xv,yv,_=make_samples(val_flights,window,horizon)
    Xv=relative_positions(Xv)
    Xv=xsc.transform(Xv.reshape(-1,12)).reshape(Xv.shape).astype('float32')
    yv=ysc.transform(yv).astype('float32')
    train_loader=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr)),batch_size=cfg.batch,shuffle=True)
    val_loader=DataLoader(TensorDataset(torch.from_numpy(Xv),torch.from_numpy(yv)),batch_size=cfg.batch)
    net=Net(cfg.hidden).to('cpu')
    optim=torch.optim.Adam(net.parameters(),lr=cfg.lr)
    best=float('inf'); best_weights=None; patience=0; log=[]
    for epoch in range(cfg.epochs):
        net.train(); losses=[]
        for X,y in train_loader:
            optim.zero_grad(); pred=net(X); loss=nn.functional.mse_loss(pred,y); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.)
            optim.step(); losses.append(loss.item())
        net.eval(); vl=[]
        with torch.no_grad():
            for X,y in val_loader:
                vl.append(nn.functional.mse_loss(net(X),y).item())
        avg=float(np.mean(vl)); log.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'val_loss':avg})
        if avg < best-1e-5:
            best=avg; best_weights={k:v.detach().clone() for k,v in net.state_dict().items()}; patience=0
        else:
            patience+=1
        if patience>=cfg.patience: break
    net.load_state_dict(best_weights)
    torch.save({'state':net.state_dict(),'x_mean':xsc.mean_,'x_scale':xsc.scale_,
                'y_mean':ysc.mean_,'y_scale':ysc.scale_, 'horizon_steps':horizon},out/f'net_{horizon}.pt')
    return {'net':net,'xsc':xsc,'ysc':ysc,'history':log}


def predict(model, obs, idx, window):
    xx=obs[idx-window+1:idx+1].copy()
    xx[:,:3]-=xx[-1,:3]
    xx=model['xsc'].transform(xx)
    with torch.inference_mode():
        delta=model['net'](torch.from_numpy(xx[None].astype('float32'))).numpy()
    return model['ysc'].inverse_transform(delta)[0]+obs[idx,:6]


def baseline(obs,idx,h):
    dt=h/FS
    curr=obs[idx,:6]; p=curr[:3]; v=curr[3:6]; a=obs[idx,6:9]
    return {'CP':np.r_[p,v], 'CV':np.r_[p+dt*v,v],
            'CA':np.r_[p+dt*v+.5*dt*dt*a,v+dt*a]}


def errors(truth,pred):
    delta=np.asarray(pred)-np.asarray(truth)
    p=np.linalg.norm(delta[:,:3],axis=1)
    v=np.linalg.norm(delta[:,3:6],axis=1)
    return {'pos_rmse_m':float(np.sqrt(np.mean(p**2))),
            'pos_mae_m':float(p.mean()),'pos_p95_m':float(np.quantile(p,.95)),
            'vel_rmse_mps':float(np.sqrt(np.mean(v**2))),'samples':len(p)}


def assess_fixed(models,flights,window):
    rows=[]
    for h in HORIZONS:
        targets=[]; estimates={'LSTM':[],'CP':[],'CV':[],'CA':[]}
        for flight in flights:
            obs,truth=flight['obs'],flight['truth']
            for idx in range(window-1,len(obs)-h):
                targets.append(truth[idx+h]); estimates['LSTM'].append(predict(models[h],obs,idx,window))
                for key,v in baseline(obs,idx,h).items(): estimates[key].append(v)
        targets=np.asarray(targets)
        for key,preds in estimates.items():
            rows.append({'horizon_s':h/FS,'method':key,**errors(targets,np.asarray(preds))})
    return pd.DataFrame(rows)


def simulate_delayed_adaptation(models,flights,window,low,high,alpha=.2):
    """Causal scheduler: only target observations whose timestamps are <= current time influence horizon."""
    all_rows=[]; event_id=0
    for flight in flights:
        obs,truth=flight['obs'],flight['truth']
        pending=[]; ema=None; chosen=10
        for idx in range(window-1,len(obs)-max(HORIZONS)):
            # Mature all previous forecasts only at their target observation time.
            while pending and pending[0][0] <= idx:
                target_idx,_,saved_pred=heapq.heappop(pending)
                err=float(np.linalg.norm(saved_pred[:3]-truth[target_idx,:3]))
                ema=err if ema is None else alpha*err+(1-alpha)*ema
                chosen=20 if ema<low else (10 if ema<high else 5)
            h=chosen
            pred=predict(models[h],obs,idx,window)
            event_id+=1
            heapq.heappush(pending,(idx+h,event_id,pred))
            actual=truth[idx+h]
            all_rows.append({'flight':flight['seed'],'origin_idx':idx,'target_idx':idx+h,
                             'horizon_s':h/FS,'pos_error_m':float(np.linalg.norm(pred[:3]-actual[:3])),
                             'ema_at_decision':ema if ema is not None else np.nan,
                             'pred_x':float(pred[0]),'true_x':float(actual[0])})
    return pd.DataFrame(all_rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=8)
    ap.add_argument('--train-flights',type=int,default=12); ap.add_argument('--val-flights',type=int,default=4)
    ap.add_argument('--test-flights',type=int,default=4); ap.add_argument('--seconds',type=int,default=28)
    ap.add_argument('--window',type=int,default=40); ap.add_argument('--batch',type=int,default=128)
    ap.add_argument('--hidden',type=int,default=48); ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--patience',type=int,default=4); ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--output',default='outputs'); cfg=ap.parse_args()
    assert cfg.seconds*FS>cfg.window+max(HORIZONS)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    torch.set_num_threads(2)
    output=Path(cfg.output);output.mkdir(exist_ok=True,parents=True)
    all_flights=[synth_flight(cfg.seed*100+i,seconds=cfg.seconds) for i in range(cfg.train_flights+cfg.val_flights+cfg.test_flights)]
    tr=all_flights[:cfg.train_flights];va=all_flights[cfg.train_flights:cfg.train_flights+cfg.val_flights];te=all_flights[cfg.train_flights+cfg.val_flights:]
    print('Datasets (independent flights):',len(tr),len(va),len(te),flush=True)
    models={};start=time.perf_counter()
    for h in HORIZONS:
        models[h]=train_model(tr,va,h,cfg,output)
        print('Horizon',h/FS,'best validation loss',round(min(x['val_loss'] for x in models[h]['history']),5),flush=True)
    fixed=assess_fixed(models,te,cfg.window)
    fixed.to_csv(output/'fixed_horizon_metrics.csv',index=False)
    # Threshold selection on validation flights ONLY; this is exploratory and needs stronger validation.
    val_errors=[]
    for h in HORIZONS:
        errs=[]
        for flight in va:
            for idx in range(cfg.window-1,len(flight['obs'])-h,3):
                p=predict(models[h],flight['obs'],idx,cfg.window)
                errs.append(np.linalg.norm(p[:3]-flight['truth'][idx+h,:3]))
        val_errors.extend(errs)
    low,high=np.quantile(val_errors,[.35,.75])
    adapt=simulate_delayed_adaptation(models,te,cfg.window,float(low),float(high))
    adapt.to_csv(output/'causal_adaptive_decisions.csv',index=False)
    summary={'seed':cfg.seed,'config':vars(cfg),'thresholds_from_validation_m':{'low':float(low),'high':float(high)},
             'adaptive_position_mean_absolute_error_m':float(adapt.pos_error_m.mean()),
             'horizon_fraction':{str(k):float(v) for k,v in adapt.horizon_s.value_counts(normalize=True).items()},
             'warning':'Adaptive errors at different selected horizons MUST NOT be compared directly with fixed-horizon RMSE. Results are exploratory synthetic-data-only.',
             'walltime_seconds':time.perf_counter()-start}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    plt.figure(figsize=(7,4));
    for method in ('CP','CV','CA','LSTM'):
        s=fixed[fixed.method==method];plt.plot(s.horizon_s,s.pos_rmse_m,marker='o',label=method)
    plt.xlabel('Prediction horizon (s)');plt.ylabel('3D position RMSE (m)');plt.legend();plt.tight_layout()
    plt.savefig(output/'fixed_horizon_comparison.png',dpi=180);plt.close()
    subset=adapt[adapt.flight==te[0]['seed']]
    plt.figure(figsize=(8,3));plt.step(subset.origin_idx/FS,subset.horizon_s,where='post');
    plt.xlabel('Time (s)');plt.ylabel('Selected horizon (s)');plt.tight_layout();plt.savefig(output/'adaptive_timeline.png',dpi=180);plt.close()
    print('\n',fixed[['horizon_s','method','pos_rmse_m','vel_rmse_mps']].round(3).to_string(index=False),flush=True)
    print('Adaptive horizon proportions:',summary['horizon_fraction'],flush=True)
    print('Output directory:',output.resolve(),flush=True)

if __name__=='__main__': main()
