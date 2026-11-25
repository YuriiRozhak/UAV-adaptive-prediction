"""Research-only hybrid ablation: CA predictor plus learned LSTM residual.

A prediction at t+h is CA(observed[t], h) + LSTM(history of observations).
All test comparisons use the exact same forecast origins and horizons.
"""
import argparse
import copy
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from experiment_v2 import synth_flight, Net, FS, HORIZONS


def arrays(flights, h, window=40, stride=1):
    X=[]; R=[]; B=[]; T=[]; ids=[]
    dt=h/FS
    for f in flights:
        obs,truth=f['obs'],f['truth']
        for i in range(window-1,len(obs)-h,stride):
            window_input=obs[i-window+1:i+1].copy()
            window_input[:,:3]-=window_input[-1,:3]
            x=obs[i,:6]
            acc=obs[i,6:9]
            base=np.r_[x[:3]+x[3:]*dt+.5*acc*dt**2,x[3:]+acc*dt]
            X.append(window_input)
            B.append(base)
            T.append(truth[i+h])
            R.append(truth[i+h]-base)
            ids.append(f['seed'])
    return (np.asarray(X,dtype='float32'),np.asarray(R,dtype='float32'),
            np.asarray(B,dtype='float32'),np.asarray(T,dtype='float32'),np.asarray(ids))


def train_hybrid(train,val,h,seed,epochs,hidden=48):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    X,R,_,_,_=arrays(train,h,stride=4)
    Xv,Rv,_,_,_=arrays(val,h,stride=4)
    xs=StandardScaler().fit(X.reshape(-1,12))
    ys=StandardScaler().fit(R)
    def trans(A,B):
        return (torch.from_numpy(xs.transform(A.reshape(-1,12)).reshape(A.shape).astype('float32')),
                torch.from_numpy(ys.transform(B).astype('float32')))
    tr=DataLoader(TensorDataset(*trans(X,R)),batch_size=128,shuffle=True)
    va=DataLoader(TensorDataset(*trans(Xv,Rv)),batch_size=128)
    net=Net(hidden=hidden)
    opt=torch.optim.Adam(net.parameters(),lr=.001)
    best=np.inf; best_state=None; stale=0; history=[]
    for ep in range(epochs):
        net.train()
        for a,b in tr:
            opt.zero_grad(); loss=((net(a)-b)**2).mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
        net.eval(); losses=[]
        with torch.inference_mode():
            for a,b in va:
                losses.append(float(((net(a)-b)**2).mean()))
        vl=float(np.mean(losses)); history.append(vl)
        if vl < best - 1e-6:
            best=vl; best_state=copy.deepcopy(net.state_dict()); stale=0
        else: stale+=1
        if stale>=5:break
    net.load_state_dict(best_state)
    return net,xs,ys,history


def eval_hybrid(model,flights,h):
    net,xs,ys,_=model
    X,_,base,truth,ids=arrays(flights,h)
    Xs=xs.transform(X.reshape(-1,12)).reshape(X.shape).astype('float32')
    net.eval(); predicted=[]
    with torch.inference_mode():
        for j in range(0,len(Xs),256):
            predicted.append(net(torch.from_numpy(Xs[j:j+256])).numpy())
    residual=ys.inverse_transform(np.concatenate(predicted))
    pred=base+residual
    ca_err=np.linalg.norm(base[:,:3]-truth[:,:3],axis=1)
    hybrid_err=np.linalg.norm(pred[:,:3]-truth[:,:3],axis=1)
    rows=[]
    for uid in np.unique(ids):
        mask=ids==uid
        rows.append({'test_flight_id':int(uid),'horizon_s':h/FS,
                     'CA_RMSE_m':float(np.sqrt(np.mean(ca_err[mask]**2))),
                     'hybrid_RMSE_m':float(np.sqrt(np.mean(hybrid_err[mask]**2))),
                     'CA_MAE_m':float(ca_err[mask].mean()),
                     'hybrid_MAE_m':float(hybrid_err[mask].mean()),
                     'n':int(mask.sum())})
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--hidden',type=int,default=48)
    p.add_argument('--output',default='results')
    p.add_argument('--train-seeds',nargs='+',type=int,default=[11,22])
    args=p.parse_args()
    torch.set_num_threads(2)
    tr=[synth_flight(202600+i,seconds=28) for i in range(12)]
    va=[synth_flight(202612+i,seconds=28) for i in range(4)]
    te=[synth_flight(950000+i,seconds=28) for i in range(8)]
    out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    rows=[]
    for seed in args.train_seeds:
        for h in HORIZONS:
            mod=train_hybrid(tr,va,h,seed,args.epochs,args.hidden)
            res=eval_hybrid(mod,te,h)
            res.insert(0,'training_seed',seed)
            rows.append(res)
            print('seed',seed,'horizon',h/FS,'baseline RMSE',round(float(np.sqrt(np.mean(res.CA_RMSE_m**2))),4),
                  'hybrid RMSE',round(float(np.sqrt(np.mean(res.hybrid_RMSE_m**2))),4),flush=True)
    raw=pd.concat(rows,ignore_index=True)
    raw.to_csv(out/'hybrid_per_flight.csv',index=False)
    summary=raw.groupby(['training_seed','horizon_s']).agg(
        CA_RMSE_m=('CA_RMSE_m',lambda x:float(np.sqrt(np.mean(x**2)))),
        hybrid_RMSE_m=('hybrid_RMSE_m',lambda x:float(np.sqrt(np.mean(x**2)))),
        CA_MAE_m=('CA_MAE_m','mean'),hybrid_MAE_m=('hybrid_MAE_m','mean'),
        test_flights=('test_flight_id','count')).reset_index()
    summary['relative_RMSE_improvement_percent']=100*(summary.CA_RMSE_m-summary.hybrid_RMSE_m)/summary.CA_RMSE_m
    summary.to_csv(out/'hybrid_ablation_summary.csv',index=False)
    print('\n',summary.round(4).to_string(index=False))

if __name__=='__main__':main()
