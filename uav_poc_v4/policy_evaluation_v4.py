"""Offline methodological evaluation of causal horizon-selection policies.
Synthetic test data only. No interfaces to real vehicles.
"""
import argparse
import heapq
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_v2 import synth_flight, Net, FS, HORIZONS

WINDOW=40
ISSUE_STEP=5


def load_models(folder):
    models={}
    for h in HORIZONS:
        data=torch.load(folder/f'net_{h}.pt',map_location='cpu',weights_only=False)
        model=Net(hidden=48)
        model.load_state_dict(data['state']);model.eval()
        models[h]=(model, data)
    return models


def predict_all(flight, models):
    """Counterfactual predictions for every horizon at identical issue times."""
    obs,truth=flight['obs'],flight['truth']
    origins=np.arange(WINDOW-1,len(obs)-max(HORIZONS),ISSUE_STEP)
    X=np.stack([obs[i-WINDOW+1:i+1].copy() for i in origins]).astype('float32')
    X[:,:,:3]-=X[:,-1:,:3]
    forecasts={}
    with torch.inference_mode():
        for h,(net,checkpoint) in models.items():
            scale=checkpoint['x_scale']; mean=checkpoint['x_mean']
            xs=((X-mean)/scale).astype('float32')
            preds=[]
            for j in range(0,len(xs),128):
                preds.append(net(torch.from_numpy(xs[j:j+128])).numpy())
            y=np.concatenate(preds)*checkpoint['y_scale']+checkpoint['y_mean']
            forecasts[h]=y+obs[origins,:6]
    return origins,forecasts


def calibration(flights,models):
    # Calibrate benchmark CA error scale on validation only.
    squares={h:[] for h in HORIZONS}
    for f in flights:
        obs,truth=f['obs'],f['truth']; origins,_=predict_all(f,models)
        for h in HORIZONS:
            dt=h/FS
            est=obs[origins,:3]+dt*obs[origins,3:6]+.5*dt*dt*obs[origins,6:9]
            err=np.linalg.norm(est-truth[origins+h,:3],axis=1)
            squares[h].extend((err**2).tolist())
    return {h:float(np.sqrt(np.mean(squares[h]))) for h in HORIZONS}


def policy_actions(origins,forecast,truth, policy,low,high,seed=0):
    if policy.startswith('fixed_'):
        return np.full(len(origins),int(policy.split('_')[1]),dtype=int)
    if policy=='cycling':
        return np.asarray([HORIZONS[k%3] for k in range(len(origins))],dtype=int)
    if policy=='random_uniform':
        return np.random.default_rng(seed).choice(HORIZONS,len(origins))
    if policy!='adaptive':raise ValueError(policy)
    pending=[]; ema=None; current=10; actions=[]
    for k,origin in enumerate(origins):
        while pending and pending[0][0]<=origin:
            _,_,position_error=heapq.heappop(pending)
            ema=position_error if ema is None else .2*position_error+.8*ema
            current=20 if ema<low else (10 if ema<high else 5)
        h=current;actions.append(h)
        err=np.linalg.norm(forecast[h][k,:3]-truth[origin+h,:3])
        heapq.heappush(pending,(int(origin+h),k,float(err)))
    return np.asarray(actions,dtype=int)


def evaluate_policy(f,origins,forecast,actions,ca_scale,penalty,policy):
    truth=f['truth']; sel_preds=np.stack([forecast[int(h)][i,:3] for i,h in enumerate(actions)])
    target=np.stack([truth[int(origin+h),:3] for origin,h in zip(origins,actions)])
    err=np.linalg.norm(sel_preds-target,axis=1)
    norm_loss=np.asarray([(err[i]/ca_scale[int(h)])**2 for i,h in enumerate(actions)])
    horizon_penalty=penalty*(1-actions/max(HORIZONS))
    utility_loss=norm_loss+horizon_penalty
    return {'flight_id':int(f['seed']),'policy':policy,'n':len(actions),
            'raw_mae_m_not_comparable_across_policies':float(err.mean()),
            'mean_normalized_squared_error':float(norm_loss.mean()),
            'mean_horizon_s':float(actions.mean()/FS),
            'mean_utility_loss_lower_better':float(utility_loss.mean()),
            'fraction_025':float(np.mean(actions==5)),
            'fraction_050':float(np.mean(actions==10)),
            'fraction_100':float(np.mean(actions==20))}


def bootstrap_compare(df, reference='adaptive', B=1500):
    rng=np.random.default_rng(202603)
    pivot=df.pivot(index='flight_id',columns='policy',values='mean_utility_loss_lower_better')
    out=[]
    for comparator in pivot.columns:
        if comparator==reference:continue
        d=(pivot[reference]-pivot[comparator]).to_numpy()
        draws=np.mean(rng.choice(d,size=(B,len(d)),replace=True),axis=1)
        out.append({'reference':reference,'comparator':comparator,
                    'mean_difference_negative_favors_adaptive':float(d.mean()),
                    'bootstrap95_low':float(np.quantile(draws,.025)),
                    'bootstrap95_high':float(np.quantile(draws,.975)),
                    'flights':len(d)})
    return pd.DataFrame(out)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--models',default='.')
    ap.add_argument('--output',default='results')
    ap.add_argument('--test-flights',type=int,default=12)
    ap.add_argument('--seconds',type=int,default=28)
    ap.add_argument('--penalties',type=float,nargs='+',default=[0,0.25,0.5,1.0])
    args=ap.parse_args()
    torch.set_num_threads(2)
    out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    models=load_models(Path(args.models))
    val=[synth_flight(202612+i,seconds=args.seconds) for i in range(4)]
    test=[synth_flight(990000+i,seconds=args.seconds) for i in range(args.test_flights)]
    scales=calibration(val,models)
    original=json.loads((Path(args.models)/'v2_training_summary.json').read_text())
    low=original['thresholds_from_validation_m']['low']
    high=original['thresholds_from_validation_m']['high']
    print('CA RMSE calibration',scales, 'thresholds',low,high,flush=True)
    all_rows=[]; action_rows=[]
    policies=['adaptive','fixed_5','fixed_10','fixed_20','cycling','random_uniform']
    for f in test:
        origins,forecast=predict_all(f,models)
        decisions={p:policy_actions(origins,forecast,f['truth'],p,low,high,seed=int(f['seed'])) for p in policies}
        for p,actions in decisions.items():
            for penalty in args.penalties:
                row=evaluate_policy(f,origins,forecast,actions,scales,penalty,p)
                row['horizon_penalty_weight']=penalty
                all_rows.append(row)
            for i,h in enumerate(actions):
                action_rows.append({'flight_id':f['seed'],'origin_idx':int(origins[i]),'policy':p,'horizon_s':h/FS})
    df=pd.DataFrame(all_rows)
    df.to_csv(out/'per_flight_policy_metrics.csv',index=False)
    pd.DataFrame(action_rows).to_csv(out/'policy_actions.csv',index=False)
    summary=df.groupby(['horizon_penalty_weight','policy']).agg(
        mean_utility_loss=('mean_utility_loss_lower_better','mean'),
        mean_normalized_squared_error=('mean_normalized_squared_error','mean'),
        avg_horizon_s=('mean_horizon_s','mean'),
        flights=('flight_id','nunique')).reset_index()
    summary.to_csv(out/'policy_summary.csv',index=False)
    cis=[]
    for weight in args.penalties:
        x=bootstrap_compare(df[df.horizon_penalty_weight==weight])
        x['horizon_penalty_weight']=weight
        cis.append(x)
    pd.concat(cis,ignore_index=True).to_csv(out/'paired_bootstrap.csv',index=False)
    metadata={'CA_RMSE_validation_scales_m':scales,'validation_thresholds_m':{'low':low,'high':high},
              'penalty_definition':'normalized squared position error + lambda*(1 - horizon_steps/20)',
              'comparison_caveat':'A priori research utility. Mean mixed-horizon raw MAE is not evidence of improvement.',
              'test_generator':'experiment_v2.synth_flight','test_flights':args.test_flights,
              'model_training':'pretrained POC v2 LSTM; one training seed; validation from v2 synthetic generator',
              'adaptation':'past matured LSTM error EMA; no future information at decision time'}
    (out/'protocol.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(summary.round(4).to_string(index=False),flush=True)
    print('Results:',out.resolve())

if __name__=='__main__':main()
