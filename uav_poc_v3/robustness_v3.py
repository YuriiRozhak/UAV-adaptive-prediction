"""Offline, non-operational UAV state-forecast evaluation, version 3.

Reuses already-trained synthetic v2 checkpoints, and adds:
 - independent held-out synthetic flights / repeated evaluation batches
 - robustness to observation noise
 - causal delayed decisions at a fixed issue cadence
 - SAME-origin, SAME-horizon pairwise controls
 - cluster (flight-level) bootstrap confidence intervals

No real-flight validation and no GNSS-outage claims.
"""
import argparse
import heapq
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from experiment_v2 import synth_flight, Net, baseline, FS, HORIZONS


def load_models(model_dir, hidden=48):
    models = {}
    for horizon in HORIZONS:
        ckpt = torch.load(Path(model_dir) / f'net_{horizon}.pt', map_location='cpu', weights_only=False)
        net = Net(hidden=hidden)
        net.load_state_dict(ckpt['state'])
        net.eval()
        xsc = StandardScaler()
        xsc.mean_ = np.asarray(ckpt['x_mean'])
        xsc.scale_ = np.asarray(ckpt['x_scale'])
        xsc.var_ = xsc.scale_ ** 2
        xsc.n_features_in_ = 12
        ysc = StandardScaler()
        ysc.mean_ = np.asarray(ckpt['y_mean'])
        ysc.scale_ = np.asarray(ckpt['y_scale'])
        ysc.var_ = ysc.scale_ ** 2
        ysc.n_features_in_ = 6
        models[horizon] = (net, xsc, ysc)
    return models


def inference(model, observation, idx, window):
    net, xsc, ysc = model
    array = observation[idx-window+1:idx+1].copy()
    array[:, :3] -= array[-1, :3]
    array = xsc.transform(array).astype('float32')
    with torch.inference_mode():
        pred = net(torch.from_numpy(array[None])).numpy()
    return (ysc.inverse_transform(pred)[0] + observation[idx, :6]).astype('float64')


def evaluate_flight(flight, models, thresholds, window=40, stride=5, alpha=.2, regime='base'):
    """Every issued forecast is scored only after outcome maturity for decision-making.
    Forecast quality is scored retrospectively and at identical prediction horizons.
    """
    obs, truth = flight['obs'], flight['truth']
    low, high = thresholds
    queue=[]
    ema=None
    selected=10
    event_id=0
    rows=[]
    # Same common issue origins across every method and regime.
    for idx in range(window-1, len(obs)-max(HORIZONS), stride):
        # Mature predictions from prior issue times before choosing current horizon.
        while queue and queue[0][0] <= idx:
            target_idx, _, pred = heapq.heappop(queue)
            error = float(np.linalg.norm(pred[:3] - truth[target_idx, :3]))
            ema = error if ema is None else (alpha*error + (1-alpha)*ema)
            selected = 20 if ema < low else 10 if ema < high else 5
        h = selected
        pred = inference(models[h], obs, idx, window)
        comps = baseline(obs, idx, h)
        event_id += 1
        heapq.heappush(queue, (idx+h, event_id, pred))
        actual = truth[idx+h]
        errors = {'LSTM': float(np.linalg.norm(pred[:3]-actual[:3]))}
        for name,p in comps.items():
            errors[name] = float(np.linalg.norm(p[:3]-actual[:3]))
        rows.append({
            'flight_id': flight['seed'], 'regime':regime,
            'issue_idx':idx, 'target_idx':idx+h,
            'horizon_s':h/FS,
            'used_ema_m':float(ema) if ema is not None else np.nan,
            'LSTM_error_m':errors['LSTM'], 'CP_error_m':errors['CP'],
            'CV_error_m':errors['CV'], 'CA_error_m':errors['CA']
        })
    return rows


def summarize_events(df, bootstraps=800, seed=78):
    rng=np.random.default_rng(seed)
    summary=[]
    names=('LSTM','CP','CV','CA')
    for regime,subset in df.groupby('regime',sort=False):
        # Compute metrics for each chosen horizon, paired on exact same (flight, issue, target).
        for h,part in subset.groupby('horizon_s'):
            for name in names:
                vals=part[f'{name}_error_m'].to_numpy()
                summary.append({'regime':regime,'horizon_s':h,'method':name,
                                'count':len(vals),'rmse_3d_m':float(np.sqrt(np.mean(vals**2))),
                                'mae_3d_m':float(np.mean(vals))})
    # One forecast from LSTM vs one CA at exact same forecast target/horizon.
    # Flight cluster bootstrap for MAE difference: positive => CA better than LSTM.
    paired=[]
    for regime,part in df.groupby('regime',sort=False):
        flights=list(part.flight_id.unique())
        blocks=[part[part.flight_id==f] for f in flights]
        def diff(blocks):
            combined=pd.concat(blocks,ignore_index=True)
            return float((combined.LSTM_error_m - combined.CA_error_m).mean())
        point=diff(blocks)
        boots=[]
        for _ in range(bootstraps):
            sampled=rng.integers(0,len(blocks),size=len(blocks))
            boots.append(diff([blocks[j] for j in sampled]))
        paired.append({'regime':regime,'flights':len(flights),
                       'observations':len(part),
                       'LSTM_minus_CA_MAE_m':point,
                       'CI95_low_m':float(np.quantile(boots,.025)),
                       'CI95_high_m':float(np.quantile(boots,.975)),
                       'average_selected_horizon_s':float(part.horizon_s.mean()),
                       'LSTM_MAE_m':float(part.LSTM_error_m.mean()),
                       'CA_MAE_m':float(part.CA_error_m.mean())})
    return pd.DataFrame(summary),pd.DataFrame(paired)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',default='.')
    parser.add_argument('--output',default='results')
    parser.add_argument('--flights-per-regime',type=int,default=8)
    parser.add_argument('--seconds',type=int,default=28)
    parser.add_argument('--stride',type=int,default=5)
    parser.add_argument('--hidden',type=int,default=48)
    parser.add_argument('--bootstraps',type=int,default=500)
    args=parser.parse_args()
    torch.set_num_threads(2)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    models=load_models(args.models,args.hidden)
    threshold_path=Path(args.models)/'summary.json'
    old=json.loads(threshold_path.read_text(encoding='utf-8'))
    low=float(old['thresholds_from_validation_m']['low'])
    high=float(old['thresholds_from_validation_m']['high'])
    all_rows=[]
    # Test seeds distinct from train/val/test seeds in v2 (202600+).
    for j,(regime,noise) in enumerate([('nominal_noise',1.0),('noise_x2',2.0),('noise_x4',4.0)]):
        for k in range(args.flights_per_regime):
            seed=900000+j*1000+k
            f=synth_flight(seed,seconds=args.seconds,noise=noise)
            all_rows.extend(evaluate_flight(f,models,(low,high),stride=args.stride,regime=regime))
        print(regime,'completed',flush=True)
    events=pd.DataFrame(all_rows)
    events.to_csv(output/'paired_forecasts.csv',index=False)
    metrics,paired=summarize_events(events,bootstraps=args.bootstraps)
    metrics.to_csv(output/'matched_horizon_metrics.csv',index=False)
    paired.to_csv(output/'paired_flight_bootstrap.csv',index=False)
    print('\nPaired comparisons (positive means LSTM worse):')
    print(paired.round(4).to_string(index=False))
    print('\nMetrics:',metrics.head(12).round(3).to_string(index=False))

    fig,ax=plt.subplots(figsize=(8.2,4.2))
    locations=np.arange(len(paired))
    width=.33
    ax.bar(locations-width/2,paired['LSTM_MAE_m'],width,label='LSTM')
    ax.bar(locations+width/2,paired['CA_MAE_m'],width,label='Constant acceleration')
    ax.set_xticks(locations,paired['regime'])
    ax.set_ylabel('Position MAE, m (same issue and horizon)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/'paired_error_noise.png',dpi=200)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.2,4.2))
    for r,part in events.groupby('regime',sort=False):
        hist=part['horizon_s'].value_counts(normalize=True)
        ax.plot([.25,.5,1.0],[hist.get(h,0.) for h in (.25,.5,1.)],marker='o',label=r)
    ax.set_xlabel('Selected horizon, s')
    ax.set_ylabel('Fraction of issued forecasts')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/'selected_horizons_noise.png',dpi=200)
    plt.close(fig)

    (output/'experiment_config.json').write_text(json.dumps({
        'trained_model_source':str(args.models),'model_training_seed':old['seed'],
        'thresholds_calibrated_on_v2_validation_m':{'low':low,'high':high},
        'independent_test_seed_scheme':'900000 + regime_index * 1000 + flight_index',
        'test_flights_per_regime':args.flights_per_regime,'seconds':args.seconds,
        'issued_forecast_interval_s':args.stride/FS,
        'bootstraps':args.bootstraps,
        'limits':['Synthetic data only','Same trained models across all noise levels',
                  'No independent retraining-seed repeats',
                  'No realistic missing-GNSS inputs',
                  'Bootstrap CI reflects only held-out flight variation']
    },ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
