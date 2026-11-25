"""Out-of-generator retrospective research benchmark; no autonomous controls.

Protocol fixed before examination of new test results:
lambda=5; h={5,10,20}; CA-RMSE normalisation calibrated on original-generator
held-out calibration set; error regression on original-generator training only.
In addition to v5 predictor all baseline methods can run on the same issue times.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_v2 import synth_flight, FS, HORIZONS
from policy_evaluation_v4 import load_models, calibration, policy_actions
from error_policy_v5 import build_flight_data, fit_regressors, estimate, choose_actions, assess, comparison_ci

LAM=5.0

def shifted_flight(seed, seconds=28, variant='shifted'):
    """Independent trajectory generator using piecewise-constant+periodic acceleration.
    Unlike v2's smooth interpolated acceleration anchors, has abrupt transitions,
    body-frame-ish turns and different observation-noise distribution.
    """
    r=np.random.default_rng(seed); n=seconds*FS; dt=1/FS
    t=np.arange(n)*dt
    a=np.zeros((n,3)); p=np.zeros((n,3)); v=np.zeros((n,3)); gyro=np.zeros((n,3))
    p[0]=r.uniform([-20,-20,4],[20,20,12]);v[0]=r.uniform([-2,-2,-.3],[2,2,.3])
    phase=r.uniform(0,2*np.pi,size=3)
    amps=r.uniform(.35,.95,size=3); amps[2]*=.5
    block_duration=r.integers(24,70)
    block_vals=r.uniform(-.75,.75,size=(int(np.ceil(n/block_duration))+1,3));block_vals[:,2]*=.5
    freq=r.uniform(.3,1.2,size=3)
    for i in range(1,n):
        a[i]=block_vals[i//block_duration]+amps*np.sin(2*np.pi*freq*t[i]+phase)*.25
        if variant=='severe': a[i]*=1.7
        v[i]=.999*v[i-1]+a[i]*dt
        p[i]=p[i-1]+v[i-1]*dt+.5*a[i]*dt**2
    yaw=np.unwrap(np.arctan2(v[:,1],v[:,0]));gyro[:,2]=np.gradient(yaw,dt)
    gyro[:,0]=.02*np.sin(t*.9); gyro[:,1]=.02*np.cos(t*.6)
    truth=np.c_[p,v].astype('float32')
    obs=np.c_[p,v,a,gyro]
    noise=np.array([.10]*3+[.07]*3+[.14]*3+[.035]*3)
    if variant=='severe':noise*=1.8
    # independent Laplace observation noise, not v2 Gaussian
    obs+=r.laplace(0,noise/np.sqrt(2),size=obs.shape)
    return {'seed':seed,'obs':obs.astype('float32'),'truth':truth}

def score_CA(f,d,scale):
    origins=d['origins'];obs=f['obs'];truth=f['truth']
    m={}
    for h in HORIZONS:
        dt=h/FS
        p=obs[origins,:3]+dt*obs[origins,3:6]+.5*(dt**2)*obs[origins,6:9]
        err=np.linalg.norm(p-truth[origins+h,:3],axis=1)
        m[h]=(err/scale[h])**2
    return m

def per_horizon_metrics(f,d,scale):
    records=[];ca=score_CA(f,d,scale)
    for h in HORIZONS:
        lstm=d['position_error'][h]
        records.append({'flight_id':f['seed'],'horizon_s':h/FS,
                        'CA_position_RMSE_m':float(np.sqrt(np.mean(ca[h]*scale[h]**2))),
                        'LSTM_position_RMSE_m':float(np.sqrt(np.mean(lstm**2))),
                        'CA_n':len(lstm)})
    return records

def eval_ca_policy(d, ca, acts, lam, name):
    e=np.array([ca[int(h)][i] for i,h in enumerate(acts)])
    return {'flight_id':d['flight_id'],'policy':name,'utility':float(np.mean(e+lam*(1-acts/20))),
            'norm_sq_error':float(e.mean()),'mean_horizon_s':float(np.mean(acts)/FS)}

def run(cfg):
    start=time.monotonic()
    out=Path(cfg.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    models=load_models(Path(__file__).parent)
    scale_flights=[synth_flight(720000+i,seconds=28) for i in range(8)]
    scales=calibration(scale_flights,models)
    train=[build_flight_data(synth_flight(730000+i,seconds=28),models,scales) for i in range(20)]
    # Three independent regressor seeds. Base LSTM checkpoint is fixed, noted as limitation.
    regressors={s:fit_regressors(train,s)[0] for s in (11,22,33)}
    # New, previously unexamined test IDs. No hyperparameter tuning on them.
    cohorts={
      'same_generator': [synth_flight(860000+i,seconds=28) for i in range(cfg.flights)],
      'shifted_generator': [shifted_flight(870000+i,seconds=28) for i in range(cfg.flights)],
      'severe_shift': [shifted_flight(880000+i,seconds=28,variant='severe') for i in range(cfg.flights)]
    }
    saved=json.loads((Path(__file__).parent/'v2_training_summary.json').read_text())
    low=saved['thresholds_from_validation_m']['low'];high=saved['thresholds_from_validation_m']['high']
    rows=[];errs=[];error_cal=[]
    for cohort, flights in cohorts.items():
      for f in flights:
        d=build_flight_data(f,models,scales);ca=score_CA(f,d,scales)
        for item in per_horizon_metrics(f,d,scales):
            item['cohort']=cohort;errs.append(item)
        v4=policy_actions(d['origins'],d['forecast'],d['truth'],'adaptive',low,high)
        for seed,reg in regressors.items():
          pred=estimate(d,reg)
          decisions={'v5_learned':choose_actions(pred,LAM),
                     'v4_threshold':v4,
                     **{f'fixed_{h/FS:.2f}':np.full(len(d['origins']),h,dtype=int) for h in HORIZONS}}
          for h_idx,h in enumerate(HORIZONS):
             # Calibration: empirical observed squared normalized error vs forecasted score
             error_cal.append({'cohort':cohort,'seed':seed,'flight_id':f['seed'],'horizon_s':h/FS,
                               'predicted_mean':float(pred[:,h_idx].mean()),
                               'observed_mean':float(d['sqnorm'][h].mean())})
          for name,acts in decisions.items():
            for base in ('LSTM','CA'):
              if base=='LSTM':
                score=assess(d,acts,LAM,name)
                utility=score['utility'];norm=score['normalized_sq_error']
              else:
                score=eval_ca_policy(d,ca,acts,LAM,name)
                utility=score['utility'];norm=score['norm_sq_error']
              rows.append({'cohort':cohort,'training_seed':seed,'flight_id':f['seed'],
                           'prediction_source':base,'policy':name,'utility':utility,
                           'norm_sq_error':norm,'mean_horizon_s':float(acts.mean()/FS),
                           'pct_025':float(np.mean(acts==5)),'pct_050':float(np.mean(acts==10)),
                           'pct_100':float(np.mean(acts==20))})
    df=pd.DataFrame(rows);df.to_csv(out/'per_flight_results.csv',index=False)
    pd.DataFrame(errs).to_csv(out/'fixed_horizon_rmse.csv',index=False)
    pd.DataFrame(error_cal).to_csv(out/'error_calibration.csv',index=False)
    summary=(df.groupby(['cohort','prediction_source','policy'])
             .agg(mean_utility=('utility','mean'),std_utility=('utility','std'),
                  mean_horizon_s=('mean_horizon_s','mean'),n=('utility','size'))
             .reset_index().sort_values(['cohort','prediction_source','mean_utility']))
    summary.to_csv(out/'policy_summary.csv',index=False)
    rng=np.random.default_rng(2026); comparisons=[]
    for (cohort,seed,source),sub in df.groupby(['cohort','training_seed','prediction_source']):
      pivot=sub.pivot(index='flight_id',columns='policy',values='utility')
      for other in ['fixed_0.25','fixed_0.50','fixed_1.00','v4_threshold']:
        delta=(pivot['v5_learned']-pivot[other]).to_numpy()
        boot=rng.choice(delta,size=(2000,len(delta)),replace=True).mean(axis=1)
        comparisons.append({'cohort':cohort,'training_seed':seed,'prediction_source':source,
          'comparator':other,'mean_delta':float(delta.mean()),
          'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975)),
          'n_flights':len(delta)})
    comp=pd.DataFrame(comparisons);comp.to_csv(out/'bootstrap_comparisons.csv',index=False)
    protocol={'fixed_lambda':LAM,'test_flights_per_cohort':cfg.flights,
      'regressor_seeds':[11,22,33], 'pretrained_LSTM_seeds':1,
      'cohorts':['same_generator','shifted_generator','severe_shift'],
      'train_regressor_seeds':'730000..730019','calibration_seeds':'720000..720007',
      'test_seeds':'860000+,870000+,880000+','CA_scale_m':scales,
      'evaluation':'normalized position squared error + lambda*(1-h/20)',
      'limits':['Only synthetic simulations, not real flight validation',
         '3 regressor seeds do NOT represent 3 LSTM training seeds',
         'Lambda=5 previously explored on v5, so pre-fixed only for this v6 experiment',
         'Shifted trajectory generator is still simplified physics',
         'Cohort bootstrap CI covers flights conditional on one pretrained LSTM model'],
      'elapsed_sec':round(time.monotonic()-start,2)}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nKey comparisons:',flush=True)
    print(comp[(comp.training_seed==11)&(comp.comparator=='fixed_0.25')].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('elapsed',protocol['elapsed_sec'])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--flights',type=int,default=12);p.add_argument('--output',default='results');run(p.parse_args())
