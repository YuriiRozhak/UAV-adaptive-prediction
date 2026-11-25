"""Offline synthetic trajectory forecast-error estimation and horizon selection.
Research-only, not connected to any vehicle or flight-controller.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from experiment_v2 import synth_flight, FS, HORIZONS
from policy_evaluation_v4 import load_models,predict_all,calibration,policy_actions

WINDOW=40; ISSUE_STEP=5

def build_flight_data(f, models, scales):
    obs,truth=f['obs'],f['truth']; origins,forecasts=predict_all(f,models)
    # Features at issue time only (no future labels). All forecasts are available at issue time.
    features={}; sqnorm={}; realerror={}
    for h in HORIZONS:
        dt=h/FS; current=obs[origins]
        ca_p=current[:,:3]+dt*current[:,3:6]+.5*dt*dt*current[:,6:9]
        cv_p=current[:,:3]+dt*current[:,3:6]
        win_acc=obs[:,6:9]; win_gyr=obs[:,9:12]
        # Moving average of acceleration difference and jerk (available past only)
        acc_change=np.asarray([np.linalg.norm(win_acc[max(0,i-5):i+1][-1]-win_acc[max(0,i-5):i+1][0]) for i in origins])
        gyro_mag=np.linalg.norm(current[:,9:12],axis=1)
        accel_mag=np.linalg.norm(current[:,6:9],axis=1)
        # Difference between neural and physical extrapolation indicates model disagreement.
        disagreement=np.linalg.norm(forecasts[h][:,:3]-ca_p,axis=1)
        fut_disp=np.linalg.norm(forecasts[h][:,:3]-current[:,:3],axis=1)
        speed=np.linalg.norm(current[:,3:6],axis=1)
        cv_ca=np.linalg.norm(ca_p-cv_p,axis=1)
        # Use scaled derived signals only, never target truth or future sensor readings.
        features[h]=np.column_stack([np.log1p(speed),accel_mag,gyro_mag,acc_change,
                                     disagreement,cv_ca,fut_disp,dt*np.ones(len(origins))])
        realerror[h]=np.linalg.norm(forecasts[h][:,:3]-truth[origins+h,:3],axis=1)
        sqnorm[h]=(realerror[h]/scales[h])**2
    return {'flight_id':f['seed'],'origins':origins,'features':features,'sqnorm':sqnorm,
            'position_error':realerror,'forecast':forecasts,'truth':truth}

def fit_regressors(caldata,seed, cap=100):
    regressors={}; training={}
    for h in HORIZONS:
        X=np.vstack([d['features'][h] for d in caldata]); y=np.hstack([d['sqnorm'][h] for d in caldata])
        y=np.clip(y,0,cap)
        reg=HistGradientBoostingRegressor(loss='squared_error',learning_rate=.06,max_iter=80,
                max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=20,
                early_stopping=False,random_state=seed)
        reg.fit(X,y); regressors[h]=reg
        training[h]={'n':len(y),'mean_target_clipped':float(y.mean())}
    return regressors,training

def estimate(d,models):
    return np.column_stack([np.maximum(.0,models[h].predict(d['features'][h])) for h in HORIZONS])

def choose_actions(est,penalty):
    hs=np.array(HORIZONS)
    loss=est+penalty*(1-hs/max(HORIZONS))
    return hs[np.argmin(loss,axis=1)]

def assess(d,acts,penalty,name):
    errs=np.asarray([d['sqnorm'][int(h)][i] for i,h in enumerate(acts)])
    pen=penalty*(1-acts/20)
    return {'flight_id':d['flight_id'],'policy':name,'lambda':penalty,'utility':float(np.mean(errs+pen)),
            'normalized_sq_error':float(np.mean(errs)),'mean_horizon_s':float(np.mean(acts)/FS),
            'fraction_025':float(np.mean(acts==5)),'fraction_050':float(np.mean(acts==10)),
            'fraction_100':float(np.mean(acts==20))}

def comparison_ci(rows,reference='learned',B=2000):
    pivot=rows.pivot(index='flight_id',columns='policy',values='utility')
    rng=np.random.default_rng(20261114)
    comparisons=[]
    for method in pivot.columns:
        if method==reference:continue
        diff=(pivot[reference]-pivot[method]).to_numpy()
        draws=np.mean(rng.choice(diff,size=(B,len(diff)),replace=True),axis=1)
        comparisons.append({'reference':reference,'comparator':method,'delta_negative_favors_learned':float(diff.mean()),
            'ci_low':float(np.quantile(draws,.025)),'ci_high':float(np.quantile(draws,.975)),
            'n_flights':len(diff)})
    return pd.DataFrame(comparisons)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cal-flights',type=int,default=20)
    parser.add_argument('--test-flights',type=int,default=24)
    parser.add_argument('--seconds',type=int,default=28)
    parser.add_argument('--lambda-value',type=float,default=.5)
    parser.add_argument('--output',default='results')
    a=parser.parse_args(); started=time.monotonic()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(2)
    models=load_models(Path(__file__).parent)
    # Scales are independently estimated; calibration and test trajectories are disjoint.
    scale_flights=[synth_flight(720000+i,seconds=a.seconds) for i in range(8)]
    scale=calibration(scale_flights,models)
    calibration_flights=[synth_flight(730000+i,seconds=a.seconds) for i in range(a.cal_flights)]
    test_flights=[synth_flight(740000+i,seconds=a.seconds) for i in range(a.test_flights)]
    print('Scaling RMSE:',scale,flush=True)
    caldata=[build_flight_data(f,models,scale) for f in calibration_flights]
    testdata=[build_flight_data(f,models,scale) for f in test_flights]
    regressors,train_stats=fit_regressors(caldata,11)
    # Fixed thresholds were selected in v2, not using v5 test data.
    saved=json.loads((Path(__file__).parent/'v2_training_summary.json').read_text())
    low=saved['thresholds_from_validation_m']['low']; high=saved['thresholds_from_validation_m']['high']
    rows=[]; action_rows=[]; eval_rows=[]
    for d in testdata:
        estimates=estimate(d,regressors)
        for idx,h in enumerate(HORIZONS):
            actual=d['sqnorm'][h]
            for e,y in zip(estimates[:,idx],actual):
                eval_rows.append({'flight_id':d['flight_id'],'horizon_s':h/FS,'estimated_loss':float(e),'actual_loss':float(y)})
        actions={'learned':choose_actions(estimates,a.lambda_value)}
        actions.update({f'fixed_{h}':np.full(len(d['origins']),h,dtype=int) for h in HORIZONS})
        actions['threshold_v4']=policy_actions(d['origins'],d['forecast'],d['truth'],'adaptive',low,high)
        actions['random_uniform']=np.random.default_rng(d['flight_id']).choice(HORIZONS,len(d['origins']))
        # Hindsight oracle diagnostic, not a policy usable at decision time.
        scores=np.column_stack([d['sqnorm'][h]+a.lambda_value*(1-h/20) for h in HORIZONS])
        actions['oracle_hindsight']=np.asarray(HORIZONS)[np.argmin(scores,axis=1)]
        for name,act in actions.items():
            rows.append(assess(d,act,a.lambda_value,name))
            for origin,h in zip(d['origins'],act):
                action_rows.append({'flight_id':d['flight_id'],'origin_idx':int(origin),'policy':name,'horizon_s':h/FS})
    df=pd.DataFrame(rows);df.to_csv(out/'per_flight_metrics.csv',index=False)
    adf=pd.DataFrame(action_rows);adf.to_csv(out/'policy_actions.csv',index=False)
    edf=pd.DataFrame(eval_rows);edf.to_csv(out/'error_regressor_predictions.csv',index=False)
    summary=df.groupby('policy').agg(utility=('utility','mean'),norm_sq_error=('normalized_sq_error','mean'),
            horizon_s=('mean_horizon_s','mean'),pct_025=('fraction_025','mean'),pct_050=('fraction_050','mean'),
            pct_100=('fraction_100','mean')).reset_index().sort_values('utility')
    summary.to_csv(out/'policy_summary.csv',index=False)
    compare=comparison_ci(df);compare.to_csv(out/'paired_bootstrap.csv',index=False)
    # Post hoc matched-mix shuffle: preserves each flight's horizon counts, changes only temporal alignment.
    rng=np.random.default_rng(77); shuffled=[]
    for d in testdata:
        act=choose_actions(estimate(d,regressors),a.lambda_value)
        expected=assess(d,act,a.lambda_value,'learned')['utility']
        mixed=[]
        for i in range(200):
            perm=rng.permutation(act)
            mixed.append(assess(d,perm,a.lambda_value,'shuffled')['utility'])
        shuffled.append({'flight_id':d['flight_id'],'learned_utility':expected,
                         'shuffled_mean_utility':float(np.mean(mixed)),
                         'delta_learned_minus_shuffle':float(expected-np.mean(mixed))})
    shuffled=pd.DataFrame(shuffled);shuffled.to_csv(out/'matched_mix.csv',index=False)
    delta=shuffled.delta_learned_minus_shuffle.to_numpy()
    draw=np.mean(rng.choice(delta,size=(3000,len(delta)),replace=True),axis=1)
    matched={'mean_delta':float(np.mean(delta)),'ci95':[float(np.quantile(draw,.025)),float(np.quantile(draw,.975))]}
    info={'settings':vars(a),'scale_CA_RMSE_validation':scale,'calibration_flights':a.cal_flights,
          'test_flights':a.test_flights,'training_regressor_stats':train_stats,
          'threshold_v4_m':{'low':low,'high':high},'matched_mix':matched,
          'elapsed_s':round(time.monotonic()-started,2),
          'limits':['synthetic trajectories only','single pretrained LSTM seed per horizon',
            'hand-defined utility lambda','no flight controller or real world navigation validation',
            'hindsight oracle is noncausal diagnostic only']}
    (out/'protocol.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\n',summary.round(4).to_string(index=False),flush=True)
    print('\nConfidence intervals\n',compare.round(4).to_string(index=False),flush=True)
    print('\nMatched horizon mix:',matched,flush=True)
    print('Duration s',info['elapsed_s'],flush=True)

if __name__=='__main__':main()
