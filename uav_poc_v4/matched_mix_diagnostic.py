"""Diagnostic: does time allocation improve over shuffling the SAME horizon counts?
Shuffled actions are post-hoc counterfactuals, not deployable causal policies.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_v2 import synth_flight, FS, HORIZONS
from policy_evaluation_v4 import load_models,predict_all,policy_actions


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',default='.')
    ap.add_argument('--output',default='results');ap.add_argument('--flights',type=int,default=12)
    ap.add_argument('--permutations',type=int,default=300);a=ap.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
    torch_seed=202603
    rng=np.random.default_rng(torch_seed)
    models=load_models(Path(a.models))
    old=json.loads((Path(a.models)/'v2_training_summary.json').read_text())
    low=old['thresholds_from_validation_m']['low'];high=old['thresholds_from_validation_m']['high']
    validation=[synth_flight(202612+i,seconds=28) for i in range(4)]
    val_loss={h:[] for h in HORIZONS}
    for f in validation:
        origins,forecast=predict_all(f,models)
        for h in HORIZONS:
            e=np.linalg.norm(forecast[h][:,:3]-f['truth'][origins+h,:3],axis=1)
            val_loss[h].extend((e*e).tolist())
    # For each horizon measure relative difficulty against validation LSTM RMS, not raw errors.
    scales={h:float(np.sqrt(np.mean(val_loss[h]))) for h in HORIZONS}
    rows=[]
    for flight_id in range(a.flights):
        f=synth_flight(990000+flight_id,seconds=28)
        origins,forecast=predict_all(f,models)
        actions=policy_actions(origins,forecast,f['truth'],'adaptive',low,high)
        losses=np.stack([np.linalg.norm(forecast[h][:,:3]-f['truth'][origins+h,:3],axis=1)**2/(scales[h]**2) for h in HORIZONS],axis=1)
        ix={h:i for i,h in enumerate(HORIZONS)}
        current=np.mean([losses[k,ix[int(h)]] for k,h in enumerate(actions)])
        random_shuffles=[]
        for _ in range(a.permutations):
            mix=rng.permutation(actions)
            random_shuffles.append(float(np.mean([losses[k,ix[int(h)]] for k,h in enumerate(mix)])))
        random_shuffles=np.asarray(random_shuffles)
        rows.append({'flight_id':f['seed'],'adaptive_normalized_loss':current,
          'shuffled_same_horizon_counts_mean':float(random_shuffles.mean()),
          'adaptive_minus_shuffled_negative_favors_adaptive':float(current-random_shuffles.mean()),
          'fraction_shuffles_beaten_by_adaptive':float(np.mean(current<random_shuffles)),
          'issue_count':len(actions)})
    d=pd.DataFrame(rows);d.to_csv(out/'matched_mix_per_flight.csv',index=False)
    effect=d.adaptive_minus_shuffled_negative_favors_adaptive.to_numpy()
    boot=np.mean(rng.choice(effect,size=(3000,len(effect)),replace=True),axis=1)
    result={'n_flights':len(d),'shuffles_per_flight':a.permutations,
       'mean_adaptive_loss':float(d.adaptive_normalized_loss.mean()),
       'mean_shuffled_loss':float(d.shuffled_same_horizon_counts_mean.mean()),
       'mean_adaptive_minus_shuffled':float(effect.mean()),
       'flight_bootstrap95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
       'interpretation':'Negative delta suggests beneficial timing beyond selected horizon fractions. Post-hoc matched-mix comparator is NOT deployable.',
       'validation_LSTM_RMSE_scales_m':scales}
    (out/'matched_mix_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
