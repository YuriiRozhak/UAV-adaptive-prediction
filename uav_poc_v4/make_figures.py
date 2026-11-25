from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).parent/'results'
s=pd.read_csv(p/'policy_summary.csv')
for name in ['adaptive','fixed_5','fixed_10','fixed_20','random_uniform']:
    x=s[s.policy==name].sort_values('horizon_penalty_weight')
    plt.plot(x.horizon_penalty_weight,x.mean_utility_loss,marker='o',label=name)
plt.xlabel('Horizon penalty weight λ');plt.ylabel('Mean utility loss (lower is better)');plt.legend()
plt.tight_layout();plt.savefig(p/'policy_tradeoff.png',dpi=200);plt.close()
a=pd.read_csv(p/'policy_actions.csv')
f=a.flight_id.iloc[0]
for name in ['adaptive','fixed_10']:
    x=a[(a.flight_id==f)&(a.policy==name)]
    plt.step(x.origin_idx/20,x.horizon_s,where='post',label=name)
plt.xlabel('Synthetic time (s)');plt.ylabel('Chosen horizon (s)');plt.legend()
plt.tight_layout();plt.savefig(p/'causal_horizons_example.png',dpi=200);plt.close()
