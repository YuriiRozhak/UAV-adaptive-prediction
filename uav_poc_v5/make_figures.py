from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent
for suffix in ('results','results_lambda2','results_lambda5'):
    folder=ROOT/suffix
    df=pd.read_csv(folder/'policy_summary.csv')
    fig,ax=plt.subplots(figsize=(9,4.5))
    filtered=df[~df.policy.eq('oracle_hindsight')]
    ax.bar(filtered.policy,filtered.utility)
    ax.tick_params(axis='x',rotation=25)
    ax.set(ylabel='Mean utility loss (lower better)',title=f'Policy comparison — {suffix}')
    fig.tight_layout();fig.savefig(folder/'policy_comparison.png',dpi=180);plt.close(fig)
    actions=pd.read_csv(folder/'policy_actions.csv')
    sel=actions[(actions.policy=='learned') & (actions.flight_id==740000)]
    fig,ax=plt.subplots(figsize=(9,3.4))
    ax.step(sel.origin_idx/20,sel.horizon_s,where='post')
    ax.set(xlabel='Issue time (s)',ylabel='Horizon (s)',ylim=(.15,1.1),title='Learned horizon policy: example trajectory')
    fig.tight_layout();fig.savefig(folder/'selected_horizon_example.png',dpi=180);plt.close(fig)
