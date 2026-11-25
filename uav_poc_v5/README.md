# UAV trajectory prediction POC v5 (research-only)

Offline experiments on synthetic trajectories using pretrained v2 LSTM and a separately trained predictor of expected forecast error per horizon.

Contents:
- `experiment_v2.py` — synthetic trajectory generator and pretrained LSTM architecture.
- `policy_evaluation_v4.py` — v4 forecasts and threshold policy, reused for control.
- `net_5.pt`, `net_10.pt`, `net_20.pt` — pretrained checkpoints from POC v2.
- `v2_training_summary.json` — prior validation thresholds.
- `error_policy_v5.py` — full v5 experiment.
- `make_figures.py` — graph production.
- `results*/` — real computed outputs for three illustrative λ values.
- `REPORT_UA.md` — full protocol, numeric results, caveats.

Run from project directory:
```
python error_policy_v5.py --lambda-value 0.5 --output results
python error_policy_v5.py --lambda-value 2 --output results_lambda2
python error_policy_v5.py --lambda-value 5 --output results_lambda5
python make_figures.py
```

There are NO controller interfaces. Synthetic results do not establish real-world navigation performance. `oracle_hindsight` and shuffled controls are retrospective diagnostics, not deployable policies.
