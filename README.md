# Adaptive UAV Navigation-State Prediction

This repository collects successive offline research prototypes for short-term UAV navigation-state forecasting and adaptive prediction-horizon selection. The experiments use **synthetic trajectories** and compare neural forecasts with kinematic baselines and alternative selection policies.

These prototypes do not control a UAV or implement a navigation fallback. Synthetic results alone do not establish real-world flight or GNSS-outage performance.

## Versions

| Version | Focus | Documentation |
| --- | --- | --- |
| v2 | LSTM forecasts and kinematic baselines; independent trajectory splits and causally delayed adaptive selection. | [README](uav_poc_v2/README.md), [audit (Ukrainian)](uav_poc_v2/AUDIT_UA.md) |
| v3 | Noise robustness, flight-level bootstrap evaluation, and a separate constant-acceleration + LSTM residual pilot. | [README](uav_poc_v3/README.md), [report (Ukrainian)](uav_poc_v3/REPORT_UA.md) |
| v4 | Causal horizon-selection evaluation, including a matched-mix diagnostic. | [README](uav_poc_v4/README.md), [report (Ukrainian)](uav_poc_v4/REPORT_UA.md) |
| v5 | Horizon selection using a separately trained forecast-error predictor, with experiments at multiple utility weights. | [README](uav_poc_v5/README.md), [report (Ukrainian)](uav_poc_v5/REPORT_UA.md) |
| v6 | Independent synthetic shift study of adaptive horizon-selection policies. | [README](uav_poc_v6/README.md), [report (Ukrainian)](uav_poc_v6/REPORT_UA.md) |

The available folders start at v2; v1 is referenced in the v2 documentation but is not included here. Later versions reuse components and pretrained weights from earlier experiments. Each version's folder contains its own scripts, dependencies and research outputs.

## Setup

Install Python and the dependencies for the version you want to run. For example, from the repository root:

```bash
cd uav_poc_v2
python -m pip install -r requirements.txt
```

The version-specific requirements list PyTorch, NumPy, pandas, scikit-learn and Matplotlib. Run the commands below from the corresponding version directory. Outputs are written to the specified output directories; existing results there may be overwritten.

## Run the experiments

### v2: baseline and adaptive prediction

```bash
python experiment.py --epochs 8 --train-flights 12 --val-flights 4 --test-flights 4 --seconds 28 --output outputs
```

The [v2 README](uav_poc_v2/README.md) also provides a faster smoke test and details of the model and evaluation.

### v3: robustness

```bash
python robustness_v3.py --models . --output results
```

This command requires the included v2 checkpoints and `summary.json` in the v3 directory. See the [v3 report](uav_poc_v3/REPORT_UA.md) for the separate hybrid residual experiment and its configuration.

### v4: horizon-selection policies

```bash
python policy_evaluation_v4.py --models . --output results
python matched_mix_diagnostic.py --models . --output results
python make_figures.py
```

### v5: forecast-error-based policy

```bash
python error_policy_v5.py --lambda-value 0.5 --output results
python error_policy_v5.py --lambda-value 2 --output results_lambda2
python error_policy_v5.py --lambda-value 5 --output results_lambda5
python make_figures.py
```

### v6: synthetic shift study

```bash
python confirmatory_v6.py --flights 12 --output results
```

Consult each version's README and report for its precise protocol, outputs, model dependencies and limitations. Later experiments use previously trained model checkpoints; reproducing their evaluation is distinct from retraining those models.

## Interpreting the results

- The data are synthetic, and the motion generator is not a validated real-flight dynamics model.
- Forecast errors at different horizons represent different prediction tasks. A lower aggregate error for a policy that selects shorter horizons does not by itself establish superiority; consider matched-horizon or matched-mix comparisons.
- Some experiments use delayed ground-truth feedback. Such feedback may be unavailable during a real GNSS outage.
- Retrospective oracle and shuffled controls in v5 are diagnostics, not deployable policies.
- Refer to each report for experimental sample sizes, uncertainty, results and version-specific caveats. Real-flight validation and deployment-hardware timing remain necessary before operational conclusions can be drawn.
