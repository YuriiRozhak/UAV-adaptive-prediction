# POC v2 — adaptive short-term navigation-state prediction (research simulation)

This is a **synthetic-data, offline research prototype**, not an autopilot, estimator replacement, or GNSS-denied navigation solution. It has no interfaces to flight hardware or mission execution.

## Reproduce

```bash
python -m pip install -r requirements.txt
python experiment.py --epochs 8 --train-flights 12 --val-flights 4 --test-flights 4 --seconds 28 --output outputs
```

For a fast engineering smoke test:

```bash
python experiment.py --epochs 2 --train-flights 3 --val-flights 2 --test-flights 2 --seconds 8 --hidden 16 --output smoke
```

Random seed is `2026` by default. Network is a two-layer LSTM (48 hidden units in the default *experimental* run; adjust `--hidden 128` to match the dissertation's proposed full configuration). Windows are 40 samples at 20 Hz. Prediction horizons are 0.25 / 0.5 / 1 s. Inputs contain 12 channels and outputs contain 6 channels. Training uses Adam and normalized relative state changes.

## Methodology improvements relative to v1

1. **Independent trajectories** are assigned to training, validation and testing to reduce sequence-overlap leakage.
2. **Consistent acceleration/velocity/position integration** replaces the v1 turn logic that could create discontinuities or an incorrect acceleration/velocity relationship.
3. **Relative displacements and velocity changes** are used as learning targets instead of absolute future coordinates; input coordinates are origin-centered.
4. **Adaptation is causally delayed:** the evaluation of prediction made at time `t` may update the selection rule only at `t + h`, when its target becomes available.
5. **Adaptive thresholds are selected exclusively from validation data**, not from test data. This is a minimal tuning scheme, not a robust hyperparameter search.
6. **Fixed-horizon methods are compared at the same horizons** and on the same held-out flights; aggregation over different horizons is explicitly discouraged.
7. **No hypothetical winning results** are inserted: all CSV metrics are computed from the code execution.

## Files

- `experiment.py`: dataset generator, LSTM, training, baselines, causal adaptation and plotting
- `outputs/fixed_horizon_metrics.csv`: held-out test metrics of CP, CV, CA and LSTM at identical horizons
- `outputs/causal_adaptive_decisions.csv`: timed adaptive decisions, including horizon and matured-error information
- `outputs/summary.json`: configuration, thresholds and horizon usage
- `outputs/net_*.pt`: pretrained research-only checkpoints
- `outputs/fixed_horizon_comparison.png`, `outputs/adaptive_timeline.png`: plots

## Important limitations

- The generator produces smoothly varying synthetic motion, not a realistic flight dynamics model. Strong kinematic baselines can outperform the LSTM, as the first tests indeed show.
- Three horizon models have independent weights, rather than one horizon-conditioned model.
- Adaptation selects a different prediction task as error varies. A smaller average error for the adaptive policy *by itself* would not establish superiority over fixed horizons because the horizons differ. A future study needs a matched-horizon utility or calibration criterion and an ablation of the controller.
- The adaptive controller observes ground truth after the horizon delay. During a real GNSS outage, that feedback is unavailable; this proof of concept cannot claim to solve outage navigation.
- The model output here is purely a forecast evaluated against synthetic ground truth. It is not fed into a flight control loop.
- One seed and 4 held-out trajectories do not support strong generalization claims. Report multiple seeds, confidence intervals and real-flight validation before publication.
- The inference timing is not benchmarked on deployment hardware.

## Suggested next research tests

- Repeat over ≥5 independent random seeds and report mean/standard deviation and bootstrap CIs by flight.
- Add a horizon-conditioned single-model ablation, same-parameter-budget models, and a fixed-horizon calibration matched to operational forecast horizons.
- Add realistic sensor corruption and a strictly masked-input experiment; note that a GNSS outage cannot be validated by allowing unavailable true coordinates as input or feedback.
- Validate on independent publicly accessible flight datasets with clearly documented ground-truth provenance and data splits.
