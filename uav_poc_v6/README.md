# POC v6 — Independent Synthetic Shift Study

Code and results for an offline **synthetic** benchmark of adaptive prediction-horizon policies.

Run with `python confirmatory_v6.py --flights 12 --output results` after installing `requirements.txt`.

Core files: `confirmatory_v6.py` (new), `error_policy_v5.py` (forecast error models), `policy_evaluation_v4.py` (pretrained model inference), `experiment_v2.py` (base generator); `net_*.pt` are previously trained frozen model weights. `REPORT_UA.md` explains methods, results and limitations.

Nothing in this project implements a flight controller, a navigation fallback, or real-time UAV operation. Any scientific use must distinguish synthetic simulation findings from real-world evidence.
