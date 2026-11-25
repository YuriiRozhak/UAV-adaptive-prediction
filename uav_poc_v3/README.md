# UAV forecasting POC v3 — offline research

Код **не керує БПЛА**. Це синтетична демонстрація методики оцінювання нейромережевих прогнозів.

- `experiment_v2.py` — копія логіки генератора, кінематичних baseline і LSTM з POC v2.
- `robustness_v3.py` — незалежні тестові траєкторії, три рівні шуму, порівняння на збіжних горизонтах, причинний адаптивний вибір, bootstrap за польотами.
- `hybrid_residual_v3.py` — окремий пілот із навчанням гібридної моделі CA + LSTM residual та двома seed.
- `REPORT_UA.md` — методика, справжні числові результати й обмеження.
- `results/` — CSV-таблиці, графіки, конфігурації.
- `net_5.pt`, `net_10.pt`, `net_20.pt` — попередні checkpoint'и v2 для перевірки шуму.

Для запуску: `python -m pip install -r requirements.txt`, потім `python robustness_v3.py --models . --output results` (за умови, що `summary.json` з v2 також є у цій папці), або команди з `REPORT_UA.md`.
