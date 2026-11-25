# POC v4: causal horizon-selection evaluation

Відтворюваний дослід адаптивного вибору горизонту на синтетичних траєкторіях із попередньо навченими LSTM моделями POC v2. Порівняння фіксованих і адаптивних політик та контроль matched-mix.

## Запуск

```bash
python -m pip install -r requirements.txt
python policy_evaluation_v4.py --models . --output results
python matched_mix_diagnostic.py --models . --output results
python make_figures.py
```

Результати та обмеження — `REPORT_UA.md`. Це дослідницький офлайн-код без інтеграції з БПЛА.
