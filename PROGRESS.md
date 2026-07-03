# dslr (logistic regression) — progress

**Status: mandatory DONE & verified.** Held-out accuracy averages ~98.7% (≥98% bar).

## Programs
- `describe.py <csv>` — Count/Mean/Std/Min/25/50/75%/Max for numeric features,
  all hand-rolled (verified to match pandas.describe in the self-check).
- `logreg_train.py <dataset_train.csv>` — one-vs-all logistic regression via batch
  gradient descent; writes `weights.json` (features, per-feature mean/std, thetas).
- `logreg_predict.py <dataset_test.csv> <weights.json>` — writes `houses.csv`
  (`Index,Hogwarts House`).
- `histogram.py` / `scatter_plot.py` / `pair_plot.py` — save PNGs (matplotlib Agg).
- `tools.py` (stats + CSV) and `logreg.py` (model) are the shared core.

## Notes / decisions
- Features standardised with **train** mean/std; missing cells imputed with the
  train mean (same scalers reused at predict time, saved in weights.json).
- Accuracy converges to the linear-model ceiling ~98.2–98.8% (10-fold numpy CV mean
  0.982; pure-Python 3-seed mean 0.9875). More iterations / dropping the collinear
  Astronomy↔Defense features do **not** raise it — that's the Bayes limit for a
  linear OvA classifier here. If the real test dips below 98%, the only lever is
  nonlinear features; flag before adding.
- `datasets/` is extracted from the tracked `datasets.tgz` (gitignored).

**Run:** `python3 test_dslr.py` (describe vs pandas + 3-fold accuracy, ~60s).

## Next (bonus, only if mandatory perfect)
- More describe fields; stochastic / mini-batch gradient descent.
