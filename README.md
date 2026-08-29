# dslr

One-vs-all logistic regression written from scratch, applied to the Hogwarts house
classification dataset. Held-out accuracy averages about 98.7%. A 42 Abu Dhabi project.

## Run

```
python3 describe.py datasets/dataset_train.csv
python3 logreg_train.py datasets/dataset_train.csv          # writes weights.json
python3 logreg_predict.py datasets/dataset_test.csv weights.json   # writes houses.csv
```

Visualisations save to PNG:

```
python3 histogram.py
python3 scatter_plot.py
python3 pair_plot.py
```

## Test

```
python3 test_dslr.py
```

Checks `describe` against pandas and runs a 3-fold accuracy check. Takes about a
minute.

## Notes

- `datasets/` is extracted from the tracked `datasets.tgz` and is gitignored.
- Features are standardised with the training mean and standard deviation; missing
  cells are imputed with the training mean. Both scalers are saved in
  `weights.json` and reused at predict time.
- Accuracy sits at the ceiling for a linear one-vs-all classifier on this data.
  More iterations and dropping the collinear Astronomy and Defense features do not
  raise it. Nonlinear features would be the only lever.
- `tools.py` and `logreg.py` hold the shared statistics and model code.
- `PROGRESS.md` is the development log.
