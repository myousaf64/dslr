"""Self-check for dslr. Run: python3 test_dslr.py

- cross-checks hand-rolled describe stats against pandas (test-only reference)
- holds out 20% of the train set and asserts classifier accuracy >= 98%
"""
import random
import pandas as pd
import tools
import logreg

TRAIN = 'datasets/dataset_train.csv'


def check_describe():
    header, rows = tools.load(TRAIN)
    df = pd.read_csv(TRAIN)
    for f in tools.numeric_features(header):
        if f not in df.columns:
            continue
        col = tools.column(header, rows, f)
        ref = df[f].describe()
        assert abs(tools.mean(col) - ref['mean']) < 1e-6, f
        assert abs(tools.std(col) - ref['std']) < 1e-6, f
        assert abs(tools.percentile(col, 25) - ref['25%']) < 1e-6, f
        assert abs(tools.percentile(col, 50) - ref['50%']) < 1e-6, f
        assert abs(tools.maximum(col) - ref['max']) < 1e-6, f
    print('describe: matches pandas reference')


def check_accuracy():
    """3-fold-style average over seeds; the model converges ~98.2% (linear ceiling).

    Averaged to avoid betting the 98% bar on one thin 320-row split. iters=1000
    since accuracy has already converged by then (keeps the check ~40s).
    """
    header, rows = tools.load(TRAIN)
    feats = tools.numeric_features(header)
    hi = header.index('Hogwarts House')
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

    accs = []
    for seed in (1, 2, 3):
        r = list(rows)
        random.seed(seed)
        random.shuffle(r)
        cut = int(len(r) * 0.8)
        tr, va = r[:cut], r[cut:]
        means, stds = logreg.fit_scalers(header, tr, feats)
        Xtr = logreg.build_X(header, tr, feats, means, stds)
        ytr = [x[hi] for x in tr]
        w = logreg.train_ova(Xtr, ytr, houses, iters=1000)
        Xva = logreg.build_X(header, va, feats, means, stds)
        yva = [x[hi] for x in va]
        preds = logreg.predict(Xva, w, houses)
        accs.append(sum(p == t for p, t in zip(preds, yva)) / len(yva))
    mean_acc = sum(accs) / len(accs)
    print(f'held-out accuracy per fold: {[round(a, 4) for a in accs]}, mean={mean_acc:.4f}')
    assert mean_acc >= 0.98, f'mean accuracy {mean_acc:.4f} below 0.98 target'


if __name__ == '__main__':
    check_describe()
    check_accuracy()
    print('OK: dslr checks pass')
