#!/usr/bin/env python3
"""logreg_train.py <dataset_train.csv> — train OvA logistic regression, save weights.json."""
import sys
import json
import tools
import logreg

HOUSES = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
WEIGHTS_FILE = 'weights.json'


def main():
    if len(sys.argv) < 2:
        print('usage: logreg_train.py <dataset_train.csv>')
        sys.exit(1)
    header, rows = tools.load(sys.argv[1])
    feats = tools.numeric_features(header)
    means, stds = logreg.fit_scalers(header, rows, feats)

    X = logreg.build_X(header, rows, feats, means, stds)
    houses_col = [row[header.index('Hogwarts House')] for row in rows]
    weights = logreg.train_ova(X, houses_col, HOUSES)

    with open(WEIGHTS_FILE, 'w') as f:
        json.dump({'features': feats, 'means': means, 'stds': stds,
                   'houses': HOUSES, 'weights': weights}, f, indent=2)
    print(f'Trained on {len(rows)} rows, {len(feats)} features -> {WEIGHTS_FILE}')


if __name__ == '__main__':
    main()
