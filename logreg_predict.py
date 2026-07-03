#!/usr/bin/env python3
"""logreg_predict.py <dataset_test.csv> <weights.json> — write houses.csv."""
import sys
import json
import csv
import tools
import logreg

OUT_FILE = 'houses.csv'


def main():
    if len(sys.argv) < 3:
        print('usage: logreg_predict.py <dataset_test.csv> <weights.json>')
        sys.exit(1)
    header, rows = tools.load(sys.argv[1])
    with open(sys.argv[2]) as f:
        model = json.load(f)

    X = logreg.build_X(header, rows, model['features'], model['means'], model['stds'])
    preds = logreg.predict(X, model['weights'], model['houses'])

    with open(OUT_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Index', 'Hogwarts House'])
        for i, house in enumerate(preds):
            w.writerow([i, house])
    print(f'Wrote {len(preds)} predictions -> {OUT_FILE}')


if __name__ == '__main__':
    main()
