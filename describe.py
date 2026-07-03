#!/usr/bin/env python3
"""describe.py <dataset.csv> — print stats for every numeric feature.

Reimplements pandas' describe from scratch (forbidden to use describe/mean/std/
percentile helpers). Same rows as pandas: Count, Mean, Std, Min, 25/50/75%, Max.
"""
import sys
import tools

STATS = [
    ('Count', tools.count),
    ('Mean', tools.mean),
    ('Std', tools.std),
    ('Min', tools.minimum),
    ('25%', lambda xs: tools.percentile(xs, 25)),
    ('50%', lambda xs: tools.percentile(xs, 50)),
    ('75%', lambda xs: tools.percentile(xs, 75)),
    ('Max', tools.maximum),
]


def main():
    if len(sys.argv) < 2:
        print('usage: describe.py <dataset.csv>')
        sys.exit(1)
    header, rows = tools.load(sys.argv[1])
    feats = tools.numeric_features(header)
    cols = {f: tools.column(header, rows, f) for f in feats}
    # keep only features that have at least one numeric value
    feats = [f for f in feats if tools.clean(cols[f])]

    widths = {f: max(len(f), 14) for f in feats}
    print(' ' * 6 + ' '.join(f'{f:>{widths[f]}}' for f in feats))
    for label, fn in STATS:
        cells = []
        for f in feats:
            try:
                cells.append(f'{fn(cols[f]):>{widths[f]}.6f}')
            except (ValueError, ZeroDivisionError):
                cells.append(f'{"NaN":>{widths[f]}}')
        print(f'{label:<6}' + ' '.join(cells))


if __name__ == '__main__':
    main()
