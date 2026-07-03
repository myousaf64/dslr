#!/usr/bin/env python3
"""pair_plot.py [dataset_train.csv] — scatter-plot matrix of all features.

Used to decide which features feed the logistic regression. Saves pair_plot.png.
"""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tools

HOUSES = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
COLORS = dict(zip(HOUSES, ['C0', 'C1', 'C2', 'C3']))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'datasets/dataset_train.csv'
    header, rows = tools.load(path)
    feats = tools.numeric_features(header)
    house_of = [r[header.index('Hogwarts House')] for r in rows]
    data = {f: tools.column(header, rows, f) for f in feats}
    pt_color = [COLORS[h] for h in house_of]

    n = len(feats)
    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n))
    for i, fi in enumerate(feats):
        for j, fj in enumerate(feats):
            ax = axes[i][j]
            if i == j:
                for h in HOUSES:
                    vals = [v for v, ho in zip(data[fi], house_of) if ho == h and v is not None]
                    ax.hist(vals, bins=15, alpha=0.5, color=COLORS[h])
            else:
                xs = [(data[fj][k], data[fi][k], pt_color[k]) for k in range(len(rows))
                      if data[fj][k] is not None and data[fi][k] is not None]
                ax.scatter([x[0] for x in xs], [x[1] for x in xs],
                           s=2, alpha=0.4, c=[x[2] for x in xs])
            if i == n - 1:
                ax.set_xlabel(fj, fontsize=5)
            if j == 0:
                ax.set_ylabel(fi, fontsize=5)
            ax.tick_params(labelsize=3)
    fig.tight_layout()
    fig.savefig('pair_plot.png', dpi=90)
    print('saved pair_plot.png')


if __name__ == '__main__':
    main()
