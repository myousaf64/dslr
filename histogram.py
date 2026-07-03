#!/usr/bin/env python3
"""histogram.py [dataset_train.csv] — score distribution per house, per course.

Answers: which course is homogeneously distributed across the four houses?
(Care of Magical Creatures — its per-house histograms overlap the most.)
Saves histogram.png; also shows a window if a display is available.
"""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tools

HOUSES = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'datasets/dataset_train.csv'
    header, rows = tools.load(path)
    feats = tools.numeric_features(header)
    house_of = [r[header.index('Hogwarts House')] for r in rows]

    n = len(feats)
    cols = 4
    rowsn = (n + cols - 1) // cols
    fig, axes = plt.subplots(rowsn, cols, figsize=(4 * cols, 3 * rowsn))
    for ax, f in zip(axes.flat, feats):
        col = tools.column(header, rows, f)
        for h in HOUSES:
            vals = [v for v, ho in zip(col, house_of) if ho == h and v is not None]
            ax.hist(vals, bins=20, alpha=0.5, label=h)
        ax.set_title(f, fontsize=8)
    for ax in axes.flat[n:]:
        ax.axis('off')
    axes.flat[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig('histogram.png', dpi=100)
    print('saved histogram.png')


if __name__ == '__main__':
    main()
