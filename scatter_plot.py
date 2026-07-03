#!/usr/bin/env python3
"""scatter_plot.py [dataset_train.csv] — the two most similar features.

Answers: which two features are similar? (Astronomy and Defense Against the
Dark Arts — near-perfectly correlated.) Saves scatter_plot.png.
"""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tools

A, B = 'Astronomy', 'Defense Against the Dark Arts'
HOUSES = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'datasets/dataset_train.csv'
    header, rows = tools.load(path)
    xa = tools.column(header, rows, A)
    xb = tools.column(header, rows, B)
    house_of = [r[header.index('Hogwarts House')] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 6))
    for h in HOUSES:
        pts = [(a, b) for a, b, ho in zip(xa, xb, house_of)
               if ho == h and a is not None and b is not None]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=6, alpha=0.5, label=h)
    ax.set_xlabel(A)
    ax.set_ylabel(B)
    ax.legend()
    fig.tight_layout()
    fig.savefig('scatter_plot.png', dpi=100)
    print('saved scatter_plot.png')


if __name__ == '__main__':
    main()
