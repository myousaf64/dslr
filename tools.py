"""Shared helpers for dslr: CSV loading + hand-rolled statistics.

No stat library does the work here (no pandas.describe, np.mean, etc.); every
statistic is computed from elementary operations, per the subject.
"""
import csv
import math

NON_FEATURE = {'Index', 'Hogwarts House', 'First Name', 'Last Name',
               'Birthday', 'Best Hand'}


def load(path):
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r]
    return header, rows


def numeric_features(header):
    return [h for h in header if h not in NON_FEATURE]


def column(header, rows, name):
    """Return the column as floats, using None for empty cells."""
    j = header.index(name)
    out = []
    for row in rows:
        try:
            out.append(float(row[j]))
        except (ValueError, IndexError):
            out.append(None)
    return out


# --- statistics (operate on lists that may contain None) ---
def clean(xs):
    return [x for x in xs if x is not None]


def count(xs):
    return float(len(clean(xs)))


def mean(xs):
    c = clean(xs)
    return sum(c) / len(c)


def std(xs):
    c = clean(xs)
    m = sum(c) / len(c)
    return math.sqrt(sum((x - m) ** 2 for x in c) / (len(c) - 1))  # sample (ddof=1)


def minimum(xs):
    return min(clean(xs))


def maximum(xs):
    return max(clean(xs))


def percentile(xs, p):
    """Linear-interpolation percentile (numpy/pandas default), p in [0,100]."""
    c = sorted(clean(xs))
    n = len(c)
    if n == 1:
        return c[0]
    rank = (p / 100) * (n - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return c[lo]
    return c[lo] * (hi - rank) + c[hi] * (rank - lo)
