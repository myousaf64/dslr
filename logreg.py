"""One-vs-all logistic regression core, hand-written (no ML library).

Features are standardised with train-set mean/std; missing cells are imputed
with the train-set mean. A bias term (1.0) is prepended to every sample.
"""
import math
import tools


def sigmoid(z):
    if z < 0:                       # numerically stable both ways
        e = math.exp(z)
        return e / (1 + e)
    return 1 / (1 + math.exp(-z))


def build_X(header, rows, feats, means, stds):
    """Standardised design matrix with bias, imputing NaN with feature mean."""
    idx = {f: header.index(f) for f in feats}
    X = []
    for row in rows:
        aug = [1.0]
        for f in feats:
            try:
                v = float(row[idx[f]])
            except (ValueError, IndexError):
                v = means[f]
            aug.append((v - means[f]) / stds[f])
        X.append(aug)
    return X


def train_binary(X, y, lr, iters):
    n = len(X[0])
    theta = [0.0] * n
    m = len(X)
    for _ in range(iters):
        grad = [0.0] * n
        for xi, yi in zip(X, y):
            h = sigmoid(sum(t * x for t, x in zip(theta, xi))) - yi
            for j in range(n):
                grad[j] += h * xi[j]
        theta = [t - lr * g / m for t, g in zip(theta, grad)]
    return theta


def train_ova(X, houses_col, houses, lr=1.0, iters=2000):
    weights = {}
    for house in houses:
        y = [1.0 if h == house else 0.0 for h in houses_col]
        weights[house] = train_binary(X, y, lr, iters)
    return weights


def predict(X, weights, houses):
    out = []
    for xi in X:
        scores = {h: sigmoid(sum(t * x for t, x in zip(weights[h], xi))) for h in houses}
        out.append(max(houses, key=lambda h: scores[h]))
    return out


def fit_scalers(header, rows, feats):
    means, stds = {}, {}
    for f in feats:
        col = tools.column(header, rows, f)
        means[f] = tools.mean(col)
        stds[f] = tools.std(col)
    return means, stds
