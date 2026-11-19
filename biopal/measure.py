"""BiOPAL checkpoint measurements: signed loss gaps per cohort with 95% intervals. Positive means worse.

baseline        model loss - loss of a simple model on sequence composition, on the diagnostic rows
generalization  loss on diagnostic rows - loss on fit rows
layer           loss of a linear probe on the layer that feeds the task head - loss of the same probe on another layer

Losses are the training loss per row: cross-entropy, or squared error on the standardized target.
"""
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

MIN_ROWS = 32   # a cohort is measured, and can be proposed, only with this many rows in both samples


def fit_linear(X, y, classification, num_out):
    """Standardize X and fit a linear model (logistic or ridge). The result is written as weights of a task head:
    coef (num_out, d) and intercept (num_out,) on the standardized input."""
    mean, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mean) / sd
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if classification:
            m = LogisticRegression(C=1.0, max_iter=2000).fit(Z, y)
            coef, icpt = np.zeros((num_out, X.shape[1])), np.full(num_out, -30.)   # a class without rows: never
            if len(m.classes_) == 2:   # one logit in scikit-learn: two logits with the same probabilities
                coef[m.classes_], icpt[m.classes_] = np.concatenate([-m.coef_, m.coef_]) / 2, np.concatenate(
                    [-m.intercept_, m.intercept_]) / 2
            else:
                coef[m.classes_], icpt[m.classes_] = m.coef_, m.intercept_
        else:
            m = Ridge(alpha=1.0).fit(Z, y)
            coef, icpt = m.coef_[None], np.array([m.intercept_])
    return dict(mean=mean, sd=sd, coef=coef, intercept=icpt, sklearn=m)


def linear_loss(p, X, y, classification):
    """Per-row loss of a fitted linear model."""
    out = ((X - p["mean"]) / p["sd"]) @ p["coef"].T + p["intercept"]
    if not classification:
        return (out[:, 0] - y) ** 2
    out = out - out.max(1, keepdims=True)
    return -(out - np.log(np.exp(out).sum(1, keepdims=True)))[np.arange(len(y)), y]


def interval(mean, se):
    return dict(gap=float(mean), lo=float(mean - 1.96 * se), hi=float(mean + 1.96 * se))


def paired(d):
    """Mean of per-row differences with its 95% interval."""
    return interval(d.mean(), d.std(ddof=1) / np.sqrt(len(d)))


def unpaired(a, b):
    return interval(a.mean() - b.mean(), np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def checkpoint(rows, s, layers=(), current=None):
    """Measure the model as it is now.

    rows(indices, layers) -> (per-row loss, {layer: pooled output}) in evaluation mode
    s        the fixed inputs: sample rows s.D and s.F, labels s.y, cohort masks s.cohorts, baseline loss s.base
             (aligned with s.D), s.classification, s.num_out
    layers   hidden states to probe; `current` is the one that feeds the head now

    Returns the gap table, the fitted probes (a probe can become the task head), the pooled outputs of `current`
    on the fit sample, and the mean loss on D.
    """
    loss_d, feat_d = rows(s.D, layers)
    loss_f, feat_f = rows(s.F, layers)
    probes = {l: fit_linear(feat_f[l], s.y[s.F], s.classification, s.num_out) for l in layers}
    probe_loss = {l: linear_loss(probes[l], feat_d[l], s.y[s.D], s.classification) for l in layers}
    table = []
    for name, mask in [("all", np.ones(len(s.y), bool))] + list(s.cohorts.items()):
        d, f = mask[s.D], mask[s.F]
        if min(d.sum(), f.sum()) < MIN_ROWS:
            continue
        found = [("baseline", None, paired(loss_d[d] - s.base[d])),
                 ("generalization", None, unpaired(loss_d[d], loss_f[f]))]
        found += [("layer", l, paired(probe_loss[current][d] - probe_loss[l][d])) for l in layers if l != current]
        for kind, layer, gap in found:
            table.append(dict(id=f"r{len(table) + 1:02d}", measurement=kind, layer=layer, cohort=name,
                              n=int(d.sum()), **gap))
    return dict(table=table, probes=probes, head_input=feat_f.get(current), panel_loss=float(loss_d.mean()),
                fit_loss=float(loss_f.mean()))
