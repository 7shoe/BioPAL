"""Planted weaknesses must show up with the right sign on the right cohort, for both objectives."""
import types

import numpy as np
import pytest

from biopal import measure


def setting(classification, seed=0, n=3000):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n) if classification else rng.normal(size=n)
    weak = np.arange(n) % 3 == 0                              # the planted cohort
    s = types.SimpleNamespace(D=np.arange(0, n, 2), F=np.arange(1, n, 2), y=y, classification=classification,
                              num_out=2 if classification else 1, cohorts=dict(weak=weak, fine=~weak, tiny=np.arange(n) < 20))
    s.base = np.full(len(s.D), .5)
    signal = (2. * y - 1) if classification else y
    feats = {1: rng.normal(size=(n, 6)), 2: rng.normal(size=(n, 6))}
    feats[1][:, 0] += 3 * signal                              # layer 1 separates every row
    feats[2][:, 0] += 3 * signal * ~weak                      # layer 2 lost the signal on the weak cohort
    return rng, s, feats


def row(table, kind, cohort, layer=None):
    return next(r for r in table if (r["measurement"], r["cohort"], r["layer"]) == (kind, cohort, layer))


@pytest.mark.parametrize("classification", [True, False])
def test_planted_gaps_have_the_right_sign(classification):
    rng, s, feats = setting(classification)
    on_d = np.isin(np.arange(len(s.y)), s.D)

    def rows(idx, layers=()):   # the model: worse than the baseline on the weak cohort, and only on unseen rows
        loss = .3 + .01 * rng.random(len(idx)) + .5 * (s.cohorts["weak"][idx] & on_d[idx])
        return loss, {l: feats[l][idx] for l in layers}

    m = measure.checkpoint(rows, s, layers=(1, 2), current=2)
    t = m["table"]
    assert {r["cohort"] for r in t} == {"all", "weak", "fine"}             # the 20-row cohort is not measured
    assert row(t, "baseline", "weak")["lo"] > 0 > row(t, "baseline", "fine")["hi"]
    assert row(t, "generalization", "weak")["lo"] > 0
    assert row(t, "generalization", "fine")["lo"] < 0 < row(t, "generalization", "fine")["hi"]
    assert row(t, "layer", "weak", 1)["lo"] > 0                            # layer 1 reads the weak cohort better
    assert abs(row(t, "layer", "fine", 1)["gap"]) < .05
    assert all(r["lo"] <= r["gap"] <= r["hi"] and r["id"] for r in t)
    assert not any(r["layer"] == 2 for r in t)                             # no gap of the current layer to itself
    assert m["head_input"].shape == (len(s.F), 6) and abs(m["panel_loss"] - np.mean(rows(s.D)[0])) < .01


@pytest.mark.parametrize("classification", [True, False])
def test_probe_loss_matches_scikit_learn(classification):
    _, s, feats = setting(classification)
    X, y = feats[1], s.y
    p = measure.fit_linear(X[s.F], y[s.F], classification, s.num_out)
    Z = (X[s.D] - p["mean"]) / p["sd"]
    if classification:
        want = -np.log(p["sklearn"].predict_proba(Z)[np.arange(len(s.D)), y[s.D]])
    else:
        want = (p["sklearn"].predict(Z) - y[s.D]) ** 2
    assert np.allclose(measure.linear_loss(p, X[s.D], y[s.D], classification), want, atol=1e-9)


def test_a_class_missing_from_the_sample_keeps_head_shape():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(200, 4)), rng.integers(0, 2, 200) * 2       # classes 0 and 2 of three
    p = measure.fit_linear(X, y, True, 3)
    assert p["coef"].shape == (3, 4) and np.isfinite(measure.linear_loss(p, X, y, True)).all()
