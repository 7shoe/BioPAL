import numpy as np
import pandas as pd

from biopal import panel


def frame(n=600, seed=0, regression=False):
    rng = np.random.default_rng(seed)
    seqs = ["".join(rng.choice(list("ACGT"), 40, p=[.4 - g, .1 + g, .1 + g, .4 - g]))
            for g in rng.uniform(0, .3, n)]
    seqs[:30] = seqs[30:60]                                    # exact duplicates
    y = rng.normal(size=n) if regression else rng.integers(0, 2, n)
    return pd.DataFrame(dict(sequence=seqs, y=y))


def test_split_is_disjoint_complete_and_stratified():
    f = frame()
    r = panel.split(f, "y", True, .15, seed=42)
    rows = np.concatenate(list(r.values()))
    assert sorted(rows) == list(range(len(f)))                 # disjoint and complete
    for c in (0, 1):                                           # each class gives about 7.5% to D and to A
        n = (f.y == c).sum()
        for k in "DA":
            assert abs((f.y[r[k]] == c).sum() / n - .075) < .03
    where = {s: k for k, v in r.items() for s in f.sequence[v]}
    assert all(where[s] == k for k, v in r.items() for s in f.sequence[v])   # duplicates stay together


def test_split_depends_only_on_seed_and_ignores_global_rng():
    f = frame(regression=True)
    np.random.seed(1)
    a = panel.split(f, "y", False, .15, seed=42)
    np.random.seed(2)
    b = panel.split(f, "y", False, .15, seed=42)
    c = panel.split(f, "y", False, .15, seed=43)
    assert panel.manifest(a) == panel.manifest(b) != panel.manifest(c)
    thirds = np.digitize(f.y, np.quantile(f.y, [1 / 3, 2 / 3]))
    assert all(abs((thirds[a["D"]] == t).mean() - 1 / 3) < .08 for t in range(3))


def test_composition_and_cohorts_use_fit_cutoffs():
    f = frame()
    comp = panel.composition(f.sequence.tolist(), "dna")
    assert comp.shape == (len(f), 64) and np.allclose(comp.sum(1), 1)
    fit = np.arange(300)
    masks, defs = panel.cohorts(f, "y", True, "dna", comp, fit)
    assert set(masks) == {f"{a}:{b}" for a in ("gc", "kmer_entropy") for b in ("low", "mid", "high")} | {
        "label:0", "label:1"}
    gc = f.sequence.map(lambda s: (s.count("G") + s.count("C")) / len(s)).to_numpy()
    lo = np.quantile(gc[fit], 1 / 3)
    assert np.array_equal(masks["gc:low"], gc <= lo)           # cutoff from the fit rows, applied to all rows
    assert (masks["gc:low"] | masks["gc:mid"] | masks["gc:high"]).all()
    assert defs["gc:high"][0] == "gc" and "{f}" in defs["gc:high"][1]


def test_protein_cohorts_annotation_and_masked_names():
    rng = np.random.default_rng(0)
    f = pd.DataFrame(dict(sequence=["".join(rng.choice(list(panel.AA), 30)) for _ in range(200)],
                          y=rng.normal(size=200), mutation_depth=rng.integers(1, 8, 200)))
    comp = panel.composition(f.sequence.tolist(), "protein")
    masks, defs = panel.cohorts(f, "y", False, "protein", comp, np.arange(200))
    assert {"mutation_depth:1-2", "mutation_depth:3-4", "mutation_depth:5+", "target:low",
            "hydrophobicity:high", "net_charge:mid"} <= set(masks)
    assert masks["mutation_depth:5+"].sum() == (f.mutation_depth >= 5).sum()
    real = panel.names(defs, "task_x", "protein", "esm", masked=False)
    masked = panel.names(defs, "task_x", "protein", "esm", masked=True)
    assert all(k == v for k, v in real.items()) and set(real) == set(masked)
    assert len(set(masked.values())) == len(masked)            # one opaque id per name
    assert not any(k in " ".join(masked.values()) for k in ("gc", "hydro", "charge", "mutation", "protein", "esm"))
