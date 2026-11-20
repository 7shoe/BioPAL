"""BiOPAL data roles and cohorts

split()        train rows -> fit F (all gradients), diagnostic D (measurements), private A (trial scoring)
stream()       the train rows of every update when D and A return to training after the gates
composition()  features of the simple baseline model
cohorts()      named row groups: sequence-composition thirds, task annotations, label groups
names()        what the expert reads for each name: the real one (informed) or an opaque id (masked)
"""
import hashlib
import math

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
HYDROPATHY = [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,   # Kyte-Doolittle, in AA order
              1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]

ABOUT = dict(   # shown to the informed expert only
    gc="fraction of G and C bases",
    kmer_entropy="Shannon entropy of 3-mer usage; low values mean repetitive, low-complexity sequence",
    hydrophobicity="mean Kyte-Doolittle hydropathy of the residues",
    net_charge="number of K and R residues minus number of D and E residues",
    mutation_depth="number of amino-acid substitutions relative to the wild-type protein",
    label="the task's class label",
    target="the task's regression target, in thirds of the fit rows",
    nt_tasks_official_splice_sites_donors=(
        "Nucleotide Transformer benchmark. 600-nt genomic windows; the positive class contains a splice "
        "donor site, the exon-to-intron boundary recognized by the spliceosome."),
    proteingym_gfp_fluorescence=(
        "ProteinGym deep mutational scan of Aequorea victoria GFP (238 residues). Every sequence is a variant "
        "with one or more substitutions; the target is its fluorescence brightness."),
    fish_gue_ctcf=(
        "Fish_GUE benchmark. 800-nt genomic windows from fish genomes; the positive class contains a binding "
        "site of CTCF, the zinc-finger protein found at the boundaries of chromatin domains."),
    nt_tasks_official_h3k27ac=(
        "Nucleotide Transformer benchmark. 1000-nt human genomic windows; the positive class lies in a region "
        "that carries H3K27ac, the histone acetylation mark of active promoters and enhancers."),
    rna_dst_splice_site_acceptor=(
        "RNA downstream-task benchmark. 400-nt pre-mRNA windows from many species; the positive class has a "
        "splice acceptor site, the intron-to-exon boundary, at its centre."),
    rna_dst_mean_ribosome_load=(
        "RNA downstream-task benchmark. 5' untranslated regions of 25 to 100 nt placed before a reporter gene in "
        "a massively parallel assay in human cells; the target is the mean ribosome load of the reporter mRNA, "
        "a measure of how actively it is translated."),
    cds_melting_point=(
        "Protein-property benchmark. Proteins of several organisms; the target is the melting temperature in "
        "degrees C, measured by thermal proteome profiling in cells or lysates."),
)


def split(frame, label, classification, frac, seed):
    """Row positions in the train frame for F, D and A. D and A each take frac/2 of every stratum (class, or target
    third); rows that share an exact sequence stay in one set. Uses its own generator, not the training seeds."""
    y = frame[label].to_numpy()
    strata = y if classification else np.digitize(y, np.quantile(y, [1 / 3, 2 / 3]))
    group = frame.groupby("sequence", sort=False).ngroup().to_numpy()   # numbered by first appearance
    first = np.unique(group, return_index=True)[1]                      # one row per distinct sequence
    rng = np.random.default_rng([seed, 2026])
    role = np.zeros(len(first), int)                                    # per distinct sequence: 0 F, 1 D, 2 A
    for s in np.unique(strata[first]):
        g = rng.permutation(np.flatnonzero(strata[first] == s))
        k = int(round(len(g) * frac / 2))
        role[g[:k]], role[g[k:2 * k]] = 1, 2
    return {name: np.flatnonzero(role[group] == i) for i, name in enumerate("FDA")}


def stream(roles, n, batch_size, epochs, rng, until, every):
    """Train rows of every update when D and A return to training during the run: -> (batches, release).

    The run has as many updates as one without a panel, epochs * ceil(n / batch_size). Updates 1..release are full
    batches from back-to-back passes over F, each pass in a fresh order. release is `until` of the run moved up to
    a multiple of `every` (OPAL's check interval): it depends on nothing a gate does, so every arm of a comparison
    gets the same batches. After it each row is visited as often as it still lacks to `epochs` visits (D and A
    rows `epochs` times, F rows the rest of theirs), in one shuffled order, in batches of equal size. At the end
    every train row was visited `epochs` times, as without a panel. That needs until <= len(F) / n: held out
    for longer, F rows pass `epochs` visits before the release and the others get what is left of the run.
    Rarely a batch holds a row twice (where two passes meet, or two late visits of one row); it then counts twice.
    release is None, and all updates pass over F, when no update is left to return the rows in."""
    total, fit = epochs * math.ceil(n / batch_size), roles["F"]
    release = min(math.ceil(until * total / every) * every, total)

    def passes(updates):
        rows = np.concatenate([fit[rng.permutation(len(fit))]
                               for _ in range(math.ceil(updates * batch_size / len(fit)))])
        return [rows[s:s + batch_size] for s in range(0, updates * batch_size, batch_size)]

    before = passes(release)
    if release == total:
        return before, None
    # the rows of D and A alone are owed more visits than updates are left: no batch below is empty
    owed = np.repeat(np.arange(n), np.clip(epochs - np.bincount(np.concatenate(before), minlength=n), 0, None))
    owed = owed[rng.permutation(len(owed))][:(total - release) * batch_size]
    return before + np.array_split(owed, total - release), release


def manifest(roles):
    """Sizes and hashes of the three sets, to check that every arm of a seed trains on the same rows."""
    return {k: dict(n=len(v), sha256=hashlib.sha256(np.asarray(v, np.int64).tobytes()).hexdigest())
            for k, v in roles.items()}


def composition(seqs, modality):
    """Row-normalized counts per sequence: amino acids (protein) or 3-mers over ACGT (dna, rna; U counts as T)."""
    lut = np.full(256, -1)
    for i, c in enumerate(AA if modality == "protein" else "ACGT"):
        lut[ord(c)] = i
    if modality != "protein":
        lut[ord("U")] = 3
    width = 20 if modality == "protein" else 64
    out = np.zeros((len(seqs), width))
    for r, s in enumerate(seqs):
        v = lut[np.frombuffer(s.encode("ascii", "replace"), np.uint8)]
        if modality == "protein":
            k = v[v >= 0]
        else:
            k = (16 * v[:-2] + 4 * v[1:-1] + v[2:])[(v[:-2] >= 0) & (v[1:-1] >= 0) & (v[2:] >= 0)]
        out[r] = np.bincount(k, minlength=width)
    return out / np.maximum(out.sum(1, keepdims=True), 1)


def cohorts(frame, label, classification, modality, comp, fit):
    """{cohort: boolean mask over train rows} and {cohort: (feature, definition)}; cutoffs come from the fit rows.
    A definition is a template with {f} for the feature's displayed name."""
    seqs = frame["sequence"]
    if modality == "protein":
        feats = dict(hydrophobicity=comp @ np.array(HYDROPATHY),
                     net_charge=seqs.map(lambda s: s.count("K") + s.count("R") - s.count("D") - s.count("E"))
                     .to_numpy(float))
    else:
        feats = dict(gc=seqs.map(lambda s: (s.count("G") + s.count("C")) / len(s)).to_numpy(float),
                     kmer_entropy=-(comp * np.log2(np.where(comp > 0, comp, 1))).sum(1))
    if not classification:
        feats["target"] = frame[label].to_numpy(float)
    masks, defs = {}, {}

    def add(feature, level, mask, text):
        masks[f"{feature}:{level}"], defs[f"{feature}:{level}"] = np.asarray(mask), (feature, text)

    for f, v in feats.items():
        lo, hi = np.quantile(v[fit], [1 / 3, 2 / 3])
        add(f, "low", v <= lo, f"bottom third of {{f}} (at most {lo:.4g})")
        add(f, "mid", (v > lo) & (v <= hi), f"middle third of {{f}} (above {lo:.4g}, at most {hi:.4g})")
        add(f, "high", v > hi, f"top third of {{f}} (above {hi:.4g})")
    if "mutation_depth" in frame:
        d = frame["mutation_depth"].to_numpy()
        add("mutation_depth", "1-2", d <= 2, "{f} of 1 or 2")
        add("mutation_depth", "3-4", (d >= 3) & (d <= 4), "{f} of 3 or 4")
        add("mutation_depth", "5+", d >= 5, "{f} of 5 or more")
    if classification:
        for c in sorted(frame[label].unique()):
            add("label", str(c), frame[label].to_numpy() == c, f"{{f}} equal to {c}")
    return masks, defs


def names(defs, task, modality, model, masked):
    """Displayed name for the task, modality, model, every feature and every cohort. Masked: opaque ids that stay
    the same for the whole run; structure and numbers are kept, biological meaning is not."""
    real = [task, modality, model] + list(dict.fromkeys(f for f, _ in defs.values())) + list(defs)
    if not masked:
        return {r: r for r in real}
    features = list(dict.fromkeys(f for f, _ in defs.values()))
    out = {task: "task_T", modality: "symbol", model: "model_M"}
    out.update({f: f"feature_{i + 1}" for i, f in enumerate(features)})
    out.update({c: f"cohort_{i + 1:02d}" for i, c in enumerate(defs)})
    return out
