"""The trainer (finetune/finetune.py) refuses a run whose splits hold the same model input. Sequences that differ only
beyond --max-chars pass the full-length check and become one input once they are cut."""
import collections
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "trainer", Path(__file__).resolve().parents[2] / "finetune" / "finetune.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def frames(rows, max_chars=0):
    """Rows (split, sequence) as the trainer holds them once sequences are cut."""
    df = pd.DataFrame(rows, columns=["split", "sequence"])
    if max_chars:
        df["sequence"] = df["sequence"].str[:max_chars]
    return {s: df[df["split"].eq(s)].reset_index(drop=True) for s in ("train", "val", "test")}


def test_cut_sequences_can_be_one_input_in_two_splits():
    rows = [("train", "ACDA"), ("val", "GHIK"), ("test", "ACDC")]
    assert trainer.shared_inputs(frames(rows)) == {}                    # distinct at full length
    assert trainer.shared_inputs(frames(rows, 4)) == {}
    assert trainer.shared_inputs(frames(rows, 3)) == {("train", "test"): dict(inputs=1, rows=1, example="ACD")}


def test_shared_inputs_counts_distinct_inputs_and_the_rows_that_carry_them():
    rows = [("train", "ACDA"), ("train", "ACDB"), ("train", "GHIA"), ("val", "ACDE"), ("val", "KLMN"),
            ("test", "ACDF"), ("test", "ACDG"), ("test", "GHIB"), ("test", "PQRS")]
    assert trainer.shared_inputs(frames(rows, 3)) == {
        ("train", "val"): dict(inputs=1, rows=1, example="ACD"),
        ("train", "test"): dict(inputs=2, rows=3, example="ACD"),       # ACD twice and GHI once in test
        ("val", "test"): dict(inputs=1, rows=2, example="ACD")}
    assert trainer.shared_inputs(frames([("train", "ACD"), ("train", "ACD"), ("val", "GHI"), ("test", "KLM")])) == {}


class ReachedTheModel(Exception):
    """The data passed every check: the trainer went on to load the backbone."""


@pytest.fixture
def train(tmp_path, monkeypatch):
    """Run the trainer on a four-row protein task up to the point where it would load the backbone."""
    import smoke_backbone                                               # on the path once the trainer is imported

    models, _ = trainer.registries()
    model = next(m for m in sorted(models) if m in trainer.FLAGS and models[m]["modality"] == "protein")
    task = tmp_path / "curated" / "cut_fixture"
    task.mkdir(parents=True)
    rows = [("t1", "train", "ACDA", 1.), ("t2", "train", "GHIA", 2.), ("v", "val", "KLMN", 3.), ("e", "test", "acdc", 4.)]
    pd.DataFrame(rows, columns=["series_id", "split", "sequence", "y_reg"]).to_parquet(task / "task_regression.parquet")
    monkeypatch.setattr(trainer, "registries", lambda: (
        models, dict(cut_fixture=dict(modality="protein", tasks=["regression"]))))
    monkeypatch.setattr(trainer, "machine_paths", lambda: (task.parent, collections.defaultdict(lambda: tmp_path)))

    def stop(*args, **kwargs):
        raise ReachedTheModel

    monkeypatch.setattr(smoke_backbone, "load_backbone", stop)

    def run(*flags):
        monkeypatch.setattr(sys, "argv", ["finetune.py", "--model", model, "--task", "cut_fixture", "--cpu", "--det",
                                          "off", "--out", str(tmp_path / f"run{len(list(tmp_path.iterdir()))}"), *flags])
        trainer.main()

    return run


def test_the_trainer_refuses_a_cut_that_gives_two_splits_the_same_input(train):
    for flags in ([], ["--max-chars", "4"], ["--max-chars", "9"]):      # ACDA and acdc stay distinct
        with pytest.raises(ReachedTheModel):
            train(*flags)
    with pytest.raises(ValueError, match=r"--max-chars 3 gives two splits the same input: train/test share 1 "
                                         r"\(1 of 1 test rows, e\.g\. 'ACD'\)"):
        train("--max-chars", "3")
    with pytest.raises(SystemExit):                                     # a negative length is not a cut from the end
        train("--max-chars", "-1")
