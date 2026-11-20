"""The local expert through the normal trainer entry point, finetune/finetune.py: the real argument parser, data
loading, OPAL wrapping, gate, trial and result files. Only the backbone is a toy and the model server a fake."""
import collections
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
import torch

import toy
from local_server import REVISION
from test_outer import events
from test_trainer_inputs import trainer

NOTE = "In our hands the GC-rich rows are the noisy ones."


class Tokenizer:
    pad_token_id, unk_token_id = 0, None

    def __call__(self, sequences, **kwargs):
        ids = torch.tensor([["ACGT".index(c) for c in s] for s in sequences])
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids))

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True)


class Backbone(torch.nn.Module):
    """The toy backbone behind the interface of a transformers model."""

    def __init__(self):
        super().__init__()
        self.core = toy.Backbone()

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        hidden = self.core(input_ids)
        return types.SimpleNamespace(hidden_states=hidden, last_hidden_state=hidden[-1])


@pytest.fixture
def train(tmp_path, monkeypatch):
    """Run finetune.py on a synthetic DNA task with one gate; returns the function and the run directory."""
    import smoke_backbone

    models, _ = trainer.registries()
    model = next(m for m in sorted(models) if models[m]["modality"] == "dna" and set(trainer.FLAGS.get(m, "?")) <= {
        "trust", "env"})
    frame = toy.data(1500).drop_duplicates("sequence").reset_index(drop=True)
    frame = frame.assign(series_id=[f"s{i}" for i in range(len(frame))], y_cls=frame.y,
                         split=["train"] * 1200 + ["val"] * 150 + ["test"] * (len(frame) - 1350)).drop(columns="y")
    folder = tmp_path / "curated" / "toy_fixture"
    folder.mkdir(parents=True)
    frame.to_parquet(folder / "task_classification.parquet")
    monkeypatch.setattr(trainer, "registries", lambda: (
        models, dict(toy_fixture=dict(modality="dna", tasks=["classification"]))))
    monkeypatch.setattr(trainer, "machine_paths", lambda: (folder.parent, collections.defaultdict(lambda: tmp_path)))

    def load(*args, **kwargs):
        torch.manual_seed(0)
        backbone = Backbone()
        return Tokenizer(), backbone, backbone

    monkeypatch.setattr(smoke_backbone, "load_backbone", load)
    out = tmp_path / "run"

    def run(*flags):
        monkeypatch.setattr(sys, "argv", [
            "finetune.py", "--model", model, "--task", "toy_fixture", "--cpu", "--det", "off", "--out", str(out),
            "--mode", "keep", "--panel-frac", "0.3", "--epochs", "1", "--lr", "3e-3", "--decisions", "2",
            "--check-every", "10", "--bio-policy", "expert", "--outer", "both", "--outer-decisions", "1",
            "--outer-steps", "8", "--no-test", *flags])
        trainer.main()

    return run, out


def test_the_trainer_runs_a_gate_with_a_local_expert_and_a_private_note(train, endpoint, tmp_path):
    run, out = train
    note = tmp_path / "note.txt"
    note.write_text(f"task: toy_fixture\nlabel: simulated institutional knowledge\n\n{NOTE}\n")
    run("--expert-backend", "local", "--expert-url", endpoint.url, "--expert-model", "toy-llm",
        "--expert-revision", REVISION, "--expert-shadow", "masked", "--expert-note", str(note),
        "--expert-request", '{"chat_template_kwargs": {"enable_thinking": false}}', "--expert-timeout", "5")
    informed, masked = (b["messages"][1]["content"] for b in endpoint.posts())   # one gate, one request per role
    assert NOTE in informed and NOTE not in masked and "toy_fixture" in informed and "task_T" in masked
    assert endpoint.posts()[0]["chat_template_kwargs"] == dict(enable_thinking=False)
    assert NOTE not in " ".join(sys.argv) and str(note) in sys.argv      # the path travels, the text does not

    proposals, trial, decision = (events(out, kind)[0] for kind in ("proposals", "trial", "decision"))
    menu = events(out, "table")[0]["menu"]
    said = proposals["informed"]
    assert said["ok"] and said["action"] == "set_cohort" and said["target"] == menu["set_cohort"][0] == "gc:low"
    branch = trial["branches"][0]                                       # the proposal was scored on the private rows
    assert (branch["action"], branch["target"], branch["by"]) == ("set_cohort", "gc:low", ["informed", "masked"])
    assert set(branch["horizons"]["8"]) == {"mean_d", "se", "passes"}
    assert decision["adopted"] is branch["horizons"]["8"]["passes"]     # the trial, not the expert, decides
    assert decision["proposals"]["informed"]["outcome"] in ("passed", "did not pass")

    result = json.loads((out / "result.json").read_text())
    about = result["biopal"]["expert"]
    assert about["backend"] == "local" and about["model"] == "toy-llm" and about["revision"] == REVISION
    assert about["serving_version"] == "0.0-fake" and about["calls"] == 2 == result["biopal"]["expert_calls"]
    assert about["failures"] == dict(informed=0, masked=0) and len(about["latency_seconds"]["each"]) == 2
    assert about["decoding"]["chat_template_kwargs"] == dict(enable_thinking=False)
    assert about["note"]["public"] is False and len(about["note"]["sha256"]) == 64 and "text" not in about["note"]
    assert result["biopal"]["expert_cost_usd"] == 0 and result["budget"] is None and result["test"] is None
    assert result["biopal"]["adopted"] == ([dict(gate=0, update=30, action="set_cohort", target="gc:low")]
                                           if decision["adopted"] else [])
    written = [p for p in out.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl", ".txt", ".md", ".py")]
    holding = sorted(p.name for p in written if NOTE in p.read_text())
    assert holding == [] and len(written) > 8                           # no file of the run holds the note's text


@pytest.mark.parametrize("flags, message", [
    (["--expert-budget", "1", "--expert-note", "note.txt"], "never sent to a hosted expert"),
    (["--expert-backend", "local", "--expert-model", "m", "--expert-revision", "r",
      "--expert-url", "http://localhost:8000/v1"], "loopback IP"),
    (["--expert-backend", "local", "--expert-model", "m", "--expert-revision", "r",
      "--expert-url", "https://api.example.com/v1"], "loopback IP"),
    (["--expert-backend", "local", "--expert-model", "m"], "needs --expert-model"),
    (["--expert-backend", "local", "--expert-revision", "r"], "needs --expert-model"),
    (["--expert-backend", "local", "--expert-model", "m", "--expert-revision", "r", "--expert-budget", "1"],
     "a local call costs nothing"),
    (["--expert-backend", "local", "--expert-model", "m", "--expert-revision", "r", "--expert-request", "[1]"],
     "must be a JSON object"),
    ([], "--expert-backend local, or --expert-script")])
def test_the_trainer_refuses_flags_that_would_misroute_a_brief(train, capsys, flags, message):
    run, out = train
    with pytest.raises(SystemExit):
        run(*flags)
    assert message in capsys.readouterr().err and not out.exists()


def test_the_trainer_stops_before_training_when_the_local_server_is_not_there(train):
    run, out = train
    with pytest.raises(RuntimeError, match="no model list from http://127.0.0.1:9/v1/models"):
        run("--expert-backend", "local", "--expert-url", "http://127.0.0.1:9/v1", "--expert-model", "toy-llm",
            "--expert-revision", REVISION, "--expert-timeout", "2")
    assert not (out / "train.jsonl").exists()                           # no update was run
