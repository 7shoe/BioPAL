"""The expert call against a fake `claude` binary: what is sent, how answers are checked, budget, no retries."""
import json
import stat

import pytest

import toy
from biopal import expert, ledger
from test_outer import events, script

FAKE = r'''#!/usr/bin/env python3
import json, os, re, sys, time
args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps(args) + "\n")
mode, model, text = os.environ.get("FAKE_MODE", "ok"), args[args.index("--model") + 1], args[args.index("-p") + 1]
if mode == "sleep":
    time.sleep(30)
if mode == "garbage":
    print("Not logged in")
    sys.exit(1)
offered = re.search(r"set_cohort <cohort>.*On offer: (.*)", text).group(1).split(", ")
answer = dict(layer_evidence="block_2", evidence_rows=["r01"], action="set_cohort", target=offered[0],
              predicted_sign="lower", hypothesis="the first cohort on offer")
if mode == "junk":
    answer = dict(action="dance")
if mode == "unavailable":
    answer["target"] = "no_such_cohort"
if mode.startswith("force"):
    answer["force"] = True
out = dict(type="result", is_error=mode == "error", result="it failed" if mode == "error" else json.dumps(answer),
           total_cost_usd=float(os.environ.get("FAKE_COST", "0.012")),
           modelUsage={"claude-opus-5-5" if mode == "other_model" else model: {}})
if mode == "helper":
    out["modelUsage"] = {model: {"costUSD": .011}, "claude-haiku-5-5": {"costUSD": .001}}
if mode == "bad_structured":                 # the result text is a valid answer, structured_output is not
    out["structured_output"] = "see the result text"
elif mode not in ("text", "force_text"):
    out["structured_output"] = answer
print(json.dumps(out))
'''


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """A fake claude binary; returns a function that lists the argument vectors of the calls made so far."""
    exe = tmp_path / "claude"
    exe.write_text(FAKE)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("BIOPAL_CLAUDE", str(exe))
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "calls.jsonl"))
    return lambda: [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()] if (
        tmp_path / "calls.jsonl").exists() else []


def flag(call, name):
    return call[call.index(name) + 1]


def test_briefs_differ_only_in_names_and_background(tmp_path, cli):
    plan = script(tmp_path, informed=[dict(action="set_cohort", target="gc:high")],
                  masked=[dict(action="set_readout", target=2)])
    args, t, bio = toy.make(tmp_path, bio_policy="expert", expert_shadow="masked", expert_script=plan)
    seen = {}
    real = bio.expert.propose
    bio.expert.propose = lambda role, gate, text: (seen.setdefault((gate, role), text), real(role, gate, text))[1]
    toy.train(args, t, bio)
    for gate in (0, 1):                                                 # the second brief also carries history
        informed, masked = seen[gate, "informed"], seen[gate, "masked"]
        for word in ("gc", "entropy", "toy_task", "toy_model", "dna", "label", "G and C bases"):
            assert word in informed and word not in masked              # no biological name reaches the masked expert
        back = masked
        for name, shown in sorted(bio.names["masked"].items(), key=lambda kv: -len(kv[1])):
            back = back.replace(shown, name)
        strip = lambda text: [x for x in text.splitlines() if not x.startswith(("Background:", "  gate "))]   # noqa: E731
        assert strip(back) == strip(informed)                           # same structure, same numbers, same actions
        assert "| layer block_2 |" in masked and "block_2, block_3" in masked   # layer order is kept
    assert "you proposed set_cohort gc:high; trial:" in seen[1, "informed"]
    own = next(x for x in seen[1, "masked"].splitlines() if x.startswith("  gate 1"))
    assert "you proposed set_readout block_2; trial:" in own and "not adopted" not in own and "reject" not in own
    assert seen[1, "informed"].count("  gate ") == seen[1, "masked"].count("  gate ") == 1   # equal history depth


def test_live_expert_sends_one_capped_call_per_proposal_and_shares_branches(tmp_path, cli):
    book = tmp_path / "ledger.jsonl"
    args, t, bio = toy.make(tmp_path / "run", bio_policy="expert", expert_shadow="masked", expert_budget=1.,
                            ledger=book)
    _, summary = toy.train(args, t, bio)
    calls = cli()
    assert len(calls) == 4 == summary["expert_calls"] and summary["expert_cost_usd"] == pytest.approx(.048)
    for call in calls:
        assert flag(call, "--model") == "claude-haiku-5-5" and flag(call, "--effort") == "low"
        assert flag(call, "--tools") == "" and "--no-session-persistence" in call and "--fallback-model" not in call
        assert json.loads(flag(call, "--json-schema"))["required"] and flag(call, "--system-prompt") == expert.SYSTEM
    assert [flag(c, "--max-budget-usd") for c in calls] == ["1.0000", "0.9880", "0.9760", "0.9640"]   # what is left
    assert ledger.spent(book, "expert") == pytest.approx(.048) and ledger.spent(book, "opal") == 0
    first = events(tmp_path / "run", "proposals")[0]
    assert first["informed"]["target"] == first["masked"]["target"] == "gc:low"   # the masked id maps back
    assert first["informed"]["predicted_sign"] == "lower" and first["informed"]["evidence_rows"] == ["r01"]
    assert [b["by"] for b in events(tmp_path / "run", "trial")[0]["branches"]] == [["informed", "masked"]]
    decision = events(tmp_path / "run", "decision")[0]["proposals"]
    assert decision["informed"]["sign_correct"] in (True, False)
    folder = tmp_path / "run" / "expert"
    assert sorted(f.name for f in folder.iterdir())[:2] == ["gate1_informed_brief.txt", "gate1_informed_reply.json"]
    assert "cohort_01" in (folder / "gate1_masked_brief.txt").read_text()


def test_budget_is_a_ceiling_across_runs_and_exhaustion_goes_offline(tmp_path, cli):
    book = tmp_path / "ledger.jsonl"
    ledger.charge(book, "expert", .17, run="an earlier run")
    ledger.charge(book, "opal", 5., run="an earlier run")              # the other role's spend does not count
    args, t, bio = toy.make(tmp_path / "run", bio_policy="expert", expert_shadow="masked", expert_budget=.2, ledger=book)
    reference, _ = toy.train(*toy.make(tmp_path / "ref"))
    toy.train(args, t, bio)
    assert [flag(c, "--max-budget-usd") for c in cli()] == ["0.0300", "0.0180"]   # the third call is never made
    second = events(tmp_path / "run", "proposals")[1]
    assert second["informed"]["ok"] is False and "budget used up" in second["informed"]["error"]
    assert second["masked"]["action"] == "keep" and len(events(tmp_path / "run", "trial")) == 1
    assert ledger.spent(book, "expert") == pytest.approx(.194)


@pytest.mark.parametrize("mode, message", [("junk", "schema"), ("unavailable", "unavailable action"),
                                           ("garbage", "no JSON"), ("error", "reported an error"),
                                           ("sleep", "TimeoutExpired"), ("other_model", "billed to"),
                                           ("force_text", "unexpected field 'force'"),
                                           ("force", "unexpected field 'force'"),
                                           ("bad_structured", "schema: not an object")])
def test_a_failed_call_keeps_the_incumbent_and_is_not_retried(tmp_path, cli, monkeypatch, mode, message):
    monkeypatch.setenv("FAKE_MODE", mode)
    reference, _ = toy.train(*toy.make(tmp_path / "ref"))
    args, t, bio = toy.make(tmp_path / "run", bio_policy="expert", expert_budget=1., expert_timeout=1.)
    losses, summary = toy.train(args, t, bio)
    assert losses == reference and summary["adopted"] == [] and not events(tmp_path / "run", "trial")
    proposals = events(tmp_path / "run", "proposals")
    assert all(p["informed"]["ok"] is False and message in p["informed"]["error"] for p in proposals)
    assert len(cli()) == (1 if mode == "other_model" else 2)           # one call per gate; none after a wrong model
    assert events(tmp_path / "run", "decision")[0]["proposals"]["informed"]["outcome"] == "failed"


def test_a_small_helper_call_on_a_cheaper_model_does_not_switch_the_expert_off(tmp_path, cli, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "helper")
    args, t, bio = toy.make(tmp_path, bio_policy="expert", expert_budget=1., expert_model="claude-opus-5-5",
                            expert_effort="medium")
    toy.train(args, t, bio)
    assert all(p["informed"]["ok"] for p in events(tmp_path, "proposals")) and len(cli()) == 2
    assert {(flag(c, "--model"), flag(c, "--effort")) for c in cli()} == {("claude-opus-5-5", "medium")}


def test_an_answer_in_the_result_text_is_parsed(tmp_path, cli, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "text")
    args, t, bio = toy.make(tmp_path, bio_policy="expert", expert_budget=1.)
    toy.train(args, t, bio)
    assert events(tmp_path, "proposals")[0]["informed"]["ok"] is True


def test_check_accepts_only_what_the_menu_offers():
    menu = dict(set_cohort=["gc:high"], ordinary_weighting=False, set_readout=[2, 3])
    ok = dict(action="set_cohort", target="cohort_01", predicted_sign="lower")
    assert expert.check(ok, menu, {"cohort_01": "gc:high"}) == ("set_cohort", "gc:high")
    with pytest.raises(ValueError, match="unavailable action"):        # the masked expert never read this name
        expert.check(dict(ok, target="gc:high"), menu, {"cohort_01": "gc:high"})
    assert expert.check(dict(action="set_readout", target="block_3"), menu, {"block_3": 3}) == ("set_readout", 3)
    with pytest.raises(ValueError, match="unavailable action"):        # nor did any expert read a bare number
        expert.check(dict(action="set_readout", target=3), menu, {"block_3": 3})
    assert expert.check(dict(action="set_readout", target=3), menu, {"block_3": 3, 3: 3}) == ("set_readout", 3)
    assert expert.check(dict(action="keep", target=""), menu) == ("keep", None)
    for bad in (dict(action="ordinary_weighting"), dict(action="set_readout", target="block_4"),
                dict(action="set_cohort", target="gc:high", predicted_sign="up"), "keep", dict(action="rm")):
        with pytest.raises(ValueError):
            expert.check(bad, menu, {"block_3": 3})


GOOD = dict(layer_evidence="block_2", evidence_rows=["r01"], action="set_cohort", target="cohort_01",
            predicted_sign="lower", hypothesis="h")


@pytest.mark.parametrize("answer, message", [
    (dict(GOOD, force=True), "unexpected field 'force'"),
    ({k: v for k, v in GOOD.items() if k != "target"}, "missing field 'target'"),
    (dict(GOOD, hypothesis=None), "'hypothesis' is not a string"),
    (dict(GOOD, action=["keep"]), "'action' is not a string"),
    (dict(GOOD, action="dance"), "'action' is 'dance'"),
    (dict(GOOD, predicted_sign="up"), "'predicted_sign' is 'up'"),
    (dict(GOOD, evidence_rows=["a", "b", "c"]), "'evidence_rows' has more than 2 items"),
    (dict(GOOD, evidence_rows="r01"), "'evidence_rows' is not a list of strings"),
    (dict(GOOD, evidence_rows=[1]), "'evidence_rows' is not a list of strings"),
    ("keep", "not an object"), ([GOOD], "not an object"), (None, "not an object")])
def test_validate_enforces_the_schema_sent_to_the_cli(answer, message):
    assert expert.validate(dict(GOOD)) == GOOD
    with pytest.raises(ValueError, match="does not follow the schema.*" + message):
        expert.validate(answer)


def test_the_result_text_is_read_only_when_structured_output_is_absent():
    text = json.dumps(GOOD)
    assert expert.extract(dict(structured_output=GOOD, result="not read")) == GOOD
    assert expert.extract(dict(result=text)) == GOOD                    # a valid plain-JSON reply still works
    for malformed in ("see the result text", None, [GOOD], dict(GOOD, force=True), dict(action="keep")):
        with pytest.raises(ValueError, match="schema"):                 # a valid result text does not rescue it
            expert.extract(dict(structured_output=malformed, result=text))
    for result in ("```json\n" + text + "\n```", "I would keep.", "", None, json.dumps(dict(GOOD, force=True))):
        with pytest.raises(ValueError, match="schema"):
            expert.extract(dict(result=result))
    with pytest.raises(ValueError, match="schema"):
        expert.extract({})
