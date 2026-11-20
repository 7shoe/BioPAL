"""Replay of saved gates: what each proposer reads, when scores are read, how choices are joined to the recorded
trials, and the edge cases of the summaries."""
import json
import shutil

import numpy as np
import pytest
import scipy.stats

import toy
from biopal import expert, replay
from local_server import REVISION
from test_expert import cli, flag   # noqa: F401  (cli: the fake claude binary)
from test_outer import events, script

NOTE = "In our hands the GC-rich rows are the noisy ones."
FORCED = dict(action="set_cohort", target="gc:high", force=True)


def audit(path, **plan):
    """A fixed-trajectory audit on the toy trainer: scripted proposals (none: keep), a masked shadow so that both
    briefs are saved, and every option scored after 4 and 8 updates at each of three gates."""
    args, t, bio = toy.make(path / "run", bio_policy="expert", expert_script=script(path, **plan),
                            expert_shadow="masked", outer_audit=[4, 8], outer_steps=8, outer_decisions=3)
    toy.train(args, t, bio)
    (path / "run" / "config.json").write_text(json.dumps(dict(recipe=dict(
        model="toy_model", task="toy_task", det="strict"))))
    return path / "run"


@pytest.fixture(scope="module")
def kept(tmp_path_factory):
    """The usual audit: every gate keeps, so ordinary weighting is in force throughout. Read-only."""
    return audit(tmp_path_factory.mktemp("kept"))


@pytest.fixture(scope="module")
def weighted(tmp_path_factory):
    """An audit whose first gate adopts a cohort: at gates 2 and 3 a return to ordinary weighting is legal."""
    return audit(tmp_path_factory.mktemp("weighted"), informed=[FORCED])


def served(endpoint, *more):
    return ["--local-model", "toy-llm", "--local-revision", REVISION, "--local-url", endpoint.url, *more]


def rewrite(run, change):
    """Edit the records of a run's log in place."""
    log = [json.loads(x) for x in (run / "outer.jsonl").read_text().splitlines()]
    for record in log:
        change(record)
    (run / "outer.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))


def text(body):
    return body["messages"][1]["content"]


def test_each_proposer_reads_the_saved_brief_without_history_and_only_one_reads_the_note(tmp_path, kept, endpoint, note):
    assert "  gate 1 after update 30: you proposed keep" in (kept / "expert" / "gate2_informed_brief.txt").read_text()
    result = replay.main([str(kept), "--out", str(tmp_path / "out"), "--note", str(note), *served(endpoint)])
    posts = endpoint.posts()
    assert len(posts) == 9                                              # three gates, three conditions, one call each
    for i in range(3):
        masked, informed, noted = (text(b) for b in posts[3 * i:3 * i + 3])
        for sent in (masked, informed, noted):
            assert "EARLIER GATES" not in sent and "you proposed" not in sent
            assert "\n\nACTIONS: choose exactly one\n" in sent
        assert "cohort_01" in masked and "gc:low" not in masked and NOTE not in masked and NOTE not in informed
        block = noted[noted.index(expert.NOTE_OPEN):noted.index(expert.NOTE_CLOSE) + len(expert.NOTE_CLOSE) + 2]
        assert NOTE in block and noted.replace(block, "") == informed   # same measurements, menu and order
        saved = (kept / "expert" / f"gate{i + 1}_informed_brief.txt").read_text()
        assert informed == replay.without_history(saved) and informed in saved.replace(
            saved[saved.index("EARLIER GATES"):saved.index("ACTIONS: choose")], "")
    for gate in result["gates"]:                                        # the masked id maps back to the real cohort
        assert {gate["picks"][p]["choice"] for p in ("local:masked", "local:informed", "local:informed+note")} == {
            "set_cohort gc:low"}
        assert gate["picks"]["local:informed"]["at"]["8"]["mean_d"] == gate["menu"]["8"]["set_cohort gc:low"]
    pair = result["by_horizon"]["8"]["paired"]["local:informed+note minus local:informed"]["by_run"]["run"]
    assert pair["gates"] == 3 and pair["disagreements"] == 0 and pair["mean_difference"] == 0
    assert result["example"]["gate"] == 1 and result["example"]["tie"] is True   # the note changed nothing: a tie
    about = result["proposers"]["local:informed+note"]
    assert about["calls"] == 3 and about["failures"] == 0 and about["model"] == "toy-llm" and about["seconds_max"] > 0
    assert about["revision"] == REVISION and about["serving_version"] == "0.0-fake" and about["decoding"]["temperature"] == 0
    shared = (tmp_path / "out" / "result.json").read_text() + (tmp_path / "out" / "proposals.jsonl").read_text()
    assert NOTE not in shared and "the first cohort on offer" not in shared   # no note, no reply text
    assert result["notes"]["toy_task"] == dict(task="toy_task", label="simulated institutional knowledge",
                                               public=False, sha256=expert.load_note(note)["sha256"],
                                               chars=len(expert.load_note(note)["text"]))
    local = tmp_path / "out" / "local" / "run"
    assert json.loads((local / "gate1_local_informed+note_reply.json").read_text())["choices"]
    assert NOTE not in "".join(f.read_text() for f in local.glob("*_brief.txt"))
    assert json.loads((tmp_path / "out" / "result.json").read_text())["primary_horizon"] == 8


def test_every_answer_is_on_disk_before_a_private_score_is_read(tmp_path, kept, endpoint, monkeypatch):
    seen, real = [], replay.trials

    def trials(path):
        seen.append([json.loads(x) for x in (tmp_path / "out" / "proposals.jsonl").read_text().splitlines()])
        return real(path)

    monkeypatch.setattr(replay, "trials", trials)
    replay.main([str(kept), "--out", str(tmp_path / "out"), *served(endpoint)])
    assert len(seen) == 1 and len(seen[0]) == 3 * 4                     # keep, the rule, masked and informed per gate
    assert {r["proposer"] for r in seen[0]} == {"always_keep", "worst_gap", "local:masked", "local:informed"}
    assert "trial" not in json.dumps(replay.checkpoints(kept)["gates"]) and len(endpoint.posts()) == 6
    with pytest.raises(FileExistsError):                                # a result is never written over
        replay.main([str(kept), "--out", str(tmp_path / "out")])


def test_keep_scores_zero_and_a_failed_call_is_not_a_chosen_keep(tmp_path, kept, endpoint):
    endpoint.mode = "keep"
    chosen = replay.main([str(kept), "--out", str(tmp_path / "a"), *served(endpoint)])
    endpoint.mode = "junk"
    failed = replay.main([str(kept), "--out", str(tmp_path / "b"), *served(endpoint)])
    for result, keeps, failures in ((chosen, 3, 0), (failed, 0, 3)):
        row = result["by_horizon"]["8"]["proposers"]["local:informed"]["by_run"]["run"]
        assert row == dict(gates=3, mean_d=0., changes=0, keeps=keeps, failures=failures,
                           mean_percentile=row["mean_percentile"])
        pick = result["gates"][0]["picks"]["local:informed"]
        assert pick["choice"] == "keep" and pick["at"]["8"]["mean_d"] == 0 and pick["at"]["4"]["se"] == 0
        assert pick["ok"] is (failures == 0) and pick["failure"] == (None if keeps else "malformed")
    assert failed["proposers"]["local:informed"]["failure_kinds"] == ["malformed"]
    assert (tmp_path / "b" / "local" / "run" / "gate1_local_informed_error.txt").read_text().startswith("ValueError")
    always = chosen["by_horizon"]["8"]["proposers"]["always_keep"]["by_run"]["run"]
    assert always["mean_d"] == 0 and always["keeps"] == 3
    gate = chosen["gates"][0]                                           # keep's place among the menu's scores
    scores = list(gate["menu"]["8"].values())
    assert gate["menu"]["8"]["keep"] == 0 and gate["picks"]["always_keep"]["at"]["8"]["percentile"] == pytest.approx(
        (sum(s > 0 for s in scores) + (sum(s == 0 for s in scores) - 1) / 2) / (len(scores) - 1))
    assert gate["uniform"]["8"] == pytest.approx(np.mean(scores))       # the expectation over every legal action
    assert chosen["by_horizon"]["8"]["proposers"]["uniform"]["by_run"]["run"]["mean_d"] == pytest.approx(
        np.mean([g["uniform"]["8"] for g in chosen["gates"]]))


def test_controls_are_reported_and_are_not_choices(tmp_path, kept, endpoint):
    endpoint.mode = "readout"                                           # the local model switches the head to block 2
    result = replay.main([str(kept), "--out", str(tmp_path / "out"), *served(endpoint)])
    for gate in result["gates"]:
        assert not any("refit" in x or "repeat" in x for x in gate["legal"]) and len(gate["legal"]) == 11
        control = gate["controls"]["8"]
        assert control["incumbent_repeat"] == 0 and control["refit_head_block"] == 4   # the repeat is exact on CPU
        pick = gate["picks"]["local:informed"]
        assert pick["choice"] == "set_readout block_2" and pick["at"]["8"]["mean_d"] == gate["menu"]["8"]["set_readout block_2"]
        assert pick["at"]["8"]["beyond_refit"] == pytest.approx(pick["at"]["8"]["mean_d"] - control["refit_head"])
        assert gate["picks"]["worst_gap"]["at"]["8"]["beyond_refit"] is None   # only a readout switch has it


def test_ordinary_weighting_is_audited_joined_and_checked_for_coverage(tmp_path, weighted, endpoint):
    menus = [t["menu"]["ordinary_weighting"] for t in events(weighted, "table")]
    assert menus == [False, True, True]
    for trial in events(weighted, "trial")[1:]:                         # the audit scored the way back
        assert [b["by"] for b in trial["branches"] if b["action"] == "ordinary_weighting"] == [["audit"]]
    endpoint.mode = "ordinary"
    result = replay.main([str(weighted), "--out", str(tmp_path / "full"), *served(endpoint)])
    first, second, third = result["gates"]
    assert "ordinary_weighting" not in first["legal"] and first["picks"]["local:informed"]["failure"] == "off_menu"
    for gate in (second, third):
        assert "ordinary_weighting" in gate["legal"] and "set_cohort gc:high" not in gate["legal"]
        pick = gate["picks"]["local:informed"]
        assert pick["choice"] == "ordinary_weighting" and pick["ok"]
        assert pick["at"]["8"]["mean_d"] == gate["menu"]["8"]["ordinary_weighting"] != 0
        assert gate["uniform"]["8"] == pytest.approx(np.mean(list(gate["menu"]["8"].values())))
    assert all(c["complete"] and not c["missing"] and not c["failed"] for c in result["coverage"])
    assert result["by_horizon"]["8"]["complete_gates"] == dict(run=[1, 2, 3])

    old = tmp_path / "old" / "run"                                      # a log from before: gate 2 never scored the
    shutil.copytree(weighted, old)                                      # way back, and one branch of gate 3 failed

    def damage(record):
        if record["event"] == "trial" and record["gate"] == 1:
            record["branches"] = [b for b in record["branches"] if b["action"] != "ordinary_weighting"]
        if record["event"] == "trial" and record["gate"] == 2:
            record["branches"] = [dict(action=b["action"], target=b["target"], by=b["by"], failed="Non-finite loss")
                                  if b["target"] == "gc:low" else b for b in record["branches"]]

    rewrite(old, damage)
    endpoint.requests.clear()
    result = replay.main([str(old), "--out", str(tmp_path / "partial"), *served(endpoint)])
    at8 = {c["gate"]: c for c in result["coverage"] if c["horizon"] == 8}
    assert at8[1]["complete"] and at8[1]["scored"] == at8[1]["legal"] == 11
    assert at8[2]["missing"] == ["ordinary_weighting"] and not at8[2]["failed"] and not at8[2]["complete"]
    assert at8[3]["failed"] == ["set_cohort gc:low"] and not at8[3]["missing"] and at8[3]["scored"] == 10
    assert result["gates"][1]["picks"]["local:informed"]["at"]["8"] is None   # chosen, but never scored
    assert result["gates"][1]["uniform"] == {} and result["gates"][2]["menu"]["8"]["set_cohort gc:low"] is None
    for horizon in ("4", "8"):                                          # every proposer on the same complete gate
        at = result["by_horizon"][horizon]
        assert at["complete_gates"] == dict(run=[1])
        assert {p: v["by_run"]["run"]["gates"] for p, v in at["proposers"].items()} == dict.fromkeys(
            ["always_keep", "worst_gap", "local:masked", "local:informed", "uniform"], 1)
        assert at["paired"]["local:informed minus local:masked"]["by_run"]["run"]["gates"] == 1


def test_a_hosted_proposer_has_one_budget_for_all_gates_and_reads_the_public_brief(tmp_path, kept, endpoint, note,
                                                                                   cli):
    hosted = ["--hosted-model", "claude-haiku-5-5", "--hosted-budget", "0.03"]
    with pytest.raises(SystemExit):                                     # the briefs' export has to be confirmed
        replay.main([str(kept), "--out", str(tmp_path / "no"), *hosted])
    assert not cli() and not (tmp_path / "no").exists()
    result = replay.main([str(kept), "--out", str(tmp_path / "out"), *hosted, "--exportable", "--note", str(note),
                          *served(endpoint)])
    calls = cli()                                                       # $0.012 a call: the third gate cannot be paid
    assert [flag(c, "--max-budget-usd") for c in calls] == ["0.0300", "0.0180"]
    assert {(flag(c, "--model"), flag(c, "--effort")) for c in calls} == {("claude-haiku-5-5", "low")}
    informed = [text(b) for b in endpoint.posts()][1::3]
    for call, local_text in zip(calls, informed):                       # the text the local model read, no note
        assert flag(call, "-p") == local_text and NOTE not in json.dumps(call) and "gc:low" in flag(call, "-p")
    picks = [g["picks"]["hosted:informed"] for g in result["gates"]]
    assert [p["ok"] for p in picks] == [True, True, False] and picks[2]["failure"] == "backend"
    assert picks[2]["choice"] == "keep" and picks[0]["choice"] == "set_cohort gc:low"
    about = result["proposers"]["hosted:informed"]
    assert about["calls"] == 3 and about["failures"] == 1 and about["budget_usd"] == .03
    assert about["cost_usd"] == pytest.approx(.024) and "budget used up" in about["stopped"]
    assert "hosted:informed minus local:informed" in result["by_horizon"]["8"]["paired"]


def test_a_log_without_recorded_names_is_replayed_only_when_the_brief_settles_them(tmp_path, kept, endpoint):
    def forget(record):
        if record["event"] == "schedule":
            del record["names"], record["expert"]

    old = tmp_path / "old" / "run"
    shutil.copytree(kept, old)
    rewrite(old, forget)
    result = replay.main([str(old), "--out", str(tmp_path / "out"), *served(endpoint)])
    assert [g["picks"]["local:masked"]["choice"] for g in result["gates"]] == ["set_cohort gc:low"] * 3
    run, brief = replay.checkpoints(old), (old / "expert" / "gate1_masked_brief.txt").read_text()
    names = replay.names_of(run, "masked", run["gates"][0], brief)
    recorded = replay.checkpoints(kept)["schedule"]["names"]["masked"]   # what the trainer logs since
    assert names.pop("all") == "all" and names == {c: recorded[c] for c in names}
    assert names["gc:low"] == "cohort_01" and len(names) == 8
    twice = brief.replace("| cohort_02 |", "| cohort_01 |")              # two cohorts under one displayed name
    with pytest.raises(ValueError, match="does not show which of its cohort names is which"):
        replay.names_of(run, "masked", run["gates"][0], twice)
    short = "\n".join(x for x in brief.splitlines() if not x.startswith("r05 |"))   # a row of the log is missing
    with pytest.raises(ValueError, match="does not show which"):
        replay.names_of(run, "masked", run["gates"][0], short)
    (old / "expert" / "gate1_masked_brief.txt").write_text(twice)
    with pytest.raises(ValueError, match="does not show which"):
        replay.main([str(old), "--out", str(tmp_path / "refused"), *served(endpoint)])
    other = brief.replace("On offer: cohort_01, cohort_02", "On offer: cohort_02, cohort_01")
    for source in (run, replay.checkpoints(kept)):                      # recorded names or not: the menu must match
        with pytest.raises(ValueError, match="does not offer the cohorts of the log's menu"):
            replay.names_of(source, "masked", source["gates"][0], other)
    with pytest.raises(ValueError, match="not a brief this version can replay"):
        replay.without_history("TASK\nno sections here")


def test_a_note_is_used_only_for_runs_of_its_task(tmp_path, kept, endpoint, note):
    other = tmp_path / "copy" / "other_run"
    shutil.copytree(kept, other)
    (other / "config.json").write_text(json.dumps(dict(recipe=dict(model="toy_model", task="another_task", det="strict"))))
    result = replay.main([str(kept), str(other), "--out", str(tmp_path / "out"), "--note", str(note), *served(endpoint)])
    sent = [text(b) for b in endpoint.posts()]
    assert len(sent) == 3 * 3 + 3 * 2 and sum(NOTE in s for s in sent) == 3 and all(NOTE not in s for s in sent[9:])
    assert [r["note"] for r in result["runs"]] == [expert.load_note(note)["sha256"], None]
    assert all("local:informed+note" not in g["picks"] for g in result["gates"] if g["run"] == "other_run")
    at = result["by_horizon"]["8"]
    pair = at["paired"]["local:informed+note minus local:informed"]
    assert pair["by_run"]["run"]["gates"] == 3 and pair["by_run"]["other_run"] == dict(
        gates=0, disagreements=0, mean_difference=None, per_gate=[])
    assert pair["by_task"] == dict(toy_task=dict(runs=1, mean=0.), another_task=dict(runs=0, mean=None))
    assert at["proposers"]["local:informed"]["by_task"]["another_task"]["runs"] == 1
    (tmp_path / "stray.txt").write_text("task: a_third_task\n\nNothing to do with these runs.\n")
    for notes in ([str(tmp_path / "stray.txt")], [str(note), str(note)]):   # no run of that task; two notes for one
        with pytest.raises(SystemExit):
            replay.main([str(kept), "--out", str(tmp_path / "refused"), "--note", *notes, *served(endpoint)])
    with pytest.raises(SystemExit):                                     # a note has no reader without a local model
        replay.main([str(kept), "--out", str(tmp_path / "refused"), "--note", str(note)])
    (other / "expert" / "gate2_masked_brief.txt").unlink()              # an audit made without the masked shadow
    with pytest.raises(SystemExit):
        replay.main([str(other), "--out", str(tmp_path / "refused"), *served(endpoint)])
    assert not (tmp_path / "refused").exists() and len(endpoint.posts()) == 15


def test_a_public_note_is_shown_with_the_example_gate(tmp_path, kept, endpoint, capsys):
    path = tmp_path / "public.txt"
    path.write_text(f"task: toy_task\nlabel: simulated institutional knowledge\nvisibility: public\n\n{NOTE}\n")
    result = replay.main([str(kept), "--out", str(tmp_path / "out"), "--note", str(path), "--example-gate", "2",
                          *served(endpoint)])
    out = capsys.readouterr().out
    assert result["notes"]["toy_task"]["text"] == NOTE and result["example"]["gate"] == 2
    for part in ("Example, chosen before any score was read: run gate 2 (update 80)",
                 f"note (simulated institutional knowledge): {NOTE}", "local:informed+note  set_cohort gc:low:",
                 "the note changed nothing at this gate", "3 gates placed, none dropped; 3 with a trial",
                 "Paired, same gates: local:informed+note minus local:informed", "run gate 1 (update 30) at 8: 11/11",
                 "Incumbent repeat: largest |loss difference| 0 over 6 scores"):
        assert part in out
    with pytest.raises(SystemExit):
        replay.main([str(kept), "--out", str(tmp_path / "no"), "--example-gate", "4"])


def row(cohort, gap, lo, measurement="baseline"):
    return dict(id="r", measurement=measurement, layer=None, cohort=cohort, n=50, gap=gap, lo=lo, hi=2 * gap - lo)


def test_worst_gap_rule_filters_on_the_interval_before_it_takes_the_largest_gap():
    menu = dict(set_cohort=["gc:low", "gc:mid", "gc:high"], ordinary_weighting=True, set_readout=[2])
    table = [row("all", .9, .5), row("gc:high", .5, -.1), row("gc:low", .2, .05), row("gc:mid", .1, .01),
             row("label:1", .8, .3),                                    # in force, so not on offer
             row("gc:mid", 2., 1., "generalization")]                   # not a baseline gap
    assert replay.worst_gap(table, menu) == ("set_cohort", "gc:low")    # gc:high is larger, but its interval holds 0
    assert replay.worst_gap([r for r in table if r["cohort"] != "gc:low"], menu) == ("set_cohort", "gc:mid")
    assert replay.worst_gap([row("gc:high", .5, 0.), row("gc:low", -.2, -.3)], menu) == replay.KEEP   # none above 0
    assert replay.worst_gap([], menu) == replay.KEEP and replay.worst_gap(table, dict(set_readout=[2])) == replay.KEEP
    tied = [row("gc:high", .3, .1), row("gc:mid", .3, .2), row("gc:low", .1, .05)]
    assert replay.worst_gap(tied, menu) == replay.worst_gap(tied[::-1], menu) == ("set_cohort", "gc:mid")


def test_spearman_uses_mean_ranks_for_ties_and_is_undefined_for_a_constant():
    assert replay.ranks([10, 20, 10, 30, 10]).tolist() == [2, 4, 2, 5, 2]
    rng = np.random.default_rng(0)
    for _ in range(20):
        x, y = rng.integers(0, 4, 12), rng.integers(0, 4, 12)          # many ties
        if len(set(x)) > 1 and len(set(y)) > 1:
            assert replay.spearman(x.tolist(), y.tolist()) == pytest.approx(scipy.stats.spearmanr(x, y).statistic)
    assert replay.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1) and replay.spearman([1, 2, 2], [5, 7, 7]) == pytest.approx(1)
    for x, y in (([0, 0, 0], [0, 0, 0]), ([1, 2, 3], [4, 4, 4]), ([2, 2], [1, 3]), ([1], [1]), ([], [])):
        assert replay.spearman(x, y) is None                            # not 1.0


def test_percentile_and_score():
    assert replay.percentile(-1., [-1., 0., .5, 2.]) == 1 and replay.percentile(2., [-1., 0., .5, 2.]) == 0
    assert replay.percentile(0., [0., 0., 0.]) == .5 and replay.percentile(0., [-1., 0., 0., 1.]) == pytest.approx(.5)
    assert replay.percentile(0., [0.]) is None
    branches = {("set_cohort", "a"): dict(horizons={"8": dict(mean_d=-.2, se=.1)}),
                ("set_cohort", "b"): dict(failed="Non-finite"),
                ("set_cohort", "c"): dict(horizons={"8": dict(mean_d=float("nan"), se=.1)})}
    assert replay.score(branches, replay.KEEP, 8) == (0., 0.) and replay.score(branches, ("set_cohort", "a"), 8) == (-.2, .1)
    assert all(replay.score(branches, action, h) is None for action, h in (
        (("set_cohort", "a"), 16), (("set_cohort", "b"), 8), (("set_cohort", "c"), 8), (("set_readout", 2), 8)))
    assert replay.legal(dict(set_cohort=["a"], ordinary_weighting=True, set_readout=[2])) == [
        replay.KEEP, ("set_cohort", "a"), ("ordinary_weighting", None), ("set_readout", 2)]
    assert replay.label(("set_readout", 2)) == "set_readout block_2" and replay.label(replay.KEEP) == "keep"
