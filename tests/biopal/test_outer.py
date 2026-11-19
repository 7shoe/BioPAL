import json

import toy
from biopal import outer


def test_spaced_falls_on_checks_inside_the_run():
    assert outer.spaced(2132, 5, 50) == [350, 700, 1100, 1450, 1850]
    assert outer.spaced(2132, 1, 50) == [1100]
    assert outer.spaced(120, 5, 50) == [50]                    # a short run holds fewer distinct checks
    assert outer.spaced(214, 2, 50) == [50, 150]               # never in the last `every` updates
    assert outer.spaced(40, 3, 50) == []
    assert all(u % 10 == 0 and 10 <= u <= 190 for u in outer.spaced(200, 10, 10))


def test_measuring_for_opal_leaves_training_bitwise_unchanged(tmp_path):
    plain, _ = toy.train(*toy.make(tmp_path / "a", wrapped=False))
    keep, summary = toy.train(*toy.make(tmp_path / "b", bio_policy="fixed"))
    assert plain == keep                                       # eval passes on D and F change no update
    assert summary["opal_decisions"] and summary["opal_cost_usd"] == 0


def test_opal_sees_the_panel_record_once_per_decision(tmp_path):
    messages = []

    def backend(workspace, message, system):
        messages.append(message)
        return "keep"

    for policy, extra in (("off", []), ("fixed", ["baseline_gap", "generalization_gap"])):
        messages.clear()
        args, t, bio = toy.make(tmp_path / policy, backend=backend, bio_policy=policy, decisions=3)
        toy.train(args, t, bio)
        assert len(messages) == len(bio.decisions) == 3
        shown = [json.loads(line) for line in (tmp_path / policy / "outer.jsonl").read_text().splitlines()]
        shown = [r for r in shown if r["event"] == "shown_to_opal"]
        assert [r["update"] for r in shown] == bio.decisions
        for message, record in zip(messages, shown):
            for key in ["panel_loss", "readout_block"] + extra:
                assert message.count(key) == 1 and f"{record[key]:.4g}"[:5] in message
            assert ("baseline_gap" in message) == bool(extra)


# ---------------------------------------------------------------------------- gates and the matched trial
import re

import numpy as np
import pytest
import torch


def script(tmp_path, **plan):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "script.json").write_text(json.dumps(plan))
    return tmp_path / "script.json"


def events(path, kind):
    return [r for r in map(json.loads, (path / "outer.jsonl").read_text().splitlines()) if r["event"] == kind]


def score(bio, t, rule):
    """Replace the loss on A by the real loss plus rule(active cohort), to decide a trial's outcome in a test."""
    real = t.rows

    def rows(indices, layers=None):
        loss, feats = real(indices, layers)
        return (loss + rule(bio.cohort), feats) if indices is bio.s.A else (loss, feats)

    t.rows = rows


def cohort(name):
    return dict(action="set_cohort", target=name)


def test_gate_placement_respects_epochs_decisions_and_the_trial_horizon():
    five = [350, 700, 1100, 1450, 1850]
    assert outer.place(five, five, 64, 1066, 2132, 50) == (five, [])      # a decision at the gate itself is fine
    assert outer.place([1000], [], 128, 1066, 2132, 50) == ([1100], [])   # a 128-update window must not cross an epoch
    assert outer.place([1000], [], 64, 1066, 2132, 50) == ([1000], [])
    assert outer.place([300], [350], 64, 1066, 2132, 50) == ([350], [])   # nor hold an OPAL decision
    assert outer.place([300], [350, 450], 128, 1066, 2132, 50) == ([450], [])
    assert outer.place([300, 300], [], 64, 1066, 2132, 50) == ([300, 350], [])
    assert outer.place([2100], [], 64, 1066, 2132, 50) == ([], [2100])    # no room left: dropped, and reported
    assert outer.place([50], [], 8, 50, 200, 50) == ([], [50])            # never at an epoch's last update


def test_adoption_rule_win_loss_tie_and_non_finite():
    rng = np.random.default_rng(0)
    base = rng.random(400)
    noise = rng.normal(0, .05, 400)
    assert outer.verdict(base - .02 + noise, base)["passes"]              # falls by more than one standard error
    assert not outer.verdict(base + .02 + noise, base)["passes"]
    assert not outer.verdict(base - .001 + noise, base)["passes"]         # a fall inside the noise is not adopted
    tie = outer.verdict(base.copy(), base)
    assert tie["mean_d"] == 0 and not tie["passes"]
    assert not outer.verdict(np.where(np.arange(400) == 3, np.nan, base - 1), base)["passes"]


def test_weighted_gradient_matches_a_hand_computation(tmp_path):
    args, t, bio = toy.make(tmp_path)
    bio.set_cohort("gc:high")
    mask, fit = bio.s.cohorts["gc:high"], t.roles["F"]
    share = mask[fit].mean()
    assert abs(bio.W[fit].mean().item() - 1) < 1e-6                       # the mean weight over the fit set is 1
    inside, outside = fit[mask[fit]][:8], fit[~mask[fit]][:8]
    for indices in (np.concatenate([inside, outside]), outside):          # a mixed batch, and one without the cohort
        w = bio.weights(indices)
        want = torch.tensor([.8 + .2 / share if mask[i] else .8 for i in indices], dtype=torch.float32)
        assert torch.allclose(w, want)
        t.update(indices, None)                                           # the plain gradient of the next batch state
        t.optimizer.zero_grad()
        ids = torch.tensor([["ACGT".index(c) for c in s] for s in t.frame.sequence[indices]])
        y = torch.tensor(t.frame.y.to_numpy()[indices])
        each = torch.nn.functional.cross_entropy(t.net["head"](t.net["backbone"](ids)[-1].mean(1)), y, reduction="none")
        params = list(t.net.parameters())
        by_hand = torch.autograd.grad((each * want).sum() / len(indices), params)
        clip = min(1., 1. / (torch.sqrt(sum((g ** 2).sum() for g in by_hand)).item() + 1e-6))
        before = [p.detach().clone() for p in params]
        t.update(indices, w)
        assert all(torch.allclose(p.grad, g * clip, atol=1e-7) for p, g in zip(params, by_hand))
        assert any(not torch.equal(p, b) for p, b in zip(params, before))
    plain = torch.autograd.grad(each.mean(), params, retain_graph=True) if False else None   # (kept simple)
    assert bio.weights(outside).unique().tolist() == pytest.approx([.8])  # no cohort row: every weight is 1 - alpha
    bio.set_cohort(None)
    assert bio.weights(outside) is None


CLEAN = re.compile(r'(program_ms_per_update": )[0-9.]+|Timing:.*|Profile of one update:.*|wall_ms_per_update": [0-9.]+')


def editing_backend(messages):
    def backend(workspace, message, system):                 # decision 1 installs a new, stateful program version
        messages.append(CLEAN.sub("", message))
        if len(messages) == 1:
            with open(f"{workspace}/updater.py", "a") as f:
                f.write("\n# scripted no-op edit\n")
            return "edit"
        return "keep"
    return backend


@pytest.mark.parametrize("backend", ["keep", "edit"])
def test_a_rejected_proposal_leaves_the_run_bitwise_as_if_uninterrupted(tmp_path, backend):
    plain, seen = [], []
    reference, _ = toy.train(*toy.make(tmp_path / "a", backend=editing_backend(plain) if backend == "edit" else "keep"))
    args, t, bio = toy.make(tmp_path / "b", backend=editing_backend(seen) if backend == "edit" else "keep",
                            bio_policy="expert", expert_script=script(tmp_path, informed=[cohort("gc:high"), cohort("label:1")]))
    assert bio.gates == [30, 140] == bio.decisions                      # each gate opens right after an OPAL decision
    cap = getattr(t.optimizer, "max_decisions", None)
    calls_at_scoring = []
    score(bio, t, lambda active: 1. if active else 0.)                  # every candidate loses on A
    real = t.rows
    t.rows = lambda i, layers=None: (calls_at_scoring.append(len(seen)) if i is bio.s.A else None, real(i, layers))[1]
    losses, summary = toy.train(args, t, bio)
    assert losses == reference                                          # model, OPAL state, scheduler and RNG all rewound
    assert summary["adopted"] == [] and summary["final_config"] == dict(cohort=None)
    assert summary["opal_calls_in_trials"] == 0 and getattr(t.optimizer, "max_decisions", None) == cap
    trials = events(tmp_path / "b", "trial")
    assert [len(tr["branches"]) for tr in trials] == [1, 1]
    for tr, gate in zip(trials, bio.gates):                             # the incumbent branch is what training then does
        assert tr["incumbent"]["losses"] == losses[gate:gate + 8]
        assert tr["branches"][0]["losses"] != tr["incumbent"]["losses"] and not tr["branches"][0]["horizons"]["8"]["passes"]
    if backend == "edit":
        assert seen == plain and len(seen) == 2                         # OPAL's agent got the same messages: no extra
        assert not any("resum" in m for m in seen)                      # call and no note about a rewind
        assert calls_at_scoring and set(calls_at_scoring[:2]) == {1}    # no agent call inside the first gate's branches
        assert events(tmp_path / "b", "decision")[0]["opal_program_changed"] is False


def test_a_winning_proposal_is_adopted_persists_and_commits_the_branch_state(tmp_path):
    plan = dict(informed=[cohort("gc:high"), dict(action="ordinary_weighting")])
    args, t, bio = toy.make(tmp_path / "b", bio_policy="expert", expert_script=script(tmp_path, **plan))
    score(bio, t, lambda active: -1. if active else 0.)                 # weighting any cohort wins on A
    losses, summary = toy.train(args, t, bio)
    assert summary["adopted"] == [dict(gate=0, update=30, action="set_cohort", target="gc:high")]
    assert summary["final_config"] == dict(cohort="gc:high")           # gate 2 proposed going back, lost, so it stays
    first, second = events(tmp_path / "b", "trial")
    assert first["branches"][0]["losses"] == losses[30:38]              # training continues exactly as the winning branch
    assert first["incumbent"]["losses"] != losses[30:38]
    assert second["incumbent"]["losses"] == losses[140:148]
    tables = events(tmp_path / "b", "table")
    assert "gc:high" in tables[0]["menu"]["set_cohort"] and not tables[0]["menu"]["ordinary_weighting"]
    assert "gc:high" not in tables[1]["menu"]["set_cohort"] and tables[1]["menu"]["ordinary_weighting"]
    args, t, bio = toy.make(tmp_path / "c")                             # the same switch without any gate or trial
    reference, _ = toy.train(args, t, bio, switch={30: "gc:high"})
    assert losses == reference
    assert losses != toy.train(*toy.make(tmp_path / "d"))[0]


def test_tie_and_failed_candidates_keep_the_incumbent(tmp_path):
    reference, _ = toy.train(*toy.make(tmp_path / "a"))
    plan = script(tmp_path, informed=[cohort("gc:high"), cohort("no_such_cohort")])
    for case in ("tie", "update", "private"):
        args, t, bio = toy.make(tmp_path / case, bio_policy="expert", expert_script=plan)
        real_rows, real_update = t.rows, t.update
        if case == "tie":                                               # both branches score the same on A
            t.rows = lambda i, layers=None: (np.ones(len(i)), {}) if i is bio.s.A else real_rows(i, layers)
        elif case == "update":                                          # the candidate's training loss blows up

            def update(indices, weights=None):
                if weights is not None:
                    raise RuntimeError("Non-finite training loss")
                return real_update(indices, weights)
            t.update = update
        else:                                                           # the candidate's loss on A is not finite
            score(bio, t, lambda active: np.nan if active else 0.)
        losses, summary = toy.train(args, t, bio)
        assert losses == reference and summary["adopted"] == []
        branch = events(tmp_path / case, "trial")[0]["branches"][0]
        decisions = events(tmp_path / case, "decision")
        if case == "tie":
            assert branch["horizons"]["8"] == dict(mean_d=0., se=0., passes=False)
        else:
            assert "Non-finite" in branch["failed"] and decisions[0]["proposals"]["informed"]["outcome"] == "failed"
        second = decisions[1]["proposals"]["informed"]                  # an action that is not on offer: no branch
        assert second["ok"] is False and "unavailable" in second["error"] and len(events(tmp_path / case, "trial")) == 1


def test_a_non_finite_incumbent_fails_the_run(tmp_path):
    args, t, bio = toy.make(tmp_path, bio_policy="expert", expert_script=script(tmp_path, informed=[cohort("gc:high")]))
    score(bio, t, lambda active: 0. if active else np.nan)
    with pytest.raises(RuntimeError, match="incumbent branch failed"):
        toy.train(args, t, bio)
    assert events(tmp_path, "incumbent_failed")[0]["gate"] == 0


def test_shadow_proposal_is_scored_once_never_adopted_and_never_shown_as_rejected(tmp_path):
    plan = script(tmp_path, informed=[cohort("gc:high"), cohort("gc:low")], masked=[cohort("gc:high"), cohort("label:1")])
    args, t, bio = toy.make(tmp_path, bio_policy="expert", expert_shadow="masked", expert_script=plan)
    order = []
    score(bio, t, lambda active: {"label:1": -1., None: 0.}.get(active, 1.))   # only the shadow's second proposal wins
    real = t.rows

    def rows(indices, layers=None):
        if indices is bio.s.A:                                          # both proposals are on disk before A is read
            order.append(len(events(tmp_path, "proposals")))
        return real(indices, layers)

    t.rows = rows
    _, summary = toy.train(args, t, bio)
    first, second = events(tmp_path, "trial")
    assert [b["by"] for b in first["branches"]] == [["informed", "masked"]]          # the same action: one branch
    assert [b["by"] for b in second["branches"]] == [["informed"], ["masked"]]
    assert order == [1, 1, 2, 2, 2]                                     # one candidate + incumbent, then two + incumbent
    assert summary["adopted"] == [] and summary["final_config"] == dict(cohort=None)
    shadow = [h for h in bio.history if h["role"] == "masked"]
    assert [h["outcome"] for h in shadow] == ["did not pass", "passed"]  # its own trial result, not "not adopted"
    assert [h["outcome"] for h in bio.history if h["role"] == "informed"] == ["did not pass", "did not pass"]
    assert events(tmp_path, "proposals")[1]["masked"]["target"] == "label:1"
    assert events(tmp_path, "decision")[1]["proposals"]["masked"]["trial"]["passes"] is True


@pytest.mark.parametrize("classification", [True, False])
def test_audit_scores_every_option_and_a_repeat_of_the_incumbent(tmp_path, classification):
    reference, _ = toy.train(*toy.make(tmp_path / "a", classification=classification))
    args, t, bio = toy.make(tmp_path / "b", classification=classification, outer_audit=[4, 12], outer_steps=8)
    assert bio.horizon == 12 and bio.roles == []
    losses, summary = toy.train(args, t, bio)
    assert losses == reference and summary["adopted"] == []            # an audit never changes the run
    trial = events(tmp_path / "b", "trial")[0]
    offered = events(tmp_path / "b", "table")[0]["menu"]["set_cohort"]
    assert [b["target"] for b in trial["branches"] if b["action"] == "set_cohort"] == offered and len(offered) >= 6
    assert all(set(b["horizons"]) == {"4", "8", "12"} and len(b["losses"]) == 12 for b in trial["branches"])
    repeat = next(b for b in trial["branches"] if b["action"] == "incumbent_repeat")
    assert all(v["mean_d"] == 0 and v["se"] == 0 for v in repeat["horizons"].values())   # the trial's noise floor on CPU
    assert any(v["mean_d"] != 0 for b in trial["branches"] if b["target"] for v in b["horizons"].values())
