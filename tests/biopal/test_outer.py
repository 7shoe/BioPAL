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
