"""Readout switch: the head reads an earlier block through a probe-fitted head; blocks above it stay and do not move."""
import copy
import json

import numpy as np
import pytest
import torch

import toy
from biopal import measure
from test_outer import events, script


def warm(tmp_path, updates=30, **kw):
    """A toy run after some real updates, so the optimizer holds momentum and decays weights."""
    args, t, bio = toy.make(tmp_path, **kw)
    t.net.train()
    fit = t.roles["F"]
    for i in range(updates):
        t.update(fit[16 * i:16 * i + 16])
    return args, t, bio, fit


def upper(t, layer):
    """Parameters of the blocks above `layer` and of the final norm: bypassed when the head reads `layer`."""
    b = t.net["backbone"]
    return [p for block in b.blocks[layer:] for p in block.parameters()] + list(b.norm.parameters())


@pytest.mark.parametrize("classification", [True, False])
def test_installed_head_reproduces_the_fitted_probe(tmp_path, classification):
    args, t, bio, _ = warm(tmp_path, classification=classification)
    assert bio.layers == [2, 3, 4] and bio.block() == 4
    m = measure.checkpoint(t.rows, bio.s, bio.layers, bio.block())
    before = float(torch.as_tensor(m["head_input"]).pow(2).mean().sqrt())
    checkpoint = copy.deepcopy(t.net.state_dict())
    for layer in (2, 3, 4):                                            # 4: the same block, head refitted
        t.net.load_state_dict(checkpoint)                              # a probe is installed at the state it was fitted on
        bio.install(layer, m)
        assert bio.block() == layer == int(t.net["readout"].layer)
        loss, feats = t.rows(bio.s.D, [layer])                         # the model, read out through the new head
        probe, y = m["probes"][layer], bio.s.y[bio.s.D]
        Z = (feats[layer] - probe["mean"]) / probe["sd"]               # the probe's own standardization
        with torch.no_grad():
            out = t.net["head"](t.net["readout"](torch.as_tensor(feats[layer], dtype=torch.float32)))
        if classification:
            assert np.allclose(out.softmax(-1).numpy(), probe["sklearn"].predict_proba(Z), atol=1e-5)
            want = -np.log(probe["sklearn"].predict_proba(Z)[np.arange(len(y)), y])
        else:
            assert np.allclose(out[:, 0].numpy(), probe["sklearn"].predict(Z), atol=1e-4)
            want = (probe["sklearn"].predict(Z) - y) ** 2
        assert np.allclose(loss, want, atol=1e-4)                      # per-row loss equals the probe's
        _, fit_feats = t.rows(bio.s.F, [layer])
        with torch.no_grad():
            size = t.net["readout"](torch.as_tensor(fit_feats[layer], dtype=torch.float32)).pow(2).mean().sqrt()
        assert float(size) == pytest.approx(before, rel=1e-3)          # the head's input keeps its overall size
        assert not torch.allclose(t.net["readout"].mu, torch.zeros(16))


@pytest.mark.parametrize("wrapped", [True, False])
def test_bypassed_blocks_stay_unchanged_under_momentum_and_weight_decay(tmp_path, wrapped):
    args, t, bio, fit = warm(tmp_path, wrapped=wrapped)
    if bio is None:                                                    # plain AdamW: borrow the outer level's install
        _, _, bio, _ = warm(tmp_path / "helper")
        bio.t = t
    base = t.optimizer if not wrapped else t.optimizer.user_optimizers[0]
    state = base.state[upper(t, 2)[0]]
    assert state["exp_avg"].abs().sum() > 0 and base.defaults["weight_decay"] > 0   # momentum and decay are live
    bio.install(2, measure.checkpoint(t.rows, bio.s, bio.layers, bio.block()))
    held = [p.detach().clone() for p in upper(t, 2)]
    lower = [p.detach().clone() for p in t.net["backbone"].blocks[0].parameters()]
    head = t.net["head"].weight.detach().clone()
    for i in range(5):                                                 # real optimizer steps
        t.update(fit[16 * i:16 * i + 16])
    assert all(p.grad is None for p in upper(t, 2))                    # not in the forward pass: no gradient at all
    assert all(torch.equal(p, h) for p, h in zip(upper(t, 2), held))   # and bitwise where they were
    assert all(not torch.equal(p, l) for p, l in zip(t.net["backbone"].blocks[0].parameters(), lower))
    assert not torch.equal(t.net["head"].weight, head)

    saved = copy.deepcopy(t.net.state_dict())                          # what best.pt would hold under this route
    with torch.no_grad():
        t.net.eval()
        x = torch.randint(0, 4, (8, 24))
        want = t.net["head"](t.net["readout"](t.net["backbone"](x)[2].mean(1)))
        t.net.train()

    bio.install(4, measure.checkpoint(t.rows, bio.s, bio.layers, bio.block()))   # switch back to the final block
    assert bio.block() == 4
    t.update(fit[:16])
    assert all(p.grad is not None for p in upper(t, 2))                # the upper blocks train again
    assert any(not torch.equal(p, h) for p, h in zip(upper(t, 2), held))

    t.net.load_state_dict(saved)                                       # restoring the checkpoint restores its route
    assert bio.block() == 2
    assert all(torch.equal(p, h) for p, h in zip(upper(t, 2), held))
    assert torch.equal(t.rows(np.arange(8))[1].get(0, torch.empty(0)), torch.empty(0))   # rows still runs
    t.net.eval()
    with torch.no_grad():
        route = int(t.net["readout"].layer)
        got = t.net["head"](t.net["readout"](t.net["backbone"](x)[route].mean(1)))
    assert torch.equal(got, want)


def test_the_head_reads_only_the_chosen_block(tmp_path):
    args, t, bio, _ = warm(tmp_path)
    bio.install(2, measure.checkpoint(t.rows, bio.s, bio.layers, bio.block()))
    before = t.rows(bio.s.D)[0]
    with torch.no_grad():
        for p in upper(t, 2):
            p.add_(1.)                                                 # scramble everything above block 2
    assert np.array_equal(t.rows(bio.s.D)[0], before)


def test_forced_switch_through_a_gate_and_back(tmp_path):
    seen = []
    plan = script(tmp_path, informed=[dict(action="set_readout", target=2, force=True),
                                      dict(action="set_readout", target=4, force=True)])
    args, t, bio = toy.make(tmp_path, backend=lambda w, m, s: (seen.append(m), "keep")[1], decisions=3,
                            bio_policy="expert", expert_script=plan)
    losses, summary = toy.train(args, t, bio)
    assert [a["target"] for a in summary["adopted"]] == [2, 4] and summary["final_config"]["readout_block"] == 4
    first, second = events(tmp_path, "trial")
    assert first["branches"][0]["losses"] == losses[30:38]             # the committed updates are the branch's
    tables = events(tmp_path, "table")
    assert tables[0]["menu"]["set_readout"] == [2, 3] and tables[1]["menu"]["set_readout"] == [3, 4]
    assert tables[1]["config"]["readout_block"] == 2
    layer_rows = [r for r in tables[1]["table"] if r["measurement"] == "layer"]
    assert {r["layer"] for r in layer_rows} == {3, 4}                  # gaps are relative to the block in use
    checks = events(tmp_path, "bypassed")
    assert [c["unchanged"] for c in checks] == [True, True] and checks[0]["readout_block"] == 2
    assert checks[0]["idle_tensors"] == 6 and checks[1]["idle_tensors"] == 0   # 2 blocks + norm; none after going back
    assert summary["bypassed_unchanged"] is True
    shown = [r["readout_block"] for r in events(tmp_path, "shown_to_opal")]
    assert bio.decisions == [30, 80, 140] and bio.gates == [30, 140]   # OPAL decides before the gate at its update
    assert shown == [4, 2, 2] and '"readout_block": 2' in seen[1]      # OPAL is told which block feeds the head


def test_readout_menu_follows_outer_flag_and_audit_tries_layers(tmp_path):
    args, t, bio = toy.make(tmp_path / "a", outer="cohort", outer_audit=[4])
    toy.train(args, t, bio)
    assert "set_readout" not in events(tmp_path / "a", "table")[0]["menu"]
    args, t, bio = toy.make(tmp_path / "b", outer="readout", outer_audit=[4])
    reference, _ = toy.train(*toy.make(tmp_path / "c"))
    losses, _ = toy.train(args, t, bio)
    assert losses == reference                                         # trying a switch in a branch leaves no trace
    branches = events(tmp_path / "b", "trial")[0]["branches"]
    assert [(b["action"], b["target"]) for b in branches] == [("set_readout", 2), ("set_readout", 3),
                                                              ("refit_head", 4), ("incumbent_repeat", None)]
    assert all(np.isfinite(b["horizons"]["4"]["mean_d"]) for b in branches)
