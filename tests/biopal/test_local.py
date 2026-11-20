"""The local expert against a fake OpenAI-compatible server: one request per proposal, what is sent, who reads the
note, how failures end, and that a brief neither leaves the host nor falls back to a hosted model."""
import json
import os
import stat

import pytest

import toy
from biopal import expert
from local_server import REVISION
from test_outer import events, script

NOTE = "In our hands the GC-rich rows are the noisy ones."


def local(endpoint, **flags):
    return dict(bio_policy="expert", expert_backend="local", expert_url=endpoint.url, expert_model="toy-llm",
                expert_revision=REVISION, **flags)


def text(body):
    return body["messages"][1]["content"]


def test_local_expert_sends_one_request_per_proposal_and_needs_no_budget(tmp_path, endpoint):
    thinking_off = dict(chat_template_kwargs=dict(enable_thinking=False))
    args, t, bio = toy.make(tmp_path, **local(endpoint, expert_shadow="masked", expert_request=thinking_off))
    _, summary = toy.train(args, t, bio)
    posts = endpoint.posts()
    assert len(posts) == 4 == summary["expert_calls"] and summary["expert_cost_usd"] == 0   # two gates, two roles
    for body in posts:
        assert body["model"] == "toy-llm" and body["temperature"] == 0 and body["seed"] == args.seed
        assert body["chat_template_kwargs"] == dict(enable_thinking=False)
        assert body["response_format"]["json_schema"]["schema"] == expert.SCHEMA
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        assert body["messages"][0]["content"] == expert.SYSTEM
    first = events(tmp_path, "proposals")[0]
    assert first["informed"]["ok"] and first["informed"]["target"] == first["masked"]["target"] == "gc:low"
    assert "gc:low" in text(posts[0]) and "cohort_01" in text(posts[1]) and "gc:low" not in text(posts[1])
    about = summary["expert"]
    assert about["backend"] == "local" and about["model"] == "toy-llm" and about["serving_version"] == "0.0-fake"
    assert about["revision"] == REVISION and about["revision_in_served_path"] is True
    assert about["decoding"] == dict(temperature=0, top_p=1, seed=args.seed, max_tokens=512, **thinking_off)
    assert about["calls"] == 4 and about["failures"] == dict(informed=0, masked=0) and about["note"] is None
    assert len(about["latency_seconds"]["each"]) == 4 and about["latency_seconds"]["max"] > 0
    assert events(tmp_path, "schedule")[0]["expert"]["served_root"].endswith(REVISION)   # known before the first gate
    assert json.loads((tmp_path / "expert" / "gate1_informed_reply.json").read_text())["choices"]


def test_only_the_informed_local_expert_reads_the_note_and_nothing_else_changes(tmp_path, endpoint, note):
    endpoint.mode = "keep"                                              # both runs then pass the same checkpoints
    args, t, bio = toy.make(tmp_path / "with", **local(endpoint, expert_shadow="masked", expert_note=note))
    _, summary = toy.train(args, t, bio)
    noted = [text(b) for b in endpoint.posts()]
    endpoint.requests.clear()
    toy.train(*toy.make(tmp_path / "without", **local(endpoint, expert_shadow="masked")))
    plain = [text(b) for b in endpoint.posts()]
    assert len(noted) == len(plain) == 4
    for i, (a, b) in enumerate(zip(noted, plain)):
        if i % 2:                                                       # the masked expert: no note, same text
            assert a == b and NOTE not in a
            continue
        block = a[a.index(expert.NOTE_OPEN):a.index(expert.NOTE_CLOSE) + len(expert.NOTE_CLOSE) + 2]
        assert NOTE in block and "Upweighting them has not paid off." in block
        assert a.replace(block, "") == b                                # measurements, menu and order: byte for byte
        assert a.index("Background: ") < a.index(expert.NOTE_OPEN) < a.index("CHECKPOINT after update")
        assert (tmp_path / "with" / "expert" / f"gate{i // 2 + 1}_informed_brief.txt").read_text() == b
    record = summary["expert"]["note"]                                  # the note's identity, not its text
    assert record == dict(task="toy_task", label="simulated institutional knowledge", public=False,
                          sha256=expert.load_note(note)["sha256"], chars=len(expert.load_note(note)["text"]))
    assert events(tmp_path / "with", "schedule")[0]["expert"]["note"] == record
    for path in [tmp_path / "with" / "outer.jsonl", *(tmp_path / "with" / "expert").glob("*_brief.txt")]:
        assert NOTE not in path.read_text()
    assert NOTE not in json.dumps(summary)


@pytest.mark.parametrize("visibility", ["private", "public"])
def test_a_reply_that_repeats_a_private_note_stays_in_the_reply_file(tmp_path, endpoint, visibility):
    path = tmp_path / "note.txt"
    path.write_text(f"task: toy_task\nvisibility: {visibility}\n\n{NOTE}\n")
    endpoint.mode = "echo"                                              # the hypothesis is the note, word for word
    args, t, bio = toy.make(tmp_path / "run", **local(endpoint, expert_note=path))
    _, summary = toy.train(args, t, bio)
    said = events(tmp_path / "run", "proposals")[0]["informed"]
    assert said["ok"] and said["action"] == "set_cohort" and said["predicted_sign"] == "lower"
    shared = (tmp_path / "run" / "outer.jsonl").read_text() + json.dumps(summary)
    if visibility == "private":
        assert NOTE not in shared and said["hypothesis"] == expert.WITHHELD
        assert said["evidence_rows"] == ["r01"] and said["layer_evidence"] == "block_2"   # row ids and a block stay
    else:
        assert said["hypothesis"] == NOTE and said["evidence_rows"] == ["r01", "the note"]
        assert summary["expert"]["note"]["text"] == NOTE
    assert NOTE in (tmp_path / "run" / "expert" / "gate1_informed_reply.json").read_text()   # local, as received


@pytest.mark.parametrize("mode, kind", [("junk", "malformed"), ("prose", "malformed"), ("think", "malformed"),
                                        ("truncated", "malformed"), ("off_menu", "off_menu"),
                                        ("http500", "transport"), ("redirect", "transport"), ("sleep", "timeout")])
def test_a_failed_local_call_keeps_the_incumbent_without_retry_or_hosted_fallback(tmp_path, endpoint, monkeypatch,
                                                                                  mode, kind):
    claude = tmp_path / "claude"                                        # would leave a trace if anything called it
    claude.write_text(f"#!/bin/sh\necho called >> {tmp_path / 'claude_calls'}\n")
    claude.chmod(claude.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("BIOPAL_CLAUDE", str(claude))
    reference, _ = toy.train(*toy.make(tmp_path / "ref"))
    args, t, bio = toy.make(tmp_path / "run", **local(endpoint, expert_timeout=1.))
    endpoint.mode = mode
    losses, summary = toy.train(args, t, bio)
    assert losses == reference and summary["adopted"] == [] and not events(tmp_path / "run", "trial")
    proposals = events(tmp_path / "run", "proposals")
    assert all(p["informed"]["ok"] is False and p["informed"]["failure"] == kind for p in proposals)
    assert len(endpoint.posts()) == 2 == summary["expert_calls"]       # one request per gate, none repeated
    assert not any("elsewhere" in path for _, path, _ in endpoint.requests)   # the redirect was not followed
    assert not (tmp_path / "claude_calls").exists() and summary["expert_cost_usd"] == 0
    assert summary["expert"]["failures"] == dict(informed=2)
    decision = events(tmp_path / "run", "decision")[0]["proposals"]["informed"]
    assert decision["outcome"] == "failed" and decision["failure"] == kind


def test_a_failure_with_a_private_note_is_logged_as_its_kind_only(tmp_path, endpoint, note):
    endpoint.mode = "off_menu"
    args, t, bio = toy.make(tmp_path, **local(endpoint, expert_shadow="masked", expert_note=note))
    toy.train(args, t, bio)
    first = events(tmp_path, "proposals")[0]
    assert first["informed"]["error"].startswith("off_menu (") and "no_such_cohort" not in first["informed"]["error"]
    assert "no_such_cohort" in first["masked"]["error"]                 # no note in that brief: the text is kept


def test_a_deliberate_keep_is_an_answer_and_not_a_failure(tmp_path, endpoint):
    endpoint.mode = "keep"
    args, t, bio = toy.make(tmp_path, **local(endpoint))
    _, summary = toy.train(args, t, bio)
    said = events(tmp_path, "decision")[0]["proposals"]["informed"]
    assert said["ok"] is True and said["outcome"] == "kept" and said["failure"] is None and said["error"] is None
    assert summary["expert"]["failures"] == dict(informed=0) and not events(tmp_path, "trial")


def test_a_role_can_only_choose_a_target_under_the_name_it_was_shown(tmp_path, endpoint):
    endpoint.mode = "real_name"                                         # answers "gc:low" to both roles
    args, t, bio = toy.make(tmp_path, **local(endpoint, expert_shadow="masked"))
    toy.train(args, t, bio)
    first = events(tmp_path, "proposals")[0]
    assert first["informed"]["ok"] and first["informed"]["target"] == "gc:low"
    assert first["masked"]["ok"] is False and first["masked"]["failure"] == "off_menu"


@pytest.mark.parametrize("url", ["http://localhost:8000/v1", "http://10.1.2.3:8000/v1", "https://127.0.0.1:8000/v1",
                                 "http://example.org:8000/v1", "http://127.0.0.1/v1", "http://u:p@127.0.0.1:8000/v1",
                                 "http://127.0.0.1:8000/v1?next=http://example.org", "127.0.0.1:8000", ""])
def test_only_a_loopback_ip_address_is_accepted(url, tmp_path):
    with pytest.raises(ValueError, match="loopback IP"):
        expert.loopback(url)
    with pytest.raises(ValueError, match="loopback IP"):
        expert.Local(url, "toy-llm", REVISION, 1., tmp_path)


def test_loopback_addresses():
    assert expert.loopback("http://127.0.0.1:8000/v1/") == "http://127.0.0.1:8000/v1"
    assert expert.loopback("http://127.8.9.10:1234") and expert.loopback("http://[::1]:8000/v1")


def test_proxy_settings_of_the_environment_are_not_used(tmp_path, endpoint, monkeypatch):
    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")                  # nothing listens there
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    client = expert.Local(endpoint.url, "toy-llm", REVISION, 5., tmp_path)
    assert client.served["serving_version"] == "0.0-fake"
    assert client.propose("informed", 0, "ACTIONS\n  set_cohort <cohort>: On offer: gc:low, gc:high")["target"] == "gc:low"


def test_only_a_listener_that_can_take_the_connection_counts_as_an_owner_of_the_port(tmp_path):
    def row(address, uid, state="0A"):
        return f"   0: {address} 00000000:0000 {state} 00000000:00000000 00:00000000 00000000 {uid} 0 1 1 0 100 0 0 10 0"

    v4, v6 = tmp_path / "tcp", tmp_path / "tcp6"
    v4.write_text("\n".join(["sl local_address ...",
                             row("0100007F:1F40", 1000),                # 127.0.0.1:8000, mine
                             row("174FDD8C:1F40", 2000),                # the same port on the machine's own address
                             row("0200007F:1F40", 2001),                # ... and on 127.0.0.2
                             row("0100007F:1F41", 2002),                # another port
                             row("0100007F:1F40", 2003, state="01")]))  # a connection, not a listener
    v6.write_text("sl local_address ...\n" + row("00000000000000000000000001000000:1F40", 2004))   # [::1]:8000
    assert expert.listeners(8000, "127.0.0.1", (v4, v6)) == {1000}
    v4.write_text(v4.read_text() + "\n" + row("00000000:1F40", 3000))  # every address: this one can take it
    v6.write_text(v6.read_text() + "\n" + row("0000000000000000FFFF00000100007F:1F40", 3001))   # ::ffff:127.0.0.1
    assert expert.listeners(8000, "127.0.0.1", (v4, v6)) == {1000, 3000, 3001}
    assert expert.listeners(8000, "::1", (v4, v6)) == {2004, 3000}
    assert expert.listeners(8000, "127.0.0.1", (tmp_path / "missing",)) is None


def test_the_server_is_checked_before_training_starts(tmp_path, endpoint, monkeypatch):
    def client(model="toy-llm", revision=REVISION, url=None, **more):
        return expert.Local(url or endpoint.url, model, revision, 2., tmp_path, **more)

    port = int(endpoint.url.split(":")[2].split("/")[0])
    assert expert.listeners(port, "127.0.0.1") == {os.getuid()} and client().served["port_owner_checked"] is True
    with pytest.raises(RuntimeError, match="does not serve 'another-model'; it serves \\['toy-llm'\\]"):
        client("another-model")
    with pytest.raises(RuntimeError, match="not revision ffff"):       # the server's path names another commit
        client(revision="f" * 40)
    with pytest.raises(RuntimeError, match="no model list"):
        client(url="http://127.0.0.1:9/v1")
    with pytest.raises(ValueError, match="model or messages"):
        client(request=dict(model="something-else"))
    endpoint.root, endpoint.version = "toy/llm", None                   # served by name, by a server without /version
    served = client(revision="f" * 40).served
    assert served["revision_in_served_path"] is False and served["serving_version"] is None
    monkeypatch.setattr(expert, "listeners", lambda port, host: {os.getuid(), os.getuid() + 1})
    with pytest.raises(RuntimeError, match="another user of this machine"):
        client()
    assert len(endpoint.posts()) == 0                                   # nothing was sent to it


def test_a_note_is_bound_to_its_task_and_to_the_local_expert(tmp_path, endpoint, note):
    assert expert.load_note(note, "toy_task")["text"].startswith(NOTE) and expert.load_note(note)["public"] is False
    with pytest.raises(ValueError, match="note for task 'toy_task', not for 'another_task'"):
        expert.load_note(note, "another_task")
    for bad in ("just a sentence\n\nand text", "label: x\n\ntext", "task: toy_task\n\n  \n",
                "task: toy_task\nvisiblity: public\n\ntext", "task: toy_task\nvisibility: shared\n\ntext",
                "task: toy_task\ntask: other\n\ntext"):
        (tmp_path / "bad.txt").write_text(bad)
        with pytest.raises(ValueError, match="note"):
            expert.load_note(tmp_path / "bad.txt")
    (tmp_path / "other.txt").write_text("task: another_task\n\nSomething about another task.\n")
    with pytest.raises(ValueError, match="note for task 'another_task', not for 'toy_task'"):
        toy.make(tmp_path / "a", **local(endpoint, expert_note=tmp_path / "other.txt"))
    for hosted_or_scripted in (dict(expert_budget=1.), dict(expert_script=script(tmp_path, informed=[]))):
        with pytest.raises(ValueError, match="local expert only"):     # no note in a Claude call or a scripted run
            toy.make(tmp_path / "b", bio_policy="expert", expert_note=note, **hosted_or_scripted)
    assert not endpoint.posts()


def test_with_note_changes_nothing_but_the_note_block():
    brief = "TASK\nx\nBackground: about the task.\n\nCHECKPOINT after update 3 of 9\nrest"
    noted = expert.with_note(brief, dict(text="first line\nsecond line"))
    assert noted == ("TASK\nx\nBackground: about the task.\n\n" + expert.NOTE_OPEN + "\nfirst line\nsecond line\n"
                     + expert.NOTE_CLOSE + "\n\nCHECKPOINT after update 3 of 9\nrest")
