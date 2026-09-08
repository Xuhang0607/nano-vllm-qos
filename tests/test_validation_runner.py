import os
import socket

import pytest

from benchmarks.run_validation import check_port, save_json, validate_resume


@pytest.mark.skipif(os.name != "posix", reason="validation runner requires WSL/Linux socket semantics")
def test_port_probe_rejects_an_existing_listener():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(OSError):
            check_port(port)
    check_port(port)


def test_resume_rejects_changed_sources_or_experiment_parameters():
    before = {"seed": 1, "source_sha256": {"engine.py": "abc"}, "resume": False}
    validate_resume(before, {**before, "resume": True})
    with pytest.raises(ValueError, match="seed"):
        validate_resume(before, {**before, "seed": 2})
    with pytest.raises(ValueError, match="source_sha256"):
        validate_resume(before, {**before, "source_sha256": {"engine.py": "changed"}})


def test_save_json_replaces_complete_document_without_leaving_temporary_file(tmp_path):
    import json

    path = tmp_path / "status.json"
    save_json(path, {"state": "running"})
    save_json(path, {"state": "complete"})
    assert json.loads(path.read_text())["state"] == "complete"
    assert not path.with_suffix(".json.tmp").exists()


def test_admission_suite_alternates_order_and_restores_candidate_setting(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from benchmarks import run_validation as runner

    args = SimpleNamespace(
        kv_admission_lookahead=32, rates=[2, 3], repeats=2, resume=False,
        output_dir=tmp_path, base_url="http://unused", timeout=1,
        seed=7, arrival_process="uniform", max_inflight=8,
    )
    starts = []

    @contextmanager
    def fake_server(args, policy, label):
        starts.append((args.kv_admission_lookahead, policy))
        yield

    def compare(before, after, **kwargs):
        assert before == [0, 0]
        assert after == [32, 32]
        assert kwargs == {"allow_admission_change": True}
        return {}

    monkeypatch.setattr(runner, "sustained_probe_workload", lambda args, rate: (2, []))
    monkeypatch.setattr(runner, "server", fake_server)
    monkeypatch.setattr(runner, "run_live_benchmark", lambda *a, **kw: args.kv_admission_lookahead)
    monkeypatch.setattr(runner, "write_results", lambda *a: None)
    monkeypatch.setattr(runner, "print_report", lambda *a: None)
    monkeypatch.setattr(runner, "write_status", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "save_json", lambda *a: None)
    monkeypatch.setattr(runner, "aggregate_runs", lambda runs, name: runs)
    monkeypatch.setattr(runner, "compare_groups", compare)
    monkeypatch.setattr(runner, "render_markdown", lambda report: "test")
    runner.admission_suite(args)
    assert starts == [(0, "none"), (32, "none"), (32, "none"), (0, "none")] * 2
    assert args.kv_admission_lookahead == 32
