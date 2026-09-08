from types import SimpleNamespace

import pytest

from benchmarks.serve_device_context import wrap_factory


@pytest.mark.parametrize("mode", ["legacy_cpu", "scoped"])
def test_factory_sets_mode_after_engine_creation_and_labels_metrics(mode):
    calls = []
    original = {"requests": 1}

    def factory():
        calls.append("created")
        return SimpleNamespace(get_scheduler_metrics=lambda: original)

    engine = wrap_factory(factory, mode, calls.append)()
    assert calls == (["created", "cpu"] if mode == "legacy_cpu" else ["created"])
    assert engine.get_scheduler_metrics() == {"requests": 1, "device_context_mode": mode}
    assert original == {"requests": 1}


def test_unknown_mode_rejected_before_engine_creation():
    with pytest.raises(ValueError, match="mode"):
        wrap_factory(None, "invalid", None)


def test_device_suite_alternates_modes(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from benchmarks import run_validation as runner

    args = SimpleNamespace(rates=[3], repeats=3, resume=False, output_dir=tmp_path,
                           base_url="unused", timeout=1, seed=7,
                           arrival_process="uniform", max_inflight=8)
    starts = []

    @contextmanager
    def server(args, policy, label):
        assert policy == "none"
        starts.append(args.device_context_mode)
        yield

    def compare(left, right, **kwargs):
        assert left == ["legacy_cpu"] * 3
        assert right == ["scoped"] * 3
        assert kwargs == {"allow_device_context_change": True}
        return {}

    monkeypatch.setattr(runner, "sustained_probe_workload", lambda *a: (2, []))
    monkeypatch.setattr(runner, "server", server)
    monkeypatch.setattr(runner, "run_live_benchmark", lambda *a, **k: args.device_context_mode)
    for name in ("write_results", "print_report", "write_status", "save_json"):
        monkeypatch.setattr(runner, name, lambda *a, **k: None)
    monkeypatch.setattr(runner, "aggregate_runs", lambda runs, name: runs)
    monkeypatch.setattr(runner, "compare_groups", compare)
    monkeypatch.setattr(runner, "render_markdown", lambda *a: "test")
    runner.device_suite(args)
    assert starts == ["legacy_cpu", "scoped", "scoped", "legacy_cpu", "legacy_cpu", "scoped"]
