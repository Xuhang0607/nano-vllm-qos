import json
from itertools import count
from types import SimpleNamespace

import pytest

from benchmarks.analyze_scheduler_trace import analyze

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.scheduler_trace import SchedulerTrace
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def config(path=None):
    return SimpleNamespace(
        max_num_seqs=4, max_num_batched_tokens=16, eos=-1,
        kvcache_block_size=4, num_kvcache_blocks=4,
        scheduling_policy="fcfs", qos_prefill_ms_per_token=1.0,
        qos_best_effort_slo_ms=1000.0, qos_priority_boost_ms=20.0,
        qos_aging_ms_per_step=1.0,
        qos_decode_ms_per_token=1.0, qos_ewma_alpha=0.2,
        kv_reclaim_policy="recompute", scheduler_trace_path=path,
    )


def run_pressure(path, monkeypatch, active_limit=None):
    monkeypatch.setattr(Sequence, "block_size", 4)
    monkeypatch.setattr(Sequence, "counter", count())
    settings = config(path)
    settings.max_num_active_seqs = active_limit
    scheduler = Scheduler(settings, clock=lambda: 1.0)
    for start in (100, 200):
        scheduler.add(Sequence(list(range(start, start + 8)),
                               SamplingParams(max_tokens=3, ignore_eos=True)))
    steps = []
    for _ in range(20):
        if scheduler.is_finished():
            break
        seqs, prefill = scheduler.schedule()
        if active_limit is not None:
            assert scheduler._resident_count() <= active_limit
        steps.append((prefill, [seq.seq_id for seq in seqs]))
        scheduler.postprocess(seqs, [999] * len(seqs), prefill)
    assert scheduler.is_finished()
    return scheduler, steps


def test_trace_preserves_schedule_and_captures_first_exhaustion(tmp_path, monkeypatch):
    plain, plain_steps = run_pressure(None, monkeypatch)
    path = tmp_path / "trace.jsonl"
    traced, traced_steps = run_pressure(path, monkeypatch)
    assert traced_steps == plain_steps
    assert traced.completed_metrics == plain.completed_metrics
    assert traced.block_manager.cache_metrics() == plain.block_manager.cache_metrics()
    rows = read_rows(path)
    pressure = next(row for row in rows if row["event"] == "append_pressure")
    assert pressure["free_blocks"] == 0
    assert pressure["selected"][0]["next_decode_needs_page"]
    admission = [row for row in rows if row["event"] == "prefill_admitted"]
    assert admission[1]["free_blocks"] == 0
    assert any(row["event"] == "preempted" for row in rows)
    assert any(row["event"] == "finished" for row in rows)
    assert "token_ids" not in path.read_text()
    assert "prompt" not in path.read_text()
    report = analyze(rows)
    assert report["first_pressure"] == pressure
    assert report["captured_append_pressure_events"] >= 1
    assert report["preceding_events"]


def test_recorder_is_bounded_and_deduplicates_overlapping_windows(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = SchedulerTrace(path, {}, history=3, following=2, max_triggers=2)
    for index in range(20):
        trace.record({"event": "step", "value": index}, trigger=index in (5, 6))
    rows = read_rows(path)[1:]
    ids = [row["event_id"] for row in rows]
    assert ids == list(range(3, 9))
    assert not trace.enabled
    assert len(trace.history) <= 3


def test_no_incident_writes_only_header_and_existing_run_is_protected(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = SchedulerTrace(path, {})
    trace.record({"event": "step"})
    assert len(read_rows(path)) == 1
    with pytest.raises(FileExistsError):
        SchedulerTrace(path, {})


def test_write_failure_disables_recorder_without_breaking_scheduler(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    trace = SchedulerTrace(path, {})
    def fail(*args, **kwargs):
        raise OSError("simulated disk full")
    monkeypatch.setattr(type(path), "open", fail)
    trace.record({"event": "pressure"}, trigger=True)
    assert not trace.enabled


def test_byte_budget_disables_capture(tmp_path):
    trace = SchedulerTrace(tmp_path / "trace.jsonl", {}, max_bytes=512)
    trace.record({"event": "pressure", "data": "x" * 512}, trigger=True)
    assert not trace.enabled
    assert trace.bytes_written <= 512


def test_disabled_trace_does_not_read_clock_or_build_snapshots():
    def fail():
        raise AssertionError("disabled trace must not read clock")
    scheduler = Scheduler(config(), clock=fail)
    scheduler._trace("step", (object(),), trigger=True)


def test_queue_snapshots_are_truncated_but_total_count_is_retained(tmp_path, monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)
    path = tmp_path / "trace.jsonl"
    scheduler = Scheduler(config(path), clock=lambda: 1.0)
    for _ in range(70):
        scheduler.add(Sequence([1, 2]))
    scheduler._trace("probe", trigger=True)
    row = read_rows(path)[-1]
    assert row["waiting_count"] == 70
    assert len(row["waiting"]) == 64


def test_active_limit_prevents_deterministic_overadmission(monkeypatch):
    baseline, _ = run_pressure(None, monkeypatch)
    limited, _ = run_pressure(None, monkeypatch, active_limit=1)
    assert baseline.kv_reclaim_events > 0
    assert limited.kv_reclaim_events == 0
    assert len(limited.completed_metrics) == 2
    assert [item.completion_tokens for item in limited.completed_metrics] == [3, 3]


@pytest.mark.parametrize("policy", ["fcfs", "pals"])
def test_resident_partial_prefill_progresses_when_new_request_is_ahead(monkeypatch, policy):
    monkeypatch.setattr(Sequence, "block_size", 4)
    settings = config()
    settings.max_num_active_seqs = 1
    settings.max_num_batched_tokens = 4
    settings.scheduling_policy = policy
    scheduler = Scheduler(settings, clock=lambda: 1.0)
    resident = Sequence(list(range(8)), SamplingParams(max_tokens=3, ignore_eos=True))
    newcomer = Sequence(list(range(100, 108)), SamplingParams(max_tokens=3, ignore_eos=True))
    scheduler.add(resident)
    seqs, prefill = scheduler.schedule()
    scheduler.postprocess(seqs, [999], prefill)
    scheduler.add(newcomer)
    scheduler.waiting.remove(newcomer)
    scheduler.waiting.appendleft(newcomer)
    seqs, prefill = scheduler.schedule()
    assert seqs == [resident]
    assert prefill
    assert not newcomer.block_table
    scheduler.postprocess(seqs, [999], prefill)
    for _ in range(20):
        if scheduler.is_finished():
            break
        seqs, prefill = scheduler.schedule()
        assert scheduler._resident_count() <= 1
        scheduler.postprocess(seqs, [999] * len(seqs), prefill)
    assert scheduler.is_finished()


def test_cancel_releases_resident_slot(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)
    settings = config()
    settings.max_num_active_seqs = 1
    scheduler = Scheduler(settings, clock=lambda: 1.0)
    first, second = (Sequence([index] * 8) for index in (1, 2))
    scheduler.add(first)
    scheduler.add(second)
    seqs, prefill = scheduler.schedule()
    assert seqs == [first]
    scheduler.postprocess(seqs, [999], prefill)
    assert scheduler.cancel(first.seq_id)
    seqs, _ = scheduler.schedule()
    assert seqs == [second]


def test_invalid_active_limit_is_rejected():
    settings = config()
    settings.max_num_active_seqs = 0
    with pytest.raises(ValueError, match="max_num_active_seqs"):
        Scheduler(settings)


def test_pending_restore_does_not_drop_already_scheduled_decode(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)
    settings = config()
    settings.max_num_active_seqs = 2
    scheduler = Scheduler(settings, clock=lambda: 1.0)
    live = Sequence([1, 2, 3], SamplingParams(max_tokens=4, ignore_eos=True))
    scheduler.add(live)
    seqs, prefill = scheduler.schedule()
    scheduler.postprocess(seqs, [999], prefill)
    # Pending restores consume a resident slot, but must not suppress ready decode.
    scheduler.pending_restores[999] = object()
    newcomer = Sequence([100, 101])
    scheduler.add(newcomer)
    seqs, prefill = scheduler.schedule()
    assert not prefill
    assert seqs == [live]
    assert live in scheduler.running
    assert not newcomer.block_table


def test_pending_restore_holds_slot_until_ready(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)
    settings = config()
    settings.max_num_active_seqs = 1
    scheduler = Scheduler(settings, clock=lambda: 1.0)
    scheduler.pending_restores[999] = object()
    newcomer = Sequence([100, 101])
    scheduler.add(newcomer)
    assert scheduler.schedule() == ([], True)
    assert not newcomer.block_table
