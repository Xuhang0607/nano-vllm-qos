import pytest
from contextlib import contextmanager

from benchmarks.profile_engine_phases import measure_method


def test_measure_restores_method_after_failure():
    class Target:
        def run(self):
            raise ValueError("expected")

    target = Target()
    totals = {}
    with pytest.raises(ValueError, match="expected"):
        with measure_method(target, "run", totals, "run"):
            target.run()
    assert "run" not in vars(target)
    assert totals["run"]["calls"] == 1
    assert totals["run"]["wall_ms"] >= 0


def test_measure_preserves_return_and_instance_override():
    class Target:
        pass

    target = Target()
    original = lambda value: value + 1
    target.run = original
    totals = {}
    with measure_method(target, "run", totals, "run"):
        assert target.run(4) == 5
    assert target.run is original


def test_range_closed_and_method_restored_on_error():
    class Target:
        def run(self):
            raise RuntimeError("failure")

    events = []

    @contextmanager
    def region(label):
        events.append((label, "enter"))
        try:
            yield
        finally:
            events.append((label, "exit"))

    target = Target()
    with pytest.raises(RuntimeError, match="failure"):
        with measure_method(target, "run", {}, "sample", region):
            target.run()
    assert events == [("sample", "enter"), ("sample", "exit")]
    assert "run" not in vars(target)
