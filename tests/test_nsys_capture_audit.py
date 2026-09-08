import sqlite3

import pytest

from benchmarks.audit_nsys_capture import audit_capture


@pytest.mark.parametrize("kernels", [0, 2])
def test_audit_does_not_substitute_host_api_for_gpu_data(tmp_path, kernels):
    path = tmp_path / "capture.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE DIAGNOSTIC_EVENT (severity INTEGER, text TEXT);
            INSERT INTO DIAGNOSTIC_EVENT VALUES (2, 'driver warning');
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            INSERT INTO StringIds VALUES (1, 'cudaLaunchKernel');
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, nameId INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (0, 1000000, 1);
        """)
        if kernels:
            db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER)")
            db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?)", [(i,) for i in range(kernels)])
    result = audit_capture(path)
    assert result["gpu_kernel_data_present"] == bool(kernels)
    assert result["top_cuda_apis"][0]["inclusive_host_ms"] == 1
    assert result["diagnostics"][0]["text"] == "driver warning"
    if not kernels:
        assert result["gpu_timing_status"] == "unavailable"


def test_missing_capture_does_not_create_database(tmp_path):
    path = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        audit_capture(path)
    assert not path.exists()
