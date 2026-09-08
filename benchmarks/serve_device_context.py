"""Benchmark-only server entry point isolating persistent CPU device dispatch."""

import os


def wrap_factory(factory, mode, set_default_device):
    if mode not in ("legacy_cpu", "scoped"):
        raise ValueError("device ablation mode must be legacy_cpu or scoped")

    def create(*args, **kwargs):
        engine = factory(*args, **kwargs)
        # The serving worker invokes its factory on the inference thread.
        if mode == "legacy_cpu":
            set_default_device("cpu")
        original_metrics = engine.get_scheduler_metrics

        def metrics():
            return {**original_metrics(), "device_context_mode": mode}

        engine.get_scheduler_metrics = metrics
        return engine

    return create


def main():
    import nanovllm
    import torch
    from nanovllm.serve.__main__ import main as serve

    mode = os.environ.get("NANOVLLM_BENCH_DEVICE_MODE", "")
    nanovllm.LLM = wrap_factory(nanovllm.LLM, mode, torch.set_default_device)
    serve()


if __name__ == "__main__":
    main()
