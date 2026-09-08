"""Export or compare isolated-process Prefill/Decode device-context logits."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/d/models/Qwen3-0.6B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("LEGACY", "SCOPED"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    import torch
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.compare:
        left, right = [torch.load(path, map_location="cpu", weights_only=True) for path in args.compare]
        records = [{"phase": phase, "logits_equal": torch.equal(a, b),
                    "max_abs_error": (a.float() - b.float()).abs().max().item(),
                    "argmax_equal": torch.equal(a.argmax(-1), b.argmax(-1))}
                   for phase, a, b in zip(("prefill", "decode"), left, right)]
        passed = left.shape == right.shape and len(left) == 2 and all(item["logits_equal"] for item in records)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump({"scope": __doc__, "passed": passed, "comparisons": records}, stream, indent=2)
        print(json.dumps(records))
        if not passed:
            raise SystemExit(1)
        return
    from nanovllm import LLM, SamplingParams
    from nanovllm.utils.context import reset_context

    engine = LLM(args.model, enforce_eager=not args.cuda_graph, max_model_len=512,
                 max_num_batched_tokens=128, max_num_seqs=8,
                 num_kvcache_blocks_override=4)
    logits = []
    try:
        if args.legacy:
            torch.set_default_device("cpu")
        engine.add_request([100] * 33, SamplingParams(max_tokens=2, ignore_eos=True))
        runner = engine.model_runner
        for expected_prefill in (True, False):
            seqs, is_prefill = engine.scheduler.schedule()
            assert is_prefill == expected_prefill
            try:
                inputs, positions = (runner.prepare_prefill(seqs) if is_prefill
                                     else runner.prepare_decode(seqs))
                logits.append(runner.run_model(inputs, positions, is_prefill).cpu())
            finally:
                reset_context()
            # Fixed continuation gives both processes identical Decode inputs.
            engine.scheduler.postprocess(seqs, [100] * len(seqs), is_prefill)
        assert engine.is_finished()
        assert len(engine.scheduler.block_manager.free_block_ids) == 4
    finally:
        engine.exit()
    with args.output.open("xb") as stream:
        torch.save(torch.stack(logits), stream)
    print(json.dumps({"mode": "legacy" if args.legacy else "scoped", "output": str(args.output)}))


if __name__ == "__main__":
    main()
