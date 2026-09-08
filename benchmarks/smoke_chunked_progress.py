"""GPU execution smoke for small-KV, mixed-length chunked prefill; not a speed test."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/d/models/Qwen3-0.6B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.qos import RequestQoS

    engine = LLM(args.model, enforce_eager=True, max_model_len=1024,
                 max_num_batched_tokens=128, max_num_seqs=4,
                 num_kvcache_blocks_override=4, scheduling_policy="pals",
                 prefix_cache_backend="radix", kv_reclaim_policy="recompute")
    results = []
    try:
        for horizon in (0, 32):
            engine.scheduler.kv_admission_lookahead = horizon
            before = engine.scheduler.step_id
            outputs = engine.generate(
                [[100 + horizon] * 513, [200 + horizon] * 769, [300 + horizon] * 257],
                SamplingParams(max_tokens=4, temperature=0.1, ignore_eos=True),
                request_qos=RequestQoS(priority=10), use_tqdm=False,
            )
            assert len(outputs) == 3
            assert all(len(item["token_ids"]) == 4 for item in outputs)
            assert engine.scheduler.is_finished()
            assert len(engine.scheduler.block_manager.free_block_ids) == 4
            results.append({"horizon": horizon, "completed": len(outputs),
                            "output_lengths": [len(item["token_ids"]) for item in outputs],
                            "scheduler_steps": engine.scheduler.step_id - before})
    finally:
        engine.exit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"scope": "GPU execution smoke, no speed or quality claim", "model": args.model,
              "kv_pages": 4, "step_token_budget": 128, "runs": results}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
