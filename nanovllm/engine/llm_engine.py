import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.qos import RequestQoS
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        qos: RequestQoS | None = None,
    ):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params, qos=qos)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        started = perf_counter()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        elapsed_ms = (perf_counter() - started) * 1000.0
        observed_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else len(seqs)
        self.scheduler.observe_execution(is_prefill, observed_tokens, elapsed_ms)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        request_qos: RequestQoS | list[RequestQoS] | None = None,
        use_tqdm: bool = True,
    ) -> list[dict]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        elif len(sampling_params) != len(prompts):
            raise ValueError("sampling_params must have the same length as prompts")
        if request_qos is None:
            request_qos = [RequestQoS()] * len(prompts)
        elif not isinstance(request_qos, list):
            request_qos = [request_qos] * len(prompts)
        elif len(request_qos) != len(prompts):
            raise ValueError("request_qos must have the same length as prompts")
        submitted_seq_ids = []
        for prompt, sp, qos in zip(prompts, sampling_params, request_qos):
            submitted_seq_ids.append(self.add_request(prompt, sp, qos))
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [
            {
                "text": self.tokenizer.decode(outputs[seq_id]),
                "token_ids": outputs[seq_id],
                "metrics": self.scheduler.request_metrics(seq_id).to_dict(),
            }
            for seq_id in submitted_seq_ids
        ]
        return outputs

    def get_scheduler_metrics(self):
        return self.scheduler.metrics()
