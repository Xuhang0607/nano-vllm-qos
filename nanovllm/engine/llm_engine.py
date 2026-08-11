import atexit
from collections import deque
from dataclasses import fields
from time import perf_counter

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config, effective_context_window
from nanovllm.engine.hierarchical_cache import (
    CacheTier,
    KVCacheGeometry,
    KVTransferPlanner,
    StorageTierProfile,
)
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.qos import RequestQoS
from nanovllm.engine.remote_restore import (
    RemoteKVRestoreService,
    RemotePageDescriptor,
    RemotePrefixCatalog,
    completed_kv_page_span,
)
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.storage_backend import (
    KVCacheIdentity,
    make_kv_catalog_key,
    make_kv_page_key,
)
from nanovllm.sampling_params import SamplingParams


class LLMEngine:

    def __init__(self, model, **kwargs):
        kv_storage_backend = kwargs.pop("kv_storage_backend", None)
        self.kv_storage_backend = kv_storage_backend
        remote_prefix_catalog = kwargs.pop("remote_prefix_catalog", None)
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        if kv_storage_backend is not None and config.tensor_parallel_size != 1:
            raise ValueError(
                "automatic remote KV restore currently requires "
                "tensor_parallel_size=1"
            )
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
        config.max_model_len = effective_context_window(
            config.max_model_len,
            config.num_kvcache_blocks,
            config.kvcache_block_size,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.kv_cache_identity = KVCacheIdentity(
            model_id=config.kv_cache_model_id,
            model_revision=config.kv_cache_model_revision,
            dtype=self.model_runner.kv_page_io.layout.dtype,
            tp_size=config.tensor_parallel_size,
            page_size=config.kvcache_block_size,
        )
        self.remote_restore_service = None
        if kv_storage_backend is not None:
            layout = self.model_runner.kv_page_io.layout
            planner = None
            if config.remote_kv_cost_aware:
                dtype_bytes = layout.nbytes // (
                    2
                    * layout.num_layers
                    * layout.block_size
                    * layout.num_kv_heads
                    * layout.head_dim
                )
                planner = KVTransferPlanner(
                    KVCacheGeometry(
                        num_layers=layout.num_layers,
                        num_kv_heads=layout.num_kv_heads,
                        head_dim=layout.head_dim,
                        dtype_bytes=dtype_bytes,
                    ),
                    {
                        CacheTier.MOONCAKE: StorageTierProfile(
                            bandwidth_gbps=config.remote_kv_bandwidth_gbps,
                            fixed_latency_ms=config.remote_kv_fixed_latency_ms,
                            congestion_multiplier=(
                                config.remote_kv_congestion_multiplier
                            ),
                        )
                    },
                )
            self.remote_restore_service = RemoteKVRestoreService(
                kv_storage_backend,
                catalog=remote_prefix_catalog or RemotePrefixCatalog(),
                planner=planner,
                catalog_key=(
                    make_kv_catalog_key(self.kv_cache_identity, tp_rank=0)
                    if config.remote_kv_persist_catalog
                    else None
                ),
                catalog_identity=(self.kv_cache_identity.digest, 0),
                catalog_timeout_s=config.remote_kv_catalog_timeout_s,
            )
        self.scheduler = Scheduler(
            config,
            remote_restore_service=self.remote_restore_service,
        )
        self.config = config
        self.pending_remote_writebacks = []
        self.remote_writeback_errors = deque(maxlen=128)
        self._step_token_events = ()
        self._exited = False
        atexit.register(self.exit)

    def exit(self):
        if self._exited:
            return
        self._exited = True
        try:
            if self.remote_restore_service is not None:
                self._process_remote_writebacks(wait=True)
                self.remote_restore_service.close()
        finally:
            try:
                self.model_runner.call("exit")
                del self.model_runner
                for p in self.ps:
                    p.join()
            finally:
                if self.kv_storage_backend is not None:
                    self.kv_storage_backend.close()

    def _describe_remote_prefix(self, token_ids, num_pages):
        block_size = self.config.kvcache_block_size
        required_tokens = num_pages * block_size
        if num_pages <= 0 or len(token_ids) < required_tokens:
            raise ValueError("token_ids do not contain every remote KV page")
        block_keys = tuple(
            tuple(token_ids[index * block_size:(index + 1) * block_size])
            for index in range(num_pages)
        )
        descriptors = tuple(
            RemotePageDescriptor(
                object_key=make_kv_page_key(
                    self.kv_cache_identity,
                    tp_rank=0,
                    page_index=index,
                    prefix_token_ids=token_ids[:(index + 1) * block_size],
                ),
                identity_digest=self.kv_cache_identity.digest,
                tp_rank=0,
                page_index=index,
            )
            for index in range(num_pages)
        )
        return block_keys, descriptors

    def register_remote_prefix(self, token_ids, envelopes, timeout=None):
        if self.remote_restore_service is None:
            raise RuntimeError("LLMEngine was created without kv_storage_backend")
        envelopes = tuple(envelopes)
        if not envelopes:
            raise ValueError("at least one remote KV page is required")
        block_keys, descriptors = self._describe_remote_prefix(
            token_ids,
            len(envelopes),
        )
        self.remote_restore_service.publish_prefix(
            block_keys,
            descriptors,
            envelopes,
            timeout=timeout,
        )
        return descriptors

    def _enqueue_remote_writebacks(self, seqs):
        service = self.remote_restore_service
        if service is None or not self.config.remote_kv_writeback:
            return

        block_size = self.config.kvcache_block_size
        for seq in seqs:
            previous_pages, completed_pages = completed_kv_page_span(
                seq.num_cached_tokens,
                seq.num_scheduled_tokens,
                block_size,
            )
            if (
                completed_pages == previous_pages
                or completed_pages < self.config.remote_kv_min_prefix_blocks
            ):
                continue
            try:
                block_keys, descriptors = self._describe_remote_prefix(
                    seq.token_ids,
                    completed_pages,
                )
                required = service.pages_requiring_write(
                    block_keys,
                    descriptors,
                )
                envelopes = {
                    index: self.model_runner.call(
                        "export_kv_page",
                        seq.block_table[index],
                        self.kv_cache_identity.digest,
                        index,
                    )
                    for index in required
                }
                pending = service.submit_writeback(
                    block_keys,
                    descriptors,
                    envelopes,
                )
                if pending is not None:
                    self.pending_remote_writebacks.append(pending)
            except Exception as exc:  # noqa: BLE001 - cache writes must not fail inference
                service.record_writeback_export_failure()
                self.remote_writeback_errors.append(
                    f"{type(exc).__name__}: {exc}"
                )

    def _process_remote_writebacks(self, wait=False):
        if self.remote_restore_service is None:
            return
        for pending in tuple(self.pending_remote_writebacks):
            if not wait and not pending.done:
                continue
            try:
                self.remote_restore_service.complete_writeback(
                    pending,
                    timeout=10 if wait else None,
                )
            except Exception as exc:  # noqa: BLE001 - cache writes must not fail inference
                self.remote_restore_service.fail_writeback(pending)
                self.remote_writeback_errors.append(
                    f"{type(exc).__name__}: {exc}"
                )
            finally:
                self.pending_remote_writebacks.remove(pending)
        self.remote_restore_service.flush_catalog_saves()

    def _process_ready_remote_restores(self):
        for state in self.scheduler.ready_remote_restores():
            try:
                envelopes = state.transfer.result()
                for page, block_id, envelope in zip(
                    state.transfer.pages,
                    state.transfer.block_ids,
                    envelopes,
                ):
                    self.model_runner.call(
                        "import_kv_page",
                        block_id,
                        envelope,
                        page.identity_digest,
                        page.page_index,
                    )
                self.scheduler.complete_remote_restore(state.sequence.seq_id)
            except Exception as exc:  # noqa: BLE001 - remote failures fall back to prefill
                self.scheduler.fail_remote_restore(state.sequence.seq_id, exc)

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

    def cancel_request(self, seq_id: int) -> bool:
        return self.scheduler.cancel(seq_id)

    def take_step_token_events(self):
        events = self._step_token_events
        self._step_token_events = ()
        return events

    def step(self):
        self._step_token_events = ()
        while True:
            self._process_remote_writebacks()
            self._process_ready_remote_restores()
            seqs, is_prefill = self.scheduler.schedule()
            if seqs:
                break
            if not self.scheduler.has_pending_restores:
                raise RuntimeError("scheduler produced no runnable requests")
            self.remote_restore_service.wait_for_any(
                state.transfer
                for state in self.scheduler.pending_restores.values()
            )
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        started = perf_counter()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        elapsed_ms = (perf_counter() - started) * 1000.0
        observed_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else len(seqs)
        self.scheduler.observe_execution(is_prefill, observed_tokens, elapsed_ms)
        self._enqueue_remote_writebacks(seqs)
        previous_completion_lengths = {
            seq.seq_id: seq.num_completion_tokens for seq in seqs
        }
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        self._step_token_events = tuple(
            (seq.seq_id, token_id)
            for seq, token_id in zip(seqs, token_ids)
            if seq.num_completion_tokens > previous_completion_lengths[seq.seq_id]
        )
        self._process_remote_writebacks()
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

    def get_request_metrics(self, seq_id: int):
        metrics = self.scheduler.request_metrics(seq_id)
        return metrics.to_dict() if metrics is not None else None
