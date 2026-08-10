from copy import copy
from enum import Enum, auto
from itertools import count
from time import perf_counter

from nanovllm.engine.qos import RequestMetrics, RequestQoS
from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params=None,
        qos: RequestQoS | None = None,
        arrival_time: float | None = None,
    ):
        sampling_params = sampling_params or SamplingParams()
        qos = qos or RequestQoS()
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.qos = qos
        self.arrival_time = arrival_time
        self.arrival_step = 0
        self.first_scheduled_time = None
        self.first_token_time = None
        self.finished_time = None
        self.preemption_count = 0
        self.last_budget_ms = None

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def mark_arrived(self, step: int, now: float):
        self.arrival_step = step
        if self.arrival_time is None:
            self.arrival_time = now

    def mark_scheduled(self, now: float):
        if self.first_scheduled_time is None:
            self.first_scheduled_time = now

    def append_token(self, token_id: int, now: float | None = None):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        if self.first_token_time is None:
            self.first_token_time = perf_counter() if now is None else now

    def mark_finished(self, now: float):
        self.finished_time = now

    def metrics(self) -> RequestMetrics:
        if self.arrival_time is None or self.first_token_time is None or self.finished_time is None:
            raise RuntimeError("request metrics are only available after completion")
        scheduled_time = self.first_scheduled_time or self.first_token_time
        queue_ms = (scheduled_time - self.arrival_time) * 1000.0
        ttft_ms = (self.first_token_time - self.arrival_time) * 1000.0
        e2e_ms = (self.finished_time - self.arrival_time) * 1000.0
        tpot_ms = None
        if self.num_completion_tokens > 1:
            tpot_ms = (e2e_ms - ttft_ms) / (self.num_completion_tokens - 1)
        return RequestMetrics(
            seq_id=self.seq_id,
            request_class=self.qos.request_class,
            priority=self.qos.priority,
            prompt_tokens=self.num_prompt_tokens,
            completion_tokens=self.num_completion_tokens,
            queue_ms=queue_ms,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            e2e_ms=e2e_ms,
            preemptions=self.preemption_count,
            ttft_slo_ms=self.qos.ttft_slo_ms,
            tpot_slo_ms=self.qos.tpot_slo_ms,
            e2e_slo_ms=self.qos.e2e_slo_ms,
            ttft_slo_met=(ttft_ms <= self.qos.ttft_slo_ms) if self.qos.ttft_slo_ms else None,
            tpot_slo_met=(tpot_ms <= self.qos.tpot_slo_ms)
            if self.qos.tpot_slo_ms is not None and tpot_ms is not None
            else None,
            e2e_slo_met=(e2e_ms <= self.qos.e2e_slo_ms) if self.qos.e2e_slo_ms else None,
        )

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
