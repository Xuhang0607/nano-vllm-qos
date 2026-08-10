from dataclasses import asdict, dataclass
from statistics import median
from typing import TYPE_CHECKING, Iterable, Optional
from typing import Sequence as TypingSequence

if TYPE_CHECKING:
    from nanovllm.engine.sequence import Sequence


@dataclass(slots=True, frozen=True)
class RequestQoS:
    priority: int = 0
    ttft_slo_ms: Optional[float] = None
    tpot_slo_ms: Optional[float] = None
    e2e_slo_ms: Optional[float] = None
    request_class: str = "best_effort"

    def __post_init__(self):
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.ttft_slo_ms is not None and self.ttft_slo_ms <= 0:
            raise ValueError("ttft_slo_ms must be positive")
        if self.tpot_slo_ms is not None and self.tpot_slo_ms <= 0:
            raise ValueError("tpot_slo_ms must be positive")
        if self.e2e_slo_ms is not None and self.e2e_slo_ms <= 0:
            raise ValueError("e2e_slo_ms must be positive")
        if not self.request_class:
            raise ValueError("request_class must not be empty")


@dataclass(slots=True, frozen=True)
class RequestMetrics:
    seq_id: int
    request_class: str
    priority: int
    prompt_tokens: int
    completion_tokens: int
    queue_ms: float
    ttft_ms: float
    tpot_ms: Optional[float]
    e2e_ms: float
    preemptions: int
    kv_restore_wait_ms: float
    kv_restored_tokens: int
    kv_restore_failures: int
    ttft_slo_ms: Optional[float]
    tpot_slo_ms: Optional[float]
    e2e_slo_ms: Optional[float]
    ttft_slo_met: Optional[bool]
    tpot_slo_met: Optional[bool]
    e2e_slo_met: Optional[bool]

    def to_dict(self):
        return asdict(self)


@dataclass(slots=True)
class ExecutionTimeEstimator:
    prefill_ms_per_token: float
    decode_ms_per_token: float
    alpha: float

    def __post_init__(self):
        if self.prefill_ms_per_token <= 0 or self.decode_ms_per_token <= 0:
            raise ValueError("initial execution-time estimates must be positive")
        if not 0 < self.alpha <= 1:
            raise ValueError("EWMA alpha must be in (0, 1]")

    def observe(self, is_prefill: bool, num_tokens: int, elapsed_ms: float):
        if num_tokens <= 0 or elapsed_ms <= 0:
            return
        sample = elapsed_ms / num_tokens
        if is_prefill:
            self.prefill_ms_per_token = (
                self.alpha * sample + (1.0 - self.alpha) * self.prefill_ms_per_token
            )
        else:
            self.decode_ms_per_token = (
                self.alpha * sample + (1.0 - self.alpha) * self.decode_ms_per_token
            )

    def remaining_prefill_ms(self, seq: "Sequence") -> float:
        remaining_tokens = max(0, seq.num_tokens - seq.num_cached_tokens)
        return remaining_tokens * self.prefill_ms_per_token

    def remaining_decode_ms(self, seq: "Sequence") -> float:
        remaining_tokens = max(0, seq.max_tokens - seq.num_completion_tokens)
        return remaining_tokens * self.decode_ms_per_token


def percentile(values: Iterable[float], ratio: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_metrics(metrics: TypingSequence[RequestMetrics]):
    if not metrics:
        return {
            "completed_requests": 0,
            "queue_ms_p50": 0.0,
            "queue_ms_p95": 0.0,
            "ttft_ms_p50": 0.0,
            "ttft_ms_p95": 0.0,
            "tpot_ms_p50": 0.0,
            "tpot_ms_p95": 0.0,
            "e2e_ms_p50": 0.0,
            "e2e_ms_p95": 0.0,
            "preemptions": 0,
            "kv_restore_wait_ms_p50": 0.0,
            "kv_restore_wait_ms_p95": 0.0,
            "kv_restored_tokens": 0,
            "kv_restore_failures": 0,
            "ttft_slo_attainment": None,
            "tpot_slo_attainment": None,
            "e2e_slo_attainment": None,
        }

    queue_values = [item.queue_ms for item in metrics]
    ttft_values = [item.ttft_ms for item in metrics]
    tpot_values = [item.tpot_ms for item in metrics if item.tpot_ms is not None]
    e2e_values = [item.e2e_ms for item in metrics]
    restore_wait_values = [item.kv_restore_wait_ms for item in metrics]
    ttft_slo = [item.ttft_slo_met for item in metrics if item.ttft_slo_met is not None]
    tpot_slo = [item.tpot_slo_met for item in metrics if item.tpot_slo_met is not None]
    e2e_slo = [item.e2e_slo_met for item in metrics if item.e2e_slo_met is not None]
    return {
        "completed_requests": len(metrics),
        "queue_ms_p50": median(queue_values),
        "queue_ms_p95": percentile(queue_values, 0.95),
        "ttft_ms_p50": median(ttft_values),
        "ttft_ms_p95": percentile(ttft_values, 0.95),
        "tpot_ms_p50": median(tpot_values) if tpot_values else 0.0,
        "tpot_ms_p95": percentile(tpot_values, 0.95),
        "e2e_ms_p50": median(e2e_values),
        "e2e_ms_p95": percentile(e2e_values, 0.95),
        "preemptions": sum(item.preemptions for item in metrics),
        "kv_restore_wait_ms_p50": median(restore_wait_values),
        "kv_restore_wait_ms_p95": percentile(restore_wait_values, 0.95),
        "kv_restored_tokens": sum(item.kv_restored_tokens for item in metrics),
        "kv_restore_failures": sum(item.kv_restore_failures for item in metrics),
        "ttft_slo_attainment": sum(ttft_slo) / len(ttft_slo) if ttft_slo else None,
        "tpot_slo_attainment": sum(tpot_slo) / len(tpot_slo) if tpot_slo else None,
        "e2e_slo_attainment": sum(e2e_slo) / len(e2e_slo) if e2e_slo else None,
    }


class SchedulingPolicy:
    def should_schedule_prefill(self, waiting, running, step: int, now: float) -> bool:
        return bool(waiting)

    def select_waiting(self, waiting, step: int, now: float):
        raise NotImplementedError

    def select_running(self, running, step: int, now: float):
        raise NotImplementedError

    def select_preemption_victim(self, running, step: int, now: float):
        raise NotImplementedError

    def budget_ms(self, seq: "Sequence", step: int, now: float) -> Optional[float]:
        return None


class FCFSPolicy(SchedulingPolicy):
    def select_waiting(self, waiting, step: int, now: float):
        return waiting[0]

    def select_preemption_victim(self, running, step: int, now: float):
        return running[-1] if running else None

    def select_running(self, running, step: int, now: float):
        return running[0]


class PriorityAwareLatencyBudgetPolicy(SchedulingPolicy):
    def __init__(
        self,
        estimator: ExecutionTimeEstimator,
        best_effort_slo_ms: float,
        priority_boost_ms: float,
        aging_ms_per_step: float,
    ):
        self.estimator = estimator
        self.best_effort_slo_ms = best_effort_slo_ms
        self.priority_boost_ms = priority_boost_ms
        self.aging_ms_per_step = aging_ms_per_step

    def _remaining_to_finish_ms(self, seq: "Sequence"):
        remaining_decode_tokens = max(0, seq.max_tokens - seq.num_completion_tokens)
        remaining = 0.0
        if seq.is_prefill:
            remaining += self.estimator.remaining_prefill_ms(seq)
            remaining_decode_tokens = max(0, remaining_decode_tokens - 1)
        remaining += remaining_decode_tokens * self.estimator.decode_ms_per_token
        return remaining

    def _raw_budget_ms(self, seq: "Sequence", elapsed_ms: float):
        candidates = []
        if seq.first_token_time is None:
            if seq.qos.ttft_slo_ms is not None:
                candidates.append(
                    seq.qos.ttft_slo_ms
                    - elapsed_ms
                    - self.estimator.remaining_prefill_ms(seq)
                )
        elif seq.qos.tpot_slo_ms is not None:
            actual_ttft_ms = (seq.first_token_time - seq.arrival_time) * 1000.0
            next_token_target_ms = (
                actual_ttft_ms
                + seq.num_completion_tokens * seq.qos.tpot_slo_ms
            )
            remaining_to_next_token = (
                self.estimator.remaining_prefill_ms(seq)
                if seq.is_prefill
                else self.estimator.decode_ms_per_token
            )
            candidates.append(
                next_token_target_ms - elapsed_ms - remaining_to_next_token
            )

        if seq.qos.e2e_slo_ms is not None:
            candidates.append(
                seq.qos.e2e_slo_ms
                - elapsed_ms
                - self._remaining_to_finish_ms(seq)
            )
        if not candidates:
            candidates.append(
                self.best_effort_slo_ms
                - elapsed_ms
                - self._remaining_to_finish_ms(seq)
            )
        return min(candidates)

    def budget_ms(self, seq: "Sequence", step: int, now: float) -> float:
        elapsed_ms = max(0.0, (now - seq.arrival_time) * 1000.0)
        raw_budget = self._raw_budget_ms(seq, elapsed_ms)
        priority_credit = seq.qos.priority * self.priority_boost_ms
        aging_credit = max(0, step - seq.arrival_step) * self.aging_ms_per_step
        return raw_budget - priority_credit - aging_credit

    def select_waiting(self, waiting, step: int, now: float):
        return min(
            waiting,
            key=lambda seq: (self.budget_ms(seq, step, now), seq.arrival_step, seq.seq_id),
        )

    def select_running(self, running, step: int, now: float):
        return min(
            running,
            key=lambda seq: (self.budget_ms(seq, step, now), seq.arrival_step, seq.seq_id),
        )

    def should_schedule_prefill(self, waiting, running, step: int, now: float) -> bool:
        if not waiting:
            return False
        if not running:
            return True
        waiting_seq = self.select_waiting(waiting, step, now)
        running_seq = self.select_running(running, step, now)
        return self.budget_ms(waiting_seq, step, now) <= self.budget_ms(
            running_seq, step, now
        )

    def select_preemption_victim(self, running, step: int, now: float):
        if not running:
            return None
        return max(
            running,
            key=lambda seq: (self.budget_ms(seq, step, now), seq.arrival_step, seq.seq_id),
        )


def create_policy(name: str, estimator: ExecutionTimeEstimator, config) -> SchedulingPolicy:
    if name == "fcfs":
        return FCFSPolicy()
    if name == "pals":
        return PriorityAwareLatencyBudgetPolicy(
            estimator=estimator,
            best_effort_slo_ms=config.qos_best_effort_slo_ms,
            priority_boost_ms=config.qos_priority_boost_ms,
            aging_ms_per_step=config.qos_aging_ms_per_step,
        )
    raise ValueError(f"unsupported scheduling policy: {name}")
