from dataclasses import dataclass
from itertools import count
from time import perf_counter, sleep


class MockTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        text = "".join(
            f"<|{message['role']}|>\n{message['content']}\n" for message in messages
        )
        if add_generation_prompt:
            text += "<|assistant|>\n"
        return [ord(character) for character in text] if tokenize else text

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token_id) for token_id in token_ids)


@dataclass(slots=True)
class _MockSequence:
    prompt_ids: list[int]
    output_ids: list[int]
    max_tokens: int
    arrival_time: float
    first_token_time: float | None = None


class MockEngine:
    """Dependency-light streaming backend used only for UI and API development."""

    def __init__(self, step_delay_s=0.018):
        self.tokenizer = MockTokenizer()
        self.step_delay_s = step_delay_s
        self._ids = count()
        self._active = {}
        self._events = ()
        self._completed = {}
        self._completed_count = 0
        self._cancelled_count = 0
        self._known_prompts = set()
        self._cache_lookups = 0
        self._cache_hits = 0

    def add_request(self, prompt, sampling_params, qos=None):
        seq_id = next(self._ids)
        prompt = list(prompt)
        prompt_key = tuple(prompt)
        self._cache_lookups += 1
        if prompt_key in self._known_prompts:
            self._cache_hits += 1
        self._known_prompts.add(prompt_key)
        response = (
            "这是 nano-vLLM 的本地流式响应。当前请求已经进入连续批处理队列，"
            "服务端会逐 token 返回结果，并同步记录 TTFT、端到端延迟和前缀缓存指标。"
        )
        self._active[seq_id] = _MockSequence(
            prompt_ids=prompt,
            output_ids=[ord(character) for character in response],
            max_tokens=sampling_params.max_tokens,
            arrival_time=perf_counter(),
        )
        return seq_id

    def step(self):
        sleep(self.step_delay_s)
        events = []
        outputs = []
        for seq_id, sequence in tuple(self._active.items()):
            index = len(self._completed.get(("partial", seq_id), []))
            limit = min(sequence.max_tokens, len(sequence.output_ids))
            generated = self._completed.setdefault(("partial", seq_id), [])
            if index < limit:
                token_id = sequence.output_ids[index]
                generated.append(token_id)
                events.append((seq_id, token_id))
                if sequence.first_token_time is None:
                    sequence.first_token_time = perf_counter()
            if len(generated) >= limit:
                now = perf_counter()
                ttft_ms = (sequence.first_token_time - sequence.arrival_time) * 1000.0
                e2e_ms = (now - sequence.arrival_time) * 1000.0
                tpot_ms = (
                    (e2e_ms - ttft_ms) / (len(generated) - 1)
                    if len(generated) > 1
                    else None
                )
                self._completed[seq_id] = {
                    "seq_id": seq_id,
                    "prompt_tokens": len(sequence.prompt_ids),
                    "completion_tokens": len(generated),
                    "queue_ms": 0.0,
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms,
                    "e2e_ms": e2e_ms,
                    "preemptions": 0,
                    "kv_restore_wait_ms": 0.0,
                    "kv_restored_tokens": 0,
                    "kv_restore_failures": 0,
                }
                outputs.append((seq_id, list(generated)))
                self._active.pop(seq_id)
                self._completed.pop(("partial", seq_id), None)
                self._completed_count += 1
        self._events = tuple(events)
        return outputs, -max(1, len(events))

    def take_step_token_events(self):
        events = self._events
        self._events = ()
        return events

    def cancel_request(self, seq_id):
        if self._active.pop(seq_id, None) is None:
            return False
        self._completed.pop(("partial", seq_id), None)
        self._cancelled_count += 1
        return True

    def get_request_metrics(self, seq_id):
        return self._completed.get(seq_id)

    def get_scheduler_metrics(self):
        return {
            "policy": "pals",
            "max_model_len": 4096,
            "completed_requests": self._completed_count,
            "cancelled_requests": self._cancelled_count,
            "waiting_requests": 0,
            "running_requests": len(self._active),
            "ttft_ms_p50": 0.0,
            "ttft_ms_p95": 0.0,
            "tpot_ms_p50": 0.0,
            "tpot_ms_p95": 0.0,
            "e2e_ms_p50": 0.0,
            "e2e_ms_p95": 0.0,
            "prefix_cache_backend": "radix",
            "prefix_cache_lookups": self._cache_lookups,
            "prefix_cache_hit_blocks": self._cache_hits,
            "prefix_cache_block_hit_rate": (
                self._cache_hits / self._cache_lookups if self._cache_lookups else 0.0
            ),
            "remote_restore_completed": 0,
            "kv_restored_tokens": 0,
        }

    def exit(self):
        self._active.clear()
