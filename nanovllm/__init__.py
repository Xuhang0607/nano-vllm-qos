from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.qos import RequestQoS

__all__ = ["LLM", "SamplingParams", "RequestQoS"]


def __getattr__(name):
    if name == "LLM":
        from nanovllm.llm import LLM

        return LLM
    raise AttributeError(name)
