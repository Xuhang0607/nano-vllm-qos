from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["developer", "system", "user", "assistant"]
    content: str = Field(min_length=1)


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=1.0, gt=1e-10)
    top_p: float = Field(default=1.0, gt=0, le=1)
    max_tokens: int | None = Field(default=128, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    n: int = Field(default=1, ge=1)
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    priority: int = Field(default=0, ge=0)
    ttft_slo_ms: float | None = Field(default=None, gt=0)
    tpot_slo_ms: float | None = Field(default=None, gt=0)
    e2e_slo_ms: float | None = Field(default=None, gt=0)
    request_class: str = Field(default="interactive", min_length=1)

    @model_validator(mode="after")
    def validate_supported_options(self):
        if self.n != 1:
            raise ValueError("nano-vLLM currently supports n=1 only")
        if self.top_p != 1.0:
            raise ValueError("nano-vLLM currently supports top_p=1 only")
        if self.stop is not None:
            raise ValueError("nano-vLLM does not currently support stop sequences")
        return self

    @property
    def output_limit(self):
        return self.max_completion_tokens or self.max_tokens or 128

    @property
    def include_usage(self):
        return bool(self.stream_options and self.stream_options.include_usage)
