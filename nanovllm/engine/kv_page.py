from __future__ import annotations

import hashlib
import hmac
import json
import math
import struct
from dataclasses import dataclass

_MAGIC = b"NVKVPAGE"
_VERSION = 1
_PREFIX = struct.Struct("<8sHIQ")
_CHECKSUM_SIZE = 16
_MAX_HEADER_BYTES = 64 * 1024
_DEFAULT_MAX_PAYLOAD_BYTES = 8 * 1024**3
_DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}


class KVPageFormatError(ValueError):
    pass


class KVPageCompatibilityError(ValueError):
    pass


def canonical_dtype(dtype: str) -> str:
    value = str(dtype).removeprefix("torch.").removeprefix("numpy.")
    aliases = {
        "half": "float16",
        "fp16": "float16",
        "bf16": "bfloat16",
        "float": "float32",
        "fp32": "float32",
    }
    value = aliases.get(value, value)
    if value not in _DTYPE_BYTES:
        raise ValueError(f"unsupported KV page dtype: {dtype}")
    return value


@dataclass(slots=True, frozen=True)
class KVPageLayout:
    num_layers: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    dtype: str

    def __post_init__(self):
        if min(
            self.num_layers,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
        ) <= 0:
            raise ValueError("KV page layout dimensions must be positive")
        object.__setattr__(self, "dtype", canonical_dtype(self.dtype))

    @property
    def shape(self):
        return (
            2,
            self.num_layers,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
        )

    @property
    def nbytes(self):
        return math.prod(self.shape) * _DTYPE_BYTES[self.dtype]

    def to_dict(self):
        return {
            "num_layers": self.num_layers,
            "block_size": self.block_size,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise KVPageFormatError("KV page layout must be an object")
        expected = {
            "num_layers",
            "block_size",
            "num_kv_heads",
            "head_dim",
            "dtype",
        }
        if set(value) != expected:
            raise KVPageFormatError("KV page layout fields are invalid")
        try:
            return cls(**value)
        except (TypeError, ValueError) as exc:
            raise KVPageFormatError(f"invalid KV page layout: {exc}") from exc


@dataclass(slots=True, frozen=True)
class KVPageMetadata:
    identity_digest: str
    tp_rank: int
    page_index: int
    layout: KVPageLayout

    def __post_init__(self):
        if not self.identity_digest or not self.identity_digest.isascii():
            raise ValueError("identity_digest must be a non-empty ASCII string")
        if self.tp_rank < 0 or self.page_index < 0:
            raise ValueError("tp_rank and page_index must be non-negative")

    def to_dict(self):
        return {
            "identity_digest": self.identity_digest,
            "tp_rank": self.tp_rank,
            "page_index": self.page_index,
            "layout": self.layout.to_dict(),
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise KVPageFormatError("KV page metadata must be an object")
        if set(value) != {"identity_digest", "tp_rank", "page_index", "layout"}:
            raise KVPageFormatError("KV page metadata fields are invalid")
        try:
            return cls(
                identity_digest=value["identity_digest"],
                tp_rank=value["tp_rank"],
                page_index=value["page_index"],
                layout=KVPageLayout.from_dict(value["layout"]),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, KVPageFormatError):
                raise
            raise KVPageFormatError(f"invalid KV page metadata: {exc}") from exc


@dataclass(slots=True, frozen=True)
class KVPageRecord:
    metadata: KVPageMetadata
    payload: bytes


class KVPageCodec:
    @classmethod
    def encode(cls, metadata: KVPageMetadata, payload) -> bytes:
        payload = bytes(payload)
        if len(payload) != metadata.layout.nbytes:
            raise KVPageCompatibilityError(
                f"KV page payload has {len(payload)} bytes, "
                f"expected {metadata.layout.nbytes}"
            )

        header = json.dumps(
            metadata.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        if len(header) > _MAX_HEADER_BYTES:
            raise KVPageFormatError("KV page header is too large")
        prefix = _PREFIX.pack(_MAGIC, _VERSION, len(header), len(payload))
        checksum = hashlib.blake2b(
            prefix + header + payload,
            digest_size=_CHECKSUM_SIZE,
        ).digest()
        return prefix + header + payload + checksum

    @classmethod
    def decode(
        cls,
        blob,
        *,
        expected_identity_digest: str | None = None,
        expected_tp_rank: int | None = None,
        expected_page_index: int | None = None,
        expected_layout: KVPageLayout | None = None,
        max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> KVPageRecord:
        view = memoryview(blob)
        minimum_size = _PREFIX.size + _CHECKSUM_SIZE
        if len(view) < minimum_size:
            raise KVPageFormatError("KV page envelope is truncated")
        if max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")

        magic, version, header_size, payload_size = _PREFIX.unpack(
            view[: _PREFIX.size]
        )
        if magic != _MAGIC:
            raise KVPageFormatError("KV page magic is invalid")
        if version != _VERSION:
            raise KVPageFormatError(f"unsupported KV page version: {version}")
        if header_size > _MAX_HEADER_BYTES:
            raise KVPageFormatError("KV page header is too large")
        if payload_size > max_payload_bytes:
            raise KVPageFormatError("KV page payload exceeds the configured limit")

        expected_size = _PREFIX.size + header_size + payload_size + _CHECKSUM_SIZE
        if len(view) != expected_size:
            raise KVPageFormatError(
                f"KV page envelope has {len(view)} bytes, expected {expected_size}"
            )

        checksum_start = expected_size - _CHECKSUM_SIZE
        expected_checksum = hashlib.blake2b(
            view[:checksum_start],
            digest_size=_CHECKSUM_SIZE,
        ).digest()
        actual_checksum = bytes(view[checksum_start:])
        if not hmac.compare_digest(expected_checksum, actual_checksum):
            raise KVPageFormatError("KV page checksum mismatch")

        header_start = _PREFIX.size
        payload_start = header_start + header_size
        try:
            header = json.loads(bytes(view[header_start:payload_start]).decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise KVPageFormatError("KV page header is not valid JSON") from exc

        metadata = KVPageMetadata.from_dict(header)
        if metadata.layout.nbytes != payload_size:
            raise KVPageFormatError(
                "KV page layout byte size does not match the encoded payload"
            )
        cls._check_compatibility(
            metadata,
            expected_identity_digest,
            expected_tp_rank,
            expected_page_index,
            expected_layout,
        )
        return KVPageRecord(
            metadata=metadata,
            payload=bytes(view[payload_start:checksum_start]),
        )

    @staticmethod
    def _check_compatibility(
        metadata: KVPageMetadata,
        expected_identity_digest: str | None,
        expected_tp_rank: int | None,
        expected_page_index: int | None,
        expected_layout: KVPageLayout | None,
    ):
        checks = (
            ("identity", expected_identity_digest, metadata.identity_digest),
            ("TP rank", expected_tp_rank, metadata.tp_rank),
            ("page index", expected_page_index, metadata.page_index),
            ("layout", expected_layout, metadata.layout),
        )
        for name, expected, actual in checks:
            if expected is not None and expected != actual:
                raise KVPageCompatibilityError(
                    f"KV page {name} mismatch: expected {expected}, got {actual}"
                )


class TorchKVPageIO:
    """Pack and restore one physical KV block across K/V and all layers."""

    def __init__(self, kv_cache):
        if getattr(kv_cache, "ndim", None) != 6:
            raise ValueError(
                "kv_cache must have shape "
                "[2, layers, blocks, block_size, kv_heads, head_dim]"
            )
        if kv_cache.shape[0] != 2:
            raise ValueError("kv_cache axis 0 must contain K and V")
        self.kv_cache = kv_cache
        self.layout = KVPageLayout(
            num_layers=kv_cache.shape[1],
            block_size=kv_cache.shape[3],
            num_kv_heads=kv_cache.shape[4],
            head_dim=kv_cache.shape[5],
            dtype=str(kv_cache.dtype),
        )

    @property
    def num_blocks(self):
        return self.kv_cache.shape[2]

    def _target(self, block_id: int):
        if not 0 <= block_id < self.num_blocks:
            raise IndexError(f"KV block id {block_id} is out of range")
        return self.kv_cache[:, :, block_id]

    def export_page(self, block_id: int) -> bytes:
        import torch

        page = self._target(block_id)
        is_cuda = page.device.type == "cuda"
        staging = torch.empty(
            self.layout.shape,
            dtype=page.dtype,
            device="cpu",
            pin_memory=is_cuda,
        )
        staging.copy_(page, non_blocking=is_cuda)
        if is_cuda:
            torch.cuda.current_stream(page.device).synchronize()
        return staging.view(torch.uint8).reshape(-1).numpy().tobytes()

    def import_page(self, block_id: int, payload) -> None:
        import torch

        payload = bytes(payload)
        if len(payload) != self.layout.nbytes:
            raise KVPageCompatibilityError(
                f"KV page payload has {len(payload)} bytes, "
                f"expected {self.layout.nbytes}"
            )

        target = self._target(block_id)
        is_cuda = target.device.type == "cuda"
        staging = torch.empty(
            self.layout.shape,
            dtype=target.dtype,
            device="cpu",
            pin_memory=is_cuda,
        )
        source = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        staging.view(torch.uint8).reshape(-1).copy_(source)
        target.copy_(staging, non_blocking=is_cuda)
        if is_cuda:
            torch.cuda.current_stream(target.device).synchronize()


class AsyncKVPageStore:
    def __init__(self, coordinator):
        self.coordinator = coordinator

    async def put(self, key: str, metadata: KVPageMetadata, payload) -> None:
        envelope = KVPageCodec.encode(metadata, payload)
        await self.coordinator.put(key, envelope)

    async def get(
        self,
        key: str,
        *,
        expected_identity_digest: str | None = None,
        expected_tp_rank: int | None = None,
        expected_page_index: int | None = None,
        expected_layout: KVPageLayout | None = None,
    ) -> KVPageRecord | None:
        envelope = await self.coordinator.get(key)
        if envelope is None:
            return None
        return KVPageCodec.decode(
            envelope,
            expected_identity_digest=expected_identity_digest,
            expected_tp_rank=expected_tp_rank,
            expected_page_index=expected_page_index,
            expected_layout=expected_layout,
        )

    async def evict(self, key: str) -> None:
        await self.coordinator.evict(key)
