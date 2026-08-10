from dataclasses import dataclass
from hashlib import blake2b
from typing import Iterable, Mapping, Protocol


class KVStorageBackend(Protocol):
    def put(self, key: str, payload: bytes): ...

    def get(self, key: str) -> bytes | None: ...

    def exists(self, key: str) -> bool: ...

    def remove(self, key: str): ...

    def put_many(self, objects: Mapping[str, bytes]): ...

    def get_many(self, keys: Iterable[str]) -> dict[str, bytes | None]: ...

    def exists_many(self, keys: Iterable[str]) -> dict[str, bool]: ...

    def remove_many(self, keys: Iterable[str]): ...

    def close(self): ...


class InMemoryKVStore:
    def __init__(self):
        self.objects = {}

    def put(self, key: str, payload: bytes):
        self.objects[key] = bytes(payload)

    def get(self, key: str):
        return self.objects.get(key)

    def exists(self, key: str):
        return key in self.objects

    def remove(self, key: str):
        self.objects.pop(key, None)

    def put_many(self, objects: Mapping[str, bytes]):
        for key, payload in objects.items():
            self.put(key, payload)

    def get_many(self, keys: Iterable[str]):
        return {key: self.get(key) for key in keys}

    def exists_many(self, keys: Iterable[str]):
        return {key: self.exists(key) for key in keys}

    def remove_many(self, keys: Iterable[str]):
        for key in keys:
            self.remove(key)

    def close(self):
        pass


@dataclass(slots=True, frozen=True)
class MooncakeStoreConfig:
    local_hostname: str = "localhost"
    metadata_server: str = "P2PHANDSHAKE"
    global_segment_size: int = 512 * 1024 * 1024
    local_buffer_size: int = 128 * 1024 * 1024
    protocol: str = "tcp"
    rdma_devices: str = ""
    master_server_addr: str = "127.0.0.1:50051"

    def __post_init__(self):
        if not self.local_hostname:
            raise ValueError("local_hostname must not be empty")
        if not self.metadata_server:
            raise ValueError("metadata_server must not be empty")
        if self.global_segment_size < 0 or self.local_buffer_size < 0:
            raise ValueError("Mooncake buffer sizes must be non-negative")
        if self.protocol not in ("tcp", "rdma", "efa"):
            raise ValueError(f"unsupported Mooncake protocol: {self.protocol}")
        if not self.master_server_addr:
            raise ValueError("master_server_addr must not be empty")


@dataclass(slots=True, frozen=True)
class KVCacheIdentity:
    model_id: str
    model_revision: str
    dtype: str
    tp_size: int
    page_size: int
    layout_version: str = "nanovllm-kv-v1"

    def __post_init__(self):
        if not all(
            (self.model_id, self.model_revision, self.dtype, self.layout_version)
        ):
            raise ValueError("KV cache identity strings must not be empty")
        if self.tp_size <= 0 or self.page_size <= 0:
            raise ValueError("tp_size and page_size must be positive")

    @property
    def digest(self):
        encoded = "|".join(
            (
                self.layout_version,
                self.model_id,
                self.model_revision,
                self.dtype,
                str(self.tp_size),
                str(self.page_size),
            )
        ).encode("utf-8")
        return blake2b(encoded, digest_size=12).hexdigest()


class MooncakeKVStore:
    """Small adapter around MooncakeDistributedStore's byte-object API."""

    def __init__(self, config: MooncakeStoreConfig, store=None):
        if store is None:
            try:
                from mooncake.store import MooncakeDistributedStore
            except ImportError as exc:
                raise RuntimeError(
                    "Mooncake support requires the optional mooncake-transfer-engine package"
                ) from exc
            store = MooncakeDistributedStore()
        self.store = store
        self.closed = False
        self.store.setup(
            config.local_hostname,
            config.metadata_server,
            config.global_segment_size,
            config.local_buffer_size,
            config.protocol,
            config.rdma_devices,
            config.master_server_addr,
        )

    @staticmethod
    def _check_status(status, operation: str):
        if status not in (None, 0):
            raise RuntimeError(f"Mooncake {operation} failed with status {status}")

    def put(self, key: str, payload: bytes):
        self._check_status(self.store.put(key, payload), "put")

    def get(self, key: str):
        return self.store.get(key)

    def exists(self, key: str):
        result = self.store.is_exist(key)
        if result < 0:
            raise RuntimeError(f"Mooncake is_exist failed with status {result}")
        return result == 1

    def remove(self, key: str):
        self._check_status(self.store.remove(key), "remove")

    def put_many(self, objects: Mapping[str, bytes]):
        keys = list(objects)
        payloads = [objects[key] for key in keys]
        self._check_status(self.store.upsert_batch(keys, payloads), "upsert_batch")

    def get_many(self, keys: Iterable[str]):
        keys = list(keys)
        payloads = self.store.get_batch(keys)
        if len(payloads) != len(keys):
            raise RuntimeError("Mooncake get_batch returned an unexpected result count")
        return dict(zip(keys, payloads))

    def exists_many(self, keys: Iterable[str]):
        keys = list(keys)
        statuses = self.store.batch_is_exist(keys)
        if len(statuses) != len(keys) or any(status < 0 for status in statuses):
            raise RuntimeError("Mooncake batch_is_exist failed")
        return {key: status == 1 for key, status in zip(keys, statuses)}

    def remove_many(self, keys: Iterable[str]):
        keys = list(keys)
        statuses = self.store.batch_remove(keys)
        if len(statuses) != len(keys) or any(status != 0 for status in statuses):
            raise RuntimeError("Mooncake batch_remove failed")

    def close(self):
        if not self.closed:
            self.store.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def make_kv_page_key(
    identity: KVCacheIdentity,
    tp_rank: int,
    page_index: int,
    prefix_token_ids: Iterable[int],
    namespace: str = "nanovllm-kv",
):
    if not namespace:
        raise ValueError("namespace must not be empty")
    if not 0 <= tp_rank < identity.tp_size or page_index < 0:
        raise ValueError("tp_rank or page_index is outside the cache identity")

    prefix_digest = blake2b(digest_size=16)
    has_tokens = False
    for token_id in prefix_token_ids:
        if token_id < 0 or token_id >= 2**32:
            raise ValueError("token ids must fit in uint32")
        prefix_digest.update(token_id.to_bytes(4, "little"))
        has_tokens = True
    if not has_tokens:
        raise ValueError("prefix_token_ids must not be empty")
    return (
        f"{namespace}/{identity.digest}/tp-{tp_rank}/page-{page_index}/"
        f"{prefix_digest.hexdigest()}"
    )
