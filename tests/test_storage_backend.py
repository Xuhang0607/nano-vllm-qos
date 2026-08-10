import pytest

from nanovllm.engine.storage_backend import (
    InMemoryKVStore,
    KVCacheIdentity,
    MooncakeKVStore,
    MooncakeStoreConfig,
    make_kv_page_key,
)


class FakeMooncakeStore:
    def __init__(self):
        self.setup_args = None
        self.objects = {}
        self.close_count = 0

    def setup(self, *args):
        self.setup_args = args

    def put(self, key, payload):
        self.objects[key] = payload
        return 0

    def get(self, key):
        return self.objects.get(key)

    def is_exist(self, key):
        return int(key in self.objects)

    def remove(self, key):
        self.objects.pop(key, None)
        return 0

    def upsert_batch(self, keys, payloads):
        self.objects.update(zip(keys, payloads))
        return 0

    def get_batch(self, keys):
        return [self.objects.get(key) for key in keys]

    def batch_is_exist(self, keys):
        return [int(key in self.objects) for key in keys]

    def batch_remove(self, keys):
        for key in keys:
            self.objects.pop(key, None)
        return [0] * len(keys)

    def close(self):
        self.close_count += 1


def exercise_backend(backend):
    backend.put("page", b"kv-data")
    assert backend.exists("page")
    assert backend.get("page") == b"kv-data"
    backend.remove("page")
    assert not backend.exists("page")
    backend.put_many({"page-a": b"a", "page-b": b"b"})
    assert backend.exists_many(["page-a", "missing"]) == {
        "page-a": True,
        "missing": False,
    }
    assert backend.get_many(["page-a", "page-b"]) == {
        "page-a": b"a",
        "page-b": b"b",
    }
    backend.remove_many(["page-a", "page-b"])
    assert not backend.exists("page-a")


def test_in_memory_backend_contract():
    exercise_backend(InMemoryKVStore())


def test_mooncake_adapter_contract_and_lifecycle():
    fake_store = FakeMooncakeStore()
    config = MooncakeStoreConfig(protocol="tcp")
    backend = MooncakeKVStore(config, store=fake_store)

    assert fake_store.setup_args[4] == "tcp"
    exercise_backend(backend)
    backend.close()
    backend.close()
    assert fake_store.close_count == 1


def test_mooncake_adapter_surfaces_status_errors():
    fake_store = FakeMooncakeStore()
    fake_store.put = lambda key, payload: -7
    backend = MooncakeKVStore(MooncakeStoreConfig(), store=fake_store)
    with pytest.raises(RuntimeError, match="status -7"):
        backend.put("page", b"data")


def test_page_key_is_stable_and_isolates_model_rank_and_prefix():
    identity = KVCacheIdentity("Qwen/Qwen3-0.6B", "main", "bf16", 2, 256)
    first = make_kv_page_key(identity, 0, 3, [1, 2, 3, 4])
    assert first == make_kv_page_key(identity, 0, 3, [1, 2, 3, 4])
    assert first != make_kv_page_key(identity, 1, 3, [1, 2, 3, 4])
    assert first != make_kv_page_key(identity, 0, 3, [1, 2, 3, 5])
    other_dtype = KVCacheIdentity("Qwen/Qwen3-0.6B", "main", "fp16", 2, 256)
    assert first != make_kv_page_key(other_dtype, 0, 3, [1, 2, 3, 4])


def test_page_key_rejects_invalid_tokens():
    identity = KVCacheIdentity("model", "main", "bf16", 1, 256)
    with pytest.raises(ValueError):
        make_kv_page_key(identity, 0, 0, [])
    with pytest.raises(ValueError):
        make_kv_page_key(identity, 0, 0, [-1])
    with pytest.raises(ValueError):
        make_kv_page_key(identity, 1, 0, [1])
