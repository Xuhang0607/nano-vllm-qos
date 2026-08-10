import argparse
from os import urandom

from nanovllm.engine.storage_backend import (
    KVCacheIdentity,
    MooncakeKVStore,
    MooncakeStoreConfig,
    make_kv_page_key,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", default="127.0.0.1:50051")
    parser.add_argument("--metadata", default="P2PHANDSHAKE")
    parser.add_argument("--hostname", default="localhost")
    parser.add_argument("--protocol", choices=("tcp", "rdma", "efa"), default="tcp")
    parser.add_argument("--rdma-devices", default="")
    parser.add_argument("--payload-bytes", type=int, default=1024 * 1024)
    args = parser.parse_args()
    if args.payload_bytes <= 0:
        parser.error("payload-bytes must be positive")

    config = MooncakeStoreConfig(
        local_hostname=args.hostname,
        metadata_server=args.metadata,
        protocol=args.protocol,
        rdma_devices=args.rdma_devices,
        master_server_addr=args.master,
    )
    identity = KVCacheIdentity(
        model_id="smoke-model",
        model_revision="test",
        dtype="bf16",
        tp_size=1,
        page_size=256,
    )
    key = make_kv_page_key(identity, 0, 0, range(256))
    payload = urandom(args.payload_bytes)
    with MooncakeKVStore(config) as store:
        store.put(key, payload)
        restored = store.get(key)
        if restored != payload:
            raise RuntimeError("Mooncake round-trip payload mismatch")
        store.remove(key)
        if store.exists(key):
            raise RuntimeError("Mooncake object still exists after removal")
    print(f"Mooncake TCP/RDMA round-trip passed: {args.payload_bytes} bytes")


if __name__ == "__main__":
    main()
