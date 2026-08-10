import json
from dataclasses import dataclass
from hashlib import blake2b
from hmac import compare_digest

CATALOG_SCHEMA = "nanovllm-remote-prefix-catalog"
CATALOG_VERSION = 1


class RemoteCatalogSnapshotError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class CatalogPageRecord:
    object_key: str
    identity_digest: str
    tp_rank: int
    page_index: int

    def __post_init__(self):
        if (
            not isinstance(self.object_key, str)
            or not self.object_key
            or not isinstance(self.identity_digest, str)
            or not self.identity_digest
        ):
            raise ValueError("catalog page keys and identities must not be empty")
        if (
            not isinstance(self.tp_rank, int)
            or isinstance(self.tp_rank, bool)
            or self.tp_rank < 0
            or not isinstance(self.page_index, int)
            or isinstance(self.page_index, bool)
            or self.page_index < 0
        ):
            raise ValueError("catalog TP ranks and page indexes must be non-negative")


@dataclass(slots=True, frozen=True)
class CatalogPrefixRecord:
    block_keys: tuple[tuple[int, ...], ...]
    pages: tuple[CatalogPageRecord, ...]

    def __post_init__(self):
        if not self.block_keys or len(self.block_keys) != len(self.pages):
            raise ValueError("catalog prefix vectors must have the same non-zero length")
        block_sizes = {len(block) for block in self.block_keys}
        if len(block_sizes) != 1 or 0 in block_sizes:
            raise ValueError("catalog token blocks must have one positive page size")
        for index, (block, page) in enumerate(zip(self.block_keys, self.pages)):
            if page.page_index != index:
                raise ValueError("catalog page indexes must start at zero")
            if any(
                not isinstance(token, int)
                or isinstance(token, bool)
                or not 0 <= token < 2**32
                for token in block
            ):
                raise ValueError("catalog token ids must fit in uint32")


@dataclass(slots=True, frozen=True)
class RemoteCatalogSnapshot:
    identity_digest: str
    tp_rank: int
    prefixes: tuple[CatalogPrefixRecord, ...]

    def __post_init__(self):
        if (
            not isinstance(self.identity_digest, str)
            or not self.identity_digest
            or not isinstance(self.tp_rank, int)
            or isinstance(self.tp_rank, bool)
            or self.tp_rank < 0
        ):
            raise ValueError("catalog identity is invalid")
        if not self.prefixes:
            raise ValueError("catalog snapshot must contain at least one prefix")
        page_sizes = {
            len(block)
            for prefix in self.prefixes
            for block in prefix.block_keys
        }
        if len(page_sizes) != 1:
            raise ValueError("catalog prefixes must use one page size")
        for prefix in self.prefixes:
            for page in prefix.pages:
                if (
                    page.identity_digest != self.identity_digest
                    or page.tp_rank != self.tp_rank
                ):
                    raise ValueError("catalog page identity does not match its snapshot")


class RemoteCatalogSnapshotCodec:
    @staticmethod
    def _payload(snapshot: RemoteCatalogSnapshot):
        return {
            "identity_digest": snapshot.identity_digest,
            "tp_rank": snapshot.tp_rank,
            "prefixes": [
                {
                    "block_keys": [list(block) for block in prefix.block_keys],
                    "pages": [
                        {
                            "object_key": page.object_key,
                            "identity_digest": page.identity_digest,
                            "tp_rank": page.tp_rank,
                            "page_index": page.page_index,
                        }
                        for page in prefix.pages
                    ],
                }
                for prefix in snapshot.prefixes
            ],
        }

    @staticmethod
    def _canonical_json(value) -> bytes:
        return json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def encode(cls, snapshot: RemoteCatalogSnapshot) -> bytes:
        payload = cls._payload(snapshot)
        payload_bytes = cls._canonical_json(payload)
        envelope = {
            "schema": CATALOG_SCHEMA,
            "version": CATALOG_VERSION,
            "checksum": blake2b(payload_bytes, digest_size=16).hexdigest(),
            "payload": payload,
        }
        return cls._canonical_json(envelope)

    @classmethod
    def decode(cls, envelope: bytes) -> RemoteCatalogSnapshot:
        try:
            document = json.loads(bytes(envelope).decode("utf-8"))
            if not isinstance(document, dict):
                raise TypeError("catalog envelope must be an object")
            if document.get("schema") != CATALOG_SCHEMA:
                raise ValueError("unsupported catalog schema")
            version = document.get("version")
            if (
                not isinstance(version, int)
                or isinstance(version, bool)
                or version != CATALOG_VERSION
            ):
                raise ValueError("unsupported catalog version")
            payload = document["payload"]
            checksum = document["checksum"]
            if not isinstance(payload, dict) or not isinstance(checksum, str):
                raise TypeError("catalog payload or checksum has an invalid type")
            actual_checksum = blake2b(
                cls._canonical_json(payload),
                digest_size=16,
            ).hexdigest()
            if not compare_digest(checksum, actual_checksum):
                raise ValueError("catalog snapshot checksum mismatch")

            prefixes = tuple(
                CatalogPrefixRecord(
                    block_keys=tuple(
                        tuple(block) for block in prefix["block_keys"]
                    ),
                    pages=tuple(
                        CatalogPageRecord(
                            object_key=page["object_key"],
                            identity_digest=page["identity_digest"],
                            tp_rank=page["tp_rank"],
                            page_index=page["page_index"],
                        )
                        for page in prefix["pages"]
                    ),
                )
                for prefix in payload["prefixes"]
            )
            return RemoteCatalogSnapshot(
                identity_digest=payload["identity_digest"],
                tp_rank=payload["tp_rank"],
                prefixes=prefixes,
            )
        except (KeyError, TypeError, UnicodeDecodeError, ValueError) as exc:
            raise RemoteCatalogSnapshotError(
                f"invalid remote catalog snapshot: {exc}"
            ) from exc
