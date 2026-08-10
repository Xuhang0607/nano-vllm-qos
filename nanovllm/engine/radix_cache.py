from dataclasses import dataclass, field
from typing import Iterable, Optional


BlockKey = tuple[int, ...]


@dataclass(eq=False, slots=True)
class RadixNode:
    edge_keys: tuple[BlockKey, ...] = ()
    block_ids: tuple[int, ...] = ()
    parent: Optional["RadixNode"] = None
    children: dict[BlockKey, "RadixNode"] = field(default_factory=dict)
    last_access: int = 0


class RadixPrefixCache:
    """Path-compressed prefix index whose symbols are page-sized token blocks."""

    def __init__(self):
        self.root = RadixNode()
        self.block_locations: dict[int, tuple[RadixNode, int]] = {}
        self.clock = 0
        self.lookups = 0
        self.queried_blocks = 0
        self.hit_blocks = 0

    @staticmethod
    def _common_prefix_length(left, right) -> int:
        length = min(len(left), len(right))
        for index in range(length):
            if left[index] != right[index]:
                return index
        return length

    def _touch(self, node: RadixNode):
        self.clock += 1
        node.last_access = self.clock

    def _index_node(self, node: RadixNode):
        for index, block_id in enumerate(node.block_ids):
            self.block_locations[block_id] = (node, index)

    def match(self, keys: Iterable[BlockKey]) -> list[int]:
        keys = tuple(keys)
        self.lookups += 1
        self.queried_blocks += len(keys)
        node = self.root
        position = 0
        matched = []

        while position < len(keys):
            child = node.children.get(keys[position])
            if child is None:
                break
            common = self._common_prefix_length(child.edge_keys, keys[position:])
            if common == 0:
                break
            matched.extend(child.block_ids[:common])
            self._touch(child)
            position += common
            if common < len(child.edge_keys):
                break
            node = child

        self.hit_blocks += len(matched)
        return matched

    def insert(self, keys: Iterable[BlockKey], block_ids: Iterable[int]):
        keys = tuple(keys)
        block_ids = tuple(block_ids)
        if len(keys) != len(block_ids):
            raise ValueError("keys and block_ids must have the same length")
        if not keys:
            return
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("a radix path cannot contain duplicate block ids")

        node = self.root
        position = 0
        while position < len(keys):
            child = node.children.get(keys[position])
            if child is None:
                child = RadixNode(
                    edge_keys=keys[position:],
                    block_ids=block_ids[position:],
                    parent=node,
                )
                node.children[child.edge_keys[0]] = child
                self._index_node(child)
                self._touch(child)
                return

            common = self._common_prefix_length(child.edge_keys, keys[position:])
            # Concurrent misses may compute duplicate physical KV blocks for the
            # same token prefix. Keep the existing path as the canonical copy.

            if common == len(child.edge_keys):
                self._touch(child)
                node = child
                position += common
                continue

            if position + common == len(keys):
                self._touch(child)
                return

            split = RadixNode(
                edge_keys=child.edge_keys[:common],
                block_ids=child.block_ids[:common],
                parent=node,
            )
            node.children[split.edge_keys[0]] = split
            child.edge_keys = child.edge_keys[common:]
            child.block_ids = child.block_ids[common:]
            child.parent = split
            split.children[child.edge_keys[0]] = child
            self._index_node(split)
            self._index_node(child)
            self._touch(split)

            position += common
            if position < len(keys):
                new_child = RadixNode(
                    edge_keys=keys[position:],
                    block_ids=block_ids[position:],
                    parent=split,
                )
                split.children[new_child.edge_keys[0]] = new_child
                self._index_node(new_child)
                self._touch(new_child)
            return

    def _drop_subtree(self, node: RadixNode):
        for child in tuple(node.children.values()):
            self._drop_subtree(child)
        for block_id in node.block_ids:
            self.block_locations.pop(block_id, None)

    def _compress(self, node: RadixNode):
        while node is not self.root:
            if len(node.children) == 1:
                child = next(iter(node.children.values()))
                offset = len(node.block_ids)
                node.edge_keys += child.edge_keys
                node.block_ids += child.block_ids
                node.children = child.children
                for grandchild in node.children.values():
                    grandchild.parent = node
                for index, child_block_id in enumerate(
                    child.block_ids, start=offset
                ):
                    self.block_locations[child_block_id] = (node, index)
                self._touch(node)
                continue
            node = node.parent

    def remove_block(self, block_id: int) -> bool:
        location = self.block_locations.get(block_id)
        if location is None:
            return False
        node, index = location

        if index == 0:
            parent = node.parent
            del parent.children[node.edge_keys[0]]
            self._drop_subtree(node)
            self._compress(parent)
            return True

        for child in tuple(node.children.values()):
            self._drop_subtree(child)
        for removed_id in node.block_ids[index:]:
            self.block_locations.pop(removed_id, None)
        node.edge_keys = node.edge_keys[:index]
        node.block_ids = node.block_ids[:index]
        node.children.clear()
        self._compress(node.parent)
        return True

    def lru_leaf_block(self, eligible_block_ids: set[int]) -> Optional[int]:
        candidates = []

        def visit(node: RadixNode):
            if not node.children:
                for block_id in reversed(node.block_ids):
                    if block_id in eligible_block_ids:
                        candidates.append((node.last_access, block_id))
                        break
                return
            for child in node.children.values():
                visit(child)

        for child in self.root.children.values():
            visit(child)
        if not candidates:
            return None
        return min(candidates)[1]

    def stats(self):
        def count_nodes(node: RadixNode):
            return sum(1 + count_nodes(child) for child in node.children.values())

        return {
            "radix_nodes": count_nodes(self.root),
            "radix_cached_blocks": len(self.block_locations),
            "radix_lookups": self.lookups,
            "radix_block_hit_rate": (
                self.hit_blocks / self.queried_blocks if self.queried_blocks else 0.0
            ),
        }

    def validate(self):
        locations = {}

        def visit(node: RadixNode):
            if node is not self.root:
                assert node.edge_keys
                assert len(node.edge_keys) == len(node.block_ids)
                assert node.parent.children[node.edge_keys[0]] is node
                assert len(node.children) != 1
                for index, block_id in enumerate(node.block_ids):
                    assert block_id not in locations
                    locations[block_id] = (node, index)
            for first_key, child in node.children.items():
                assert first_key == child.edge_keys[0]
                assert child.parent is node
                visit(child)

        visit(self.root)
        assert locations == self.block_locations
