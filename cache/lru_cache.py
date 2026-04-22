"""
LRU Cache — Doubly Linked List + HashMap
=========================================
Data Structures:
  _Node       : doubly-linked-list node (prev/next pointers)
  self._map   : dict[key -> Node]  O(1) key lookup

Time Complexity:
  get()  : O(1)  hash lookup + O(1) DLL move-to-front
  put()  : O(1)  hash insert + O(1) DLL insert/evict

Space Complexity: O(capacity)
"""

import time


class _Node:
    __slots__ = ("key", "value", "prev", "next", "timestamp")

    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.timestamp: float = time.time()
        self.prev: "_Node | None" = None
        self.next: "_Node | None" = None


class LRUCache:
    def __init__(self, capacity: int, ttl: int = 86400):
        self.capacity = capacity
        self.ttl = ttl
        self._map: dict = {}
        # Sentinel head (MRU side) and tail (LRU side)
        self._head = _Node()
        self._tail = _Node()
        self._head.next = self._tail
        self._tail.prev = self._head
        self._evictions = 0

    # ── DLL helpers ────────────────────────────────────────────────────
    def _remove(self, node: _Node) -> None:
        """Unlink node from DLL.  O(1)"""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _insert_after_head(self, node: _Node) -> None:
        """Insert node right after head sentinel (MRU position).  O(1)"""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def _move_to_front(self, node: _Node) -> None:
        """Promote node to MRU position.  O(1)"""
        self._remove(node)
        self._insert_after_head(node)

    # ── Public interface ───────────────────────────────────────────────
    def get(self, key) -> object | None:
        """
        Return cached value on hit, None on miss or TTL expiry.  O(1)
        Promotes the accessed node to MRU position.
        """
        node = self._map.get(key)
        if node is None:
            return None
        if time.time() - node.timestamp > self.ttl:
            self._remove(node)
            del self._map[key]
            return None
        self._move_to_front(node)           # O(1)
        return node.value

    def put(self, key, value) -> None:
        """
        Insert or update key.  Evicts LRU item if over capacity.  O(1)
        """
        node = self._map.get(key)
        if node is not None:
            node.value = value
            node.timestamp = time.time()
            self._move_to_front(node)       # O(1)
            return
        new_node = _Node(key, value)
        self._map[key] = new_node
        self._insert_after_head(new_node)   # O(1)
        if len(self._map) > self.capacity:
            lru_node = self._tail.prev
            self._remove(lru_node)          # O(1)
            del self._map[lru_node.key]
            self._evictions += 1

    def __len__(self) -> int:
        """Return number of cached items.  O(1)"""
        return len(self._map)

    def get_evictions(self) -> int:
        """Return total evictions since creation.  O(1)"""
        return self._evictions