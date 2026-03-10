"""
LRU Cache — Doubly Linked List + HashMap (dict)
Data Structures:
  - _Node       : doubly linked list node (prev / next pointers)
  - self._map   : dict[key -> Node]  ← the "HashMap" for O(1) key lookup
Time Complexity:
  - get()  : O(1)  — hash lookup + O(1) 
  - put()  : O(1)  — hash insert + O(1) 

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
    def __init__(self, capacity: int, ttl: int = 3600):
        self.capacity = capacity
        self.ttl = ttl

        # HashMap: key → Node (O(1) lookup)
        self._map: dict = {}

        self._head = _Node()   
        self._tail = _Node()  
        self._head.next = self._tail
        self._tail.prev = self._head

        self.evictions = 0

    def _remove(self, node: _Node) -> None:
        """Unlink a node from the DLL.  O(1)"""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node   
        next_node.prev = prev_node   

    def _insert_after_head(self, node: _Node) -> None:
        """Insert a node right after the head sentinel (MRU position).  O(1)"""
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node  
        self._head.next = node

    def _move_to_front(self, node: _Node) -> None:
        self._remove(node)
        self._insert_after_head(node)

    def get(self, key) -> object | None:
        node = self._map.get(key)
        if node is None:
            return None  # cache miss

        # TTL check — evict and report miss if expired
        if time.time() - node.timestamp > self.ttl:
            self._remove(node)
            del self._map[key]
            return None  # expired

        # Cache hit — promote to MRU position
        self._move_to_front(node)
        return node.value

    def put(self, key, value) -> None:
        node = self._map.get(key)

        if node is not None:
            # Key already exists — update value and promote to MRU
            node.value = value
            node.timestamp = time.time()
            self._move_to_front(node)
            return

        # New key — create node and insert at MRU position
        new_node = _Node(key, value)
        self._map[key] = new_node
        self._insert_after_head(new_node)

        # Evict LRU (tail.prev) if over capacity
        if len(self._map) > self.capacity:
            lru_node = self._tail.prev        
            self._remove(lru_node)
            del self._map[lru_node.key]        
            self.evictions += 1

    def get_evictions(self) -> int:
        """Return total number of evictions since creation."""
        return self.evictions

    def size(self) -> int:
        """Return the current number of cached items.  O(1)"""
        return len(self._map)