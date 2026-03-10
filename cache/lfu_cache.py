"""
LFU Cache — Min-Heap (heapq) + HashMaps with Lazy Deletion

Lazy Deletion Strategy:
  When a key's frequency changes we push a NEW heap entry and leave the stale one.
  On eviction (heappop) we skip entries where heap_freq != self._freq[key]
  or the key no longer exists.  This avoids the O(n) cost of finding and
  removing in-place heap entries.

Time Complexity:
  - get()  : O(log n) — freq bump pushes one entry onto the heap
  - put()  : O(log n) — heappush for new key; heappop loop for eviction
               (amortised O(log n) because each key causes at most one real pop)

Space Complexity: O(n + d) where d = number of lazy/stale heap entries
                 
"""

import heapq
import time


class LFUCache:
    def __init__(self, capacity: int, ttl: int = 3600):
        self.capacity = capacity
        self.ttl = ttl

        # HashMaps for O(1) key-level lookups
        self._vals: dict = {}          
        self._freq: dict = {}          
        self._ts:   dict = {}          

        # Min-heap: (frequency, insertion_order_counter, key)
        # Lower freq → higher eviction priority.
        # Tie-break on insertion_order (smaller = older = evict first).
        self._heap: list = []
        self._counter: int = 0     

        self.evictions = 0

    def _is_stale_entry(self, heap_freq: int, heap_order: int, key) -> bool:
        if key not in self._freq:
            return True                   
        return self._freq[key] != heap_freq

    def _push(self, key) -> None:
        self._counter += 1
        heapq.heappush(self._heap, (self._freq[key], self._counter, key))

    def _evict_lfu(self) -> None:
        while self._heap:
            freq, order, key = heapq.heappop(self._heap)  # O(log n)
            if not self._is_stale_entry(freq, order, key):
                del self._vals[key]
                del self._freq[key]
                del self._ts[key]
                self.evictions += 1
                return

    def get(self, key) -> object | None:
        if key not in self._vals:
            return None  # cache miss

        # TTL check
        if time.time() - self._ts[key] > self.ttl:
            del self._vals[key]
            del self._freq[key]
            del self._ts[key]
            return None 

        self._freq[key] += 1   # O(1)
        self._push(key)        # O(log n) 
        return self._vals[key]

    def put(self, key, value) -> None:
        if self.capacity == 0:
            return

        if key in self._vals:
            self._vals[key] = value
            self._ts[key] = time.time()
            self._freq[key] += 1   # O(1)
            self._push(key)        # O(log n)
            return

        if len(self._vals) >= self.capacity:
            self._evict_lfu()      

        self._vals[key] = value
        self._freq[key] = 1
        self._ts[key] = time.time()
        self._push(key)            # O(log n)

    def delete(self, key) -> None:
        if key in self._vals:
            del self._vals[key]
            del self._freq[key]
            del self._ts[key]

    def get_evictions(self) -> int:
        return self.evictions

    def size(self) -> int:
        return len(self._vals)