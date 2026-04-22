"""
LFU Cache — Min-Heap (heapq) + HashMaps with Lazy Deletion
===========================================================
Lazy Deletion Strategy:
  When a key's frequency changes we push a NEW heap entry and leave the
  stale one.  On eviction (heappop) we skip entries where the heap freq
  does not match self._freq[key].  This avoids O(n) in-place heap repair.

Time Complexity:
  get()  : O(log n) — freq bump pushes one entry onto the heap
  put()  : O(log n) — heappush for new key; heappop loop for eviction

Space Complexity: O(n + d) where d = stale heap entries (bounded by 2n)
"""

import heapq
import time


class LFUCache:
    def __init__(self, capacity: int, ttl: int = 86400):
        self.capacity = capacity
        self.ttl = ttl
        self._vals: dict = {}
        self._freq: dict = {}
        self._ts:   dict = {}
        # Min-heap: (frequency, insertion_counter, key)
        self._heap: list = []
        self._counter: int = 0
        self._evictions = 0

    # ── Heap helpers ───────────────────────────────────────────────────
    def _push(self, key) -> None:
        """Push a fresh heap entry for key at its current frequency.  O(log n)"""
        self._counter += 1
        heapq.heappush(self._heap, (self._freq[key], self._counter, key))

    def _is_stale(self, freq: int, key) -> bool:
        """True if this heap entry is outdated.  O(1)"""
        return key not in self._freq or self._freq[key] != freq

    def _evict_lfu(self) -> None:
        """Pop heap until a live entry is found; delete it.  O(log n) amortised"""
        while self._heap:
            freq, _, key = heapq.heappop(self._heap)   # O(log n)
            if not self._is_stale(freq, key):
                del self._vals[key]
                del self._freq[key]
                del self._ts[key]
                self._evictions += 1
                return

    # ── Public interface ───────────────────────────────────────────────
    def get(self, key) -> object | None:
        """
        Return cached value on hit, None on miss or TTL expiry.  O(log n)
        Bumps frequency and pushes updated heap entry on hit.
        """
        if key not in self._vals:
            return None
        if time.time() - self._ts[key] > self.ttl:
            del self._vals[key]; del self._freq[key]; del self._ts[key]
            return None
        self._freq[key] += 1   # O(1)
        self._push(key)        # O(log n)
        return self._vals[key]

    def put(self, key, value) -> None:
        """
        Insert or update key.  Evicts LFU item if over capacity.  O(log n)
        """
        if self.capacity == 0:
            return
        if key in self._vals:
            self._vals[key] = value
            self._ts[key] = time.time()
            self._freq[key] += 1
            self._push(key)        # O(log n)
            return
        if len(self._vals) >= self.capacity:
            self._evict_lfu()      # O(log n)
        self._vals[key] = value
        self._freq[key] = 1
        self._ts[key] = time.time()
        self._push(key)            # O(log n)

    def __len__(self) -> int:
        """Return number of cached items.  O(1)"""
        return len(self._vals)

    def get_evictions(self) -> int:
        """Return total evictions.  O(1)"""
        return self._evictions