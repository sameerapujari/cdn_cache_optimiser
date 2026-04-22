"""
Greedy Cache — Score-Based Eviction
=====================================
Greedy paradigm — makes locally optimal eviction choice by maximising
retained value score.  Approximates Belady's optimal without future
knowledge.  O(log n) per eviction via min-heap with lazy deletion.

Score formula (computed at eviction time for each cached item):

    score = (frequency * 0.6  +  recency_bonus * 0.4) / size

Where:
    frequency     = number of times item has been accessed (hits)
    recency_bonus = 1 / (current_time - last_access_time + 1)
                    (higher = accessed more recently)
    size          = 1 for all items (uniform, no size difference)

The item with the LOWEST score is evicted — it is the least frequented
and/or least recently used item, maximising the value retained in cache.

Time Complexity:
    get()  : O(log n) — lazy-deletion heap push on hit
    put()  : O(log n) — heap push; heappop loop on eviction
    __len__: O(1)

Space Complexity: O(n + d) where d = stale heap entries ≤ 2n
"""

from __future__ import annotations

import heapq
import time


class GreedyCache:
    """
    Greedy paradigm — makes locally optimal eviction choice by maximising
    retained value score.  Approximates Belady's optimal without future
    knowledge.  O(log n) per eviction via min-heap.
    """

    # Weights for the score function
    FREQ_WEIGHT    = 0.6
    RECENCY_WEIGHT = 0.4

    def __init__(self, capacity: int, ttl: int = 86400) -> None:
        self.capacity = capacity
        self.ttl = ttl

        # Core state
        self._vals:        dict[str, object] = {}
        self._freq:        dict[str, int]    = {}
        self._last_access: dict[str, float]  = {}   # wall-clock timestamp

        # Min-heap: (score_snapshot, counter, key)
        # Lower score → evict first.
        # Lazy deletion: stale entries are skipped on pop.
        self._heap:    list  = []
        self._counter: int   = 0      # tie-break; also detects stale entries
        self._version: dict[str, int] = {}   # key -> counter at last push

        self._evictions = 0

    # ── Score ──────────────────────────────────────────────────────────
    def _score(self, key: str) -> float:
        """
        Compute the current greedy score for a cached item.  O(1)
        Higher score = more valuable = should be RETAINED.
        """
        freq         = self._freq[key]
        recency_bonus = 1.0 / (time.time() - self._last_access[key] + 1.0)
        size         = 1.0   # uniform
        return (self.FREQ_WEIGHT * freq + self.RECENCY_WEIGHT * recency_bonus) / size

    # ── Heap helpers ───────────────────────────────────────────────────
    def _push(self, key: str) -> None:
        """Push a fresh (score, counter, key) entry onto the min-heap.  O(log n)"""
        self._counter += 1
        score = self._score(key)                           # O(1)
        self._version[key] = self._counter
        heapq.heappush(self._heap, (score, self._counter, key))  # O(log n)

    def _is_stale(self, counter: int, key: str) -> bool:
        """Return True if this heap entry is outdated.  O(1)"""
        return key not in self._version or self._version[key] != counter

    def _evict_lowest_score(self) -> None:
        """
        Pop the heap until a live entry is found; evict that item.
        O(log n) amortised — each key causes at most one real pop.
        """
        while self._heap:
            score, counter, key = heapq.heappop(self._heap)  # O(log n)
            if self._is_stale(counter, key):
                continue   # skip outdated entry
            # Evict this item
            del self._vals[key]
            del self._freq[key]
            del self._last_access[key]
            del self._version[key]
            self._evictions += 1
            return

    # ── Public interface ───────────────────────────────────────────────
    def get(self, key: str) -> object | None:
        """
        Return cached value on hit, None on miss or TTL expiry.  O(log n)
        Updates frequency, recency, and pushes fresh heap entry.
        """
        if key not in self._vals:
            return None                                    # miss
        # TTL check
        if time.time() - self._last_access[key] > self.ttl:
            del self._vals[key]; del self._freq[key]
            del self._last_access[key]; del self._version[key]
            return None
        # Hit — update state
        self._freq[key]        += 1                        # O(1)
        self._last_access[key]  = time.time()              # O(1)
        self._push(key)                                    # O(log n) — refresh score
        return self._vals[key]

    def put(self, key: str, value: object) -> None:
        """
        Insert or update key.  Evicts lowest-score item if over capacity.
        O(log n)
        """
        if self.capacity == 0:
            return
        if key in self._vals:
            self._vals[key]       = value
            self._freq[key]      += 1
            self._last_access[key] = time.time()
            self._push(key)                                # O(log n) refresh
            return
        if len(self._vals) >= self.capacity:
            self._evict_lowest_score()                     # O(log n)
        # Insert new item
        self._vals[key]        = value
        self._freq[key]        = 1
        self._last_access[key] = time.time()
        self._push(key)                                    # O(log n)

    def __len__(self) -> int:
        """Return number of cached items.  O(1)"""
        return len(self._vals)

    def get_evictions(self) -> int:
        """Return total evictions since creation.  O(1)"""
        return self._evictions
