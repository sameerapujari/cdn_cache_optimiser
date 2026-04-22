"""
Belady's Optimal Cache — Offline Greedy
==========================================
Greedy paradigm — provably optimal by exchange argument.  Always evicts
the item least needed in the future (next use furthest away or never).
Serves as the theoretical upper bound for all online policies.

Offline algorithm: requires the full request trace before simulation.
At each eviction the greedy choice is to evict the item whose next access
index in the trace is largest — this is locally and globally optimal
(proven by exchange argument: swapping any different choice cannot reduce
the total miss count).

Time Complexity:
    Preprocessing : O(n log n)  — build sorted access-position index
    get()         : O(log k)    — lazy-deletion heap push  (k = capacity)
    put()         : O(log k)    — heap push/pop on eviction
    __len__       : O(1)

Space Complexity: O(n + k)
"""

from __future__ import annotations

import bisect
import heapq
from collections import defaultdict


class BeladyCache:
    """
    Greedy paradigm — provably optimal by exchange argument.  Always evicts
    the item least needed in the future.  Serves as the theoretical upper
    bound for all online policies.
    """

    def __init__(self, capacity: int, trace: list[str]) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if not trace:
            raise ValueError("trace must be non-empty")
        self.capacity = capacity
        self._trace   = trace
        self._n       = len(trace)

        # ── Preprocessing: sorted access positions per URL  O(n log n) ──
        # _positions[url] = sorted list of all indices in trace where url appears
        self._positions: dict[str, list[int]] = defaultdict(list)
        for idx, url in enumerate(trace):                  # O(n)
            self._positions[url].append(idx)
        # Lists are already sorted (appended in order)

        # ── Runtime state ────────────────────────────────────────────────
        self._cache:   set[str]      = set()
        self._evictions: int         = 0
        self._step:    int           = 0   # current request index

        # Max-heap (negated): (-next_use_index, url)
        # Highest next_use = furthest future = evict first.
        # Lazy deletion: stale entries are skipped on pop.
        self._pq:       list         = []
        self._pq_fut:   dict[str, int] = {}   # url -> next_use stored in PQ

    # ── Preprocessing helper ──────────────────────────────────────────
    def _next_use(self, url: str, after: int) -> int:
        """
        Binary-search for the smallest position > after in url's list.
        Returns self._n (infinity) if url is never accessed again.  O(log n)
        """
        positions = self._positions.get(url, [])           # O(1)
        lo = bisect.bisect_right(positions, after)         # O(log n)
        return positions[lo] if lo < len(positions) else self._n

    def _push_pq(self, url: str) -> None:
        """Push url onto the max-heap keyed on its next use after _step.  O(log k)"""
        fut = self._next_use(url, self._step)              # O(log n)
        self._pq_fut[url] = fut
        heapq.heappush(self._pq, (-fut, url))              # O(log k)

    def _evict_one(self) -> None:
        """
        Greedy eviction: remove the item with the furthest next use.
        Lazy-deletes stale heap entries.  O(log k) amortised.
        """
        while self._pq:
            neg_fut, url = heapq.heappop(self._pq)        # O(log k)
            if url not in self._cache:
                continue   # stale: already evicted
            if self._pq_fut.get(url) != -neg_fut:
                continue   # stale: fut index was updated
            self._cache.discard(url)
            del self._pq_fut[url]
            self._evictions += 1
            return

    # ── Public interface ──────────────────────────────────────────────
    def get(self, key: str) -> object | None:
        """
        Cache lookup.  Returns sentinel string on hit, None on miss.
        Refreshes PQ entry so future evictions use an updated look-ahead.
        O(log k)
        """
        present = key in self._cache
        if present:
            self._push_pq(key)                             # O(log k) refresh
        self._step += 1
        return f"content_of_{key}" if present else None

    def put(self, key: str, value: object) -> None:
        """
        Insert key.  Evicts furthest-future item first if cache is full.
        O(log k)
        """
        if key in self._cache:
            self._push_pq(key)                             # O(log k) refresh
            return
        if len(self._cache) >= self.capacity:
            self._evict_one()                              # O(log k)
        self._cache.add(key)
        self._push_pq(key)                                 # O(log k)

    def __len__(self) -> int:
        """Return current cache occupancy.  O(1)"""
        return len(self._cache)

    def get_evictions(self) -> int:
        """Return total evictions since creation.  O(1)"""
        return self._evictions
