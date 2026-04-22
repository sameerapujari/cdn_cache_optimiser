"""
DP Admission Cache — 0/1 Knapsack Pre-Load + LRU Online
=========================================================
Dynamic Programming paradigm — optimal substructure and overlapping
subproblems.  Selects the highest-value item set within capacity
constraint.  O(n·W) where n=unique URLs, W=cache capacity.

Pre-load phase (offline):
  Each unique URL is an "item":
    value  = access frequency in the trace  (higher = more beneficial)
    weight = 1 (uniform size)
  Cache capacity W = knapsack capacity.
  Standard 0/1 Knapsack DP solves which URLs to pre-load, maximising
  the total anticipated hit count before the simulation begins.

Online phase:
  Uses LRU (via OrderedDict) for subsequent evictions.

Time Complexity:
    Pre-load DP : O(n · W) where n = distinct URLs, W = cache capacity
    get()       : O(1)  — OrderedDict lookup + move_to_end
    put()       : O(1)  — OrderedDict insert + optional LRU eviction
    __len__     : O(1)
"""

from __future__ import annotations

from collections import Counter, OrderedDict


class DPAdmissionCache:
    """
    Dynamic Programming paradigm — optimal substructure and overlapping
    subproblems.  Selects the highest-value item set within capacity
    constraint.  O(n·W) where n=unique URLs, W=cache capacity.
    """

    def __init__(self, capacity: int, trace: list[str]) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._evictions = 0

        # ── Step 1: Frequency analysis  O(n) ─────────────────────────
        freq: Counter[str] = Counter(trace)                # O(n)
        items = list(freq.items())                         # [(url, count), ...]
        n, W = len(items), capacity

        # ── Step 2: 0/1 Knapsack DP  O(n · W) ───────────────────────
        # dp[i][w] = max total freq achievable using first i items in w slots.
        # All weights = 1 (uniform).
        dp = [[0] * (W + 1) for _ in range(n + 1)]        # O(n·W) space

        for i in range(1, n + 1):                          # O(n)
            _, freq_i = items[i - 1]
            for w in range(W + 1):                         # O(W)
                dp[i][w] = dp[i - 1][w]                   # skip item i
                if w >= 1:
                    val_take = dp[i - 1][w - 1] + freq_i
                    if val_take > dp[i][w]:
                        dp[i][w] = val_take                # take item i

        # ── Step 3: Backtrack to recover admitted set  O(n) ──────────
        admitted: list[str] = []
        w = W
        for i in range(n, 0, -1):                          # O(n)
            if w >= 1 and dp[i][w] != dp[i - 1][w]:
                admitted.append(items[i - 1][0])
                w -= 1

        # ── Step 4: Pre-load into LRU (most frequent first) ──────────
        admitted.sort(key=lambda url: freq[url], reverse=True)  # O(n log n)

        self._cache: OrderedDict[str, str] = OrderedDict()
        for url in admitted:                               # O(min(n,W))
            self._cache[url] = f"content_of_{url}"

        self._freq = freq   # kept for reference / debugging

    # ── LRU online helpers ─────────────────────────────────────────────
    def _lru_evict(self) -> None:
        """Evict the least-recently-used item (leftmost in OrderedDict).  O(1)"""
        self._cache.popitem(last=False)                    # O(1)
        self._evictions += 1

    # ── Public interface ───────────────────────────────────────────────
    def get(self, key: str) -> object | None:
        """
        Return cached value on hit, None on miss.  O(1)
        Promotes item to MRU position on hit.
        """
        if key not in self._cache:
            return None                                    # miss
        self._cache.move_to_end(key)                       # O(1) LRU promote
        return self._cache[key]

    def put(self, key: str, value: object) -> None:
        """
        Insert or update key using LRU replacement policy.  O(1)
        """
        if key in self._cache:
            self._cache.move_to_end(key)                   # O(1)
            self._cache[key] = value
            return
        if len(self._cache) >= self.capacity:
            self._lru_evict()                              # O(1)
        self._cache[key] = value
        self._cache.move_to_end(key)                       # O(1)

    def __len__(self) -> int:
        """Return current cache occupancy.  O(1)"""
        return len(self._cache)

    def get_evictions(self) -> int:
        """Return total online evictions (pre-load phase excluded).  O(1)"""
        return self._evictions
