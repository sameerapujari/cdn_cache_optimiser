"""
Simulation Engine — DAA Cache Policy Comparison
================================================
Runs five cache policies on the same Zipf-distributed request trace
and prints a structured comparison table.

Policies:
  1. LRU       — Least Recently Used          (online baseline)
  2. LFU       — Least Frequently Used        (online baseline)
  3. Greedy    — Score-based eviction         (Greedy paradigm)
  4. Belady    — Optimal offline              (Greedy + exchange argument)
  5. DP + LRU  — Knapsack pre-load + LRU     (Dynamic Programming)

DAA Complexity Reference:
  LRU    : O(1)
  LFU    : O(log n)
  Greedy : O(log n)
  Belady : O(n log n)  [offline]
  DP+LRU : O(n·W)      [pre-load]  then O(1) online
"""

from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache.lru_cache      import LRUCache
from cache.lfu_cache      import LFUCache
from cache.greedy_cache   import GreedyCache
from cache.belady_cache   import BeladyCache
from cache.dp_admission   import DPAdmissionCache
from simulator.traffic_generator import TrafficGenerator


# ── Latency constants ───────────────────────────────────────────────────────
HIT_LATENCY_MS  = 1.0    # served from local cache
MISS_LATENCY_MS = 50.0   # round-trip to origin server

# ── DAA Complexity reference dict ───────────────────────────────────────────
COMPLEXITY: dict[str, str] = {
    "LRU":    "O(1)",
    "LFU":    "O(log n)",
    "Greedy": "O(log n)",
    "Belady": "O(n log n)",
    "DP+LRU": "O(n·W)",
}


class _Stats:
    """Accumulates per-algorithm simulation statistics."""

    def __init__(self) -> None:
        self.hits:      int   = 0
        self.misses:    int   = 0
        self.total_lat: float = 0.0
        self.wall_ms:   float = 0.0

    @property
    def requests(self) -> int:            # O(1)
        return self.hits + self.misses

    @property
    def hit_ratio(self) -> float:         # O(1)
        return (self.hits / self.requests * 100) if self.requests else 0.0

    @property
    def avg_latency(self) -> float:       # O(1)
        return (self.total_lat / self.requests) if self.requests else 0.0


class SimulationEngine:
    """
    Orchestrates all five cache policies against the same Zipf trace.

    Args:
        cache_capacity   : Max items each cache can hold (also knapsack W).
        num_urls         : Distinct URL count in the catalogue.
        total_requests   : Request trace length.
        alpha            : Zipf skew parameter.
        seed             : RNG seed for reproducibility.
        hit_latency_ms   : Simulated cache-hit latency (ms).
        miss_latency_ms  : Simulated cache-miss latency (ms).
        cache_ttl        : TTL for LRU/LFU/Greedy items (seconds).
    """

    def __init__(
        self,
        cache_capacity:  int   = 10,
        num_urls:        int   = 50,
        total_requests:  int   = 1000,
        alpha:           float = 1.2,
        seed:            int   = 42,
        hit_latency_ms:  float = HIT_LATENCY_MS,
        miss_latency_ms: float = MISS_LATENCY_MS,
        cache_ttl:       int   = 86400,
    ) -> None:
        self.cache_capacity  = cache_capacity
        self.num_urls        = num_urls
        self.total_requests  = total_requests
        self.alpha           = alpha
        self.seed            = seed
        self.hit_latency_ms  = hit_latency_ms
        self.miss_latency_ms = miss_latency_ms
        self.cache_ttl       = cache_ttl

    # ── Replay helper ────────────────────────────────────────────────────
    def _replay(self, cache, requests: list[str]) -> _Stats:
        """
        Replay trace through any cache object exposing get/put.
        O(n · cost_per_operation)
        """
        stats = _Stats()
        t0    = time.perf_counter()
        for key in requests:                               # O(n)
            if cache.get(key) is not None:
                stats.hits      += 1
                stats.total_lat += self.hit_latency_ms
            else:
                stats.misses    += 1
                stats.total_lat += self.miss_latency_ms
                cache.put(key, f"content_of_{key}")
        stats.wall_ms = (time.perf_counter() - t0) * 1_000
        return stats

    # ── Main run ─────────────────────────────────────────────────────────
    def run(self) -> dict:
        """Generate traffic, run all five policies, return results dict."""

        # 1. Generate trace                                O(n log N)
        tgen = TrafficGenerator(
            num_urls=self.num_urls,
            total_requests=self.total_requests,
            alpha=self.alpha,
            seed=self.seed,
        )
        requests = tgen.generate()

        # 2. Initialise caches
        cap, ttl = self.cache_capacity, self.cache_ttl
        lru    = LRUCache(capacity=cap, ttl=ttl)
        lfu    = LFUCache(capacity=cap, ttl=ttl)
        greedy = GreedyCache(capacity=cap, ttl=ttl)
        belady = BeladyCache(capacity=cap, trace=requests)  # offline — gets trace
        dp_adm = DPAdmissionCache(capacity=cap, trace=requests)

        # 3. Replay identical trace through all five      O(n) each
        lru_s    = self._replay(lru,    requests)
        lfu_s    = self._replay(lfu,    requests)
        grd_s    = self._replay(greedy, requests)
        bel_s    = self._replay(belady, requests)
        dp_s     = self._replay(dp_adm, requests)

        # 4. Print report
        self._print_report(lru_s, lfu_s, grd_s, bel_s, dp_s,
                           lru.get_evictions(), lfu.get_evictions(),
                           greedy.get_evictions(), belady.get_evictions(),
                           dp_adm.get_evictions())

        return {
            "lru_hit_ratio":      round(lru_s.hit_ratio,   2),
            "lru_avg_latency_ms": round(lru_s.avg_latency, 2),
            "lru_evictions":      lru.get_evictions(),
            "lru_wall_ms":        round(lru_s.wall_ms,     2),

            "lfu_hit_ratio":      round(lfu_s.hit_ratio,   2),
            "lfu_avg_latency_ms": round(lfu_s.avg_latency, 2),
            "lfu_evictions":      lfu.get_evictions(),
            "lfu_wall_ms":        round(lfu_s.wall_ms,     2),

            "greedy_hit_ratio":      round(grd_s.hit_ratio,   2),
            "greedy_avg_latency_ms": round(grd_s.avg_latency, 2),
            "greedy_evictions":      greedy.get_evictions(),
            "greedy_wall_ms":        round(grd_s.wall_ms,     2),

            "belady_hit_ratio":      round(bel_s.hit_ratio,   2),
            "belady_avg_latency_ms": round(bel_s.avg_latency, 2),
            "belady_evictions":      belady.get_evictions(),
            "belady_wall_ms":        round(bel_s.wall_ms,     2),

            "dp_hit_ratio":      round(dp_s.hit_ratio,   2),
            "dp_avg_latency_ms": round(dp_s.avg_latency, 2),
            "dp_evictions":      dp_adm.get_evictions(),
            "dp_wall_ms":        round(dp_s.wall_ms,     2),

            "total_requests":   self.total_requests,
            "traffic_sequence": requests,
        }

    # ── Report printer ───────────────────────────────────────────────────
    def _print_report(
        self,
        lru: _Stats, lfu: _Stats, grd: _Stats, bel: _Stats, dp: _Stats,
        lru_ev: int, lfu_ev: int, grd_ev: int, bel_ev: int, dp_ev: int,
    ) -> None:
        W    = 106
        hdiv = "═" * W
        div  = "─" * W

        def row(label, v1, v2, v3, v4, v5, sfx=""):
            return (
                f"  {label:<18}"
                f"{str(v1)+sfx:>16}"
                f"{str(v2)+sfx:>16}"
                f"{str(v3)+sfx:>16}"
                f"{str(v4)+sfx:>16}"
                f"{str(v5)+sfx:>16}"
            )

        lines = [
            "",
            "╔" + hdiv + "╗",
            "║" + "  CDN Cache Optimizer — DAA Comparison".center(W) + "║",
            "╠" + hdiv + "╣",
            "║" + f"  {'Metric':<18}{'LRU':>16}{'LFU':>16}{'Greedy':>16}{'Belady':>16}{'DP+LRU':>16}" + "  ║",
            "║" + f"  {'DAA Strategy':<18}{'Heuristic':>16}{'Heuristic':>16}{'Greedy':>16}{'Greedy(opt)':>16}{'Dyn. Prog.':>16}" + "  ║",
            "║" + f"  {'Complexity':<18}{'O(1)':>16}{'O(log n)':>16}{'O(log n)':>16}{'O(n log n)':>16}{'O(n·W)':>16}" + "  ║",
            "║" + "  " + div + "  ║",
            "║" + row("Hit Ratio",
                       f"{lru.hit_ratio:.2f}", f"{lfu.hit_ratio:.2f}",
                       f"{grd.hit_ratio:.2f}", f"{bel.hit_ratio:.2f}",
                       f"{dp.hit_ratio:.2f}", " %") + "  ║",
            "║" + row("Avg Latency",
                       f"{lru.avg_latency:.2f}", f"{lfu.avg_latency:.2f}",
                       f"{grd.avg_latency:.2f}", f"{bel.avg_latency:.2f}",
                       f"{dp.avg_latency:.2f}", " ms") + "  ║",
            "║" + row("Evictions",
                       lru_ev, lfu_ev, grd_ev, bel_ev, dp_ev) + "  ║",
            "║" + row("Wall Time",
                       f"{lru.wall_ms:.1f}", f"{lfu.wall_ms:.1f}",
                       f"{grd.wall_ms:.1f}", f"{bel.wall_ms:.1f}",
                       f"{dp.wall_ms:.1f}", " ms") + "  ║",
            "╠" + hdiv + "╣",
            "║" + f"  Best Hit Ratio : {max(lru.hit_ratio, lfu.hit_ratio, grd.hit_ratio, bel.hit_ratio, dp.hit_ratio):.2f} %".ljust(W) + "║",
            "╚" + hdiv + "╝",
            "",
        ]
        print("\n".join(lines))

    # ── Zipf helper ──────────────────────────────────────────────────────
    @staticmethod
    def print_zipf_table(num_urls: int = 20, alpha: float = 1.2) -> None:
        """Print the theoretical Zipf popularity table to stdout."""
        tgen  = TrafficGenerator(num_urls=num_urls, alpha=alpha, total_requests=1)
        table = tgen.url_popularity_table()
        print(f"\n  Zipf (alpha={alpha}, {num_urls} URLs)")
        print(f"  {'Rank':<6} {'URL':<10} {'Prob':>10}")
        print("  " + "─" * 30)
        for rank, (url, prob) in enumerate(table, 1):
            bar = "█" * int(prob / 2)
            print(f"  {rank:<6} {url:<10} {prob:>8.2f} %  {bar}")
        print()


# ── Entry-point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SimulationEngine.print_zipf_table(num_urls=20, alpha=1.2)
    SimulationEngine(
        cache_capacity=10, num_urls=20, total_requests=1000,
        alpha=1.2, seed=42,
    ).run()
