from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache.lru_cache import LRUCache
from cache.lfu_cache import LFUCache
from simulator.traffic_generator import TrafficGenerator


# ---------------------------------------------------------------------------
# Simulated latency constants (milliseconds)
# ---------------------------------------------------------------------------
DEFAULT_HIT_LATENCY_MS  = 1.0    # served from local cache
DEFAULT_MISS_LATENCY_MS = 50.0   # round-trip to origin server


class _PerAlgoStats:
    """Accumulates per-algorithm simulation statistics."""

    def __init__(self) -> None:
        self.hits:       int   = 0
        self.misses:     int   = 0
        self.total_lat:  float = 0.0   # cumulative latency (ms)

    @property
    def requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_ratio(self) -> float:
        return (self.hits / self.requests * 100) if self.requests else 0.0

    @property
    def avg_latency(self) -> float:
        return (self.total_lat / self.requests) if self.requests else 0.0


class SimulationEngine:
    """
    Runs LRU and LFU caches against identical Zipf-distributed traffic.

    Args:
        cache_capacity      : Max items each cache can hold.
        num_urls            : Number of distinct URLs in the catalogue.
        total_requests      : Number of requests to simulate.
        alpha               : Zipf skew (higher = more skewed, more cache-friendly).
        seed                : RNG seed for reproducible traffic sequences.
        hit_latency_ms      : Simulated latency (ms) for a cache hit.
        miss_latency_ms     : Simulated latency (ms) for a cache miss.
        cache_ttl           : TTL (seconds) for cached items (default: no expiry).
    """

    def __init__(
        self,
        cache_capacity:   int   = 10,
        num_urls:         int   = 20,
        total_requests:   int   = 1000,
        alpha:            float = 1.2,
        seed:             int   = 42,
        hit_latency_ms:   float = DEFAULT_HIT_LATENCY_MS,
        miss_latency_ms:  float = DEFAULT_MISS_LATENCY_MS,
        cache_ttl:        int   = 86400,   # 24 h — effectively no expiry in sim
    ):
        self.cache_capacity  = cache_capacity
        self.num_urls        = num_urls
        self.total_requests  = total_requests
        self.alpha           = alpha
        self.seed            = seed
        self.hit_latency_ms  = hit_latency_ms
        self.miss_latency_ms = miss_latency_ms
        self.cache_ttl       = cache_ttl

    # ------------------------------------------------------------------ #

    def _replay(self, cache: LRUCache | LFUCache, requests: list[str]) -> _PerAlgoStats:
        stats = _PerAlgoStats()
        for key in requests:
            hit = cache.get(key) is not None  # O(1) LRU / O(log n) LFU

            if hit:
                stats.hits     += 1
                stats.total_lat += self.hit_latency_ms
            else:
                stats.misses    += 1
                stats.total_lat += self.miss_latency_ms
                # Simulate fetching content from origin and caching it
                cache.put(key, f"content_of_{key}")   # O(1) LRU / O(log n) LFU

        return stats

    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        # 1. Generate traffic 
        tgen = TrafficGenerator(
            num_urls=self.num_urls,
            total_requests=self.total_requests,
            alpha=self.alpha,
            seed=self.seed,
        )
        requests = tgen.generate()   # O(n log num_urls)

        # 2. Initialise fresh caches 
        lru = LRUCache(capacity=self.cache_capacity, ttl=self.cache_ttl)
        lfu = LFUCache(capacity=self.cache_capacity, ttl=self.cache_ttl)

        # 3. Replay identical traffic through both
        lru_stats = self._replay(lru, requests)
        lfu_stats = self._replay(lfu, requests)

        # 4. Gather eviction counts  
        lru_evictions = lru.get_evictions()
        lfu_evictions = lfu.get_evictions()

        # 5. Print report
        self._print_report(lru_stats, lfu_stats, lru_evictions, lfu_evictions)

        return {
            "lru_hit_ratio":        round(lru_stats.hit_ratio,   2),
            "lru_avg_latency_ms":   round(lru_stats.avg_latency, 2),
            "lru_evictions":        lru_evictions,
            "lfu_hit_ratio":        round(lfu_stats.hit_ratio,   2),
            "lfu_avg_latency_ms":   round(lfu_stats.avg_latency, 2),
            "lfu_evictions":        lfu_evictions,
            "total_requests":       self.total_requests,
            "traffic_sequence":     requests,
        }

    # ------------------------------------------------------------------ #

    def _print_report(
        self,
        lru: _PerAlgoStats,
        lfu: _PerAlgoStats,
        lru_ev: int,
        lfu_ev: int,
    ) -> None:
        """Render a human-readable comparison table to stdout."""

        W = 62
        div  = "─" * W
        hdiv = "═" * W

        def row(label, lru_val, lfu_val, suffix=""):
            return (
                f"  {label:<22}"
                f"{str(lru_val) + suffix:>16}"
                f"{str(lfu_val) + suffix:>16}"
            )

        winner_hit = (
            "  ✓ LFU" if lfu.hit_ratio > lru.hit_ratio else
            "  ✓ LRU" if lru.hit_ratio > lfu.hit_ratio else
            "  = TIE"
        )
        winner_lat = (
            "  ✓ LFU" if lfu.avg_latency < lru.avg_latency else
            "  ✓ LRU" if lru.avg_latency < lfu.avg_latency else
            "  = TIE"
        )

        lines = [
            "",
            "╔" + hdiv + "╗",
            "║" + "  CDN Cache Optimizer — Simulation Results".center(W) + "║",
            "╠" + hdiv + "╣",
            "║" + f"  {'Metric':<22}{'LRU':>16}{'LFU':>16}" + "  ║",
            "║" + "  " + div + "  ║",
            "║" + row("Total Requests",
                       lru.requests, lfu.requests) + "  ║",
            "║" + row("Cache Hits",
                       lru.hits, lfu.hits) + "  ║",
            "║" + row("Cache Misses",
                       lru.misses, lfu.misses) + "  ║",
            "║" + row("Hit Ratio",
                       f"{lru.hit_ratio:.2f}", f"{lfu.hit_ratio:.2f}", " %") + "  ║",
            "║" + row("Avg Latency",
                       f"{lru.avg_latency:.2f}", f"{lfu.avg_latency:.2f}", " ms") + "  ║",
            "║" + row("Evictions",
                       lru_ev, lfu_ev) + "  ║",
            "╠" + hdiv + "╣",
            "║" + f"  Best Hit Ratio : {winner_hit}".ljust(W) + "║",
            "║" + f"  Best Latency   : {winner_lat}".ljust(W) + "║",
            "╚" + hdiv + "╝",
            "",
        ]
        print("\n".join(lines))

    # ------------------------------------------------------------------ #

    @staticmethod
    def print_zipf_table(num_urls: int = 20, alpha: float = 1.2) -> None:
        """Print the theoretical Zipf popularity table for the catalogue."""
        tgen = TrafficGenerator(num_urls=num_urls, alpha=alpha, total_requests=1)
        table = tgen.url_popularity_table()
        print(f"\n  Zipf Distribution  (alpha={alpha}, {num_urls} URLs)")
        print(f"  {'Rank':<6} {'URL':<10} {'Probability':>12}")
        print("  " + "─" * 32)
        for rank, (url, prob) in enumerate(table, start=1):
            bar = "█" * int(prob / 2)
            print(f"  {rank:<6} {url:<10} {prob:>10.2f} %  {bar}")
        print()


# ---------------------------------------------------------------------------
# Entry point — run the simulator directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  Starting CDN Cache Optimizer Simulation …")
    print("  Parameters: capacity=10, urls=20, requests=1000, alpha=1.2\n")

    # Show Zipf popularity table so results are interpretable
    SimulationEngine.print_zipf_table(num_urls=20, alpha=1.2)

    engine = SimulationEngine(
        cache_capacity=10,
        num_urls=20,
        total_requests=1000,
        alpha=1.2,
        seed=42,
        hit_latency_ms=1.0,
        miss_latency_ms=50.0,
    )
    engine.run()
