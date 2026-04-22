"""
run_sim.py — Smoke tests for all five DAA cache modules.
Run:  python run_sim.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cache.lru_cache    import LRUCache
from cache.lfu_cache    import LFUCache
from cache.greedy_cache import GreedyCache
from cache.belady_cache import BeladyCache
from cache.dp_admission import DPAdmissionCache


def chk(cond, msg):
    ok = "\033[92mPASSED\033[0m" if cond else "\033[91mFAILED\033[0m"
    print(f"  {'✓' if cond else '✗'} {msg} ... {ok}")
    assert cond, msg


# ── 1. LRU ───────────────────────────────────────────────────────────────
print("\n── LRU Cache ──────────────────────────────")
lru = LRUCache(capacity=2)
lru.put('a', 1); lru.put('b', 2)
chk(lru.get('a') == 1,        "hit on a")
lru.put('c', 3)                # evicts b (LRU)
chk(lru.get('b') is None,     "b evicted")
chk(lru.get('a') == 1,        "a still present")
chk(lru.get_evictions() == 1, "eviction count = 1")
chk(len(lru) == 2,            "__len__ = 2")

# ── 2. LFU ───────────────────────────────────────────────────────────────
print("\n── LFU Cache ──────────────────────────────")
lfu = LFUCache(capacity=2)
lfu.put('x', 10); lfu.put('y', 20)
lfu.get('x'); lfu.get('x')    # x freq=3
lfu.put('z', 30)               # evicts y (freq=1)
chk(lfu.get('y') is None,     "y evicted")
chk(lfu.get('x') == 10,       "x present")
chk(lfu.get_evictions() == 1, "eviction count = 1")
chk(len(lfu) == 2,            "__len__ = 2")

# ── 3. Greedy ─────────────────────────────────────────────────────────────
print("\n── Greedy Cache ───────────────────────────")
g = GreedyCache(capacity=2)
g.put('p', 1); g.put('q', 2)
g.get('p'); g.get('p')        # p has higher freq → higher score
g.put('r', 3)                  # should evict q (lower score)
chk(g.get('p') is not None,   "p retained (higher score)")
chk(g.get_evictions() == 1,   "eviction count = 1")
chk(len(g) == 2,              "__len__ = 2")

# ── 4. Belady ─────────────────────────────────────────────────────────────
print("\n── Belady Cache ───────────────────────────")
trace = ['a', 'b', 'c', 'a', 'b', 'c', 'a']
bel   = BeladyCache(capacity=2, trace=trace)
lru2  = LRUCache(capacity=2)
bel_h = lru_h = 0
for url in trace:
    if bel.get(url) is not None: bel_h += 1
    else: bel.put(url, url)
    if lru2.get(url) is not None: lru_h += 1
    else: lru2.put(url, url)
chk(bel_h >= lru_h,
    f"Belady hits {bel_h} >= LRU hits {lru_h}")
chk(bel.get_evictions() <= lru2.get_evictions(),
    f"Belady evictions {bel.get_evictions()} <= LRU {lru2.get_evictions()}")
chk(len(bel) <= 2, "__len__ <= capacity")

# ── 5. DP Admission ───────────────────────────────────────────────────────
print("\n── DP Admission Cache ─────────────────────")
dp_trace = ['x'] * 5 + ['y'] * 3 + ['z'] * 1
dp = DPAdmissionCache(capacity=2, trace=dp_trace)
chk('x' in dp._cache,        "x pre-loaded (highest frequency)")
chk(dp.get('x') is not None, "x cache hit immediately")
dp.put('new1', 'v1'); dp.put('new2', 'v2')
chk(len(dp) <= 2,            "capacity respected after online puts")

# ── 6. Full simulation ────────────────────────────────────────────────────
print("\n── Full DAA Simulation ────────────────────")
from simulator.simulation_engine import SimulationEngine
SimulationEngine.print_zipf_table(num_urls=20, alpha=1.2)
engine  = SimulationEngine(
    cache_capacity=10, num_urls=20, total_requests=500,
    alpha=1.2, seed=42,
)
results = engine.run()
chk(
    results["belady_hit_ratio"] >= results["lru_hit_ratio"] - 0.5,
    f"Belady ({results['belady_hit_ratio']}%) >= LRU ({results['lru_hit_ratio']}%) — optimality check"
)

print("\n  ✅  All smoke tests passed.\n")
