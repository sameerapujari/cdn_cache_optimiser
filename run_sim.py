import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cache.lru_cache import LRUCache
from cache.lfu_cache import LFUCache

# --- LRU smoke test ---
c = LRUCache(capacity=2, ttl=3600)
c.put('a', 1); c.put('b', 2)
assert c.get('a') == 1,        'LRU: hit on a'
c.put('c', 3)                  # evicts 'b' (LRU)
assert c.get('b') is None,     'LRU: b evicted'
assert c.get('a') == 1,        'LRU: a still present'
assert c.get('c') == 3,        'LRU: c present'
assert c.get_evictions() == 1, 'LRU: eviction count'
print("LRU: ALL PASSED")

# --- LFU smoke test ---
f = LFUCache(capacity=2, ttl=3600)
f.put('x', 10); f.put('y', 20)
f.get('x'); f.get('x')        # x freq=3
f.put('z', 30)                # evicts y (freq=1)
assert f.get('y') is None,    'LFU: y evicted'
assert f.get('x') == 10,      'LFU: x present'
assert f.get_evictions() == 1, 'LFU: eviction count'
print("LFU: ALL PASSED")

# --- Simulation ---
from simulator.simulation_engine import SimulationEngine
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
results = engine.run()
print(f"\nReturned dict: {results}")
