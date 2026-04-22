"""
CDN Cache Optimizer — Streamlit Dashboard
Five DAA-strategy cache policies on a live Zipf traffic trace.
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator.simulation_engine import SimulationEngine, COMPLEXITY
from simulator.traffic_generator import TrafficGenerator

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CDN Cache Optimizer — DAA",
    layout="wide",
)

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"]  { font-size: 1.55rem; font-weight: 700; }
  [data-testid="stMetricLabel"]  { font-size: 0.85rem; opacity: 0.75; }
  .algo-card {
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 4px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.03);
  }
  .badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 10px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.4px;
    vertical-align: middle;
    margin-left: 6px;
  }
  .b-lru    { background:#1e3050; color:#93c5fd; }
  .b-lfu    { background:#1a3a2a; color:#86efac; }
  .b-greedy { background:#3b2a10; color:#fcd34d; }
  .b-belady { background:#2e1a3a; color:#d8b4fe; }
  .b-dp     { background:#1a2e2e; color:#67e8f9; }
</style>
""", unsafe_allow_html=True)

# ── Title ───────────────────────────────────────────────────────────────────
st.title("CDN Cache Optimizer")
st.markdown(
    "Compare **5 cache eviction policies** — each grounded in a different "
    "Design & Analysis of Algorithms paradigm — on the **same Zipf traffic trace**."
)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Simulation Parameters")
    cache_capacity = st.slider(
        "Cache Capacity (W)", 5, 100, 10, 5,
        help="Max items per cache. Also the knapsack capacity W for DP Admission."
    )
    total_requests = st.slider(
        "Total Requests (n)", 100, 5000, 1000, 100,
        help="Length of the simulated request trace."
    )
    alpha = st.slider(
        "Zipf α (skew)", 0.5, 2.5, 1.2, 0.1,
        help="Higher α → heavier skew → top URLs dominate → easier to cache."
    )
    num_urls = st.number_input(
        "Unique URLs", 10, 500, 50, 10,
        help="Catalogue size. DP Knapsack runs over unique URLs × capacity."
    )
    seed = st.number_input("RNG Seed", 0, 9999, 42, 1)
    st.divider()

    run_btn = st.button("Run Simulation", use_container_width=True, type="primary")

# ── Info cards ──────────────────────────────────────────────────────────────
# c1, c2, c3, c4, c5 = st.columns(5)
# with c1:
#     st.markdown('<div class="algo-card"><b>LRU</b> <span class="badge b-lru">Heuristic</span><br><small>Doubly-Linked List<br>O(1) / op</small></div>', unsafe_allow_html=True)
# with c2:
#     st.markdown('<div class="algo-card"><b>LFU</b> <span class="badge b-lfu">Heuristic</span><br><small>Min-Heap + dict<br>O(log n) / op</small></div>', unsafe_allow_html=True)
# with c3:
#     st.markdown('<div class="algo-card"><b>Greedy</b> <span class="badge b-greedy">Greedy</span><br><small>Score = freq+recency<br>O(log n) / op</small></div>', unsafe_allow_html=True)
# with c4:
#     st.markdown('<div class="algo-card"><b>Belady</b> <span class="badge b-belady">Greedy Optimal</span><br><small>Furthest-future evict<br>O(n log n) offline</small></div>', unsafe_allow_html=True)
# with c5:
#     st.markdown('<div class="algo-card"><b>DP+LRU</b> <span class="badge b-dp">Dyn. Prog.</span><br><small>0/1 Knapsack pre-load<br>O(n·W) setup</small></div>', unsafe_allow_html=True)

# st.divider()

# ── Simulation run ──────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running all 5 DAA cache policies on the same Zipf trace…"):
        engine  = SimulationEngine(
            cache_capacity = cache_capacity,
            num_urls       = int(num_urls),
            total_requests = total_requests,
            alpha          = alpha,
            seed           = int(seed),
        )
        R = engine.run()   # results dict

    # ── Metric cards ─────────────────────────────────────────────────────
    st.subheader("Performance Metrics")

    base_hr  = R["lru_hit_ratio"]
    base_lat = R["lru_avg_latency_ms"]

    def delta_hr(key):
        d = round(R[key] - base_hr, 2)
        return f"{d:+.2f}% vs LRU"

    def delta_lat(key):
        d = round(R[key] - base_lat, 2)
        return f"{d:+.2f} ms vs LRU"

    cols = st.columns(5)
    for col, label, hr_key, lat_key, ev_key, wt_key in zip(
        cols,
        ["LRU", "LFU", "Greedy", "Belady", "DP+LRU"],
        ["lru_hit_ratio",    "lfu_hit_ratio",    "greedy_hit_ratio",    "belady_hit_ratio",    "dp_hit_ratio"],
        ["lru_avg_latency_ms","lfu_avg_latency_ms","greedy_avg_latency_ms","belady_avg_latency_ms","dp_avg_latency_ms"],
        ["lru_evictions",    "lfu_evictions",    "greedy_evictions",    "belady_evictions",    "dp_evictions"],
        ["lru_wall_ms",      "lfu_wall_ms",      "greedy_wall_ms",      "belady_wall_ms",      "dp_wall_ms"],
    ):
        with col:
            st.markdown(f"**{label}**")
            is_base = (hr_key == "lru_hit_ratio")
            if is_base:
                st.metric("Hit Ratio",   f"{R[hr_key]}%")
                st.metric("Avg Latency", f"{R[lat_key]} ms")
            else:
                st.metric("Hit Ratio",   f"{R[hr_key]}%",   delta=delta_hr(hr_key))
                st.metric("Avg Latency", f"{R[lat_key]} ms", delta=delta_lat(lat_key), delta_color="inverse")
            st.metric("Evictions",   R[ev_key])
            st.caption(f"Wall: {R[wt_key]} ms")

    st.divider()

    # ── Comparison charts ─────────────────────────────────────────────────
    st.subheader("Comparison Charts")

    policies = ["LRU", "LFU", "Greedy", "Belady", "DP+LRU"]
    df = pd.DataFrame({
        "Policy":           policies,
        "Hit Ratio (%)":    [R["lru_hit_ratio"],    R["lfu_hit_ratio"],    R["greedy_hit_ratio"],    R["belady_hit_ratio"],    R["dp_hit_ratio"]],
        "Avg Latency (ms)": [R["lru_avg_latency_ms"],R["lfu_avg_latency_ms"],R["greedy_avg_latency_ms"],R["belady_avg_latency_ms"],R["dp_avg_latency_ms"]],
        "Evictions":        [R["lru_evictions"],    R["lfu_evictions"],    R["greedy_evictions"],    R["belady_evictions"],    R["dp_evictions"]],
    })

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown("**Hit Ratio (%)** — higher is better")
        st.bar_chart(df, x="Policy", y="Hit Ratio (%)", color="Policy")
    with cc2:
        st.markdown("**Avg Latency (ms)** — lower is better")
        st.bar_chart(df, x="Policy", y="Avg Latency (ms)", color="Policy")
    with cc3:
        st.markdown("**Total Evictions** — lower = more selective")
        st.bar_chart(df, x="Policy", y="Evictions", color="Policy")

    st.divider()

    # ── Complexity comparison table (core DAA deliverable) ────────────────
    st.subheader("Complexity Comparison Table")
    st.markdown(
        "Theoretical complexity vs empirical performance — the **core DAA deliverable**."
    )

    cmp_df = pd.DataFrame([
        {
            "Policy":         "LRU",
            "DAA Paradigm":   "Heuristic (recency)",
            "Complexity":     COMPLEXITY["LRU"],
            "Hit Ratio (%)":  R["lru_hit_ratio"],
            "Avg Latency (ms)": R["lru_avg_latency_ms"],
            "Evictions":      R["lru_evictions"],
            "Optimality":     "Approximate",
        },
        {
            "Policy":         "LFU",
            "DAA Paradigm":   "Heuristic (frequency)",
            "Complexity":     COMPLEXITY["LFU"],
            "Hit Ratio (%)":  R["lfu_hit_ratio"],
            "Avg Latency (ms)": R["lfu_avg_latency_ms"],
            "Evictions":      R["lfu_evictions"],
            "Optimality":     "Approximate",
        },
        {
            "Policy":         "Greedy",
            "DAA Paradigm":   "Greedy — score-based",
            "Complexity":     COMPLEXITY["Greedy"],
            "Hit Ratio (%)":  R["greedy_hit_ratio"],
            "Avg Latency (ms)": R["greedy_avg_latency_ms"],
            "Evictions":      R["greedy_evictions"],
            "Optimality":     "Locally optimal",
        },
        {
            "Policy":         "Belady",
            "DAA Paradigm":   "Greedy — exchange argument",
            "Complexity":     COMPLEXITY["Belady"],
            "Hit Ratio (%)":  R["belady_hit_ratio"],
            "Avg Latency (ms)": R["belady_avg_latency_ms"],
            "Evictions":      R["belady_evictions"],
            "Optimality":     "Provably Optimal (offline)",
        },
        {
            "Policy":         "DP+LRU",
            "DAA Paradigm":   "Dynamic Programming — 0/1 Knapsack",
            "Complexity":     COMPLEXITY["DP+LRU"],
            "Hit Ratio (%)":  R["dp_hit_ratio"],
            "Avg Latency (ms)": R["dp_avg_latency_ms"],
            "Evictions":      R["dp_evictions"],
            "Optimality":     "Optimal admission, approx. eviction",
        },
    ])

    st.dataframe(
        cmp_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Hit Ratio (%)":    st.column_config.NumberColumn(format="%.2f %%"),
            "Avg Latency (ms)": st.column_config.NumberColumn(format="%.2f ms"),
        },
    )

    st.divider()

    # ── Zipf distribution table ───────────────────────────────────────────
    st.subheader("Zipf Traffic Distribution")
    st.markdown(
        f"Theoretical access probabilities for **{int(num_urls)}** URLs with **α = {alpha}**."
    )
    tgen    = TrafficGenerator(num_urls=int(num_urls), alpha=alpha, total_requests=1)
    df_zipf = pd.DataFrame(tgen.url_popularity_table(), columns=["URL", "Probability (%)"])
    st.table(df_zipf.head(20))
    if int(num_urls) > 20:
        st.caption(f"Showing top 20 of {int(num_urls)} URLs.")

else:
    st.info(
        "Adjust the parameters in the sidebar and click **▶ Run Simulation** "
        "to compare all five DAA cache policies."
    )
    st.markdown("""
### Algorithm Overview

| # | Policy | File | DAA Paradigm | Eviction Complexity |
|---|--------|------|-------------|---------------------|
| 1 | LRU | `cache/lru_cache.py` | Heuristic (recency) | **O(1)** |
| 2 | LFU | `cache/lfu_cache.py` | Heuristic (frequency) | **O(log n)** |
| 3 | Greedy | `cache/greedy_cache.py` | Greedy — score-based eviction | **O(log n)** |
| 4 | Belady | `cache/belady_cache.py` | Greedy — exchange argument (optimal) | **O(n log n)** offline |
| 5 | DP+LRU | `cache/dp_admission.py` | Dynamic Programming — 0/1 Knapsack | **O(n·W)** pre-load |
""")
