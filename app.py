import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator.simulation_engine import SimulationEngine
#from simulator.traffic_generator import TrafficGenerator

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="CDN Cache Optimizer",
    page_icon="",
    layout="wide",
)

st.title("CDN Cache Optimizer Dashboard")
st.markdown("Simulate and compare **LRU** (Least Recently Used) vs **LFU** (Least Frequently Used) cache eviction policies using Zipf-distributed traffic.")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("Simulation Parameters")
    
    cache_capacity = st.slider(
        "Cache Capacity", 
        min_value=5, max_value=100, value=20, step=5,
        help="Maximum number of items the cache can hold."
    )
    
    total_requests = st.slider(
        "Total Requests", 
        min_value=100, max_value=5000, value=1000, step=100,
        help="Total number of requests to simulate."
    )
    alpha = 1.2
    
    # alpha = st.slider(
    #     "Zipf Alpha (Skew)", 
    #     min_value=0.5, max_value=2.5, value=1.2, step=0.1,
    #     help="Higher alpha means traffic is more skewed towards a few popular URLs (highly cacheable). Lower alpha means traffic is more uniform."
    # )
    
    num_urls = st.number_input(
        "Number of Unique URLs", 
        min_value=10, max_value=500, value=50, step=10
    )
    
    run_sim = st.button("Run Simulation", use_container_width=True, type="primary")


# --- MAIN CONTENT DYNAMIC UPDATE ---
if run_sim:
    with st.spinner("Running simulation through LRU and LFU caches..."):
        
        # 1. Run simulation
        engine = SimulationEngine(
            cache_capacity=cache_capacity,
            num_urls=num_urls,
            total_requests=total_requests,
            alpha=alpha,
            seed=42, # fixed seed for reproducible comparison
        )
        results = engine.run()
        
        # 2. Display Metrics
        st.subheader("Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### LRU Cache")
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Hit Ratio", f"{results['lru_hit_ratio']}%")
            with metric_col2:
                st.metric("Avg Latency", f"{results['lru_avg_latency_ms']} ms")
                
        with col2:
            st.markdown("### LFU Cache")
            metric_col3, metric_col4 = st.columns(2)
            with metric_col3:
                # Calculate delta for LFU hit ratio relative to LRU
                hit_delta = round(results['lfu_hit_ratio'] - results['lru_hit_ratio'], 2)
                st.metric("Hit Ratio", f"{results['lfu_hit_ratio']}%", delta=f"{hit_delta}% vs LRU")
            with metric_col4:
                # Calculate delta for latency (lower is better, so flip delta colour theoretically, but st.metric handles inverse deltas with delta_color="inverse")
                lat_delta = round(results['lfu_avg_latency_ms'] - results['lru_avg_latency_ms'], 2)
                st.metric("Avg Latency", f"{results['lfu_avg_latency_ms']} ms", delta=f"{lat_delta} ms vs LRU", delta_color="inverse")

        st.divider()
        
        # 3. Charts
        st.subheader("Comparison Charts")
        
        # Prepare data for bar chart
        chart_data = pd.DataFrame({
            "Eviction Algorithm": ["LRU", "LFU"],
            "Hit Ratio (%)": [results["lru_hit_ratio"], results["lfu_hit_ratio"]],
            "Avg Latency (ms)": [results["lru_avg_latency_ms"], results["lfu_avg_latency_ms"]],
            "Total Evictions": [results["lru_evictions"], results["lfu_evictions"]]
        })
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("**Hit Ratio Comparison** (Higher is better)")
            st.bar_chart(chart_data, x="Eviction Algorithm", y="Hit Ratio (%)", color="Eviction Algorithm")
            
        with col_chart2:
            st.markdown("**Avg Latency Comparison** (Lower is better)")
            st.bar_chart(chart_data, x="Eviction Algorithm", y="Avg Latency (ms)", color="Eviction Algorithm")


        st.divider()

    #  # 4. Zipf Distribution Table
    #     st.subheader("Zipf Distribution of Generated Traffic")
    #     st.markdown(f"Theoretical probabilities for **{num_urls}** URLs with **Alpha = {alpha}**.")
        
    #     tgen = TrafficGenerator(num_urls=num_urls, alpha=alpha, total_requests=1)
    #     pop_table = tgen.url_popularity_table()
        
    #     # Convert to DataFrame for nice Streamlit table rendering
    #     df_zipf = pd.DataFrame(pop_table, columns=["URL", "Probability (%)"])
        
    #     # Display table
    #     st.table(df_zipf.head(20)) # Show top 20 to avoid massive tables if num_urls is large
    #     if num_urls > 20:
    #         st.caption(f"Showing top 20 of {num_urls} URLs.") 
            

else:
    # Initial state screen
    st.info("Adjust the parameters in the sidebar and click **Run Simulation** to see the results.")
    
