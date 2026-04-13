# 🧹 JVM GC Observability Dashboard

A Streamlit-based observability dashboard that correlates JVM garbage collection behaviour with application performance metrics.

---

## Features

- **Streaming GC log parser** — processes large logs line-by-line with pre-compiled regex (no full-file load)
- **Multi-format GC support** — JDK 9+ Unified Logging (G1GC, CMS, Parallel, ZGC), classic `-verbose:gc` output
- **JTL / JMeter CSV ingestion** — auto-detects column names, converts epoch-ms timestamps
- **Time-series alignment** — configurable bucket size (5s–120s) joins GC and latency timelines
- **Pearson correlation engine** — GC pause metrics ↔ P50/P95/P99/error rate heatmap
- **GC Storm detection** — sliding-window burst analysis
- **Latency spike detection** — σ-threshold anomaly flagging
- **Bottleneck recommendations** — auto-generated tuning advice based on detected patterns
- **Demo mode** — built-in synthetic data so you can explore without uploading anything

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## File Upload

| File | Format | Notes |
|------|--------|-------|
| GC Log | `.log` / `.txt` | JDK unified log (`-Xlog:gc*`) or classic |
| JTL | `.jtl` / `.csv` | JMeter results with `timestamp` + `elapsed` columns |

### Enable GC Logging on your JVM

```bash
# JDK 9+ (recommended)
-Xlog:gc*:file=gc.log:time,uptime:filecount=5,filesize=20m

# JDK 8
-verbose:gc -XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:gc.log
```

---

## Project Structure

```
gc_dashboard/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── sample_gc.log       # Sample GC log for testing
├── sample_jtl.csv      # Sample JMeter JTL for testing
└── README.md
```

---

## Dashboard Sections

1. **KPI Row** — Total events, Max/Avg pause, Full GC count, GC overhead %, Avg P99
2. **Alerts** — Auto-detected anomalies (STW spikes, storms, high correlation)
3. **GC Pause Timeline** — Scatter plot of all GC events coloured by type
4. **Correlated Latency vs GC** — Dual-panel time-series overlay
5. **Heap Usage** — Before/after heap trend per GC event
6. **GC Cause Breakdown** — Horizontal bar chart of allocation failure reasons
7. **Pause Distribution** — Histogram per GC type
8. **Throughput & Error Rate** — Request volume + error % over time
9. **Correlation Heatmap** — Pearson r matrix (GC metrics × latency metrics)
10. **Raw Data Tables** — GC events, bucketed time-series, storm windows
11. **Bottleneck Summary** — Automated health report + tuning recommendations
