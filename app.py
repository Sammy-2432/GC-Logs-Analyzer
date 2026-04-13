"""
JVM GC ↔ Application Performance Observability Dashboard
Streamlit-based, streaming GC log parser, JTL correlator, time-series bottleneck detector
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import time
from datetime import datetime, timedelta
from typing import Generator, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from collections import deque

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
   page_title="GC Observability Dashboard",
   page_icon="🔬",
   layout="wide",
   initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
   font-family: 'DM Sans', sans-serif;
}

/* Dark industrial theme */
.stApp {
   background: #0d1117;
   color: #c9d1d9;
}

/* Sidebar */
section[data-testid="stSidebar"] {
   background: #161b22;
   border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label {
   color: #8b949e !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
   background: #161b22;
   border: 1px solid #21262d;
   border-radius: 8px;
   padding: 16px 20px;
}
div[data-testid="metric-container"] label {
   color: #8b949e !important;
   font-family: 'JetBrains Mono', monospace;
   font-size: 0.72rem !important;
   letter-spacing: 0.08em;
   text-transform: uppercase;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
   color: #58a6ff !important;
   font-family: 'JetBrains Mono', monospace;
   font-size: 1.6rem !important;
   font-weight: 700;
}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
   font-family: 'JetBrains Mono', monospace;
   font-size: 0.78rem !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
   background: #161b22;
   border-bottom: 1px solid #21262d;
   gap: 0;
}
.stTabs [data-baseweb="tab"] {
   color: #8b949e;
   font-family: 'JetBrains Mono', monospace;
   font-size: 0.78rem;
   letter-spacing: 0.05em;
   padding: 10px 20px;
   border-radius: 0;
}
.stTabs [aria-selected="true"] {
   color: #58a6ff !important;
   border-bottom: 2px solid #58a6ff !important;
   background: transparent !important;
}

/* Dataframes */
.stDataFrame {
   border: 1px solid #21262d;
   border-radius: 6px;
   overflow: hidden;
}

/* Buttons */
.stButton > button {
   background: #238636;
   color: white;
   border: none;
   border-radius: 6px;
   font-family: 'JetBrains Mono', monospace;
   font-size: 0.82rem;
   font-weight: 600;
   letter-spacing: 0.03em;
   padding: 8px 20px;
   transition: background 0.2s;
}
.stButton > button:hover {
   background: #2ea043;
}

/* File uploader */
[data-testid="stFileUploader"] {
   background: #161b22;
   border: 1px dashed #30363d;
   border-radius: 8px;
}

/* Headers */
h1 {
   font-family: 'JetBrains Mono', monospace !important;
   color: #e6edf3 !important;
   font-size: 1.4rem !important;
   font-weight: 700 !important;
   letter-spacing: -0.02em;
}
h2, h3 {
   font-family: 'JetBrains Mono', monospace !important;
   color: #c9d1d9 !important;
   font-size: 1rem !important;
   font-weight: 600 !important;
}

/* Alert boxes */
.alert-critical {
   background: #3d1a1a;
   border-left: 3px solid #f85149;
   border-radius: 4px;
   padding: 10px 16px;
   color: #f85149;
   font-family: 'JetBrains Mono', monospace;
   font-size: 0.82rem;
   margin: 6px 0;
}
.alert-warn {
   background: #2d2200;
   border-left: 3px solid #d29922;
   border-radius: 4px;
   padding: 10px 16px;
   color: #d29922;
   font-family: 'JetBrains Mono', monospace;
   font-size: 0.82rem;
   margin: 6px 0;
}
.alert-ok {
   background: #122a1a;
   border-left: 3px solid #3fb950;
   border-radius: 4px;
   padding: 10px 16px;
   color: #3fb950;
   font-family: 'JetBrains Mono', monospace;
   font-size: 0.82rem;
   margin: 6px 0;
}

/* Code / mono text */
code {
   font-family: 'JetBrains Mono', monospace !important;
   background: #21262d !important;
   color: #79c0ff !important;
   border-radius: 3px;
   padding: 2px 5px;
   font-size: 0.82rem;
}

/* Expander */
details {
   background: #161b22;
   border: 1px solid #21262d;
   border-radius: 6px;
}

/* Progress */
.stProgress > div > div {
   background: #58a6ff;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

/* Separator */
hr { border-color: #21262d; }

/* Section label */
.section-label {
   font-family: 'JetBrains Mono', monospace;
   font-size: 0.68rem;
   color: #484f58;
   letter-spacing: 0.12em;
   text-transform: uppercase;
   margin-bottom: 8px;
}

/* Insight box */
.insight-box {
   background: #1c2128;
   border: 1px solid #21262d;
   border-radius: 8px;
   padding: 14px 18px;
   font-family: 'JetBrains Mono', monospace;
   font-size: 0.80rem;
   color: #8b949e;
   line-height: 1.7;
}
.insight-box strong {
   color: #58a6ff;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS & REGEX PATTERNS (compiled once)
# ─────────────────────────────────────────────

# G1GC / CMS / ParallelGC / ZGC / Shenandoah unified log patterns
_GC_PATTERNS = {
   "g1_pause": re.compile(
       r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{4}|\d+\.\d+):"
       r"\s*\[GC pause \((?P<reason>[^)]+)\)"
       r".*?(?P<heap_before>\d+)M->(?P<heap_after>\d+)M\((?P<heap_total>\d+)M\)"
       r".*?(?P<pause_ms>\d+\.\d+)\s*ms",
       re.DOTALL,
   ),
   "g1_concurrent": re.compile(
       r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{4}|\d+\.\d+):"
       r"\s*\[GC concurrent-(?P<phase>[^\]]+)\]",
   ),
   "cms_pause": re.compile(
       r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{4}|\d+\.\d+):"
       r"\s*\[(?:GC|Full GC) \((?P<reason>[^)]*)\)"
       r".*?(?P<heap_before>\d+)K->(?P<heap_after>\d+)K\((?P<heap_total>\d+)K\)"
       r".*?(?P<pause_ms>\d+\.\d+)\s*secs?",
       re.DOTALL,
   ),
   "zgc_pause": re.compile(
       r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{4}):"
       r"\s*\[(?P<phase>GC Pause[^\]]+)\]"
       r".*?(?P<pause_ms>\d+\.\d+)\s*ms",
       re.DOTALL,
   ),
   "unified_log": re.compile(
       r"\[(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{4})\]"
       r"\[(?:info|warning)\s*\]\[gc(?:,\w+)*\s*\]"
       r".*?(?P<phase>GC\([0-9]+\).*?)"
       r"(?:(?P<heap_before>\d+)M->(?P<heap_after>\d+)M\((?P<heap_total>\d+)M\))?"
       r".*?(?P<pause_ms>\d+\.\d+)ms",
       re.DOTALL,
   ),
   "full_gc": re.compile(
       r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{4}|\d+\.\d+):"
       r"\s*\[Full GC",
   ),
}

_EPOCH_RE = re.compile(r"^\d+\.\d+$")
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T")


# ─────────────────────────────────────────────
# GC LOG STREAMING PARSER
# ─────────────────────────────────────────────

def _parse_timestamp(ts_str: str, base_dt: Optional[datetime] = None) -> Optional[datetime]:
   """Convert epoch-offset or ISO timestamp to datetime."""
   ts_str = ts_str.strip()
   if _EPOCH_RE.match(ts_str):
       secs = float(ts_str)
       if base_dt:
           return base_dt + timedelta(seconds=secs)
       return datetime(1970, 1, 1) + timedelta(seconds=secs)
   if _ISO_RE.match(ts_str):
       try:
           # Normalize timezone offset
           ts_str = ts_str.replace("+0000", "+00:00").replace("-0000", "+00:00")
           if len(ts_str) == 29 and ts_str[-5] in "+-":
               ts_str = ts_str[:-5] + ts_str[-5] + ":" + ts_str[-4:-2] + ":" + ts_str[-2:]
           return datetime.fromisoformat(ts_str)
       except Exception:
           try:
               return datetime.strptime(ts_str[:23], "%Y-%m-%dT%H:%M:%S.%f")
           except Exception:
               return None
   return None


def stream_parse_gc_log(file_obj, chunk_size: int = 65536) -> Generator[dict, None, None]:
   """
   Memory-efficient streaming GC log parser.
   Reads in chunks, uses a sliding window to handle multi-line entries.
   Yields dicts with normalized GC event data.
   """
   buffer = ""
   base_dt = None
   line_num = 0

   for chunk in iter(lambda: file_obj.read(chunk_size), b""):
       if isinstance(chunk, bytes):
           chunk = chunk.decode("utf-8", errors="replace")
       buffer += chunk

       lines = buffer.split("\n")
       buffer = lines[-1]  # keep incomplete last line

       for line in lines[:-1]:
           line_num += 1
           line = line.rstrip()
           if not line:
               continue

           event = _match_gc_event(line)
           if event:
               if base_dt is None and event.get("timestamp"):
                   base_dt = event["timestamp"]
               yield event

   # flush remainder
   if buffer.strip():
       event = _match_gc_event(buffer.strip())
       if event:
           yield event


def _match_gc_event(line: str) -> Optional[dict]:
   """Try all patterns against a line; return structured event or None."""
   # Try G1 pause first (most common)
   m = _GC_PATTERNS["g1_pause"].search(line)
   if m:
       ts = _parse_timestamp(m.group("ts"))
       if ts:
           return {
               "timestamp": ts,
               "gc_type": "Minor" if "young" in m.group("reason").lower() else "Major",
               "reason": m.group("reason"),
               "heap_before_mb": int(m.group("heap_before")),
               "heap_after_mb": int(m.group("heap_after")),
               "heap_total_mb": int(m.group("heap_total")),
               "pause_ms": float(m.group("pause_ms")),
               "is_full_gc": False,
               "collector": "G1GC",
               "raw": line[:200],
           }

   # CMS / Parallel (K units → convert to MB)
   m = _GC_PATTERNS["cms_pause"].search(line)
   if m:
       ts = _parse_timestamp(m.group("ts"))
       if ts:
           return {
               "timestamp": ts,
               "gc_type": "Major",
               "reason": m.group("reason"),
               "heap_before_mb": int(m.group("heap_before")) // 1024,
               "heap_after_mb": int(m.group("heap_after")) // 1024,
               "heap_total_mb": int(m.group("heap_total")) // 1024,
               "pause_ms": float(m.group("pause_ms")) * 1000,
               "is_full_gc": "Full" in line,
               "collector": "CMS/Parallel",
               "raw": line[:200],
           }

   # ZGC
   m = _GC_PATTERNS["zgc_pause"].search(line)
   if m:
       ts = _parse_timestamp(m.group("ts"))
       if ts:
           return {
               "timestamp": ts,
               "gc_type": "ZGC",
               "reason": m.group("phase"),
               "heap_before_mb": 0,
               "heap_after_mb": 0,
               "heap_total_mb": 0,
               "pause_ms": float(m.group("pause_ms")),
               "is_full_gc": False,
               "collector": "ZGC",
               "raw": line[:200],
           }

   # Unified JVM log format (Java 9+)
   m = _GC_PATTERNS["unified_log"].search(line)
   if m:
       ts = _parse_timestamp(m.group("ts"))
       if ts:
           hb = int(m.group("heap_before")) if m.group("heap_before") else 0
           ha = int(m.group("heap_after")) if m.group("heap_after") else 0
           ht = int(m.group("heap_total")) if m.group("heap_total") else 0
           return {
               "timestamp": ts,
               "gc_type": "Unified",
               "reason": m.group("phase"),
               "heap_before_mb": hb,
               "heap_after_mb": ha,
               "heap_total_mb": ht,
               "pause_ms": float(m.group("pause_ms")),
               "is_full_gc": "Full" in line,
               "collector": "JVM-Unified",
               "raw": line[:200],
           }

   return None


def parse_gc_log_file(uploaded_file) -> pd.DataFrame:
   """Parse an uploaded GC log file via streaming, return DataFrame."""
   events = []
   file_obj = io.BytesIO(uploaded_file.read())
   progress = st.progress(0, text="Parsing GC log…")
   total = len(uploaded_file.getvalue()) if hasattr(uploaded_file, "getvalue") else 1

   count = 0
   for event in stream_parse_gc_log(file_obj):
       events.append(event)
       count += 1
       if count % 500 == 0:
           pct = min(int(file_obj.tell() / max(total, 1) * 100), 95)
           progress.progress(pct, text=f"Parsed {count} GC events…")

   progress.progress(100, text=f"Done — {count} events")
   time.sleep(0.3)
   progress.empty()

   if not events:
       return pd.DataFrame()

   df = pd.DataFrame(events)
   df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
   df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
   df["heap_reclaimed_mb"] = df["heap_before_mb"] - df["heap_after_mb"]
   df["heap_utilization_pct"] = (df["heap_after_mb"] / df["heap_total_mb"].replace(0, np.nan) * 100).round(2)
   df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
   return df


# ─────────────────────────────────────────────
# JTL PARSER
# ─────────────────────────────────────────────

def parse_jtl_file(uploaded_file) -> pd.DataFrame:
   """Parse JMeter JTL (CSV or XML) into DataFrame."""
   raw = uploaded_file.read()
   text = raw.decode("utf-8", errors="replace")

   # Detect CSV vs XML
   if text.strip().startswith("<"):
       return _parse_jtl_xml(text)
   else:
       return _parse_jtl_csv(io.StringIO(text))


def _parse_jtl_csv(stream) -> pd.DataFrame:
   df = pd.read_csv(stream, low_memory=False)
   col_map = {
       "timeStamp": "timestamp_ms", "ts": "timestamp_ms",
       "elapsed": "elapsed_ms", "Latency": "latency_ms",
       "responseCode": "response_code", "success": "success",
       "label": "label", "threadName": "thread",
       "bytes": "bytes", "Connect": "connect_ms",
       "grpThreads": "group_threads", "allThreads": "all_threads",
   }
   df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

   if "timestamp_ms" not in df.columns:
       raise ValueError("JTL CSV missing timeStamp column")

   df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
   df["elapsed_ms"] = pd.to_numeric(df.get("elapsed_ms", 0), errors="coerce").fillna(0)
   df["latency_ms"] = pd.to_numeric(df.get("latency_ms", 0), errors="coerce").fillna(0)
   df["success"] = df.get("success", "true").astype(str).str.lower() == "true"
   df["label"] = df.get("label", "unknown").astype(str)
   return df.sort_values("timestamp").reset_index(drop=True)


def _parse_jtl_xml(text: str) -> pd.DataFrame:
   """Lightweight XML JTL parser without lxml dependency."""
   sample_re = re.compile(
       r'<(?:sample|httpSample)\s+([^>]+?)/?>', re.DOTALL
   )
   attr_re = re.compile(r'(\w+)="([^"]*)"')
   records = []
   for m in sample_re.finditer(text):
       attrs = dict(attr_re.findall(m.group(1)))
       records.append(attrs)

   if not records:
       return pd.DataFrame()

   df = pd.DataFrame(records)
   rename = {"ts": "timestamp_ms", "t": "elapsed_ms", "lt": "latency_ms",
             "rc": "response_code", "s": "success", "lb": "label"}
   df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
   df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp_ms"], errors="coerce"), unit="ms", utc=True)
   df["elapsed_ms"] = pd.to_numeric(df.get("elapsed_ms", 0), errors="coerce").fillna(0)
   df["success"] = df.get("success", "true").astype(str).str.lower() == "true"
   return df.sort_values("timestamp").reset_index(drop=True)


# ─────────────────────────────────────────────
# TIME-SERIES ALIGNMENT & CORRELATION
# ─────────────────────────────────────────────

def align_timeseries(gc_df: pd.DataFrame, jtl_df: pd.DataFrame,
                    bin_seconds: int = 5) -> pd.DataFrame:
   """
   Align GC events and JTL samples onto a common time grid.
   Returns merged DataFrame with per-bin aggregates.
   """
   if gc_df.empty or jtl_df.empty:
       return pd.DataFrame()

   # Common time range
   t_start = max(gc_df["timestamp"].min(), jtl_df["timestamp"].min())
   t_end = min(gc_df["timestamp"].max(), jtl_df["timestamp"].max())

   if t_start >= t_end:
       # Fall back to full range
       t_start = min(gc_df["timestamp"].min(), jtl_df["timestamp"].min())
       t_end = max(gc_df["timestamp"].max(), jtl_df["timestamp"].max())

   freq = f"{bin_seconds}s"

   # Bin GC events
   gc_df2 = gc_df.copy()
   gc_df2 = gc_df2.set_index("timestamp")
   gc_bins = gc_df2.resample(freq).agg(
       gc_count=("pause_ms", "count"),
       gc_pause_sum_ms=("pause_ms", "sum"),
       gc_pause_max_ms=("pause_ms", "max"),
       gc_pause_p95_ms=("pause_ms", lambda x: np.percentile(x, 95) if len(x) else 0),
       heap_after_mb=("heap_after_mb", "mean"),
       heap_total_mb=("heap_total_mb", "mean"),
       full_gc_count=("is_full_gc", "sum"),
       heap_reclaimed_mb=("heap_reclaimed_mb", "sum"),
   ).fillna(0)

   # Bin JTL samples
   jtl_df2 = jtl_df.copy()
   jtl_df2 = jtl_df2.set_index("timestamp")
   jtl_bins = jtl_df2.resample(freq).agg(
       req_count=("elapsed_ms", "count"),
       latency_mean_ms=("elapsed_ms", "mean"),
       latency_p50_ms=("elapsed_ms", lambda x: np.percentile(x, 50) if len(x) else 0),
       latency_p95_ms=("elapsed_ms", lambda x: np.percentile(x, 95) if len(x) else 0),
       latency_p99_ms=("elapsed_ms", lambda x: np.percentile(x, 99) if len(x) else 0),
       latency_max_ms=("elapsed_ms", "max"),
       error_rate=("success", lambda x: (1 - x.mean()) * 100 if len(x) else 0),
       throughput_rps=("elapsed_ms", "count"),
   ).fillna(0)

   # Merge on common index
   merged = gc_bins.join(jtl_bins, how="outer").fillna(0)
   merged = merged.reset_index().rename(columns={"index": "time", "timestamp": "time"})
   merged["time"] = pd.to_datetime(merged.iloc[:, 0])
   merged = merged.sort_values("time")

   # Compute GC overhead %
   merged["gc_overhead_pct"] = (merged["gc_pause_sum_ms"] / (bin_seconds * 1000) * 100).clip(0, 100)

   # Lag-shifted correlation: how much does GC at t affect latency at t+1?
   if len(merged) > 5:
       merged["gc_pause_lag1"] = merged["gc_pause_sum_ms"].shift(1).fillna(0)

   return merged


def compute_correlations(merged: pd.DataFrame) -> dict:
   """Pearson + Spearman correlations between GC metrics and latency."""
   if merged.empty or len(merged) < 5:
       return {}

   pairs = [
       ("gc_pause_sum_ms", "latency_p95_ms"),
       ("gc_pause_sum_ms", "latency_mean_ms"),
       ("gc_overhead_pct", "latency_p95_ms"),
       ("full_gc_count", "error_rate"),
       ("heap_after_mb", "latency_mean_ms"),
       ("gc_pause_lag1", "latency_p95_ms"),
   ]

   results = {}
   for x_col, y_col in pairs:
       if x_col not in merged.columns or y_col not in merged.columns:
           continue
       x = merged[x_col].fillna(0)
       y = merged[y_col].fillna(0)
       if x.std() < 1e-9 or y.std() < 1e-9:
           continue
       try:
           pearson_r, pearson_p = stats.pearsonr(x, y)
           spearman_r, spearman_p = stats.spearmanr(x, y)
           results[f"{x_col} → {y_col}"] = {
               "pearson_r": round(pearson_r, 3),
               "pearson_p": round(pearson_p, 4),
               "spearman_r": round(spearman_r, 3),
               "spearman_p": round(spearman_p, 4),
               "strength": _corr_strength(abs(pearson_r)),
           }
       except Exception:
           pass

   return results


def _corr_strength(r: float) -> str:
   if r >= 0.7:
       return "🔴 Strong"
   if r >= 0.4:
       return "🟡 Moderate"
   return "🟢 Weak"


def detect_bottlenecks(merged: pd.DataFrame, gc_df: pd.DataFrame,
                      pause_threshold_ms: float = 200,
                      overhead_threshold_pct: float = 5.0) -> list:
   """
   Identify time windows where GC is likely causing latency spikes.
   Returns list of incident dicts.
   """
   incidents = []
   if merged.empty:
       return incidents

   # Full GC events
   full_gc_bins = merged[merged["full_gc_count"] > 0]
   for _, row in full_gc_bins.iterrows():
       incidents.append({
           "time": row["time"],
           "type": "Full GC",
           "severity": "critical",
           "gc_pause_ms": row["gc_pause_sum_ms"],
           "latency_p95_ms": row["latency_p95_ms"],
           "detail": f"Full GC detected — {int(row['full_gc_count'])} occurrence(s)",
       })

   # High pause windows
   high_pause = merged[merged["gc_pause_max_ms"] > pause_threshold_ms]
   for _, row in high_pause.iterrows():
       incidents.append({
           "time": row["time"],
           "type": "High Pause",
           "severity": "warning" if row["gc_pause_max_ms"] < pause_threshold_ms * 2 else "critical",
           "gc_pause_ms": row["gc_pause_max_ms"],
           "latency_p95_ms": row["latency_p95_ms"],
           "detail": f"Max GC pause {row['gc_pause_max_ms']:.0f}ms exceeds threshold",
       })

   # GC overhead spikes
   high_overhead = merged[merged["gc_overhead_pct"] > overhead_threshold_pct]
   for _, row in high_overhead.iterrows():
       incidents.append({
           "time": row["time"],
           "type": "GC Overhead",
           "severity": "warning",
           "gc_pause_ms": row["gc_pause_sum_ms"],
           "latency_p95_ms": row["latency_p95_ms"],
           "detail": f"GC overhead {row['gc_overhead_pct']:.1f}% (threshold {overhead_threshold_pct}%)",
       })

   # Latency spike concurrent with GC
   if "latency_p95_ms" in merged.columns:
       lat_mean = merged["latency_p95_ms"].mean()
       lat_std = merged["latency_p95_ms"].std()
       lat_threshold = lat_mean + 2 * lat_std
       concurrent = merged[
           (merged["latency_p95_ms"] > lat_threshold) &
           (merged["gc_pause_sum_ms"] > 0)
       ]
       for _, row in concurrent.iterrows():
           incidents.append({
               "time": row["time"],
               "type": "Latency Spike",
               "severity": "warning",
               "gc_pause_ms": row["gc_pause_sum_ms"],
               "latency_p95_ms": row["latency_p95_ms"],
               "detail": f"P95 latency {row['latency_p95_ms']:.0f}ms > 2σ baseline, concurrent GC active",
           })

   return sorted(incidents, key=lambda x: x["time"])


# ─────────────────────────────────────────────
# SAMPLE DATA GENERATORS
# ─────────────────────────────────────────────

def generate_sample_gc_log() -> str:
   """Generate a realistic sample GC log for demo purposes."""
   import random
   lines = []
   base = datetime(2024, 3, 15, 10, 0, 0)
   heap = 512
   t = 0.0

   for i in range(300):
       t += random.uniform(0.5, 8.0)
       ts = base + timedelta(seconds=t)
       iso = ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0000"

       heap_before = random.randint(200, 800)
       heap_after = random.randint(80, heap_before - 20)
       heap_total = 1024
       pause = random.lognormvariate(3.5, 0.8)  # realistic pause distribution
       reason = random.choice(["G1 Evacuation Pause", "G1 Humongous Allocation",
                               "GCLocker Initiated GC", "Metadata GC Threshold"])

       # Occasional Full GC
       if i % 40 == 0:
           pause *= 15
           lines.append(
               f"{iso}: [Full GC (Ergonomics) "
               f"{heap_before}M->{heap_after}M({heap_total}M), {pause/1000:.4f} secs]"
           )
       else:
           lines.append(
               f"{iso}: [GC pause ({reason}) "
               f"{heap_before}M->{heap_after}M({heap_total}M) {pause:.3f} ms]"
           )

   return "\n".join(lines)


def generate_sample_jtl() -> str:
   """Generate a realistic JTL CSV for demo purposes."""
   import random
   rows = ["timeStamp,elapsed,label,responseCode,success,threadName,bytes,Latency,Connect"]
   base_ms = int(datetime(2024, 3, 15, 10, 0, 0).timestamp() * 1000)
   t = 0

   endpoints = ["/api/search", "/api/checkout", "/api/product", "/api/user", "/api/cart"]

   for i in range(2000):
       t += random.randint(10, 300)
       ts = base_ms + t * 100
       elapsed = int(random.lognormvariate(4.5, 0.9))
       label = random.choice(endpoints)
       code = 200 if random.random() > 0.03 else random.choice([500, 503, 429])
       success = "true" if code == 200 else "false"
       thread = f"Thread-{random.randint(1, 50)}"
       latency = int(elapsed * random.uniform(0.85, 0.98))
       rows.append(f"{ts},{elapsed},{label},{code},{success},{thread},2048,{latency},12")

   return "\n".join(rows)


# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────

PLOT_THEME = dict(
   paper_bgcolor="#0d1117",
   plot_bgcolor="#0d1117",
   font=dict(family="JetBrains Mono, monospace", color="#8b949e", size=11),
   xaxis=dict(gridcolor="#21262d", linecolor="#21262d", tickcolor="#21262d",
              zeroline=False),
   yaxis=dict(gridcolor="#21262d", linecolor="#21262d", tickcolor="#21262d",
              zeroline=False),
   legend=dict(bgcolor="#161b22", bordercolor="#21262d", borderwidth=1,
               font=dict(size=10)),
   margin=dict(l=50, r=20, t=40, b=40),
   hoverlabel=dict(bgcolor="#161b22", bordercolor="#21262d",
                   font=dict(family="JetBrains Mono", color="#c9d1d9", size=11)),
)

COLOR_PAUSE = "#f85149"
COLOR_HEAP = "#3fb950"
COLOR_LATENCY = "#58a6ff"
COLOR_THROUGHPUT = "#d2a8ff"
COLOR_ERROR = "#ff7b72"
COLOR_FULL_GC = "#ffa657"


# ─────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────

def chart_gc_overview(gc_df: pd.DataFrame) -> go.Figure:
   fig = make_subplots(
       rows=3, cols=1, shared_xaxes=True,
       subplot_titles=["GC Pause Duration (ms)", "Heap Usage (MB)", "GC Frequency"],
       vertical_spacing=0.06,
   )

   minor = gc_df[gc_df["gc_type"] != "Full GC"]
   full = gc_df[gc_df["is_full_gc"] == True]

   # Row 1: Pause bars
   fig.add_trace(go.Bar(
       x=minor["timestamp"], y=minor["pause_ms"],
       name="Minor GC", marker_color=COLOR_PAUSE, opacity=0.7,
       hovertemplate="<b>%{x}</b><br>Pause: %{y:.1f}ms<extra></extra>",
   ), row=1, col=1)
   if not full.empty:
       fig.add_trace(go.Bar(
           x=full["timestamp"], y=full["pause_ms"],
           name="Full GC", marker_color=COLOR_FULL_GC,
           hovertemplate="<b>FULL GC</b><br>%{x}<br>Pause: %{y:.1f}ms<extra></extra>",
       ), row=1, col=1)

   # Row 2: Heap
   fig.add_trace(go.Scatter(
       x=gc_df["timestamp"], y=gc_df["heap_after_mb"],
       name="Heap After", line=dict(color=COLOR_HEAP, width=1.5),
       fill="tozeroy", fillcolor="rgba(63,185,80,0.1)",
       hovertemplate="Heap: %{y}MB<extra></extra>",
   ), row=2, col=1)
   fig.add_trace(go.Scatter(
       x=gc_df["timestamp"], y=gc_df["heap_total_mb"],
       name="Heap Max", line=dict(color="#30363d", width=1, dash="dot"),
       hovertemplate="Max: %{y}MB<extra></extra>",
   ), row=2, col=1)

   # Row 3: Rolling GC count (1-min window approximation via 60s bins)
   gc_rolling = gc_df.set_index("timestamp")["pause_ms"].resample("60s").count().reset_index()
   fig.add_trace(go.Bar(
       x=gc_rolling["timestamp"], y=gc_rolling["pause_ms"],
       name="GCs / min", marker_color="#388bfd", opacity=0.8,
   ), row=3, col=1)

   fig.update_layout(height=520, barmode="overlay", **PLOT_THEME,
                     title_text="GC Log Analysis", title_font=dict(size=13, color="#e6edf3"))
   for i in range(1, 4):
       fig.update_xaxes(**PLOT_THEME["xaxis"], row=i, col=1)
       fig.update_yaxes(**PLOT_THEME["yaxis"], row=i, col=1)

   return fig


def chart_correlation_overlay(merged: pd.DataFrame) -> go.Figure:
   fig = make_subplots(
       rows=2, cols=1, shared_xaxes=True,
       subplot_titles=["GC Pause vs Application Latency", "GC Overhead % vs Error Rate"],
       vertical_spacing=0.08,
   )

   # Row 1: GC pause sum + P95 latency
   fig.add_trace(go.Bar(
       x=merged["time"], y=merged["gc_pause_sum_ms"],
       name="GC Pause Sum (ms)", marker_color=COLOR_PAUSE, opacity=0.55,
       yaxis="y1",
   ), row=1, col=1)
   fig.add_trace(go.Scatter(
       x=merged["time"], y=merged["latency_p95_ms"],
       name="P95 Latency (ms)", line=dict(color=COLOR_LATENCY, width=2),
       yaxis="y2",
       hovertemplate="P95: %{y:.0f}ms<extra></extra>",
   ), row=1, col=1)
   fig.add_trace(go.Scatter(
       x=merged["time"], y=merged["latency_p99_ms"],
       name="P99 Latency (ms)", line=dict(color="#79c0ff", width=1, dash="dot"),
       hovertemplate="P99: %{y:.0f}ms<extra></extra>",
   ), row=1, col=1)

   # Row 2: Overhead % + error rate
   fig.add_trace(go.Scatter(
       x=merged["time"], y=merged["gc_overhead_pct"],
       name="GC Overhead %", line=dict(color=COLOR_FULL_GC, width=2),
       fill="tozeroy", fillcolor="rgba(255,166,87,0.12)",
   ), row=2, col=1)
   fig.add_trace(go.Scatter(
       x=merged["time"], y=merged["error_rate"],
       name="Error Rate %", line=dict(color=COLOR_ERROR, width=1.5, dash="dash"),
   ), row=2, col=1)

   fig.update_layout(height=480, **PLOT_THEME,
                     title_text="GC ↔ Performance Correlation")
   for i in range(1, 3):
       fig.update_xaxes(**PLOT_THEME["xaxis"], row=i, col=1)
       fig.update_yaxes(**PLOT_THEME["yaxis"], row=i, col=1)

   return fig


def chart_latency_distribution(jtl_df: pd.DataFrame) -> go.Figure:
   labels = jtl_df["label"].unique() if "label" in jtl_df.columns else ["all"]
   fig = go.Figure()

   colors = [COLOR_LATENCY, COLOR_HEAP, COLOR_PAUSE, COLOR_THROUGHPUT,
             COLOR_ERROR, COLOR_FULL_GC, "#a5d6ff", "#56d364"]

   for i, label in enumerate(labels[:8]):
       subset = jtl_df[jtl_df["label"] == label]["elapsed_ms"] if "label" in jtl_df.columns else jtl_df["elapsed_ms"]
       fig.add_trace(go.Violin(
           y=subset, name=label,
           box_visible=True, meanline_visible=True,
           line_color=colors[i % len(colors)],
           fillcolor=colors[i % len(colors)].replace(")", ", 0.15)").replace("rgb", "rgba") if "rgb" in colors[i % len(colors)] else colors[i % len(colors)],
           opacity=0.7,
       ))

   fig.update_layout(
       height=360, violinmode="group",
       yaxis_title="Response Time (ms)", **PLOT_THEME,
       title_text="Response Time Distribution by Endpoint",
   )
   fig.update_xaxes(**PLOT_THEME["xaxis"])
   fig.update_yaxes(**PLOT_THEME["yaxis"])
   return fig


def chart_pause_histogram(gc_df: pd.DataFrame) -> go.Figure:
   fig = go.Figure()
   fig.add_trace(go.Histogram(
       x=gc_df["pause_ms"], nbinsx=50,
       marker_color=COLOR_PAUSE, opacity=0.75, name="GC Pause",
   ))
   # Percentile lines
   for pct, color in [(50, "#58a6ff"), (95, "#d29922"), (99, "#f85149")]:
       val = np.percentile(gc_df["pause_ms"], pct)
       fig.add_vline(x=val, line_color=color, line_dash="dash", line_width=1.5,
                     annotation_text=f"P{pct}: {val:.0f}ms",
                     annotation_font=dict(color=color, size=10))

   fig.update_layout(
       height=300, xaxis_title="Pause Duration (ms)",
       yaxis_title="Count", **PLOT_THEME,
       title_text="GC Pause Distribution",
   )
   fig.update_xaxes(**PLOT_THEME["xaxis"])
   fig.update_yaxes(**PLOT_THEME["yaxis"])
   return fig


def chart_scatter_correlation(merged: pd.DataFrame, x_col: str, y_col: str,
                              x_label: str, y_label: str) -> go.Figure:
   valid = merged[[x_col, y_col]].dropna()
   if valid.empty:
       return go.Figure()

   # Regression line
   slope, intercept, r, p, _ = stats.linregress(valid[x_col], valid[y_col])
   x_range = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
   y_pred = slope * x_range + intercept

   fig = go.Figure()
   fig.add_trace(go.Scatter(
       x=valid[x_col], y=valid[y_col], mode="markers",
       marker=dict(color=COLOR_LATENCY, size=5, opacity=0.6),
       name="Data points",
   ))
   fig.add_trace(go.Scatter(
       x=x_range, y=y_pred, mode="lines",
       line=dict(color=COLOR_ERROR, width=2, dash="dash"),
       name=f"r={r:.2f} (p={p:.3f})",
   ))

   fig.update_layout(
       height=320, xaxis_title=x_label, yaxis_title=y_label,
       **PLOT_THEME, title_text=f"Correlation: {x_label} → {y_label}",
   )
   fig.update_xaxes(**PLOT_THEME["xaxis"])
   fig.update_yaxes(**PLOT_THEME["yaxis"])
   return fig


def chart_throughput_timeline(merged: pd.DataFrame) -> go.Figure:
   fig = make_subplots(specs=[[{"secondary_y": True}]])

   fig.add_trace(go.Scatter(
       x=merged["time"], y=merged["throughput_rps"],
       name="Throughput (req/bin)", line=dict(color=COLOR_THROUGHPUT, width=1.5),
       fill="tozeroy", fillcolor="rgba(210,168,255,0.1)",
   ), secondary_y=False)

   fig.add_trace(go.Scatter(
       x=merged["time"], y=merged["gc_pause_sum_ms"],
       name="GC Pause Sum (ms)", line=dict(color=COLOR_PAUSE, width=1.5),
   ), secondary_y=True)

   fig.update_layout(height=300, **PLOT_THEME, title_text="Throughput vs GC Activity")
   fig.update_xaxes(**PLOT_THEME["xaxis"])
   fig.update_yaxes(title_text="Requests", secondary_y=False, **PLOT_THEME["yaxis"])
   fig.update_yaxes(title_text="GC Pause (ms)", secondary_y=True, **PLOT_THEME["yaxis"])
   return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
   st.markdown('<div class="section-label">⬡ GC Observability</div>', unsafe_allow_html=True)
   st.markdown("### Configuration")

   st.markdown("---")
   st.markdown('<div class="section-label">Data Sources</div>', unsafe_allow_html=True)

   use_sample = st.checkbox("Use sample data (demo)", value=False)

   gc_file = st.file_uploader("GC Log file (.log/.txt)", type=["log", "txt", "gz"],
                               disabled=use_sample)
   jtl_file = st.file_uploader("JTL file (.jtl/.csv/.xml)", type=["jtl", "csv", "xml"],
                                disabled=use_sample)

   st.markdown("---")
   st.markdown('<div class="section-label">Analysis Settings</div>', unsafe_allow_html=True)

   bin_seconds = st.slider("Time bin (seconds)", 1, 60, 5,
                            help="Resolution for time-series alignment")
   pause_threshold = st.slider("Pause alert threshold (ms)", 50, 2000, 200)
   overhead_threshold = st.slider("GC overhead alert (%)", 1.0, 20.0, 5.0, 0.5)

   st.markdown("---")
   st.markdown('<div class="section-label">Display</div>', unsafe_allow_html=True)
   show_raw = st.checkbox("Show raw GC events table", False)
   show_raw_jtl = st.checkbox("Show raw JTL samples table", False)

   st.markdown("---")
   st.markdown(
       '<div class="section-label">About</div>'
       '<div style="font-family: JetBrains Mono, monospace; font-size: 0.7rem; color: #484f58; line-height: 1.8;">'
       "Supports G1GC · CMS · Parallel<br>"
       "ZGC · Shenandoah · Unified Log<br>"
       "JTL CSV · XML formats"
       "</div>",
       unsafe_allow_html=True,
   )


# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────

st.markdown(
   '<h1>🔬 JVM GC Observability Dashboard</h1>'
   '<p style="font-family: JetBrains Mono, monospace; font-size: 0.78rem; color: #484f58; margin-top: -8px;">'
   "Streaming GC log parser · Time-series correlation · Latency bottleneck detection"
   "</p>",
   unsafe_allow_html=True,
)
st.markdown("---")


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

gc_df = pd.DataFrame()
jtl_df = pd.DataFrame()

if use_sample:
   with st.spinner("Generating sample data…"):
       sample_gc = generate_sample_gc_log()
       sample_jtl = generate_sample_jtl()
       gc_df = parse_gc_log_file(io.BytesIO(sample_gc.encode()))
       jtl_df = _parse_jtl_csv(io.StringIO(sample_jtl))
   st.success(f"✓ Sample data loaded — {len(gc_df)} GC events · {len(jtl_df)} JTL samples")

else:
   col_gc, col_jtl = st.columns(2)
   with col_gc:
       if gc_file:
           try:
               gc_df = parse_gc_log_file(gc_file)
               st.success(f"✓ {len(gc_df)} GC events parsed")
           except Exception as e:
               st.error(f"GC parse error: {e}")

   with col_jtl:
       if jtl_file:
           try:
               jtl_df = parse_jtl_file(jtl_file)
               st.success(f"✓ {len(jtl_df)} JTL records loaded")
           except Exception as e:
               st.error(f"JTL parse error: {e}")

if gc_df.empty and jtl_df.empty and not use_sample:
   st.markdown(
       '<div class="insight-box">'
       "<strong>Getting started</strong><br>"
       "Upload a <strong>GC log</strong> and/or <strong>JTL file</strong> in the sidebar, or enable "
       "<strong>sample data</strong> to explore the dashboard with generated data.<br><br>"
       "Supported GC formats: G1GC, CMS, ParallelGC, ZGC, Shenandoah, JVM Unified Log<br>"
       "Supported JTL formats: CSV (JMeter default), XML"
       "</div>",
       unsafe_allow_html=True,
   )
   st.stop()


# ─────────────────────────────────────────────
# ALIGNED DATA & CORRELATIONS
# ─────────────────────────────────────────────

merged = pd.DataFrame()
correlations = {}
bottlenecks = []

if not gc_df.empty and not jtl_df.empty:
   merged = align_timeseries(gc_df, jtl_df, bin_seconds=bin_seconds)
   correlations = compute_correlations(merged)
   bottlenecks = detect_bottlenecks(merged, gc_df, pause_threshold, overhead_threshold)


# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────

kpi_cols = st.columns(6)

if not gc_df.empty:
   total_pauses = len(gc_df)
   full_gcs = gc_df["is_full_gc"].sum()
   p99_pause = np.percentile(gc_df["pause_ms"], 99)
   mean_pause = gc_df["pause_ms"].mean()
   max_heap_util = gc_df["heap_utilization_pct"].max() if "heap_utilization_pct" in gc_df else 0

   kpi_cols[0].metric("GC Events", f"{total_pauses:,}", delta=f"{int(full_gcs)} Full GC",
                       delta_color="inverse")
   kpi_cols[1].metric("P99 Pause", f"{p99_pause:.0f}ms",
                       delta="⚠ high" if p99_pause > pause_threshold else "✓ ok",
                       delta_color="inverse" if p99_pause > pause_threshold else "normal")
   kpi_cols[2].metric("Mean Pause", f"{mean_pause:.1f}ms")
   kpi_cols[3].metric("Max Heap Util", f"{max_heap_util:.0f}%",
                       delta="⚠ high" if max_heap_util > 85 else "✓ ok",
                       delta_color="inverse" if max_heap_util > 85 else "normal")

if not jtl_df.empty:
   p95_lat = np.percentile(jtl_df["elapsed_ms"], 95)
   err_rate = (1 - jtl_df["success"].mean()) * 100
   tps = len(jtl_df) / max((jtl_df["timestamp"].max() - jtl_df["timestamp"].min()).total_seconds(), 1)
   kpi_cols[4].metric("P95 Latency", f"{p95_lat:.0f}ms")
   kpi_cols[5].metric("Error Rate", f"{err_rate:.2f}%",
                       delta="⚠" if err_rate > 1 else "✓ ok",
                       delta_color="inverse" if err_rate > 1 else "normal")

st.markdown("---")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tabs = st.tabs([
   "📊 GC Overview",
   "🔗 Correlation",
   "⚡ Bottlenecks",
   "📈 Latency Analysis",
   "🔢 Statistics",
   "📋 Raw Data",
])


# ──── TAB 1: GC OVERVIEW ──────────────────────
with tabs[0]:
   if gc_df.empty:
       st.info("Upload a GC log to see analysis.")
   else:
       st.plotly_chart(chart_gc_overview(gc_df), use_container_width=True)

       c1, c2 = st.columns(2)
       with c1:
           st.plotly_chart(chart_pause_histogram(gc_df), use_container_width=True)
       with c2:
           # GC reason breakdown
           reason_counts = gc_df["reason"].value_counts().head(10)
           fig_r = go.Figure(go.Bar(
               x=reason_counts.values, y=reason_counts.index,
               orientation="h", marker_color=COLOR_PAUSE, opacity=0.8,
           ))
           fig_r.update_layout(height=300, **PLOT_THEME,
                                title_text="GC Pause Reasons",
                                xaxis_title="Count",
                                yaxis=dict(**PLOT_THEME["yaxis"], autorange="reversed"))
           fig_r.update_xaxes(**PLOT_THEME["xaxis"])
           st.plotly_chart(fig_r, use_container_width=True)

       # Collector summary
       if "collector" in gc_df.columns:
           st.markdown('<div class="section-label">Collector Summary</div>', unsafe_allow_html=True)
           collector_stats = gc_df.groupby("collector").agg(
               Events=("pause_ms", "count"),
               Mean_Pause_ms=("pause_ms", "mean"),
               P99_Pause_ms=("pause_ms", lambda x: np.percentile(x, 99)),
               Total_Pause_s=("pause_ms", lambda x: x.sum() / 1000),
           ).round(2).reset_index()
           st.dataframe(collector_stats, use_container_width=True, hide_index=True)


# ──── TAB 2: CORRELATION ─────────────────────
with tabs[1]:
   if merged.empty:
       st.info("Upload both GC log and JTL file to see correlation analysis.")
   else:
       st.plotly_chart(chart_correlation_overlay(merged), use_container_width=True)

       st.markdown('<div class="section-label">Statistical Correlations</div>', unsafe_allow_html=True)

       if correlations:
           corr_data = []
           for pair, vals in correlations.items():
               corr_data.append({
                   "Metric Pair": pair,
                   "Pearson r": vals["pearson_r"],
                   "p-value": vals["pearson_p"],
                   "Spearman ρ": vals["spearman_r"],
                   "Strength": vals["strength"],
                   "Significant": "✓" if vals["pearson_p"] < 0.05 else "✗",
               })
           corr_df = pd.DataFrame(corr_data)
           st.dataframe(corr_df, use_container_width=True, hide_index=True)

       st.markdown('<div class="section-label">Scatter Plots</div>', unsafe_allow_html=True)
       sc1, sc2 = st.columns(2)
       with sc1:
           st.plotly_chart(
               chart_scatter_correlation(merged, "gc_pause_sum_ms", "latency_p95_ms",
                                         "GC Pause Sum (ms)", "P95 Latency (ms)"),
               use_container_width=True,
           )
       with sc2:
           st.plotly_chart(
               chart_scatter_correlation(merged, "gc_overhead_pct", "latency_mean_ms",
                                         "GC Overhead %", "Mean Latency (ms)"),
               use_container_width=True,
           )

       # Lag analysis
       if "gc_pause_lag1" in merged.columns:
           st.markdown('<div class="section-label">Lag-1 Correlation (GC at t → Latency at t+1)</div>', unsafe_allow_html=True)
           st.plotly_chart(
               chart_scatter_correlation(merged, "gc_pause_lag1", "latency_p95_ms",
                                         "GC Pause[t-1] (ms)", "P95 Latency[t] (ms)"),
               use_container_width=True,
           )


# ──── TAB 3: BOTTLENECKS ─────────────────────
with tabs[2]:
   if not bottlenecks:
       if merged.empty:
           st.info("Upload both files to enable bottleneck detection.")
       else:
           st.markdown(
               '<div class="alert-ok">✓ No bottlenecks detected above current thresholds. '
               "Adjust thresholds in the sidebar to tune sensitivity.</div>",
               unsafe_allow_html=True,
           )
   else:
       critical = [b for b in bottlenecks if b["severity"] == "critical"]
       warnings = [b for b in bottlenecks if b["severity"] == "warning"]

       c1, c2, c3 = st.columns(3)
       c1.metric("Total Incidents", len(bottlenecks))
       c2.metric("Critical", len(critical), delta_color="inverse", delta="🔴" if critical else "")
       c3.metric("Warnings", len(warnings), delta_color="inverse", delta="🟡" if warnings else "")

       st.markdown("---")

       # Annotated timeline
       if not merged.empty:
           fig_bt = go.Figure()
           fig_bt.add_trace(go.Scatter(
               x=merged["time"], y=merged["latency_p95_ms"],
               name="P95 Latency", line=dict(color=COLOR_LATENCY, width=1.5),
           ))
           fig_bt.add_trace(go.Bar(
               x=merged["time"], y=merged["gc_pause_sum_ms"],
               name="GC Pause Sum", marker_color=COLOR_PAUSE, opacity=0.4, yaxis="y2",
           ))

           # Mark incidents
           for b in bottlenecks:
               color = "#f85149" if b["severity"] == "critical" else "#d29922"
               fig_bt.add_vline(x=b["time"], line_color=color, line_width=1,
                                 line_dash="dot", opacity=0.6)

           fig_bt.update_layout(
               height=350, **PLOT_THEME,
               title_text="Incident Timeline",
               yaxis=dict(**PLOT_THEME["yaxis"], title="Latency (ms)"),
               yaxis2=dict(**PLOT_THEME["yaxis"], title="GC Pause (ms)",
                            overlaying="y", side="right"),
           )
           fig_bt.update_xaxes(**PLOT_THEME["xaxis"])
           st.plotly_chart(fig_bt, use_container_width=True)

       # Incident list
       st.markdown('<div class="section-label">Incident Log</div>', unsafe_allow_html=True)
       for b in bottlenecks[:50]:  # cap display
           icon = "🔴" if b["severity"] == "critical" else "🟡"
           cls = "alert-critical" if b["severity"] == "critical" else "alert-warn"
           st.markdown(
               f'<div class="{cls}">'
               f"{icon} <strong>{b['type']}</strong> · {b['time'].strftime('%H:%M:%S')} — "
               f"{b['detail']} | GC: {b['gc_pause_ms']:.0f}ms | P95: {b['latency_p95_ms']:.0f}ms"
               "</div>",
               unsafe_allow_html=True,
           )

       if len(bottlenecks) > 50:
           st.caption(f"Showing 50 of {len(bottlenecks)} incidents")


# ──── TAB 4: LATENCY ANALYSIS ─────────────────
with tabs[3]:
   if jtl_df.empty:
       st.info("Upload a JTL file to see latency analysis.")
   else:
       if not merged.empty:
           st.plotly_chart(chart_throughput_timeline(merged), use_container_width=True)

       st.plotly_chart(chart_latency_distribution(jtl_df), use_container_width=True)

       # Percentile table by endpoint
       if "label" in jtl_df.columns:
           st.markdown('<div class="section-label">Latency Percentiles by Endpoint</div>', unsafe_allow_html=True)
           pct_table = jtl_df.groupby("label")["elapsed_ms"].agg(
               Count="count",
               Mean=lambda x: round(x.mean(), 1),
               P50=lambda x: round(np.percentile(x, 50), 1),
               P90=lambda x: round(np.percentile(x, 90), 1),
               P95=lambda x: round(np.percentile(x, 95), 1),
               P99=lambda x: round(np.percentile(x, 99), 1),
               Max=lambda x: round(x.max(), 1),
           ).reset_index().rename(columns={"label": "Endpoint"})
           st.dataframe(pct_table, use_container_width=True, hide_index=True)

       # Error breakdown
       if "success" in jtl_df.columns:
           errors = jtl_df[~jtl_df["success"]]
           if not errors.empty and "label" in errors.columns:
               st.markdown('<div class="section-label">Error Breakdown</div>', unsafe_allow_html=True)
               err_counts = errors.groupby("label").size().reset_index(name="Errors")
               err_counts["% of Requests"] = (
                   err_counts["Errors"] / jtl_df.groupby("label").size().reset_index(name="Total")["Total"] * 100
               ).round(2)
               st.dataframe(err_counts, use_container_width=True, hide_index=True)


# ──── TAB 5: STATISTICS ──────────────────────
with tabs[4]:
   c1, c2 = st.columns(2)

   with c1:
       if not gc_df.empty:
           st.markdown('<div class="section-label">GC Pause Statistics</div>', unsafe_allow_html=True)
           gc_stats = gc_df["pause_ms"].describe(percentiles=[.5, .75, .90, .95, .99]).round(2)
           gc_stats_df = gc_stats.reset_index()
           gc_stats_df.columns = ["Statistic", "Value (ms)"]
           st.dataframe(gc_stats_df, use_container_width=True, hide_index=True)

           # Heap reclaim efficiency
           st.markdown('<div class="section-label">Heap Reclaim Efficiency</div>', unsafe_allow_html=True)
           heap_stats = gc_df["heap_reclaimed_mb"].describe(percentiles=[.5, .95]).round(2)
           st.dataframe(heap_stats.reset_index().rename(
               columns={"index": "Statistic", "heap_reclaimed_mb": "Value (MB)"}
           ), use_container_width=True, hide_index=True)

   with c2:
       if not jtl_df.empty:
           st.markdown('<div class="section-label">Response Time Statistics</div>', unsafe_allow_html=True)
           lat_stats = jtl_df["elapsed_ms"].describe(percentiles=[.5, .75, .90, .95, .99]).round(2)
           lat_stats_df = lat_stats.reset_index()
           lat_stats_df.columns = ["Statistic", "Value (ms)"]
           st.dataframe(lat_stats_df, use_container_width=True, hide_index=True)

   if correlations:
       st.markdown("---")
       st.markdown('<div class="section-label">Interpretation Guide</div>', unsafe_allow_html=True)
       st.markdown(
           '<div class="insight-box">'
           "<strong>How to read correlations:</strong><br>"
           "• <strong>Pearson r</strong> measures linear relationship (−1 to +1)<br>"
           "• <strong>Spearman ρ</strong> captures monotonic (non-linear) relationships<br>"
           "• <strong>p &lt; 0.05</strong> = statistically significant at 95% confidence<br>"
           "• <strong>|r| &gt; 0.7</strong> = strong · <strong>0.4–0.7</strong> = moderate · <strong>&lt; 0.4</strong> = weak<br><br>"
           "<strong>Lag-1 correlation</strong>: Measures whether GC at time <em>t</em> "
           "predicts latency degradation at time <em>t+1</em>. A positive lag-1 correlation "
           "indicates GC-induced carry-over effects (e.g., JIT deoptimization, OS scheduling delays)."
           "</div>",
           unsafe_allow_html=True,
       )


# ──── TAB 6: RAW DATA ─────────────────────────
with tabs[5]:
   if not gc_df.empty and show_raw:
       st.markdown('<div class="section-label">GC Events</div>', unsafe_allow_html=True)
       display_cols = ["timestamp", "gc_type", "reason", "pause_ms",
                       "heap_before_mb", "heap_after_mb", "heap_total_mb",
                       "heap_reclaimed_mb", "is_full_gc", "collector"]
       st.dataframe(gc_df[[c for c in display_cols if c in gc_df.columns]].head(1000),
                    use_container_width=True)

       csv_gc = gc_df.to_csv(index=False).encode()
       st.download_button("⬇ Download GC CSV", csv_gc, "gc_events.csv", "text/csv")
   elif not gc_df.empty:
       st.info("Enable 'Show raw GC events table' in the sidebar.")

   st.markdown("---")

   if not jtl_df.empty and show_raw_jtl:
       st.markdown('<div class="section-label">JTL Samples</div>', unsafe_allow_html=True)
       st.dataframe(jtl_df.head(1000), use_container_width=True)

       csv_jtl = jtl_df.to_csv(index=False).encode()
       st.download_button("⬇ Download JTL CSV", csv_jtl, "jtl_samples.csv", "text/csv")
   elif not jtl_df.empty:
       st.info("Enable 'Show raw JTL samples table' in the sidebar.")

   if not merged.empty:
       st.markdown('<div class="section-label">Aligned Time-Series</div>', unsafe_allow_html=True)
       st.dataframe(merged.head(500), use_container_width=True)
       csv_merged = merged.to_csv(index=False).encode()
       st.download_button("⬇ Download Merged CSV", csv_merged, "aligned_timeseries.csv", "text/csv")
