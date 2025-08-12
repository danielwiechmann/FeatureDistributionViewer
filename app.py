# app.py
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Feature Distribution Viewer", layout="wide")

# ---------- NM selection (your list) ----------
NM_SELECTED_FEATURES = [
    # Syntactic
    "MLS", "MLC", "MLT", "CS", "CT", "cTT", "dCC", "dCT", "cPC", "cPT", "TS",
    "cNC", "cNT", "cNS", "NPpre", "NPpost",
    # Lexical density/diversity/sophistication
    "LD", "NDW", "cTTR", "rTTR", "lwVAR", "MLWc", "MLWs", "B2KBANC", "B2KBBNC",
    "T10KCOCAw", "T10KCOCAs", "JDCOCAw",
    "3GNLFa", "3GNLFf", "3GNLFs", "3GNLFtv", "3GNLFw"
]
EXCLUDE = {"TID", "Group", "COURSE"}  # not plotted as features

# ---------- Sidebar: file inputs ----------
st.sidebar.header("Load data")

meas_upload = st.sidebar.file_uploader("Upload measurements CSV (must include TID, Group)", type=["csv"])
meas_path   = st.sidebar.text_input("...or path to measurements CSV", value="")

meta_upload = st.sidebar.file_uploader("Upload meta CSV (must include TID, COURSE)", type=["csv"])
meta_path   = st.sidebar.text_input("...or path to meta CSV", value="")

load_btn = st.sidebar.button("Load / Reload")

def load_csv(uploaded, path_str):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    if path_str:
        return pd.read_csv(Path(path_str))
    return None

if "df" not in st.session_state:
    st.session_state.df = None

if load_btn:
    try:
        df_meas = load_csv(meas_upload, meas_path)
        df_meta = load_csv(meta_upload, meta_path)
        if df_meas is None:
            st.error("Please provide the measurements CSV (upload or path).")
        elif df_meta is None:
            st.error("Please provide the meta CSV (upload or path).")
        else:
            # Basic checks
            for col in ("TID", "Group"):
                if col not in df_meas.columns:
                    st.error(f"Measurements CSV must contain '{col}'."); st.stop()
            if "TID" not in df_meta.columns or "COURSE" not in df_meta.columns:
                st.error("Meta CSV must contain 'TID' and 'COURSE'."); st.stop()

            # Map Group strings to 0/1 if necessary
            if df_meas["Group"].dtype == object:
                df_meas["Group"] = df_meas["Group"].map({"AI": 1, "Human": 0})
            df_meas["Group"] = pd.to_numeric(df_meas["Group"], errors="coerce")

            # Merge COURSE onto measurements
            df = df_meas.merge(df_meta[["TID", "COURSE"]], on="TID", how="left")
            st.session_state.df = df
            st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns.")
    except Exception as e:
        st.error(f"Failed to load: {e}")

df = st.session_state.df

st.title("Feature Distribution Viewer")

if df is None:
    st.info("Upload/select both CSVs and click **Load / Reload**. Measurements must include `TID`, `Group`; meta must include `TID`, `COURSE`.")
    st.stop()

# ---------- Feature set toggle ----------
st.sidebar.header("Feature set")
feature_mode = st.sidebar.radio("Use", ["All features", "NM selection"], index=0)

# discover numeric features
num_cols_all = df.select_dtypes(include=["number"]).columns.tolist()
features_all = [c for c in num_cols_all if c not in EXCLUDE]

if feature_mode == "NM selection":
    features = [f for f in NM_SELECTED_FEATURES if f in features_all]
    missing = [f for f in NM_SELECTED_FEATURES if f not in features_all]
    if missing:
        st.sidebar.warning(f"{len(missing)} NM features not in data (skipped).")
else:
    features = features_all

if not features:
    st.error("No numeric features available to plot with the chosen feature set.")
    st.stop()

# ---------- Aggregation toggle (avoid duplicate TIDs) ----------
st.sidebar.header("Rows")
do_agg = st.sidebar.checkbox("Aggregate to one row per (Group, TID) via mean", value=True)

def aggregate_by_tid(df_in: pd.DataFrame) -> pd.DataFrame:
    num_feats = df_in.select_dtypes(include=["number"]).columns.tolist()
    num_feats = [c for c in num_feats if c != "Group"]
    agg_spec = {c: "mean" for c in num_feats}
    agg_spec["COURSE"] = "first"
    out = (df_in.groupby(["Group", "TID"], dropna=False, as_index=False).agg(agg_spec))
    return out

# diagnostics for duplicates
dup_ct = df.groupby(["Group", "TID"]).size()
n_dup_pairs = int((dup_ct > 1).sum())
if n_dup_pairs and do_agg:
    st.caption(f"Aggregated {n_dup_pairs} duplicated (Group, TID) pairs to single rows.")

df_plot = aggregate_by_tid(df) if do_agg else df.copy()

# ---------- Filters ----------
st.sidebar.header("Filters")
human_mode = st.sidebar.radio("Humans", ["All", "By COURSE"], index=0)
courses_all = sorted(df_plot["COURSE"].dropna().astype(str).unique())
selected_courses = courses_all
if human_mode == "By COURSE":
    selected_courses = st.sidebar.multiselect("Select COURSE", options=courses_all, default=courses_all)

include_ai = st.sidebar.checkbox("Include AI group", value=True)

# Feature picker
default_feat = "NPpre" if "NPpre" in features else features[0]
feat = st.sidebar.selectbox("Feature", options=features, index=features.index(default_feat))

# Build subsets
hum = df_plot[df_plot["Group"] == 0].copy()
if human_mode == "By COURSE":
    hum = hum[hum["COURSE"].astype(str).isin(selected_courses)]
ai  = df_plot[df_plot["Group"] == 1].copy() if include_ai else df_plot.iloc[0:0].copy()

hum_sub = hum[["TID", feat]].dropna()
ai_sub  = ai[["TID", feat]].dropna()

# ---------- Plot (original scale, more jitter) ----------
POINT_JITTER = 0.35
POINT_SIZE = 7

fig = go.Figure()

if not hum_sub.empty:
    fig.add_trace(go.Box(
        x=["Human (0)"] * len(hum_sub),
        y=hum_sub[feat],
        name="Human (0)",
        boxpoints="all", jitter=POINT_JITTER, pointpos=0.0,
        marker=dict(size=POINT_SIZE, color="white", line=dict(color="black", width=0.5)),
        line_color="black", fillcolor="white",
        customdata=hum_sub["TID"],
        hovertemplate=f"<b>{feat}</b><br>Group: Human (0)<br>TID: %{{customdata}}<br>value: %{{y:.6g}}<extra></extra>",
        showlegend=False
    ))

if include_ai and not ai_sub.empty:
    fig.add_trace(go.Box(
        x=["AI (1)"] * len(ai_sub),
        y=ai_sub[feat],
        name="AI (1)",
        boxpoints="all", jitter=POINT_JITTER, pointpos=0.0,
        marker=dict(size=POINT_SIZE, color="white", line=dict(color="black", width=0.5)),
        line_color="black", fillcolor="white",
        customdata=ai_sub["TID"],
        hovertemplate=f"<b>{feat}</b><br>Group: AI (1)<br>TID: %{{customdata}}<br>value: %{{y:.6g}}<extra></extra>",
        showlegend=False
    ))

fig.update_yaxes(autorange=True)
fig.update_layout(
    title=f"{feat} — distribution by group (hover shows TID)",
    xaxis_title="Group",
    yaxis_title=feat,
    template="simple_white",
    margin=dict(l=60, r=20, t=60, b=40),
    height=520
)

st.plotly_chart(fig, use_container_width=True)

# ---------- Summary + preview ----------
c1, c2, c3 = st.columns(3)
c1.metric("Human texts shown", int(hum_sub.shape[0]))
c2.metric("AI texts shown", int(ai_sub.shape[0]))
c3.metric("Feature", feat)

with st.expander("Preview filtered rows (TID & feature)"):
    prev = pd.concat([
        hum_sub.assign(GroupLabel="Human"),
        ai_sub.assign(GroupLabel="AI")
    ], ignore_index=True)
    st.dataframe(prev[["GroupLabel", "TID", feat]], use_container_width=True)
