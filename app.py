import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="LMD Detection System", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f2744 100%);
        padding: 2rem 2.5rem; border-radius: 12px;
        border: 1px solid #334155; margin-bottom: 1.5rem;
    }
    .main-header h1 { color: #f1f5f9; font-size: 1.8rem; font-weight: 600; margin: 0 0 0.3rem 0; letter-spacing: -0.02em; }
    .main-header p  { color: #94a3b8; margin: 0; font-size: 0.9rem; }
    .badge {
        display: inline-block; background: #1d4ed8; color: #bfdbfe;
        font-size: 0.7rem; font-weight: 600; padding: 2px 8px;
        border-radius: 4px; margin-right: 6px; letter-spacing: 0.05em; text-transform: uppercase;
    }
    .metric-card {
        background: #1e293b; border: 1px solid #334155;
        border-radius: 10px; padding: 1.2rem 1.5rem; text-align: center;
    }
    .metric-card .label { color: #94a3b8; font-size: 0.78rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.4rem; }
    .metric-card .value { color: #f1f5f9; font-size: 2rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
    .metric-card.danger  .value { color: #f87171; }
    .metric-card.warning .value { color: #fbbf24; }
    .metric-card.safe    .value { color: #34d399; }
    .section-label {
        color: #64748b; font-size: 0.72rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.08em;
        margin: 1.5rem 0 0.6rem 0; padding-bottom: 0.4rem;
        border-bottom: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] .stMarkdown p { color: #94a3b8; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# --- Konstanta ---
LABEL_DISPLAY = {"0": "Normal", "1": "Suspicious", "2": "Lateral Movement"}
LABEL_COLOR   = {"Normal": "#34d399", "Suspicious": "#fbbf24", "Lateral Movement": "#f87171"}

# EventID flags 
CRITICAL_LOGON = [4624, 4625, 4648, 4672]
KERBEROS       = [4768, 4769, 4770, 4771]
NTLM           = [4776]
SYSMON_EVENTS  = [1, 3, 10, 11]

# Graph feature column mapping 
GRAPH_COLS_SOURCE = {
    'in_degree':   'source_in_degree',
    'out_degree':  'source_out_degree',
    'pagerank':    'source_pagerank',
    'betweenness': 'source_betweenness',
    'clustering':  'source_clustering',
}
GRAPH_COLS_DEST = {
    'in_degree':   'dest_in_degree',
    'out_degree':  'dest_out_degree',
    'pagerank':    'dest_pagerank',
    'betweenness': 'dest_betweenness',
    'clustering':  'dest_clustering',
}


# --- Load artifacts ---
@st.cache_resource
def load_artifacts():
    model         = joblib.load("best_model.pkl")
    freq_map      = joblib.load("freq_map.pkl")
    graph_lookup  = joblib.load("graph_lookup.pkl")
    agg_lookup    = joblib.load("agg_lookup.pkl")

    with open("feature_names.json")  as f: feature_names = json.load(f)
    with open("label_mapping.json")  as f: label_mapping = json.load(f)
    with open("col_info.json")       as f: col_info      = json.load(f)

    return model, freq_map, graph_lookup, agg_lookup, feature_names, label_mapping, col_info


# --- Feature engineering pipeline ---
def run_feature_engineering(df_raw, freq_map, graph_lookup, agg_lookup, col_info, feature_names):
    df = df_raw.copy()

    source_col = col_info["source_col"]
    dest_col   = col_info["dest_col"]
    eid_col    = col_info["eid_col"]

    # Deteksi kolom timestamp
    ts_candidates = ["utctime", "systemtime", "timestamp", "datetime", "time"]
    ts_col = next((c for c in df.columns if c.lower() in ts_candidates), None)

    # 1. Temporal features
    if ts_col:
        df["_ts"] = pd.to_datetime(df[ts_col], errors="coerce")
        df["hour"]              = df["_ts"].dt.hour
        df["day_of_week"]       = df["_ts"].dt.weekday
        df["day_of_month"]      = df["_ts"].dt.day
        df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] < 18)).astype(int)
        df["is_weekend"]        = (df["day_of_week"] >= 5).astype(int)
        df = df.drop(columns=["_ts"])
    else:
        for col in ["hour", "day_of_week", "day_of_month", "is_business_hours", "is_weekend"]:
            df[col] = 0

    # 2. EventID flags
    if eid_col and eid_col in df.columns:
        eid = pd.to_numeric(df[eid_col], errors="coerce")
        df["is_critical_logon"]  = eid.isin(CRITICAL_LOGON).astype(int)
        df["is_kerberos_event"]  = eid.isin(KERBEROS).astype(int)
        df["is_ntlm_event"]      = eid.isin(NTLM).astype(int)
        df["is_failed_logon"]    = (eid == 4625).astype(int)
        df["is_explicit_cred"]   = (eid == 4648).astype(int)
        df["is_process_create"]  = (eid == 1).astype(int)
        df["is_network_conn"]    = (eid == 3).astype(int)
        df["is_process_access"]  = (eid == 10).astype(int)
        df["is_sysmon_event"]    = eid.isin(SYSMON_EVENTS).astype(int)
    else:
        eid_flag_cols = [
            "is_critical_logon", "is_kerberos_event", "is_ntlm_event",
            "is_failed_logon", "is_explicit_cred", "is_process_create",
            "is_network_conn", "is_process_access", "is_sysmon_event"
        ]
        for col in eid_flag_cols:
            df[col] = 0

    # 3. Graph features — lookup dari precomputed table, node baru -> 0
    if source_col and source_col in df.columns:
        df[source_col] = df[source_col].astype(str)
        for metric, feat_name in GRAPH_COLS_SOURCE.items():
            df[feat_name] = df[source_col].map(
                lambda ip, m=metric: graph_lookup.get(ip, {}).get(m, 0)
            )
    else:
        for feat_name in GRAPH_COLS_SOURCE.values():
            df[feat_name] = 0

    if dest_col and dest_col in df.columns:
        df[dest_col] = df[dest_col].astype(str)
        for metric, feat_name in GRAPH_COLS_DEST.items():
            df[feat_name] = df[dest_col].map(
                lambda ip, m=metric: graph_lookup.get(ip, {}).get(m, 0)
            )
    else:
        for feat_name in GRAPH_COLS_DEST.values():
            df[feat_name] = 0

    # 4. Agregasi features — lookup dari precomputed table, IP baru -> 0
    if source_col and source_col in df.columns:
        df["source_event_count"]   = df[source_col].map(
            lambda ip: agg_lookup.get(ip, {}).get("source_event_count", 0)
        )
        df["source_unique_events"] = df[source_col].map(
            lambda ip: agg_lookup.get(ip, {}).get("source_unique_events", 0)
        )
    else:
        df["source_event_count"]   = 0
        df["source_unique_events"] = 0

    # 5. Frequency encoding — lookup dari training freq_map, nilai baru -> 0
    for col, mapping in freq_map.items():
        if col in df.columns:
            df[f"{col}_freq"] = df[col].map(mapping).fillna(0)
        else:
            df[f"{col}_freq"] = 0

    # Ambil hanya 59 fitur sesuai urutan training
    missing_feats = [f for f in feature_names if f not in df.columns]
    for f in missing_feats:
        df[f] = 0

    X = df[feature_names].fillna(0).astype(float)
    X = np.nan_to_num(X.values, nan=0, posinf=0, neginf=0)
    return X, df


def resolve_label(pred, label_mapping):
    key = str(int(pred))
    return LABEL_DISPLAY.get(key, label_mapping.get(key, f"Class_{int(pred)}"))


def make_donut(counts):
    labels = list(counts.keys())
    sizes  = list(counts.values())
    colors = [LABEL_COLOR.get(l, "#64748b") for l in labels]
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="none")
    ax.pie(sizes, labels=None, colors=colors, startangle=90, counterclock=False,
           wedgeprops=dict(width=0.52, edgecolor="#0f172a", linewidth=2))
    ax.set_facecolor("none")
    patches = [mpatches.Patch(color=colors[i], label=f"{labels[i]}  ({sizes[i]:,})")
               for i in range(len(labels))]
    ax.legend(handles=patches, loc="center", frameon=False,
              labelcolor="#cbd5e1", fontsize=9, handlelength=1.2)
    fig.patch.set_alpha(0)
    return fig


# --- Main ---
def main():
    st.markdown("""
    <div class="main-header">
        <div>
            <span class="badge">XGBoost</span>
            <span class="badge">Graph Neural Feature</span>
            <span class="badge">Raw Sysmon Log</span>
        </div>
        <h1>Lateral Movement Detection System</h1>
        <p>Deteksi anomali berbasis Machine Learning pada log Windows Sysmon — PT Bukit Asam Tbk Unit Pelabuhan Tarahan</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        model, freq_map, graph_lookup, agg_lookup, feature_names, label_mapping, col_info = load_artifacts()
    except Exception as e:
        st.error(f"Gagal memuat artifacts: {e}")
        st.info("Pastikan file berikut ada di direktori yang sama: best_model.pkl, freq_map.pkl, graph_lookup.pkl, agg_lookup.pkl, feature_names.json, label_mapping.json, col_info.json")
        return

    if "df_result" not in st.session_state:
        st.session_state["df_result"] = None
    if "df_engineered" not in st.session_state:
        st.session_state["df_engineered"] = None

    # Sidebar
    with st.sidebar:
        st.markdown("### Panel Kontrol")
        st.markdown("Upload file CSV log Sysmon **mentah** (raw). Feature engineering akan dijalankan otomatis.")
        uploaded_file = st.file_uploader("Upload Raw Log CSV", type=["csv"])
        st.markdown("---")
        st.markdown("**Kelas Deteksi:**")
        for k, v in LABEL_DISPLAY.items():
            st.markdown(f"<span style='color:{LABEL_COLOR[v]}'>●</span> **{v}**", unsafe_allow_html=True)
        st.markdown("---")
       
        
    

    if uploaded_file is None:
        st.info("Upload file CSV log Sysmon mentah melalui sidebar untuk memulai deteksi.")
        return

    df_raw = pd.read_csv(uploaded_file)

    st.markdown('<div class="section-label">1 — Pratinjau Raw Data</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(5), use_container_width=True)
    st.caption(f"Total baris: **{len(df_raw):,}** | Total kolom: **{len(df_raw.columns)}**")

    # Deteksi kolom identitas yang tersedia
    id_candidates = [
        col_info.get("source_col"), col_info.get("dest_col"),
        col_info.get("eid_col"), "Computer", "Image", "UtcTime", "SystemTime"
    ]
    id_cols = [c for c in id_candidates if c and c in df_raw.columns]

    if st.button("Jalankan Deteksi", type="primary"):
        with st.spinner("Menjalankan feature engineering dan prediksi..."):

            # Feature engineering
            try:
                X, df_eng = run_feature_engineering(
                    df_raw, freq_map, graph_lookup, agg_lookup, col_info, feature_names
                )
            except Exception as e:
                st.error(f"Feature engineering gagal: {e}")
                return

            # Prediksi
            preds = model.predict(X)
            try:
                proba = model.predict_proba(X)
            except Exception:
                proba = np.zeros((len(preds), 3))

            labels = [resolve_label(p, label_mapping) for p in preds]

            # Gabungkan identitas + hasil ke df_raw
            df_result = df_raw.copy()
            df_result["HASIL_DETEKSI"]        = labels
            df_result["CONF_Normal (%)"]      = (proba[:, 0] * 100).round(2)
            df_result["CONF_Suspicious (%)"]  = (proba[:, 1] * 100).round(2)
            df_result["CONF_LateralMove (%)"] = (proba[:, 2] * 100).round(2)

            st.session_state["df_result"]     = df_result
            st.session_state["df_engineered"] = df_eng
        st.success("Deteksi selesai.")

    df_result = st.session_state.get("df_result")
    if df_result is None:
        return

    counts_raw = df_result["HASIL_DETEKSI"].value_counts().to_dict()
    total      = len(df_result)
    n_normal   = counts_raw.get("Normal", 0)
    n_sus      = counts_raw.get("Suspicious", 0)
    n_atk      = counts_raw.get("Lateral Movement", 0)

    # Metric cards
    st.markdown('<div class="section-label">2 — Ringkasan Hasil Deteksi</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="label">Total Log</div><div class="value">{total:,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card safe"><div class="label">Normal</div><div class="value">{n_normal:,}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card warning"><div class="label">Suspicious</div><div class="value">{n_sus:,}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card danger"><div class="label">Lateral Movement</div><div class="value">{n_atk:,}</div></div>', unsafe_allow_html=True)

    # Donut + interpretasi
    st.markdown('<div class="section-label">3 — Distribusi Kelas Prediksi</div>', unsafe_allow_html=True)
    ch, inf = st.columns([1, 2])
    with ch:
        if counts_raw:
            st.pyplot(make_donut(counts_raw), use_container_width=True)
    with inf:
        st.markdown("#### Interpretasi")
        pct_atk = n_atk / total * 100 if total else 0
        pct_sus = n_sus / total * 100 if total else 0
        if n_atk > 0:
            st.error(f"Terdeteksi **{n_atk:,} log Lateral Movement** ({pct_atk:.2f}%). "
                     f"Segera investigasi baris merah pada tabel di bawah.")
        if n_sus > 0:
            st.warning(f"**{n_sus:,} log Suspicious** ({pct_sus:.2f}%) memerlukan pemantauan lanjutan.")
        if n_atk == 0 and n_sus == 0:
            st.success("Tidak ada indikasi ancaman pada batch log ini.")
        st.markdown("---")
        st.caption("Confidence score per baris tersedia di kolom CONF_* pada tabel di bawah.")

    # Tabel hasil 
    st.markdown('<div class="section-label">4 — Detail Data & Confidence Score</div>', unsafe_allow_html=True)

    filter_opt = st.selectbox(
        "Filter tampilan:",
        ["Semua", "Hanya Lateral Movement", "Hanya Suspicious", "Hanya Normal"]
    )
    filter_map = {
        "Hanya Lateral Movement": "Lateral Movement",
        "Hanya Suspicious":       "Suspicious",
        "Hanya Normal":           "Normal",
    }
    df_show = (df_result[df_result["HASIL_DETEKSI"] == filter_map[filter_opt]]
               if filter_opt in filter_map else df_result)

    # Kolom identitas di depan, lalu confidence dll
    conf_cols = ["HASIL_DETEKSI", "CONF_Normal (%)", "CONF_Suspicious (%)", "CONF_LateralMove (%)"]
    others    = [c for c in df_show.columns if c not in conf_cols and c not in id_cols]
    df_show   = df_show[conf_cols + id_cols + others].reset_index(drop=True)

    def highlight_row(row):
        label = row["HASIL_DETEKSI"]
        if label == "Lateral Movement":
            return ["background-color: #450a0a; color: #fca5a5"] * len(row)
        elif label == "Suspicious":
            return ["background-color: #451a03; color: #fcd34d"] * len(row)
        return [""] * len(row)

    MAX_DISPLAY = 2000
    if len(df_show) > MAX_DISPLAY:
        st.warning(f"Menampilkan {MAX_DISPLAY:,} dari {len(df_show):,} baris. Download untuk data lengkap.")
        df_show = df_show.head(MAX_DISPLAY)

    st.dataframe(df_show.style.apply(highlight_row, axis=1), use_container_width=True, height=420)
    st.caption(
        f"Ditampilkan: {len(df_show):,} baris | "
        f"LM: {(df_show['HASIL_DETEKSI']=='Lateral Movement').sum():,} | "
        f"Suspicious: {(df_show['HASIL_DETEKSI']=='Suspicious').sum():,} | "
        f"Normal: {(df_show['HASIL_DETEKSI']=='Normal').sum():,}"
    )

    # Download hasil prediksi
    st.markdown('<div class="section-label">5 — Ekspor Laporan</div>', unsafe_allow_html=True)
    csv_out = df_result.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Laporan Deteksi Lengkap (CSV)",
        data=csv_out,
        file_name="laporan_deteksi_lateral_movement.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
