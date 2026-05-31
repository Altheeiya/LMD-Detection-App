import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Konfigurasi Halaman ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LMD Detection System",
    page_icon="",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
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

# ── Konstanta label ────────────────────────────────────────────────────────────
LABEL_DISPLAY = {"0": "Normal", "1": "Suspicious", "2": "Lateral Movement"}
LABEL_COLOR   = {"Normal": "#34d399", "Suspicious": "#fbbf24", "Lateral Movement": "#f87171"}

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    with open("feature_names.json", "r") as f:
        feature_names = json.load(f)
    with open("label_mapping.json", "r") as f:
        label_mapping = json.load(f)
    return model, feature_names, label_mapping

def resolve_label(pred, label_mapping):
    key = str(int(pred))
    if key in label_mapping:
        raw = label_mapping[key]
        return LABEL_DISPLAY.get(key, str(raw))
    return LABEL_DISPLAY.get(key, f"Class_{int(pred)}")

def make_donut(counts: dict):
    labels = list(counts.keys())
    sizes  = list(counts.values())
    colors = [LABEL_COLOR.get(l, "#64748b") for l in labels]
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="none")
    ax.pie(sizes, labels=None, colors=colors, startangle=90, counterclock=False,
           wedgeprops=dict(width=0.52, edgecolor="#0f172a", linewidth=2))
    ax.set_facecolor("none")
    patches = [mpatches.Patch(color=colors[i], label=f"{labels[i]}  ({sizes[i]:,})") for i in range(len(labels))]
    ax.legend(handles=patches, loc="center", frameon=False,
              labelcolor="#cbd5e1", fontsize=9, handlelength=1.2)
    fig.patch.set_alpha(0)
    return fig

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="main-header">
        <div>
            <span class="badge">XGBoost</span>
            <span class="badge">Graph Neural Feature</span>
            <span class="badge">Sysmon Log</span>
        </div>
        <h1> Lateral Movement Detection System</h1>
        <p>Deteksi anomali berbasis Machine Learning pada log Windows Sysmon — PT Bukit Asam Tbk Unit Pelabuhan Tarahan</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        model, feature_names, label_mapping = load_artifacts()
    except Exception as e:
        st.error(f"Gagal memuat model atau artifak: {e}")
        return

    if "df_result" not in st.session_state:
        st.session_state["df_result"] = None

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("###  Panel Kontrol")
        st.markdown("Upload file CSV yang sudah berisi **59 fitur** hasil feature engineering dari notebook.")
        uploaded_file = st.file_uploader("Upload Data Uji (CSV)", type=["csv"])
        st.markdown("---")
        st.markdown("**Kelas Deteksi:**")
        for k, v in LABEL_DISPLAY.items():
            color = LABEL_COLOR[v]
            st.markdown(f"<span style='color:{color}'>●</span> **{v}**", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Model Info:**")
        st.caption("Algoritma: XGBoost (multi:softprob)")
        st.caption("Fitur: 59 (Temporal + EventID + Graph + Freq)")
        st.caption("MCC: 0.8956 | Precision: 0.9738")

    if uploaded_file is None:
        st.info(" Silakan upload file CSV melalui sidebar untuk memulai deteksi.")
        return

    df_input = pd.read_csv(uploaded_file)

    st.markdown('<div class="section-label">1 — Pratinjau Data Input</div>', unsafe_allow_html=True)
    st.dataframe(df_input.head(5), use_container_width=True)
    st.caption(f"Total baris: **{len(df_input):,}** | Total kolom: **{len(df_input.columns)}**")

    missing = [c for c in feature_names if c not in df_input.columns]
    if missing:
        st.error(f" {len(missing)} kolom tidak ditemukan: `{'`, `'.join(missing[:5])}`{'...' if len(missing) > 5 else ''}")
        st.stop()

    if st.button(" Jalankan Deteksi", type="primary"):
        with st.spinner("Model XGBoost sedang menganalisis log..."):
            X = df_input[feature_names].fillna(0).astype(float)
            preds = model.predict(X)
            try:
                proba = model.predict_proba(X)
            except Exception:
                proba = np.zeros((len(preds), 3))

            labels = [resolve_label(p, label_mapping) for p in preds]
            df_result = df_input.copy()
            df_result["HASIL_DETEKSI"]       = labels
            df_result["CONF_Normal (%)"]     = (proba[:, 0] * 100).round(2)
            df_result["CONF_Suspicious (%)"] = (proba[:, 1] * 100).round(2)
            df_result["CONF_LateralMove (%)"]= (proba[:, 2] * 100).round(2)
            st.session_state["df_result"] = df_result
            st.success(" Deteksi selesai!")

    df_result = st.session_state.get("df_result")
    if df_result is None:
        return

    counts_raw = df_result["HASIL_DETEKSI"].value_counts().to_dict()
    total    = len(df_result)
    n_normal = counts_raw.get("Normal", 0)
    n_sus    = counts_raw.get("Suspicious", 0)
    n_atk    = counts_raw.get("Lateral Movement", 0)

    # ── Metric cards ───────────────────────────────────────────────────────────
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

    # ── Donut + interpretasi ───────────────────────────────────────────────────
    st.markdown('<div class="section-label">3 — Distribusi Kelas Prediksi</div>', unsafe_allow_html=True)
    ch, inf = st.columns([1, 2])
    with ch:
        if counts_raw:
            st.pyplot(make_donut(counts_raw), use_container_width=True)
    with inf:
        st.markdown("#### Interpretasi Singkat")
        pct_atk = n_atk / total * 100 if total else 0
        pct_sus = n_sus / total * 100 if total else 0
        if n_atk > 0:
            st.error(f" Terdeteksi **{n_atk:,} log Lateral Movement** ({pct_atk:.2f}%). "
                     f"Segera investigasi baris merah pada tabel di bawah.")
        if n_sus > 0:
            st.warning(f" **{n_sus:,} log Suspicious** ({pct_sus:.2f}%) memerlukan pemantauan lanjutan.")
        if n_atk == 0 and n_sus == 0:
            st.success(" Tidak ada indikasi ancaman pada batch log ini.")
        st.markdown("---")
        st.caption("Confidence score per baris tersedia di kolom `CONF_*` pada tabel di bawah.")

    # ── Tabel hasil ────────────────────────────────────────────────────────────
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
    df_show = df_result[df_result["HASIL_DETEKSI"] == filter_map[filter_opt]] if filter_opt in filter_map else df_result

    priority = ["HASIL_DETEKSI", "CONF_Normal (%)", "CONF_Suspicious (%)", "CONF_LateralMove (%)"]
    others   = [c for c in df_show.columns if c not in priority]
    df_show  = df_show[priority + others].reset_index(drop=True)

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
        f" LM: {(df_show['HASIL_DETEKSI']=='Lateral Movement').sum():,} | "
        f" Suspicious: {(df_show['HASIL_DETEKSI']=='Suspicious').sum():,} | "
        f" Normal: {(df_show['HASIL_DETEKSI']=='Normal').sum():,}"
    )

    # ── Download ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">5 — Ekspor Laporan</div>', unsafe_allow_html=True)
    csv_out = df_result.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=" Download Laporan Deteksi Lengkap (CSV)",
        data=csv_out,
        file_name="laporan_deteksi_lateral_movement.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
