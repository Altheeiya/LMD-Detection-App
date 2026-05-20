import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard Deteksi Lateral Movement", layout="wide")

# Fungsi untuk memuat model dan artifak 
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_model.pkl')
    
    with open('feature_names.json', 'r') as f:
        feature_names = json.load(f)
        
    with open('label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
        
    return model, feature_names, label_mapping

def main():
    st.title("Sistem Deteksi Anomali Lateral Movement")
    st.write("Aplikasi deteksi intrusi berbasis Machine Learning (XGBoost) dan Graph Network Analysis untuk log Windows Sysmon.")
    
    # Load Model
    try:
        model, feature_names, label_mapping = load_artifacts()
    except Exception as e:
        st.error(f"Gagal memuat model atau artifak: {e}")
        return

    if "df_result" not in st.session_state:
        st.session_state["df_result"] = None
    if "df_input" not in st.session_state:
        st.session_state["df_input"] = None

    # Normalisasi label_mapping: pastikan mapping dari numeric string -> label name
    # Bisa jadi file memiliki format {"0": "Normal"} atau {"Normal": 0}
    if all(k.isdigit() for k in label_mapping.keys()):
        label_map = {k: v for k, v in label_mapping.items()}
    else:
        # invert mapping if values are numeric
        try:
            if all(isinstance(v, int) or (isinstance(v, str) and v.isdigit()) for v in label_mapping.values()):
                label_map = {str(v): k for k, v in label_mapping.items()}
            else:
                # fallback: coerce both to strings (best-effort)
                label_map = {str(k): str(v) for k, v in label_mapping.items()}
        except Exception:
            label_map = {str(k): str(v) for k, v in label_mapping.items()}

    # Sidebar untuk Input
    st.sidebar.header("Panel Kontrol")
    st.sidebar.write("Silakan unggah file CSV berisi fitur-fitur yang telah diekstraksi.")
    
    uploaded_file = st.sidebar.file_uploader("Upload Data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        st.subheader("1. Pratinjau Data Input")
        df_input = pd.read_csv(uploaded_file)
        st.session_state["df_input"] = df_input
        st.dataframe(df_input.head(10))
        
        # Validasi Kolom
        missing_cols = [col for col in feature_names if col not in df_input.columns]
        
        if missing_cols:
            st.error(f"Data tidak valid! Kolom berikut hilang: {missing_cols[:5]}...")
        else:
            # Tombol Prediksi
            if st.sidebar.button("Jalankan Deteksi"):
                with st.spinner("Memproses data menggunakan XGBoost..."):
                    # Pastikan urutan kolom sesuai dengan saat training
                    X_input = df_input[feature_names].fillna(0)
                    
                    # Lakukan prediksi
                    predictions = model.predict(X_input)
                    
                    # Map angka prediksi ke label aslinya
                    predicted_labels = [label_map.get(str(p), f"Class_{p}") for p in predictions]
                    
                    # Gabungkan hasil ke dataframe asli
                    df_result = df_input.copy()
                    df_result['HASIL_DETEKSI'] = predicted_labels
                    st.session_state["df_result"] = df_result
                    
                    st.success("Deteksi Selesai!")
    else:
        st.info("Menunggu unggahan data uji. Silakan gunakan menu di sidebar.")

    df_result = st.session_state.get("df_result")
    if df_result is not None:
        # Tampilkan metrik ringkasan
        st.subheader("2. Ringkasan Hasil Deteksi")

        col1, col2, col3 = st.columns(3)
        total_data = len(df_result)
        normal_count = (df_result['HASIL_DETEKSI'] == label_map.get('0', 'Normal')).sum()
        anomaly_count = total_data - normal_count

        col1.metric("Total Aktivitas (Baris)", total_data)
        col2.metric("Aktivitas Normal", normal_count)
        col3.metric("Indikasi Serangan/Anomali", anomaly_count)

        # Tampilkan Data Frame Hasil
        st.subheader("3. Detail Data Terdampak")

        # Filter dinamis berdasarkan label yang sebenarnya
        unique_labels = list(df_result['HASIL_DETEKSI'].unique())
        options = ["Tampilkan Semua"] + [f"{code} - {label_map.get(str(code), str(code))}" for code in unique_labels]
        filter_opsi = st.selectbox("Filter Tampilan Data:", options, key="hasil_filter")

        if filter_opsi == "Tampilkan Semua":
            df_display = df_result
        else:
            selected_code = str(filter_opsi).split(" - ")[0]
            df_display = df_result[df_result['HASIL_DETEKSI'] == selected_code]

        # Batasi ukuran tampilan agar tidak menyebabkan error styling besar
        max_cells = 262144
        n_cells = df_display.shape[0] * df_display.shape[1]

        if n_cells > max_cells:
            st.warning(f"Dataset terlalu besar untuk ditampilkan seluruhnya ({n_cells} sel). Menampilkan 100 baris teratas.")
            st.dataframe(df_display.head(100))
        else:
            st.dataframe(
                df_display,
                column_config={
                    "HASIL_DETEKSI": st.column_config.TextColumn(
                        "HASIL_DETEKSI",
                        help="Status deteksi anomali",
                    )
                }
            )

        # Opsi Unduh Hasil (selalu tersedia)
        csv_output = df_result.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Unduh Laporan Deteksi (CSV)",
            data=csv_output,
            file_name='hasil_deteksi_lateral_movement.csv',
            mime='text/csv',
        )

        # Debug: tampilkan mapping dan label unik jika diminta
        if st.checkbox("Tampilkan debug mapping & label unik", value=False):
            st.write("label_map:", label_map)
            st.write("Unique HASIL_DETEKSI:", df_result['HASIL_DETEKSI'].unique().tolist())
            st.write(df_result['HASIL_DETEKSI'].value_counts().to_dict())

if __name__ == '__main__':
    main()
