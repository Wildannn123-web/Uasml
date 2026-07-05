import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------------------------
# CONFIG & KONSTRUKSI UI
# ------------------------------------------------------------------
st.set_page_config(page_title="Prediksi Penjualan Toko Buku", layout="wide", page_icon="📚")

st.title("📚 Dashboard Prediksi Penjualan Toko Buku")
st.markdown("""
Aplikasi ini membantu Anda menganalisis data penjualan historis, melatih beberapa model Machine Learning secara otomatis, 
dan menggunakannya untuk memprediksi angka penjualan di masa mendatang.
""")

# --- LANGKAH 1: UPLOAD DATA ---
st.header("1️⃣ Upload & Eksplorasi Data")
uploaded_file = st.file_uploader("Unggah file CSV data penjualan Anda", type=["csv"])

if not uploaded_file:
    st.info("💡 Silakan unggah file CSV data penjualan terlebih dahulu di atas untuk memulai.")
    st.stop()

# Load dan normalisasi nama kolom (hapus spasi di awal/akhir)
df = pd.read_csv(uploaded_file)
df.columns = [str(col).strip() for col in df.columns]

st.success(f"✅ Data berhasil dimuat: {df.shape[0]} baris dan {df.shape[1]} kolom ditemukan.")

# Tampilkan preview data & ringkasan info
col_preview, col_info = st.columns([2, 1])
with col_preview:
    st.subheader("Preview Data (Top 5 Baris)")
    st.dataframe(df.head(5), use_container_width=True)

with col_info:
    st.subheader("Ringkasan Data")
    # Hitung jumlah kolom numerik dan kategorikal
    num_counts = df.select_dtypes(include=["number"]).shape[1]
    cat_counts = df.select_dtypes(exclude=["number"]).shape[1]
    st.write(f"- 🔢 Kolom Angka (Numerik): **{num_counts}**")
    st.write(f"- 🔤 Kolom Teks (Kategorikal): **{cat_counts}**")
    st.write(f"- ⚠️ Total Baris Kosong: **{df.isna().sum().sum()}**")

st.divider()

# --- LANGKAH 2: KONFIGURASI TARGET & FITUR ---
st.header("2️⃣ Konfigurasi Model")
st.caption("Tentukan apa yang ingin ditebak (Target) dan faktor apa saja yang memengaruhinya (Fitur).")

col_target, col_features = st.columns(2)
with col_target:
    target_col = st.selectbox(
        "🎯 Pilih Kolom Target (Variabel yang ingin diprediksi):", 
        df.columns.tolist(),
        index=df.columns.tolist().index("total") if "total" in df.columns else 0
    )

with col_features:
    # Filter kolom identitas/nama pelanggan otomatis agar tidak merusak model
    available_features = [
        c for c in df.columns 
        if c != target_col and not any(kw in c.lower() for kw in ["nama", "customer", "pelanggan", "id"])
    ]
    selected_features = st.multiselect(
        "🧩 Pilih Kolom Fitur (Faktor pendukung):", 
        available_features, 
        default=available_features[:min(4, len(available_features))]
    )

if not selected_features:
    st.warning("⚠️ Anda harus memilih minimal satu fitur untuk melatih model.")
    st.stop()

st.divider()

# --- LANGKAH 3: TRAINING & EVALUASI ---
st.header("3️⃣ Pelatihan & Perbandingan Model")

if st.button("🚀 Mulai Latih Model", type="primary"):
    with st.spinner("Sedang memproses data dan melatih 5 model sekaligus..."):
        
        # Bersihkan data dari target yang kosong
        work_df = df.dropna(subset=[target_col]).copy()
        
        if len(work_df) < 10:
            st.error("❌ Data terlalu sedikit (minimal 10 baris setelah pembersihan data kosong) untuk melatih model.")
            st.stop()
            
        X = work_df[selected_features].copy()
        y = work_df[target_col].astype(float)
        
        # Rekayasa Fitur otomatis untuk tipe Tanggal/Waktu
        for col in X.columns:
            if "tanggal" in col.lower() or "date" in col.lower():
                parsed = pd.to_datetime(X[col], errors='coerce')
                X[f"{col}_bulan"] = parsed.dt.month.fillna(1)
                X[f"{col}_tahun"] = parsed.dt.year.fillna(2026)
                X[f"{col}_hari"] = parsed.dt.day.fillna(1)
                X = X.drop(columns=[col])

        # Split Data (80% Training, 20% Testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Pipeline Preprocessing (Handle data kosong & standarisasi skala)
        num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
        
        preprocessor = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
        ])
        
        # Daftar 5 Model untuk dibandingkan
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        
        results = []
        best_r2 = -float('inf')
        best_model_pipe = None
        best_name = ""
        
        for name, model in models.items():
            pipe = Pipeline([("prep", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            
            # Hitung Metrik Performa
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            
            # Hitung MAPE (Persentase Kesalahan) dengan aman
            denom = np.where(y_test == 0, np.nan, y_test)
            mape = np.nanmean(np.abs((y_test - preds) / denom)) * 100
            
            results.append({
                "Model": name, 
                "Akurasi (R² Score)": r2, 
                "Rata-rata Selisih (MAE)": mae,
                "Error Persen (MAPE)": mape
            })
            
            if r2 > best_r2:
                best_r2 = r2
                best_model_pipe = pipe
                best_name = name
                
        # Simpan semua variabel penting ke session_state agar tidak hilang saat input prediksi
        st.session_state["best_pipe"] = best_model_pipe
        st.session_state["features"] = selected_features
        st.session_state["target"] = target_col
        st.session_state["final_cols"] = X.columns.tolist()
        st.session_state["best_name"] = best_name
        st.session_state["results_df"] = pd.DataFrame(results).sort_values("Akurasi (R² Score)", ascending=False)

# Tampilkan hasil jika data hasil training sudah tersedia di Session State
if "results_df" in st.session_state:
    res_df = st.session_state["results_df"]
    best_name = st.session_state["best_name"]
    
    # Grid Tampilan Grafik Hasil
    col_table, col_chart = st.columns([5, 4])
    
    with col_table:
        st.subheader("📋 Tabel Perbandingan Performa")
        st.dataframe(
            res_df.style.format({
                "Akurasi (R² Score)": "{:.4f}",
                "Rata-rata Selisih (MAE)": "{:,.2f}",
                "Error Persen (MAPE)": "{:.2f}%"
            }),
            use_container_width=True
        )
        st.success(f"🏆 Model terbaik yang dipilih untuk prediksi: **{best_name}**")
        
    with col_chart:
        st.subheader("📊 Visualisasi Akurasi")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.barplot(data=res_df, x="Akurasi (R² Score)", y="Model", ax=ax, palette="viridis")
        ax.set_title("Perbandingan R² Score (Makin Tinggi Makin Bagus)")
        st.pyplot(fig)

st.divider()

# --- LANGKAH 4: PREDIKSI DATA BARU ---
if "best_pipe" in st.session_state:
    st.header("4️⃣ Simulator Prediksi Penjualan Baru")
    st.markdown(f"Masukkan data baru di bawah ini untuk mensimulasikan nilai perkiraan **{st.session_state['target']}**.")
    
    # Bungkus dalam form agar aplikasi tidak reload setiap kali user mengetik/memilih input
    with st.form("form_prediksi"):
        input_data = {}
        cols = st.columns(min(3, len(st.session_state["features"])))
        
        for idx, feat in enumerate(st.session_state["features"]):
            with cols[idx % len(cols)]:
                # Form dinamis berdasarkan tipe data asli kolom
                if "tanggal" in feat.lower() or "date" in feat.lower():
                    input_data[feat] = st.date_input(feat)
                elif pd.api.types.is_numeric_dtype(df[feat]):
                    median_val = float(df[feat].median()) if not df[feat].isna().all() else 0.0
                    input_data[feat] = st.number_input(feat, value=median_val)
                else:
                    unique_options = df[feat].dropna().unique().tolist()
                    input_data[feat] = st.selectbox(feat, unique_options if unique_options else ["-"])
                    
        submit_btn = st.form_submit_button("🔮 Hitung Nilai Prediksi", type="primary")
        
    if submit_btn:
        # Konversi input data ke DataFrame
        pred_df = pd.DataFrame([input_data])
        
        # Samakan transformasi format tanggal dengan data training
        for col in pred_df.columns:
            if "tanggal" in col.lower() or "date" in col.lower():
                parsed = pd.to_datetime(pred_df[col], errors='coerce')
                pred_df[f"{col}_bulan"] = parsed.dt.month
                pred_df[f"{col}_tahun"] = parsed.dt.year
                pred_df[f"{col}_hari"] = parsed.dt.day
                pred_df = pred_df.drop(columns=[col])
                
        # Susun urutan kolom agar persis sama dengan saat melatih model
        pred_df = pred_df[st.session_state["final_cols"]]
        
        # Eksekusi Prediksi
        try:
            hasil_prediksi = st.session_state["best_pipe"].predict(pred_df)[0]
            
            # Tampilkan Hasil Utama
            st.markdown("---")
            col_metric, col_note = st.columns([1, 2])
            with col_metric:
                st.metric(
                    label=f"Hasil Estimasi {st.session_state['target']}", 
                    value=f"{hasil_prediksi:,.2f}"
                )
            with col_note:
                st.info(
                    f"💡 Prediksi ini dihitung menggunakan model **{st.session_state['best_name']}**. "
                    "Pastikan karakteristik data baru yang Anda masukkan berada dalam rentang wajar data historis "
                    "yang Anda unggah di atas agar hasil akurat."
                )
        except Exception as e:
            st.error(f"Gagal melakukan prediksi. Error teknis: {e}")
else:
    st.info("ℹ️ Menu simulator prediksi akan muncul di sini secara otomatis setelah Anda mengklik tombol **'Mulai Latih Model'** di atas.")
