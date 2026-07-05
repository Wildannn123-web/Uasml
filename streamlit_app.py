import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Config & UI Awal
st.set_page_config(page_title="Prediksi Toko Buku", layout="wide", page_icon="📚")
st.title("📚 Prediksi Penjualan Toko Buku")

# --- 1. UPLOAD & PERSIAPAN DATA ---
uploaded_file = st.file_uploader("Upload file CSV penjualan", type=["csv"])

if not uploaded_file:
    st.info("⬆️ Upload file CSV untuk mulai.")
    st.stop()

# Load dan bersihkan nama kolom
df = pd.read_csv(uploaded_file)
df.columns = [str(col).strip() for col in df.columns]

st.success(f"Data berhasil dimuat: {df.shape[0]} baris.")
st.dataframe(df.head(5), use_container_width=True)

# Pilih Target & Fitur
col1, col2 = st.columns(2)
with col1:
    target_col = st.selectbox("🎯 Target (yang ingin diprediksi)", df.columns.tolist())
with col2:
    available_features = [c for c in df.columns if c != target_col and "nama" not in c.lower()]
    selected_features = st.multiselect("🧩 Fitur (faktor pendukung)", available_features, default=available_features[:3])

if not selected_features:
    st.warning("Pilih minimal satu fitur!")
    st.stop()

# --- 2. TRAIN & COMPARE MODEL ---
if st.button("🚀 Latih dan Bandingkan Model", type="primary"):
    # Proses Fitur Tanggal & Handle Missing Value Target
    work_df = df.dropna(subset=[target_col]).copy()
    X = work_df[selected_features].copy()
    y = work_df[target_col].astype(float)
    
    # Deteksi & Extract Tanggal otomatis
    for col in X.columns:
        if "tanggal" in col.lower() or "date" in col.lower():
            parsed = pd.to_datetime(X[col], errors='coerce')
            X[f"{col}_month"] = parsed.dt.month.fillna(1)
            X[f"{col}_year"] = parsed.dt.year.fillna(2026)
            X = X.drop(columns=[col])

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessing Pipeline
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])
    
    # Model List (Disederhanakan jadi 3 yang paling mewakili)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
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
        
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        results.append({"Model": name, "R2 (Akurasi)": round(r2, 4), "MAE (Selisih)": round(mae, 2)})
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_pipe = pipe
            best_name = name
            
    # Simpan hasil ke session state agar bisa dipakai untuk prediksi
    st.session_state["best_pipe"] = best_model_pipe
    st.session_state["features"] = selected_features
    st.session_state["target"] = target_col
    st.session_state["final_cols"] = X.columns.tolist()
    
    # Tampilkan Hasil
    st.divider()
    st.subheader("📊 Hasil Perbandingan Model")
    res_df = pd.DataFrame(results).sort_values("R2 (Akurasi)", ascending=False)
    st.dataframe(res_df, use_container_width=True)
    st.success(f"🏆 Model Terbaik: **{best_name}** dengan R² Score: {best_r2:.4f}")

# --- 3. INPUT PREDIKSI BARU ---
if "best_pipe" in st.session_state:
    st.divider()
    st.subheader("🔮 Coba Prediksi Penjualan Baru")
    st.caption(f"Menggunakan fitur yang dipilih sebelumnya untuk memprediksi **{st.session_state['target']}**")
    
    # Gunakan Form agar tidak re-run tiap kali input berubah
    with st.form("prediction_form"):
        input_data = {}
        cols = st.columns(min(3, len(st.session_state["features"])))
        
        for idx, feat in enumerate(st.session_state["features"]):
            with cols[idx % len(cols)]:
                if "tanggal" in feat.lower() or "date" in feat.lower():
                    input_data[feat] = st.date_input(feat)
                elif pd.api.types.is_numeric_dtype(df[feat]):
                    input_data[feat] = st.number_input(feat, value=float(df[feat].median()))
                else:
                    input_data[feat] = st.selectbox(feat, df[feat].dropna().unique())
                    
        submit_btn = st.form_submit_button("Hitung Prediksi", type="primary")
        
    if submit_btn:
        pred_df = pd.DataFrame([input_data])
        
        # Samakan transformasi tanggal dengan saat training
        for col in pred_df.columns:
            if "tanggal" in col.lower() or "date" in col.lower():
                parsed = pd.to_datetime(pred_df[col], errors='coerce')
                pred_df[f"{col}_month"] = parsed.dt.month
                pred_df[f"{col}_year"] = parsed.dt.year
                pred_df = pred_df.drop(columns=[col])
                
        # Urutkan kolom sesuai data training
        pred_df = pred_df[st.session_state["final_cols"]]
        
        # Jalankan Prediksi
        hasil = st.session_state["best_pipe"].predict(pred_df)[0]
        st.metric(label=f"Hasil Perkiraan {st.session_state['target']}", value=f"{hasil:,.0f}")
