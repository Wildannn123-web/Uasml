import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Pengaturan halaman Streamlit
st.set_page_config(page_title="Sistem Prediksi Toko Buku", layout="wide")
st.title("🔮 Sistem Prediksi Pembelian Toko Buku")
st.write("Aplikasi ini berfungsi untuk memprediksi kategori produk (`jenis_item`) yang dibeli berdasarkan parameter transaksi.")

# 1. Panel Unggah Data Penjualan (CSV)
st.sidebar.header("📁 Unggah Data Penjualan")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV Penjualan Toko Buku", type=["csv"])

if uploaded_file is not None:
    # Membaca file data yang diunggah
    df = pd.read_csv(uploaded_file)
    
    # --- FITUR PILIH KOLOM TABEL INTERAKTIF ---
    st.markdown("### 📋 Eksplorasi Data Transaksi")
    all_columns = df.columns.tolist()
    
    # Pengguna bisa memilih kolom mana saja dari tabel yang ingin ditampilkan ke layar
    selected_columns = st.multiselect(
        "Pilih kolom dari tabel yang ingin kamu tampilkan:", 
        options=all_columns, 
        default=all_columns
    )
    
    if selected_columns:
        st.dataframe(df[selected_columns].head(10), use_container_width=True)
    else:
        st.warning("Silakan pilih minimal satu kolom untuk menampilkan data tabel.")
    
    # --- PREPROCESSING & TRAINING MODEL OTOMATIS ---
    # Pembersihan data dari baris kosong pada kolom krusial
    df = df.dropna(subset=['jenis_item', 'jumlah', 'total', 'tanggal pembelian'])
    
    # Ekstraksi informasi bulan dan hari dari kolom tanggal
    df['tanggal pembelian'] = pd.to_datetime(df['tanggal pembelian'], errors='coerce')
    df = df.dropna(subset=['tanggal pembelian'])
    df['bulan_pembelian'] = df['tanggal pembelian'].dt.month
    df['hari_pembelian'] = df['tanggal pembelian'].dt.dayofweek
    
    # Encode label target (jenis_item) menjadi angka numeric
    le_item = LabelEncoder()
    df['item_encoded'] = le_item.fit_transform(df['jenis_item'])
    
    # Menentukan Fitur (X) dan Target (y)
    X = df[['jumlah', 'total', 'bulan_pembelian', 'hari_pembelian']]
    y = df['item_encoded']
    
    # Split data menjadi 80% Training dan 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standarisasi skala fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Melatih Model Prediksi (Menggunakan Random Forest)
    @st.cache_resource
    def train_prediction_model(X_tr, y_tr, X_te, y_te):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        acc = accuracy_score(y_te, preds)
        return model, acc

    model, akurasi = train_prediction_model(X_train_scaled, y_train, X_test_scaled, y_test)
    
    st.markdown("---")
    st.success(f"🤖 **Status Model:** AI Prediksi Berhasil Dilatih! (Akurasi Model: {akurasi:.2%})")
    
    # --- FORM INPUT SIMULASI DATA TRANSAKSI BARU ---
    st.markdown("### 🛠️ Form Input Transaksi Baru (Simulasi Prediksi)")
    st.write("Masukkan parameter di bawah ini untuk memprediksi jenis item apa yang dibeli oleh pelanggan:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_jumlah = st.number_input("Jumlah Barang (Qty):", min_value=1, max_value=100, value=1)
        input_total = st.number_input("Total Harga Transaksi (Rp):", min_value=1000, max_value=100000000, value=150000, step=10000)
    
    with col2:
        # Pilihan Bulan Transaksi
        list_bulan = {
            "Januari": 1, "Februari": 2, "Maret": 3, "April": 4, "Mei": 5, "Juni": 6,
            "Juli": 7, "Agustus": 8, "September": 9, "Oktober": 10, "November": 11, "Desember": 12
        }
        input_bulan_nama = st.selectbox("Bulan Transaksi:", list(list_bulan.keys()))
        input_bulan = list_bulan[input_bulan_nama]
        
        # Pilihan Hari Transaksi
        list_hari = {
            "Senin": 0, "Selasa": 1, "Rabu": 2, "Kamis": 3, "Jumat": 4, "Sabtu": 5, "Minggu": 6
        }
        input_hari_nama = st.selectbox("Hari Transaksi:", list(list_hari.keys()))
        input_hari = list_hari[input_hari_nama]
        
    # Tombol Aksi untuk Menjalankan Prediksi
    if st.button("🔮 Jalankan Prediksi"):
        # Menyusun data input menjadi array
        user_data = np.array([[input_jumlah, input_total, input_bulan, input_hari]])
        
        # Penyesuaian skala data input
        user_data_scaled = scaler.transform(user_data)
        
        # Proses Prediksi
        pred_encoded = model.predict(user_data_scaled)
        hasil_prediksi = le_item.inverse_transform(pred_encoded)[0]
        
        # Menghitung probabilitas akurasi tebakan model
        pred_proba = model.predict_proba(user_data_scaled)
        confidence = np.max(pred_proba) * 100
        
        # Menampilkan Hasil Prediksi Utama
        st.markdown("### 📊 Hasil Analisis Prediksi:")
        st.info(f"Berdasarkan kombinasi data tersebut, jenis item yang diprediksi adalah: **{hasil_prediksi}**")
        st.caption(f"Tingkat Keyakinan Model (Confidence Score): {confidence:.2f}%")

else:
    st.info("💡 Silakan unggah file data transaksi penjualan `.csv` pada bilah menu di sebelah kiri untuk mengaktifkan sistem prediksi.")
