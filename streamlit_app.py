import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Pengaturan Judul Aplikasi
st.set_page_config(page_title="Sistem Prediksi Penjualan Toko Buku", layout="wide")
st.title("📚 Sistem Prediksi & Rekomendasi Item Toko Buku")
st.write("Aplikasi ini memprediksi jenis produk yang dibeli dan membandingkan performa model Machine Learning.")

# 1. Fitur Upload File CSV
st.sidebar.header("📁 Upload & Pengaturan Data")
uploaded_file = st.sidebar.file_uploader("Unggah file CSV Penjualan Toko Buku", type=["csv"])

if uploaded_file is not None:
    # Membaca data yang diunggah
    df = pd.read_csv(uploaded_file)
    
    # Menampilkan Preview Data Utama
    st.subheader("📋 Sampel Data Transaksi")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Fitur Interaktif: Memilih Kolom dari Tabel untuk Eksplorasi
    st.markdown("---")
    st.subheader("🔍 Eksplorasi Kolom Tabel")
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Pilih kolom yang ingin kamu lihat detailnya:", all_columns, default=all_columns[:4])
    
    if selected_columns:
        st.write(df[selected_columns].head(5))
    
    # --- PREPROCESSING DATA ---
    # Membersihkan data dari baris kosong pada kolom krusial
    df = df.dropna(subset=['jenis_item', 'jumlah', 'total', 'tanggal pembelian'])
    
    # Rekayasa Fitur Tanggal
    df['tanggal pembelian'] = pd.to_datetime(df['tanggal pembelian'], errors='coerce')
    df = df.dropna(subset=['tanggal pembelian'])
    df['bulan_pembelian'] = df['tanggal pembelian'].dt.month
    df['hari_pembelian'] = df['tanggal pembelian'].dt.dayofweek
    
    # Encode Target (jenis_item)
    le_item = LabelEncoder()
    df['item_encoded'] = le_item.fit_transform(df['jenis_item'])
    
    # Menentukan Fitur (X) dan Target (y)
    X = df[['jumlah', 'total', 'bulan_pembelian', 'hari_pembelian']]
    y = df['item_encoded']
    
    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standarisasi Fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- PEMILIHAN DAN PERBANDINGAN MODEL ---
    st.markdown("---")
    st.subheader("🤖 Parameter dan Perbandingan Model Machine Learning")
    
    # Pilihan model interaktif di Sidebar
    st.sidebar.header("⚙️ Pilih Model untuk Dibandingkan")
    choose_lr = st.sidebar.checkbox("Logistic Regression", value=True)
    choose_rf = st.sidebar.checkbox("Random Forest", value=True)
    choose_gb = st.sidebar.checkbox("Gradient Boosting", value=True)
    
    # Menyiapkan model yang dipilih oleh user
    models = {}
    if choose_lr:
        models["Logistic Regression"] = LogisticRegression(max_iter=1000, random_state=42)
    if choose_rf:
        # User bisa mengatur hyperparameter n_estimators lewat slider!
        rf_estimators = st.sidebar.slider("RF: Jumlah Trees (n_estimators)", 10, 200, 100, step=10)
        models["Random Forest"] = RandomForestClassifier(n_estimators=rf_estimators, random_state=42)
    if choose_gb:
        gb_estimators = st.sidebar.slider("GB: Jumlah Trees (n_estimators)", 10, 200, 100, step=10)
        models["Gradient Boosting"] = GradientBoostingClassifier(n_estimators=gb_estimators, random_state=42)
        
    if st.button("🚀 Jalankan Perbandingan Model"):
        if not models:
            st.warning("Silakan pilih minimal satu model di sidebar sebelah kiri!")
        else:
            results = {}
            
            # Progress bar untuk visualisasi loading proses training
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (name, model) in enumerate(models.items()):
                status_text.text(f"Sedang melatih dan mengevaluasi {name}...")
                
                # Training model
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
                
                # Menghitung Metrik Evaluasi
                acc = accuracy_score(y_test, predictions)
                prec = precision_score(y_test, predictions, average='macro', zero_division=0)
                rec = recall_score(y_test, predictions, average='macro', zero_division=0)
                f1 = f1_score(y_test, predictions, average='macro', zero_division=0)
                
                results[name] = {
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1-Score": f1
                }
                
                # Update progress bar
                progress_bar.progress(int((idx + 1) / len(models) * 100))
                
            status_text.text("Proses training selesai! ✅")
            
            # Mengubah hasil ke DataFrame
            df_results = pd.DataFrame(results).T
            
            # Menampilkan Tabel Hasil Perbandingan
            st.write("### 📊 Tabel Hasil Performa Model")
            st.dataframe(df_results.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
            
            # Menampilkan Grafik Perbandingan menggunakan Matplotlib/Seaborn
            st.write("### 📈 Visualisasi Akurasi dan F1-Score")
            fig, ax = plt.subplots(figsize=(10, 5))
            df_results[['Accuracy', 'F1-Score']].plot(kind='bar', ax=ax)
            ax.set_ylabel('Skor')
            ax.set_title('Perbandingan Model (Lebih tinggi lebih baik)')
            plt.xticks(rotation=0)
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig)
            
            # Kesimpulan Otomatis
            best_model_name = df_results['F1-Score'].idxmax()
            st.success(f"**Kesimpulan:** Berdasarkan nilai F1-Score, model **{best_model_name}** adalah model terbaik untuk memprediksi jenis item penjualan pada dataset ini!")

else:
    st.info("💡 Silakan unggah file data transaksi penjualan toko buku Anda (.csv) pada panel di sebelah kiri untuk memulai analisa.")
