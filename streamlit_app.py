import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Book Sales Predictor", layout="wide", page_icon="📚")

st.title("📚 Book Sales Predictor & Model Comparison")
st.markdown("""
Aplikasi ini memungkinkan Anda memilih **variabel target** yang ingin diprediksi (seperti Gross Sales, Publisher Revenue, atau Units Sold), 
membandingkan performa 3 model Machine Learning, dan melakukan prediksi interaktif secara langsung!
""")

# --- Load Data ---
@st.cache_data
def load_data():
    # Pastikan file Books_Data_Clean.csv berada di folder yang sama dengan app.py
    df = pd.read_csv('Books_Data_Clean.csv')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'Books_Data_Clean.csv' tidak ditemukan. Pastikan file tersebut berada di direktori yang sama dengan aplikasi ini.")
    st.stop()

# --- Sidebar: Pemilihan Target & Parameter ---
st.sidebar.header("⚙️ Pengaturan Prediksi")

# User bisa memilih apa yang ingin diprediksi
target_options = {
    'Gross Sales': 'gross sales',
    'Publisher Revenue': 'publisher revenue',
    'Units Sold': 'units sold'
}
selected_target_label = st.sidebar.selectbox("Pilih Variabel yang Ingin Diprediksi:", list(target_options.keys()))
target_column = target_options[selected_target_label]

# Pilihan Model untuk Prediksi Interaktif
model_choice = st.sidebar.selectbox("Pilih Model untuk Prediksi Interaktif:", ['Random Forest', 'Gradient Boosting', 'Linear Regression'])

st.sidebar.markdown("---")
st.sidebar.write("Dataset Terbaca:", df.shape[0], "baris data.")

# --- Persiapan Fitur & Target ---
# Fitur numerik yang digunakan (menghapus target terpilih dari daftar fitur agar tidak bocor)
all_num_features = ['Book_average_rating', 'Book_ratings_count', 'sale price', 'units sold', 'Publishing Year', 'gross sales', 'publisher revenue']
num_features = [f for f in all_num_features if f != target_column and f in df.columns]
cat_features = ['genre', 'Author_Rating']

X = df[num_features + cat_features]
y = df[target_column]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Preprocessing Pipeline ---
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

# --- Pelatihan Model & Evaluasi ---
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

trained_pipelines = {}
results = {}

# Kita gunakan spinner agar user tahu proses training sedang berjalan saat target diganti
with st.spinner(f"Sedang melatih model untuk memprediksi **{selected_target_label}**..."):
    for name, model in models.items():
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Metrik
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2 Score': r2}
        trained_pipelines[name] = clf

# --- Layout Utama: Tampilan Hasil ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"📊 Performa Model ({selected_target_label})")
    df_results = pd.DataFrame(results).T
    st.dataframe(df_results.style.highlight_max(subset=['R2 Score'], color='#90EE90').highlight_min(subset=['MAE', 'RMSE'], color='#90EE90'))

    # Visualisasi R2 Score
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.barplot(x=df_results.index, y=df_results['R2 Score'], palette='viridis', ax=ax)
    ax.set_title('Perbandingan R2 Score antar Model')
    ax.set_ylabel('R2 Score')
    ax.set_ylim(0, 1.1)
    for i, v in enumerate(df_results['R2 Score']):
        ax.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
    st.pyplot(fig)

with col2:
    st.subheader(f"🔮 Input Data untuk Prediksi {selected_target_label}")
    st.markdown("Masukkan spesifikasi buku baru di bawah ini untuk melihat hasil prediksi:")

    # Form Input Dinamis berdasarkan Fitur yang Digunakan
    input_data = {}
    
    with st.form("prediction_form"):
        # Input untuk Fitur Kategorikal terlebih dahulu agar rapi
        st.write("**Fitur Kategorikal**")
        input_data['genre'] = st.selectbox("Genre Buku:", df['genre'].dropna().unique())
        input_data['Author_Rating'] = st.selectbox("Rating Penulis (Author Rating):", df['Author_Rating'].dropna().unique())
        
        st.write("---")
        st.write("**Fitur Numerik**")
        
        # Input dinamis hanya untuk fitur yang tidak menjadi target saat ini
        if 'Publishing Year' in num_features:
            input_data['Publishing Year'] = st.number_input("Tahun Terbit:", min_value=1800, max_value=2026, value=2020)
        
        if 'Book_average_rating' in num_features:
            input_data['Book_average_rating'] = st.slider("Rata-rata Rating Buku:", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
            
        if 'Book_ratings_count' in num_features:
            input_data['Book_ratings_count'] = st.number_input("Jumlah Rating Buku (Counts):", min_value=0, value=50000, step=500)
            
        if 'sale price' in num_features:
            input_data['sale price'] = st.number_input("Harga Jual (Sale Price dalam $):", min_value=0.0, value=5.99, step=0.5)
            
        if 'units sold' in num_features:
            input_data['units sold'] = st.number_input("Unit Terjual (Units Sold):", min_value=0, value=1000, step=10)
            
        if 'gross sales' in num_features:
            input_data['gross sales'] = st.number_input("Gross Sales ($):", min_value=0.0, value=10000.0, step=100.0)
            
        if 'publisher revenue' in num_features:
            input_data['publisher revenue'] = st.number_input("Publisher Revenue ($):", min_value=0.0, value=6000.0, step=100.0)

        # Tombol submit form
        submitted = st.form_submit_button("Prediksi Sekarang!")
        
        if submitted:
            # Ubah input menjadi DataFrame satu baris
            input_df = pd.DataFrame([input_data])
            
            # Memanggil pipeline model terpilih
            selected_pipeline = trained_pipelines[model_choice]
            prediction = selected_pipeline.predict(input_df)[0]
            
            st.success("### Hasil Prediksi:")
            if target_column == 'units sold':
                st.metric(label=f"Estimasi {selected_target_label}", value=f"{int(round(prediction))} Unit")
            else:
                st.metric(label=f"Estimasi {selected_target_label}", value=f"${prediction:,.2f}")
