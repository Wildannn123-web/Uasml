import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


st.set_page_config(page_title="Prediksi Penjualan Toko Buku", layout="wide")
st.title("📈 Prediksi Penjualan Toko Buku")
st.caption("Upload data penjualan, pilih fitur yang relevan, latih beberapa model, lalu prediksi penjualan berikutnya.")


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = [str(col).strip() for col in data.columns]
    return data


def is_date_like(col_name: str) -> bool:
    lowered = col_name.lower()
    return "tanggal" in lowered or "date" in lowered


def summarize_columns(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Buat ringkasan tiap kolom (selain target) supaya user tahu fitur mana yang layak dipakai."""
    rows = []
    target_numeric = pd.to_numeric(data[target_col], errors="coerce")

    for col in data.columns:
        if col == target_col:
            continue

        series = data[col]
        missing_pct = series.isna().mean() * 100
        n_unique = series.nunique(dropna=True)

        if is_date_like(col):
            col_type = "Tanggal"
            contoh = pd.to_datetime(series, errors="coerce").dropna()
            contoh_val = str(contoh.iloc[0].date()) if not contoh.empty else "-"
            korelasi = "-"
        elif pd.api.types.is_numeric_dtype(series):
            col_type = "Angka"
            contoh_val = str(series.dropna().iloc[0]) if series.notna().any() else "-"
            corr = pd.to_numeric(series, errors="coerce").corr(target_numeric)
            korelasi = f"{corr:.2f}" if pd.notna(corr) else "-"
        else:
            col_type = "Kategori"
            contoh_val = str(series.dropna().astype(str).iloc[0]) if series.notna().any() else "-"
            korelasi = "-"

        rows.append(
            {
                "Kolom": col,
                "Tipe": col_type,
                "Contoh Nilai": contoh_val,
                "Jumlah Unik": n_unique,
                "% Kosong": round(missing_pct, 1),
                "Korelasi ke Target": korelasi,
            }
        )

    return pd.DataFrame(rows)


def prepare_data(data: pd.DataFrame, target_col: str, selected_features: list[str]):
    data = normalize_columns(data)
    data = data.dropna(subset=[target_col])

    work = data[selected_features + [target_col]].copy()
    date_cols = []

    for col in selected_features:
        if is_date_like(col):
            parsed = pd.to_datetime(work[col], errors="coerce")
            if parsed.notna().sum() > 0:
                work[f"{col}_year"] = parsed.dt.year
                work[f"{col}_month"] = parsed.dt.month
                work[f"{col}_day"] = parsed.dt.day
                work[f"{col}_dayofweek"] = parsed.dt.dayofweek
                date_cols.append(col)

    work = work.drop(columns=date_cols)
    X = work.drop(columns=[target_col])
    y = work[target_col].astype(float)
    return X, y


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )


def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


@st.cache_data(show_spinner=False)
def train_and_compare(data: pd.DataFrame, target_col: str, selected_features: list[str]):
    X, y = prepare_data(data, target_col, selected_features)

    if len(X) < 10:
        raise ValueError("Data terlalu sedikit untuk training dan testing.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Buat preprocessor tetap/konsisten berdasarkan X_train secara keseluruhan
    preprocessor = build_preprocessor(X_train)

    results = []
    trained_models = {}

    for name, model in get_models().items():
        pipeline = Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        denominator = np.where(y_test == 0, np.nan, y_test)
        mape = np.nanmean(np.abs((y_test - preds) / denominator)) * 100

        results.append(
            {
                "Metode": name,
                "R2": r2_score(y_test, preds),
                "MAE": mean_absolute_error(y_test, preds),
                "RMSE": mean_squared_error(y_test, preds) ** 0.5,
                "MAPE (%)": mape,
            }
        )
        trained_models[name] = pipeline

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    best_name = results_df.iloc[0]["Metode"]

    # Simpan urutan kolom X yang asli untuk validasi input prediksi nanti
    return results_df, trained_models[best_name], best_name, list(X.columns)


def get_feature_importance(pipeline: Pipeline):
    """Ambil feature importance kalau model best-nya berbasis tree. Return None kalau tidak ada."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None

    preprocessor = pipeline.named_steps["preprocess"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return None

    importance_df = pd.DataFrame(
        {"Fitur": feature_names, "Kontribusi": model.feature_importances_}
    ).sort_values("Kontribusi", ascending=False)
    importance_df["Fitur"] = importance_df["Fitur"].str.replace(r"^(num__|cat__)", "", regex=True)
    return importance_df.head(15)


def build_prediction_input(data: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    values = {}
    cols = st.columns(min(3, len(selected_features)))

    for idx, feature in enumerate(selected_features):
        with cols[idx % len(cols)]:
            series = data[feature]

            if is_date_like(feature):
                default_date = pd.to_datetime(series, errors="coerce").dropna()
                default_value = default_date.max().date() if not default_date.empty else pd.Timestamp.today().date()
                values[feature] = st.date_input(feature, value=default_value)
            elif pd.api.types.is_numeric_dtype(series):
                clean = pd.to_numeric(series, errors="coerce")
                default_value = float(clean.median()) if clean.notna().any() else 0.0
                values[feature] = st.number_input(feature, value=default_value)
            else:
                options = sorted(series.dropna().astype(str).unique().tolist())
                if options:
                    values[feature] = st.selectbox(feature, options)
                else:
                    values[feature] = st.text_input(feature)

    row = pd.DataFrame([values])
    return row


# ----------------------------------------------------------------------------
# 1. Upload data
# ----------------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload file CSV penjualan", type=["csv"])

if uploaded_file is None:
    st.info("Upload file CSV untuk mulai training dan prediksi.")
    st.stop()

df = normalize_columns(pd.read_csv(uploaded_file))

st.subheader("Preview Data")
st.dataframe(df.head(20), use_container_width=True)

numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
default_target = "total" if "total" in df.columns else (numeric_columns[-1] if numeric_columns else df.columns[-1])

# ----------------------------------------------------------------------------
# 2. Konfigurasi target & fitur — dipindah ke sidebar biar area utama fokus
#    ke hasil dan prediksi penjualan
# ----------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Konfigurasi Model")

    target_col = st.selectbox(
        "Kolom target (yang mau diprediksi)",
        df.columns.tolist(),
        index=df.columns.tolist().index(default_target),
        help="Biasanya ini kolom total penjualan, omzet, atau jumlah unit terjual.",
    )

    st.markdown("**Pilih fitur yang mempengaruhi penjualan**")
    st.caption(
        "Lihat ringkasan tiap kolom di bawah sebelum memilih. Untuk kolom angka, "
        "'Korelasi ke Target' mendekati 1 atau -1 berarti hubungannya kuat dengan penjualan."
    )

    summary_df = summarize_columns(df, target_col)
    st.dataframe(
        summary_df.set_index("Kolom"),
        use_container_width=True,
        height=min(300, 40 + 35 * len(summary_df)),
    )

    available_features = [col for col in df.columns if col != target_col]
    numeric_available = summary_df.loc[summary_df["Tipe"] == "Angka", "Kolom"].tolist()
    categorical_available = summary_df.loc[summary_df["Tipe"] == "Kategori", "Kolom"].tolist()
    date_available = summary_df.loc[summary_df["Tipe"] == "Tanggal", "Kolom"].tolist()

    quick_col1, quick_col2, quick_col3 = st.columns(3)
    if quick_col1.button("➕ Semua Angka", use_container_width=True):
        st.session_state["feature_picker"] = list(
            set(st.session_state.get("feature_picker", []) + numeric_available)
        )
    if quick_col2.button("➕ Semua Kategori", use_container_width=True):
        st.session_state["feature_picker"] = list(
            set(st.session_state.get("feature_picker", []) + categorical_available)
        )
    if quick_col3.button("➕ Semua Tanggal", use_container_width=True):
        st.session_state["feature_picker"] = list(
            set(st.session_state.get("feature_picker", []) + date_available)
        )

    if "feature_picker" not in st.session_state:
        default_features = [
            col for col in ["jenis_item", "jumlah", "tanggal pembelian"] if col in available_features
        ]
        if not default_features:
            default_features = available_features[: min(3, len(available_features))]
        st.session_state["feature_picker"] = default_features

    selected_features = st.multiselect(
        "Fitur yang dipakai model",
        available_features,
        key="feature_picker",
        help="Pilih kolom-kolom yang menurutmu ikut menentukan besar-kecilnya penjualan, "
        "misalnya jenis produk, jumlah stok, harga, hari/tanggal transaksi, dll.",
    )

    train_clicked = st.button("🚀 Latih dan Bandingkan Model", type="primary", use_container_width=True)

if not selected_features:
    st.warning("Pilih minimal satu kolom fitur di sidebar sebelah kiri.")
    st.stop()

if train_clicked:
    with st.spinner("Melatih beberapa metode dan menghitung skor..."):
        try:
            results_df, best_model, best_name, final_columns = train_and_compare(
                df, target_col, selected_features
            )
            st.session_state["results_df"] = results_df
            st.session_state["best_model"] = best_model
            st.session_state["best_name"] = best_name
            st.session_state["trained_features"] = selected_features
            st.session_state["final_columns"] = final_columns
        except Exception as exc:
            st.error(f"Training gagal: {exc}")
            st.stop()

if "results_df" not in st.session_state:
    st.info("Atur target & fitur di sidebar, lalu klik **Latih dan Bandingkan Model** untuk mulai.")
    st.stop()

results_df = st.session_state["results_df"]
best_model = st.session_state["best_model"]
best_name = st.session_state["best_name"]
selected_features = st.session_state["trained_features"]
final_columns = st.session_state["final_columns"]

# ----------------------------------------------------------------------------
# 3. Hasil training
# ----------------------------------------------------------------------------

st.subheader("📊 Perbandingan Metode")
st.dataframe(
    results_df.style.format(
        {"R2": "{:.4f}", "MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE (%)": "{:.2f}"}
    ),
    use_container_width=True,
)

best_r2 = results_df.iloc[0]["R2"]
st.success(f"Model terbaik: **{best_name}** dengan R2 = {best_r2:.4f}. Semakin dekat R2 ke 1, semakin baik.")
st.bar_chart(results_df.set_index("Metode")["R2"])

importance_df = get_feature_importance(best_model)
if importance_df is not None:
    st.markdown("**Fitur paling berpengaruh terhadap penjualan (menurut model terbaik):**")
    st.bar_chart(importance_df.set_index("Fitur")["Kontribusi"])

# ----------------------------------------------------------------------------
# 4. Prediksi penjualan
# ----------------------------------------------------------------------------

st.subheader("🔮 Prediksi Penjualan")
st.caption("Masukkan nilai untuk tiap fitur di bawah, lalu klik Prediksi untuk memperkirakan penjualan.")
input_df = build_prediction_input(df, selected_features)

if st.button("Prediksi", type="primary"):
    # Penanganan logika ekstraksi tanggal untuk data input baru yang diprediksi
    prepared_input = input_df.copy()
    date_cols = []

    for col in selected_features:
        if is_date_like(col):
            parsed = pd.to_datetime(prepared_input[col], errors="coerce")
            prepared_input[f"{col}_year"] = parsed.dt.year
            prepared_input[f"{col}_month"] = parsed.dt.month
            prepared_input[f"{col}_day"] = parsed.dt.day
            prepared_input[f"{col}_dayofweek"] = parsed.dt.dayofweek
            date_cols.append(col)

    prepared_input = prepared_input.drop(columns=date_cols)

    # Memastikan urutan kolom input prediksi 100% sama dengan kolom X saat ditraining
    prepared_input = prepared_input[final_columns]

    # Eksekusi Prediksi
    prediction = best_model.predict(prepared_input)[0]
    st.metric("Estimasi Penjualan", f"{prediction:,.0f}")
