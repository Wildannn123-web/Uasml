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

MODEL_DESCRIPTIONS = {
    "Linear Regression": "Model paling sederhana, mengasumsikan hubungan lurus antar variabel. Cepat, tapi kurang akurat jika datanya rumit.",
    "Ridge Regression": "Mirip Linear Regression tapi lebih tahan terhadap data yang saling berkorelasi.",
    "Random Forest": "Gabungan banyak 'pohon keputusan'. Biasanya cukup akurat dan tahan terhadap data yang berisik.",
    "Gradient Boosting": "Membangun model secara bertahap, tiap tahap memperbaiki kesalahan tahap sebelumnya. Sering paling akurat, tapi sedikit lebih lambat.",
    "Extra Trees": "Mirip Random Forest tapi lebih acak saat membelah data, kadang lebih cepat dan lebih stabil.",
}

METRIC_INFO = {
    "R2": "Seberapa baik model menjelaskan variasi data. Rentang biasanya 0–1, makin dekat ke 1 makin baik.",
    "MAE": "Rata-rata selisih (dalam satuan asli) antara prediksi dan angka sebenarnya. Makin kecil makin baik.",
    "RMSE": "Mirip MAE, tapi memberi 'hukuman' lebih besar untuk kesalahan yang sangat meleset. Makin kecil makin baik.",
    "MAPE (%)": "Rata-rata persentase kesalahan prediksi. Makin kecil makin baik (di bawah 10% biasanya dianggap bagus).",
}


st.set_page_config(page_title="Prediksi Penjualan Toko Buku", layout="wide", page_icon="📚")
st.title("📚 Prediksi Penjualan Toko Buku")
st.caption(
    "Upload data penjualan → latih beberapa model sekaligus → lihat model mana yang paling akurat → "
    "coba prediksi angka penjualan baru."
)


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = [str(col).strip() for col in data.columns]
    return data


CUSTOMER_ID_KEYWORDS = ["nama customer", "nama pelanggan", "customer name", "nama_customer", "nama_pelanggan"]


def is_customer_identity_column(col: str) -> bool:
    """Deteksi kolom berisi nama/identitas customer, yang tidak relevan sebagai fitur prediksi."""
    lowered = col.lower().strip()
    if lowered in CUSTOMER_ID_KEYWORDS:
        return True
    has_nama = "nama" in lowered
    has_person_ref = "customer" in lowered or "pelanggan" in lowered
    return has_nama and has_person_ref


def prepare_data(data: pd.DataFrame, target_col: str, selected_features: list[str]):
    data = normalize_columns(data)
    data = data.dropna(subset=[target_col])

    work = data[selected_features + [target_col]].copy()
    date_cols = []

    for col in selected_features:
        lowered = col.lower()
        if "tanggal" in lowered or "date" in lowered:
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
        raise ValueError("Data terlalu sedikit untuk training dan testing (minimal 10 baris).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X_train)

    results = []
    trained_models = {}
    predictions_by_model = {}

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
        predictions_by_model[name] = preds

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    best_name = results_df.iloc[0]["Metode"]

    return (
        results_df,
        trained_models[best_name],
        best_name,
        list(X.columns),
        y_test.reset_index(drop=True),
        pd.Series(predictions_by_model[best_name]),
    )


def build_prediction_input(data: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    values = {}
    cols = st.columns(min(3, len(selected_features)))

    for idx, feature in enumerate(selected_features):
        with cols[idx % len(cols)]:
            series = data[feature]
            lowered = feature.lower()

            if "tanggal" in lowered or "date" in lowered:
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


def get_feature_importance(pipeline, final_columns) -> pd.DataFrame | None:
    """Ambil feature importance / koefisien model jika tersedia, dipetakan ke nama kolom asli."""
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]

    try:
        feature_names_out = preprocessor.get_feature_names_out()
    except Exception:
        return None

    if hasattr(model, "feature_importances_"):
        raw_importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        raw_importance = np.abs(np.ravel(model.coef_))
    else:
        return None

    imp_df = pd.DataFrame({"fitur_olahan": feature_names_out, "skor": raw_importance})

    # Petakan nama fitur hasil encoding kembali ke kolom asli agar mudah dibaca
    def map_to_original(name: str) -> str:
        cleaned = name.split("__", 1)[-1]
        for col in final_columns:
            if cleaned == col or cleaned.startswith(col + "_"):
                return col
        return cleaned

    imp_df["fitur"] = imp_df["fitur_olahan"].apply(map_to_original)
    grouped = imp_df.groupby("fitur")["skor"].sum().sort_values(ascending=False)
    return grouped.reset_index().rename(columns={"skor": "Tingkat Pengaruh"})


tab_data, tab_train, tab_predict = st.tabs(["1️⃣ Data", "2️⃣ Latih & Bandingkan Model", "3️⃣ Prediksi"])


# ------------------------------------------------------------------
# TAB 1: DATA
# ------------------------------------------------------------------
with tab_data:
    st.subheader("Upload Data Penjualan")
    uploaded_file = st.file_uploader("Upload file CSV penjualan", type=["csv"])

    if uploaded_file is None:
        st.info("⬆️ Upload file CSV untuk mulai. Pastikan file berisi kolom target (misalnya jumlah/total penjualan) dan kolom fitur pendukung (jenis barang, tanggal, dll).")
        st.stop()

    df = normalize_columns(pd.read_csv(uploaded_file))
    st.session_state["df"] = df

    st.success(f"Data berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom.")
    st.dataframe(df.head(20), use_container_width=True)

    with st.expander("ℹ️ Ringkasan singkat data"):
        st.write("**Tipe data tiap kolom:**")
        st.dataframe(df.dtypes.astype(str).rename("Tipe Data"), use_container_width=True)
        st.write("**Jumlah data kosong per kolom:**")
        st.dataframe(df.isna().sum().rename("Jumlah Kosong"), use_container_width=True)

    st.divider()
    st.subheader("Pilih Target & Fitur")
    st.caption("Target = angka yang ingin diprediksi. Fitur = informasi yang dipakai model untuk menebak target.")

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    default_target = "total" if "total" in df.columns else (numeric_columns[-1] if numeric_columns else df.columns[-1])

    col1, col2 = st.columns([1, 2])
    with col1:
        target_col = st.selectbox(
            "🎯 Kolom target yang ingin diprediksi",
            df.columns.tolist(),
            index=df.columns.tolist().index(default_target),
        )

    with col2:
        available_features = [
            col for col in df.columns
            if col != target_col and not is_customer_identity_column(col)
        ]
        excluded_customer_cols = [
            col for col in df.columns
            if col != target_col and is_customer_identity_column(col)
        ]

        default_features = [col for col in ["jenis_item", "jumlah", "tanggal pembelian"] if col in available_features]
        if not default_features:
            default_features = available_features[: min(3, len(available_features))]

        selected_features = st.multiselect(
            "🧩 Kolom fitur yang dipakai model",
            available_features,
            default=default_features,
        )

        if excluded_customer_cols:
            st.caption(
                f"ℹ️ Kolom {', '.join(excluded_customer_cols)} disembunyikan dari pilihan fitur "
                "karena berisi nama/identitas customer, bukan pola penjualan yang berguna untuk prediksi."
            )

    st.session_state["target_col"] = target_col
    st.session_state["selected_features"] = selected_features

    if not selected_features:
        st.warning("Pilih minimal satu kolom fitur sebelum lanjut ke tab berikutnya.")
    else:
        st.info("Data siap. Lanjut ke tab **2️⃣ Latih & Bandingkan Model**.")


# ------------------------------------------------------------------
# TAB 2: TRAINING
# ------------------------------------------------------------------
with tab_train:
    if "df" not in st.session_state or not st.session_state.get("selected_features"):
        st.warning("Lengkapi dulu langkah di tab **1️⃣ Data**.")
        st.stop()

    df = st.session_state["df"]
    target_col = st.session_state["target_col"]
    selected_features = st.session_state["selected_features"]

    st.subheader("Latih & Bandingkan Beberapa Model")
    st.caption(
        f"Model akan dilatih untuk memprediksi **{target_col}** menggunakan fitur: "
        f"{', '.join(selected_features)}. Data otomatis dibagi 80% untuk belajar dan 20% untuk uji."
    )

    with st.expander("ℹ️ Apa arti metrik-metrik di tabel hasil?"):
        for metric, desc in METRIC_INFO.items():
            st.markdown(f"- **{metric}**: {desc}")

    if st.button("🚀 Latih dan Bandingkan Model", type="primary"):
        with st.spinner("Melatih 5 model dan menghitung skor masing-masing..."):
            try:
                (
                    results_df,
                    best_model,
                    best_name,
                    final_columns,
                    y_test,
                    best_preds,
                ) = train_and_compare(df, target_col, selected_features)
                st.session_state["results_df"] = results_df
                st.session_state["best_model"] = best_model
                st.session_state["best_name"] = best_name
                st.session_state["final_columns"] = final_columns
                st.session_state["y_test"] = y_test
                st.session_state["best_preds"] = best_preds
            except Exception as exc:
                st.error(f"Training gagal: {exc}")
                st.stop()

    if "results_df" not in st.session_state:
        st.info("Klik tombol di atas untuk mulai melatih model.")
        st.stop()

    results_df = st.session_state["results_df"]
    best_name = st.session_state["best_name"]
    best_model = st.session_state["best_model"]
    final_columns = st.session_state["final_columns"]
    y_test = st.session_state["y_test"]
    best_preds = st.session_state["best_preds"]

    st.divider()
    st.subheader("📊 Tabel Perbandingan Model")

    def highlight_best(row):
        return ["background-color: #d4edda" if row["Metode"] == best_name else "" for _ in row]

    st.dataframe(
        results_df.style.apply(highlight_best, axis=1).format(
            {"R2": "{:.4f}", "MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE (%)": "{:.2f}"}
        ),
        use_container_width=True,
    )

    best_r2 = results_df.iloc[0]["R2"]
    st.success(
        f"🏆 Model terbaik: **{best_name}** (R² = {best_r2:.4f}). "
        f"{MODEL_DESCRIPTIONS.get(best_name, '')}"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Perbandingan skor R² antar model (makin tinggi makin baik)")
        st.bar_chart(results_df.set_index("Metode")["R2"])
    with col_b:
        st.caption(f"Prediksi vs Angka Sebenarnya — model {best_name} (data uji)")
        chart_df = pd.DataFrame({"Aktual": y_test, "Prediksi": best_preds})
        st.scatter_chart(chart_df, x="Aktual", y="Prediksi")

    importance_df = get_feature_importance(best_model, final_columns)
    if importance_df is not None and not importance_df.empty:
        st.divider()
        st.subheader("🔍 Fitur Apa yang Paling Berpengaruh?")
        st.caption("Semakin tinggi nilainya, semakin besar pengaruh fitur tersebut terhadap prediksi model terbaik.")
        st.bar_chart(importance_df.set_index("fitur")["Tingkat Pengaruh"])

    st.info("Model terbaik sudah siap dipakai. Lanjut ke tab **3️⃣ Prediksi**.")


# ------------------------------------------------------------------
# TAB 3: PREDIKSI
# ------------------------------------------------------------------
with tab_predict:
    if "best_model" not in st.session_state:
        st.warning("Latih model dulu di tab **2️⃣ Latih & Bandingkan Model**.")
        st.stop()

    df = st.session_state["df"]
    selected_features = st.session_state["selected_features"]
    final_columns = st.session_state["final_columns"]
    best_model = st.session_state["best_model"]
    best_name = st.session_state["best_name"]
    target_col = st.session_state["target_col"]

    st.subheader("Coba Prediksi Penjualan Baru")
    st.caption(f"Menggunakan model terbaik: **{best_name}**. Isi nilai fitur di bawah lalu klik Prediksi.")

    input_df = build_prediction_input(df, selected_features)

    if st.button("🔮 Prediksi", type="primary"):
        prepared_input = input_df.copy()
        date_cols = []

        for col in selected_features:
            lowered = col.lower()
            if "tanggal" in lowered or "date" in lowered:
                parsed = pd.to_datetime(prepared_input[col], errors="coerce")
                prepared_input[f"{col}_year"] = parsed.dt.year
                prepared_input[f"{col}_month"] = parsed.dt.month
                prepared_input[f"{col}_day"] = parsed.dt.day
                prepared_input[f"{col}_dayofweek"] = parsed.dt.dayofweek
                date_cols.append(col)

        prepared_input = prepared_input.drop(columns=date_cols)
        prepared_input = prepared_input[final_columns]

        prediction = best_model.predict(prepared_input)[0]
        st.metric(f"Perkiraan {target_col}", f"{prediction:,.0f}")
        st.caption(
            "Angka ini adalah perkiraan, bukan kepastian. Semakin tinggi R² model di tab sebelumnya, "
            "semakin bisa diandalkan angka ini."
        )
