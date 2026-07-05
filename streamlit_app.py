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
st.title("Prediksi Penjualan Toko Buku")


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = [str(col).strip() for col in data.columns]
    return data


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
        raise ValueError("Data terlalu sedikit untuk training dan testing.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results = []
    trained_models = {}

    for name, model in get_models().items():
        pipeline = Pipeline(
            [
                ("preprocess", build_preprocessor(X_train)),
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
    return results_df, trained_models[best_name], best_name, list(X.columns)


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
    for col in row.columns:
        if "tanggal" in col.lower() or "date" in col.lower():
            row[col] = pd.to_datetime(row[col]).astype(str)
    return row


uploaded_file = st.file_uploader("Upload file CSV penjualan", type=["csv"])

if uploaded_file is None:
    st.info("Upload file CSV untuk mulai training dan prediksi.")
    st.stop()

df = normalize_columns(pd.read_csv(uploaded_file))

st.subheader("Preview Data")
st.dataframe(df.head(20), use_container_width=True)

numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
default_target = "total" if "total" in df.columns else (numeric_columns[-1] if numeric_columns else df.columns[-1])

left, right = st.columns([1, 2])
with left:
    target_col = st.selectbox(
        "Kolom target yang ingin diprediksi",
        df.columns.tolist(),
        index=df.columns.tolist().index(default_target),
    )

with right:
    available_features = [col for col in df.columns if col != target_col]
    default_features = [col for col in ["jenis_item", "jumlah", "tanggal pembelian"] if col in available_features]
    if not default_features:
        default_features = available_features[: min(3, len(available_features))]

    selected_features = st.multiselect(
        "Kolom fitur yang dipakai model",
        available_features,
        default=default_features,
    )

if not selected_features:
    st.warning("Pilih minimal satu kolom fitur.")
    st.stop()

if st.button("Latih dan Bandingkan Model", type="primary"):
    with st.spinner("Melatih beberapa metode dan menghitung skor..."):
        try:
            results_df, best_model, best_name, final_columns = train_and_compare(
                df, target_col, selected_features
            )
            st.session_state["results_df"] = results_df
            st.session_state["best_model"] = best_model
            st.session_state["best_name"] = best_name
            st.session_state["selected_features"] = selected_features
            st.session_state["final_columns"] = final_columns
        except Exception as exc:
            st.error(f"Training gagal: {exc}")
            st.stop()

if "results_df" not in st.session_state:
    st.stop()

results_df = st.session_state["results_df"]
best_model = st.session_state["best_model"]
best_name = st.session_state["best_name"]
selected_features = st.session_state["selected_features"]

st.subheader("Perbandingan Metode")
st.dataframe(
    results_df.style.format(
        {"R2": "{:.4f}", "MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE (%)": "{:.2f}"}
    ),
    use_container_width=True,
)

best_r2 = results_df.iloc[0]["R2"]
st.success(f"Model terbaik: {best_name} dengan R2 = {best_r2:.4f}. Semakin dekat R2 ke 1, semakin baik.")
st.bar_chart(results_df.set_index("Metode")["R2"])

st.subheader("Prediksi Penjualan")
input_df = build_prediction_input(df, selected_features)

if st.button("Prediksi"):
    prepared_input, _ = prepare_data(
        pd.concat([input_df.assign(**{target_col: 0}), df[selected_features + [target_col]].head(1)], ignore_index=True).head(1),
        target_col,
        selected_features,
    )
    prediction = best_model.predict(prepared_input)[0]
    st.metric("Hasil Prediksi", f"{prediction:,.0f}")
