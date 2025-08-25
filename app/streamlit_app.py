# app/streamlit_app.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["ABSL_LOGGING_STDERR_THRESHOLD"] = "3"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import yfinance as yf

try:
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

#  page config
st.set_page_config(
    page_title="Financial Time-Series Forecasting (LSTM)",
    page_icon="📈",
    layout="wide",
)

ARTIFACT_DIR   = "models"
DEFAULT_COLS_5 = ["Open", "High", "Low", "Close", "Volume"]
ALT_COLS_4     = ["High", "Low", "Open", "Volume"]

# small metrics helpers 
def rmse(y, yhat) -> float:
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.sqrt(((y - yhat) ** 2).mean()))

def smape(y, yhat) -> float:
    y, yhat = np.asarray(y, dtype=np.float64), np.asarray(yhat, dtype=np.float64)
    return float(100.0 * np.mean(np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + 1e-8)))

# data fetch (cached) 
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_prices(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for '{ticker}' with period='{period}'.")
    df = df.rename(columns=str.capitalize)
    keep = [c for c in DEFAULT_COLS_5 if c in df.columns]
    return df[keep].dropna()

# model/scalers loading (cached) 
@st.cache_resource(show_spinner=False)
def _load_model_tfkeras(path: str):
    from tensorflow.keras.models import load_model
    return load_model(path, compile=False)

@st.cache_resource(show_spinner=False)
def _load_model_keras3(path: str):
    import keras
    return keras.models.load_model(path, safe_mode=False, compile=False)

def load_artifacts():
    model_path = os.path.join(ARTIFACT_DIR, "model.h5")
    sx_path    = os.path.join(ARTIFACT_DIR, "scaler_x.pkl")
    sy_path    = os.path.join(ARTIFACT_DIR, "scaler_y.pkl")

    if not (os.path.exists(model_path) and os.path.exists(sx_path) and os.path.exists(sy_path)):
        raise FileNotFoundError(
            "Missing artifacts. Expected in 'models/': model.h5, scaler_x.pkl, scaler_y.pkl"
        )

    # Keras 3 is more tolerant to mixed configs; try first
    try:
        model = _load_model_keras3(model_path)
        loader = "keras.models.load_model (Keras 3)"
    except Exception:
        model = _load_model_tfkeras(model_path)
        loader = "tensorflow.keras.models.load_model"

    scaler_x = joblib.load(sx_path)
    scaler_y = joblib.load(sy_path)
    return model, scaler_x, scaler_y, loader

# scaler-driven input builder (robust to 4-vs-5 features)
def build_inputs_to_match(model, raw_df: pd.DataFrame, scaler_x, window_from_ui: int):
    """
    Build model inputs that match the *trained* artifacts.

    - Feature count is taken from scaler_x (n_features_in_) -> avoids transform() errors.
    - Timesteps are taken from model.input_shape.
    - Returns: X (np.ndarray), cols used (list[str]), timesteps (int).
    """
    # 1) timesteps & model feature count from model.input_shape
    try:
        in_shape = model.input_shape 
    except AttributeError:
        in_shape = model.inputs[0].shape.as_list()

    if len(in_shape) != 3:
        raise ValueError(f"Unexpected input shape {in_shape}. Expect (None, timesteps, features).")

    timesteps     = int(in_shape[1])
    n_feats_model = int(in_shape[2])

    # 2) feature count from scaler (what transform() enforces)
    n_feats_scaler = int(getattr(scaler_x, "n_features_in_", n_feats_model))

    # 3) choose columns to match scaler
    if n_feats_scaler == 5 and all(c in raw_df.columns for c in DEFAULT_COLS_5):
        cols = DEFAULT_COLS_5
    elif n_feats_scaler == 4 and all(c in raw_df.columns for c in ALT_COLS_4):
        cols = ALT_COLS_4
    else:
        numeric_cols = [c for c in raw_df.columns if np.issubdtype(raw_df[c].dtype, np.number)]
        cols = numeric_cols[:n_feats_scaler]
        if len(cols) != n_feats_scaler:
            raise ValueError(
                f"Could not find {n_feats_scaler} numeric columns. Available: {list(raw_df.columns)}"
            )

    # 4) warn if model/scaler disagree (we proceed with scaler’s count)
    if n_feats_model != n_feats_scaler:
        st.warning(
            f"Feature count mismatch: model expects {n_feats_model}, scaler expects {n_feats_scaler}. "
            f"Proceeding with scaler’s order: {cols}"
        )

    # 5) scale then create (N, 1, F) or (N-T, T, F)
    try:
        scaled = raw_df[cols].copy()
        scaled[cols] = scaler_x.transform(scaled[cols])
    except Exception as e:
        raise RuntimeError(
            f"Feature scaler failed. Ensure scaler_x was fit on these columns in this order: {cols}. "
            f"Original error: {e}"
        )

    vals = scaled.values
    if timesteps <= 1:
        X = vals.reshape(-1, 1, n_feats_scaler)          # (N, 1, F)
    else:
        X = np.array([vals[i - timesteps:i] for i in range(timesteps, len(vals))])  # (N-T, T, F)
        if window_from_ui != timesteps:
            st.info(f"Model expects window={timesteps}; UI window={window_from_ui}. Using model's window.")

    return X, cols, timesteps

# UI 
with st.sidebar:
    st.header("Settings")
    ticker    = st.text_input("Ticker", "NVDA").strip().upper()
    period    = st.selectbox("History period", ["2y", "5y", "10y"], index=1)
    ui_window = st.slider("Lookback window (days)", 20, 120, 80, step=5)

st.title("📈 Financial Time-Series Forecasting (LSTM)")
st.caption("Educational demo — not financial advice. Model served with pre-trained artifacts and time-based windowing.")

# Orchestration with visible progress & errors 
with st.status("Initializing…", expanded=True) as s:
    # Load artifacts
    try:
        s.write("• Loading model & scalers…")
        model, scaler_x, scaler_y, loader = load_artifacts()
        s.write(f"  ↳ Loaded via **{loader}**. input_shape: `{getattr(model, 'input_shape', None)}`")
    except Exception as e:
        st.error("Failed to load model/scalers.")
        st.exception(e)
        st.stop()

    # Fetch data
    try:
        s.write(f"• Fetching price data for **{ticker}** ({period})…")
        raw_df = fetch_prices(ticker, period)
        s.write(f"  ↳ {len(raw_df):,} rows from {raw_df.index.min().date()} to {raw_df.index.max().date()}")
    except Exception as e:
        st.error("Failed to fetch price data.")
        st.exception(e)
        st.stop()

    if len(raw_df) < 30:
        st.warning("Not enough rows returned from yfinance. Try a longer period or a different ticker.")
        st.stop()

    # Build inputs
    try:
        s.write("• Preparing model inputs to match the trained shape…")
        X, used_cols, timesteps = build_inputs_to_match(model, raw_df, scaler_x, ui_window)
        s.write(f"  ↳ Using columns: {used_cols} | timesteps={timesteps} | X.shape={X.shape}")
    except Exception as e:
        st.error("Failed to build model inputs.")
        st.exception(e)
        st.stop()

    # Predict
    try:
        s.write("• Running inference…")
        yhat_scaled = model.predict(X, verbose=0)
        yhat = scaler_y.inverse_transform(yhat_scaled).reshape(-1)  # ensure 1-D
    except Exception as e:
        st.error("Model prediction failed (shape/version mismatch).")
        st.exception(e)
        st.stop()

    # Align y_true to predictions
    # If you trained SAME-day target:
    y_true = raw_df["Close"].iloc[-len(yhat):].values
    # If you trained NEXT-day target, use this instead:
    # y_true = raw_df["Close"].shift(-1).dropna().iloc[-len(yhat):].values

    # Ensure 1-D & matched lengths
    y_true = np.asarray(y_true).reshape(-1)
    n = min(len(y_true), len(yhat))
    y_true = y_true[-n:]
    yhat   = yhat[-n:]
    plot_index = raw_df.index[-n:]

    # Baseline (naive last) for comparison (align lengths)
    if n > 1:
        naive_pred = y_true[:-1]
        naive_true = y_true[1:]
        base_rmse  = rmse(naive_true, naive_pred)
        base_smape = smape(naive_true, naive_pred)
    else:
        base_rmse = base_smape = float("nan")

    lstm_rmse  = rmse(y_true, yhat)
    lstm_smape = smape(y_true, yhat)

    s.write("• Done.")
    s.update(label="Ready", state="complete")

#  Results 
c1, c2, c3, c4 = st.columns(4)
c1.metric("RMSE (LSTM)", f"{lstm_rmse:.2f}")
c2.metric("sMAPE (LSTM)", f"{lstm_smape:.2f}%")
c3.metric("RMSE (Naive last)", f"{base_rmse:.2f}")
c4.metric("sMAPE (Naive last)", f"{base_smape:.2f}%")

st.subheader(f"{ticker} — Actual vs Predicted Close")
plot_df = pd.DataFrame({"Actual": y_true, "Predicted": yhat}, index=plot_index)
st.line_chart(plot_df)

st.subheader("Latest forecast")
last_close = float(plot_df["Actual"].iloc[-1])
last_pred  = float(plot_df["Predicted"].iloc[-1])
delta = last_pred - last_close
cc1, cc2, cc3 = st.columns(3)
cc1.metric("Last Close", f"{last_close:,.2f} USD")
cc2.metric("Next Pred",  f"{last_pred:,.2f} USD", delta=f"{delta:+.2f}")
cc3.metric("Model window", f"{timesteps}")

with st.expander("Debug details"):
    st.write("Model input shape:", getattr(model, "input_shape", None))
    st.write("Data columns:", list(raw_df.columns))
    st.write("Using columns:", used_cols)
    st.write("X.shape:", X.shape, "pred len:", len(yhat))
    st.dataframe(raw_df.tail())
