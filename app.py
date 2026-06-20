"""
Agricultural Produce Price Predictor — Nigeria
Final Year Project, Software Engineering
Streamlit deployment app

Run with:
    pip install streamlit scikit-learn pandas numpy
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nigeria Crop Price Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — earthy green + warm cream palette, clean editorial feel
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@300;400;500;600&display=swap');

/* Root palette */
:root {
    --green-deep:   #1B4332;
    --green-mid:    #2D6A4F;
    --green-light:  #40916C;
    --green-pale:   #D8F3DC;
    --cream:        #FAFAF5;
    --sand:         #F1EDE3;
    --text-dark:    #1A1A1A;
    --text-muted:   #6B7280;
    --gold:         #D4A017;
    --border:       #E5E0D5;
}

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--cream);
    color: var(--text-dark);
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1100px; }

/* ── HEADER ── */
.app-header {
    background: linear-gradient(135deg, var(--green-deep) 0%, var(--green-mid) 100%);
    border-radius: 16px;
    padding: 2.5rem 2.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.app-header::after {
    content: "🌾";
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.15;
}
.app-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: white;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
    line-height: 1.2;
}
.app-header p {
    color: rgba(255,255,255,0.75);
    font-size: 0.95rem;
    margin: 0;
    font-weight: 300;
}
.app-header .badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.9);
    margin-top: 0.75rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-weight: 500;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background-color: var(--sand);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .sidebar-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: var(--green-deep);
    margin-bottom: 0.25rem;
}
[data-testid="stSidebar"] .sidebar-sub {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 1.5rem;
    line-height: 1.5;
}

/* ── CARDS ── */
.result-card {
    background: white;
    border-radius: 16px;
    padding: 2rem 2.2rem;
    border: 1px solid var(--border);
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 1.25rem;
}
.result-card.primary {
    background: linear-gradient(135deg, var(--green-deep) 0%, var(--green-light) 100%);
    border: none;
    color: white;
}
.result-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 0.4rem;
}
.result-card .value {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.result-card .sub {
    font-size: 0.85rem;
    opacity: 0.75;
}
.result-card.primary .label { color: rgba(255,255,255,0.75); }
.result-card.primary .value { color: white; }
.result-card.primary .sub   { color: rgba(255,255,255,0.7); }

/* ── METRIC GRID ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-tile {
    background: white;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    border: 1px solid var(--border);
    text-align: center;
}
.metric-tile .mt-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    margin-bottom: 0.35rem;
}
.metric-tile .mt-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: var(--green-deep);
    line-height: 1;
}
.metric-tile .mt-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
}

/* ── INSIGHT BOX ── */
.insight-box {
    background: var(--green-pale);
    border-left: 4px solid var(--green-mid);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin-top: 1.25rem;
    font-size: 0.88rem;
    color: var(--green-deep);
    line-height: 1.6;
}
.insight-box strong { font-weight: 600; }

/* ── SECTION LABEL ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.5rem 0 0.75rem;
}

/* ── TREND ROW ── */
.trend-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.trend-chip {
    background: var(--sand);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.45rem 0.85rem;
    font-size: 0.82rem;
    color: var(--text-dark);
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.trend-chip .chip-label { color: var(--text-muted); font-size: 0.72rem; }

/* ── PREDICT BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, var(--green-deep), var(--green-light)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    transition: opacity 0.2s !important;
    box-shadow: 0 4px 14px rgba(27,67,50,0.3) !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── SELECT / INPUT ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {
    border-radius: 8px !important;
    border-color: var(--border) !important;
    background: white !important;
}

/* ── DIVIDER ── */
.custom-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* ── TABLE ── */
.price-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    margin-top: 0.5rem;
}
.price-table th {
    background: var(--sand);
    color: var(--text-muted);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    padding: 0.6rem 0.9rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
}
.price-table td {
    padding: 0.6rem 0.9rem;
    border-bottom: 1px solid var(--border);
    color: var(--text-dark);
}
.price-table tr:last-child td { border-bottom: none; }
.price-table tr:hover td { background: var(--green-pale); }
.badge-best {
    background: var(--green-pale);
    color: var(--green-deep);
    border-radius: 6px;
    padding: 0.15rem 0.5rem;
    font-size: 0.72rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADING  (cached — runs once on startup)
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset_engineered.csv")

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_resource(show_spinner=False)
def build_models(df):
    price_cols = [
        c for c in df.columns
        if c.startswith("price_")
        and "_lag" not in c
        and "_roll" not in c
        and "_deflated" not in c
    ]
    LAG_STEPS    = [1, 3, 6, 12]
    ROLL_WINDOWS = [3, 6]
    BASE_FEATURES = [
        "year", "month", "month_sin", "month_cos",
        "cpi", "fuel_price", "temperature", "rainfall",
        "latitude", "longitude", "cpi_x_fuel",
        "admin1_enc", "market_enc",
    ]

    # ── Macro lookup (mean per year) ──
    macro_lookup = (
        df.groupby("year")[["cpi", "fuel_price", "temperature", "rainfall"]]
        .mean().round(4)
    )

    # ── Extrapolate macro to future years via linear trend ──
    future_macro = {}
    years_arr = macro_lookup.index.values.reshape(-1, 1)
    for col in ["cpi", "fuel_price", "temperature", "rainfall"]:
        reg = LinearRegression().fit(years_arr, macro_lookup[col].values)
        for yr in range(2025, 2036):
            future_macro.setdefault(yr, {})[col] = round(float(reg.predict([[yr]])[0]), 4)

    # ── Market metadata ──
    market_meta = (
        df.groupby(["market", "admin1"])[["latitude", "longitude", "admin1_enc", "market_enc"]]
        .first()
    )

    # ── Price history per market per commodity (last 12 obs) ──
    price_history = {}
    for market, grp in df.groupby("market"):
        price_history[market] = {}
        for col in price_cols:
            hist = grp[["date", col]].dropna().sort_values("date")
            price_history[market][col] = hist[col].tolist()[-12:]

    # ── Global commodity medians (fallback when no market history) ──
    commodity_medians = {col: df[col].median() for col in price_cols}
    base_cpi = float(macro_lookup.iloc[0]["cpi"])

    # ── Train RF per commodity on full dataset ──
    trained_models = {}
    for col in price_cols:
        comm_feats = (
            [f"{col}_lag{l}" for l in LAG_STEPS]
            + [f"{col}_roll_mean{w}" for w in ROLL_WINDOWS]
            + [f"{col}_roll_std3", f"{col}_deflated_lag1", f"{col}_deflated_lag3"]
        )
        feat_cols = BASE_FEATURES + comm_feats
        sub = df[df[col].notna()].dropna(subset=feat_cols).copy()
        rf = RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )
        rf.fit(sub[feat_cols].values, sub[col].values)
        trained_models[col] = {"model": rf, "features": feat_cols}

    return {
        "price_cols":       price_cols,
        "macro_lookup":     macro_lookup,
        "future_macro":     future_macro,
        "market_meta":      market_meta,
        "price_history":    price_history,
        "commodity_medians":commodity_medians,
        "base_cpi":         base_cpi,
        "trained_models":   trained_models,
        "BASE_FEATURES":    BASE_FEATURES,
    }


def predict_price(commodity_col, market, state, year, month, bundle):
    """
    Predict price per kg for commodity at a given market, year and month.
    Returns (predicted_price_ngn, macro_context_dict) or raises ValueError.
    """
    macro_lookup  = bundle["macro_lookup"]
    future_macro  = bundle["future_macro"]
    market_meta   = bundle["market_meta"]
    price_history = bundle["price_history"]
    commodity_medians = bundle["commodity_medians"]
    base_cpi      = bundle["base_cpi"]
    trained_models = bundle["trained_models"]

    # ── Macro ──
    if year in macro_lookup.index:
        macro = macro_lookup.loc[year].to_dict()
    elif year in future_macro:
        macro = future_macro[year]
    else:
        macro = macro_lookup.iloc[-1].to_dict()

    # ── Market metadata ──
    try:
        meta = market_meta.xs(market, level="market").loc[state]
    except (KeyError, TypeError):
        try:
            meta = market_meta.xs(market, level="market").iloc[0]
        except Exception:
            raise ValueError(f"Market '{market}' not found in dataset.")

    # ── Lag / rolling features from historical prices ──
    hist = price_history.get(market, {}).get(commodity_col, [])
    if len(hist) == 0:
        fallback = commodity_medians.get(commodity_col, 100.0)
        hist = [fallback] * 12
    while len(hist) < 12:
        hist = [hist[0]] + hist

    lag1  = hist[-1]
    lag3  = hist[-3]
    lag6  = hist[-6]
    lag12 = hist[-12]
    roll3 = float(np.mean(hist[-3:]))
    roll6 = float(np.mean(hist[-6:]))
    std3  = float(np.std(hist[-3:])) if len(hist) >= 3 else 0.0
    defl1 = lag1 / macro["cpi"] * base_cpi
    defl3 = lag3 / macro["cpi"] * base_cpi

    month_sin  = np.sin(2 * np.pi * month / 12)
    month_cos  = np.cos(2 * np.pi * month / 12)
    cpi_x_fuel = macro["cpi"] * macro["fuel_price"]

    feat_vals = {
        "year": year, "month": month,
        "month_sin": month_sin, "month_cos": month_cos,
        "cpi": macro["cpi"], "fuel_price": macro["fuel_price"],
        "temperature": macro["temperature"], "rainfall": macro["rainfall"],
        "latitude": float(meta["latitude"]), "longitude": float(meta["longitude"]),
        "cpi_x_fuel": cpi_x_fuel,
        "admin1_enc": int(meta["admin1_enc"]), "market_enc": int(meta["market_enc"]),
        f"{commodity_col}_lag1":  lag1,  f"{commodity_col}_lag3":  lag3,
        f"{commodity_col}_lag6":  lag6,  f"{commodity_col}_lag12": lag12,
        f"{commodity_col}_roll_mean3": roll3, f"{commodity_col}_roll_mean6": roll6,
        f"{commodity_col}_roll_std3":  std3,
        f"{commodity_col}_deflated_lag1": defl1,
        f"{commodity_col}_deflated_lag3": defl3,
    }

    model_info   = trained_models[commodity_col]
    feat_vector  = np.array([feat_vals[f] for f in model_info["features"]]).reshape(1, -1)
    pred         = float(model_info["model"].predict(feat_vector)[0])
    pred         = max(pred, 0.0)

    macro_ctx = {
        "CPI":         round(macro["cpi"], 1),
        "Fuel (₦/L)":  round(macro["fuel_price"], 1),
        "Temp (°C)":   round(macro["temperature"], 1),
        "Rainfall":    round(macro["rainfall"], 2),
    }
    return round(pred, 2), macro_ctx


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

COMMODITY_DISPLAY = {
    "price_yam_per_kg":                 "Yam",
    "price_cassava_gari_yellow_per_kg": "Cassava Meal / Gari (Yellow)",
    "price_gari_white_per_kg":          "Gari (White)",
    "price_rice_local_per_kg":          "Rice (Local)",
    "price_maize_white_per_kg":         "Maize (White)",
    "price_maize_yellow_per_kg":        "Maize (Yellow)",
    "price_rice_imported_per_kg":       "Rice (Imported)",
}
COMMODITY_ICON = {
    "Yam":                          "🍠",
    "Cassava Meal / Gari (Yellow)": "🌿",
    "Gari (White)":                 "🌾",
    "Rice (Local)":                 "🍚",
    "Maize (White)":                "🌽",
    "Maize (Yellow)":               "🌽",
    "Rice (Imported)":              "🍚",
}
MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MODEL_ACCURACY = {
    "Yam":                          85.52,
    "Cassava Meal / Gari (Yellow)": 84.61,
    "Gari (White)":                 82.81,
    "Rice (Local)":                 76.01,
    "Maize (White)":                90.69,
    "Maize (Yellow)":               89.70,
    "Rice (Imported)":              77.79,
}


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

# ── Load & build ──
with st.spinner("Loading models — this takes about 30 seconds on first run…"):
    df = load_data()
    bundle = build_models(df)

# ── Header ──
st.markdown("""
<div class="app-header">
    <h1>Nigeria Crop Price Predictor</h1>
    <p>Predict agricultural produce prices per kilogram across Nigerian states and markets</p>
    <span class="badge">Final Year Project · Software Engineering · RF Model · 76–91% Accuracy</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar — inputs ──
with st.sidebar:
    st.markdown('<div class="sidebar-title">Prediction Inputs</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-sub">Select the crop, location, and time period to get a predicted market price.</div>',
        unsafe_allow_html=True,
    )

    # Market → state mapping
    market_state_map = (
        df[["market", "admin1"]].drop_duplicates()
        .sort_values(["admin1", "market"])
        .set_index("market")["admin1"].to_dict()
    )
    state_markets_map = {}
    for mkt, state in market_state_map.items():
        state_markets_map.setdefault(state, []).append(mkt)

    # 1. Crop
    st.markdown("**Crop**")
    commodity_name = st.selectbox(
        "Crop", options=list(COMMODITY_DISPLAY.values()), label_visibility="collapsed"
    )
    commodity_col = {v: k for k, v in COMMODITY_DISPLAY.items()}[commodity_name]

    # 2. State
    st.markdown("**State**")
    all_states = sorted(state_markets_map.keys())
    selected_state = st.selectbox("State", options=all_states, label_visibility="collapsed")

    # 3. Market (cascades from state)
    st.markdown("**Market**")
    available_markets = sorted(state_markets_map.get(selected_state, []))
    selected_market = st.selectbox("Market", options=available_markets, label_visibility="collapsed")

    # 4. Year
    st.markdown("**Year**")
    selected_year = st.selectbox(
        "Year", options=list(range(2025, 2031)), index=0, label_visibility="collapsed"
    )

    # 5. Month
    st.markdown("**Month**")
    month_options = MONTH_NAMES[1:]
    selected_month_name = st.selectbox("Month", options=month_options, label_visibility="collapsed")
    selected_month = month_options.index(selected_month_name) + 1

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Predict Price", use_container_width=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:0.75rem;color:var(--text-muted);line-height:1.6'>"
        f"Model: <b>Random Forest</b><br>"
        f"Training data: 2009–2024<br>"
        f"Sources: WFP Nigeria, CBN<br>"
        f"Unit: Nigerian Naira per kg"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

if not predict_btn:
    # Welcome state
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="result-card">
            <div class="label">How to use</div>
            <p style="margin:0.5rem 0 0;font-size:0.92rem;line-height:1.7;color:#374151">
                Select a <b>crop</b>, <b>state</b>, <b>market</b>, <b>year</b>, and <b>month</b>
                in the sidebar, then click <b>Predict Price</b>.<br><br>
                The model will return an estimated price per kilogram in Nigerian Naira (₦),
                along with the macroeconomic context used to generate the prediction.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-label">Model accuracy by crop</div>', unsafe_allow_html=True)
        acc_rows = "".join(
            f"<tr><td>{COMMODITY_ICON.get(c,'🌾')} {c}</td>"
            f"<td style='text-align:right;font-weight:600;color:var(--green-deep)'>{a:.1f}%</td></tr>"
            for c, a in MODEL_ACCURACY.items()
        )
        st.markdown(
            f"<table class='price-table'>"
            f"<thead><tr><th>Crop</th><th style='text-align:right'>Accuracy</th></tr></thead>"
            f"<tbody>{acc_rows}</tbody></table>",
            unsafe_allow_html=True,
        )

    st.markdown("""
    <div class="insight-box">
        <strong>About this predictor</strong><br>
        Trained on WFP Nigeria food price survey data (2009–2024) covering 58 markets
        across 14 states. The Random Forest model uses macroeconomic indicators (CPI,
        fuel price), climate data (temperature, rainfall), geographic features, and
        12 months of historical price lags to generate predictions.
    </div>
    """, unsafe_allow_html=True)

else:
    # ── RUN PREDICTION ──
    try:
        pred_price, macro_ctx = predict_price(
            commodity_col, selected_market, selected_state,
            selected_year, selected_month, bundle
        )

        icon     = COMMODITY_ICON.get(commodity_name, "🌾")
        accuracy = MODEL_ACCURACY.get(commodity_name, 80.0)

        # ── Primary result ──
        st.markdown(f"""
        <div class="result-card primary">
            <div class="label">Predicted Price — {commodity_name}</div>
            <div class="value">₦{pred_price:,.2f}</div>
            <div class="sub">
                {icon} per kilogram &nbsp;·&nbsp;
                {selected_market}, {selected_state} &nbsp;·&nbsp;
                {MONTH_NAMES[selected_month]} {selected_year}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Context metrics ──
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="label">Per 50 kg bag</div>
                <div class="value" style="font-size:2.2rem">₦{pred_price*50:,.0f}</div>
                <div class="sub">Standard wholesale unit</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="label">Per 100 kg bag</div>
                <div class="value" style="font-size:2.2rem">₦{pred_price*100:,.0f}</div>
                <div class="sub">Large wholesale unit</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="result-card">
                <div class="label">Model accuracy</div>
                <div class="value" style="font-size:2.2rem;color:var(--green-mid)">{accuracy:.1f}%</div>
                <div class="sub">Random Forest · test set MAPE</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Macro context ──
        st.markdown('<div class="section-label">Macroeconomic context used for this prediction</div>',
                    unsafe_allow_html=True)
        macro_html = "".join(
            f'<div class="trend-chip"><span class="chip-label">{k}</span> <b>{v}</b></div>'
            for k, v in macro_ctx.items()
        )
        st.markdown(f'<div class="trend-row">{macro_html}</div>', unsafe_allow_html=True)

        # ── Historical price trend for this market ──
        hist_prices = bundle["price_history"].get(selected_market, {}).get(commodity_col, [])
        if hist_prices and len(hist_prices) >= 2:
            st.markdown('<div class="section-label">Recent price history at this market</div>',
                        unsafe_allow_html=True)

            # Build sparkline data using Streamlit chart
            chart_data = pd.DataFrame({
                "Price (₦/kg)": hist_prices
            })
            st.line_chart(chart_data, height=160, use_container_width=True)

        # ── All-crop price table for same location ──
        st.markdown(
            f'<div class="section-label">All crop predictions — {selected_market}, {MONTH_NAMES[selected_month]} {selected_year}</div>',
            unsafe_allow_html=True,
        )

        all_rows = []
        for col_key, col_name in COMMODITY_DISPLAY.items():
            try:
                p, _ = predict_price(
                    col_key, selected_market, selected_state,
                    selected_year, selected_month, bundle
                )
                is_selected = col_key == commodity_col
                all_rows.append((col_name, p, is_selected))
            except Exception:
                pass

        all_rows.sort(key=lambda x: x[1], reverse=True)
        rows_html = ""
        for name, price, is_sel in all_rows:
            badge = ' <span class="badge-best">selected</span>' if is_sel else ""
            rows_html += (
                f"<tr{'style=background:var(--green-pale)' if is_sel else ''}>"
                f"<td>{COMMODITY_ICON.get(name,'🌾')} {name}{badge}</td>"
                f"<td style='text-align:right;font-family:monospace;font-weight:600'>"
                f"₦{price:,.2f}/kg</td>"
                f"<td style='text-align:right'>₦{price*50:,.0f}</td>"
                f"<td style='text-align:right'>₦{price*100:,.0f}</td>"
                f"</tr>"
            )

        st.markdown(
            f"<table class='price-table'>"
            f"<thead><tr>"
            f"<th>Crop</th>"
            f"<th style='text-align:right'>Per kg</th>"
            f"<th style='text-align:right'>Per 50 kg</th>"
            f"<th style='text-align:right'>Per 100 kg</th>"
            f"</tr></thead>"
            f"<tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )

        # ── Insight ──
        hist_mean = float(np.mean(hist_prices)) if hist_prices else pred_price
        pct_change = ((pred_price - hist_mean) / hist_mean * 100) if hist_mean > 0 else 0
        direction  = "higher than" if pct_change > 0 else "lower than"
        abs_pct    = abs(pct_change)

        st.markdown(f"""
        <div class="insight-box">
            <strong>Prediction note</strong><br>
            The predicted price of <b>₦{pred_price:,.2f}/kg</b> for {commodity_name} at {selected_market}
            in {MONTH_NAMES[selected_month]} {selected_year} is approximately
            <b>{abs_pct:.1f}% {direction}</b> the recent average price recorded at this market.
            The model achieves <b>{accuracy:.1f}% accuracy</b> on held-out test data for this commodity.
            Predictions for future years use extrapolated macroeconomic indicators and may carry
            wider uncertainty than predictions within the 2009–2024 historical window.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info("Try selecting a different market or commodity. Not all markets have price history for all crops.")
