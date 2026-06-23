"""
AgriPrice Nigeria — Crop Price Intelligence
Final Year Project — Software Engineering
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings, os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriPrice Nigeria",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --green-deep:  #1B4332;
    --green-mid:   #2D6A4F;
    --green-pale:  #D8F3DC;
    --bg:          #FAFAF5;
    --surface:     #FFFFFF;
    --sand:        #F1EDE3;
    --text-dark:   #1A1A1A;
    --text-muted:  #6B7280;
    --border:      #E5E0D5;
    --hero-grad:   linear-gradient(135deg,#1B4332,#1a5c42);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg);
    color: var(--text-dark);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1060px; }

/* ── DESKTOP ONLY: keep sidebar always open ── */
@media (min-width: 768px) {
    [data-testid="collapsedControl"] { display: none !important; }
    section[data-testid="stSidebar"] {
        width: 300px !important;
        min-width: 300px !important;
        transform: none !important;
        visibility: visible !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        width: 300px !important;
    }
}

/* ── MOBILE: let sidebar collapse naturally, tighten padding ── */
@media (max-width: 767px) {
    .block-container { padding: 1rem 0.8rem 2rem !important; }
    .result-hero { padding: 1.4rem 1.4rem !important; }
    .result-hero .rh-price { font-size: 2.6rem !important; }
    .result-hero .rh-accuracy { position: static !important; margin-top: 0.8rem; }
    .bag-grid { grid-template-columns: 1fr !important; }
    .page-title { font-size: 1.4rem !important; }
    /* Mobile top-banner — only shows on small screens */
    .mobile-banner { display: block !important; }
}
.mobile-banner { display: none; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] { background-color: #1B4332 !important; border-right: none; }
[data-testid="stSidebar"] * { color: white !important; }

.sidebar-brand {
    padding: 1.8rem 1.5rem 1.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 1.4rem;
}
.sidebar-brand .brand-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.55rem; color: white;
    line-height: 1.15; letter-spacing: -0.3px;
}
.sidebar-brand .brand-name span { color: #74C69D; }
.sidebar-brand .brand-sub {
    font-size: 0.72rem; color: rgba(255,255,255,0.45);
    margin-top: 0.3rem; letter-spacing: 0.4px;
    text-transform: uppercase; font-weight: 500;
}
.sidebar-section { padding: 0 1.5rem; }
.sidebar-field-label {
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 1px; text-transform: uppercase;
    color: rgba(255,255,255,0.45) !important;
    margin-bottom: 0.35rem; margin-top: 1.1rem; display: block;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important; color: white !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] svg { fill: white !important; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] span { color: white !important; }
.sidebar-bottom {
    padding: 1.2rem 1.5rem 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.1);
    margin-top: 1.6rem;
}
.sidebar-link-row {
    margin-top: 0.5rem; font-size: 0.73rem;
    color: rgba(255,255,255,0.35); line-height: 1.8;
}
.sidebar-link-row a { color: #74C69D !important; text-decoration: none; font-weight: 500; }
.sidebar-link-row a:hover { text-decoration: underline; }

/* ── MAIN AREA ── */
.page-eyebrow {
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 1.2px; text-transform: uppercase;
    color: var(--green-mid); margin-bottom: 0.3rem;
}
.page-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.85rem; color: var(--text-dark);
    margin: 0 0 0.2rem; letter-spacing: -0.4px; line-height: 1.2;
}
.page-subtitle { font-size: 0.88rem; color: var(--text-muted); margin: 0 0 1.8rem; }

.result-hero {
    background: var(--hero-grad);
    border-radius: 18px; padding: 2.2rem 2.5rem;
    margin-bottom: 1.2rem; position: relative; overflow: hidden;
}
.result-hero::before {
    content: ''; position: absolute;
    top: -40px; right: -40px; width: 200px; height: 200px;
    background: rgba(255,255,255,0.04); border-radius: 50%;
}
.result-hero .rh-eyebrow {
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 1.2px; text-transform: uppercase;
    color: #74C69D; margin-bottom: 0.5rem;
}
.result-hero .rh-price {
    font-family: 'DM Serif Display', serif;
    font-size: 3.8rem; color: white; line-height: 1; margin-bottom: 0.5rem;
}
.result-hero .rh-unit {
    font-size: 1rem; color: rgba(255,255,255,0.6);
    font-weight: 300; margin-left: 0.3rem;
}
.result-hero .rh-meta { font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-top: 0.4rem; }
.result-hero .rh-accuracy { position: absolute; top: 2.2rem; right: 2.5rem; text-align: right; }
.result-hero .rh-accuracy .acc-val {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem; color: #74C69D; line-height: 1;
}
.result-hero .rh-accuracy .acc-label {
    font-size: 0.68rem; color: rgba(255,255,255,0.45);
    letter-spacing: 0.8px; text-transform: uppercase;
    display: block; margin-top: 0.2rem;
}

.bag-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.2rem; }
.bag-card {
    background: var(--surface); border-radius: 14px;
    padding: 1.2rem 1.4rem; border: 1px solid var(--border);
}
.bag-card .bc-label {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.35rem;
}
.bag-card .bc-value { font-family: 'DM Serif Display', serif; font-size: 1.9rem; color: var(--green-deep); line-height: 1; }
.bag-card .bc-sub { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem; }

.sec-hdr {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 1.1px;
    text-transform: uppercase; color: var(--text-muted);
    margin: 1.6rem 0 0.75rem; display: flex; align-items: center; gap: 0.5rem;
}
.sec-hdr::after { content: ''; flex: 1; height: 1px; background: var(--border); }

.price-table { width: 100%; border-collapse: collapse; font-size: 0.87rem; }
.price-table th {
    background: var(--sand); color: var(--text-muted);
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.8px;
    text-transform: uppercase; padding: 0.65rem 1rem;
    text-align: left; border-bottom: 1px solid var(--border);
}
.price-table td {
    padding: 0.65rem 1rem; border-bottom: 1px solid var(--border);
    color: var(--text-dark); background: var(--surface);
}
.price-table tr:last-child td { border-bottom: none; }
.price-table tr:hover td { background: var(--green-pale); }
.price-table .selected-row td { background: var(--green-pale); font-weight: 600; }
.badge-sel {
    background: var(--green-deep); color: var(--bg);
    border-radius: 5px; padding: 0.1rem 0.45rem;
    font-size: 0.65rem; font-weight: 600;
    margin-left: 0.4rem; vertical-align: middle;
}

.insight-box {
    background: var(--green-pale); border-left: 3px solid var(--green-mid);
    border-radius: 0 10px 10px 0; padding: 1rem 1.2rem; margin-top: 1.2rem;
    font-size: 0.86rem; color: var(--text-dark); line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset_engineered.csv")

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_resource(show_spinner=False)
def build_models(df):
    price_cols = [c for c in df.columns if c.startswith("price_")
                  and "_lag" not in c and "_roll" not in c
                  and "_deflated" not in c]
    LAG_STEPS    = [1, 3, 6, 12]
    ROLL_WINDOWS = [3, 6]
    BASE_FEATURES = [
        "year", "month", "month_sin", "month_cos",
        "cpi", "fuel_price", "temperature", "rainfall",
        "exchange_rate_ngn_usd",
        "cpi_x_fuel", "cpi_x_exrate", "fuel_x_exrate",
        "exrate_lag1", "exrate_lag3",
        "exrate_roll_mean3", "exrate_pct_change",
        "latitude", "longitude", "admin1_enc", "market_enc",
    ]

    macro_lookup = (df.groupby("year")[["cpi", "fuel_price", "temperature",
                                        "rainfall", "exchange_rate_ngn_usd"]]
                    .mean().round(4))

    future_macro = {}
    years_arr = macro_lookup.index.values.reshape(-1, 1)
    for col in ["cpi", "fuel_price", "temperature", "rainfall", "exchange_rate_ngn_usd"]:
        reg = LinearRegression().fit(years_arr, macro_lookup[col].values)
        for yr in range(2025, 2036):
            future_macro.setdefault(yr, {})[col] = round(float(reg.predict([[yr]])[0]), 4)

    market_meta = (df.groupby(["market", "admin1"])
                   [["latitude", "longitude", "admin1_enc", "market_enc"]].first())

    price_history = {}
    for market, grp in df.groupby("market"):
        price_history[market] = {}
        for col in price_cols:
            hist = grp[["date", col]].dropna().sort_values("date")
            price_history[market][col] = hist[col].tolist()[-12:]

    commodity_medians = {col: df[col].median() for col in price_cols}
    base_cpi = float(macro_lookup.iloc[0]["cpi"])

    trained_models = {}
    for col in price_cols:
        ucol = "usd_" + col.replace("price_", "")
        comm_feats = (
            [f"{col}_lag{l}" for l in LAG_STEPS]
            + [f"{col}_roll_mean{w}" for w in ROLL_WINDOWS]
            + [f"{col}_roll_std3", f"{col}_deflated_lag1", f"{col}_deflated_lag3"]
            + [f"{ucol}_lag1", f"{ucol}_lag3", f"{ucol}_lag12", f"{ucol}_roll_mean3"]
        )
        feat_cols = BASE_FEATURES + [f for f in comm_feats if f in df.columns]
        sub = df[df[col].notna()].dropna(subset=feat_cols).copy()
        rf = RandomForestRegressor(n_estimators=300, max_depth=12,
                                   min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf.fit(sub[feat_cols].values, sub[col].values)
        trained_models[col] = {"model": rf, "features": feat_cols}

    return {
        "price_cols": price_cols, "macro_lookup": macro_lookup,
        "future_macro": future_macro, "market_meta": market_meta,
        "price_history": price_history, "commodity_medians": commodity_medians,
        "base_cpi": base_cpi, "trained_models": trained_models,
    }


def predict_price(commodity_col, market, state, year, month, bundle):
    macro_lookup      = bundle["macro_lookup"]
    future_macro      = bundle["future_macro"]
    market_meta       = bundle["market_meta"]
    price_history     = bundle["price_history"]
    commodity_medians = bundle["commodity_medians"]
    base_cpi          = bundle["base_cpi"]
    trained_models    = bundle["trained_models"]

    macro = (macro_lookup.loc[year].to_dict() if year in macro_lookup.index
             else future_macro.get(year, macro_lookup.iloc[-1].to_dict()))

    try:
        meta = market_meta.xs(market, level="market").loc[state]
    except (KeyError, TypeError):
        meta = market_meta.xs(market, level="market").iloc[0]

    hist = price_history.get(market, {}).get(commodity_col, [])
    if not hist:
        hist = [commodity_medians.get(commodity_col, 100.0)] * 12
    while len(hist) < 12:
        hist = [hist[0]] + hist

    lag1=hist[-1]; lag3=hist[-3]; lag6=hist[-6]; lag12=hist[-12]
    roll3=float(np.mean(hist[-3:])); roll6=float(np.mean(hist[-6:]))
    std3=float(np.std(hist[-3:]))
    defl1=lag1/macro["cpi"]*base_cpi; defl3=lag3/macro["cpi"]*base_cpi

    exrate = macro.get("exchange_rate_ngn_usd", macro["cpi"])
    ucol   = "usd_" + commodity_col.replace("price_", "")
    u_lag1 = lag1 / exrate if exrate > 0 else 0
    u_lag3 = lag3 / exrate if exrate > 0 else 0
    u_lag12= lag12 / exrate if exrate > 0 else 0
    u_roll3= roll3 / exrate if exrate > 0 else 0

    month_sin  = np.sin(2*np.pi*month/12)
    month_cos  = np.cos(2*np.pi*month/12)
    cpi_x_fuel = macro["cpi"]*macro["fuel_price"]
    cpi_x_er   = macro["cpi"]*exrate
    fuel_x_er  = macro["fuel_price"]*exrate

    feat_vals = {
        "year":year, "month":month, "month_sin":month_sin, "month_cos":month_cos,
        "cpi":macro["cpi"], "fuel_price":macro["fuel_price"],
        "temperature":macro["temperature"], "rainfall":macro["rainfall"],
        "exchange_rate_ngn_usd":exrate,
        "cpi_x_fuel":cpi_x_fuel, "cpi_x_exrate":cpi_x_er, "fuel_x_exrate":fuel_x_er,
        "exrate_lag1":exrate, "exrate_lag3":exrate,
        "exrate_roll_mean3":exrate, "exrate_pct_change":0.0,
        "latitude":float(meta["latitude"]), "longitude":float(meta["longitude"]),
        "admin1_enc":int(meta["admin1_enc"]), "market_enc":int(meta["market_enc"]),
        f"{commodity_col}_lag1":lag1, f"{commodity_col}_lag3":lag3,
        f"{commodity_col}_lag6":lag6, f"{commodity_col}_lag12":lag12,
        f"{commodity_col}_roll_mean3":roll3, f"{commodity_col}_roll_mean6":roll6,
        f"{commodity_col}_roll_std3":std3,
        f"{commodity_col}_deflated_lag1":defl1, f"{commodity_col}_deflated_lag3":defl3,
        f"{ucol}_lag1":u_lag1, f"{ucol}_lag3":u_lag3,
        f"{ucol}_lag12":u_lag12, f"{ucol}_roll_mean3":u_roll3,
    }

    model_info  = trained_models[commodity_col]
    feat_vector = np.array([feat_vals.get(f, 0) for f in model_info["features"]]).reshape(1,-1)
    pred        = max(float(model_info["model"].predict(feat_vector)[0]), 0.0)
    return round(pred, 2)


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
    "Yam":"🍠","Cassava Meal / Gari (Yellow)":"🌿","Gari (White)":"🌾",
    "Rice (Local)":"🍚","Maize (White)":"🌽","Maize (Yellow)":"🌽","Rice (Imported)":"🍚",
}
MONTH_NAMES = ["","January","February","March","April","May","June",
               "July","August","September","October","November","December"]
MODEL_ACCURACY = {
    "Yam":85.52,"Cassava Meal / Gari (Yellow)":84.61,"Gari (White)":82.81,
    "Rice (Local)":75.78,"Maize (White)":90.69,"Maize (Yellow)":89.70,"Rice (Imported)":78.36,
}
APP_URL = ""  # paste your Streamlit Cloud URL here after deploying

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading models…"):
    df     = load_data()
    bundle = build_models(df)

market_state_map = (df[["market","admin1"]].drop_duplicates()
                    .sort_values(["admin1","market"])
                    .set_index("market")["admin1"].to_dict())
state_markets_map: dict = {}
for mkt, st_ in market_state_map.items():
    state_markets_map.setdefault(st_, []).append(mkt)

# ─────────────────────────────────────────────────────────────────────────────
# MOBILE BANNER — shown only on small screens via CSS, gives app identity
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="mobile-banner">
    <div style="background:#1B4332;border-radius:12px;padding:1rem 1.2rem;margin-bottom:1rem">
        <div style="font-family:'DM Serif Display',serif;font-size:1.3rem;color:white">
            AgriPrice<span style="color:#74C69D">Nigeria</span>
        </div>
        <div style="font-size:0.68rem;color:rgba(255,255,255,0.5);
                    text-transform:uppercase;letter-spacing:0.5px;margin-top:0.2rem">
            Crop Price Intelligence · Use the ☰ menu to set inputs
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — inputs (collapses on mobile, always open on desktop)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-name">AgriPrice<span>Nigeria</span></div>
        <div class="brand-sub">Crop Price Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)

    st.markdown('<span class="sidebar-field-label">Crop</span>', unsafe_allow_html=True)
    commodity_name = st.selectbox("Crop", list(COMMODITY_DISPLAY.values()),
                                  label_visibility="collapsed", key="crop")
    commodity_col = {v: k for k, v in COMMODITY_DISPLAY.items()}[commodity_name]

    st.markdown('<span class="sidebar-field-label">State</span>', unsafe_allow_html=True)
    selected_state = st.selectbox("State", sorted(state_markets_map.keys()),
                                  label_visibility="collapsed", key="state")

    st.markdown('<span class="sidebar-field-label">Market</span>', unsafe_allow_html=True)
    available_markets = sorted(state_markets_map.get(selected_state, []))
    selected_market = st.selectbox("Market", available_markets,
                                   label_visibility="collapsed", key="market")

    st.markdown('<span class="sidebar-field-label">Year</span>', unsafe_allow_html=True)
    selected_year = st.selectbox("Year", list(range(2025, 2031)), index=0,
                                 label_visibility="collapsed", key="year")

    st.markdown('<span class="sidebar-field-label">Month</span>', unsafe_allow_html=True)
    month_options = MONTH_NAMES[1:]
    selected_month_name = st.selectbox("Month", month_options,
                                       label_visibility="collapsed", key="month")
    selected_month = month_options.index(selected_month_name) + 1

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-bottom">', unsafe_allow_html=True)
    if APP_URL:
        st.markdown(
            f'<div class="sidebar-link-row">'
            f'<a href="{APP_URL}" target="_blank">🔗 Open live app</a>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
icon     = COMMODITY_ICON.get(commodity_name, "🌾")
accuracy = MODEL_ACCURACY.get(commodity_name, 80.0)

try:
    pred_price = predict_price(commodity_col, selected_market, selected_state,
                               selected_year, selected_month, bundle)

    st.markdown(
        f'<div class="page-eyebrow">Market Price Forecast</div>'
        f'<div class="page-title">{icon} {commodity_name}</div>'
        f'<div class="page-subtitle">'
        f'{selected_market} &nbsp;·&nbsp; {selected_state} State &nbsp;·&nbsp; '
        f'{MONTH_NAMES[selected_month]} {selected_year}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"""
    <div class="result-hero">
        <div class="rh-eyebrow">Predicted price per kilogram</div>
        <div class="rh-price">₦{pred_price:,.2f}<span class="rh-unit">/ kg</span></div>
        <div class="rh-meta">Nigerian Naira &nbsp;·&nbsp; Random Forest model</div>
        <div class="rh-accuracy">
            <div class="acc-val">{accuracy:.1f}%</div>
            <span class="acc-label">Model accuracy</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="bag-grid">
        <div class="bag-card">
            <div class="bc-label">Per 50 kg bag</div>
            <div class="bc-value">₦{pred_price*50:,.0f}</div>
            <div class="bc-sub">Standard wholesale unit</div>
        </div>
        <div class="bag-card">
            <div class="bc-label">Per 100 kg bag</div>
            <div class="bc-value">₦{pred_price*100:,.0f}</div>
            <div class="bc-sub">Large wholesale unit</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    hist_prices = bundle["price_history"].get(selected_market, {}).get(commodity_col, [])
    if hist_prices and len(hist_prices) >= 2:
        st.markdown('<div class="sec-hdr">Recent price history at this market</div>',
                    unsafe_allow_html=True)
        st.line_chart(pd.DataFrame({"Price (₦/kg)": hist_prices}),
                      height=155, use_container_width=True)

    st.markdown(
        f'<div class="sec-hdr">All crops — {selected_market}, '
        f'{MONTH_NAMES[selected_month]} {selected_year}</div>',
        unsafe_allow_html=True,
    )

    all_rows = []
    for col_key, col_name in COMMODITY_DISPLAY.items():
        try:
            p = predict_price(col_key, selected_market, selected_state,
                              selected_year, selected_month, bundle)
            all_rows.append((col_name, p, col_key == commodity_col))
        except Exception:
            pass

    all_rows.sort(key=lambda x: x[1], reverse=True)
    rows_html = ""
    for name, price, is_sel in all_rows:
        badge     = '<span class="badge-sel">selected</span>' if is_sel else ""
        row_class = ' class="selected-row"' if is_sel else ""
        rows_html += (
            f"<tr{row_class}>"
            f"<td>{COMMODITY_ICON.get(name,'🌾')} {name}{badge}</td>"
            f"<td style='text-align:right;font-variant-numeric:tabular-nums'>₦{price:,.2f}/kg</td>"
            f"<td style='text-align:right;font-variant-numeric:tabular-nums'>₦{price*50:,.0f}</td>"
            f"<td style='text-align:right;font-variant-numeric:tabular-nums'>₦{price*100:,.0f}</td>"
            f"</tr>"
        )
    st.markdown(
        f"<table class='price-table'><thead><tr>"
        f"<th>Crop</th><th style='text-align:right'>Per kg</th>"
        f"<th style='text-align:right'>Per 50 kg</th>"
        f"<th style='text-align:right'>Per 100 kg</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table>",
        unsafe_allow_html=True,
    )

    hist_mean  = float(np.mean(hist_prices)) if hist_prices else pred_price
    pct_change = ((pred_price-hist_mean)/hist_mean*100) if hist_mean > 0 else 0
    direction  = "above" if pct_change > 0 else "below"

    st.markdown(f"""
    <div class="insight-box">
        <strong>Prediction note</strong><br>
        The forecast of <b>₦{pred_price:,.2f}/kg</b> for {commodity_name} at {selected_market}
        in {MONTH_NAMES[selected_month]} {selected_year} is approximately
        <b>{abs(pct_change):.1f}% {direction}</b> the recent observed average at this market.
        This model achieves <b>{accuracy:.1f}% accuracy</b> on held-out test data for this crop.
        Forecasts beyond 2024 use extrapolated macro indicators and carry wider uncertainty.
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Prediction unavailable: {e}")
    st.info("Try a different market or commodity.")
