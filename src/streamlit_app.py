import sys
import os
import json
sys.path.append(os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from feature_engineering import engineer_features

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ — Customer Retention Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}


/* Main background */
.stApp {
    background: #0a0a0f;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f18 !important;
    border-right: 1px solid #1e1e2e;
}

section[data-testid="stSidebar"] {
    min-width: 320px !important;
    max-width: 320px !important;
}

[data-testid="stSidebar"] * {
    color: #a0a0b8 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0f0f18;
    border-bottom: 1px solid #1e1e2e;
    padding: 0 8px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #5a5a7a !important;
    font-size: 13px;
    font-weight: 500;
    padding: 12px 20px;
    border: none;
    border-bottom: 2px solid transparent;
    letter-spacing: 0.02em;
}

.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #ffffff !important;
    border-bottom: 2px solid #7c6af7 !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #0f0f18;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 20px 24px;
}

[data-testid="metric-container"] label {
    color: #5a5a7a !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #ffffff !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c6af7, #a78bfa) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 14px 28px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(124, 106, 247, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(124, 106, 247, 0.45) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #0f0f18 !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 8px !important;
    color: #e0e0f0 !important;
}

/* Number inputs */
[data-testid="stNumberInput"] > div > div > input {
    background: #0f0f18 !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 8px !important;
    color: #e0e0f0 !important;
}

/* Sliders */
[data-testid="stSlider"] > div > div > div > div {
    background: #7c6af7 !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #0f0f18 !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 10px !important;
}

/* Info/success/error boxes */
[data-testid="stInfo"] {
    background: rgba(124, 106, 247, 0.08) !important;
    border: 1px solid rgba(124, 106, 247, 0.25) !important;
    border-radius: 10px !important;
    color: #c4bbff !important;
}

[data-testid="stSuccess"] {
    background: rgba(62, 207, 142, 0.08) !important;
    border: 1px solid rgba(62, 207, 142, 0.25) !important;
    border-radius: 10px !important;
    color: #a0ecd0 !important;
}

[data-testid="stError"] {
    background: rgba(240, 107, 92, 0.08) !important;
    border: 1px solid rgba(240, 107, 92, 0.25) !important;
    border-radius: 10px !important;
}

/* Radio */
[data-testid="stRadio"] label {
    color: #a0a0b8 !important;
}

/* Divider */
hr {
    border-color: #1e1e2e !important;
}

/* Text colors */
h1, h2, h3 {
    color: #ffffff !important;
}

p, label {
    color: #a0a0b8;
}

/* Custom card class */
.stat-card {
    background: #0f0f18;
    border: 1px solid #1e1e2e;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 16px;
}

.risk-badge-high {
    background: rgba(240,107,92,0.15);
    border: 1px solid rgba(240,107,92,0.4);
    color: #f06b5c;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 13px;
    display: inline-block;
}

.risk-badge-medium {
    background: rgba(244,168,53,0.15);
    border: 1px solid rgba(244,168,53,0.4);
    color: #f4a835;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 13px;
    display: inline-block;
}

.risk-badge-low {
    background: rgba(62,207,142,0.15);
    border: 1px solid rgba(62,207,142,0.4);
    color: #3ecf8e;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 13px;
    display: inline-block;
}

.section-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #5a5a7a;
    margin-bottom: 16px;
    display: block;
}

.action-card {
    background: #0f0f18;
    border: 1px solid #1e1e2e;
    border-left: 3px solid #7c6af7;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}

.action-card-high {
    border-left-color: #f06b5c;
}

.action-card-medium {
    border-left-color: #f4a835;
}

.action-title {
    color: #e0e0f0;
    font-weight: 500;
    font-size: 14px;
    margin-bottom: 4px;
}

.action-reason {
    color: #5a5a7a;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme ──────────────────────────────────────────
PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#a0a0b8", size=12),
    xaxis=dict(gridcolor="#1e1e2e", linecolor="#1e1e2e", zerolinecolor="#1e1e2e"),
    yaxis=dict(gridcolor="#1e1e2e", linecolor="#1e1e2e", zerolinecolor="#1e1e2e"),
    margin=dict(t=40, b=20, l=20, r=20),
)

COLORS = {
    "primary": "#7c6af7",
    "green":   "#3ecf8e",
    "red":     "#f06b5c",
    "amber":   "#f4a835",
    "teal":    "#2dd4bf",
    "surface": "#0f0f18",
    "border":  "#1e1e2e",
}

ROOT       = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(ROOT, "models")
DATA_PATH  = os.path.join(ROOT, "data", "Customer-Churn-Records.csv")


# ── Load artifacts ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    pipeline  = joblib.load(os.path.join(MODELS_DIR, "pipeline.pkl"))
    explainer = joblib.load(os.path.join(MODELS_DIR, "explainer.pkl"))
    names     = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    return pipeline, explainer, names["all_feature_names"]

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    pipeline, explainer, feature_names = load_artifacts()
    MODEL_OK = True
except Exception as e:
    st.error(f"Model not found. Run `python src/train.py` first.\n\n{e}")
    st.stop()

df = load_data()

metrics_path = os.path.join(MODELS_DIR, "latest_metrics.json")
METRICS = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 24px'>
        <div style='font-size:22px; font-weight:700; color:#ffffff; letter-spacing:-0.02em'>
            🏦 ChurnIQ
        </div>
        <div style='font-size:11px; color:#5a5a7a; margin-top:4px; letter-spacing:0.05em'>
            CUSTOMER RETENTION INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style='background:#0a1628; border:1px solid #1e3a5f; border-radius:10px; padding:14px 16px; margin-bottom:16px'>
        <div style='font-size:11px; color:#3b82f6; font-weight:600; letter-spacing:0.08em; margin-bottom:6px'>
            ● MODEL STATUS
        </div>
        <div style='font-size:13px; color:#93c5fd; font-weight:500'>XGBoost v2.0 · Active</div>
    </div>
    """, unsafe_allow_html=True)

    if METRICS:
        col1, col2 = st.columns(2)
        col1.metric("F1", f"{METRICS.get('f1_score', 0):.3f}")
        col2.metric("AUC", f"{METRICS.get('roc_auc', 0):.3f}")
        col1.metric("Recall", f"{METRICS.get('recall', 0):.3f}")
        col2.metric("Accuracy", f"{METRICS.get('accuracy', 0):.3f}")

    st.divider()

    st.markdown("""
    <div style='font-size:11px; color:#5a5a7a; font-weight:600; letter-spacing:0.08em; margin-bottom:10px'>
        TECH STACK
    </div>
    """, unsafe_allow_html=True)

    for tech in ["XGBoost 3.2", "SHAP 0.51", "FastAPI 0.136", "MLflow 3.12", "Evidently AI", "Streamlit"]:
        st.markdown(f"""
        <div style='font-size:12px; color:#a0a0b8; padding:4px 0; 
             border-bottom:1px solid #1e1e2e; display:flex; align-items:center; gap:8px'>
            <span style='color:#7c6af7'>▸</span> {tech}
        </div>
        """, unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  📊  Overview",
    "  🎯  Risk Scorer",
    "  🔍  SHAP Analysis",
    "  📈  Cohort Intelligence",
])


# ══════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW / EDA
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div style='padding: 32px 0 24px'>
        <div style='font-size:11px; font-weight:600; color:#7c6af7; 
             letter-spacing:0.12em; text-transform:uppercase; margin-bottom:8px'>
            PLATFORM OVERVIEW
        </div>
        <h1 style='font-size:32px; font-weight:700; letter-spacing:-0.02em; margin:0; color:#fff'>
            Customer Churn Intelligence
        </h1>
        <p style='color:#5a5a7a; margin-top:8px; font-size:15px'>
            Real-time churn prediction with explainable AI and retention recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI strip
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Churned", f"{df['Exited'].sum():,}", delta=None)
    c3.metric("Churn Rate", f"{df['Exited'].mean():.1%}")
    c4.metric("Avg Balance", f"£{df['Balance'].mean():,.0f}")

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Pie(
            labels=["Retained", "Churned"],
            values=[df['Exited'].value_counts()[0], df['Exited'].value_counts()[1]],
            hole=0.65,
            marker_colors=[COLORS["green"], COLORS["red"]],
            textfont=dict(color="#ffffff", size=13),
        ))
        fig.update_layout(
            title=dict(text="Churn Distribution", font=dict(color="#fff", size=15)),
            showlegend=True,
            legend=dict(font=dict(color="#a0a0b8")),
            **PLOT_THEME,
            annotations=[dict(
                text=f"<b>{df['Exited'].mean():.1%}</b><br>Churn",
                x=0.5, y=0.5, font=dict(size=18, color="#ffffff"),
                showarrow=False
            )]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.histogram(
            df, x="Age", color="Exited", barmode="overlay",
            color_discrete_map={0: COLORS["green"], 1: COLORS["red"]},
            opacity=0.8, nbins=40,
        )
        fig2.update_layout(
            title=dict(text="Age Distribution by Churn", font=dict(color="#fff", size=15)),
            legend=dict(font=dict(color="#a0a0b8"), title_text=""),
            **PLOT_THEME,
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        geo = df.groupby("Geography")["Exited"].mean().reset_index()
        geo.columns = ["Geography", "Churn Rate"]
        fig3 = px.bar(
            geo, x="Geography", y="Churn Rate",
            color="Churn Rate",
            color_continuous_scale=[[0, COLORS["green"]], [0.5, COLORS["amber"]], [1, COLORS["red"]]],
            text=geo["Churn Rate"].apply(lambda x: f"{x:.1%}"),
        )
        fig3.update_traces(textposition="outside", textfont=dict(color="#fff"))
        fig3.update_layout(
            title=dict(text="Churn Rate by Geography", font=dict(color="#fff", size=15)),
            showlegend=False, coloraxis_showscale=False,
            **PLOT_THEME,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        prod = df.groupby("NumOfProducts")["Exited"].mean().reset_index()
        prod.columns = ["Products", "Churn Rate"]
        fig4 = px.bar(
            prod, x="Products", y="Churn Rate",
            color="Churn Rate",
            color_continuous_scale=[[0, COLORS["green"]], [0.5, COLORS["amber"]], [1, COLORS["red"]]],
            text=prod["Churn Rate"].apply(lambda x: f"{x:.1%}"),
        )
        fig4.update_traces(textposition="outside", textfont=dict(color="#fff"))
        fig4.update_layout(
            title=dict(text="Churn Rate by Number of Products", font=dict(color="#fff", size=15)),
            showlegend=False, coloraxis_showscale=False,
            **PLOT_THEME,
        )
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2: RISK SCORER
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style='padding: 32px 0 24px'>
        <div style='font-size:11px; font-weight:600; color:#7c6af7; 
             letter-spacing:0.12em; text-transform:uppercase; margin-bottom:8px'>
            INDIVIDUAL ASSESSMENT
        </div>
        <h1 style='font-size:32px; font-weight:700; letter-spacing:-0.02em; margin:0; color:#fff'>
            Customer Risk Scorer
        </h1>
        <p style='color:#5a5a7a; margin-top:8px; font-size:15px'>
            Enter customer profile to get churn probability, SHAP explanation, and retention actions
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<span class="section-label">Demographics</span>', unsafe_allow_html=True)
        geography    = st.selectbox("Geography", ["France", "Germany", "Spain"], label_visibility="visible")
        gender       = st.selectbox("Gender", ["Female", "Male"])
        age          = st.slider("Age", 18, 92, 42)
        credit_score = st.slider("Credit Score", 300, 900, 650)

    with col2:
        st.markdown('<span class="section-label">Financial Profile</span>', unsafe_allow_html=True)
        tenure  = st.slider("Tenure (years)", 0, 10, 3)
        balance = st.number_input("Balance (£)", 0.0, 300000.0, 60000.0, step=1000.0)
        salary  = st.number_input("Estimated Salary (£)", 0.0, 250000.0, 80000.0, step=5000.0)

    with col3:
        st.markdown('<span class="section-label">Product & Engagement</span>', unsafe_allow_html=True)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_card     = st.radio("Has Credit Card", [1, 0],
                                format_func=lambda x: "Yes" if x else "No", horizontal=True)
        is_active    = st.radio("Active Member", [1, 0],
                                format_func=lambda x: "Yes" if x else "No", horizontal=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("🎯  Analyse Churn Risk", use_container_width=True):
        input_df = pd.DataFrame([{
            "Geography": geography, "Gender": gender,
            "CreditScore": credit_score, "Age": age, "Tenure": tenure,
            "Balance": balance, "NumOfProducts": num_products,
            "HasCrCard": has_card, "IsActiveMember": is_active,
            "EstimatedSalary": salary,
        }])
        input_df = engineer_features(input_df)

        prob  = float(pipeline.predict_proba(input_df)[0, 1])
        label = "HIGH" if prob >= 0.65 else ("MEDIUM" if prob >= 0.35 else "LOW")
        badge_class = f"risk-badge-{label.lower()}"

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown("---")

        # Result header
        badge_html = f'<span class="{badge_class}">{label} RISK</span>'
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:16px; padding:8px 0 24px'>
            <div>
                <div style='font-size:13px; color:#5a5a7a; margin-bottom:4px'>Churn Risk Assessment</div>
                <div style='display:flex; align-items:center; gap:12px'>
                    <span style='font-size:42px; font-weight:700; color:#fff'>{prob:.1%}</span>
                    {badge_html}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        res1, res2 = st.columns([1, 1])

        with res1:
            # Gauge
            gauge_color = {"HIGH": COLORS["red"], "MEDIUM": COLORS["amber"], "LOW": COLORS["green"]}[label]
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Churn Probability %", "font": {"color": "#a0a0b8", "size": 13}},
                number={"suffix": "%", "valueformat": ".1f", "font": {"color": "#ffffff", "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#5a5a7a",
                             "tickfont": {"color": "#5a5a7a"}},
                    "bar": {"color": gauge_color, "thickness": 0.7},
                    "bgcolor": "#1e1e2e",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 35],  "color": "rgba(62,207,142,0.08)"},
                        {"range": [35, 65], "color": "rgba(244,168,53,0.08)"},
                        {"range": [65, 100],"color": "rgba(240,107,92,0.08)"},
                    ],
                    "threshold": {
                        "line": {"color": gauge_color, "width": 3},
                        "thickness": 0.8,
                        "value": prob * 100,
                    },
                },
            ))
            fig.update_layout(
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                margin=dict(t=50, b=20, l=30, r=30),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Business impact
            revenue_at_risk = prob * 50000
            saveable = revenue_at_risk * 0.30
            st.markdown(f"""
            <div style='background:#0a0a0f; border:1px solid #1e1e2e; border-radius:12px; padding:18px 20px'>
                <div style='font-size:11px; color:#5a5a7a; font-weight:600; 
                     letter-spacing:0.1em; text-transform:uppercase; margin-bottom:14px'>
                    Business Impact
                </div>
                <div style='display:flex; justify-content:space-between; margin-bottom:10px'>
                    <span style='color:#a0a0b8; font-size:13px'>Revenue at risk</span>
                    <span style='color:#f06b5c; font-weight:600; font-size:13px'>£{revenue_at_risk:,.0f}</span>
                </div>
                <div style='display:flex; justify-content:space-between'>
                    <span style='color:#a0a0b8; font-size:13px'>Est. recoverable</span>
                    <span style='color:#3ecf8e; font-weight:600; font-size:13px'>£{saveable:,.0f}</span>
                </div>
                <div style='font-size:11px; color:#3a3a4a; margin-top:10px'>
                    Based on £50k avg annual revenue · 30% retention success rate
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res2:
            # Retention actions
            st.markdown("""
            <div style='font-size:11px; color:#5a5a7a; font-weight:600; 
                 letter-spacing:0.1em; text-transform:uppercase; margin-bottom:14px'>
                Recommended Actions
            </div>
            """, unsafe_allow_html=True)

            actions = []
            if balance == 0:
                actions.append(("HIGH", "Offer savings account with 4.5% APY",
                                 "Zero balance is the #1 churn predictor in this dataset"))
            if is_active == 0 and age >= 45:
                actions.append(("HIGH", "Assign dedicated relationship manager",
                                 "Inactive senior customers have 3× higher churn rate"))
            if num_products == 1:
                actions.append(("HIGH", "Bundle additional products with 20% discount",
                                 "Single-product customers churn significantly more"))
            if geography == "Germany":
                actions.append(("MEDIUM", "Enrol in Germany loyalty cashback programme",
                                 "Germany has 32% churn rate vs 16% France average"))
            if credit_score < 500:
                actions.append(("MEDIUM", "Offer credit-building programme",
                                 "Low credit score signals financial stress"))
            if tenure <= 1:
                actions.append(("MEDIUM", "Trigger 90-day new customer success journey",
                                 "First-year customers have the highest exit probability"))
            if not actions:
                actions.append(("LOW", "Schedule quarterly relationship check-in",
                                 "No high-priority risk factors detected"))

            for impact, action, reason in actions[:4]:
                color = {"HIGH": "#f06b5c", "MEDIUM": "#f4a835", "LOW": "#3ecf8e"}[impact]
                st.markdown(f"""
                <div style='background:#0a0a0f; border:1px solid #1e1e2e; 
                     border-left:3px solid {color}; border-radius:10px; 
                     padding:14px 16px; margin-bottom:10px'>
                    <div style='font-size:10px; color:{color}; font-weight:600; 
                         letter-spacing:0.08em; margin-bottom:5px'>{impact} PRIORITY</div>
                    <div style='color:#e0e0f0; font-size:13px; font-weight:500; margin-bottom:4px'>
                        {action}
                    </div>
                    <div style='color:#5a5a7a; font-size:12px'>{reason}</div>
                </div>
                """, unsafe_allow_html=True)

        # Store for SHAP tab
        st.session_state["last_input"] = input_df
        st.session_state["last_prob"]  = prob


# ══════════════════════════════════════════════════════════════
# TAB 3: SHAP ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style='padding: 32px 0 24px'>
        <div style='font-size:11px; font-weight:600; color:#7c6af7; 
             letter-spacing:0.12em; text-transform:uppercase; margin-bottom:8px'>
            EXPLAINABLE AI
        </div>
        <h1 style='font-size:32px; font-weight:700; letter-spacing:-0.02em; margin:0; color:#fff'>
            SHAP Feature Analysis
        </h1>
        <p style='color:#5a5a7a; margin-top:8px; font-size:15px'>
            Understand why the model made its prediction — feature by feature
        </p>
    </div>
    """, unsafe_allow_html=True)

    if "last_input" not in st.session_state:
        st.markdown("""
        <div style='background:rgba(124,106,247,0.06); border:1px solid rgba(124,106,247,0.2); 
             border-radius:12px; padding:24px; text-align:center; color:#7c6af7'>
            <div style='font-size:32px; margin-bottom:10px'>👆</div>
            <div style='font-weight:600; margin-bottom:4px'>No prediction yet</div>
            <div style='font-size:13px; color:#5a5a7a'>
                Go to Risk Scorer tab, enter a customer profile and click Analyse
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        input_df = st.session_state["last_input"]
        prob     = st.session_state["last_prob"]
        label    = "HIGH" if prob >= 0.65 else ("MEDIUM" if prob >= 0.35 else "LOW")
        badge_color = {"HIGH": "#f06b5c", "MEDIUM": "#f4a835", "LOW": "#3ecf8e"}[label]

        st.markdown(f"""
        <div style='background:#0a0a0f; border:1px solid #1e1e2e; border-radius:12px; 
             padding:18px 22px; margin-bottom:24px; display:flex; align-items:center; gap:16px'>
            <div>
                <div style='font-size:12px; color:#5a5a7a; margin-bottom:2px'>Last prediction</div>
                <div style='font-size:24px; font-weight:700; color:#fff'>{prob:.1%} churn probability</div>
            </div>
            <div style='background:rgba(124,106,247,0.12); border:1px solid rgba(124,106,247,0.3); 
                 color:{badge_color}; padding:6px 14px; border-radius:20px; font-weight:600; font-size:12px'>
                {label} RISK
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            preprocessor = pipeline.named_steps["preprocessor"]
            X_t = preprocessor.transform(input_df)
            sv  = explainer(X_t)
            vals = sv.values[0] if len(sv.values.shape) == 2 else sv.values[0, :, 1]

            shap_df = pd.DataFrame({
                "Feature":     feature_names[:len(vals)],
                "SHAP Value":  vals,
                "abs":         np.abs(vals),
            }).sort_values("abs", ascending=True).tail(12)

            colors = [COLORS["red"] if v > 0 else COLORS["green"]
                      for v in shap_df["SHAP Value"]]

            fig = go.Figure(go.Bar(
                x=shap_df["SHAP Value"],
                y=shap_df["Feature"],
                orientation="h",
                marker=dict(color=colors, opacity=0.9),
                text=[f"{v:+.3f}" for v in shap_df["SHAP Value"]],
                textposition="outside",
                textfont=dict(color="#a0a0b8", size=11),
            ))
            fig.update_layout(
                title=dict(
                    text="Feature Impact on Churn Probability",
                    font=dict(color="#fff", size=16)
                ),
                xaxis_title="SHAP Value",
                height=460,
                **PLOT_THEME,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div style='display:flex; gap:20px; margin-top:8px'>
                <div style='display:flex; align-items:center; gap:8px'>
                    <div style='width:12px; height:12px; background:#f06b5c; border-radius:2px'></div>
                    <span style='font-size:12px; color:#5a5a7a'>Increases churn risk</span>
                </div>
                <div style='display:flex; align-items:center; gap:8px'>
                    <div style='width:12px; height:12px; background:#3ecf8e; border-radius:2px'></div>
                    <span style='font-size:12px; color:#5a5a7a'>Decreases churn risk</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"SHAP error: {e}")

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='padding:16px 0'>
        <div style='font-size:11px; color:#5a5a7a; font-weight:600; 
             letter-spacing:0.1em; text-transform:uppercase; margin-bottom:12px'>
            Why SHAP matters in Indian Banking
        </div>
        <p style='color:#a0a0b8; font-size:14px; line-height:1.7'>
            RBI's model risk management guidelines require financial institutions to explain 
            credit and retention decisions to customers. SHAP provides the exact feature-level 
            attribution regulators require — making this not just a nice-to-have, but a 
            <strong style='color:#e0e0f0'>compliance requirement</strong> for production ML systems 
            at Indian banks and NBFCs.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 4: COHORT INTELLIGENCE
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div style='padding: 32px 0 24px'>
        <div style='font-size:11px; font-weight:600; color:#7c6af7; 
             letter-spacing:0.12em; text-transform:uppercase; margin-bottom:8px'>
            SEGMENT INTELLIGENCE
        </div>
        <h1 style='font-size:32px; font-weight:700; letter-spacing:-0.02em; margin:0; color:#fff'>
            Cohort Analysis
        </h1>
        <p style='color:#5a5a7a; margin-top:8px; font-size:15px'>
            Identify highest-risk customer segments by geography, age, and product usage
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        pivot = df.groupby(["Geography", "NumOfProducts"])["Exited"].mean().reset_index()
        heat  = pivot.pivot(index="Geography", columns="NumOfProducts", values="Exited")
        fig   = px.imshow(
            heat, text_auto=".0%",
            color_continuous_scale=[[0, "#0a2e1a"], [0.4, "#f4a835"], [1, "#f06b5c"]],
            aspect="auto",
        )
        fig.update_layout(
            title=dict(text="Churn Rate: Geography × Product Count",
                       font=dict(color="#fff", size=15)),
            **PLOT_THEME,
            coloraxis_showscale=False,
        )
        fig.update_traces(textfont=dict(color="#fff", size=12))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cross = df.groupby(["IsActiveMember", "Geography"])["Exited"].mean().reset_index()
        cross["Status"] = cross["IsActiveMember"].map({0: "Inactive", 1: "Active"})
        fig2 = px.bar(
            cross, x="Geography", y="Exited", color="Status", barmode="group",
            color_discrete_map={"Active": COLORS["green"], "Inactive": COLORS["red"]},
            opacity=0.85,
        )
        fig2.update_layout(
            title=dict(text="Active vs Inactive Churn Rate",
                       font=dict(color="#fff", size=15)),
            legend=dict(font=dict(color="#a0a0b8"), title_text=""),
            yaxis_tickformat=".0%",
            **PLOT_THEME,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Age group heatmap
    df["age_group"] = pd.cut(
        df["Age"], bins=[18, 30, 40, 50, 60, 100],
        labels=["18–30", "31–40", "41–50", "51–60", "60+"]
    )
    age_geo = df.groupby(["age_group", "Geography"], observed=True)["Exited"].mean().reset_index()
    heat2   = age_geo.pivot(index="age_group", columns="Geography", values="Exited")
    fig3    = px.imshow(
        heat2, text_auto=".0%",
        color_continuous_scale=[[0, "#0a2e1a"], [0.4, "#f4a835"], [1, "#f06b5c"]],
    )
    fig3.update_layout(
        title=dict(text="Churn Rate: Age Group × Geography",
                   font=dict(color="#fff", size=15)),
        **PLOT_THEME,
        coloraxis_showscale=False,
    )
    fig3.update_traces(textfont=dict(color="#fff", size=13))
    st.plotly_chart(fig3, use_container_width=True)

    # Revenue at risk
    st.markdown("""
    <div style='font-size:11px; color:#5a5a7a; font-weight:600; 
         letter-spacing:0.1em; text-transform:uppercase; margin:24px 0 12px'>
        Revenue at Risk by Geography
    </div>
    """, unsafe_allow_html=True)

    rev = df.groupby("Geography").agg(
        customers=("Exited", "count"),
        churn_rate=("Exited", "mean"),
    ).reset_index()
    rev["revenue_at_risk"] = rev["customers"] * rev["churn_rate"] * 50000

    fig4 = px.bar(
        rev, x="Geography", y="revenue_at_risk",
        color="revenue_at_risk",
        color_continuous_scale=[[0, COLORS["amber"]], [1, COLORS["red"]]],
        text=rev["revenue_at_risk"].apply(lambda x: f"£{x/1e6:.1f}M"),
    )
    fig4.update_traces(textposition="outside", textfont=dict(color="#fff"))
    fig4.update_layout(
        yaxis_tickformat="£,.0f",
        showlegend=False,
        coloraxis_showscale=False,
        **PLOT_THEME,
    )
    st.plotly_chart(fig4, use_container_width=True)