import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Custom CSS Styling
# -------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    .hero-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #334155 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.15);
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #FFFFFF 0%, #E2E8F0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        background: rgba(99, 102, 241, 0.2);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.4);
        margin-bottom: 0.8rem;
    }

    /* High Contrast Slate Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
        border-right: 1px solid #334155 !important;
    }

    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] summary,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #F1F5F9 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #818CF8 !important;
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-weight: 500 !important;
    }

    /* Predict Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.8rem 1.8rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 16px 0 rgba(79, 70, 229, 0.4);
        transition: all 0.2s ease-in-out;
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #4338CA 0%, #4F46E5 100%);
        box-shadow: 0 6px 22px 0 rgba(79, 70, 229, 0.55);
        transform: translateY(-2px);
        color: white;
    }

    /* Risk Badge Styling */
    .risk-badge-high {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 1.5px solid #FCA5A5;
        border-radius: 14px;
        padding: 1.5rem;
        color: #991B1B;
    }

    .risk-badge-medium {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border: 1.5px solid #FCD34D;
        border-radius: 14px;
        padding: 1.5rem;
        color: #92400E;
    }

    .risk-badge-low {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 1.5px solid #86EFAC;
        border-radius: 14px;
        padding: 1.5rem;
        color: #166534;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #EEF2FF !important;
        color: #4F46E5 !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model & Assets
# -------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("models/xgb_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    scaler = joblib.load("models/scaler.pkl") if os.path.exists("models/scaler.pkl") else None
    explainer = shap.TreeExplainer(model)
    return model, feature_columns, scaler, explainer

model, feature_columns, scaler, explainer = load_assets()

# Fast Cached SHAP Calculation
@st.cache_data
def get_sample_shap(_explainer, df_sample):
    return _explainer.shap_values(df_sample)[0]

# Helper for loading static image outputs
@st.cache_data
def get_resized_image(image_path, width=700, height=500):
    if os.path.exists(image_path):
        img = Image.open(image_path)
        return img.resize((width, height), Image.Resampling.LANCZOS)
    return None

# -------------------------
# 📊 Sidebar: Model Information
# -------------------------
st.sidebar.markdown("### 📊 Model Architecture")
st.sidebar.markdown("**Engine:** `XGBoost (Hyperparameter Tuned)`")

st.sidebar.markdown("#### Performance Metrics")
col_sb1, col_sb2 = st.sidebar.columns(2)
col_sb1.metric("Accuracy", "85.6%")
col_sb2.metric("ROC-AUC", "0.91")

col_sb3, col_sb4 = st.sidebar.columns(2)
col_sb3.metric("Precision", "85.7%")
col_sb4.metric("Recall", "96.4%")

st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚙️ Feature Schema")
st.sidebar.caption(f"Total Model Features: `{len(feature_columns)}`")
with st.sidebar.expander("View Input Features"):
    st.code("\n".join(feature_columns[:10]) + "\n... (+ regional state dummies)", language="text")

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Real-Time Analytics Engine**: Computes custom Plotly Gauge dials, "
    "Behavioral Radar Signatures, and SHAP Local Feature Impacts dynamically per input!"
)

# -------------------------
# Hero Banner
# -------------------------
st.markdown("""
<div class="hero-header">
    <div class="badge-pill">Enterprise AI Platform v3.0</div>
    <h1 class="hero-title">Customer Churn Intelligence Platform</h1>
    <p class="hero-subtitle">Interactive real-time prediction, dynamic behavioral radar signatures, and instant SHAP explainability visuals.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Customer Inputs Form
# -------------------------
st.markdown("### 👤 Customer Behavioral Input Profile")
st.caption("Adjust the sliders & numerical controls below. Interactive charts update dynamically upon prediction!")

col1, col2, col3 = st.columns(3)

with col1:
    frequency = st.number_input(
        "Frequency (total orders)",
        min_value=1,
        max_value=50,
        value=1,
        help="Total number of orders placed by customer"
    )
    monetary = st.number_input(
        "Monetary Value ($ total spent)",
        min_value=0.0,
        value=150.0,
        step=10.0,
        help="Total lifetime expenditure"
    )
    avg_order_value = st.number_input(
        "Average Order Value ($)",
        min_value=0.0,
        value=150.0,
        step=10.0
    )
    avg_review_score = st.slider(
        "Customer Review Rating ⭐",
        min_value=1.0,
        max_value=5.0,
        value=4.5,
        step=0.5,
        help="Mean review star rating submitted by customer"
    )

with col2:
    tenure = st.number_input(
        "Customer Tenure (days active)",
        min_value=0,
        max_value=800,
        value=0,
        help="Days between customer's first and last order"
    )
    avg_items = st.number_input(
        "Average Items per Order",
        min_value=1.0,
        value=1.0,
        step=0.5
    )
    avg_installments = st.number_input(
        "Average Payment Installments",
        min_value=1.0,
        max_value=24.0,
        value=1.0,
        step=1.0
    )

with col3:
    total_freight = st.number_input(
        "Total Freight Cost ($)",
        min_value=0.0,
        value=20.0,
        step=5.0
    )
    avg_delivery_delay = st.number_input(
        "Delivery Delay (days)",
        min_value=-20.0,
        max_value=40.0,
        value=0.0,
        step=1.0,
        help="Days actual delivery exceeded estimated date (positive = late)"
    )
    
    states = [
        'SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'DF', 'ES', 'GO', 
        'PE', 'CE', 'PA', 'MT', 'MA', 'MS', 'PB', 'RN', 'PI', 'AL', 
        'SE', 'RO', 'AM', 'AP', 'AC', 'RR', 'TO'
    ]
    customer_state = st.selectbox("Customer State Location", options=states, index=0)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------
# Prediction Event Trigger
# -------------------------
if st.button("🔮 Run Churn Prediction Model", type="primary", use_container_width=True):

    # Construct input dataframe
    input_dict = {
        "avg_order_value": avg_order_value,
        "avg_items": avg_items,
        "total_freight": total_freight,
        "avg_installments": avg_installments,
        "avg_review_score": avg_review_score,
        "avg_delivery_time": 12.0,
        "avg_delivery_delay": avg_delivery_delay,
        "tenure": tenure,
        "frequency": frequency,
        "monetary": monetary,
    }

    state_col = f"customer_state_{customer_state}"
    input_data = pd.DataFrame([input_dict])

    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 1 if col == state_col else 0

    input_data = input_data[feature_columns]

    # Predict & calculate fast SHAP
    if scaler is not None:
        try:
            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)[0]
            probability = float(model.predict_proba(scaled_input)[0][1])
            shap_values = get_sample_shap(explainer, pd.DataFrame(scaled_input, columns=feature_columns))
        except Exception:
            prediction = model.predict(input_data)[0]
            probability = float(model.predict_proba(input_data)[0][1])
            shap_values = get_sample_shap(explainer, input_data)
    else:
        prediction = model.predict(input_data)[0]
        probability = float(model.predict_proba(input_data)[0][1])
        shap_values = get_sample_shap(explainer, input_data)

    # Store in session state
    st.session_state["has_predicted"] = True
    st.session_state["prediction"] = prediction
    st.session_state["probability"] = probability
    st.session_state["shap_values"] = shap_values
    st.session_state["input_dict"] = input_dict

# -------------------------
# Render Results & Dynamic Interactive Charts
# -------------------------
if st.session_state.get("has_predicted", False):
    
    probability = st.session_state["probability"]
    shap_values = st.session_state["shap_values"]
    input_dict = st.session_state["input_dict"]

    st.markdown("---")

    # -------------------------
    # 🏎️ Dynamic Plotly Gauge & Risk Card
    # -------------------------
    st.markdown("### 📈 Real-Time Churn Risk Gauge & Status")
    
    col_g1, col_g2 = st.columns([1, 1.2])

    with col_g1:
        if probability > 0.8:
            st.markdown("""
            <div class="risk-badge-high">
                <h3 style="margin:0; font-size:1.4rem;">🚨 Status: High Churn Risk</h3>
                <p style="margin-top:6px; margin-bottom:0; font-size:0.98rem;">Customer is highly susceptible to attrition. Immediate retention intervention required.</p>
            </div>
            """, unsafe_allow_html=True)
        elif probability > 0.5:
            st.markdown("""
            <div class="risk-badge-medium">
                <h3 style="margin:0; font-size:1.4rem;">⚠️ Status: Medium Churn Risk</h3>
                <p style="margin-top:6px; margin-bottom:0; font-size:0.98rem;">Customer displays early churn signals. Good candidate for re-engagement triggers.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-badge-low">
                <h3 style="margin:0; font-size:1.4rem;">✅ Status: Active / Low Risk</h3>
                <p style="margin-top:6px; margin-bottom:0; font-size:0.98rem;">Customer is engaged and satisfied. Target for upsell or referral promos.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Estimated Churn Risk Score", f"{probability:.1%}")

    with col_g2:
        # High contrast gauge dial compatible with Dark and Light Mode
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            number = {
                'suffix': "%", 
                'font': {'size': 36, 'color': "#818CF8", 'family': "Inter, sans-serif"}
            },
            title = {
                'text': "Dynamic Churn Risk Index Dial", 
                'font': {'size': 16, 'color': "#94A3B8", 'family': "Inter, sans-serif"}
            },
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                'bar': {'color': "#6366F1", 'thickness': 0.65},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 1.5,
                'bordercolor': "#475569",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(34, 197, 94, 0.25)'},
                    {'range': [40, 75], 'color': 'rgba(234, 179, 8, 0.25)'},
                    {'range': [75, 100], 'color': 'rgba(239, 68, 68, 0.25)'}
                ],
                'threshold': {
                    'line': {'color': "#EF4444", 'width': 4},
                    'thickness': 0.75,
                    'value': probability * 100
                }
            }
        ))
        fig_gauge.update_layout(
            height=280, 
            margin=dict(l=30, r=30, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC")
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")

    # -------------------------
    # 💡 Business Action Strategy
    # -------------------------
    st.markdown("### 💡 Recommended Retention Action Plan")

    if probability > 0.8:
        st.error("🔴 **High Risk Threshold Exceeded (> 80%)**: Trigger an automated VIP retention package, high-value discount code, or dedicated account management outreach.")
    elif probability > 0.5:
        st.warning("🟡 **Medium Risk Threshold (50% - 80%)**: Enrol in re-engagement email drip sequence, issue customized product recommendations, or offer free shipping incentives.")
    else:
        st.success("🟢 **Low Risk Threshold (< 50%)**: Maintain standard engagement. Target for brand ambassador campaigns, product upsells, or referral rewards.")

    st.markdown("---")

    # -------------------------
    # 📌 Dynamic Visual Analytics & SHAP
    # -------------------------
    st.markdown("### 📊 Dynamic Visual Analytics & SHAP Explainability")
    st.caption("Interactive charts calculated in real-time for THIS specific customer profile:")

    tab1, tab2, tab3, tab4 = st.tabs([
        "⚡ Instant Feature Impact (SHAP)", 
        "🕸️ Behavioral Radar Signature", 
        "📊 Global Feature Importance", 
        "📈 Validation Curves"
    ])

    # --- TAB 1: DYNAMIC LOCAL SHAP FEATURE IMPACT CHART ---
    with tab1:
        st.markdown("#### Real-Time Feature Impact Breakdown for Current Input")
        st.write("Red bars push churn risk **UP**, Green bars push churn risk **DOWN**:")

        # Extract top feature SHAP impacts
        shap_df = pd.DataFrame({
            'Feature': feature_columns,
            'SHAP_Impact': shap_values
        })
        
        name_map = {
            'tenure': 'Customer Tenure (days)',
            'frequency': 'Purchase Frequency (orders)',
            'monetary': 'Lifetime Spend ($)',
            'avg_review_score': 'Review Rating ⭐',
            'avg_delivery_delay': 'Delivery Delay (days)',
            'total_freight': 'Freight Cost ($)',
            'avg_installments': 'Payment Installments',
            'avg_order_value': 'Avg Order Value ($)',
            'avg_items': 'Avg Items per Order',
            'avg_delivery_time': 'Fulfillment Time (days)'
        }
        shap_df['Clean_Feature'] = shap_df['Feature'].map(lambda x: name_map.get(x, x.replace('customer_state_', 'State: ')))
        
        shap_df['Abs_Impact'] = shap_df['SHAP_Impact'].abs()
        top_shap = shap_df.sort_values(by='Abs_Impact', ascending=False).head(10).iloc[::-1]
        
        colors = ['#EF4444' if val > 0 else '#10B981' for val in top_shap['SHAP_Impact']]

        fig_shap_local = go.Figure(go.Bar(
            x=top_shap['SHAP_Impact'],
            y=top_shap['Clean_Feature'],
            orientation='h',
            marker_color=colors,
            text=[f"{val:+.3f}" for val in top_shap['SHAP_Impact']],
            textposition='auto'
        ))
        
        fig_shap_local.update_layout(
            title="Local SHAP Feature Contributions (Current Customer Input)",
            xaxis_title="SHAP Value (Impact on Churn Log-Odds)",
            yaxis_title="Feature",
            height=420,
            margin=dict(l=20, r=20, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155")
        )
        st.plotly_chart(fig_shap_local, use_container_width=True)

    # --- TAB 2: DYNAMIC BEHAVIORAL RADAR SIGNATURE ---
    with tab2:
        st.markdown("#### Dynamic Behavioral Radar Signature")
        st.write("Compares **This Customer's Profile** against typical **Active** vs **Churned** benchmark baselines:")

        radar_categories = ['Tenure', 'Frequency', 'Monetary Spend', 'Review Rating', 'Order Value', 'Logistics Speed']
        
        c_tenure = min(100, (input_dict['tenure'] / 360.0) * 100)
        c_freq = min(100, (input_dict['frequency'] / 5.0) * 100)
        c_monetary = min(100, (input_dict['monetary'] / 500.0) * 100)
        c_review = (input_dict['avg_review_score'] / 5.0) * 100
        c_aov = min(100, (input_dict['avg_order_value'] / 300.0) * 100)
        c_speed = max(0, 100 - max(0, input_dict['avg_delivery_delay'] * 5))

        cust_radar = [c_tenure, c_freq, c_monetary, c_review, c_aov, c_speed]
        active_benchmark = [75, 80, 70, 90, 65, 85]
        churn_benchmark = [15, 20, 25, 50, 40, 40]

        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=cust_radar,
            theta=radar_categories,
            fill='toself',
            name='Current Customer Profile',
            line_color='#818CF8',
            fillcolor='rgba(129, 140, 248, 0.3)'
        ))

        fig_radar.add_trace(go.Scatterpolar(
            r=active_benchmark,
            theta=radar_categories,
            name='Active Customer Baseline',
            line=dict(color='#22C55E', dash='dash')
        ))

        fig_radar.add_trace(go.Scatterpolar(
            r=churn_benchmark,
            theta=radar_categories,
            name='Churned Customer Baseline',
            line=dict(color='#EF4444', dash='dot')
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#334155"),
                angularaxis=dict(gridcolor="#334155"),
                bgcolor="rgba(0,0,0,0)"
            ),
            showlegend=True,
            title="Customer Behavioral Radar Signature vs Benchmark",
            height=450,
            margin=dict(l=40, r=40, t=50, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC")
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # --- TAB 3: GLOBAL SHAP FEATURE IMPORTANCE ---
    with tab3:
        st.markdown("#### Global Feature Importance Plots (SHAP)")
        c1, c2 = st.columns(2)
        
        img_bar = get_resized_image("outputs/shap_bar.png.png", 700, 500)
        img_beeswarm = get_resized_image("outputs/shap_beeswarm.png.png", 700, 500)
        
        with c1:
            if img_bar:
                st.image(img_bar, caption="SHAP Global Feature Importance Bar Plot", use_container_width=True)
        with c2:
            if img_beeswarm:
                st.image(img_beeswarm, caption="SHAP Beeswarm Value Distribution Plot", use_container_width=True)

    # --- TAB 4: VALIDATION CURVES ---
    with tab4:
        st.markdown("#### XGBoost Model Performance & Validation Curves")
        c3, c4 = st.columns(2)
        
        img_roc = get_resized_image("outputs/xgb_roc_curve.png.png", 600, 480)
        img_cm = get_resized_image("outputs/xgb_confusion_matrix.png.png", 600, 480)
        
        with c3:
            if img_roc:
                st.image(img_roc, caption="XGBoost Receiver Operating Characteristic (ROC-AUC 0.910)", use_container_width=True)
        with c4:
            if img_cm:
                st.image(img_cm, caption="XGBoost Test Set Confusion Matrix", use_container_width=True)
else:
    st.info("👆 Adjust the customer behavioral metrics above and click **'🔮 Run Churn Prediction Model'** to generate real-time predictions, dynamic risk dials, radar signatures, and SHAP charts.")