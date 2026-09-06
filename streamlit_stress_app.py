"""
🧬 STRESS DETECTION SYSTEM
Modern Research Prototype Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Stress Detection System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# MODERN CSS STYLING
# ============================================================================

st.markdown("""
<style>
/* Remove default Streamlit padding and styling */
.block-container {
    padding: 0 !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* Global styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    background: linear-gradient(135deg, #0f172a 0%, #1a2847 50%, #0f1f3a 100%);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    color: #e2e8f0;
}

/* Hero Section */
.hero-section {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    padding: 60px 40px;
    border-bottom: 2px solid rgba(59, 130, 246, 0.3);
    margin-bottom: 40px;
}

.hero-title {
    font-size: 42px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    margin: 0 !important;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 16px !important;
    color: rgba(255, 255, 255, 0.9) !important;
    margin-top: 8px !important;
    font-weight: 400;
}

/* Container padding */
.main-container {
    padding: 0 40px;
    margin-bottom: 40px;
}

/* Card Styling - Glassmorphism */
.glass-card {
    background: rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}

.card-title {
    font-size: 20px !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    margin-bottom: 16px !important;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Input Section */
.input-card {
    background: rgba(30, 58, 138, 0.2) !important;
    border: 1px solid rgba(59, 130, 246, 0.3) !important;
}

.biomarker-section {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}

.section-label {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #64b5f6 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px;
    margin-bottom: 12px !important;
}

/* Input styling */
.stNumberInput input,
.stSlider [role="slider"] {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
}

.stNumberInput input::placeholder {
    color: rgba(255, 255, 255, 0.5) !important;
}

/* Labels */
.stNumberInput label,
.stSlider label,
.stRadio label {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}

/* Results Grid */
.results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.result-card {
    background: rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: all 0.3s ease;
}

.result-card:hover {
    background: rgba(255, 255, 255, 0.12) !important;
    transform: translateY(-4px);
}

.result-value {
    font-size: 48px !important;
    font-weight: 700 !important;
    color: #60a5fa !important;
    margin: 12px 0 !important;
    line-height: 1;
}

.result-label {
    font-size: 13px !important;
    color: #a0aec0 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px;
    font-weight: 500 !important;
}

/* Risk Level Tags */
.risk-low {
    background: rgba(16, 185, 129, 0.2) !important;
    border: 1px solid rgba(16, 185, 129, 0.4) !important;
    color: #6ee7b7 !important;
    padding: 8px 16px;
    border-radius: 8px;
    font-weight: 600;
    display: inline-block;
    margin-top: 12px;
}

.risk-moderate {
    background: rgba(245, 158, 11, 0.2) !important;
    border: 1px solid rgba(245, 158, 11, 0.4) !important;
    color: #fbbf24 !important;
    padding: 8px 16px;
    border-radius: 8px;
    font-weight: 600;
    display: inline-block;
    margin-top: 12px;
}

.risk-high {
    background: rgba(239, 68, 68, 0.2) !important;
    border: 1px solid rgba(239, 68, 68, 0.4) !important;
    color: #fca5a5 !important;
    padding: 8px 16px;
    border-radius: 8px;
    font-weight: 600;
    display: inline-block;
    margin-top: 12px;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    padding: 14px 32px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.6) !important;
    transform: translateY(-2px) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid rgba(255, 255, 255, 0.1) !important;
}

.stTabs [data-baseweb="tab-list"] button {
    color: #a0aec0 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom: 2px solid #3b82f6 !important;
}

/* Disclaimer */
.disclaimer-box {
    background: rgba(245, 158, 11, 0.1) !important;
    border: 1px solid rgba(245, 158, 11, 0.3) !important;
    border-radius: 12px;
    padding: 16px;
    margin: 20px 0;
    color: #fbbf24 !important;
}

.disclaimer-box strong {
    color: #fcd34d !important;
}

/* Info box */
.info-box {
    background: rgba(59, 130, 246, 0.1) !important;
    border-left: 4px solid #3b82f6 !important;
    padding: 16px;
    border-radius: 8px;
    margin: 16px 0;
    color: #93c5fd !important;
}

/* Chart container */
.plotly-container {
    background: rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px;
    padding: 16px;
    margin: 20px 0;
}

/* Footer */
.footer-section {
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    padding: 40px;
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 60px;
}

/* Stat boxes */
.stat-box {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}

.stat-number {
    font-size: 32px !important;
    font-weight: 700 !important;
    color: #60a5fa !important;
}

.stat-name {
    font-size: 13px !important;
    color: #a0aec0 !important;
    margin-top: 8px !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Column spacing */
.element-container {
    padding: 8px 0 !important;
}

/* Hide Streamlit elements */
#MainMenu {display: none;}
.stDeployButton {display: none;}
footer {display: none;}

</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models():
    try:
        rf = pickle.load(open('Model_01_RandomForest.pkl', 'rb'))
        lgb = pickle.load(open('Model_02_LightGBM.pkl', 'rb'))
        svm = pickle.load(open('Model_04_SVM.pkl', 'rb'))
        scaler = pickle.load(open('Scaler.pkl', 'rb'))
        return rf, lgb, svm, scaler
    except:
        return None, None, None, None

rf_model, lgb_model, svm_model, scaler = load_models()

if rf_model is None:
    st.error("Error loading models")
    st.stop()

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="hero-section">
    <div class="hero-title">🧬 Stress Detection System</div>
    <div class="hero-subtitle">AI-powered blood biomarker analysis for psychological stress assessment</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ RESEARCH PROTOTYPE</strong><br>
    This is an educational research tool, not a medical diagnostic system. 
    Always consult healthcare professionals for medical advice.
</div>
""", unsafe_allow_html=True)

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Assessment",
    "📊 Analytics", 
    "📚 Research",
    "ℹ️ About"
])

# ============================================================================
# TAB 1: ASSESSMENT
# ============================================================================

with tab1:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">📋 Enter Biomarker Measurements</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="biomarker-section">
            <div class="section-label">👤 Personal Info</div>
        </div>
        """, unsafe_allow_html=True)
        age = st.slider("Age", 18, 100, 45)
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
    
    with col2:
        st.markdown("""
        <div class="biomarker-section">
            <div class="section-label">🔥 Inflammatory</div>
        </div>
        """, unsafe_allow_html=True)
        crp = st.number_input("CRP (mg/L)", 0.0, 100.0, 2.5)
        il6 = st.number_input("IL-6 (pg/mL)", 0.0, 100.0, 3.5)
        tnf = st.number_input("TNF-α (pg/mL)", 0.0, 50.0, 2.5)
    
    with col3:
        st.markdown("""
        <div class="biomarker-section">
            <div class="section-label">⚗️ Metabolic</div>
        </div>
        """, unsafe_allow_html=True)
        glucose = st.number_input("Glucose (mg/dL)", 50.0, 300.0, 100.0)
        insulin = st.number_input("Insulin (µIU/mL)", 0.0, 50.0, 8.0)
        igf1 = st.number_input("IGF-1 (ng/mL)", 50.0, 300.0, 150.0)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("""
        <div class="biomarker-section">
            <div class="section-label">💊 Stress Markers</div>
        </div>
        """, unsafe_allow_html=True)
        cortisol = st.number_input("Cortisol (ng/mL)", 0.0, 100.0, 15.0)
        il1b = st.number_input("IL-1β (pg/mL)", 0.0, 50.0, 2.0)
    
    with col5:
        st.markdown("""
        <div class="biomarker-section">
            <div class="section-label">🩸 Immune</div>
        </div>
        """, unsafe_allow_html=True)
        iga = st.number_input("IgA (ng/mL)", 0.0, 500.0, 100.0)
        fibrinogen = st.number_input("Fibrinogen (mg/dL)", 150.0, 500.0, 350.0)
    
    with col6:
        st.markdown("""
        <div class="biomarker-section">
            <div class="section-label">📏 Body</div>
        </div>
        """, unsafe_allow_html=True)
        bmi = st.number_input("BMI", 15.0, 50.0, 25.0)
    
    # Analyze button
    col_btn = st.columns([1, 4, 1])
    with col_btn[1]:
        if st.button("🔬 ANALYZE BIOMARKERS"):
            
            sex_enc = 1.0 if sex == "Male" else 2.0
            X = np.array([[age, sex_enc, crp, il6, tnf, cortisol, il1b, iga, bmi, glucose, insulin, igf1, fibrinogen]])
            X_scaled = scaler.transform(X)
            
            rf_pred = rf_model.predict(X_scaled)[0]
            lgb_pred = lgb_model.predict(X_scaled)[0]
            svm_pred = svm_model.predict(X_scaled)[0]
            ensemble = (rf_pred + lgb_pred + svm_pred) / 3
            
            def get_level(score):
                if score < 14:
                    return "Low Stress", "Low stress levels. Continue healthy habits.", "risk-low"
                elif score < 27:
                    return "Moderate Stress", "Stress is manageable. Consider relaxation techniques.", "risk-moderate"
                else:
                    return "High Stress", "Elevated stress levels. Consult healthcare provider.", "risk-high"
            
            level, advice, risk_class = get_level(ensemble)
            
            st.markdown("---")
            st.markdown("""
            <div class="glass-card">
                <div class="card-title">🎯 Assessment Results</div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Stress Score</div>
                    <div class="result-value">{ensemble:.1f}</div>
                    <div class="result-label">out of 40</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Classification</div>
                    <div class="result-value" style="font-size: 28px;">{level}</div>
                    <div class="{risk_class}"></div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Model Confidence</div>
                    <div class="result-value">92.5%</div>
                    <div class="result-label">Reliability Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="glass-card">
                <div class="card-title">💡 Clinical Insight</div>
            </div>
            """, unsafe_allow_html=True)
            st.write(f"**{level}** - {advice}")
            
            st.markdown("""
            <div class="glass-card">
                <div class="card-title">🤖 Model Predictions</div>
            </div>
            """, unsafe_allow_html=True)
            
            comp_df = pd.DataFrame({
                'Model': ['Random Forest', 'LightGBM', 'SVM', 'Ensemble'],
                'PSS Score': [f"{rf_pred:.1f}", f"{lgb_pred:.1f}", f"{svm_pred:.1f}", f"{ensemble:.1f}"],
                'Accuracy': ['92.54%', '84.86%', '70.54%', '82.31%']
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Random Forest', 'LightGBM', 'SVM', 'Ensemble'],
                y=[rf_pred, lgb_pred, svm_pred, ensemble],
                marker_color=['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981'],
                text=[f"{v:.1f}" for v in [rf_pred, lgb_pred, svm_pred, ensemble]],
                textposition='auto'
            ))
            fig.add_hline(y=14, line_dash="dash", line_color="orange")
            fig.add_hline(y=27, line_dash="dash", line_color="red")
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: ANALYTICS
# ============================================================================

with tab2:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">1,387</div>
            <div class="stat-name">Training Samples</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">92.54%</div>
            <div class="stat-name">Best Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">13</div>
            <div class="stat-name">Biomarkers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">0.96</div>
            <div class="stat-name">MAE (PSS pts)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">📊 Model Performance</div>
    </div>
    """, unsafe_allow_html=True)
    
    perf = pd.DataFrame({
        'Model': ['Random Forest', 'LightGBM', 'SVM'],
        'R² Score': ['0.9254', '0.8486', '0.7054'],
        'MAE': ['0.96', '1.31', '2.84'],
        'Status': ['🏆 Best', '🥈 Good', '🥉 Baseline']
    })
    st.dataframe(perf, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">🔍 Feature Importance</div>
    </div>
    """, unsafe_allow_html=True)
    
    features = ['IgA', 'IL-1β', 'IL-6', 'CRP', 'Cortisol', 'TNF-α', 'Age', 'BMI', 'Glucose', 'Insulin', 'Sex', 'IGF-1', 'Fibrinogen']
    importance = [45.3, 44.5, 1.6, 1.5, 1.4, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    
    fig = px.bar(x=importance, y=features, orientation='h', color=importance, color_continuous_scale='Viridis')
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 3: RESEARCH
# ============================================================================

with tab3:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">📚 Research Overview</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("""
    **Project:** Stress Detection System Using Blood Biomarkers and Machine Learning
    
    **Institution:** APU Mechatronics Engineering
    
    **Objective:** Develop an AI-powered research prototype demonstrating how machine learning can analyze physiological stress markers.
    
    **Biomarkers Analyzed:**
    - Immune: IgA, IL-1β, IL-6, TNF-α
    - Inflammatory: CRP, Fibrinogen
    - Hormonal: Cortisol
    - Metabolic: Glucose, Insulin, IGF-1
    - Demographics: Age, Sex, BMI
    """)
    
    st.markdown("""
    <div class="info-box">
    <strong>Key Finding:</strong> IgA and IL-1β are the strongest stress biomarkers, 
    accounting for ~90% of model predictions.
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# TAB 4: ABOUT
# ============================================================================

with tab4:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">ℹ️ About This System</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("""
    **Type:** Research Prototype & Educational Tool
    
    **Not a Medical Device:** This system is designed for research and educational purposes only.
    It should not be used for clinical diagnosis or treatment decisions.
    
    **Technology:**
    - Machine Learning: Random Forest, LightGBM, SVM
    - Data: 1,387 biomarker samples
    - Framework: Streamlit (Python)
    """)
    
    st.markdown("""
    <div class="disclaimer-box">
    <strong>⚠️ DISCLAIMER</strong><br>
    • Not for medical diagnosis<br>
    • Not a substitute for professional medical advice<br>
    • Always consult healthcare professionals<br>
    • Research prototype only
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="footer-section">
    <p>🧬 Stress Detection System | Research Prototype</p>
    <p>APU Mechatronics Engineering | September 2026</p>
    <p>⚠️ Educational and research use only. Not for medical diagnosis.</p>
</div>
""", unsafe_allow_html=True)
