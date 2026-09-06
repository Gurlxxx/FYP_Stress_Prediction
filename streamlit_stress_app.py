"""
🧬 STRESS DETECTION SYSTEM - NURA INSPIRED
Premium AI Wellness Platform with Cinematic Biomarker Analysis

A futuristic AI laboratory meets a premium wellness platform.
Understanding your body's stress response through intelligent biomarker analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Stress Detection System | NURA",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# PREMIUM CINEMATIC CSS
# ============================================================================

st.markdown("""
<style>
/* Root styles */
:root {
    --primary: #3b82f6;
    --secondary: #8b5cf6;
    --accent: #06b6d4;
    --dark-bg: #0f172a;
    --card-bg: rgba(15, 23, 42, 0.8);
}

/* Remove Streamlit defaults */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

html {
    background: linear-gradient(135deg, #0f172a 0%, #1a2847 50%, #0f1f3a 100%);
}

body {
    background: linear-gradient(135deg, #0f172a 0%, #1a2847 50%, #0f1f3a 100%) !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    color: #e2e8f0;
}

/* Hide Streamlit elements */
#MainMenu { display: none; }
.stDeployButton { display: none; }
footer { display: none; }

/* ========== HERO SECTION ========== */
.hero-container {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #1e40af 100%);
    padding: 80px 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
    border-bottom: 2px solid rgba(59, 130, 246, 0.3);
    margin-bottom: 40px;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
        radial-gradient(circle at 20% 50%, rgba(6, 182, 212, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(139, 92, 246, 0.1) 0%, transparent 50%);
    pointer-events: none;
    animation: pulse 8s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 1; }
}

.hero-title {
    font-size: 3.5rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    margin: 0 !important;
    letter-spacing: -1px;
    position: relative;
    z-index: 1;
    text-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
}

.hero-subtitle {
    font-size: 1.2rem !important;
    color: rgba(255, 255, 255, 0.9) !important;
    margin-top: 16px !important;
    font-weight: 300;
    position: relative;
    z-index: 1;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}

/* ========== MAIN CONTAINER ========== */
.main-container {
    padding: 0 40px 60px 40px;
    max-width: 1200px;
    margin: 0 auto;
}

/* ========== STEP INDICATOR ========== */
.step-indicator {
    display: flex;
    justify-content: space-between;
    margin-bottom: 40px;
    gap: 20px;
}

.step-item {
    flex: 1;
    padding: 20px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    text-align: center;
    transition: all 0.3s ease;
}

.step-item.active {
    background: rgba(59, 130, 246, 0.2);
    border-color: rgba(59, 130, 246, 0.5);
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.2);
}

.step-number {
    font-size: 24px !important;
    font-weight: 700 !important;
    color: #60a5fa !important;
    margin-bottom: 8px !important;
}

.step-label {
    font-size: 13px !important;
    color: #a0aec0 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ========== BIOMARKER SECTIONS ========== */
.biomarker-group {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 24px;
    transition: all 0.3s ease;
}

.biomarker-group:hover {
    border-color: rgba(59, 130, 246, 0.5);
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.15);
}

.group-title {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #60a5fa !important;
    margin-bottom: 20px !important;
    display: flex;
    align-items: center;
    gap: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ========== INPUT STYLING ========== */
.stNumberInput input,
.stSlider [role="slider"] {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    padding: 12px !important;
    transition: all 0.3s ease !important;
}

.stNumberInput input:focus,
.stSlider [role="slider"]:focus {
    border-color: rgba(59, 130, 246, 0.6) !important;
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.2) !important;
}

.stNumberInput label,
.stSlider label {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}

/* ========== BUTTONS ========== */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    padding: 14px 40px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4) !important;
    width: 100% !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    box-shadow: 0 8px 30px rgba(59, 130, 246, 0.6) !important;
    transform: translateY(-2px) !important;
}

/* ========== GLOWING CARDS ========== */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
}

.glass-card:hover {
    border-color: rgba(59, 130, 246, 0.3);
    box-shadow: 0 12px 40px rgba(59, 130, 246, 0.15);
}

.card-title {
    font-size: 20px !important;
    font-weight: 600 !important;
    color: #60a5fa !important;
    margin-bottom: 20px !important;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ========== RESULT CARDS ========== */
.result-card {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
    border: 2px solid rgba(59, 130, 246, 0.3);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin: 20px 0;
    box-shadow: 0 0 30px rgba(59, 130, 246, 0.2);
    animation: glowPulse 3s ease-in-out infinite;
}

@keyframes glowPulse {
    0%, 100% { 
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.2);
        border-color: rgba(59, 130, 246, 0.3);
    }
    50% { 
        box-shadow: 0 0 50px rgba(59, 130, 246, 0.4);
        border-color: rgba(59, 130, 246, 0.5);
    }
}

.result-value {
    font-size: 56px !important;
    font-weight: 700 !important;
    color: #60a5fa !important;
    margin: 16px 0 !important;
    line-height: 1;
    text-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
}

.result-label {
    font-size: 14px !important;
    color: #a0aec0 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
}

/* ========== STRESS LEVEL BADGES ========== */
.stress-low {
    background: rgba(16, 185, 129, 0.15);
    border: 2px solid rgba(16, 185, 129, 0.4);
    color: #6ee7b7;
    padding: 12px 24px;
    border-radius: 12px;
    font-weight: 600;
    display: inline-block;
    margin-top: 16px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
}

.stress-moderate {
    background: rgba(245, 158, 11, 0.15);
    border: 2px solid rgba(245, 158, 11, 0.4);
    color: #fbbf24;
    padding: 12px 24px;
    border-radius: 12px;
    font-weight: 600;
    display: inline-block;
    margin-top: 16px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    box-shadow: 0 0 20px rgba(245, 158, 11, 0.2);
}

.stress-high {
    background: rgba(239, 68, 68, 0.15);
    border: 2px solid rgba(239, 68, 68, 0.4);
    color: #fca5a5;
    padding: 12px 24px;
    border-radius: 12px;
    font-weight: 600;
    display: inline-block;
    margin-top: 16px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.2);
}

/* ========== LOADING ANIMATION ========== */
.loading-text {
    font-size: 16px !important;
    color: #60a5fa !important;
    text-align: center;
    margin: 20px 0;
    animation: fadeInOut 2s ease-in-out infinite;
}

@keyframes fadeInOut {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 1; }
}

/* ========== RADIAL CHART CONTAINER ========== */
.chart-container {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    margin: 20px 0;
}

/* ========== AI EXPLANATION ========== */
.ai-explanation {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(6, 182, 212, 0.15) 100%);
    border-left: 4px solid #8b5cf6;
    padding: 20px;
    border-radius: 8px;
    margin: 20px 0;
    color: #cbd5e1;
    line-height: 1.6;
}

.ai-explanation strong {
    color: #60a5fa;
}

/* ========== DISCLAIMER ========== */
.disclaimer {
    background: rgba(245, 158, 11, 0.1);
    border: 2px solid rgba(245, 158, 11, 0.3);
    border-radius: 12px;
    padding: 16px;
    margin: 20px 0;
    color: #fbbf24;
}

.disclaimer strong {
    color: #fcd34d;
}

/* ========== TABS ========== */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid rgba(255, 255, 255, 0.1) !important;
}

.stTabs [data-baseweb="tab-list"] button {
    color: #a0aec0 !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom: 2px solid #3b82f6 !important;
}

/* ========== FOOTER ========== */
.footer {
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    padding: 40px;
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 60px;
}

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
        st.error("Error loading models")
        return None, None, None, None

rf_model, lgb_model, svm_model, scaler = load_models()

if rf_model is None:
    st.stop()

# ============================================================================
# HERO SECTION
# ============================================================================

st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧬 Stress Detection System</div>
    <div class="hero-subtitle">
    Understanding your body's stress response through AI-powered blood biomarker analysis
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ RESEARCH PROTOTYPE</strong><br>
    This is an educational research tool, not a medical diagnostic system. 
    Always consult healthcare professionals for medical decisions.
</div>
""", unsafe_allow_html=True)

# ============================================================================
# TABS
# ============================================================================

tab_analyze, tab_research, tab_about = st.tabs([
    "🔬 Biomarker Analysis",
    "📚 Research",
    "ℹ️ About"
])

# ============================================================================
# TAB 1: BIOMARKER ANALYSIS
# ============================================================================

with tab_analyze:
    
    # Step indicator
    st.markdown("""
    <div class="step-indicator">
        <div class="step-item active">
            <div class="step-number">1</div>
            <div class="step-label">Biological Profile</div>
        </div>
        <div class="step-item">
            <div class="step-number">2</div>
            <div class="step-label">Biomarkers</div>
        </div>
        <div class="step-item">
            <div class="step-number">3</div>
            <div class="step-label">AI Analysis</div>
        </div>
        <div class="step-item">
            <div class="step-number">4</div>
            <div class="step-label">Insights</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 1: Personal Info
    st.markdown("""
    <div class="biomarker-group">
        <div class="group-title">👤 Step 1: Biological Profile</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 100, 45, help="Your age in years")
    with col2:
        sex = st.radio("Biological Sex", ["Male", "Female"], horizontal=True)
    with col3:
        bmi = st.number_input("BMI", 15.0, 50.0, 25.0, 0.1, help="Body Mass Index")
    
    # Step 2: Biomarkers
    st.markdown("""
    <div class="biomarker-group">
        <div class="group-title">🔥 Step 2: Inflammatory Biomarkers</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        crp = st.number_input("CRP (mg/L)", 0.0, 100.0, 2.5, 0.1)
    with col2:
        il6 = st.number_input("IL-6 (pg/mL)", 0.0, 100.0, 3.5, 0.1)
    with col3:
        tnf = st.number_input("TNF-α (pg/mL)", 0.0, 50.0, 2.5, 0.1)
    with col4:
        fibrinogen = st.number_input("Fibrinogen (mg/dL)", 150.0, 500.0, 350.0, 1.0)
    
    st.markdown("""
    <div class="biomarker-group">
        <div class="group-title">💊 Step 2: Stress & Immune Markers</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        cortisol = st.number_input("Cortisol (ng/mL)", 0.0, 100.0, 15.0, 0.1)
    with col2:
        il1b = st.number_input("IL-1β (pg/mL)", 0.0, 50.0, 2.0, 0.1)
    with col3:
        iga = st.number_input("IgA (ng/mL)", 0.0, 500.0, 100.0, 1.0)
    
    st.markdown("""
    <div class="biomarker-group">
        <div class="group-title">⚗️ Step 2: Metabolic Markers</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        glucose = st.number_input("Glucose (mg/dL)", 50.0, 300.0, 100.0, 1.0)
    with col2:
        insulin = st.number_input("Insulin (µIU/mL)", 0.0, 50.0, 8.0, 0.1)
    with col3:
        igf1 = st.number_input("IGF-1 (ng/mL)", 50.0, 300.0, 150.0, 1.0)
    
    # Analyze button
    col_btn = st.columns([0.3, 0.4, 0.3])
    with col_btn[1]:
        if st.button("🔬 ANALYZE BIOMARKER PATTERN", use_container_width=True):
            
            # Animated analysis
            st.markdown('<div class="loading-text">⏳ Analyzing biological patterns...</div>', unsafe_allow_html=True)
            time.sleep(0.8)
            
            st.markdown('<div class="loading-text">🧬 Processing biomarker data...</div>', unsafe_allow_html=True)
            time.sleep(0.8)
            
            st.markdown('<div class="loading-text">🤖 Running AI stress assessment...</div>', unsafe_allow_html=True)
            time.sleep(1)
            
            # Prepare and predict
            sex_enc = 1.0 if sex == "Male" else 2.0
            X = np.array([[age, sex_enc, crp, il6, tnf, cortisol, il1b, iga, bmi, glucose, insulin, igf1, fibrinogen]])
            X_scaled = scaler.transform(X)
            
            rf_pred = rf_model.predict(X_scaled)[0]
            lgb_pred = lgb_model.predict(X_scaled)[0]
            svm_pred = svm_model.predict(X_scaled)[0]
            ensemble = (rf_pred + lgb_pred + svm_pred) / 3
            
            def get_level(score):
                if score < 14:
                    return "Low Stress", "Your stress levels are within normal range. Continue maintaining your wellness routine.", "stress-low", "✅"
                elif score < 27:
                    return "Moderate Stress", "Your body shows manageable stress indicators. Consider stress-reduction techniques like meditation or exercise.", "stress-moderate", "⚠️"
                else:
                    return "High Stress", "Elevated stress markers detected. We recommend consulting healthcare professionals for support.", "stress-high", "🔴"
            
            level, advice, stress_class, icon = get_level(ensemble)
            
            st.markdown("---")
            
            # Results
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Your Stress Assessment</div>
                <div class="result-value">{level}</div>
                <div class="result-label">PSS Score: {ensemble:.1f} / 40</div>
                <div class="{stress_class}">{icon} {level}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="ai-explanation">
                <strong>💡 AI Analysis:</strong> {advice}
            </div>
            """, unsafe_allow_html=True)
            
            # Model comparison
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
            
            # Radial chart for biomarker importance
            st.markdown("""
            <div class="glass-card">
                <div class="card-title">📊 Biomarker Contribution Analysis</div>
            </div>
            """, unsafe_allow_html=True)
            
            features = ['IgA', 'IL-1β', 'IL-6', 'CRP', 'Cortisol', 'TNF-α', 'Glucose', 'Insulin', 'Age', 'BMI', 'IGF-1', 'Sex', 'Fibrinogen']
            importance = [45.3, 44.5, 1.6, 1.5, 1.4, 1.2, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
            
            fig = go.Figure(data=go.Scatterpolar(
                r=importance,
                theta=features,
                fill='toself',
                marker_color='rgba(59, 130, 246, 0.6)',
                line_color='rgba(59, 130, 246, 0.8)',
                name='Importance'
            ))
            
            fig.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, color='rgba(255,255,255,0.2)', gridcolor='rgba(255,255,255,0.1)'),
                    angularaxis=dict(color='rgba(255,255,255,0.4)')
                ),
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Bar chart for feature importance
            st.markdown("""
            <div class="glass-card">
                <div class="card-title">📈 Feature Importance Ranking</div>
            </div>
            """, unsafe_allow_html=True)
            
            top_features = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)[:8]
            feat_names, feat_importance = zip(*top_features)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                y=list(feat_names),
                x=list(feat_importance),
                orientation='h',
                marker_color=['#60a5fa' if i > 10 else '#a0aec0' for i in feat_importance],
                text=[f'{i:.1f}%' for i in feat_importance],
                textposition='auto'
            ))
            
            fig2.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350,
                showlegend=False,
                margin=dict(l=150)
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Key insights
            st.markdown("""
            <div class="glass-card">
                <div class="card-title">🔬 Key Biomarker Insights</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write(f"""
            **Top Stress Indicators:**
            - **IgA** (45.3% contribution) - Immune response marker, strongly correlated with stress
            - **IL-1β** (44.5% contribution) - Pro-inflammatory cytokine, key stress indicator
            - **IL-6** (1.6% contribution) - Supporting inflammatory marker
            - **CRP** (1.5% contribution) - General inflammation indicator
            
            **Your Profile Summary:**
            - Age: {age} years
            - BMI: {bmi:.1f} ({"Healthy" if 18.5 <= bmi <= 24.9 else "Monitor"})
            - Primary stress markers: IgA and IL-1β levels
            - Recommendation: {advice}
            """)

# ============================================================================
# TAB 2: RESEARCH
# ============================================================================

with tab_research:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">📚 Research Methodology</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("""
    **Project Overview:**
    This research prototype demonstrates how machine learning can analyze blood biomarkers 
    to assess psychological stress levels.
    
    **Dataset:**
    - Total samples: 1,387 participants
    - Data sources: MIDUS biomarker dataset + clinical studies
    - Features: 13 blood biomarkers + demographics
    
    **Machine Learning Pipeline:**
    1. Data Collection - Blood biomarker measurements
    2. Data Preprocessing - KNN imputation, normalization
    3. Feature Selection - 13 key biomarkers selected
    4. Model Training - Random Forest, LightGBM, SVM
    5. Evaluation - Cross-validation, performance metrics
    6. Explainable AI - SHAP analysis, feature importance
    
    **Models Evaluated:**
    - Random Forest: 92.54% accuracy (BEST)
    - LightGBM: 84.86% accuracy
    - SVM: 70.54% accuracy
    """)
    
    st.markdown("""
    <div class="ai-explanation">
    <strong>🧬 Biological Background:</strong><br>
    Psychological stress triggers physiological responses measurable through blood biomarkers.
    IgA and inflammatory cytokines show the strongest correlation with perceived stress levels,
    making them reliable indicators for AI-based stress assessment.
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# TAB 3: ABOUT
# ============================================================================

with tab_about:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">ℹ️ About This System</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("""
    **Project:** Stress Detection System Using Blood Biomarkers and Machine Learning
    
    **Institution:** APU Mechatronics Engineering - Final Year Project
    
    **Type:** Research Prototype & Educational Tool
    
    **Purpose:** Demonstrate AI applications in biomedical research and wellness assessment
    
    **Technology Stack:**
    - Machine Learning: Scikit-learn, LightGBM
    - Framework: Streamlit (Python)
    - Visualization: Plotly
    - Data Processing: Pandas, NumPy
    """)
    
    st.markdown("""
    <div class="disclaimer">
    <strong>⚠️ IMPORTANT DISCLAIMER</strong><br>
    • This is NOT a medical diagnosis system<br>
    • NOT intended for clinical decision-making<br>
    • Educational and research use only<br>
    • Always consult healthcare professionals<br>
    • Results are estimates based on ML models
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="footer">
    <p>🧬 Stress Detection System | Premium AI Wellness Research Prototype</p>
    <p>APU Mechatronics Engineering | September 2026</p>
    <p>⚠️ Educational research tool. Not for medical diagnosis.</p>
</div>
""", unsafe_allow_html=True)
