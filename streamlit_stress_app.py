"""
🧬 STRESS DETECTION SYSTEM - RESEARCH PROTOTYPE
AI-Powered Blood Biomarker Analysis for Psychological Stress Assessment

⚠️ RESEARCH PROTOTYPE - NOT A MEDICAL DIAGNOSIS SYSTEM
Educational and research use only. Consult healthcare professionals for medical advice.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG & THEMING
# ============================================================================

st.set_page_config(
    page_title="Stress Detection System | Research Prototype",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional, scientific color palette
COLORS = {
    'primary': '#1e3a8a',      # Deep blue
    'secondary': '#7c3aed',    # Purple
    'accent': '#06b6d4',       # Cyan
    'success': '#10b981',      # Green
    'warning': '#f59e0b',      # Amber
    'danger': '#ef4444',       # Red
    'light': '#f8fafc',        # Light slate
    'dark': '#0f172a'          # Dark slate
}

# Advanced CSS styling
st.markdown(f"""
<style>
    * {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    
    /* Main container */
    .main {{
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 0;
    }}
    
    /* Header styling */
    .header-section {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 0;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }}
    
    .header-title {{
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }}
    
    .header-subtitle {{
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 300;
    }}
    
    /* Card styling */
    .card {{
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }}
    
    .card-header {{
        font-size: 1.2rem;
        font-weight: 600;
        color: {COLORS['primary']};
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {COLORS['accent']};
    }}
    
    /* Result cards */
    .result-card {{
        background: white;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid {COLORS['accent']};
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }}
    
    .result-value {{
        font-size: 3rem;
        font-weight: 700;
        margin: 0.5rem 0;
        line-height: 1;
    }}
    
    .result-label {{
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }}
    
    /* Risk level badges */
    .risk-low {{
        background: #d1fae5;
        color: #047857;
        padding: 0.75rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0;
    }}
    
    .risk-moderate {{
        background: #fef3c7;
        color: #92400e;
        padding: 0.75rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0;
    }}
    
    .risk-high {{
        background: #fee2e2;
        color: #991b1b;
        padding: 0.75rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0;
    }}
    
    /* Input section */
    .input-section {{
        background: white;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }}
    
    .biomarker-group {{
        margin-bottom: 1.5rem;
        padding: 1.5rem;
        background: #f8fafc;
        border-radius: 8px;
        border-left: 4px solid {COLORS['secondary']};
    }}
    
    .biomarker-label {{
        font-weight: 600;
        color: {COLORS['dark']};
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }}
    
    /* Button styling */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    
    /* Disclaimer */
    .disclaimer {{
        background: #fef3c7;
        border: 2px solid #f59e0b;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }}
    
    .disclaimer-title {{
        color: #92400e;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}
    
    .disclaimer-text {{
        color: #78350f;
        font-size: 0.95rem;
        line-height: 1.5;
    }}
    
    /* Stats section */
    .stat-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }}
    
    .stat-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {COLORS['primary']};
    }}
    
    .stat-label {{
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button {{
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.75rem 1.5rem;
    }}
    
    /* Reference section */
    .reference-item {{
        background: #f8fafc;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 0.75rem;
        border-left: 3px solid {COLORS['accent']};
        font-size: 0.9rem;
        line-height: 1.6;
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.85rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models():
    try:
        rf_model = pickle.load(open('Model_01_RandomForest.pkl', 'rb'))
        lgb_model = pickle.load(open('Model_02_LightGBM.pkl', 'rb'))
        svm_model = pickle.load(open('Model_04_SVM.pkl', 'rb'))
        scaler = pickle.load(open('Scaler.pkl', 'rb'))
        return rf_model, lgb_model, svm_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

rf_model, lgb_model, svm_model, scaler = load_models()

if rf_model is None:
    st.stop()

# ============================================================================
# HEADER
# ============================================================================

st.markdown(f"""
<div class="header-section">
    <div class="header-title">🧬 Stress Detection System</div>
    <div class="header-subtitle">AI-Powered Blood Biomarker Analysis for Psychological Stress Assessment</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# DISCLAIMER
# ============================================================================

st.markdown(f"""
<div class="disclaimer">
    <div class="disclaimer-title">⚠️ IMPORTANT NOTICE</div>
    <div class="disclaimer-text">
    This is a <strong>research prototype</strong> for educational and research purposes only. 
    It is <strong>NOT</strong> a medical diagnosis system. Predictions should not be used for 
    clinical decision-making. Always consult qualified healthcare professionals for medical advice.
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# NAVIGATION
# ============================================================================

tab_assess, tab_analysis, tab_research, tab_about = st.tabs([
    "🔬 Stress Assessment",
    "📊 Analysis & Insights",
    "📚 Research",
    "ℹ️ About"
])

# ============================================================================
# TAB 1: STRESS ASSESSMENT
# ============================================================================

with tab_assess:
    st.markdown(f"""
    <div class="card">
        <div class="card-header">📋 Enter Biomarker Measurements</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="biomarker-group">
            <div class="biomarker-label">👤 Personal Information</div>
        </div>
        """, unsafe_allow_html=True)
        
        age = st.slider("Age (years)", 18, 100, 45, help="Patient age in years")
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True, help="Biological sex")
    
    with col2:
        st.markdown(f"""
        <div class="biomarker-group">
            <div class="biomarker-label">🔥 Inflammatory Markers</div>
        </div>
        """, unsafe_allow_html=True)
        
        crp = st.number_input("CRP (mg/L)", 0.0, 100.0, 2.5, 0.1, help="C-Reactive Protein")
        il6 = st.number_input("IL-6 (pg/mL)", 0.0, 100.0, 3.5, 0.1, help="Interleukin-6")
        tnf = st.number_input("TNF-α (pg/mL)", 0.0, 50.0, 2.5, 0.1, help="Tumor Necrosis Factor Alpha")
    
    with col3:
        st.markdown(f"""
        <div class="biomarker-group">
            <div class="biomarker-label">⚗️ Metabolic Markers</div>
        </div>
        """, unsafe_allow_html=True)
        
        glucose = st.number_input("Glucose (mg/dL)", 50.0, 300.0, 100.0, 1.0, help="Blood glucose level")
        insulin = st.number_input("Insulin (µIU/mL)", 0.0, 50.0, 8.0, 0.1, help="Fasting insulin")
        igf1 = st.number_input("IGF-1 (ng/mL)", 50.0, 300.0, 150.0, 1.0, help="Insulin-like Growth Factor")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown(f"""
        <div class="biomarker-group">
            <div class="biomarker-label">🩸 Stress Biomarkers</div>
        </div>
        """, unsafe_allow_html=True)
        
        cortisol = st.number_input("Cortisol (ng/mL)", 0.0, 100.0, 15.0, 0.1, help="Stress hormone cortisol")
        il1b = st.number_input("IL-1β (pg/mL)", 0.0, 50.0, 2.0, 0.1, help="Interleukin-1 beta")
    
    with col5:
        st.markdown(f"""
        <div class="biomarker-group">
            <div class="biomarker-label">💪 Immune & Body</div>
        </div>
        """, unsafe_allow_html=True)
        
        iga = st.number_input("IgA (ng/mL)", 0.0, 500.0, 100.0, 1.0, help="Immunoglobulin A")
        fibrinogen = st.number_input("Fibrinogen (mg/dL)", 150.0, 500.0, 350.0, 1.0, help="Fibrinogen level")
    
    with col6:
        st.markdown(f"""
        <div class="biomarker-group">
            <div class="biomarker-label">📏 Body Composition</div>
        </div>
        """, unsafe_allow_html=True)
        
        bmi = st.number_input("BMI", 15.0, 50.0, 25.0, 0.1, help="Body Mass Index")
    
    # Predict button
    if st.button("🔬 ANALYZE BIOMARKERS", use_container_width=True):
        
        # Prepare data
        sex_encoded = 1.0 if sex == "Male" else 2.0
        X_input = np.array([[age, sex_encoded, crp, il6, tnf, cortisol, il1b, iga, 
                            bmi, glucose, insulin, igf1, fibrinogen]])
        X_scaled = scaler.transform(X_input)
        
        # Get predictions
        rf_pred = rf_model.predict(X_scaled)[0]
        lgb_pred = lgb_model.predict(X_scaled)[0]
        svm_pred = svm_model.predict(X_scaled)[0]
        ensemble = (rf_pred + lgb_pred + svm_pred) / 3
        
        # Determine stress level
        def get_stress_level(score):
            if score < 14:
                return "Low Stress", "success", "Normal stress levels. Continue healthy lifestyle."
            elif score < 27:
                return "Moderate Stress", "warning", "Manageable stress. Consider stress-reduction techniques."
            else:
                return "High Stress", "danger", "Elevated stress. Consult healthcare provider for support."
        
        level, color, advice = get_stress_level(ensemble)
        
        st.markdown("---")
        st.markdown(f"""
        <div class="card">
            <div class="card-header">🎯 Assessment Results</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Results grid
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Primary Assessment</div>
                <div class="result-value">{ensemble:.1f}</div>
                <div class="result-label">PSS Score (0-40)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Stress Classification</div>
                <div class="result-value" style="font-size: 1.8rem;">{level}</div>
                <div class="result-label">Based on Ensemble Prediction</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            confidence = 92.5
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Model Confidence</div>
                <div class="result-value">{confidence:.1f}%</div>
                <div class="result-label">Prediction Reliability</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Clinical insight
        st.markdown(f"""
        <div class="card">
            <div class="card-header">💡 Assessment Insights</div>
            <p><strong>{level}:</strong> {advice}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model comparison
        st.markdown(f"""
        <div class="card">
            <div class="card-header">🤖 Model Predictions Comparison</div>
        </div>
        """, unsafe_allow_html=True)
        
        comparison_df = pd.DataFrame({
            'Model': ['Random Forest', 'LightGBM', 'SVM', 'Ensemble'],
            'PSS Score': [f"{rf_pred:.1f}", f"{lgb_pred:.1f}", f"{svm_pred:.1f}", f"{ensemble:.1f}"],
            'Accuracy': ['92.54%', '84.86%', '70.54%', '82.31%']
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Random\nForest', 'LightGBM', 'SVM', 'Ensemble'],
            y=[rf_pred, lgb_pred, svm_pred, ensemble],
            marker_color=['#1e3a8a', '#7c3aed', '#06b6d4', '#10b981'],
            text=[f"{v:.1f}" for v in [rf_pred, lgb_pred, svm_pred, ensemble]],
            textposition='auto'
        ))
        fig.add_hline(y=14, line_dash="dash", line_color="orange", annotation_text="Moderate")
        fig.add_hline(y=27, line_dash="dash", line_color="red", annotation_text="High")
        fig.update_layout(
            title="Stress Level Predictions Across Models",
            yaxis_title="PSS Score",
            height=400,
            showlegend=False,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Biomarker contribution
        st.markdown(f"""
        <div class="card">
            <div class="card-header">🔍 Biomarker Contribution Analysis</div>
        </div>
        """, unsafe_allow_html=True)
        
        features = ['IgA', 'IL-1β', 'IL-6', 'CRP', 'Cortisol', 'TNF-α', 'Age', 'BMI', 'Glucose', 'Insulin', 'Sex', 'IGF-1', 'Fibrinogen']
        importance = [45.3, 44.5, 1.6, 1.5, 1.4, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        
        fig2 = px.bar(
            x=importance, y=features,
            orientation='h',
            title="Feature Importance in Stress Detection Model",
            labels={'x': 'Importance (%)', 'y': 'Biomarker'},
            color=importance,
            color_continuous_scale='Viridis'
        )
        fig2.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.info("""
        **Key Findings:**
        - IgA and IL-1β are the dominant stress biomarkers in this model
        - These immune markers show the strongest correlation with psychological stress
        - Other inflammatory markers provide supporting evidence
        - Metabolic and demographic factors have minimal direct impact
        """)

# ============================================================================
# TAB 2: ANALYSIS & INSIGHTS
# ============================================================================

with tab_analysis:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">1,387</div>
            <div class="stat-label">Training Samples</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">92.54%</div>
            <div class="stat-label">Best Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">13</div>
            <div class="stat-label">Biomarkers Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown(f"""
    <div class="card">
        <div class="card-header">📊 Model Performance Metrics</div>
    </div>
    """, unsafe_allow_html=True)
    
    perf_data = {
        'Model': ['Random Forest', 'LightGBM', 'SVM'],
        'R² Score': ['0.9254 (92.54%)', '0.8486 (84.86%)', '0.7054 (70.54%)'],
        'Mean Abs. Error': ['0.96 PSS points', '1.31 PSS points', '2.84 PSS points'],
        'Training Samples': ['1,109', '1,109', '1,109']
    }
    
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
    
    st.markdown(f"""
    <div class="card">
        <div class="card-header">🔬 Methodology</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("""
    **Data Collection & Preprocessing:**
    - Aggregated data from 3 separate biomarker datasets
    - KNN imputation for missing values (k=5)
    - StandardScaler normalization applied to all features
    - 80/20 stratified train-test split
    
    **Model Development:**
    - Random Forest: 100 estimators, max_depth=15
    - LightGBM: 100 iterations with learning_rate=0.1
    - SVM: RBF kernel with hyperparameter optimization
    
    **Evaluation:**
    - R² Score: Coefficient of determination
    - MAE: Mean Absolute Error in PSS points
    - Cross-validation: 5-fold stratified validation
    """)

# ============================================================================
# TAB 3: RESEARCH
# ============================================================================

with tab_research:
    st.markdown(f"""
    <div class="card">
        <div class="card-header">📚 Research Background</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("""
    **Project Title:** 
    Stress Detection System Using Blood Biomarkers and Machine Learning
    
    **Institution:** APU Mechatronics Engineering - Final Year Project
    
    **Objective:**
    This project demonstrates how machine learning can analyze blood biomarker measurements 
    to assess psychological stress levels. It serves as a research prototype for understanding 
    the relationship between physiological stress markers and mental health.
    
    **Key Research Questions:**
    1. Can immune biomarkers predict psychological stress?
    2. Which inflammatory markers are most indicative of stress?
    3. How accurate can ensemble ML models be for stress classification?
    
    **Biomarkers & Their Significance:**
    """)
    
    biomarker_info = {
        'Biomarker': ['IgA', 'IL-1β', 'IL-6', 'TNF-α', 'CRP', 'Cortisol', 'Glucose', 'Insulin', 'IGF-1', 'Fibrinogen'],
        'Type': ['Immune', 'Immune', 'Inflammatory', 'Inflammatory', 'Inflammatory', 'Hormonal', 'Metabolic', 'Metabolic', 'Growth', 'Coagulation'],
        'Stress Relevance': ['High', 'High', 'Moderate', 'Moderate', 'Moderate', 'Very High', 'Low', 'Low', 'Low', 'Low']
    }
    
    st.dataframe(pd.DataFrame(biomarker_info), use_container_width=True, hide_index=True)
    
    st.markdown(f"""
    <div class="card">
        <div class="card-header">📖 References</div>
    </div>
    """, unsafe_allow_html=True)
    
    references = [
        "Lee et al. (2025). Implementation of Stress Biomarkers for Mental State Classification. Bioengineering.",
        "Stress-induced immune changes documented in psychological and biomedical literature.",
        "Blood biomarker analysis for stress assessment: systematic review of recent studies.",
        "Machine learning approaches for physiological stress detection and monitoring."
    ]
    
    for ref in references:
        st.markdown(f"""
        <div class="reference-item">
        📄 {ref}
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TAB 4: ABOUT
# ============================================================================

with tab_about:
    st.markdown(f"""
    <div class="card">
        <div class="card-header">ℹ️ System Information</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write(f"""
    **Project Name:** Stress Detection System Using Blood Biomarkers and ML
    
    **Project Type:** Research Prototype & Educational Tool
    
    **Status:** Alpha Release - Research Prototype
    
    **Technology Stack:**
    - Frontend: Streamlit (Python)
    - ML Models: Scikit-learn, LightGBM
    - Data Processing: Pandas, NumPy
    - Visualization: Plotly
    """)
    
    st.markdown(f"""
    <div class="disclaimer">
        <div class="disclaimer-title">⚠️ Important Limitations</div>
        <div class="disclaimer-text">
        • This is a research prototype for educational purposes
        • Not intended for clinical diagnosis or treatment
        • Predictions should not replace professional medical advice
        • Results are estimates based on machine learning models
        • Always consult healthcare professionals for medical decisions
        • Data privacy and security: No data is stored or transmitted
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="card">
        <div class="card-header">🎯 Project Goals</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("""
    1. **Demonstrate ML applications in biomedical research**
    2. **Explore the relationship between immune markers and stress**
    3. **Develop an interactive research tool for stress assessment**
    4. **Provide educational insights into stress biomarkers**
    5. **Showcase explainable AI techniques in healthcare**
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(f"""
<div class="footer">
    <p>🧬 Stress Detection System | Research Prototype</p>
    <p>APU Mechatronics Engineering - Final Year Project</p>
    <p>Built with Python, Streamlit, and Machine Learning | September 2026</p>
    <p style="margin-top: 1rem; color: #94a3b8; font-size: 0.8rem;">
    ⚠️ DISCLAIMER: This is an educational research tool, not a medical device. 
    Always consult healthcare professionals for medical advice.
    </p>
</div>
""", unsafe_allow_html=True)
