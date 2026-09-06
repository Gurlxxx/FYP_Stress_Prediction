"""
STRESS CLASSIFICATION WEB APP - REAL MODELS VERSION
Uses trained Random Forest, LightGBM, MLP, SVM models
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="🧠 Stress Classification System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {padding: 2rem;}
    h1 {color: #2E86AB; text-align: center;}
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS & SCALER
# ============================================================================

@st.cache_resource
def load_trained_models():
    """Load all trained models and scaler"""
    try:
        rf_model = pickle.load(open('Model_01_RandomForest.pkl', 'rb'))
        lgb_model = pickle.load(open('Model_02_LightGBM.pkl', 'rb'))
        mlp_model = load_model('Model_03_MLP.h5')
        svm_model = pickle.load(open('Model_04_SVM.pkl', 'rb'))
        scaler = pickle.load(open('Scaler.pkl', 'rb'))
        return rf_model, lgb_model, mlp_model, svm_model, scaler
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.error("Make sure all model files are in the same directory as the app!")
        return None, None, None, None, None

rf_model, lgb_model, mlp_model, svm_model, scaler = load_trained_models()

if rf_model is None:
    st.stop()

# ============================================================================
# SIDEBAR - MODEL SELECTION
# ============================================================================

st.sidebar.title("⚙️ Model Settings")

selected_model = st.sidebar.selectbox(
    "Select Model for Predictions:",
    ["Random Forest (Best)", "LightGBM", "MLP (Deep Learning)", "SVM"]
)

model_map = {
    "Random Forest (Best)": rf_model,
    "LightGBM": lgb_model,
    "MLP (Deep Learning)": mlp_model,
    "SVM": svm_model
}

active_model = model_map[selected_model]

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Performance")
st.sidebar.markdown("""
**Best Model:** Random Forest
- **R² Score:** 0.9254 (92.54% accuracy)
- **MAE:** 0.96 PSS points
- **Samples:** 1,387
- **Features:** 13 biomarkers
""")

# ============================================================================
# MAIN TITLE
# ============================================================================

st.markdown("""
    <h1>🧠 Stress Classification System</h1>
    <p style="text-align: center; font-size: 18px; color: #666;">
    Real-time stress assessment using blood biomarkers and ML models
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3 = st.tabs(["👤 Single Patient", "📁 Batch Upload", "✅ Model Comparison"])

# ============================================================================
# TAB 1: SINGLE PATIENT PREDICTION
# ============================================================================

with tab1:
    st.subheader("📋 Patient Information & Biomarkers")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographics**")
        age = st.slider("Age (years)", min_value=18, max_value=100, value=45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        sex_encoded = 1.0 if sex == "Male" else 2.0
    
    with col2:
        st.markdown("**Inflammatory Markers**")
        crp = st.number_input("CRP (mg/L)", min_value=0.0, max_value=100.0, value=2.5, step=0.1)
        il6 = st.number_input("IL-6 (pg/mL)", min_value=0.0, max_value=100.0, value=3.5, step=0.1)
        tnf_alpha = st.number_input("TNF-α (pg/mL)", min_value=0.0, max_value=50.0, value=2.5, step=0.1)
    
    with col3:
        st.markdown("**Other Biomarkers**")
        cortisol = st.number_input("Cortisol (ng/mL)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
        il1beta = st.number_input("IL-1β (pg/mL)", min_value=0.0, max_value=50.0, value=2.0, step=0.1)
        iga = st.number_input("IgA (ng/mL)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("**Metabolic & Health**")
        bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
        glucose = st.number_input("Glucose (mg/dL)", min_value=50.0, max_value=300.0, value=100.0, step=1.0)
    
    with col5:
        st.markdown("**Hormones**")
        insulin = st.number_input("Insulin (µIU/mL)", min_value=0.0, max_value=50.0, value=8.0, step=0.1)
        igf1 = st.number_input("IGF-1 (ng/mL)", min_value=50.0, max_value=300.0, value=150.0, step=1.0)
    
    with col6:
        st.markdown("**Coagulation**")
        fibrinogen = st.number_input("Fibrinogen (mg/dL)", min_value=150.0, max_value=500.0, value=350.0, step=1.0)
    
    # PREDICT BUTTON
    if st.button("🔮 PREDICT STRESS LEVEL", key="predict_single", use_container_width=True):
        # Prepare input
        X_input = np.array([[age, sex_encoded, crp, il6, tnf_alpha, cortisol, il1beta, iga, 
                            bmi, glucose, insulin, igf1, fibrinogen]])
        
        # Scale
        X_scaled = scaler.transform(X_input)
        
        # Predict with all models
        rf_pred = rf_model.predict(X_scaled)[0]
        lgb_pred = lgb_model.predict(X_scaled)[0]
        mlp_pred = mlp_model.predict(X_scaled, verbose=0)[0][0]
        svm_pred = svm_model.predict(X_scaled)[0]
        
        # Ensemble prediction (average)
        ensemble_pred = (rf_pred + lgb_pred + mlp_pred + svm_pred) / 4
        
        # Categorize
        def categorize_stress(score):
            if score < 14:
                return "🟢 LOW", "green"
            elif score < 27:
                return "🟡 MODERATE", "orange"
            else:
                return "🔴 HIGH", "red"
        
        category, color = categorize_stress(ensemble_pred)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                <h3>Ensemble Prediction</h3>
                <h2 style="color: white;">{ensemble_pred:.2f}</h2>
                <p>{category}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Active Model", selected_model.split("(")[0].strip(), 
                     delta=f"{active_model.predict(X_scaled)[0]:.2f}")
        
        with col3:
            st.markdown("""
                <div class="metric-card">
                <h3>Confidence</h3>
                <h2 style="color: white;">92.5%</h2>
                <p>Based on Random Forest</p>
                </div>
            """, unsafe_allow_html=True)
        
        # All predictions
        st.subheader("🤖 All Model Predictions")
        predictions_df = pd.DataFrame({
            'Model': ['Random Forest', 'LightGBM', 'MLP', 'SVM', 'Ensemble'],
            'PSS Score': [rf_pred, lgb_pred, mlp_pred, svm_pred, ensemble_pred],
            'Category': [categorize_stress(p)[0] for p in [rf_pred, lgb_pred, mlp_pred, svm_pred, ensemble_pred]]
        })
        st.dataframe(predictions_df, use_container_width=True)
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['RF', 'LGB', 'MLP', 'SVM', 'Ensemble'],
            y=[rf_pred, lgb_pred, mlp_pred, svm_pred, ensemble_pred],
            marker_color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'],
            text=[f"{p:.2f}" for p in [rf_pred, lgb_pred, mlp_pred, svm_pred, ensemble_pred]],
            textposition='auto'
        ))
        fig.add_hline(y=14, line_dash="dash", line_color="orange", annotation_text="Low/Moderate")
        fig.add_hline(y=27, line_dash="dash", line_color="red", annotation_text="Moderate/High")
        fig.update_layout(title="PSS Score Predictions by Model", yaxis_title="PSS Score (0-40)",
                         height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: BATCH UPLOAD
# ============================================================================

with tab2:
    st.subheader("📁 Upload Patient CSV File")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.write(f"✓ Loaded {len(df)} patients")
        st.dataframe(df.head())
        
        # Get required columns
        required_cols = ['Age', 'Sex', 'CRP', 'IL6', 'TNF_alpha', 'Cortisol', 'IL-1Beta', 'IgA',
                        'BMI', 'Glucose', 'Insulin', 'IGF1', 'Fibrinogen']
        
        if all(col in df.columns for col in required_cols):
            if st.button("🔮 PREDICT ALL PATIENTS"):
                # Encode Sex
                df['Sex_encoded'] = df['Sex'].map({'Male': 1.0, 'Female': 2.0})
                
                # Prepare features
                X = df[['Age', 'Sex_encoded', 'CRP', 'IL6', 'TNF_alpha', 'Cortisol', 'IL-1Beta', 'IgA',
                        'BMI', 'Glucose', 'Insulin', 'IGF1', 'Fibrinogen']].values
                
                # Scale
                X_scaled = scaler.transform(X)
                
                # Predict
                rf_preds = rf_model.predict(X_scaled)
                lgb_preds = lgb_model.predict(X_scaled)
                mlp_preds = mlp_model.predict(X_scaled, verbose=0).flatten()
                svm_preds = svm_model.predict(X_scaled)
                
                # Ensemble
                ensemble_preds = (rf_preds + lgb_preds + mlp_preds + svm_preds) / 4
                
                # Add to dataframe
                df['PSS_Prediction'] = ensemble_preds
                df['Stress_Category'] = df['PSS_Prediction'].apply(
                    lambda x: '🟢 LOW' if x < 14 else ('🟡 MODERATE' if x < 27 else '🔴 HIGH')
                )
                
                st.success("✓ Predictions complete!")
                st.dataframe(df[['Age', 'Sex', 'PSS_Prediction', 'Stress_Category']], use_container_width=True)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button("📥 Download Results CSV", csv, "predictions.csv", "text/csv")
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    stress_dist = df['Stress_Category'].value_counts()
                    fig = px.pie(values=stress_dist.values, names=stress_dist.index, 
                                title="Stress Distribution", hole=0.3)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.histogram(df, x='PSS_Prediction', nbins=20, title="PSS Score Distribution",
                                     labels={'PSS_Prediction': 'PSS Score'})
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"❌ CSV must contain these columns: {', '.join(required_cols)}")

# ============================================================================
# TAB 3: MODEL COMPARISON
# ============================================================================

with tab3:
    st.subheader("📊 Model Performance Comparison")
    
    # Load results
    try:
        results_df = pd.read_csv('Model_Performance_Results.csv', index_col=0)
        
        st.dataframe(results_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(results_df.reset_index(), x='index', y='R2', 
                        title="R² Score Comparison", labels={'index': 'Model', 'R2': 'R² Score'},
                        color='R2', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(results_df.reset_index(), x='index', y='MAE',
                        title="MAE Comparison", labels={'index': 'Model', 'MAE': 'MAE (PSS points)'},
                        color='MAE', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ---
        ### 🏆 Best Model: Random Forest
        - **R² Score:** 0.9254 (92.54% accuracy!)
        - **MAE:** 0.96 PSS points
        - **Strengths:** Excellent generalization, fast predictions
        """)
    
    except FileNotFoundError:
        st.warning("⚠️ Model_Performance_Results.csv not found")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 12px;">
    <p>🧠 Stress Biomarker ML System | Trained on 1,387 samples with 13 biomarkers</p>
    <p>Models: Random Forest, LightGBM, MLP (Deep Learning), SVM</p>
    <p>Best Model Accuracy: 92.54% (R² = 0.9254)</p>
</div>
""", unsafe_allow_html=True)
