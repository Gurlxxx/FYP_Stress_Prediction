"""
STRESS PREDICTION WEB APPLICATION
Based on Lee et al. 2025 + Enhanced Parameters
Using Streamlit Framework

Run with: streamlit run streamlit_stress_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Stress Classification System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #065A82;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #1C7293;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .metric-box {
        background-color: #E0F2F1;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #065A82;
    }
    .high-stress {
        background-color: #FFEBEE;
        border-left-color: #E53935;
    }
    .moderate-stress {
        background-color: #FFF3E0;
        border-left-color: #FF9800;
    }
    .low-stress {
        background-color: #E8F5E9;
        border-left-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MOCK MODELS (Replace with real trained models later)
# ============================================================================

class MockModel:
    """Mock model for demonstration - will be replaced with real trained models"""
    
    def __init__(self, name):
        self.name = name
        self.accuracy = {"Random Forest": 0.940, "LightGBM": 0.962, 
                        "LSTM": 0.974, "Transformer": 0.982}[name]
    
    def predict(self, X):
        """Mock prediction based on cortisol level"""
        cortisol = X[2] if isinstance(X, np.ndarray) else X['Cortisol']
        
        # Simple heuristic: higher cortisol = higher stress
        if cortisol < 30:
            return 0  # Low
        elif cortisol < 60:
            return 1  # Moderate
        else:
            return 2  # High
    
    def predict_proba(self, X):
        """Mock probability predictions"""
        pred = self.predict(X)
        # Create probability array with highest prob at predicted class
        proba = np.array([0.1, 0.1, 0.1])
        proba[pred] = 0.7
        proba = proba / proba.sum()
        return np.array([proba])

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_stress_color(stress_level):
    """Return color based on stress level"""
    colors = {0: "🟢 LOW", 1: "🟡 MODERATE", 2: "🔴 HIGH"}
    return colors.get(stress_level, "UNKNOWN")

def get_stress_emoji(stress_level):
    """Return emoji for stress level"""
    emojis = {0: "😊", 1: "😐", 2: "😟"}
    return emojis.get(stress_level, "❓")

def get_biomarker_status(biomarker_name, value):
    """Determine if biomarker value is normal, elevated, or high"""
    # Reference ranges (simplified)
    ranges = {
        "Cortisol": (5, 25),  # ng/mL
        "IL6": (1, 5),        # pg/mL
        "TNF_Alpha": (0.5, 3),  # pg/mL
        "CRP": (0.5, 3)       # mg/L
    }
    
    if biomarker_name in ranges:
        low, high = ranges[biomarker_name]
        if value < low:
            return "Low", "🔵"
        elif value > high:
            return "High", "🔴"
        else:
            return "Normal", "🟢"
    return "Unknown", "⚪"

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 APP SETTINGS")
selected_model = st.sidebar.selectbox(
    "Select ML Model:",
    ["Transformer", "LSTM", "LightGBM", "Random Forest"],
    help="Transformer has highest accuracy (98.2%)"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 MODEL INFO")
model_info = {
    "Transformer": {"accuracy": 98.2, "f1": 0.981, "roc_auc": 0.997},
    "LSTM": {"accuracy": 97.4, "f1": 0.973, "roc_auc": 0.994},
    "LightGBM": {"accuracy": 96.2, "f1": 0.960, "roc_auc": 0.992},
    "Random Forest": {"accuracy": 94.0, "f1": 0.935, "roc_auc": 0.980}
}

info = model_info[selected_model]
col1, col2, col3 = st.sidebar.columns(3)
col1.metric("Accuracy", f"{info['accuracy']:.1f}%")
col2.metric("F1-Score", f"{info['f1']:.2f}")
col3.metric("ROC-AUC", f"{info['roc_auc']:.3f}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 ABOUT")
st.sidebar.info(
    """
    **Stress Classification System**
    
    Based on Lee et al. 2025 - Deep Neural Network stress prediction
    
    Uses blood biomarkers + demographics for real-time stress assessment
    
    Models: Random Forest, LightGBM, LSTM, Transformer
    """
)

# ============================================================================
# MAIN TITLE
# ============================================================================
st.markdown('<div class="main-header">🧠 Stress Classification System</div>', 
            unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #666;'>Real-time psychological stress assessment using blood biomarkers and demographic parameters</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ============================================================================
# TAB LAYOUT
# ============================================================================
tab1, tab2, tab3 = st.tabs([
    "👤 Single Patient", 
    "📁 Batch Upload",
    "📈 Model Comparison"
])

# ============================================================================
# TAB 1: SINGLE PATIENT PREDICTION
# ============================================================================
with tab1:
    st.markdown('<div class="sub-header">📋 Patient Information & Biomarker Input</div>', 
                unsafe_allow_html=True)
    
    # Create columns for input form
    col1, col2, col3 = st.columns(3)
    
    # DEMOGRAPHICS
    with col1:
        st.markdown("**Demographics**")
        age = st.number_input(
            "Age (years):",
            min_value=18,
            max_value=100,
            value=45,
            help="Patient age in years"
        )
        gender = st.selectbox(
            "Gender:",
            ["Male", "Female"],
            help="Biological sex"
        )
    
    with col2:
        st.markdown("**Lifestyle & Health**")
        lifestyle = st.selectbox(
            "Lifestyle Activity Level:",
            ["Sedentary", "Moderate", "Active"],
            help="Physical activity level"
        )
        disease = st.selectbox(
            "Chronic Disease Status:",
            ["No", "Yes"],
            help="Any diagnosed chronic disease?"
        )
    
    with col3:
        st.markdown("**Social Factors**")
        caregiver = st.selectbox(
            "Caregiver Status:",
            ["No", "Yes"],
            help="Primary caregiver for family member?"
        )
        employment = st.selectbox(
            "Employment Status:",
            ["Employed", "Self-employed", "Unemployed", "Retired", "Student"],
            help="Current employment status"
        )
    
    st.markdown("---")
    
    # BIOMARKERS
    st.markdown('<div class="sub-header">🩸 Blood Biomarkers</div>', 
                unsafe_allow_html=True)
    
    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    
    with bcol1:
        cortisol = st.number_input(
            "Cortisol (ng/mL):",
            min_value=0.0,
            max_value=500.0,
            value=50.0,
            step=1.0,
            help="Cortisol level - HPA axis marker"
        )
    
    with bcol2:
        il6 = st.number_input(
            "IL-6 (pg/mL):",
            min_value=0.0,
            max_value=500.0,
            value=25.0,
            step=0.1,
            help="Interleukin-6 - pro-inflammatory cytokine"
        )
    
    with bcol3:
        tnf_alpha = st.number_input(
            "TNF-α (pg/mL):",
            min_value=0.0,
            max_value=50.0,
            value=3.5,
            step=0.1,
            help="Tumor Necrosis Factor-alpha"
        )
    
    with bcol4:
        crp = st.number_input(
            "CRP (mg/L):",
            min_value=0.0,
            max_value=20.0,
            value=2.1,
            step=0.1,
            help="C-Reactive Protein - systemic inflammation"
        )
    
    st.markdown("---")
    
    # PREDICT BUTTON
    col_button = st.columns([1, 3, 1])
    with col_button[1]:
        predict_button = st.button(
            "🔮 PREDICT STRESS LEVEL",
            key="predict_single",
            use_container_width=True
        )
    
    if predict_button:
        # Prepare data
        input_data = {
            'Age': age,
            'Gender': 1 if gender == "Female" else 0,
            'Cortisol': cortisol,
            'IL6': il6,
            'TNF_Alpha': tnf_alpha,
            'CRP': crp,
            'Lifestyle': 0 if lifestyle == "Sedentary" else (1 if lifestyle == "Moderate" else 2),
            'Disease': 1 if disease == "Yes" else 0,
            'Caregiver': 1 if caregiver == "Yes" else 0
        }
        
        # Normalize (mock normalization)
        input_array = np.array([cortisol, il6, tnf_alpha, crp, age, 
                               input_data['Gender'], input_data['Lifestyle'],
                               input_data['Disease'], input_data['Caregiver']])
        
        # Get predictions from mock model
        model = MockModel(selected_model)
        prediction = model.predict(input_array)
        proba = model.predict_proba(input_array)[0]
        
        # Display results
        st.markdown("---")
        st.markdown('<div class="sub-header">📊 PREDICTION RESULTS</div>', 
                    unsafe_allow_html=True)
        
        stress_names = ["Low Stress", "Moderate Stress", "High Stress"]
        stress_emojis = ["😊", "😐", "😟"]
        stress_colors = ["#4CAF50", "#FF9800", "#E53935"]
        
        # Main prediction box
        result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
        
        with result_col2:
            st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, {stress_colors[prediction]} 0%, {stress_colors[prediction]}22 100%);
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                border: 3px solid {stress_colors[prediction]};
            '>
                <p style='font-size: 2rem; margin: 0;'>{stress_emojis[prediction]}</p>
                <p style='font-size: 2rem; font-weight: bold; margin: 10px 0 0 0;'>{stress_names[prediction].upper()}</p>
                <p style='font-size: 1.5rem; margin: 10px 0; opacity: 0.8;'>{proba[prediction]*100:.1f}% Confidence</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model comparison
        st.markdown("**All Models' Predictions:**")
        
        model_names = ["Random Forest", "LightGBM", "LSTM", "Transformer"]
        results_data = []
        
        for model_name in model_names:
            m = MockModel(model_name)
            pred = m.predict(input_array)
            prob = m.predict_proba(input_array)[0]
            results_data.append({
                'Model': model_name,
                'Prediction': stress_names[pred],
                'Confidence': f"{prob[pred]*100:.1f}%",
                'Emoji': stress_emojis[pred]
            })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Biomarker analysis
        st.markdown('<div class="sub-header">🩸 Biomarker Analysis</div>', 
                    unsafe_allow_html=True)
        
        bio_col1, bio_col2, bio_col3, bio_col4 = st.columns(4)
        
        biomarkers = [
            ("Cortisol", cortisol, "ng/mL"),
            ("IL-6", il6, "pg/mL"),
            ("TNF-α", tnf_alpha, "pg/mL"),
            ("CRP", crp, "mg/L")
        ]
        
        for col, (name, value, unit) in zip([bio_col1, bio_col2, bio_col3, bio_col4], biomarkers):
            with col:
                status, emoji = get_biomarker_status(name, value)
                col.markdown(f"""
                <div style='
                    background-color: #f0f2f6;
                    padding: 15px;
                    border-radius: 10px;
                    text-align: center;
                '>
                    <p style='margin: 0; font-size: 1.2rem;'>{emoji}</p>
                    <p style='margin: 5px 0 0 0; font-weight: bold;'>{name}</p>
                    <p style='margin: 5px 0; font-size: 1.3rem; font-weight: bold;'>{value:.1f}</p>
                    <p style='margin: 0; font-size: 0.8rem; color: #666;'>{unit}</p>
                    <p style='margin: 5px 0 0 0; font-size: 0.8rem; color: #666;'>{status}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # SHAP-like explanation
        st.markdown('<div class="sub-header">💡 What Drives This Prediction?</div>', 
                    unsafe_allow_html=True)
        
        # Create feature importance mock visualization
        features = ['Cortisol', 'IL-6', 'TNF-α', 'CRP', 'Age', 'Lifestyle', 'Disease', 'Caregiver']
        importance = np.array([0.28, 0.19, 0.15, 0.12, 0.08, 0.06, 0.06, 0.06])
        
        fig_shap, ax = plt.subplots(figsize=(10, 6))
        colors_bar = ['#E53935' if imp > 0.15 else '#FF9800' if imp > 0.10 else '#4CAF50' for imp in importance]
        ax.barh(features, importance, color=colors_bar)
        ax.set_xlabel('Feature Importance (SHAP)', fontweight='bold')
        ax.set_title('Which Biomarkers Drove This Prediction?', fontweight='bold', fontsize=12)
        ax.set_xlim(0, 0.3)
        
        for i, v in enumerate(importance):
            ax.text(v + 0.005, i, f'{v:.2f}', va='center', fontweight='bold')
        
        st.pyplot(fig_shap)
        
        st.info(
            """
            🔍 **Interpretation:**
            - **Cortisol** (28%) - Most important: Higher cortisol → Higher stress prediction
            - **IL-6** (19%) - Second important: Pro-inflammatory marker
            - **TNF-α** (15%) - Inflammatory response
            - **Demographics** - Age, lifestyle have minimal impact compared to biomarkers
            """
        )

# ============================================================================
# TAB 2: BATCH UPLOAD
# ============================================================================
with tab2:
    st.markdown('<div class="sub-header">📁 Upload CSV File with Multiple Patients</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **CSV Format Required:**
    - Column names: Age, Gender, Cortisol, IL6, TNF_Alpha, CRP, Lifestyle, Disease, Caregiver
    - Gender: Male/Female
    - Lifestyle: Sedentary/Moderate/Active
    - Disease: No/Yes
    - Caregiver: No/Yes
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="Maximum 10MB"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df_input = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_cols = ['Age', 'Gender', 'Cortisol', 'IL6', 'TNF_Alpha', 'CRP', 
                           'Lifestyle', 'Disease', 'Caregiver']
            
            if all(col in df_input.columns for col in required_cols):
                st.success(f"✅ File loaded successfully! {len(df_input)} patients detected.")
                
                # Process predictions
                st.markdown("**Processing...**")
                progress_bar = st.progress(0)
                
                predictions_list = []
                confidences_list = []
                
                model = MockModel(selected_model)
                
                for idx, row in df_input.iterrows():
                    # Prepare input
                    cortisol = row['Cortisol']
                    il6 = row['IL6']
                    tnf_alpha = row['TNF_Alpha']
                    crp = row['CRP']
                    age = row['Age']
                    gender = 1 if row['Gender'] == "Female" else 0
                    lifestyle = 0 if row['Lifestyle'] == "Sedentary" else (1 if row['Lifestyle'] == "Moderate" else 2)
                    disease = 1 if row['Disease'] == "Yes" else 0
                    caregiver = 1 if row['Caregiver'] == "Yes" else 0
                    
                    input_array = np.array([cortisol, il6, tnf_alpha, crp, age, 
                                          gender, lifestyle, disease, caregiver])
                    
                    # Predict
                    pred = model.predict(input_array)
                    proba = model.predict_proba(input_array)[0]
                    confidence = proba[pred]
                    
                    predictions_list.append(['Low', 'Moderate', 'High'][pred])
                    confidences_list.append(f"{confidence*100:.1f}%")
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(df_input))
                
                # Add predictions to dataframe
                df_output = df_input.copy()
                df_output['Predicted_Stress'] = predictions_list
                df_output['Confidence'] = confidences_list
                
                st.markdown("---")
                st.markdown('<div class="sub-header">📊 Results Table</div>', 
                            unsafe_allow_html=True)
                
                # Display results
                st.dataframe(df_output, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.markdown('<div class="sub-header">📈 Results Summary</div>', 
                            unsafe_allow_html=True)
                
                # Statistics
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                
                total = len(df_output)
                high_count = (df_output['Predicted_Stress'] == 'High').sum()
                moderate_count = (df_output['Predicted_Stress'] == 'Moderate').sum()
                low_count = (df_output['Predicted_Stress'] == 'Low').sum()
                
                stats_col1.metric("Total Patients", total)
                stats_col2.metric("🔴 High Stress", high_count, f"{high_count/total*100:.1f}%")
                stats_col3.metric("🟡 Moderate Stress", moderate_count, f"{moderate_count/total*100:.1f}%")
                stats_col4.metric("🟢 Low Stress", low_count, f"{low_count/total*100:.1f}%")
                
                st.markdown("---")
                
                # Visualizations
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Pie chart
                    stress_counts = df_output['Predicted_Stress'].value_counts()
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=stress_counts.index,
                        values=stress_counts.values,
                        marker=dict(colors=['#E53935', '#FF9800', '#4CAF50']),
                        hole=0.3
                    )])
                    fig_pie.update_layout(
                        title="Stress Level Distribution",
                        showlegend=True,
                        height=400
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with viz_col2:
                    # Bar chart by age group
                    df_output['Age_Group'] = pd.cut(df_output['Age'], 
                                                    bins=[0, 30, 40, 50, 60, 100],
                                                    labels=['<30', '30-40', '40-50', '50-60', '60+'])
                    age_stress = pd.crosstab(df_output['Age_Group'], df_output['Predicted_Stress'])
                    
                    fig_bar = go.Figure()
                    for stress in ['Low', 'Moderate', 'High']:
                        if stress in age_stress.columns:
                            fig_bar.add_trace(go.Bar(
                                x=age_stress.index,
                                y=age_stress[stress],
                                name=stress,
                                marker=dict(color={'Low': '#4CAF50', 'Moderate': '#FF9800', 'High': '#E53935'}[stress])
                            ))
                    
                    fig_bar.update_layout(
                        title="Stress by Age Group",
                        xaxis_title="Age Group",
                        yaxis_title="Count",
                        barmode='stack',
                        height=400
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                st.markdown("---")
                
                # Download button
                csv = df_output.to_csv(index=False)
                st.download_button(
                    "📥 Download Results as CSV",
                    csv,
                    f"stress_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
                
            else:
                st.error("❌ CSV is missing required columns. Please check the format.")
                st.write("Required columns:", required_cols)
                st.write("Found columns:", list(df_input.columns))
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# ============================================================================
# TAB 3: MODEL COMPARISON
# ============================================================================
with tab3:
    st.markdown('<div class="sub-header">📊 Model Performance Comparison</div>', 
                unsafe_allow_html=True)
    
    # Performance metrics table
    model_performance = pd.DataFrame({
        'Model': ['Random Forest', 'LightGBM', 'LSTM', 'Transformer'],
        'Accuracy': [94.0, 96.2, 97.4, 98.2],
        'F1-Score': [0.935, 0.960, 0.973, 0.981],
        'ROC-AUC': [0.980, 0.992, 0.994, 0.997],
        'Precision': [0.938, 0.962, 0.975, 0.982],
        'Recall': [0.932, 0.958, 0.971, 0.980]
    })
    
    st.dataframe(model_performance, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Comparison charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Accuracy comparison
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Bar(
            x=model_performance['Model'],
            y=model_performance['Accuracy'],
            marker=dict(color=['#FF9800', '#FFC107', '#8BC34A', '#4CAF50']),
            text=model_performance['Accuracy'].round(1),
            textposition='auto',
        ))
        fig_acc.update_layout(
            title="Classification Accuracy",
            yaxis_title="Accuracy (%)",
            xaxis_title="Model",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with chart_col2:
        # ROC-AUC comparison
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Bar(
            x=model_performance['Model'],
            y=model_performance['ROC-AUC'],
            marker=dict(color=['#FF9800', '#FFC107', '#8BC34A', '#4CAF50']),
            text=model_performance['ROC-AUC'].round(3),
            textposition='auto',
        ))
        fig_roc.update_layout(
            title="ROC-AUC Score",
            yaxis_title="ROC-AUC",
            xaxis_title="Model",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    
    st.markdown("---")
    
    # Multi-metric radar chart
    st.markdown("**Comprehensive Model Comparison**")
    
    fig_radar = go.Figure()
    
    for idx, row in model_performance.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['Accuracy']/100, row['F1-Score'], row['ROC-AUC'], 
               row['Precision'], row['Recall']],
            theta=['Accuracy', 'F1-Score', 'ROC-AUC', 'Precision', 'Recall'],
            fill='toself',
            name=row['Model']
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    st.markdown("---")
    
    # Key insights
    st.markdown('<div class="sub-header">🔍 Key Insights</div>', 
                unsafe_allow_html=True)
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("""
        ✅ **Why Transformer Wins:**
        - Multi-head attention captures temporal patterns
        - Better at detecting stress accumulation over time
        - Highest accuracy (98.2%) and balanced F1-score
        - Robust across all stress levels
        """)
    
    with insight_col2:
        st.markdown("""
        📊 **Model Recommendations:**
        - **Real-time prediction**: Use Transformer
        - **Fast inference**: Use Random Forest
        - **Production**: Ensemble all 4 models
        - **Research**: Deep learning (LSTM/Transformer)
        """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>
    🧠 <strong>Stress Classification System</strong> | 
    Based on Lee et al. 2025 | 
    Mechatronics Engineering FYP
    </p>
    <p>
    For research purposes only. Not a substitute for professional medical advice.
    </p>
</div>
""", unsafe_allow_html=True)
