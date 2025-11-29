import streamlit as st
import pandas as pd
import plotly.express as px
from preprocess_utils import (
    load_data, clean_data, encode_gender, lifestyle_risk,
    encode_ap_status, add_bmi, add_pulse_pressure
)
from model_utils import load_model, predict_with_pipeline
from visualization_utils import (
    z_test_and_effect, chi_square_and_cramers_v,
    plot_cohens_d, plot_cramers_v, plot_age_group_distribution
)

st.set_page_config(
    page_title="🫀 Cardiovascular Health Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #e63946;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    df = load_data("data/cardio_train.csv")
    df = clean_data(df)
    df = encode_gender(df)
    df = encode_ap_status(df)
    df = add_bmi(df)
    df = add_pulse_pressure(df)
    df = lifestyle_risk(df)
    df["age_years"] = (df["age"] / 365).round(1)
    return df

df = load_and_preprocess_data()

st.sidebar.title("🩺 Cardiovascular Dashboard")
section = st.sidebar.radio(
    "Navigate to Section:",
    [
        "📘 Overview & Cleaning Report",
        "📊 Exploratory Data Analysis (EDA)",
        "📈 Statistical Analysis & Feature Importance",
        "🤖 Model Performance Summary",
        "💓 Live Prediction"
    ]
)

if section == "📘 Overview & Cleaning Report":
    st.markdown('<div class="main-header">📘 Data Overview & Cleaning Summary</div>', unsafe_allow_html=True)
    st.write("### Dataset Summary")
    st.dataframe(df.describe().T)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{len(df):,}")
    col2.metric("Features", f"{len(df.columns)}")
    col3.metric("CVD Cases", f"{df['cardio'].sum():,}")
    col4.metric("CVD Rate", f"{df['cardio'].mean()*100:.1f}%")
    st.subheader("Target Distribution")
    fig = px.pie(
        df, names="cardio",
        title="Cardiovascular Disease Distribution",
        color="cardio",
        color_discrete_map={0: "#3498db", 1: "#e74c3c"},
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

elif section == "📊 Exploratory Data Analysis (EDA)":
    st.markdown('<div class="main-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧍‍♂️ Demographics & Lifestyle",
        "🩺 Blood Pressure & BMI",
        "📦 Feature Distributions by CVD",
        "🔗 Correlations",
        "📊 Age Groups",
        "📋 Summary & Balance"
    ])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            gender_counts = df.groupby(['gender', 'cardio']).size().reset_index(name='count')
            gender_counts['gender'] = gender_counts['gender'].map({0: 'Female', 1: 'Male'})
            fig_gender = px.bar(
                gender_counts, x='gender', y='count', color='cardio',
                title='Gender Distribution', barmode='group', text='count',
                color_discrete_map={0: '#3498db', 1: '#e74c3c'}
            )
            fig_gender.update_traces(textposition='outside')
            st.plotly_chart(fig_gender, use_container_width=True)
        with col2:
            chol_counts = df.groupby(['cholesterol', 'cardio']).size().reset_index(name='count')
            fig_chol = px.bar(
                chol_counts, x='cholesterol', y='count', color='cardio',
                title='Cholesterol Levels', barmode='group', text='count',
                color_discrete_map={0: '#3498db', 1: '#e74c3c'}
            )
            fig_chol.update_traces(textposition='outside')
            st.plotly_chart(fig_chol, use_container_width=True)
        with col3:
            gluc_counts = df.groupby(['gluc', 'cardio']).size().reset_index(name='count')
            fig_gluc = px.bar(
                gluc_counts, x='gluc', y='count', color='cardio',
                title='Glucose Levels', barmode='group', text='count',
                color_discrete_map={0: '#3498db', 1: '#e74c3c'}
            )
            fig_gluc.update_traces(textposition='outside')
            st.plotly_chart(fig_gluc, use_container_width=True)
        with col4:
            lifestyle_counts = df.groupby(['lifestyle_risk', 'cardio']).size().reset_index(name='count')
            fig_lifestyle = px.bar(
                lifestyle_counts, x='lifestyle_risk', y='count', color='cardio',
                title='Lifestyle Risk Score', barmode='group', text='count',
                color_discrete_map={0: '#3498db', 1: '#e74c3c'}
            )
            fig_lifestyle.update_traces(textposition='outside')
            st.plotly_chart(fig_lifestyle, use_container_width=True)

    with tab2:
        bp_map = {0: 'Hypotension', 1: 'Normal', 2: 'Elevated', 3: 'Stage 1 HTN', 4: 'Stage 2 HTN', 5: 'Severe HTN'}
        df['bp_status_label'] = df['ap_status'].map(bp_map)
        col1, col2 = st.columns(2)
        with col1:
            bp_cardio = df.groupby(['bp_status_label', 'cardio']).size().reset_index(name='count')
            fig_bp_cardio = px.bar(
                bp_cardio, x='bp_status_label', y='count', color='cardio',
                title='CVD Cases by BP Status', barmode='group',
                color_discrete_map={0: '#3498db', 1: '#e74c3c'}
            )
            st.plotly_chart(fig_bp_cardio, use_container_width=True)
        with col2:
            bp_rate = df.groupby('bp_status_label')['cardio'].agg(['sum', 'count']).reset_index()
            bp_rate['rate'] = (bp_rate['sum'] / bp_rate['count']) * 100
            fig_bp_rate = px.bar(
                bp_rate, x='bp_status_label', y='rate', text='rate', color='rate',
                title='CVD Rate by BP Status (%)', color_continuous_scale='Reds'
            )
            fig_bp_rate.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig_bp_rate, use_container_width=True)
        df['bmi_category'] = pd.cut(
            df['BMI'], bins=[0, 18.5, 25, 30, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
        col3, col4 = st.columns(2)
        with col3:
            bmi_cardio = df.groupby(['bmi_category', 'cardio']).size().reset_index(name='count')
            fig_bmi_cardio = px.bar(
                bmi_cardio, x='bmi_category', y='count', color='cardio',
                title='CVD Cases by BMI Category', barmode='group',
                color_discrete_map={0: '#3498db', 1: '#e74c3c'}
            )
            st.plotly_chart(fig_bmi_cardio, use_container_width=True)
        with col4:
            bmi_rate = df.groupby('bmi_category')['cardio'].agg(['sum', 'count']).reset_index()
            bmi_rate['rate'] = (bmi_rate['sum'] / bmi_rate['count']) * 100
            fig_bmi_rate = px.bar(
                bmi_rate, x='bmi_category', y='rate', text='rate', color='rate',
                title='CVD Rate by BMI Category (%)', color_continuous_scale='Oranges'
            )
            fig_bmi_rate.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig_bmi_rate, use_container_width=True)

    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_age = px.box(df, x='cardio', y='age_years', color='cardio', title='Age Distribution by CVD', color_discrete_map={0: '#3498db', 1: '#e74c3c'})
            st.plotly_chart(fig_age, use_container_width=True)
        with col2:
            fig_bmi = px.box(df, x='cardio', y='BMI', color='cardio', title='BMI Distribution by CVD', color_discrete_map={0: '#3498db', 1: '#e74c3c'})
            st.plotly_chart(fig_bmi, use_container_width=True)
        with col3:
            fig_pp = px.box(df, x='cardio', y='pulse_pressure', color='cardio', title='Pulse Pressure by CVD', color_discrete_map={0: '#3498db', 1: '#e74c3c'})
            st.plotly_chart(fig_pp, use_container_width=True)

    with tab4:
        corr_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'BMI', 'pulse_pressure', 'lifestyle_risk', 'ap_status', 'cardio']
        corr_matrix = df[corr_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto='.2f', aspect='auto', title='Correlation Matrix - All Features', color_continuous_scale='RdBu_r', zmin=-1, zmax=1, height=700)
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab5:
        fig_age_group = plot_age_group_distribution(df)
        st.pyplot(fig_age_group)

    with tab6:
        col1, col2 = st.columns(2)
        with col1:
            class_counts = df['cardio'].value_counts().reset_index()
            class_counts.columns = ['CVD Status', 'Count']
            class_counts['CVD Status'] = class_counts['CVD Status'].map({0: 'Negative', 1: 'Positive'})
            fig_balance = px.pie(
                class_counts, values='Count', names='CVD Status',
                title='CVD Class Distribution',
                color='CVD Status',
                color_discrete_map={'Negative': '#3498db', 'Positive': '#e74c3c'},
                hole=0.4
            )
            st.plotly_chart(fig_balance, use_container_width=True)
        with col2:
            summary_cols = ['age_years', 'BMI', 'ap_hi', 'ap_lo', 'pulse_pressure', 'weight', 'height']
            summary = df.groupby('cardio')[summary_cols].agg(['mean', 'std', 'median']).round(2)
            summary.columns = [f'{col}_{stat}' for col, stat in summary.columns]
            summary.index = ['No CVD (0)', 'CVD (1)']
            st.dataframe(summary.T, use_container_width=True)

elif section == "📈 Statistical Analysis & Feature Importance":
    st.markdown('<div class="main-header">📈 Statistical Tests & Feature Importance</div>', unsafe_allow_html=True)
    numeric_features = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'BMI', 'pulse_pressure']
    cat_features = ['gender', 'cholesterol', 'gluc', 'ap_status', 'lifestyle_risk']
    st.subheader("Numerical Feature Analysis (Z-Test & Cohen's d)")
    effect_df = z_test_and_effect(df, numeric_features)
    st.dataframe(effect_df)
    st.plotly_chart(plot_cohens_d(effect_df), use_container_width=True)
    st.subheader("Categorical Feature Analysis (Chi-Square & Cramer's V)")
    cat_df = chi_square_and_cramers_v(df, cat_features)
    st.dataframe(cat_df)
    st.plotly_chart(plot_cramers_v(cat_df), use_container_width=True)

elif section == "🤖 Model Performance Summary":
    st.markdown('<div class="main-header">🤖 Model Performance Summary</div>', unsafe_allow_html=True)
    st.info("This section summarizes model metrics from your training results.")
    try:
        metrics_df = pd.read_csv("models/Models_Metrics.csv")
        st.dataframe(metrics_df)
        fig = px.bar(
            metrics_df[metrics_df["Dataset"] == "Test"],
            x="Model", y="Accuracy", color="F1 Score",
            title="Model Accuracy vs F1 Score (Test Set)",
            color_continuous_scale="RdYlGn"
        )
        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.warning("Models_Metrics.csv not found. Please place it in the app directory.")

elif section == "💓 Live Prediction":
    st.markdown('<div class="main-header">💓 Live Cardiovascular Risk Prediction</div>', unsafe_allow_html=True)

    @st.cache_resource
    def load_pipeline():
        return load_model("models/ensemble_pipeline.pkl")

    pipe = load_pipeline()
    st.success("✅ Model pipeline loaded successfully!")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age (years)", 30, 80, 50)
        gender = st.selectbox("Gender", ["Female", "Male"])
        height = st.number_input("Height (cm)", 140, 200, 170)
        weight = st.number_input("Weight (kg)", 40, 150, 70)
        ap_hi = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
        ap_lo = st.number_input("Diastolic BP (mmHg)", 40, 130, 80)
    with col2:
        cholesterol_label = st.selectbox(
            "Cholesterol Level",
            ["Normal", "Above Normal", "Well Above Normal"]
        )
        cholesterol = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[cholesterol_label]

        gluc_label = st.selectbox(
            "Glucose Level",
            ["Normal", "Above Normal", "Well Above Normal"]
        )
        gluc = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[gluc_label]

        smoke = 1 if st.radio("Smokes?", ["No", "Yes"]) == "Yes" else 0
        alco = 1 if st.radio("Consumes Alcohol?", ["No", "Yes"]) == "Yes" else 0
        active = 1 if st.radio("Physically Active?", ["No", "Yes"]) == "Yes" else 0


    bmi = add_bmi(weight=weight, height=height)
    pulse_pressure = add_pulse_pressure(ap_hi, ap_lo)
    ap_status = encode_ap_status([ap_hi, ap_lo])
    lifestyle_r = lifestyle_risk(smoke, alco, active)

    input_df = pd.DataFrame({
        "age": [age * 365],
        "gender": [1 if gender == "Male" else 0],
        "height": [height],
        "weight": [weight],
        "ap_hi": [ap_hi],
        "ap_lo": [ap_lo],
        "cholesterol": [cholesterol],
        "gluc": [gluc],
        "lifestyle_risk": [lifestyle_r],
        "BMI": [bmi],
        "pulse_pressure": [pulse_pressure],
        "ap_status": [ap_status]
    })

    st.subheader("🧾 Input Data Preview")
    st.dataframe(input_df)

    if st.button("🔍 Predict Risk"):
        pred, prob = predict_with_pipeline(pipe, input_df)
        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ **High Risk** — Probability: {prob:.2f}")
        else:
            st.success(f"✅ **Low Risk** — Probability: {prob:.2f}")
        st.info("⚕️ Note: For research use only — not a medical diagnosis.")
