import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from preprocess_utils import (
    load_data, clean_data, encode_gender, lifestyle_risk,
    encode_ap_status, add_bmi, add_pulse_pressure
)
from visualization_utils import (
    z_test_and_effect, chi_square_and_cramers_v,
    plot_cohens_d, plot_cramers_v, plot_age_group_distribution
)

st.set_page_config(
    page_title="🫀 Cardiovascular Health Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    st.write("### Dataset Summary")
    st.dataframe(df.describe().T)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{len(df):,}")
    col2.metric("Features", f"{len(df.columns)}")
    col3.metric("CVD Cases", f"{df['cardio'].sum():,}")
    col4.metric("CVD Rate", f"{df['cardio'].mean()*100:.1f}%")
    fig = px.pie(df, names="cardio", title="Cardiovascular Disease Distribution", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

elif section == "📊 Exploratory Data Analysis (EDA)":
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
            st.plotly_chart(px.bar(gender_counts, x='gender', y='count', color='cardio', barmode='group'), use_container_width=True)
        with col2:
            chol_counts = df.groupby(['cholesterol', 'cardio']).size().reset_index(name='count')
            st.plotly_chart(px.bar(chol_counts, x='cholesterol', y='count', color='cardio', barmode='group'), use_container_width=True)
        with col3:
            gluc_counts = df.groupby(['gluc', 'cardio']).size().reset_index(name='count')
            st.plotly_chart(px.bar(gluc_counts, x='gluc', y='count', color='cardio', barmode='group'), use_container_width=True)
        with col4:
            lifestyle_counts = df.groupby(['lifestyle_risk', 'cardio']).size().reset_index(name='count')
            st.plotly_chart(px.bar(lifestyle_counts, x='lifestyle_risk', y='count', color='cardio', barmode='group'), use_container_width=True)

    with tab2:
        bp_map = {0: 'Hypotension', 1: 'Normal', 2: 'Elevated', 3: 'Stage 1 HTN', 4: 'Stage 2 HTN', 5: 'Severe HTN'}
        df['bp_status_label'] = df['ap_status'].map(bp_map)
        col1, col2 = st.columns(2)
        with col1:
            bp_cardio = df.groupby(['bp_status_label', 'cardio']).size().reset_index(name='count')
            st.plotly_chart(px.bar(bp_cardio, x='bp_status_label', y='count', color='cardio', barmode='group'), use_container_width=True)
        with col2:
            bp_rate = df.groupby('bp_status_label')['cardio'].agg(['sum', 'count']).reset_index()
            bp_rate['rate'] = (bp_rate['sum'] / bp_rate['count']) * 100
            st.plotly_chart(px.bar(bp_rate, x='bp_status_label', y='rate', text='rate'), use_container_width=True)
        df['bmi_category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        col3, col4 = st.columns(2)
        with col3:
            bmi_cardio = df.groupby(['bmi_category', 'cardio']).size().reset_index(name='count')
            st.plotly_chart(px.bar(bmi_cardio, x='bmi_category', y='count', color='cardio', barmode='group'), use_container_width=True)
        with col4:
            bmi_rate = df.groupby('bmi_category')['cardio'].agg(['sum', 'count']).reset_index()
            bmi_rate['rate'] = (bmi_rate['sum'] / bmi_rate['count']) * 100
            st.plotly_chart(px.bar(bmi_rate, x='bmi_category', y='rate', text='rate'), use_container_width=True)

    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(px.box(df, x='cardio', y='age_years', color='cardio'), use_container_width=True)
        with col2:
            st.plotly_chart(px.box(df, x='cardio', y='BMI', color='cardio'), use_container_width=True)
        with col3:
            st.plotly_chart(px.box(df, x='cardio', y='pulse_pressure', color='cardio'), use_container_width=True)

    with tab4:
        corr_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'BMI', 'pulse_pressure', 'lifestyle_risk', 'ap_status', 'cardio']
        corr_matrix = df[corr_cols].corr()
        st.plotly_chart(px.imshow(corr_matrix, text_auto='.2f', aspect='auto'), use_container_width=True)

    with tab5:
        st.pyplot(plot_age_group_distribution(df))

    with tab6:
        class_counts = df['cardio'].value_counts().reset_index()
        class_counts.columns = ['CVD Status', 'Count']
        class_counts['CVD Status'] = class_counts['CVD Status'].map({0: 'Negative', 1: 'Positive'})
        st.plotly_chart(px.pie(class_counts, values='Count', names='CVD Status', hole=0.4), use_container_width=True)
        summary_cols = ['age_years', 'BMI', 'ap_hi', 'ap_lo', 'pulse_pressure', 'weight', 'height']
        summary = df.groupby('cardio')[summary_cols].agg(['mean', 'std', 'median']).round(2)
        summary.columns = [f'{col}_{stat}' for col, stat in summary.columns]
        summary.index = ['No CVD (0)', 'CVD (1)']
        st.dataframe(summary.T, use_container_width=True)

elif section == "📈 Statistical Analysis & Feature Importance":
    numeric_features = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'BMI', 'pulse_pressure']
    cat_features = ['gender', 'cholesterol', 'gluc', 'ap_status', 'lifestyle_risk']
    effect_df = z_test_and_effect(df, numeric_features)
    st.dataframe(effect_df)
    st.plotly_chart(plot_cohens_d(effect_df), use_container_width=True)
    cat_df = chi_square_and_cramers_v(df, cat_features)
    st.dataframe(cat_df)
    st.plotly_chart(plot_cramers_v(cat_df), use_container_width=True)

elif section == "🤖 Model Performance Summary":
    try:
        metrics_df = pd.read_csv("models/Models_Metrics.csv")
        st.dataframe(metrics_df)
        subset = metrics_df[metrics_df["Dataset"] == "Test"]
        st.plotly_chart(px.bar(subset, x="Model", y="Accuracy", color="F1 Score"), use_container_width=True)
    except FileNotFoundError:
        st.warning("Models_Metrics.csv not found.")

elif section == "💓 Live Prediction":
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age (years)", 30, 80, 50)
        gender = st.selectbox("Gender", ["Female", "Male"])
        height = st.number_input("Height (cm)", 140, 200, 170)
        weight = st.number_input("Weight (kg)", 40, 150, 70)
        ap_hi = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
        ap_lo = st.number_input("Diastolic BP (mmHg)", 40, 130, 80)

    with col2:
        cholesterol_label = st.selectbox("Cholesterol Level",
            ["Normal", "Above Normal", "Well Above Normal"]
        )
        cholesterol = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[cholesterol_label]

        gluc_label = st.selectbox("Glucose Level",
            ["Normal", "Above Normal", "Well Above Normal"]
        )
        gluc = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[gluc_label]

        smoke = 1 if st.radio("Smokes?", ["No", "Yes"]) == "Yes" else 0
        alco = 1 if st.radio("Consumes Alcohol?", ["No", "Yes"]) == "Yes" else 0
        active = 1 if st.radio("Physically Active?", ["No", "Yes"]) == "Yes" else 0

    input_data = {
        "age": age,
        "gender": 1 if gender == "Male" else 0,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active
    }

    st.subheader("Input Data Preview")
    preview_df = pd.DataFrame([input_data])
    st.dataframe(preview_df)

    if st.button("🔍 Predict Risk"):
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
            result = response.json()

            pred = result["prediction"]
            prob = result["probability"]

            st.markdown("---")

            if pred == 1:
                st.error(f"⚠️ High Risk — Probability: {prob:.2f}")
            else:
                st.success(f"✅ Low Risk — Probability: {prob:.2f}")

        except Exception as e:
            st.error("Could not connect to the Flask API. Make sure it is running.")
            st.write(e)

