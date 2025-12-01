# Binary-Classification-CardioVascular-Disease-Prediction
# Cardiovascular Disease Prediction Dashboard
A Machine learning project that predicts cardiovascular disease risk using a **Stacked Ensemble Model**, an interactive **Streamlit Dashboard**, a **Flask API**, and **MLflow** for experiment tracking.

---

## 📸 Screenshots

### MLflow Experiments
<img width="1910" height="769" alt="image" src="https://github.com/user-attachments/assets/832b9a41-f33e-43d0-b85e-106cd91de644" />

<img width="1918" height="855" alt="image" src="https://github.com/user-attachments/assets/5b5efa30-178c-4636-997c-416c025c38da" />


### Live Prediction (Streamlit UI)
<img width="1869" height="816" alt="image" src="https://github.com/user-attachments/assets/b70ef631-1018-431f-8feb-83f35647cdfe" />


---

## 🚀 Features

### ✔ Streamlit Dashboard
- Data exploration (EDA)
- Feature importance & statistical visualizations
- Model performance overview
- **Live prediction form**
- CSV-style preview of input data
- Sends JSON to the Flask API

### ✔ Flask API
- Receives model input as JSON
- Reconstructs engineered features:
  - BMI  
  - Pulse Pressure  
  - Blood Pressure Stage  
  - Lifestyle Risk  
  - Age in days  
- Returns prediction + probability
- Optional MLflow logging for API usage

### ✔ MLflow Tracking
Tracks:
- Training experiments  
- Streamlit live predictions  
- Flask API predictions  

---

## 🧪 How to Run the Project

### 1️⃣ Install Dependencies
####  run pip install -r requirements.txt

### 2️⃣ Start the Flask API
#### run python app.py

### 3️⃣ Run the Streamlit Dashboard
#### streamlit run Dashboard.py


---

## 🔧 How Prediction Works

1. User enters health data into Streamlit.
2. Streamlit shows a CSV-style preview of the input.
3. Streamlit sends the input as **JSON** to the Flask `/predict` endpoint.
4. Flask performs feature engineering using:
   - `add_bmi`
   - `add_pulse_pressure`
   - `encode_ap_status`
   - `lifestyle_risk`
5. Model predicts:
   - `prediction`: 0 = Low Risk, 1 = High Risk  
   - `probability`: model confidence  
6. Streamlit displays the result to the user.

---

## 🧠 Model Overview

- **Stacked Ensemble model**
  - XGBoost
  - CatBoost
  - LGBM
  - XGBoost (meta-learner)
- **Feature Engineering**
  - BMI  
  - Pulse pressure  
  - Blood pressure stage encoding  
  - Lifestyle risk score  
  - Age converted from years → days  

## 👥 Contributors

- [Omar Nashat](https://github.com/OmarNashat1124)
- [Rawan Khalid](https://github.com/RawanElOlemy)
- [Mazen Mohamed](https://github.com/mazenmuhamed14)
- [Felopater Ashraf](https://github.com/FelopaterAshraf)
- [Shahd Sayed](https://github.com/ShahdSayed4)
- [Jana Ashraf](https://github.com/janaalkot)
