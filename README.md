# Binary-Classification-CardioVascular-Disease-Prediction
# 🫀 Cardiovascular Disease Prediction Dashboard
A machine learning project that predicts cardiovascular disease risk using a **Stacked Ensemble Model**, an interactive **Streamlit Dashboard**, a **Flask API**, and **MLflow** for experiment tracking.

---

## 📸 Screenshots

### MLflow Experiments
<img width="1910" height="769" alt="image" src="https://github.com/user-attachments/assets/832b9a41-f33e-43d0-b85e-106cd91de644" />

<img width="1918" height="855" alt="image" src="https://github.com/user-attachments/assets/5b5efa30-178c-4636-997c-416c025c38da" />


### Live Prediction (Streamlit UI)
<img width="1869" height="816" alt="image" src="https://github.com/user-attachments/assets/b70ef631-1018-431f-8feb-83f35647cdfe" />


## 📁 Project Structure


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
