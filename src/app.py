from flask import Flask, request, jsonify
import pandas as pd
import mlflow
from model_utils import load_model
from preprocess_utils import (
    add_bmi, add_pulse_pressure, encode_ap_status, lifestyle_risk
)

app = Flask(__name__)

pipe = load_model("models/ensemble_pipeline.pkl")

@app.post("/predict")
def predict():
    data = request.get_json()

    df = pd.DataFrame([{
        "age": data["age"] * 365,
        "gender": data["gender"],
        "height": data["height"],
        "weight": data["weight"],
        "ap_hi": data["ap_hi"],
        "ap_lo": data["ap_lo"],
        "cholesterol": data["cholesterol"],
        "gluc": data["gluc"],
        "lifestyle_risk": lifestyle_risk(data["smoke"], data["alco"], data["active"]),
        "BMI": add_bmi(weight=data["weight"], height=data["height"]),
        "pulse_pressure": add_pulse_pressure(data["ap_hi"], data["ap_lo"]),
        "ap_status": encode_ap_status([data["ap_hi"], data["ap_lo"]])
    }])

    pred = pipe.predict(df)[0]
    prob = pipe.predict_proba(df)[0][1]

    mlflow.set_experiment("Flask_API_Predictions")
    with mlflow.start_run(run_name="API_Prediction", nested=True):
        mlflow.log_param("input", data)
        mlflow.log_metric("prediction", int(pred))
        mlflow.log_metric("probability", float(prob))

    return jsonify({
        "prediction": int(pred),
        "probability": float(prob)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
