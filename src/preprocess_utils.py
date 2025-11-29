import pandas as pd
import numpy as np
import os

def load_data(path = "cardio_train.csv"):
    """Load data from specified path of file already within the directory leave empty"""
    df = pd.read_csv(path, sep = ";")
    return df
def set_id_index(df):
    """Set 'id' column as the DataFrame index if it exists."""
    df = df.set_index("id")
    return df

def clean_data(df):
    """Drops null values and duplicates"""
    df = df.dropna()
    df = df.drop_duplicates()
    return df
    

def encode_gender(df):
    """Encode Gender into 0 for Female 1 for Male"""
    df["gender"] = df["gender"].astype(int).map({1: 0, 2: 1})
    return df

def lifestyle_risk(data, alco=None, active=None):
    """
    Generate 'lifestyle_risk' feature.
    Works for both:
      - Full DataFrames (training)
      - Single numeric inputs (Streamlit)
    """
    if isinstance(data, pd.DataFrame):
        data["lifestyle_risk"] = (2 * data["smoke"]) + data["alco"] + (1 - data["active"])
        data.drop(["alco", "active", "smoke"], axis=1, inplace=True)
        return data
    elif all(isinstance(x, (int, float)) for x in [data, alco, active]):
        lifestyle_score = (2 * data) + alco + (1 - active)
        return lifestyle_score

    elif isinstance(data, dict):
        smoke = data.get("smoke", 0)
        alco = data.get("alco", 0)
        active = data.get("active", 1)
        lifestyle_score = (2 * smoke) + alco + (1 - active)
        return lifestyle_score

    else:
        raise ValueError("Unsupported input type for lifestyle_risk")


def encode_ap_status(data):
    """
    Encode ap_status into numerical categories.
    Handles both DataFrames (for training) and scalars (for single prediction).
    """

    status_map = {
        "Hypotension": 0,
        "Normal": 1,
        "Elevated": 2,
        "Stage 1 Hypertension": 3,
        "Stage 2 Hypertension": 4,
        "Severe Hypertension": 5
    }

    def ap_status_single(row):
        if row["ap_hi"] < 90 and row["ap_lo"] < 60:
            return "Hypotension"
        elif row["ap_hi"] < 120 and row["ap_lo"] < 80:
            return "Normal"
        elif 120 <= row["ap_hi"] <= 129 and row["ap_lo"] < 80:
            return "Elevated"
        elif (130 <= row["ap_hi"] <= 139) or (80 <= row["ap_lo"] <= 89):
            return "Stage 1 Hypertension"
        elif row["ap_hi"] >= 180 or row["ap_lo"] >= 120:
            return "Severe Hypertension"
        elif row["ap_hi"] >= 140 or row["ap_lo"] >= 90:
            return "Stage 2 Hypertension"
        else:
            return "Uncategorized"

    import pandas as pd
    if isinstance(data, pd.DataFrame):
        data["ap_status"] = data.apply(ap_status_single, axis=1).map(status_map)
        return data
    
    elif isinstance(data, (tuple, list)):
        ap_hi, ap_lo = data
        row = {"ap_hi": ap_hi, "ap_lo": ap_lo}
        label = ap_status_single(row)
        return status_map.get(label, -1)

    elif isinstance(data, dict):
        label = ap_status_single(data)
        return status_map.get(label, -1)

    else:
        raise ValueError("Unsupported input type for encode_ap_status")


def IQR(series):
    """Replace outliers using IQR rule (values outside 1.5 × IQR are clipped)."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    return series.clip(lower, upper)



def impute_outliers(df,features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo','cholesterol', 'gluc']):
    """Impute outliers using IQR within each cardio category."""
    
    exclude = ['gluc', 'lifstyle_risk', 'cholesterol']

    for feature in features:
        if feature not in exclude:
            for category in df["cardio"].unique():
                df.loc[df["cardio"] == category, feature] = IQR(df.loc[df["cardio"] == category, feature])

    return df

def add_bmi(data=None, weight=None, height=None):
    """
    Compute BMI (Body Mass Index).
    Works for both:
      - Full DataFrames (training)
      - Single numeric inputs (Streamlit)
    """

    import pandas as pd

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)
        return df

    elif weight is not None and height is not None and all(isinstance(x, (int, float)) for x in [weight, height]):
        bmi_value = weight / ((height / 100) ** 2)
        return bmi_value

    else:
        raise ValueError("Unsupported input type for add_bmi. Expected a DataFrame or numeric weight/height.")



def add_pulse_pressure(ap_hi, ap_lo=None):
    """
    Generate 'pulse_pressure' feature.
    Works for both:
      - Full DataFrames (training)
      - Single numeric inputs (Streamlit)
    """

    if isinstance(ap_hi, pd.DataFrame):
        df = ap_hi.copy()
        df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
        return df

    elif isinstance(ap_hi, (int, float)) and isinstance(ap_lo, (int, float)):
        pulse_pressure = ap_hi - ap_lo
        return pulse_pressure

    elif isinstance(ap_hi, dict):
        hi = ap_hi.get("ap_hi")
        lo = ap_hi.get("ap_lo")
        if hi is not None and lo is not None:
            return hi - lo
        else:
            raise ValueError("Dictionary input must include 'ap_hi' and 'ap_lo' keys.")

    else:
        raise ValueError("Unsupported input type for add_pulse")


def clean_partial_duplicates(df):
    """Remove partial duplicates across main biometric columns."""
    partial_duplicates = df[df.duplicated(subset=['age','gender','height', 'weight','ap_hi','ap_lo','cholesterol','lifestyle_risk'])]
    df.drop(partial_duplicates.index,inplace = True)
    return df

def clean_bp(df):
    """Remove rows where ap_hi <= ap_lo (invalid blood pressure readings)."""
    df = df[df["ap_hi"] > df["ap_lo"]]
    return df



def save_data(df, path="data/cleaned_data.csv", index=True):
    """Save DataFrame to a CSV file safely."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)
    print(f"Data saved successfully at: {path}")
