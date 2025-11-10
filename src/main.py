import pandas as pd
from preprocess_utils import (
    load_data,
    set_id_index,
    clean_data,
    clean_bp,
    clean_partial_duplicates,
    encode_gender,
    lifestyle_risk,
    encode_ap_status,
    add_bmi,
    add_pulse_pressure,
    impute_outliers,
    save_data
)
from visualization_utils import (
    plot_feature_distribution,
    plot_correlation_heatmap,
    plot_target_relationships,
    plot_categorical_counts,
    plot_age_group_distribution
)
from model_utils import train_stacked_ensemble_pipeline, save_model

df = load_data("D:\DEPI\Final Project\Milestone 4\data\cardio_train.csv")

print(f"Data loaded successfully. Shape: {df.shape}")

print("🧹 Running full preprocessing pipeline...")

df = clean_data(df)

df = encode_gender(df)

df = lifestyle_risk(df)

df = encode_ap_status(df)


df = impute_outliers(df)

df = set_id_index(df)

df = clean_partial_duplicates(df)

df = clean_bp(df)

df = add_bmi(df)

df = add_pulse_pressure(df)






save_data(df, "data/cleaned_data.csv")

print("Data cleaning and feature engineering complete.")
print(f"Final shape after preprocessing: {df.shape}\n")



# EDA Summary

# print("Running exploratory data analysis...")
# for fig in plot_feature_distribution(df, ["age", "height", "weight", "BMI"]):
#     fig.show()

# plot_correlation_heatmap(df).show()
# for fig in plot_target_relationships(df, "cardio", ["age", "BMI", "ap_hi", "ap_lo"]):
#     fig.show()
# for fig in plot_categorical_counts(df, ["cholesterol", "gluc", "gender", "lifestyle_risk", "ap_status"]):
#     fig.show()
# plot_age_group_distribution(df).show()
# print("✅ EDA complete.\n")



print("Training Stacked Ensemble pipeline with preprocessing...")
pipe, ensemble_metrics = train_stacked_ensemble_pipeline(df)
print(ensemble_metrics)


save_model(pipe, "models/ensemble_pipeline.pkl")

print("\nTraining complete. Model pipeline saved successfully.")

