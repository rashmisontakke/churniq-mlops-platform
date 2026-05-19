import shap
import pickle
import numpy as np
import pandas as pd

def get_shap_values(input_df: pd.DataFrame):
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    return shap_values, explainer.expected_value

def get_top_features(input_df: pd.DataFrame, n=5):
    shap_vals, _ = get_shap_values(input_df)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    feature_impact = pd.Series(
        np.abs(shap_vals[0]), index=input_df.columns
    ).sort_values(ascending=False).head(n)
    return feature_impact