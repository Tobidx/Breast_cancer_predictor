#!/usr/bin/env python
# coding: utf-8

# In[6]:


# app.py
import gradio as gr
import joblib
import json
import numpy as np

# Load the model and scaler
model = joblib.load('xgboost_breast_cancer__model.joblib')
scaler = joblib.load('scalerr.joblib')

# Load feature names
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

def predict_cancer(*features):
    # Convert inputs to numpy array
    input_data = np.array(features).reshape(1, -1)
    
    # Scale the input data
    scaled_input = scaler.transform(input_data)
    
    # Make prediction
    prediction_proba = model.predict_proba(scaled_input)[0, 1]
    
    # Apply threshold
    prediction = "Malignant" if prediction_proba >= 0.4 else "Benign"
    
    return f"Prediction: {prediction}\nProbability of being malignant: {prediction_proba:.2f}"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_cancer,
    inputs=[gr.Number(label=name) for name in feature_names],
    outputs="text",
    title="Breast Cancer Prediction",
    description="Enter the feature values to predict whether a breast mass is benign or malignant."
)

iface.launch()


# In[ ]:




