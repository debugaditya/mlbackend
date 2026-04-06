from fastapi import FastAPI
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import pickle
import keras

# Intercepts version mismatches between Colab and your local PC
class SafeDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains to access your API
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],  # Allows all headers
)
# Load everything once when the server starts
model = keras.models.load_model("model.h5", custom_objects={'Dense': SafeDense})
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("features.pkl", "rb"))

@app.post("/predict")
def predict(data: dict):
    # THE FIX: Force floats so pandas creates "Chest_Pain_1.0" instead of "Chest_Pain_1"
    df = pd.DataFrame([data], dtype=float)
    
    # Find which columns were categorical in training
    categorical_cols = [col for col in df.columns if col not in feature_names]
    
    # Force pandas to one-hot encode them
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols)
        
    # Align columns exactly to match the model's expectations
    df = df.reindex(columns=feature_names, fill_value=0)
    
    # Scale the continuous 'Age' column
    df['Age'] = scaler.transform(df[['Age']])
    
    # Predict!
    pred = model.predict(df)[0][0]
    
    return {
        "probability": float(pred),
        "prediction": int(pred > 0.5)
    }

@app.head("/health")
def health():
    return {"status": "healthy"}
