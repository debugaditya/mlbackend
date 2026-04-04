from fastapi import FastAPI
import numpy as np
import pandas as pd
import pickle
import keras

# Intercepts version mismatches between Colab and your local PC
class SafeDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)

app = FastAPI()

# Load everything once when the server starts
model = keras.models.load_model("model.h5", custom_objects={'Dense': SafeDense})
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("features.pkl", "rb"))

@app.post("/predict")
def predict(data: dict):
    # 1. Convert incoming JSON to DataFrame
    df = pd.DataFrame([data])
    
    # 2. THE FIX: Find which columns were categorical in training
    # If a column name from the incoming data isn't directly in the final features list, 
    # it means it was one-hot encoded during training!
    categorical_cols = [col for col in df.columns if col not in feature_names]
    
    # 3. Force pandas to one-hot encode them (even if they are integers)
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols)
        
    # 4. Align columns exactly to match the model's expectations
    df = df.reindex(columns=feature_names, fill_value=0)
    
    # 5. Scale the continuous 'Age' column
    df['Age'] = scaler.transform(df[['Age']])
    
    # 6. Predict!
    pred = model.predict(df)[0][0]
    
    return {
        "probability": float(pred),
        "prediction": int(pred > 0.5)
    }

@app.get("/health")
def health():
    return {"status": "healthy"}