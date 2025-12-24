import os
from enum import Enum

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- Configuration ---
MODEL_DIR = "models"
MODELS = {
    "logistic": "model_logistic_regression.pkl",
    "random_forest": "model_random_forest.pkl",
    "svm": "model_svm.pkl"
}
SCALER_FILE = "scaler_stroke.pkl"

# --- App Initialization ---
app = FastAPI(
    title="Stroke Risk Prediction API",
    description="API for predicting stroke risk based on medical symptoms using Logistic Regression, Random Forest, or SVM.",
    version="1.0.0"
)

# Global variables to store loaded models
loaded_models = {}
scaler = None


# --- Pydantic Schemas ---

class ModelType(str, Enum):
    logistic = "logistic"
    random_forest = "random_forest"
    svm = "svm"


class StrokeInput(BaseModel):
    # Binary Symptoms (0 or 1)
    chest_pain: int = Field(..., description="1 if Chest Pain present, else 0", ge=0, le=1)
    shortness_of_breath: int = Field(..., description="1 if Shortness of Breath present, else 0", ge=0, le=1)
    irregular_heartbeat: int = Field(..., description="1 if Irregular Heartbeat present, else 0", ge=0, le=1)
    fatigue_weakness: int = Field(..., description="1 if Fatigue & Weakness present, else 0", ge=0, le=1)
    dizziness: int = Field(..., description="1 if Dizziness present, else 0", ge=0, le=1)
    swelling_edema: int = Field(..., description="1 if Swelling (Edema) present, else 0", ge=0, le=1)
    pain_neck_jaw: int = Field(..., description="1 if Pain in Neck/Jaw/Shoulder/Back present, else 0", ge=0, le=1)
    excessive_sweating: int = Field(..., description="1 if Excessive Sweating present, else 0", ge=0, le=1)
    persistent_cough: int = Field(..., description="1 if Persistent Cough present, else 0", ge=0, le=1)
    nausea_vomiting: int = Field(..., description="1 if Nausea/Vomiting present, else 0", ge=0, le=1)
    high_blood_pressure: int = Field(..., description="1 if High Blood Pressure present, else 0", ge=0, le=1)
    chest_discomfort_activity: int = Field(..., description="1 if Chest Discomfort during activity, else 0", ge=0, le=1)
    cold_hands_feet: int = Field(..., description="1 if Cold Hands/Feet present, else 0", ge=0, le=1)
    snoring_sleep_apnea: int = Field(..., description="1 if Snoring/Sleep Apnea present, else 0", ge=0, le=1)
    anxiety_feeling_doom: int = Field(..., description="1 if Anxiety/Feeling of Doom present, else 0", ge=0, le=1)

    # Numerical Features
    age: int = Field(..., description="Age of the patient", ge=18, le=100)
    stroke_risk_percentage: float = Field(..., description="Estimated Stroke Risk Percentage (0-100)", ge=0.0, le=100.0)

    class Config:
        schema_extra = {
            "example": {
                "chest_pain": 1,
                "shortness_of_breath": 0,
                "irregular_heartbeat": 1,
                "fatigue_weakness": 1,
                "dizziness": 1,
                "swelling_edema": 0,
                "pain_neck_jaw": 1,
                "excessive_sweating": 0,
                "persistent_cough": 1,
                "nausea_vomiting": 1,
                "high_blood_pressure": 1,
                "chest_discomfort_activity": 0,
                "cold_hands_feet": 1,
                "snoring_sleep_apnea": 1,
                "anxiety_feeling_doom": 1,
                "age": 80,
                "stroke_risk_percentage": 55.0
            }
        }


class PredictionOutput(BaseModel):
    model_used: str
    prediction_label: str = Field(..., description="'At Risk' or 'Not At Risk'")
    prediction_score: int = Field(..., description="1 for At Risk, 0 for Not At Risk")
    probability: dict | None = Field(None, description="Probability distribution if supported by model")


# --- Startup Event ---

@app.on_event("startup")
def load_artifacts():
    """Load machine learning models and scalers into memory on startup."""
    global scaler

    # Load Scaler
    scaler_path = os.path.join(MODEL_DIR, SCALER_FILE)
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    scaler = joblib.load(scaler_path)
    print(f"Loaded Scaler: {SCALER_FILE}")

    # Load Models
    for model_name, filename in MODELS.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            loaded_models[model_name] = joblib.load(path)
            print(f"Loaded Model: {model_name}")
        else:
            print(f"Warning: Model {model_name} not found at {path}")


# --- Helper Functions ---

def preprocess_input(input_data: StrokeInput) -> np.ndarray:
    """
    Converts Pydantic input to the numpy array format expected by the model.
    Order must match training: [15 binary symptoms, Age, Stroke Risk %]
    """
    # 1. Extract Binary Features in exact order of training
    binary_features = [
        input_data.chest_pain,
        input_data.shortness_of_breath,
        input_data.irregular_heartbeat,
        input_data.fatigue_weakness,
        input_data.dizziness,
        input_data.swelling_edema,
        input_data.pain_neck_jaw,
        input_data.excessive_sweating,
        input_data.persistent_cough,
        input_data.nausea_vomiting,
        input_data.high_blood_pressure,
        input_data.chest_discomfort_activity,
        input_data.cold_hands_feet,
        input_data.snoring_sleep_apnea,
        input_data.anxiety_feeling_doom
    ]

    # 2. Extract Numerical Features
    # Note: Training logic used scaler.fit_transform(data[['Age', 'Stroke Risk (%)']])
    numerical_features = [[input_data.age, input_data.stroke_risk_percentage]]

    # 3. Scale Numerical Features
    if scaler is None:
        raise HTTPException(status_code=500, detail="Scaler not loaded.")
    scaled_numerical = scaler.transform(numerical_features)

    # 4. Concatenate [Binary... , Scaled_Age, Scaled_Risk]
    # binary_features needs to be 2D array (1, 15) to concatenate with (1, 2)
    final_input = np.concatenate([np.array([binary_features]), scaled_numerical], axis=1)

    return final_input


# --- Endpoints ---

@app.get("/health", tags=["Health"])
def health_check():
    """Checks if the API is running and models are loaded."""
    return {
        "status": "active",
        "loaded_models": list(loaded_models.keys()),
        "scaler_loaded": scaler is not None
    }


@app.post("/predict/{model_type}", response_model=PredictionOutput, tags=["Prediction"])
def predict(model_type: ModelType, input_data: StrokeInput):
    """
    Predict stroke risk using a specific model.

    - **logistic**: Best for interpretability and speed.
    - **random_forest**: Best for complex, non-linear patterns.
    - **svm**: Good for high-dimensional separation.
    """

    if model_type.value not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {model_type} not found or failed to load.")

    model = loaded_models[model_type.value]

    try:
        # Preprocess
        processed_data = preprocess_input(input_data)

        # Predict
        prediction = model.predict(processed_data)[0]

        # Probability (if supported)
        probs = None
        if hasattr(model, "predict_proba"):
            prob_array = model.predict_proba(processed_data)[0]
            probs = {
                "not_at_risk": float(prob_array[0]),
                "at_risk": float(prob_array[1])
            }

        return PredictionOutput(
            model_used=model_type.value,
            prediction_label="At Risk" if prediction == 1 else "Not At Risk",
            prediction_score=int(prediction),
            probability=probs
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
