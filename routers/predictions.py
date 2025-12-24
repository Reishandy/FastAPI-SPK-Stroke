import os
import joblib
import numpy as np
from datetime import datetime
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from schemas import ModelType, StrokeInput, PredictionOutput, PredictionHistoryItem, PredictionDetail
from routers.users import get_current_user  # Protected Dependency

# --- Configuration ---
# Assuming you run uvicorn from the root project folder
MODEL_DIR = "models"
MODELS_MAP = {
    "logistic": "model_logistic_regression.pkl",
    "random_forest": "model_random_forest.pkl",
    "svm": "model_svm.pkl"
}
SCALER_FILE = "scaler_stroke.pkl"

# --- Global State for ML Artifacts ---
loaded_models = {}
scaler = None

# --- Mock Database for History ---
# In a real app, this would be a MongoDB collection
fake_prediction_db = []

router = APIRouter(prefix="/predict", tags=["Prediction & History"])


# --- Helper Functions ---

def load_artifacts():
    """
    Load models and scaler if they aren't already loaded.
    """
    global scaler, loaded_models

    # 1. Load Scaler
    if scaler is None:
        scaler_path = os.path.join(MODEL_DIR, SCALER_FILE)
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"[System] Loaded Scaler: {SCALER_FILE}")
        else:
            print(f"[Error] Scaler not found at {scaler_path}")

    # 2. Load Models
    for model_name, filename in MODELS_MAP.items():
        if model_name not in loaded_models:
            path = os.path.join(MODEL_DIR, filename)
            if os.path.exists(path):
                loaded_models[model_name] = joblib.load(path)
                print(f"[System] Loaded Model: {model_name}")
            else:
                print(f"[Warning] Model {model_name} not found at {path}")


def preprocess_input(input_data: StrokeInput) -> np.ndarray:
    """
    Converts Pydantic input to the numpy array format expected by the model.
    Logic taken from original main.py.
    """
    if scaler is None:
        raise HTTPException(status_code=500, detail="Scaler is not loaded.")

    # 1. Extract Binary Features (Order matters!)
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
    numerical_features = [[input_data.age, input_data.stroke_risk_percentage]]

    # 3. Scale Numerical Features
    scaled_numerical = scaler.transform(numerical_features)

    # 4. Concatenate [Binary... , Scaled_Age, Scaled_Risk]
    # binary_features needs to be 2D array (1, 15) to concatenate with (1, 2)
    final_input = np.concatenate([np.array([binary_features]), scaled_numerical], axis=1)

    return final_input


# --- Endpoints ---

@router.post("/{model_type}", response_model=PredictionOutput)
async def predict_stroke_risk(
        model_type: ModelType,
        input_data: StrokeInput,
        current_user: dict = Depends(get_current_user)
):
    """
    Predicts stroke risk and saves the result to the user's history.
    Requires Authentication.
    """
    # 0. Ensure artifacts are loaded
    load_artifacts()

    # 1. Select Model
    if model_type.value not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {model_type} not available.")

    model = loaded_models[model_type.value]

    try:
        # 2. Preprocess Data
        processed_data = preprocess_input(input_data)

        # 3. Run Inference
        prediction = model.predict(processed_data)[0]

        # 4. Get Probability (if available)
        probs = None
        if hasattr(model, "predict_proba"):
            prob_array = model.predict_proba(processed_data)[0]
            # Assuming class 0 = Not At Risk, class 1 = At Risk
            probs = {
                "not_at_risk": float(prob_array[0]),
                "at_risk": float(prob_array[1])
            }

        # 5. Construct Result
        result_data = {
            "model_used": model_type.value,
            "prediction_label": "At Risk" if prediction == 1 else "Not At Risk",
            "prediction_score": int(prediction),
            "probability": probs
        }

        # 6. SAVE TO HISTORY (Mock DB Operation)
        # In Mongo: await db.prediction_logs.insert_one(log_entry)
        log_entry = {
            "id": f"log_{len(fake_prediction_db) + 1}",
            "user_email": current_user["email"],  # Link to user
            "timestamp": datetime.utcnow().isoformat(),
            "input_data": input_data.dict(),
            **result_data  # Flattens model_used, label, score, probability
        }
        fake_prediction_db.append(log_entry)
        print(f"[DB] Saved prediction for user {current_user['email']}")

        return result_data

    except Exception as e:
        # Log the full error to console for debugging
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/history", response_model=List[PredictionHistoryItem])
async def get_prediction_history(
        skip: int = 0,
        limit: int = 10,
        current_user: dict = Depends(get_current_user)
):
    """
    Retrieve past predictions for the logged-in user.
    """
    # 1. Filter DB for current user
    user_logs = [log for log in fake_prediction_db if log["user_email"] == current_user["email"]]

    # 2. Apply pagination (skip/limit) -> Sort by newest first usually
    # For mock list, we just slice. 
    # In Mongo: find({'user_id': ...}).sort('timestamp', -1).skip(skip).limit(limit)
    start = len(user_logs) - skip - limit
    end = len(user_logs) - skip

    # Simple list slicing for mock purposes (reversing to show newest first)
    user_logs_sorted = user_logs[::-1]
    paginated_logs = user_logs_sorted[skip: skip + limit]

    return paginated_logs


@router.get("/history/{log_id}", response_model=PredictionDetail)
async def get_history_detail(log_id: str, current_user: dict = Depends(get_current_user)):
    """
    Get full details of a specific prediction.
    """
    # 1. Find log
    log = next((item for item in fake_prediction_db if item["id"] == log_id), None)

    # 2. Security Check: Does log exist and belong to user?
    if not log:
        raise HTTPException(status_code=404, detail="History item not found")

    if log["user_email"] != current_user["email"]:
        raise HTTPException(status_code=403, detail="Not authorized to view this record")

    return log