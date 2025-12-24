import os
from datetime import datetime
from typing import List

import joblib
import numpy as np
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException

from app.database import prediction_logs_collection
from app.routers.users import get_current_user
from app.schemas import ModelType, StrokeInput, PredictionOutput, PredictionHistoryItem, PredictionDetail

# --- Configuration ---
MODEL_DIR = "app/models"
MODELS_MAP = {
    "logistic": "model_logistic_regression.pkl",
    "random_forest": "model_random_forest.pkl",
    "svm": "model_svm.pkl"
}
SCALER_FILE = "scaler_stroke.pkl"

# Global State
loaded_models = {}
scaler = None

router = APIRouter(prefix="/predict", tags=["Prediction & History"])


# --- Helper Functions ---
def load_artifacts():
    global scaler, loaded_models
    if scaler is None:
        scaler_path = os.path.join(MODEL_DIR, SCALER_FILE)
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"[System] Loaded Scaler: {SCALER_FILE}")
        else:
            print(f"[Error] Scaler not found at {scaler_path}")

    for model_name, filename in MODELS_MAP.items():
        if model_name not in loaded_models:
            path = os.path.join(MODEL_DIR, filename)
            if os.path.exists(path):
                loaded_models[model_name] = joblib.load(path)
                print(f"[System] Loaded Model: {model_name}")


def preprocess_input(input_data: StrokeInput) -> np.ndarray:
    if scaler is None:
        raise HTTPException(status_code=500, detail="Scaler is not loaded.")

    binary_features = [
        input_data.chest_pain, input_data.shortness_of_breath, input_data.irregular_heartbeat,
        input_data.fatigue_weakness, input_data.dizziness, input_data.swelling_edema,
        input_data.pain_neck_jaw, input_data.excessive_sweating, input_data.persistent_cough,
        input_data.nausea_vomiting, input_data.high_blood_pressure,
        input_data.chest_discomfort_activity, input_data.cold_hands_feet,
        input_data.snoring_sleep_apnea, input_data.anxiety_feeling_doom
    ]

    numerical_features = [[input_data.age, input_data.stroke_risk_percentage]]
    scaled_numerical = scaler.transform(numerical_features)
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
        prob_at_risk = 0.0
        if hasattr(model, "predict_proba"):
            prob_array = model.predict_proba(processed_data)[0]
            # Assuming class 0 = Not At Risk, class 1 = At Risk
            probs = {
                "not_at_risk": float(prob_array[0]),
                "at_risk": float(prob_array[1])
            }
            prob_at_risk = float(prob_array[1])

        # Result object structure
        result_obj = {
            "label": "At Risk" if prediction == 1 else "Not At Risk",
            "score": int(prediction),
            "probability_at_risk": prob_at_risk
        }

        # SAVE TO DB (as per requested schema)
        log_entry = {
            "user_id": ObjectId(current_user["id"]),
            "timestamp": datetime.utcnow(),
            "model_version": "2.0.0",  # Hardcoded or dynamic
            "model_used": model_type.value,
            "input_data": input_data.dict(),
            "result": result_obj
        }

        await prediction_logs_collection.insert_one(log_entry)

        # Return response matching the Pydantic schema
        return {
            "model_used": model_type.value,
            "prediction_label": result_obj["label"],
            "prediction_score": result_obj["score"],
            "probability": probs
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/history", response_model=List[PredictionHistoryItem])
async def get_prediction_history(
        skip: int = 0,
        limit: int = 10,
        current_user: dict = Depends(get_current_user)
):
    cursor = prediction_logs_collection.find(
        {"user_id": ObjectId(current_user["id"])}
    ).sort("timestamp", -1).skip(skip).limit(limit)

    logs = []
    async for doc in cursor:
        # Map DB structure back to Pydantic Response
        logs.append(PredictionHistoryItem(
            id=str(doc["_id"]),
            timestamp=doc["timestamp"].isoformat(),
            model_used=doc["model_used"],
            prediction_label=doc["result"]["label"],
            prediction_score=doc["result"]["score"]
        ))

    return logs


@router.get("/history/{log_id}", response_model=PredictionDetail)
async def get_history_detail(log_id: str, current_user: dict = Depends(get_current_user)):
    try:
        obj_id = ObjectId(log_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid Log ID")

    log = await prediction_logs_collection.find_one({
        "_id": obj_id,
        "user_id": ObjectId(current_user["id"])
    })

    if not log:
        raise HTTPException(status_code=404, detail="History item not found")

    # Reconstruct probability dict from stored 'probability_at_risk' if needed,
    # or just return None if strict reconstruction isn't possible/needed for detail view.
    # Here we approximate to fit the schema.
    p_at_risk = log["result"].get("probability_at_risk", 0.0)

    return PredictionDetail(
        id=str(log["_id"]),
        timestamp=log["timestamp"].isoformat(),
        model_used=log["model_used"],
        prediction_label=log["result"]["label"],
        prediction_score=log["result"]["score"],
        input_data=log["input_data"],
        probability={
            "at_risk": p_at_risk,
            "not_at_risk": 1.0 - p_at_risk
        }
    )