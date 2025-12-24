from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, EmailStr

# --- Enums ---
class ModelType(str, Enum):
    logistic = "logistic"
    random_forest = "random_forest"
    svm = "svm"

# --- Auth & User Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str

class PersonalDefaults(BaseModel):
    """Data to auto-fill forms based on user history."""
    age: Optional[int] = Field(None, ge=18, le=100)
    high_blood_pressure: Optional[int] = Field(None, ge=0, le=1)
    # Add other common defaults if needed

class UserResponse(BaseModel):
    id: str
    email: EmailStr
    full_name: str
    personal_defaults: Optional[PersonalDefaults] = None

# --- Prediction Schemas (Migrated from your main.py) ---
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
    prediction_label: str
    prediction_score: int
    probability: Optional[Dict[str, float]] = None

class PredictionHistoryItem(BaseModel):
    id: str
    timestamp: str
    model_used: str
    prediction_label: str
    prediction_score: int

class PredictionDetail(PredictionHistoryItem):
    input_data: StrokeInput
    probability: Optional[Dict[str, float]]