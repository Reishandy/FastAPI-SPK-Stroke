import os

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://0.0.0.0:29997")
DB_NAME = os.getenv("DB_NAME", "stroke_risk_db")

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

# Collections
users_collection = db["users"]
prediction_logs_collection = db["prediction_logs"]


# Helper to fix ObjectId serialization for Pydantic
def fix_id(doc):
    if doc and "_id" in doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc
