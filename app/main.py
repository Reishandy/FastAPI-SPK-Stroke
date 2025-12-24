from fastapi import FastAPI
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import json

from app.routers import auth, users, predictions

# Load environment variables from a .env file (if present)
load_dotenv()

app = FastAPI(
    title="Stroke Risk Prediction API",
    description="Full-stack API for stroke risk assessment with user history and personalization.",
    version="2.0.0"
)

# Read DEBUG flag from the environment. Treat common truthy values as True.
DEBUG = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

# Read CORS allowed origins from environment. Supports either:
# - a comma-separated string: "https://a.com,https://b.com"
# - a JSON list string: "[\"https://a.com\", \"https://b.com\"]"
# - the literal "*" to allow all origins
# Accepts either CORS_ALLOW_ORIGINS or ALLOWED_ORIGINS
_allowed_raw = os.getenv("CORS_ALLOW_ORIGINS", os.getenv("ALLOWED_ORIGINS", "")).strip()
_allowed_origins = []
if _allowed_raw:
    if _allowed_raw == "*":
        _allowed_origins = ["*"]
    else:
        try:
            # Try to parse JSON list first
            parsed = json.loads(_allowed_raw)
            if isinstance(parsed, list):
                _allowed_origins = [str(x) for x in parsed]
            else:
                # Fallback to comma split
                _allowed_origins = [s.strip() for s in _allowed_raw.split(",") if s.strip()]
        except Exception:
            # Not JSON -> treat as comma-separated
            _allowed_origins = [s.strip() for s in _allowed_raw.split(",") if s.strip()]

# If DEBUG is True and no explicit origins provided, allow all origins (useful for development)
if DEBUG and not _allowed_origins:
    _allowed_origins = ["*"]

# If we have any allowed origins configured, add the CORS middleware.
# This keeps the previous behavior (only adding middleware in DEBUG) but also allows
# explicit configuration via environment variables in production.
if _allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# --- Include Routers ---
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(predictions.router)


@app.get("/", tags=["Health"])
def health_check():
    return {"status": "active", "version": "1.0.0"}
