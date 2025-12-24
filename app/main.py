from fastapi import FastAPI
import os
from fastapi.middleware.cors import CORSMiddleware

from app.routers import auth, users, predictions

app = FastAPI(
    title="Stroke Risk Prediction API",
    description="Full-stack API for stroke risk assessment with user history and personalization.",
    version="2.0.0"
)

# Read DEBUG flag from the environment. Treat common truthy values as True.
DEBUG = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

# If DEBUG is enabled, permit any origin (useful for local development / testing).
if DEBUG:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
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
