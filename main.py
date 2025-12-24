from fastapi import FastAPI
from routers import auth, users, predictions

app = FastAPI(
    title="Stroke Risk Prediction API",
    description="Full-stack API for stroke risk assessment with user history and personalization.",
    version="2.0.0"
)

# --- Include Routers ---
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(predictions.router)

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "active", "version": "2.0.0"}