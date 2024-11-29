from fastapi import FastAPI
from api.routes.predictions import router as predictions_router
from api.routes.models import router as models_router

app = FastAPI()

# Include routes
app.include_router(predictions_router, prefix="/api/predict", tags=["Predictions"])
app.include_router(models_router, prefix="/api/models", tags=["Models"])

@app.get("/")
def root():
    return {"message": "Welcome to the Cancer Prediction API!"}
