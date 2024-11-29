from fastapi import FastAPI
from routes.models import router as models_router
from routes.predictions import router as predictions_router

app = FastAPI()

# Register routes
app.include_router(models_router, prefix="/api/models", tags=["Models"])
app.include_router(predictions_router, prefix="/api/predict", tags=["Predictions"])

