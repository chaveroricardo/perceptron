from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pickle
import numpy as np
from ..db import db

router = APIRouter()

class PredictionRequest(BaseModel):
    model_name: str
    vector: List[float]

class PredictionResponse(BaseModel):
    prediction: str
    probabilities: Dict[str, float]
    metrics: Dict[str, float]

@router.post("/", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    model_data = db["Esophagus"].find_one({"model_name": request.model_name})
    if not model_data:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")

    model = pickle.loads(model_data["model_binary"])
    vector = np.array(request.vector).reshape(1, -1)
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    return PredictionResponse(
        prediction="Cancer" if prediction == 1 else "Healthy",
        probabilities={
            "Cancer": probabilities[1] * 100,
            "Healthy": probabilities[0] * 100,
        },
        metrics=model_data["metrics"]
    )
