from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from bson.binary import Binary
import pickle
import numpy as np
from db import db

router = APIRouter()

# Define the input schema for prediction
class PredictionRequest(BaseModel):
    vector: List[float]
    model_name: str

# Define the output schema
class PredictionResponse(BaseModel):
    prediction: str
    probabilities: Dict[str, float]
    metrics: Dict[str, float]

@router.post("/", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    # Fetch the model from MongoDB
    model_data = db["models"].find_one({"model_name": request.model_name})
    if not model_data:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")

    # Deserialize the model
    model = pickle.loads(model_data["model_binary"])

    # Convert the vector to NumPy array
    vector = np.array(request.vector).reshape(1, -1)

    # Perform prediction
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    # Prepare response
    return PredictionResponse(
        prediction="Cancer" if prediction == 1 else "Healthy",
        probabilities={
            "Cancer": probabilities[1] * 100,
            "Healthy": probabilities[0] * 100
        },
        metrics=model_data["metrics"]
    )
