from pydantic import BaseModel
from typing import List, Dict

class PredictionRequest(BaseModel):
    vector: List[float]
    model_name: str

class PredictionResponse(BaseModel):
    prediction: str
    probabilities: Dict[str, float]
    metrics: Dict[str, float]
