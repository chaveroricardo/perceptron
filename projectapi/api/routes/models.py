from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.db import db
import pickle

router = APIRouter()

# Schemas for input and output
class ModelDetails(BaseModel):
    model_name: str
    description: str
    metrics: dict

@router.get("/", response_model=list[ModelDetails])
def list_models():
    """
    List all models stored in MongoDB.
    """
    models = db["models"].find({}, {"_id": 0, "model_name": 1, "description": 1, "metrics": 1})
    return list(models)

@router.get("/{model_name}", response_model=ModelDetails)
def get_model_details(model_name: str):
    """
    Fetch details of a specific model.
    """
    model = db["models"].find_one({"model_name": model_name}, {"_id": 0, "model_name": 1, "description": 1, "metrics": 1})
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found.")
    return model

@router.post("/")
def upload_model(model_name: str, description: str, metrics: dict, model_binary: bytes):
    """
    Upload a new model to MongoDB.
    """
    # Check if model already exists
    if db["models"].find_one({"model_name": model_name}):
        raise HTTPException(status_code=400, detail=f"Model {model_name} already exists.")
    
    # Save model
    db["models"].insert_one({
        "model_name": model_name,
        "description": description,
        "metrics": metrics,
        "model_binary": model_binary,
    })
    return {"message": f"Model {model_name} uploaded successfully."}

@router.delete("/{model_name}")
def delete_model(model_name: str):
    """
    Delete a specific model from MongoDB.
    """
    result = db["models"].delete_one({"model_name": model_name})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found.")
    return {"message": f"Model {model_name} deleted successfully."}
