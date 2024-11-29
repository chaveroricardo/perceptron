from fastapi import APIRouter, HTTPException
from db import db
import pickle

router = APIRouter()

@router.post("/models/")
def add_model(model_name: str, metrics: dict, model_binary: bytes):
    collection = db["models"]
    if collection.find_one({"model_name": model_name}):
        raise HTTPException(status_code=400, detail="Model already exists")

    model_doc = {
        "model_name": model_name,
        "metrics": metrics,
        "model_binary": Binary(model_binary),
    }
    collection.insert_one(model_doc)
    return {"message": "Model added successfully"}

@router.get("/models/{model_name}")
def get_model(model_name: str):
    collection = db["models"]
    model = collection.find_one({"model_name": model_name})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return {
        "model_name": model["model_name"],
        "metrics": model["metrics"],
    }
