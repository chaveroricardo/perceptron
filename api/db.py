import os 
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables from .env
load_dotenv()

# MongoDB connection settings
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("DATABASE_NAME")
MONGO_COLLECTION = os.getenv("COLLECTION_NAME")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]