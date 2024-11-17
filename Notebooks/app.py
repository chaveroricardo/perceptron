import streamlit as st
import numpy as np
from pymongo import MongoClient
import pickle
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# MongoDB connection settings
MONGO_CONNECTION_STRING = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("DATABASE_NAME")
MONGO_COLLECTION = os.getenv("COLLECTION_NAME")

# Connect to MongoDB
@st.cache_resource
def get_mongo_connection():
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[MONGO_DB]
    return db[MONGO_COLLECTION]  # Return the specified collection

# Load model and metrics from MongoDB
def load_model_from_mongo(model_name):
    collection = get_mongo_connection()
    document = collection.find_one({"model_name": model_name})
    if not document:
        st.error(f"Model {model_name} not found in MongoDB.")
        return None, None

    # Deserialize the model
    model_binary = document["model_binary"]
    model = pickle.loads(model_binary)
    metrics = document["metrics"]
    return model, metrics

# Get the vector from MongoDB
def get_vector_from_mongo(filename):
    collection = get_mongo_connection()
    document = collection.find_one({"filename": filename})
    if not document:
        st.error(f"Image vector for {filename} not found in MongoDB.")
        return None

    # Load the compressed vector from MongoDB
    compressed_vector_binary = document['compressed_vector_binary']
    with open("temp_vector.npz", "wb") as f:
        f.write(compressed_vector_binary)
    vector_data = np.load("temp_vector.npz")['vector']
    return vector_data

# Title and description
st.title("Cancer Prediction with Machine Learning Models")
st.write("Upload an image and select a model to predict cancer and view performance metrics.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Get the filename of the uploaded image
    uploaded_filename = uploaded_file.name

    # Select model
    model_choice = st.selectbox("Choose a model", ["Perceptron", "RandomForest", "LinearSVM"])

    if st.button("Predict"):
        # Load model and metrics from MongoDB
        model, metrics = load_model_from_mongo(model_choice)
        if model:
            # Get the vector for the uploaded image from MongoDB
            test_vector = get_vector_from_mongo(uploaded_filename)
            if test_vector is not None:
                # Predict using the model
                probas = model.predict_proba([test_vector])[0]
                prediction = model.predict([test_vector])[0]

                st.write(f"Model: {model_choice}")
                st.write(f"Prediction: {'Cancer' if prediction == 1 else 'Healthy'}")
                st.write(f"Probability of Cancer: {probas[1] * 100:.2f}%")
                st.write(f"Probability of Healthy: {probas[0] * 100:.2f}%")

                st.write("Model Metrics:")
                for metric, value in metrics.items():
                    st.write(f"{metric}: {value:.2f}")