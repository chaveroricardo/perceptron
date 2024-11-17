import streamlit as st
import numpy as np
from pymongo import MongoClient
from bson.binary import Binary
from PIL import Image
from io import BytesIO
from skimage.transform import resize
import os
from dotenv import load_dotenv
import pickle
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

# Save the image and vector to MongoDB
def save_image_and_vector_to_mongo(filename, label, image, vector):
    collection = get_mongo_connection()

    # Save the image as binary
    image_binary = Binary(image)

    # Compress the vector and save as binary
    compressed_path = f"temp_{filename}.npz"
    np.savez_compressed(compressed_path, vector=vector)
    with open(compressed_path, "rb") as f:
        vector_binary = Binary(f.read())

    # Create a document for MongoDB
    document = {
        "filename": filename,
        "label": label,  # Default label; use -1 if unknown
        "image_binary": image_binary,  # Image as binary
        "compressed_vector_binary": vector_binary,  # Compressed vector as binary
    }

    # Insert into MongoDB
    collection.insert_one(document)
    os.remove(compressed_path)  # Clean up temporary file
    st.success(f"Image and vector for {filename} saved to MongoDB!")

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
st.write("Upload an image to predict cancer and optionally save it to MongoDB.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Preprocess the image: convert to grayscale and resize to 128x128
    image = Image.open(BytesIO(uploaded_file.read())).convert("L")
    image_resized = resize(np.array(image), (128, 128))
    vector = image_resized.flatten()

    # Save the image as binary for MongoDB
    image.seek(0)  # Reset the image pointer
    image_binary = uploaded_file.getvalue()

    # Optionally save to MongoDB
    save_to_mongo = st.checkbox("Save this image to MongoDB")
    if save_to_mongo:
        label = st.selectbox("Choose a label (if known)", ["Unknown", "Cancer", "Healthy"])
        label_value = -1 if label == "Unknown" else (1 if label == "Cancer" else 0)
        save_image_and_vector_to_mongo(uploaded_file.name, label_value, image_binary, vector)

    # Select model
    model_choice = st.selectbox("Choose a model", ["Perceptron", "RandomForest", "LinearSVM"])

    if st.button("Predict"):
        # Load model and metrics from MongoDB
        model, metrics = load_model_from_mongo(model_choice)
        if model:
            # Use the current vector for prediction
            probas = model.predict_proba([vector])[0]
            prediction = model.predict([vector])[0]

            st.write(f"Model: {model_choice}")
            st.write(f"Prediction: {'Cancer' if prediction == 1 else 'Healthy'}")
            st.write(f"Probability of Cancer: {probas[1] * 100:.2f}%")
            st.write(f"Probability of Healthy: {probas[0] * 100:.2f}%")

            st.write("Model Metrics:")
            for metric, value in metrics.items():
                st.write(f"{metric}: {value:.2f}")