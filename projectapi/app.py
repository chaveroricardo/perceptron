import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from skimage.transform import resize
import numpy as np

API_URL = "http://127.0.0.1:8000/api/predict"

# Title and description
st.title("Cancer Prediction with Machine Learning Models")
st.write("Upload an image to predict cancer using our API.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Preprocess the image: convert to grayscale and resize to 128x128
    image = Image.open(BytesIO(uploaded_file.read())).convert("L")
    image_resized = resize(np.array(image), (128, 128))
    vector = image_resized.flatten().tolist()  # Convert to list for JSON

    # Select model
    model_choice = st.selectbox("Choose a model", ["Perceptron", "RandomForest", "LinearSVM"])

    if st.button("Predict"):
        # Send the vector to FastAPI
        response = requests.post(API_URL, json={"vector": vector, "model_name": model_choice})
        if response.status_code == 200:
            result = response.json()
            st.write(f"Model: {model_choice}")
            st.write(f"Prediction: {result['prediction']}")
            st.write(f"Probability of Cancer: {result['probabilities']['Cancer']:.2f}%")
            st.write(f"Probability of Healthy: {result['probabilities']['Healthy']:.2f}%")
            st.write("Model Metrics:")
            for metric, value in result["metrics"].items():
                st.write(f"{metric}: {value:.2f}")
        else:
            st.error(f"Error: {response.status_code} - {response.json()['detail']}")
