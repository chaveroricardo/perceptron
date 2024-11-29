import requests
import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
from skimage.transform import resize

# API URL for predictions
API_URL = "http://127.0.0.1:8000/api/predict/"

# Preprocess the image to generate a feature vector
def preprocess_image(image_file):
    # Convert the image to grayscale and resize to 128x128
    image = Image.open(BytesIO(image_file.read())).convert("L")
    image_resized = resize(np.array(image), (128, 128))
    return image_resized.flatten().tolist()

# Title and description
st.title("Cancer Prediction with Machine Learning Models")
st.write("Upload an image to predict cancer and view model metrics.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    vector = preprocess_image(uploaded_file)

    # Select the model
    model_choice = st.selectbox("Choose a model", ["Perceptron", "RandomForest", "LinearSVM"])

    if st.button("Predict"):
        # Send a prediction request to the FastAPI backend
        response = requests.post(API_URL, json={"vector": vector, "model_name": model_choice})

        if response.status_code == 200:
            # Parse the response
            result = response.json()

            # Display prediction results
            st.write(f"### Model: {model_choice}")
            st.write(f"**Prediction**: {result['prediction']}")
            st.write(f"**Probability of Cancer**: {result['probabilities']['Cancer']:.2f}%")
            st.write(f"**Probability of Healthy**: {result['probabilities']['Healthy']:.2f}%")

            # Display model metrics
            st.write("### Model Metrics:")
            for metric, value in result["metrics"].items():
                st.write(f"**{metric}**: {value:.2f}")
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
