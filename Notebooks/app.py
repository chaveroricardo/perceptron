import streamlit as st
import numpy as np
import pickle
from PIL import Image
from io import BytesIO

# Title and description
st.title("Cancer Prediction with Machine Learning Models")
st.write("Upload an image and select a model to predict cancer and view performance metrics.")

# Load models and metrics
@st.cache
def load_model(model_name):
    try:
        with open(f"models/{model_name}.pkl", "rb") as f:
            model_data = pickle.load(f)
        return model_data["model"], model_data["metrics"]
    except FileNotFoundError:
        st.error(f"Model file '{model_name}.pkl' not found!")
        return None, None

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Process the uploaded image (simulate preprocessing)
    image = Image.open(BytesIO(uploaded_file.read()))
    test_vector = np.random.rand(1, 67600)  # Simulated vector for demonstration

    # Select model
    model_choice = st.selectbox("Choose a model", ["Perceptron", "RandomForest", "LinearSVM"])

    if st.button("Predict"):
        # Load model and metrics
        model, metrics = load_model(model_choice.lower())
        if model:
            # Predict using the model
            probas = model.predict_proba(test_vector)[0]
            prediction = model.predict(test_vector)[0]

            st.write(f"Model: {model_choice}")
            st.write(f"Prediction: {'Cancer' if prediction == 1 else 'Healthy'}")
            st.write(f"Probability of Cancer: {probas[1] * 100:.2f}%")
            st.write(f"Probability of Healthy: {probas[0] * 100:.2f}%")

            st.write("Model Metrics:")
            for metric, value in metrics.items():
                st.write(f"{metric}: {value:.2f}")
