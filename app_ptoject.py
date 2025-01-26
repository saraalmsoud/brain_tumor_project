import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

import os
import requests

if not os.path.exists("brain_tumor_model.h5"):
    file_url = 'https://drive.google.com/uc?id=1UXXRlN0V6j_Ky0xKFQuBT9V7JMU0GVsw'
    response = requests.get(file_url)
    with open('brain_tumor_model.h5', 'wb') as file:
        file.write(response.content)



# Change the background color to a gradient from blue to gray
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #4f6d7a, #d1d9d7);  /* Blue to gray gradient */
    }
    </style>
    """, unsafe_allow_html=True
)

# Customize the title style
st.markdown(
    """
    <style>
    .stFileUploader > label {
        font-size: 30px;  /* Increase font size */
        font-weight: bold;  /* Make the text bold */
        color: white;  /* Change text color to white */
        background-color: #8A2BE2;  /* Set background color to a vibrant purple */
        padding: 15px;  /* Add padding to make it more spacious */
        border-radius: 10px;  /* Rounded corners */
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);  /* Add shadow effect */
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    body {
        background-color: #EDE7F6;  /* Light purple color */
    }
    .stApp {
        background-color: #EDE7F6;  /* Light purple for the main app background */
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Load the model (ensure this is a Keras or TensorFlow model)
model = keras.models.load_model(r"C:\Users\ssalq\OneDrive\سطح المكتب\toumr_project\brain_tumor_model.h5")

# Set up the user interface
st.title("Brain Tumor Prediction System")
st.write("Please upload an MRI image for prediction.")

# Upload the MRI image
uploaded_file = st.file_uploader("Upload the MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Process the image (ensure this step matches the preprocessing used during training)
    # For example: resize the image to match the size used by the model
    image = image.resize((224, 224))  # Resize to 224x224 as required by the model
    image_array = np.array(image) / 255.0  # Convert image to array and normalize by dividing by 255

    # Ensure the image has 3 color channels (RGB)
    if image_array.shape[-1] != 3:
        image_array = np.repeat(image_array, 3, axis=-1)

    # Add an extra dimension to match the model input shape (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array)

    # If it's a binary classification model, extract the probability for tumor (class 1)
    prediction_proba = prediction[0][1]  # Probability for class 1 (Tumor)

    # Display results
    st.write("### Prediction Result:")
    predicted_class = np.argmax(prediction, axis=1)  # Get the class with the highest probability

    if predicted_class[0] == 1:  # 1 = Tumor
        st.write("🔴 Tumor Detected.")
    else:
        st.write("🟢 No Tumor Detected.")

    # Display prediction probabilities
    st.write("### Prediction Probabilities:")
    st.write(f"No Tumor: {100 - prediction_proba * 100:.2f}%")
    st.write(f"Tumor Detected: {prediction_proba * 100:.2f}%")